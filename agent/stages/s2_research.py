"""Stage 2 — deep account research.

1. Resolve the company in SEC EDGAR and pull its latest annual report (20-F/10-K) + recent 6-K/8-Ks.
2. LLM plans web searches per research dimension; code runs them and fetches the pages.
3. Per dimension, the LLM extracts findings from keyword-windowed excerpts, each with a quote.
4. Grounding gate: code verifies every quote against the fetched text; failures are rejected.
5. Contact check for the sender (no verified source -> reported as unverified, never invented).
6. Positioning is synthesised from verified findings only, citing finding ids.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from ..config import RESEARCH_DIMENSIONS
from ..evidence import verify_quote
from ..prompts import CONTACT_SYSTEM, EXTRACT_SYSTEM, PLAN_SYSTEM, POSITION_SYSTEM
from ..schemas import (ContactCheck, ContactDraft, Finding, FindingsDraft, PlannedQuery, PositioningDraft,
                       QueryPlan, RejectedClaim, Research, SourceRef)
from ..tools.sec import SEC

LOW_VALUE_DOMAINS = ("linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "youtube.com",
                     "tiktok.com", "glassdoor", "indeed.", "zoominfo", "rocketreach", "signalhire")


def run(ctx) -> Research:
    s, log, store = ctx.settings, ctx.log, ctx.store
    profile = ctx.out["profile"]
    names = list(dict.fromkeys([profile.company_legal_name, profile.company_short_name,
                                *profile.company_name_variants, ctx.email.company]))
    names = [n for n in names if n and len(n) > 1]
    rejected: list[RejectedClaim] = []

    # ---- 1. primary filings ----------------------------------------------------
    ids = {"names_tried": names}
    sec = SEC(ctx.web, s, log)
    tickers = [v for v in names if re.fullmatch(r"[A-Z.\-]{2,6}", v)]
    company = sec.find_cik(names, tickers)
    if company:
        ids.update(company)
        log.info(f"EDGAR: CIK {company['cik']} ({company['label']}) via {company['method']}")
        annual = sec.filings(company["cik"], ("20-F", "10-K", "40-F"), limit=1)
        for f in annual:
            page = ctx.web.fetch(f["url"], headers=sec.headers)
            if page and len(page.text) > 1000:
                src = store.add(f["url"], f"{f['form']} filed {f['date']} — {f['company']}", page.text, "filing",
                                {"org_structure", "budget_signals", "stakeholder_letters", "news"})
                ids["annual_report"] = {**f, "source_id": src.id, "chars": len(page.text)}
                log.info(f"annual report {f['form']} {f['date']}: {len(page.text):,} chars -> {src.id}")
        for f in sec.filings(company["cik"], ("6-K", "8-K"), limit=4):
            page = ctx.web.fetch(f["url"], headers=sec.headers)
            if page and len(page.text) > 500:
                store.add(f["url"], f"{f['form']} filed {f['date']} — {f['description'] or f['company']}",
                          page.text, "filing", {"news", "budget_signals"})
    else:
        log.warn("company not found in SEC EDGAR; relying on web sources only")

    # ---- 2. search plan + fetch ------------------------------------------------
    dims = {k: v["goal"] for k, v in RESEARCH_DIMENSIONS.items()}
    plan = ctx.llm.structured(
        system=PLAN_SYSTEM.format(vendor=s.vendor_name, today=ctx.run_date, n=s.queries_per_dimension),
        user=(f"Company: {profile.company_legal_name} ({profile.company_short_name}), {profile.hq_country}\n"
              f"Lead context: {profile.sub_industry}; {profile.use_case}; sites: {', '.join(profile.site_locations)}\n"
              f"Dimensions:\n" + "\n".join(f"- {k}: {v}" for k, v in dims.items())),
        schema=QueryPlan)
    queries = _ensure_coverage(plan.queries, profile, ctx.run_date)
    log.info(f"{len(queries)} search queries planned")

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda q: (q, ctx.web.search(q.query, s.results_per_query)), queries))
    picked: dict[str, list] = {d: [] for d in dims}
    seen = set()
    for q, rs in results:
        for r in rs:
            host = urlparse(r.url).netloc.lower()
            if any(b in host for b in LOW_VALUE_DOMAINS):
                continue
            if r.url in seen and r.url not in [x.url for x in picked[q.dimension]]:
                # already fetched for another dimension: just add the tag later
                picked[q.dimension].append(r)
                continue
            if len(picked[q.dimension]) < s.fetch_per_dimension and r.url not in seen:
                picked[q.dimension].append(r)
                seen.add(r.url)
    pages = ctx.web.fetch_many(list(seen))
    for dim, rs in picked.items():
        for r in rs:
            p = pages.get(r.url)
            if p and len(p.text) > 300:
                store.add(r.url, p.title or r.title, p.text, "web", {dim}, r.query)
    log.info(f"sources in store: {len(store.all())} "
             f"({sum(1 for v in pages.values() if v is None)} fetches failed or blocked)")

    # ---- 3+4. extract per dimension, verify -----------------------------------
    name_kws = [profile.company_short_name, profile.company_legal_name.split()[0]] + profile.site_locations
    def extract(dim):
        return dim, _extract_dimension(ctx, dim, name_kws)
    with ThreadPoolExecutor(max_workers=max(1, s.llm_parallel)) as ex:
        drafts = list(ex.map(extract, list(dims)))
    findings: list[Finding] = []
    for dim, draft in drafts:
        if draft is None:
            continue
        ok, bad = _verify(ctx, draft.findings, f"research/{dim}", dim)
        findings += ok
        rejected += bad

    # ---- 5. contact check -------------------------------------------------------
    contact = _contact_check(ctx, profile, rejected)
    findings += contact.findings

    for i, f in enumerate(findings, 1):
        f.id = f"F{i:02d}"
    log.info(f"verified findings: {len(findings)}; rejected claims: {len(rejected)}")

    # ---- 6. positioning -----------------------------------------------------------
    positioning = ctx.llm.structured(
        system=POSITION_SYSTEM.format(vendor=s.vendor_name),
        user=(f"<lead>\n{profile.model_dump_json(indent=1)}\n</lead>\n<qualification_gaps>\n"
              + "\n".join(ctx.out["qualification"].missing) + "\n</qualification_gaps>\n<verified_findings>\n"
              + "\n".join(f"{f.id} [{f.dimension}] {f.claim} ({f.date_or_period}) — so what: {f.so_what}"
                          for f in findings) + "\n</verified_findings>"),
        schema=PositioningDraft)
    valid = {f.id for f in findings}
    for h in positioning.value_hooks:
        bad_ids = [i for i in h.finding_ids if i not in valid]
        if bad_ids:
            log.warn(f"positioning hook cited unknown findings {bad_ids}; removed those ids")
        h.finding_ids = [i for i in h.finding_ids if i in valid]
    positioning.value_hooks = [h for h in positioning.value_hooks if h.finding_ids]
    for st in positioning.likely_stakeholders:
        st.finding_ids = [i for i in st.finding_ids if i in valid]
        if st.basis == "verified" and not st.finding_ids:
            st.basis = "inferred_from_role"

    coverage = {d: sum(1 for f in findings if f.dimension == d) for d in list(dims) + ["contact"]}
    gaps = [f"No verified findings for '{d}'" for d, n in coverage.items() if n == 0 and d != "contact"]
    if not contact.publicly_verified:
        gaps.append("Sender's role and reporting line are not publicly verifiable; confirm on first call.")
    sources = [SourceRef(id=x.id, url=x.url, title=x.title, kind=x.kind, tags=sorted(x.tags)) for x in store.all()]
    return Research(company_identifiers=ids, sources=sources, findings=findings, rejected=rejected, contact=contact,
                    positioning=positioning, coverage=coverage, gaps=gaps, queries_run=queries)


def _ensure_coverage(queries: list[PlannedQuery], profile, today: str) -> list[PlannedQuery]:
    """Guarantee >= 2 queries per dimension even if the planner skipped one."""
    year = today[:4]
    co = profile.company_short_name or profile.company_legal_name
    fallback = {
        "org_structure": [f"{co} executive officers organizational structure", f"{co} vice president operations {profile.sub_industry}"],
        "budget_signals": [f"{co} capital expenditure plan {year}", f"{co} investment plan technology automation"],
        "news": [f"{co} news {year}", f"{co} press release {profile.sub_industry} {year}"],
        "stakeholder_letters": [f"{co} annual report letter from chairman", f"{co} investor presentation {year}"],
    }
    out = list(queries)
    for dim, qs in fallback.items():
        have = [q for q in out if q.dimension == dim]
        for q in qs[: max(0, 2 - len(have))]:
            out.append(PlannedQuery(dimension=dim, query=q, why="coverage fallback"))
    seen, uniq = set(), []
    for q in out:
        if q.query.lower() not in seen:
            seen.add(q.query.lower())
            uniq.append(q)
    return uniq


def _sources_block(ctx, sources, keywords) -> str:
    s = ctx.settings
    parts, used = [], 0
    # filings first: they are the most authoritative
    for src in sorted(sources, key=lambda x: (x.kind != "filing", x.id)):
        budget = s.excerpt_chars_per_source * (2 if src.kind == "filing" else 1)
        budget = min(budget, s.max_prompt_chars - used)
        if budget < 1500:
            break
        ex = ctx.store.excerpt(src.id, keywords, budget)
        used += len(ex)
        parts.append(f'<source id="{src.id}" kind="{src.kind}" url="{src.url}" title="{src.title}">\n{ex}\n</source>')
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"


def _extract_dimension(ctx, dim, name_kws):
    sources = ctx.store.with_tag(dim)
    if not sources:
        ctx.log.warn(f"no sources for {dim}")
        return None
    spec = RESEARCH_DIMENSIONS[dim]
    block = _sources_block(ctx, sources, spec["keywords"] + [k for k in name_kws if k])
    profile = ctx.out["profile"]
    return ctx.llm.structured(
        system=EXTRACT_SYSTEM.format(vendor=ctx.settings.vendor_name, dimension=dim, goal=spec["goal"],
                                     max_findings=8, today=ctx.run_date),
        user=(f"Account: {profile.company_legal_name} ({profile.company_short_name}). Lead: {ctx.email.title}; "
              f"interest: {profile.use_case} at {', '.join(profile.site_locations) or 'their sites'}.\n"
              f"Set dimension='{dim}' on every finding.\n\n{block}"),
        schema=FindingsDraft, max_tokens=6000)


def _verify(ctx, items: list[Finding], stage: str, dim: str | None):
    ok, bad = [], []
    for f in items:
        if dim:
            f.dimension = dim
        src = ctx.store.get(f.source_id)
        if not src:
            bad.append(RejectedClaim(stage=stage, claim=f.claim, source_id=f.source_id, reason="unknown source id"))
            continue
        good, score = verify_quote(f.evidence_quote, src.text)
        if good:
            f.source_id = src.id
            ok.append(f)
        else:
            bad.append(RejectedClaim(stage=stage, claim=f.claim, source_id=src.id,
                                     reason=f"quote not found in source (match {score:.0f}/100)"))
    return ok, bad


def _contact_check(ctx, profile, rejected) -> ContactCheck:
    e = ctx.email
    if not e.sender_name:
        return ContactCheck(publicly_verified=False, summary="No sender name provided.", findings=[])
    co = profile.company_short_name or profile.company_legal_name
    qs = [f'"{e.sender_name}" {co}', f'"{e.sender_name}" "{profile.company_legal_name}"',
          f'"{e.sender_name}" {co} {e.title.split(",")[0]}']
    urls = []
    for q in qs:
        for r in ctx.web.search(q, 5):
            if r.url not in urls:
                urls.append(r.url)
    surname = e.sender_name.split()[-1].lower()
    pages = ctx.web.fetch_many(urls[:8])
    for url, p in pages.items():
        # deterministic pre-filter: page must contain the surname and a company name
        if p and surname in p.text.lower() and any(n.lower() in p.text.lower()
                                                    for n in [co, profile.company_legal_name] if n):
            ctx.store.add(url, p.title, p.text, "web", {"contact"})
    srcs = ctx.store.with_tag("contact")
    ctx.log.info(f"contact check: {len(urls)} results, {len(srcs)} pages mention name + company")
    if not srcs:
        return ContactCheck(publicly_verified=False, findings=[],
                            summary=(f"No public source found that names {e.sender_name} at {co}. Title and division "
                                     f"are taken from the contact form only and must be confirmed."))
    draft = ctx.llm.structured(
        system=CONTACT_SYSTEM.format(vendor=ctx.settings.vendor_name),
        user=(f"Person: {e.sender_name}, claimed title '{e.title}', company {profile.company_legal_name} ({co}).\n"
              "Set dimension='contact' on every finding.\n\n"
              + _sources_block(ctx, srcs, [surname, e.sender_name.split()[0].lower()])),
        schema=ContactDraft)
    ok, bad = _verify(ctx, draft.findings, "research/contact", "contact")
    # a contact finding must actually contain the surname in its quote
    kept = [f for f in ok if surname in f.evidence_quote.lower()]
    for f in ok:
        if f not in kept:
            bad.append(RejectedClaim(stage="research/contact", claim=f.claim, source_id=f.source_id,
                                     reason="quote does not name the person"))
    rejected += bad
    summary = draft.summary if kept else (f"Pages mentioning the name were found, but none verifiably tie "
                                          f"{e.sender_name} to {co}. Treat title as unverified.")
    return ContactCheck(publicly_verified=bool(kept), summary=summary, findings=kept)
