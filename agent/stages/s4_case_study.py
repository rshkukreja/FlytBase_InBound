"""Stage 4 — case study and material matching.

Discovery (code): robots.txt -> sitemaps -> all default-language URLs, plus site-restricted
web searches for each referral entity named in the email and for the lead's industry/use case.
Pre-ranking (code): URL-slug keyword overlap, then page-level features: referral mentions,
share of lead keywords, region mention, section credibility.
Judgement (LLM): rubric scores for the top N pages, each backed by a verified quote.
Final score (code): weighted sum from config.CASE_STUDY_FINAL_WEIGHTS.
Nothing here names a customer: the referral entity comes from the parsed email.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from ..config import (CASE_STUDY_FINAL_WEIGHTS, CONTENT_SECTION_WEIGHTS, DEFAULT_SECTION_WEIGHT, SKIP_SECTIONS)
from ..evidence import verify_quote
from ..prompts import CASE_EXPLAIN_SYSTEM, RUBRIC_SYSTEM
from ..schemas import CaseStudyExplanation, CaseStudyMatch, ContentCandidate, RubricBatch
from ..tools.site import is_default_language, same_site, section_of, sitemap_urls, slug_tokens

STOP = {"the", "and", "for", "with", "inc", "plc", "ltd", "llc", "group", "company", "corp", "sa", "de", "of"}


def _tokens(phrases) -> set[str]:
    out = set()
    for p in phrases:
        for t in re.split(r"[^a-z0-9]+", p.lower()):
            if len(t) >= 3 and t not in STOP:
                out.add(t)
    return out


def _match(tok: str, kw: str) -> bool:
    """Light stemming: 'mining' ~ 'mine', 'inspections' ~ 'inspection'."""
    if tok == kw:
        return True
    n = min(len(tok), len(kw))
    return n >= 4 and tok[:max(4, n - 2)] == kw[:max(4, n - 2)]


def run(ctx) -> CaseStudyMatch:
    s, log, profile = ctx.settings, ctx.log, ctx.out["profile"]
    base = s.vendor_site
    referrals = [r for r in profile.referral_entities if r.strip()]
    ref_tokens = _tokens(referrals)
    kw_tokens = _tokens(profile.matching_keywords + [profile.industry, profile.sub_industry, profile.use_case])
    region_terms = [t.lower() for t in profile.region_markets if len(t) >= 3]
    country = profile.hq_country.lower()

    # ---- discovery ---------------------------------------------------------------
    all_urls = sitemap_urls(ctx.web, base, log)
    urls = [u for u in all_urls if same_site(u, base) and is_default_language(u)
            and section_of(u) not in SKIP_SECTIONS]
    candidates: dict[str, ContentCandidate] = {}
    for u in urls:
        candidates[u] = ContentCandidate(url=u, section=section_of(u), discovered_via=["sitemap"])
    domain = urlparse(base).netloc
    site_queries = [f'site:{domain} "{r}"' for r in referrals]
    site_queries.append(f"site:{domain} {profile.industry} {profile.use_case}")
    site_queries.append(f"site:{domain} {profile.sub_industry} case study")
    search_hits = 0
    for q in site_queries:
        for r in ctx.web.search(q, 8):
            if same_site(r.url, base) and is_default_language(r.url):
                u = r.url.split("#")[0].split("?")[0].rstrip("/")
                c = candidates.setdefault(u, ContentCandidate(url=u, section=section_of(u)))
                c.discovered_via.append(f"search: {q}")
                search_hits += 1
    log.info(f"content pool: {len(candidates)} default-language pages ({search_hits} site-search hits)")

    # ---- slug pre-rank ---------------------------------------------------------------
    for c in candidates.values():
        toks = slug_tokens(c.url)
        ref = sum(1 for t in toks for k in ref_tokens if _match(t, k))
        kw = sum(1 for t in toks for k in kw_tokens if _match(t, k))
        reg = sum(1 for t in toks for r in region_terms if _match(t, r))
        searched = any(v.startswith("search") for v in c.discovered_via)
        c.section_weight = CONTENT_SECTION_WEIGHTS.get(c.section, DEFAULT_SECTION_WEIGHT)
        c.slug_score = 3 * ref + kw + reg + 2 * searched + c.section_weight
    pool = sorted(candidates.values(), key=lambda c: c.slug_score, reverse=True)
    # always consider every official case study page, then the best of the rest
    official = [c for c in pool if CONTENT_SECTION_WEIGHTS.get(c.section, 0) >= 1.0 and c.slug_score > 0][:15]
    rest = [c for c in pool if c not in official]
    chosen = (official + rest)[: s.max_content_candidates]
    chosen = list({c.url: c for c in chosen}.values())

    # ---- fetch + page features --------------------------------------------------------
    pages = ctx.web.fetch_many([c.url for c in chosen])
    fetched = []
    for c in chosen:
        p = pages.get(c.url)
        if not p or len(p.text) < 200:
            continue
        ctx.vendor_pages[c.url] = p
        src = ctx.store.add(c.url, p.title, p.text, "vendor", {"vendor_content"})
        c.source_id, c.title = src.id, p.title
        low = (p.title + "\n" + p.text).lower()
        for r in referrals:
            n = low.count(r.lower())
            if n:
                c.referral_mentions += n
                c.referral_entities_found.append(r)
        words = set(re.findall(r"[a-z0-9]+", low))
        hits = sum(1 for k in kw_tokens if any(_match(w, k) for w in words if w[:3] == k[:3]))
        c.industry_share = round(hits / max(1, len(kw_tokens)), 3)
        c.region_hit = 1.0 if country and country in low else (0.6 if any(r in low for r in region_terms) else 0.0)
        fetched.append(c)
    log.info(f"fetched {len(fetched)} candidate pages; "
             f"{sum(1 for c in fetched if c.referral_mentions)} mention a referral entity")

    def pre(c):
        return (0.4 * min(c.referral_mentions, 3) / 3 + 0.3 * c.industry_share + 0.2 * c.region_hit
                + 0.1 * c.section_weight)
    top = sorted(fetched, key=pre, reverse=True)[: s.rubric_top_n]

    # ---- LLM rubric -----------------------------------------------------------------------
    if top:
        block = "\n".join(
            f'<source id="{c.source_id}" section="{c.section}" url="{c.url}" title="{c.title}">\n'
            f'{ctx.store.excerpt(c.source_id, list(kw_tokens) + referrals, 4000)}\n</source>' for c in top)
        rubric = ctx.llm.structured(
            system=RUBRIC_SYSTEM.format(vendor=s.vendor_name),
            user=(f"<lead>\nIndustry: {profile.industry} / {profile.sub_industry}\nUse case: {profile.use_case}\n"
                  f"Sites: {profile.site_count or '?'} x {', '.join(profile.asset_or_site_types)} in "
                  f"{', '.join(profile.site_locations)}\nConditions: {', '.join(profile.operating_conditions)}\n"
                  f"Pains: {'; '.join(profile.stated_pains)}\n</lead>\n<sources>\n{block}\n</sources>"),
            schema=RubricBatch)
        by_id = {r.source_id.strip("[] "): r for r in rubric.scores}
        for c in top:
            r = by_id.get(c.source_id)
            if not r:
                continue
            c.rubric = r
            c.rubric_quote_verified = verify_quote(r.evidence_quote, ctx.store.get(c.source_id).text)[0]

    # ---- final score --------------------------------------------------------------------------
    w = CASE_STUDY_FINAL_WEIGHTS
    for c in fetched:
        rub = 0.0
        if c.rubric:
            rub = (c.rubric.industry_fit + c.rubric.use_case_fit + c.rubric.environment_fit
                   + c.rubric.proof_strength) / 20
            if not c.rubric_quote_verified:
                rub *= 0.5  # unverified judgement counts half
        comp = {"rubric": rub, "referral": min(c.referral_mentions, 3) / 3, "region": c.region_hit,
                "industry": c.industry_share, "section": c.section_weight}
        c.components = {k: round(v, 3) for k, v in comp.items()}
        c.final_score = round(100 * sum(w[k] * comp[k] for k in w), 1)
    ranked = sorted(fetched, key=lambda c: c.final_score, reverse=True)
    primary = ranked[0] if ranked else None
    supporting = [c for c in ranked[1:] if c.final_score >= 0.6 * (primary.final_score if primary else 0)][:2]
    referral_urls = [c.url for c in ranked if c.referral_mentions]
    if referrals and not referral_urls:
        log.warn("no vendor content mentions the referral entity; the referral claim should be confirmed")

    explanation = None
    if primary:
        detail = "\n".join(
            f"#{i+1} score={c.final_score} components={c.components} section={c.section} url={c.url} "
            f"referrals={c.referral_entities_found} rubric={c.rubric.model_dump() if c.rubric else None}"
            for i, c in enumerate(ranked[:5]))
        explanation = ctx.llm.structured(
            system=CASE_EXPLAIN_SYSTEM.format(vendor=s.vendor_name),
            user=(f"Lead: {ctx.email.title} at {profile.company_legal_name}; {profile.use_case}; "
                  f"{', '.join(profile.site_locations)}; referred by {', '.join(referrals) or 'nobody'}.\n"
                  f"Ranking:\n{detail}\n\nTop page excerpt:\n"
                  f"{ctx.store.excerpt(primary.source_id, list(kw_tokens) + referrals, 5000)}"),
            schema=CaseStudyExplanation)
        log.info(f"primary: {primary.url} (score {primary.final_score})")

    stats = {"sitemap_urls": len(all_urls), "default_language_urls": len(urls), "pool": len(candidates),
             "fetched": len(fetched), "rubric_scored": len(top), "site_search_hits": search_hits}
    return CaseStudyMatch(referral_entities=referrals, referral_linked_urls=referral_urls, discovery_stats=stats,
                          ranked=ranked[:10], primary=primary, supporting=supporting, explanation=explanation)
