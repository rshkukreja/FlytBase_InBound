"""Stage 5 — partner identification and go-to-market motion.

Discovery (code): partner pages from homepage navigation links, sitemaps and site search.
Extraction (LLM): partner name / country / tier / type, each with a verified quote.
Decision (code, rules below): deal complexity x partner availability -> motion.
Shortlist (code): points from config.PARTNER_SHORTLIST_POINTS.
Rationale (LLM): explains the decision; cannot change it.
"""
from __future__ import annotations

from urllib.parse import urlparse

from ..config import COMPLEXITY_THRESHOLDS, PARTNER_PATH_HINTS, PARTNER_SHORTLIST_POINTS, TOP_TIERS
from ..evidence import verify_quote
from ..prompts import GTM_SYSTEM, PARTNER_EXTRACT_SYSTEM
from ..schemas import GtmRationale, GtmRecommendation, PartnerExtraction, PartnerScored
from ..tools.site import is_default_language, same_site, sitemap_urls

AUTONOMY_WORDS = ("autonom", "dock", "bvlos", "beyond visual", "remote", "drone-in-a-box", "unattended")


def _partner_page_urls(ctx) -> list[str]:
    base = ctx.settings.vendor_site
    found = []
    home = ctx.web.fetch(base + "/")
    if home:
        found += [u for u in home.links if same_site(u, base)]
    found += sitemap_urls(ctx.web, base, ctx.log)
    domain = urlparse(base).netloc
    for q in [f"site:{domain} partner program", f"site:{domain} partners directory"]:
        found += [r.url for r in ctx.web.search(q, 6)]
    content_sections = {"blog", "webinars", "tv", "case-studies", "events", "author", "authors", "industries"}
    cands = []
    for u in dict.fromkeys(found):
        segs = [x for x in urlparse(u).path.lower().split("/") if x]
        if not (same_site(u, base) and is_default_language(u) and segs):
            continue
        if segs[0] in content_sections:  # e.g. a blog post whose slug mentions a partner
            continue
        if any(seg.startswith(h) for seg in segs for h in PARTNER_PATH_HINTS):
            cands.append(u.split("#")[0].split("?")[0])
    # directory-like pages first (shortest paths), max 4
    cands = sorted(dict.fromkeys(cands), key=lambda u: (urlparse(u).path.count("/"), len(u)))
    return cands[:4]


def complexity(ctx) -> tuple[str, dict]:
    p, q, r = ctx.out["profile"], ctx.out["qualification"], ctx.out["research"]
    text = (ctx.email.raw + " " + p.use_case).lower()
    factors = {
        "multi_site (>=2 sites in scope)": bool(p.site_count and p.site_count >= 2),
        "continuous_ops (24/7)": any("24" in c or "continuous" in c.lower() for c in p.operating_conditions),
        "enterprise_buyer (files annual report with a securities regulator)":
            bool(r.company_identifiers.get("annual_report")),
        "harsh_or_remote_environment": len(p.operating_conditions) >= 2,
        "autonomous/BVLOS operations (regulatory approval likely)": any(w in text for w in AUTONOMY_WORDS),
        "strategic_referral (existing-customer relationship to protect)": bool(p.referral_entities),
        "senior_buyer (VP/Head or above)": p.seniority in ("C-level", "VP/Head"),
    }
    n = sum(factors.values())
    level = "high" if n >= COMPLEXITY_THRESHOLDS["high"] else "medium" if n >= COMPLEXITY_THRESHOLDS["medium"] else "low"
    return level, factors


def decide(level: str, in_country: list, in_region: list) -> tuple[str, str, list[str]]:
    trace = [f"Deal complexity: {level}.",
             f"Partners in lead's country: {len(in_country)}; elsewhere in region: {len(in_region)}."]
    if level == "high":
        if in_country or in_region:
            trace.append("Rule: high complexity + regional partner available -> AE owns the account and commercials; "
                         "partner delivers hardware, installation, permits and local support.")
            return "Hybrid: AE-led, partner-delivered", "System Integrator / Solution Provider", trace
        trace.append("Rule: high complexity + no regional partner -> direct AE engagement; recruit a local integrator.")
        return "Direct AE engagement", "System Integrator (to recruit)", trace
    if level == "medium":
        if in_country:
            trace.append("Rule: medium complexity + in-country partner -> partner-led with AE oversight.")
            return "Partner-led with AE oversight", "System Integrator / Solution Provider", trace
        trace.append("Rule: medium complexity + no in-country partner -> direct AE engagement.")
        return "Direct AE engagement", "Reseller (for hardware)", trace
    if in_country or in_region:
        trace.append("Rule: low complexity + regional partner -> partner-led.")
        return "Partner-led", "Reseller", trace
    trace.append("Rule: low complexity + no partner -> direct (inside sales).")
    return "Direct (inside sales)", "Reseller", trace


def run(ctx) -> GtmRecommendation:
    s, log, profile = ctx.settings, ctx.log, ctx.out["profile"]
    pages_urls = _partner_page_urls(ctx)
    log.info(f"partner pages found: {pages_urls or 'none'}")
    pages = ctx.web.fetch_many(pages_urls)
    srcs = []
    for u, p in pages.items():
        if p and len(p.text) > 200:
            srcs.append(ctx.store.add(u, p.title, p.text, "vendor", {"partners"}))

    partners, rejected = [], 0
    if srcs:
        block = "\n".join(f'<source id="{x.id}" url="{x.url}">\n{x.text[:60000]}\n</source>' for x in srcs)
        ext = ctx.llm.structured(system=PARTNER_EXTRACT_SYSTEM.format(vendor=s.vendor_name),
                                 user=f"<sources>\n{block}\n</sources>", schema=PartnerExtraction, max_tokens=8000)
        all_text = "\n".join(x.text for x in srcs)
        for pt in ext.partners:
            ok = verify_quote(pt.evidence_quote, all_text)[0] if len(pt.evidence_quote) >= 20 else False
            # short directory cards: accept if name and country both appear in the page text
            ok = ok or (pt.name.lower() in all_text.lower() and pt.country.lower() in all_text.lower())
            if ok:
                partners.append(pt)
            else:
                rejected += 1
    log.info(f"partners extracted: {len(partners)} verified, {rejected} rejected")

    # reference-account proof: partner named in the matched case-study content
    cs = ctx.out["case_study"]
    proof_text = " ".join(ctx.store.get(c.source_id).text.lower() for c in ([cs.primary] if cs.primary else [])
                          + cs.supporting if c and c.source_id)
    country = profile.hq_country.lower()
    region = {m.lower() for m in profile.region_markets}
    pts = PARTNER_SHORTLIST_POINTS
    scored = []
    for pt in partners:
        pc = pt.country.lower().strip()
        same_c = bool(country) and (pc == country or country in pc)
        same_r = (pc in region) or any(r in pc for r in region if len(r) > 3)
        name_core = pt.name.lower().replace(" ", "")
        proven = bool(proof_text) and (pt.name.lower() in proof_text or name_core in proof_text.replace(" ", ""))
        types = " ".join(pt.types).lower()
        p, reasons = 0.0, []
        if same_c: p += pts["same_country"]; reasons.append(f"based in {pt.country}")
        elif same_r: p += pts["same_region"]; reasons.append(f"in region ({pt.country})")
        if proven: p += pts["proven_on_reference_account"]; reasons.append("named in the matched reference deployment")
        if "integrator" in types: p += pts["system_integrator"]; reasons.append("system integrator")
        if "solution" in types: p += pts["solution_provider"]; reasons.append("solution provider")
        if pt.tier.lower() in TOP_TIERS: p += pts["top_tier"]; reasons.append(f"{pt.tier} tier")
        scored.append(PartnerScored(name=pt.name, country=pt.country, tier=pt.tier, types=pt.types,
                                    source_url=srcs[0].url if srcs else "", same_country=same_c,
                                    same_region=same_r or same_c, proven_on_reference_account=proven,
                                    points=p, reasons=reasons))
    in_country = [x for x in scored if x.same_country]
    in_region = [x for x in scored if x.same_region and not x.same_country]

    level, factors = complexity(ctx)
    motion, ptype, trace = decide(level, in_country, in_region)
    shortlist = sorted([x for x in scored if x.same_region or x.proven_on_reference_account],
                       key=lambda x: x.points, reverse=True)[:3]
    if shortlist:
        trace.append("Shortlist (points): " + "; ".join(f"{x.name} {x.points:g}" for x in shortlist))
    log.info(f"motion: {motion} (complexity {level}, {sum(factors.values())}/{len(factors)} factors)")

    rationale = ctx.llm.structured(
        system=GTM_SYSTEM.format(vendor=s.vendor_name),
        user=(f"Lead: {ctx.email.title}, {profile.company_legal_name}, {profile.hq_country} ({profile.region}). "
              f"{profile.use_case}; {profile.site_count or '?'} sites.\nComplexity factors: {factors}\n"
              f"Decision trace:\n- " + "\n- ".join(trace) + f"\nRecommended partner type: {ptype}\n"
              f"Shortlist: {[x.model_dump() for x in shortlist]}"),
        schema=GtmRationale)
    return GtmRecommendation(motion=motion, complexity_level=level, complexity_factors=factors, decision_trace=trace,
                             partner_pages=pages_urls, partners_found=len(partners), partners_rejected=rejected,
                             in_country=len(in_country), in_region=len(in_region), recommended_partner_type=ptype,
                             shortlist=shortlist, rationale=rationale)
