"""Stage 1 — qualification (MEDDPICC + ICP fit).

LLM: judges each MEDDPICC element and each ICP criterion, quoting the email.
Code: verifies every quote against the email, downgrades unverified judgements,
computes fit score, MEDDPICC completeness and priority from config weights."""
from __future__ import annotations

import json

from ..config import (FIT_CREDIT, ICP_CRITERIA, MEDDPICC_CREDIT, MEDDPICC_ELEMENTS, PRIORITY_RULES)
from ..evidence import verify_quote
from ..prompts import QUALIFY_SYSTEM
from ..schemas import ElementAssessment, FitLine, Qualification, QualificationDraft


def run(ctx) -> Qualification:
    e, profile = ctx.email, ctx.out["profile"]
    icp = [{k: c[k] for k in ("id", "label", "description")} for c in ICP_CRITERIA]
    user = (f"<email>\n{e.raw}\n</email>\n\n<lead_profile>\n{profile.model_dump_json(indent=1)}\n</lead_profile>\n\n"
            f"<icp_criteria>\n{json.dumps(icp, indent=1)}\n</icp_criteria>\n\n"
            f"MEDDPICC elements: {', '.join(MEDDPICC_ELEMENTS)}")
    draft = ctx.llm.structured(system=QUALIFY_SYSTEM.format(vendor=ctx.settings.vendor_name, today=ctx.run_date),
                               user=user, schema=QualificationDraft)
    return score(draft, e.raw, ctx.log)


def score(draft: QualificationDraft, email_text: str, log=None) -> Qualification:
    flags: list[str] = []

    # --- MEDDPICC: one row per element, quotes must come from the email ---------
    by_el = {}
    for el in draft.elements:
        by_el.setdefault(el.element, el)
    elements = []
    for name in MEDDPICC_ELEMENTS:
        el = by_el.get(name) or ElementAssessment(element=name, status="unknown", what_we_know="",
                                                  evidence_quote="", what_is_missing="Not assessed",
                                                  discovery_questions=[])
        if el.status != "unknown":
            ok, s = verify_quote(el.evidence_quote, email_text) if el.evidence_quote else (False, 0)
            if not ok:
                flags.append(f"{name}: quote not found in email (score {s:.0f}); downgraded "
                             f"{el.status} -> {'partial' if el.status == 'confirmed' else 'unknown'}")
                el.status = "partial" if el.status == "confirmed" else "unknown"
        elements.append(el)
    pts = sum(MEDDPICC_CREDIT[el.status] for el in elements)
    mx = 2 * len(MEDDPICC_ELEMENTS)

    # --- ICP fit -------------------------------------------------------------------
    by_c = {f.criterion_id: f for f in draft.fit_signals}
    fit_lines, fit_total = [], 0.0
    for c in ICP_CRITERIA:
        f = by_c.get(c["id"])
        assessment = f.assessment if f else "unknown"
        quote = f.evidence_quote if f else ""
        verified = bool(quote) and verify_quote(quote, email_text)[0]
        if assessment in ("met", "partial") and not verified:
            flags.append(f"ICP '{c['id']}': '{assessment}' not backed by a verifiable email quote; counted as unknown")
            assessment = "unknown"
        points = c["weight"] * FIT_CREDIT[assessment]
        fit_total += points
        fit_lines.append(FitLine(criterion_id=c["id"], label=c["label"], weight=c["weight"], assessment=assessment,
                                 points=points, evidence_quote=quote, quote_verified=verified,
                                 note=f.note if f else "not assessed"))
    max_fit = sum(c["weight"] for c in ICP_CRITERIA)
    fit_score = round(100 * fit_total / max_fit)
    pct = round(100 * pts / mx)

    for min_fit, min_pct, label, meaning in PRIORITY_RULES:
        if fit_score >= min_fit and pct >= min_pct:
            break
    reasoning = [f"ICP fit {fit_score}/100 from {sum(1 for l in fit_lines if l.points)} of {len(fit_lines)} "
                 f"criteria with email evidence.",
                 f"MEDDPICC completeness {pts}/{mx} ({pct}%): typical for a first inbound touch; "
                 f"discovery must close the gaps listed below.",
                 f"Rule applied: fit >= {min_fit} -> {label}."]
    strong = [l.label for l in fit_lines if l.assessment == "met"]
    if strong:
        reasoning.append("Strongest signals: " + ", ".join(strong) + ".")

    known = [f"{el.element}: {el.what_we_know}" for el in elements if el.status != "unknown" and el.what_we_know]
    missing = [f"{el.element}: {el.what_is_missing}" for el in elements if el.status != "confirmed"]
    if log:
        log.info(f"fit {fit_score}/100, MEDDPICC {pts}/{mx}, priority {label}; {len(flags)} verification flags")
    return Qualification(framework_rationale=draft.framework_rationale, elements=elements, verification_flags=flags,
                         fit=fit_lines, fit_score=fit_score, meddpicc_points=pts, meddpicc_max=mx, meddpicc_pct=pct,
                         priority=label, priority_meaning=meaning, priority_reasoning=reasoning, known=known,
                         missing=missing, ambiguities=draft.ambiguities, risks=draft.risks)
