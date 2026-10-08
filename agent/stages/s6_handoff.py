"""Stage 6 — AE handoff. The LLM writes the narrative parts; scores, case study and
GTM motion are copied from earlier stages by code so they cannot drift."""
from __future__ import annotations

from ..prompts import HANDOFF_SYSTEM
from ..schemas import Handoff, HandoffDraft


def run(ctx) -> Handoff:
    s, profile = ctx.settings, ctx.out["profile"]
    q, r, cs, gtm, seq = (ctx.out[k] for k in ("qualification", "research", "case_study", "gtm", "sequence"))
    findings = {f.id: f for f in r.findings}
    user = (
        f"<email>\n{ctx.email.raw}\n</email>\n"
        f"<qualification>MEDDPICC {q.meddpicc_points}/{q.meddpicc_max}; ICP fit {q.fit_score}/100; {q.priority}\n"
        f"Known:\n- " + "\n- ".join(q.known) + "\nMissing:\n- " + "\n- ".join(q.missing)
        + f"\nAmbiguities: {'; '.join(q.ambiguities)}\nRisks: {'; '.join(q.risks)}\n</qualification>\n"
        f"<sender_verification>{r.contact.summary}</sender_verification>\n"
        f"<verified_findings>\n" + "\n".join(f"{f.id} [{f.dimension}] {f.claim} ({f.date_or_period}) — {f.so_what}"
                                            for f in r.findings) + "\n</verified_findings>\n"
        f"<positioning>{r.positioning.thesis}</positioning>\n"
        f"<stakeholders>{[x.model_dump() for x in r.positioning.likely_stakeholders]}</stakeholders>\n"
        f"<case_study>{cs.primary.url if cs.primary else 'none'} — "
        f"{cs.explanation.why_strongest if cs.explanation else ''}</case_study>\n"
        f"<go_to_market>{gtm.motion}; partner type {gtm.recommended_partner_type}; shortlist "
        f"{[x.name for x in gtm.shortlist]}; {gtm.rationale.rationale if gtm.rationale else ''}</go_to_market>\n"
        f"<sequence_status>{len(seq.draft.emails)}-email sequence; email 1 goal: "
        f"{seq.draft.emails[0].goal if seq.draft.emails else ''}</sequence_status>")
    draft = ctx.llm.structured(system=HANDOFF_SYSTEM.format(vendor=s.vendor_name), user=user, schema=HandoffDraft)
    issues = []
    good = [t for t in draft.top_findings if t.finding_id in findings]
    if len(good) != len(draft.top_findings):
        issues.append("handoff cited finding ids that do not exist; removed")
    draft.top_findings = good[:3]
    if len(draft.top_findings) < 3:
        issues.append(f"only {len(draft.top_findings)} verified top findings available")
    for st in draft.others_involved:
        st.finding_ids = [i for i in st.finding_ids if i in findings]
        if st.basis == "verified" and not st.finding_ids:
            st.basis = "inferred_from_role"
    return Handoff(draft=draft, top_findings_resolved=[findings[t.finding_id] for t in draft.top_findings],
                   issues=issues)
