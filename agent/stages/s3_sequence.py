"""Stage 3 — adaptive response sequence.

Runs after research, case-study and partner stages so emails can use them.
Generate -> deterministic checks -> revise (max N rounds) loop:
  * every email cites >= 1 valid verified finding
  * every unknown/partial MEDDPICC element is targeted by some email
  * recommended proof content is used somewhere in the sequence
  * word limits, no placeholders
"""
from __future__ import annotations

import re

from ..prompts import SEQUENCE_SYSTEM
from ..schemas import Sequence, SequenceDraft

WORD_LIMITS = {1: 170}
DEFAULT_LIMIT = 150


def check(draft: SequenceDraft, valid_ids: set[str], open_elements: list[str], proof_url: str | None) -> tuple[list[str], dict]:
    issues = []
    if not draft.emails:
        return ["sequence is empty"], {}
    for em in draft.emails:
        bad = [i for i in em.finding_ids if i not in valid_ids]
        if bad:
            issues.append(f"email {em.step}: cites unknown finding ids {bad}")
        if not [i for i in em.finding_ids if i in valid_ids]:
            issues.append(f"email {em.step}: uses no verified research finding")
        words = len(re.findall(r"\S+", em.body))
        limit = WORD_LIMITS.get(em.step, DEFAULT_LIMIT)
        if words > limit + 15:
            issues.append(f"email {em.step}: {words} words (limit {limit})")
        if re.search(r"\[[^\]]{2,40}\]|\{\{|<insert|lorem", em.body, re.I):
            issues.append(f"email {em.step}: contains a placeholder")
        if not em.unknowns_targeted:
            issues.append(f"email {em.step}: targets no MEDDPICC unknown")
    coverage = {el: [em.step for em in draft.emails if el in em.unknowns_targeted] for el in open_elements}
    uncovered = [el for el, steps in coverage.items() if not steps]
    if uncovered:
        issues.append(f"MEDDPICC gaps never targeted: {uncovered}")
    if proof_url and not any(proof_url.rstrip('/') in (" ".join(em.materials) + em.body) for em in draft.emails):
        issues.append(f"recommended proof content {proof_url} is not used in any email")
    return issues, coverage


def run(ctx) -> Sequence:
    s, profile = ctx.settings, ctx.out["profile"]
    q, r, cs, gtm = (ctx.out["qualification"], ctx.out["research"], ctx.out["case_study"], ctx.out["gtm"])
    valid_ids = {f.id for f in r.findings}
    open_elements = [el.element for el in q.elements if el.status != "confirmed"]
    proof = cs.primary
    materials = []
    for c in ([cs.primary] if cs.primary else []) + cs.supporting:
        materials.append(f"- {c.url} ({c.section}): {c.rubric.summary if c.rubric else c.title}")
    user = (
        f"<email>\n{ctx.email.raw}\n</email>\n"
        f"<qualification>\nPriority: {q.priority}\nOpen MEDDPICC elements (must all be targeted): {open_elements}\n"
        + "\n".join(f"- {el.element} [{el.status}]: missing: {el.what_is_missing} | questions: "
                    f"{'; '.join(el.discovery_questions)}" for el in q.elements if el.status != "confirmed")
        + f"\nAmbiguities to resolve: {'; '.join(q.ambiguities)}\n</qualification>\n"
        f"<verified_findings>\n" + "\n".join(f"{f.id} [{f.dimension}] {f.claim} ({f.date_or_period})"
                                            for f in r.findings) + "\n</verified_findings>\n"
        f"<positioning>\n{r.positioning.thesis}\n" + "\n".join(f"- {h.hook} ({', '.join(h.finding_ids)})"
                                                               for h in r.positioning.value_hooks)
        + f"\nAvoid: {'; '.join(r.positioning.avoid)}\n</positioning>\n"
        f"<proof_content>\n" + ("\n".join(materials) or "none found") + "\n"
        + (f"Why it fits: {cs.explanation.why_strongest}\n" if cs.explanation else "") + "</proof_content>\n"
        f"<go_to_market>\n{gtm.motion}. Partner shortlist: {', '.join(x.name for x in gtm.shortlist) or 'none'}\n"
        f"</go_to_market>\n"
        f"Sender verification: {r.contact.summary}\n"
        "Write 4 emails (reply + 3 follow-ups) and 3-5 reply branches.")
    system = SEQUENCE_SYSTEM.format(vendor=s.vendor_name, today=ctx.run_date)
    draft = ctx.llm.structured(system=system, user=user, schema=SequenceDraft, max_tokens=8000, temperature=0.4)
    issues, coverage = check(draft, valid_ids, open_elements, proof.url if proof else None)
    rounds = 0
    while issues and rounds < s.max_sequence_revisions:
        rounds += 1
        ctx.log.info(f"revision {rounds}: {len(issues)} issues -> {issues[:3]}")
        draft = ctx.llm.structured(
            system=system, schema=SequenceDraft, max_tokens=8000, temperature=0.3,
            user=user + "\n\n<previous_draft>\n" + draft.model_dump_json(indent=1) + "\n</previous_draft>\n"
                 "<checker_issues>\n- " + "\n- ".join(issues) + "\n</checker_issues>\n"
                 "Fix every issue and return the full revised sequence.")
        issues, coverage = check(draft, valid_ids, open_elements, proof.url if proof else None)
    ctx.log.info(f"sequence: {len(draft.emails)} emails, {len(issues)} open issues after {rounds} revisions")
    return Sequence(draft=draft, coverage=coverage, issues=issues, revisions=rounds)
