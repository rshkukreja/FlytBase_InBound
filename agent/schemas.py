"""Typed contracts between stages. Every LLM call returns one of the *Draft / LLM
models below (validated by Pydantic); the code then post-processes them into the
final stage outputs, which are what later stages and the report consume."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MeddpiccElement = Literal[
    "Metrics", "Economic Buyer", "Decision Criteria", "Decision Process",
    "Paper Process", "Identify Pain", "Champion", "Competition",
]
Dimension = Literal["org_structure", "budget_signals", "news", "stakeholder_letters", "contact"]


# ============================ Stage 0: intake ============================
class InboundEmail(BaseModel):
    sender_name: str
    sender_email: str
    title: str
    company: str
    subject: str
    context: str
    raw: str

    @property
    def domain(self) -> str:
        return self.sender_email.split("@")[-1].lower() if "@" in self.sender_email else ""


class LeadProfile(BaseModel):
    company_legal_name: str = Field(description="Full legal company name as best known (fix accents/spelling).")
    company_short_name: str = Field(description="Common short name / acronym.")
    company_name_variants: list[str] = Field(description="Other names the company is filed or reported under, incl. English translations and stock tickers you are confident about.")
    hq_country: str
    region: str = Field(description="Commercial sales region, e.g. LATAM, EMEA, APAC, North America.")
    region_markets: list[str] = Field(description="Country names in the same commercial region (include the lead's own country and the region label itself). Used for partner/case-study matching.")
    industry: str
    sub_industry: str
    use_case: str
    asset_or_site_types: list[str]
    site_count: Optional[int] = Field(default=None, description="Number of sites in scope if stated, else null.")
    site_locations: list[str]
    operating_conditions: list[str] = Field(description="Conditions stated or directly implied by the email, e.g. '24/7', 'desert', 'remote', 'hazardous'.")
    stated_pains: list[str]
    referral_entities: list[str] = Field(description="Organisations the email says referred the lead. Exact names. Empty list if none.")
    timeline_signals: list[str]
    budget_signals: list[str]
    seniority: Literal["C-level", "VP/Head", "Director", "Manager", "Individual contributor", "Unknown"]
    matching_keywords: list[str] = Field(description="8-15 short lowercase keywords for finding relevant vendor content: industry, assets, use case, environment.")


# ========================= Stage 1: qualification =========================
class ElementAssessment(BaseModel):
    element: MeddpiccElement
    status: Literal["confirmed", "partial", "unknown"]
    what_we_know: str = Field(description="What the email tells us for this element. Empty if unknown.")
    evidence_quote: str = Field(description="Verbatim span copied from the inbound email. Empty string if unknown.")
    what_is_missing: str
    discovery_questions: list[str] = Field(description="1-3 questions that would close the gap, phrased for a senior operations buyer.")


class FitSignal(BaseModel):
    criterion_id: str
    assessment: Literal["met", "partial", "not_met", "unknown"]
    evidence_quote: str = Field(description="Verbatim span from the inbound email. Empty if unknown.")
    note: str


class QualificationDraft(BaseModel):
    framework_rationale: str = Field(description="Why MEDDPICC fits THIS lead (2-4 sentences, specific to the lead), and why BANT/SPICED would be weaker here.")
    elements: list[ElementAssessment]
    fit_signals: list[FitSignal]
    ambiguities: list[str] = Field(description="Statements in the email that are ambiguous and must be confirmed, e.g. dates relative to today, site counts vs known footprint.")
    risks: list[str]


class FitLine(BaseModel):
    criterion_id: str
    label: str
    weight: int
    assessment: str
    points: float
    evidence_quote: str
    quote_verified: bool
    note: str


class Qualification(BaseModel):
    framework: str = "MEDDPICC"
    framework_rationale: str
    elements: list[ElementAssessment]
    verification_flags: list[str]
    fit: list[FitLine]
    fit_score: int
    meddpicc_points: int
    meddpicc_max: int
    meddpicc_pct: int
    priority: str
    priority_meaning: str
    priority_reasoning: list[str]
    known: list[str]
    missing: list[str]
    ambiguities: list[str]
    risks: list[str]


# =========================== Stage 2: research ===========================
class PlannedQuery(BaseModel):
    dimension: Literal["org_structure", "budget_signals", "news", "stakeholder_letters"]
    query: str
    why: str


class QueryPlan(BaseModel):
    queries: list[PlannedQuery]


class Finding(BaseModel):
    id: str = Field(default="", description="Leave empty; assigned by the system.")
    dimension: str = Field(default="", description="Set to the dimension named in the request.")  # overwritten by code
    claim: str = Field(description="One factual statement, in your own words, fully supported by the quote.")
    source_id: str = Field(description="The [S#] id of the source the quote comes from.")
    evidence_quote: str = Field(description="Verbatim span (15-60 words) copied exactly from that source's text.")
    date_or_period: str = Field(default="", description="Date/period the fact refers to, or '' if not stated.")
    so_what: str = Field(default="", description="Why this matters for selling autonomous drone operations to this lead.")


class FindingsDraft(BaseModel):
    findings: list[Finding] = Field(default_factory=list)


class ContactDraft(BaseModel):
    summary: str
    findings: list[Finding]


class ValueHook(BaseModel):
    hook: str
    finding_ids: list[str]
    how_to_say_it: str


class Stakeholder(BaseModel):
    name_or_role: str
    why_involved: str
    basis: Literal["verified", "inferred_from_role"]
    finding_ids: list[str] = Field(default_factory=list)


class PositioningDraft(BaseModel):
    thesis: str = Field(description="2-3 sentence positioning recommendation grounded in the findings.")
    value_hooks: list[ValueHook]
    likely_stakeholders: list[Stakeholder]
    risks_and_objections: list[str]
    avoid: list[str]


class SourceRef(BaseModel):
    id: str
    url: str
    title: str
    kind: str
    tags: list[str]


class RejectedClaim(BaseModel):
    stage: str
    claim: str
    source_id: str
    reason: str


class ContactCheck(BaseModel):
    publicly_verified: bool
    summary: str
    findings: list[Finding]


class Research(BaseModel):
    company_identifiers: dict
    sources: list[SourceRef]
    findings: list[Finding]
    rejected: list[RejectedClaim]
    contact: ContactCheck
    positioning: PositioningDraft
    coverage: dict
    gaps: list[str]
    queries_run: list[PlannedQuery]


# ========================= Stage 4: case study ==========================
class RubricScore(BaseModel):
    source_id: str
    industry_fit: int = Field(ge=0, le=5)
    use_case_fit: int = Field(ge=0, le=5)
    environment_fit: int = Field(ge=0, le=5)
    proof_strength: int = Field(ge=0, le=5, description="Concrete outcomes / metrics / named customer.")
    evidence_quote: str = Field(description="Verbatim span from that page supporting the scores.")
    summary: str = Field(description="One line: what this content shows.")


class RubricBatch(BaseModel):
    scores: list[RubricScore]


class CaseStudyExplanation(BaseModel):
    why_strongest: str
    how_to_use_in_sales: str
    gaps_vs_lead: str


class ContentCandidate(BaseModel):
    url: str
    section: str
    title: str = ""
    source_id: str = ""
    discovered_via: list[str] = Field(default_factory=list)
    slug_score: float = 0
    referral_mentions: int = 0
    referral_entities_found: list[str] = Field(default_factory=list)
    industry_share: float = 0
    region_hit: float = 0
    section_weight: float = 0
    rubric: Optional[RubricScore] = None
    rubric_quote_verified: Optional[bool] = None
    components: dict = Field(default_factory=dict)
    final_score: float = 0


class CaseStudyMatch(BaseModel):
    referral_entities: list[str]
    referral_linked_urls: list[str]
    discovery_stats: dict
    ranked: list[ContentCandidate]
    primary: Optional[ContentCandidate]
    supporting: list[ContentCandidate]
    explanation: Optional[CaseStudyExplanation]


# =========================== Stage 5: partners ===========================
class PartnerLLM(BaseModel):
    name: str
    country: str
    tier: str = ""
    types: list[str] = Field(default_factory=list)
    evidence_quote: str = Field(description="Verbatim span from the page that contains the partner name.")


class PartnerExtraction(BaseModel):
    partners: list[PartnerLLM]


class PartnerScored(BaseModel):
    name: str
    country: str
    tier: str
    types: list[str]
    source_url: str
    same_country: bool
    same_region: bool
    proven_on_reference_account: bool
    points: float
    reasons: list[str]


class GtmRationale(BaseModel):
    rationale: str
    ae_role: str
    partner_role: str
    risks: list[str]


class GtmRecommendation(BaseModel):
    motion: str
    complexity_level: str
    complexity_factors: dict
    decision_trace: list[str]
    partner_pages: list[str]
    partners_found: int
    partners_rejected: int
    in_country: int
    in_region: int
    recommended_partner_type: str
    shortlist: list[PartnerScored]
    rationale: Optional[GtmRationale]


# =========================== Stage 3: sequence ===========================
class SequenceEmail(BaseModel):
    step: int
    send_timing: str = Field(description="e.g. 'Day 0 (within 2 business hours)', 'Day 3 if no reply'.")
    trigger: str
    goal: str
    unknowns_targeted: list[MeddpiccElement]
    finding_ids: list[str] = Field(description="Ids (F##) of research findings this email uses.")
    materials: list[str] = Field(description="URLs of vendor materials referenced.")
    subject: str
    body: str
    why_this_step: str = Field(description="Why this email comes at this point in the sequence.")


class ReplyBranch(BaseModel):
    if_reply: str
    next_move: str


class SequenceDraft(BaseModel):
    strategy: str
    tone_notes: str
    emails: list[SequenceEmail]
    reply_branches: list[ReplyBranch]


class Sequence(BaseModel):
    draft: SequenceDraft
    coverage: dict
    issues: list[str]
    revisions: int


# =========================== Stage 6: handoff ============================
class TopFinding(BaseModel):
    finding_id: str
    why_ae_should_care: str


class NextStep(BaseModel):
    action: str
    owner: str
    timing: str
    rationale: str
    agenda: list[str]


class HandoffDraft(BaseModel):
    who_he_is: str
    what_he_cares_about: list[str]
    others_involved: list[Stakeholder]
    top_findings: list[TopFinding]
    next_step: NextStep
    first_call_questions: list[str]
    watch_outs: list[str]


class Handoff(BaseModel):
    draft: HandoffDraft
    top_findings_resolved: list[Finding]
    issues: list[str]
