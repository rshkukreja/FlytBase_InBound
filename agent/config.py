"""All tunable logic lives here so a reviewer can see every rule and weight in one file.

Nothing in this file names the lead, the referrer, a case study, or a partner.
Those are always discovered at run time from the inbound email and the web.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")   # e.g. Gemini, Groq, OpenRouter, Ollama
    # Free-tier friendliness: minimum seconds between LLM calls, and how many may run at once
    llm_min_interval: float = float(os.getenv("LLM_MIN_SECONDS_BETWEEN_CALLS", "0"))
    llm_parallel: int = int(os.getenv("LLM_PARALLEL_CALLS", "4"))
    openai_min_output_tokens: int = int(os.getenv("LLM_MIN_OUTPUT_TOKENS", "16000"))
    reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "")  # optional, e.g. "low" for thinking models
    openai_response_format: str = os.getenv("LLM_RESPONSE_FORMAT", "json_schema")  # json_schema|json_object|none
    tavily_key: str = os.getenv("TAVILY_API_KEY", "")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "")
    vendor_name: str = os.getenv("VENDOR_NAME", "FlytBase")
    vendor_site: str = os.getenv("VENDOR_SITE", "https://flytbase.com").rstrip("/")
    http_timeout: float = 30.0
    fetch_workers: int = 6
    # research budgets
    queries_per_dimension: int = 3
    results_per_query: int = 5
    fetch_per_dimension: int = 6
    excerpt_chars_per_source: int = int(os.getenv("EXCERPT_CHARS_PER_SOURCE", "9000"))
    max_prompt_chars: int = int(os.getenv("MAX_PROMPT_CHARS", "60000"))  # lower for small local models
    # vendor content budgets
    max_content_candidates: int = 30
    rubric_top_n: int = 10
    max_sequence_revisions: int = 2
    mock: bool = False
    use_cache: bool = True


# ---------------------------------------------------------------------------
# Stage 1 — Ideal Customer Profile (vendor-level, not lead-level)
# The LLM judges each criterion against the email with a verbatim quote;
# the code turns those judgements into a score with these weights.
# ---------------------------------------------------------------------------
ICP_CRITERIA = [
    {"id": "target_industry", "weight": 20,
     "label": "Target industry",
     "description": "Heavy industry with large physical sites: mining, oil & gas, utilities/energy, ports & logistics, rail, chemicals, manufacturing, public safety, security of large perimeters."},
    {"id": "large_sites", "weight": 15,
     "label": "Large-scale sites",
     "description": "Sites large or spread out enough that manual inspection is slow (mines, plants, pipelines, yards)."},
    {"id": "continuous_ops", "weight": 10,
     "label": "Continuous operations",
     "description": "Operations run continuously (24/7 or shift-based), which favours docked, autonomous drones over piloted flights."},
    {"id": "multi_site", "weight": 10,
     "label": "Multi-site rollout potential",
     "description": "More than one site is in scope or implied."},
    {"id": "costly_or_hazardous_manual_work", "weight": 15,
     "label": "Costly or hazardous manual inspection",
     "description": "The status quo (crews, contractors, manual rounds) is expensive, slow or dangerous."},
    {"id": "warm_referral", "weight": 10,
     "label": "Warm referral",
     "description": "Referred by an existing customer or partner."},
    {"id": "budget_or_timeline_signal", "weight": 10,
     "label": "Budget / timeline signal",
     "description": "A budget cycle, funding event or decision date is mentioned."},
    {"id": "senior_operational_owner", "weight": 10,
     "label": "Senior operational owner",
     "description": "Sender is a senior leader who owns the operational problem (Head/VP/Director of Operations, Asset Integrity, HSE, Security)."},
]
FIT_CREDIT = {"met": 1.0, "partial": 0.5, "not_met": 0.0, "unknown": 0.0}

MEDDPICC_ELEMENTS = [
    "Metrics", "Economic Buyer", "Decision Criteria", "Decision Process",
    "Paper Process", "Identify Pain", "Champion", "Competition",
]
MEDDPICC_CREDIT = {"confirmed": 2, "partial": 1, "unknown": 0}

# Priority: fit decides how much we want the deal; MEDDPICC completeness decides
# how much we know. Inbound leads are naturally low on completeness.
PRIORITY_RULES = [
    # (min_fit, min_meddpicc_pct, label, meaning)
    (75, 0, "P1 - Hot", "Strong ICP fit; route to an AE within 24h even though qualification is incomplete."),
    (55, 0, "P2 - Warm", "Good fit; BDR runs discovery before AE handoff."),
    (35, 0, "P3 - Nurture", "Partial fit; nurture sequence, re-score on reply."),
    (0, 0, "P4 - Disqualify / redirect", "Poor fit; polite redirect."),
]

# ---------------------------------------------------------------------------
# Stage 2 — research dimensions and the keywords used to pull relevant
# excerpts out of long documents (filings, reports) before the LLM reads them.
# ---------------------------------------------------------------------------
RESEARCH_DIMENSIONS = {
    "org_structure": {
        "goal": "Organizational structure, business units, operating sites and executives relevant to an operations leader in the lead's division/region; who the lead likely reports to.",
        "keywords": ["organizational structure", "organization", "executive officers", "senior management",
                     "vice president", "general manager", "operations", "business unit", "board of directors",
                     "chief executive", "chief operating", "reports to", "division", "headcount", "employees"],
    },
    "budget_signals": {
        "goal": "Capex/opex guidance, investment plans, technology/automation/digital spend, safety and contractor costs from public filings and investor material.",
        "keywords": ["capital expenditure", "capex", "investment plan", "capital investment", "investments of",
                     "us$", "million", "technology", "digital", "automation", "autonomous", "drone",
                     "maintenance", "contractor", "cost reduction", "efficiency", "safety"],
    },
    "news": {
        "goal": "Recent news, press releases and strategic announcements (last ~18 months).",
        "keywords": ["announce", "agreement", "partnership", "expansion", "production", "joint venture",
                     "acquisition", "project", "approval", "strike", "incident", "2026", "2025"],
    },
    "stakeholder_letters": {
        "goal": "Chairman/CEO letters, investor presentations, sustainability reports and earnings commentary that reveal operating priorities.",
        "keywords": ["letter", "dear shareholders", "chairman", "chief executive officer", "message from",
                     "our priorities", "strategy", "safety", "sustainab", "productivity", "operational excellence",
                     "innovation", "outlook"],
    },
}

# ---------------------------------------------------------------------------
# Stage 4 — vendor content discovery and ranking
# ---------------------------------------------------------------------------
CONTENT_SECTION_WEIGHTS = {  # first path segment on the vendor site -> credibility as sales proof
    "case-studies": 1.0, "case-study": 1.0, "customers": 1.0, "customer-stories": 1.0,
    "webinars": 0.85, "blog": 0.8, "industries": 0.7, "tv": 0.6, "events": 0.4,
}
DEFAULT_SECTION_WEIGHT = 0.5
SKIP_SECTIONS = {"author", "authors", "studio", "api", "legal", "privacy", "terms"}
CASE_STUDY_FINAL_WEIGHTS = {  # all components normalised to 0..1
    "rubric": 0.45,      # LLM judgement of industry / use-case / environment / proof (quote-verified)
    "referral": 0.25,    # content mentions an organisation that referred the lead
    "region": 0.15,      # content mentions the lead's country / region
    "industry": 0.10,    # share of lead keywords present in the content
    "section": 0.05,     # case study > webinar > blog ...
}

# ---------------------------------------------------------------------------
# Stage 5 — go-to-market motion
# ---------------------------------------------------------------------------
PARTNER_PATH_HINTS = ["partner", "reseller", "distributor", "integrator"]
PARTNER_SHORTLIST_POINTS = {
    "same_country": 3, "same_region": 1, "proven_on_reference_account": 3,
    "system_integrator": 2, "solution_provider": 1, "top_tier": 1,
}
TOP_TIERS = {"gold", "platinum", "elite", "premier", "diamond"}
COMPLEXITY_THRESHOLDS = {"high": 4, "medium": 2}  # count of complexity factors


@dataclass
class RunPaths:
    root: str
    llm_dir: str = field(init=False)
    stage_dir: str = field(init=False)

    def __post_init__(self):
        self.llm_dir = os.path.join(self.root, "llm_calls")
        self.stage_dir = os.path.join(self.root, "stages")
        os.makedirs(self.llm_dir, exist_ok=True)
        os.makedirs(self.stage_dir, exist_ok=True)
