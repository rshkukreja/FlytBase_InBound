from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .evidence import EvidenceStore
from .llm import LLM
from .runlog import RunLog
from .schemas import InboundEmail


@dataclass
class Ctx:
    settings: Settings
    llm: LLM
    web: object
    store: EvidenceStore
    log: RunLog
    email: InboundEmail
    run_date: str
    out: dict = field(default_factory=dict)   # stage key -> output model
    vendor_pages: dict = field(default_factory=dict)  # url -> Page, shared by stages 4 and 5
