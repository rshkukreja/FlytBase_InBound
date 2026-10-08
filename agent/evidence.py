"""Evidence store and grounding gate.

Every fetched document gets a stable id (S1, S2, ...). LLM stages may only make
claims as {claim, source_id, evidence_quote}. `verify_quote` then checks — in plain
code, no LLM — that the quote really occurs in that source's text. Claims that fail
are dropped from the output and listed in the report as rejected.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz

MIN_QUOTE_CHARS = 20
FUZZY_THRESHOLD = 90  # tolerate whitespace/punctuation differences from HTML->text


@dataclass
class Source:
    id: str
    url: str
    title: str
    kind: str          # input | web | filing | vendor
    text: str
    tags: set = field(default_factory=set)
    query: str = ""


class EvidenceStore:
    def __init__(self):
        self._by_id: dict[str, Source] = {}
        self._by_url: dict[str, str] = {}
        self._lock = threading.Lock()

    def add(self, url: str, title: str, text: str, kind: str, tags=(), query: str = "") -> Source:
        with self._lock:
            if url in self._by_url:
                src = self._by_id[self._by_url[url]]
                src.tags.update(tags)
                return src
            sid = f"S{len(self._by_id) + 1}"
            src = Source(sid, url, title or url, kind, text, set(tags), query)
            self._by_id[sid] = src
            self._by_url[url] = sid
            return src

    def get(self, sid: str) -> Source | None:
        return self._by_id.get(sid.strip().strip("[]"))

    def by_url(self, url: str) -> Source | None:
        sid = self._by_url.get(url)
        return self._by_id.get(sid) if sid else None

    def all(self) -> list[Source]:
        return list(self._by_id.values())

    def with_tag(self, tag: str) -> list[Source]:
        return [s for s in self._by_id.values() if tag in s.tags]

    def excerpt(self, sid: str, keywords: list[str], budget: int) -> str:
        """Keyword-windowed excerpt so long filings fit in a prompt. Windows are cut
        on line boundaries so the LLM sees whole sentences it can quote verbatim."""
        src = self.get(sid)
        if not src:
            return ""
        text = src.text
        if len(text) <= budget:
            return text
        low = text.lower()
        spans = []
        for kw in keywords:
            for m in re.finditer(re.escape(kw.lower()), low):
                spans.append((max(0, m.start() - 700), min(len(text), m.end() + 900)))
                if len(spans) > 400:
                    break
        if not spans:
            return text[:budget]
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        # score windows by number of distinct keywords they contain; keep the best within budget
        def score(w):
            seg = low[w[0]:w[1]]
            return sum(1 for kw in keywords if kw.lower() in seg)
        ranked = sorted(merged, key=score, reverse=True)
        chosen, used = [], 0
        for a, b in ranked:
            a = text.rfind("\n", 0, a) + 1
            nb = text.find("\n", b)
            b = nb if nb != -1 else len(text)
            if used + (b - a) > budget:
                continue
            chosen.append((a, b))
            used += b - a
        chosen.sort()
        return "\n[...]\n".join(text[a:b] for a, b in chosen) or text[:budget]


# ------------------------------------------------------------------ grounding
@lru_cache(maxsize=256)
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-"))
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \"'")


def verify_quote(quote: str, text: str) -> tuple[bool, float]:
    """True if `quote` occurs in `text` (exactly after normalisation, or fuzzily >= 90).
    Quotes containing an ellipsis are checked part by part."""
    q, t = normalize(quote), normalize(text)
    if len(q) < MIN_QUOTE_CHARS or not t:
        return False, 0.0
    parts = [p.strip() for p in re.split(r"\.\.\.|…|\[\.\.\.\]", q) if len(p.strip()) >= 8]
    if len(parts) > 1:
        scores = [verify_quote(p, text)[1] if len(p) >= MIN_QUOTE_CHARS else (100.0 if p in t else 0.0)
                  for p in parts]
        s = min(scores)
        return s >= FUZZY_THRESHOLD, s
    if q in t:
        return True, 100.0
    # Fuzzy: only look near anchor words, so very long filings stay fast.
    words = sorted({w for w in re.findall(r"[a-z0-9$%.,]{5,}", q)}, key=len, reverse=True)[:4]
    best = 0.0
    window = len(q)
    for w in words:
        start = 0
        for _ in range(60):
            i = t.find(w, start)
            if i == -1:
                break
            seg = t[max(0, i - window): i + window * 2]
            best = max(best, fuzz.partial_ratio(q, seg))
            if best >= 99:
                return True, best
            start = i + 1
    if not words and len(t) < 200_000:
        best = fuzz.partial_ratio(q, t)
    return best >= FUZZY_THRESHOLD, float(best)
