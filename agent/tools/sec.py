"""SEC EDGAR lookup: company name -> CIK -> latest annual report (20-F / 10-K) and
recent 6-K/8-K filings. Many foreign issuers are filed under an English name that
differs from their local name, so we resolve the CIK with EDGAR full-text search
over every name variant, then fall back to the ticker table."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from urllib.parse import quote

from rapidfuzz import fuzz


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


class SEC:
    def __init__(self, web, settings, log):
        self.web = web
        self.log = log
        ua = settings.sec_user_agent or "InboundResearchAgent research@example.com"
        self.headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    def _name_sim(self, names: list[str], label: str) -> float:
        label = re.sub(r"\(CIK \d+\)|\([A-Z.\-]{1,6}\)", "", label).lower()
        return max((fuzz.token_set_ratio(_ascii(n).lower(), label) for n in names if n), default=0)

    def find_cik(self, names: list[str], tickers: list[str]) -> dict | None:
        """Identify the company's OWN filer record. A filing that merely mentions the company
        (e.g. a shareholder's annual report) must not win, so every candidate has to match
        the company by ticker or by name; hit counts only break ties."""
        tickers = [t.upper() for t in tickers]
        # 1) SEC ticker table: exact ticker match
        table = self.web.get_json("https://www.sec.gov/files/company_tickers.json", headers=self.headers) or {}
        rows = list(table.values()) if isinstance(table, dict) else []
        for t in tickers:
            for row in rows:
                if row.get("ticker", "").upper() == t:
                    sim = self._name_sim(names, row.get("title", ""))
                    return {"cik": str(row["cik_str"]), "label": f"{row['title']} ({row['ticker']})",
                            "method": f"ticker table, ticker {t} (name similarity {sim:.0f})"}
        # 2) Full-text search, keeping only filers that ARE the company
        votes: Counter = Counter()
        labels: dict[str, str] = {}
        for name in dict.fromkeys([n for n in names if n] + [_ascii(n) for n in names if n]):
            url = ("https://efts.sec.gov/LATEST/search-index?q=" + quote(f'"{name}"')
                   + "&forms=20-F,10-K,40-F")
            data = self.web.get_json(url, headers=self.headers)
            hits = ((data or {}).get("hits") or {}).get("hits") or []
            for h in hits[:50]:
                src = h.get("_source", {})
                for cik, label in zip(src.get("ciks", []), src.get("display_names", [])):
                    votes[cik] += 1
                    labels[cik] = label
            if hits:
                self.log.info(f"EDGAR full-text '{name}': {len(hits)} hits")
        candidates = []
        for cik, n in votes.items():
            label = labels[cik]
            ticker_hit = any(f"({t})" in label.upper() for t in tickers)
            sim = self._name_sim(names, label)
            if ticker_hit or sim >= 85:
                candidates.append((100 if ticker_hit else sim, n, cik, label))
            else:
                self.log.info(f"EDGAR: ignoring {label.strip()} (mentions the company but is not it; similarity {sim:.0f})")
        if candidates:
            score, n, cik, label = max(candidates)
            return {"cik": cik.lstrip("0"), "label": label.strip(),
                    "method": f"full-text search, identity match {score:.0f} ({n} hits)"}
        # 3) Name match against the ticker table
        best = None
        for row in rows:
            sim = self._name_sim(names, row.get("title", ""))
            if sim >= 90 and (best is None or sim > best[0]):
                best = (sim, row)
        if best:
            row = best[1]
            return {"cik": str(row["cik_str"]), "label": f"{row['title']} ({row['ticker']})",
                    "method": f"name match ({best[0]:.0f})"}
        return None

    def filings(self, cik: str, forms: tuple[str, ...], limit: int = 1) -> list[dict]:
        data = self.web.get_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", headers=self.headers)
        if not data:
            return []
        recent = data.get("filings", {}).get("recent", {})
        out = []
        for form, acc, doc, date, desc in zip(recent.get("form", []), recent.get("accessionNumber", []),
                                              recent.get("primaryDocument", []), recent.get("filingDate", []),
                                              recent.get("primaryDocDescription", [""] * 10_000)):
            if form in forms:
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
                out.append({"form": form, "date": date, "url": url, "description": desc,
                            "company": data.get("name", "")})
                if len(out) >= limit:
                    break
        return out
