"""Web access: search (Tavily, falling back to DuckDuckGo) and fetch (HTML/PDF -> text).
Fetches are cached on disk by URL so reruns are cheap and reproducible."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 InboundResearchAgent/1.0")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str
    query: str


@dataclass
class Page:
    url: str
    final_url: str
    title: str
    text: str
    content_type: str
    links: list[str] = field(default_factory=list)
    status: int = 200


class Web:
    def __init__(self, settings, log, cache_dir: str = ".cache"):
        self.s = settings
        self.log = log
        self.cache_dir = os.path.join(cache_dir, "web")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.client = httpx.Client(follow_redirects=True, timeout=settings.http_timeout,
                                   headers={"User-Agent": BROWSER_UA, "Accept-Language": "en"})
        self._tavily = None
        if settings.tavily_key:
            from tavily import TavilyClient
            self._tavily = TavilyClient(api_key=settings.tavily_key)

    # ------------------------------------------------------------------ cache
    def _ck(self, kind: str, key: str) -> str:
        return os.path.join(self.cache_dir, f"{kind}_{hashlib.sha1(key.encode()).hexdigest()}.json")

    def _cache_get(self, kind, key):
        p = self._ck(kind, key)
        if self.s.use_cache and os.path.exists(p):
            self.log.bump("cache_hits")
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, kind, key, value):
        with open(self._ck(kind, key), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    # ----------------------------------------------------------------- search
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        cached = self._cache_get("search", f"{query}|{max_results}")
        if cached is not None:
            return [SearchResult(**r) for r in cached]
        self.log.bump("searches")
        results: list[SearchResult] = []
        if self._tavily:
            try:
                kwargs = {}
                m = re.search(r"\bsite:(\S+)", query)
                q = query
                if m:  # Tavily restricts domains via a parameter rather than the site: operator
                    kwargs["include_domains"] = [m.group(1)]
                    q = (query[:m.start()] + query[m.end():]).strip() or m.group(1)
                resp = self._tavily.search(query=q, max_results=max_results, search_depth="advanced", **kwargs)
                for r in resp.get("results", []):
                    results.append(SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", "")[:500],
                                                "tavily", query))
            except Exception as e:
                self.log.warn(f"Tavily search failed ({e}); falling back to DuckDuckGo")
        if not results:
            try:
                from ddgs import DDGS
                for r in DDGS().text(query, max_results=max_results):
                    results.append(SearchResult(r.get("title", ""), r.get("href", ""), r.get("body", "")[:500],
                                                "duckduckgo", query))
            except Exception as e:
                self.log.warn(f"DuckDuckGo search failed for '{query}': {e}")
        results = [r for r in results if r.url.startswith("http")]
        self.log.event("search", query=query, n=len(results), urls=[r.url for r in results])
        if results:
            self._cache_put("search", f"{query}|{max_results}", [asdict(r) for r in results])
        return results

    # ------------------------------------------------------------------ fetch
    def get_raw(self, url: str, headers: dict | None = None) -> tuple[int, str, str, str]:
        """Returns (status, final_url, content_type, body_text). Binary bodies are decoded by fetch()."""
        cached = self._cache_get("raw", url)
        if cached is not None:
            return cached["status"], cached["final_url"], cached["ctype"], cached["body"]
        self.log.bump("fetches")
        for attempt in range(3):
            try:
                r = self.client.get(url, headers=headers or {})
                ctype = r.headers.get("content-type", "").lower()
                if r.content[:5] == b"%PDF-":  # trust the bytes, not the URL (some '.pdf' links return HTML)
                    body = _pdf_to_text(r.content)
                    ctype = "application/pdf"
                else:
                    body = r.text
                if r.status_code == 429 and attempt < 2:
                    time.sleep(2 + attempt * 3)
                    continue
                if r.status_code < 400:
                    self._cache_put("raw", url, {"status": r.status_code, "final_url": str(r.url),
                                                 "ctype": ctype, "body": body})
                else:
                    self.log.bump("fetch_failures")
                self.log.event("fetch", url=url, status=r.status_code, chars=len(body))
                return r.status_code, str(r.url), ctype, body
            except Exception as e:
                if attempt == 2:
                    self.log.bump("fetch_failures")
                    self.log.event("fetch", url=url, status=0, error=str(e)[:200])
                    return 0, url, "", ""
                time.sleep(1 + attempt)
        return 0, url, "", ""

    def get_json(self, url: str, headers: dict | None = None):
        status, _, _, body = self.get_raw(url, headers=headers)
        if status >= 400 or not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    def fetch(self, url: str, headers: dict | None = None) -> Page | None:
        status, final_url, ctype, body = self.get_raw(url, headers=headers)
        if status == 0 or status >= 400 or not body:
            return None
        if ctype == "application/pdf":
            return Page(url, final_url, os.path.basename(urlparse(url).path), _clean(body), ctype, [], status)
        if "xml" in ctype and "html" not in ctype:
            return Page(url, final_url, "", body, ctype, [], status)
        return html_to_page(url, final_url, body, ctype, status)

    def fetch_many(self, urls: list[str]) -> dict[str, Page | None]:
        urls = list(dict.fromkeys(urls))
        with ThreadPoolExecutor(max_workers=self.s.fetch_workers) as ex:
            pages = list(ex.map(self.fetch, urls))
        return dict(zip(urls, pages))


def html_to_page(url, final_url, html, ctype="text/html", status=200) -> Page:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:300]
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(final_url, a["href"]).split("#")[0]
        if href.startswith("http"):
            links.append(href)
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()
    # Keep block structure so that quotes stay on sensible line boundaries.
    text = soup.get_text("\n", strip=True)
    return Page(url, final_url, title, _clean(text), ctype, list(dict.fromkeys(links)), status)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for i, page in enumerate(reader.pages[:120]):
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


class MockWeb(Web):
    """Offline stand-in used by --mock and the tests. Returns small synthetic pages."""

    def __init__(self, settings, log, cache_dir: str = ".cache"):
        self.s = settings
        self.log = log

    def search(self, query, max_results=5):
        self.log.bump("searches")
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40]
        return [SearchResult(f"Result {i} for {query}", f"https://example.org/{slug}/{i}", "snippet", "mock", query)
                for i in range(2)]

    def get_raw(self, url, headers=None):
        self.log.bump("fetches")
        if url.endswith("robots.txt"):
            return 200, url, "text/plain", f"Sitemap: {self.s.vendor_site}/sitemap.xml"
        if url.endswith("sitemap.xml"):
            return 200, url, "application/xml", (
                '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                f'<url><loc>{self.s.vendor_site}/case-studies/mining-site-inspection</loc></url>'
                f'<url><loc>{self.s.vendor_site}/es/case-studies/mining-site-inspection</loc></url>'
                f'<url><loc>{self.s.vendor_site}/blog/referral-partner-deploys-docks-at-copper-mine</loc></url>'
                '</urlset>')
        html = (f"<html><head><title>Page {url}</title></head><body><a href='{self.s.vendor_site}/partner'>Partners</a>"
                "<p>Mock page text about mining operations, autonomous drone docks and inspection in Chile.</p>"
                + "<p>Synthetic filler sentence used only for offline wiring tests of the pipeline.</p>" * 8
                + "<p>Mock Partner One Chile Gold System Integrator</p></body></html>")
        return 200, url, "text/html", html
