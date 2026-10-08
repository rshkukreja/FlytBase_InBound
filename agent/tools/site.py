"""Vendor-site discovery: robots.txt -> sitemap(s) -> every page URL, plus the links
in the homepage navigation (some pages, e.g. a partner directory, are not in any sitemap)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

LANG_SEGMENT = re.compile(r"^([a-z]{2})(-[a-z]{2,4})?$")
# ISO 639-1 codes of languages sites commonly translate into (so '/tv/...' is not mistaken for a language)
LANG_CODES = {"ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he", "hi", "hr", "hu", "id", "it",
              "ja", "ko", "lt", "lv", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "th", "tr",
              "uk", "vi", "zh"}


def _locs(xml_text: str) -> tuple[list[str], list[str]]:
    """Returns (child_sitemaps, page_urls) from a sitemap or sitemap index."""
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        return [], re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml_text)
    tag = root.tag.split("}")[-1]
    locs = [el.text.strip() for el in root.iter() if el.tag.split("}")[-1] == "loc" and el.text]
    return (locs, []) if tag == "sitemapindex" else ([], locs)


def sitemap_urls(web, base: str, log, per_map_cap: int = 5000) -> list[str]:
    """Reads every child sitemap (a per-sitemap cap, not a global one, so a huge blog
    sitemap can never crowd out the case-study sitemap)."""
    status, _, _, robots = web.get_raw(f"{base}/robots.txt")
    maps = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots or "") or [f"{base}/sitemap.xml"]
    seen_maps, pages = set(), []
    queue = list(maps)
    while queue and len(seen_maps) < 80:
        m = queue.pop(0)
        if m in seen_maps:
            continue
        seen_maps.add(m)
        status, _, _, body = web.get_raw(m)
        if status >= 400 or not body:
            continue
        children, urls = _locs(body)
        queue.extend(children)
        pages.extend(urls[:per_map_cap])
    pages = list(dict.fromkeys(pages))
    log.info(f"sitemaps read: {len(seen_maps)}, page URLs: {len(pages)}")
    return pages


def is_default_language(url: str) -> bool:
    """Drop translated duplicates like /es/..., /pt-br/... (keeps the default-language page)."""
    seg = [s for s in urlparse(url).path.lower().split("/") if s]
    if not seg:
        return True
    m = LANG_SEGMENT.match(seg[0])
    return not (m and m.group(1) in LANG_CODES and m.group(1) != "en")


def section_of(url: str) -> str:
    seg = [s for s in urlparse(url).path.split("/") if s]
    return seg[0].lower() if seg else ""


def slug_tokens(url: str) -> list[str]:
    path = urlparse(url).path.lower()
    return [t for t in re.split(r"[^a-z0-9]+", path) if len(t) > 1]


def same_site(url: str, base: str) -> bool:
    a = urlparse(url).netloc.lower().removeprefix("www.")
    b = urlparse(base).netloc.lower().removeprefix("www.")
    return a == b
