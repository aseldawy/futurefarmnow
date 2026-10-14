"""
UC IPM (University of California Integrated Pest Management) guidance fetcher.

Fetches evidence-based pest/disease management guidance from ipm.ucanr.edu.
Results are cached for 24 hours per (crop, pest) pair to avoid hammering the UC servers.

UC IPM URL patterns:
  - Pest/crop search: https://ipm.ucanr.edu/PMG/selectpest.home.html
  - Individual pest pages: https://ipm.ucanr.edu/PMG/r{crop_code}700{pest_code}.html
    (crop_code and pest_code are numeric IDs not easily predictable)
  - General crop pest list: https://ipm.ucanr.edu/agriculture.html
"""

import re
import time
import logging
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)

_SEARCH_BASE = "https://ipm.ucanr.edu/PMG/selectpest.home.html"
_AG_BASE     = "https://ipm.ucanr.edu"
_TIMEOUT     = 10
_CACHE_TTL   = 86400  # 24 hours

# {(crop, pest): (timestamp, result_dict)}
_cache: dict[tuple[str, str | None], tuple[float, dict]] = {}


# ---------------------------------------------------------------------------
# Simple HTML text extractor
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip all HTML tags, return visible text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style", "nav", "header", "footer"}
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _fetch_url(url: str) -> str:
    """Fetch a URL and return the response text."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FutureFarmNow/1.0)",
            "Accept": "text/html",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def _truncate(text: str, max_chars: int = 1200) -> str:
    """Truncate to max_chars at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(". ")
    return text[: cut + 1] if cut > 0 else text[:max_chars] + "…"


# ---------------------------------------------------------------------------
# Search and parse UC IPM pages
# ---------------------------------------------------------------------------

def _search_ipm(crop: str, pest: str | None) -> dict[str, Any]:
    """
    Search ipm.ucanr.edu for pest management info.
    Returns a dict with source_url, management_summary, treatment_threshold, monitoring_tips.
    """
    query = f"{crop} {pest}" if pest else crop
    search_url = f"https://ipm.ucanr.edu/PMG/selectpest.home.html?sel={urllib.parse.quote(query)}"

    # Try the search results page
    try:
        html = _fetch_url(search_url)
    except Exception as e:
        logger.warning("UC IPM search fetch failed for '%s': %s", query, e)
        return _fallback_result(crop, pest)

    # Look for links to specific pest pages in the search results
    pest_links = re.findall(r'href="(/PMG/[^"]+\.html)"', html)
    pest_links = [_AG_BASE + l for l in pest_links if "r" in l.lower()]

    if pest_links:
        # Fetch the first relevant pest management page
        try:
            pest_html = _fetch_url(pest_links[0])
            return _parse_pest_page(pest_html, pest_links[0], crop, pest)
        except Exception as e:
            logger.warning("UC IPM pest page fetch failed: %s", e)

    # Fall back to extracting from search results page itself
    text = _extract_text(html)
    if len(text) > 200:
        return {
            "crop":                 crop,
            "pest":                 pest,
            "source_url":           search_url,
            "management_summary":   _truncate(text, 1000),
            "treatment_threshold":  None,
            "monitoring_tips":      None,
        }

    return _fallback_result(crop, pest)


def _parse_pest_page(html: str, url: str, crop: str, pest: str | None) -> dict[str, Any]:
    """Extract structured content from a UC IPM pest management page."""
    text = _extract_text(html)

    # Try to find treatment threshold sentence
    threshold = None
    threshold_match = re.search(
        r"(treat(?:ment)?\s+threshold[^.]{0,200}\.)",
        text,
        re.IGNORECASE,
    )
    if threshold_match:
        threshold = threshold_match.group(1).strip()

    # Try to find monitoring tips (look for "monitor" or "scout" context)
    monitoring = None
    monitor_match = re.search(
        r"((?:monitor|scout|inspect|check)[^.]{0,300}\.)",
        text,
        re.IGNORECASE,
    )
    if monitor_match:
        monitoring = monitor_match.group(1).strip()

    return {
        "crop":                crop,
        "pest":                pest,
        "source_url":          url,
        "management_summary":  _truncate(text, 1200),
        "treatment_threshold": threshold,
        "monitoring_tips":     monitoring,
    }


def _fallback_result(crop: str, pest: str | None) -> dict[str, Any]:
    """Return a minimal result pointing to the UC IPM website when fetch fails."""
    base = "https://ipm.ucanr.edu/agriculture.html"
    return {
        "crop":                crop,
        "pest":                pest,
        "source_url":          base,
        "management_summary":  (
            f"UC IPM provides comprehensive pest management guidelines for {crop}. "
            f"Visit ipm.ucanr.edu for detailed information on {'pest management' if not pest else pest}."
        ),
        "treatment_threshold": None,
        "monitoring_tips":     f"Regular scouting recommended. See ipm.ucanr.edu for {crop}-specific thresholds.",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ipm_guidance(crop: str, pest: str | None = None) -> dict[str, Any]:
    """
    Retrieve UC IPM pest/disease management guidance for a crop (and optional pest).

    Results are cached 24h per (crop, pest) pair.

    Returns:
        {crop, pest, source_url, management_summary, treatment_threshold, monitoring_tips}
    """
    cache_key = (crop.lower().strip(), pest.lower().strip() if pest else None)
    now = time.time()

    if cache_key in _cache:
        ts, result = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            return result

    result = _search_ipm(crop, pest)
    _cache[cache_key] = (now, result)
    return result
