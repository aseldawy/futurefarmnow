"""
USDA AMS MyMarketNews (MARS) API client for California commodity prices.

Auth: HTTP Basic with MARS_key as username, empty password.
Strategy: Hardcode CA-specific report slugs, route commodity → slug,
fetch Report Details section, filter by commodity name.

Verified CA report slugs (from /services/v1.2/reports):
  2307 - LA Terminal Market Vegetables
  2306 - LA Terminal Market Fruit
  2309 - LA Terminal Market Nuts
  2308 - LA Terminal Market Onions and Potatoes
  2391 - Fresno Shipping Point Vegetables
  2390 - Fresno Shipping Point Fruit
  2904 - California Direct Hay Report
"""

import os
import time
import logging
import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# TTL cache for price responses: cache_key -> (monotonic_ts, result)
_price_cache: dict[str, tuple[float, Any]] = {}
_PRICE_CACHE_TTL = 1800  # 30 minutes
_PRICE_CACHE_MAX = 500   # entries; cache is cleared when full (keys come from model-chosen args)

_BASE    = "https://marsapi.ams.usda.gov/services/v1.2"
_KEY     = os.environ.get("MARS_key", "")
_TIMEOUT = 25

# Commodity prefix → report slug routing (lowercase prefix matching)
_COMMODITY_ROUTES: list[tuple[str, int]] = [
    # Vegetables → LA Terminal Market Vegetables
    ("tomato",      2307),
    ("lettuce",     2307),
    ("pepper",      2307),
    ("cucumber",    2307),
    ("broccoli",    2307),
    ("cauliflower", 2307),
    ("celery",      2307),
    ("spinach",     2307),
    ("carrot",      2307),
    ("cabbage",     2307),
    ("artichoke",   2307),
    ("asparagus",   2307),
    ("mushroom",    2307),
    ("zucchini",    2307),
    ("squash",      2307),
    ("eggplant",    2307),
    ("garlic",      2307),
    ("kale",        2307),
    ("chard",       2307),
    ("bean",        2307),
    ("pea",         2307),
    # Fruits → LA Terminal Market Fruit
    ("strawberr",   2306),
    ("grape",       2306),
    ("lemon",       2306),
    ("orange",      2306),
    ("peach",       2306),
    ("nectarine",   2306),
    ("plum",        2306),
    ("cherry",      2306),
    ("apricot",     2306),
    ("avocado",     2306),
    ("kiwi",        2306),
    ("citrus",      2306),
    ("blueberr",    2306),
    ("raspberr",    2306),
    ("fig",         2306),
    ("pomegranate", 2306),
    ("mandarin",    2306),
    ("tangerine",   2306),
    # Nuts → LA Terminal Market Nuts
    ("almond",      2309),
    ("walnut",      2309),
    ("pistachio",   2309),
    ("pecan",       2309),
    # Onions + Potatoes
    ("onion",       2308),
    ("potato",      2308),
    ("yam",         2308),
    ("sweet potato",2308),
    # Hay
    ("hay",         2904),
    ("alfalfa",     2904),
]

_REPORT_LABELS: dict[int, str] = {
    2307: "LA Terminal Market Vegetables",
    2306: "LA Terminal Market Fruit",
    2309: "LA Terminal Market Nuts",
    2308: "LA Terminal Market Onions & Potatoes",
    2391: "Fresno Shipping Point Vegetables",
    2390: "Fresno Shipping Point Fruit",
    2904: "California Direct Hay",
}


def _route_slug(commodity: str) -> int:
    """Map a commodity name to the best CA report slug."""
    c = commodity.lower().strip()
    for prefix, slug in _COMMODITY_ROUTES:
        if prefix in c:
            return slug
    return 2307  # default: LA Terminal Vegetables


def _fetch_report(slug: int, report_date: str) -> list[dict]:
    """
    Fetch Report Details section for a given slug and date.
    Returns list of commodity price rows.
    """
    url = f"{_BASE}/reports/{slug}"
    params = {"allSections": "true", "reportDate": report_date}
    resp = requests.get(url, params=params, auth=(_KEY, ""),
                        headers={"Accept": "application/json"}, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    for section in data:
        if section.get("reportSection") == "Report Details":
            return section.get("results", [])
    return []


def _filter_by_commodity(rows: list[dict], commodity: str) -> list[dict]:
    """Case-insensitive substring filter on the commodity field."""
    c = commodity.lower().strip()
    # Try exact then partial match
    exact   = [r for r in rows if r.get("commodity", "").lower() == c]
    if exact:
        return exact
    partial = [r for r in rows if c in r.get("commodity", "").lower()]
    return partial


def _format_row(r: dict) -> dict:
    """Pick the fields useful to the AI model."""
    return {
        "commodity":    r.get("commodity"),
        "variety":      r.get("variety") if r.get("variety") not in ("N/A", None) else None,
        "package":      r.get("package"),
        "origin":       r.get("origin") if r.get("origin") not in ("N/A", None) else None,
        "organic":      r.get("organic") == "Y",
        "low_price":    r.get("low_price"),
        "high_price":   r.get("high_price"),
        "market_tone":  r.get("market_tone_comments"),
        "grade":        r.get("grade") if r.get("grade") not in ("N/A", None) else None,
    }


def get_prices(commodity: str) -> dict[str, Any]:
    if not _KEY:
        raise RuntimeError("MARS_key environment variable not set")

    slug = _route_slug(commodity)
    today = datetime.date.today()
    cache_key = f"{slug}:{commodity.lower().strip()}:{today.isoformat()}"
    now = time.monotonic()
    if cache_key in _price_cache and now - _price_cache[cache_key][0] < _PRICE_CACHE_TTL:
        return _price_cache[cache_key][1]

    for delta in (0, 1, 2):
        attempt_date = (today - datetime.timedelta(days=delta)).strftime("%m/%d/%Y")
        try:
            rows = _fetch_report(slug, attempt_date)
            if rows:
                matched = _filter_by_commodity(rows, commodity)
                if not matched:
                    # Broaden: return first 10 rows of the report so the model can explain
                    matched = rows[:10]
                result = {
                    "commodity_queried": commodity,
                    "report_label":      _REPORT_LABELS.get(slug, str(slug)),
                    "report_date":       attempt_date,
                    "market":            "Los Angeles Terminal / CA",
                    "prices":            [_format_row(r) for r in matched[:8]],
                    "num_results":       len(matched),
                }
                if len(_price_cache) >= _PRICE_CACHE_MAX:
                    _price_cache.clear()
                _price_cache[cache_key] = (time.monotonic(), result)
                return result
        except requests.exceptions.HTTPError as e:
            logger.warning("MARS report %s on %s returned %s", slug, attempt_date, e)

    return {
        "commodity_queried": commodity,
        "report_label":      _REPORT_LABELS.get(slug, str(slug)),
        "report_date":       None,
        "market":            "Los Angeles Terminal / CA",
        "prices":            [],
        "num_results":       0,
        "message":           "No recent price data found. Report may not have published yet.",
    }
