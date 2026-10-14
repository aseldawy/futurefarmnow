"""
CIMIS (California Irrigation Management Information System) API client.

Strategy:
- Lat/lon coordinates are NOT accepted by the CIMIS data API (ERR2006).
- Zip code targets (Spatial CIMIS) return only ETo, solar, wind — no temps.
- Station targets return the full set including temps and humidity.
- Therefore: given a lat/lon, find the nearest active ETo station via haversine,
  then query by station number to get complete daily data.

Station list is cached in a module-level variable after the first request
(no key required for /api/station).
"""

import os
import math
import time
import logging
import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DATA_URL    = "https://et.water.ca.gov/api/data"
_STATION_URL = "https://et.water.ca.gov/api/station"
_KEY         = os.environ.get("CIMIS_appKey", "")
_TIMEOUT     = 20

# Module-level cache: list of {nbr, lat, lon, city, county} for active ETo stations
_station_cache: list[dict] | None = None

# TTL cache for daily data responses: key -> (monotonic_ts, result)
_data_cache: dict[str, tuple[float, Any]] = {}
_DATA_CACHE_TTL = 1800  # 30 minutes
_DATA_CACHE_MAX = 500   # entries; cache is cleared when full (keys come from model-chosen args)


# ---------------------------------------------------------------------------
# Station lookup
# ---------------------------------------------------------------------------

def _load_stations() -> list[dict]:
    """Fetch all active ETo stations once and cache them."""
    global _station_cache
    if _station_cache is not None:
        return _station_cache

    resp = requests.get(_STATION_URL, headers={"Accept": "application/json"}, timeout=15)
    resp.raise_for_status()
    stations = resp.json().get("Stations", [])

    active = []
    for s in stations:
        if s.get("IsActive") != "True" or s.get("IsEtoStation") != "True":
            continue
        try:
            # HmsLatitude format: "36º20'10N / 36.336222" — decimal is after " / "
            lat = float(s["HmsLatitude"].split("/")[-1].strip())
            lon = float(s["HmsLongitude"].split("/")[-1].strip())
        except (KeyError, ValueError, IndexError):
            continue
        active.append({
            "nbr":    s["StationNbr"],
            "lat":    lat,
            "lon":    lon,
            "city":   s.get("City", ""),
            "county": s.get("County", ""),
        })

    _station_cache = active
    logger.info("Loaded %d active CIMIS ETo stations", len(active))
    return active


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_station(lat: float, lon: float) -> dict:
    """Return the nearest active ETo station dict to the given coordinates."""
    stations = _load_stations()
    if not stations:
        raise RuntimeError("No active CIMIS stations found")
    return min(stations, key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))


# ---------------------------------------------------------------------------
# Data fetcher
# ---------------------------------------------------------------------------

def _parse_val(record: dict, field: str) -> float | None:
    """Extract float value from a CIMIS record field, return None if missing/null."""
    item = record.get(field)
    if not item:
        return None
    val = item.get("Value")
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fetch_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], dict]:
    if not _KEY:
        raise RuntimeError("CIMIS_appKey environment variable not set")

    station = nearest_station(lat, lon)
    cache_key = f"{station['nbr']}:{start_date}:{end_date}"
    now = time.monotonic()
    if cache_key in _data_cache and now - _data_cache[cache_key][0] < _DATA_CACHE_TTL:
        return _data_cache[cache_key][1]

    dist_km = _haversine_km(lat, lon, station["lat"], station["lon"])

    params = {
        "appKey":       _KEY,
        "targets":      station["nbr"],
        "startDate":    start_date,
        "endDate":      end_date,
        "unitOfMeasure": "E",
    }

    resp = requests.get(
        _DATA_URL,
        params=params,
        headers={"Accept": "application/json"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # Error strings come back as bare JSON strings, e.g. "[ERR1035-...]"
    if isinstance(data, str):
        raise ValueError(f"CIMIS API error: {data}")

    providers = data.get("Data", {}).get("Providers", [])
    raw_records = providers[0].get("Records", []) if providers else []

    records = []
    for r in raw_records:
        records.append({
            "date":     r.get("Date", ""),
            "eto_in":   _parse_val(r, "DayAsceEto"),
            "precip_in": _parse_val(r, "DayPrecip"),
            "tmax_f":   _parse_val(r, "DayAirTmpMax"),
            "tmin_f":   _parse_val(r, "DayAirTmpMin"),
            "tavg_f":   _parse_val(r, "DayAirTmpAvg"),
            "rh_avg":   _parse_val(r, "DayRelHumAvg"),
            "rh_max":   _parse_val(r, "DayRelHumMax"),
            "rh_min":   _parse_val(r, "DayRelHumMin"),
            "solar_ly": _parse_val(r, "DaySolRadAvg"),
            "wind_mph": _parse_val(r, "DayWindSpdAvg"),
            "dew_f":    _parse_val(r, "DayDewPnt"),
        })

    station_info = {
        "nbr":     station["nbr"],
        "city":    station["city"],
        "county":  station["county"],
        "lat":     station["lat"],
        "lon":     station["lon"],
        "dist_km": round(dist_km, 1),
    }

    result = records, station_info
    if len(_data_cache) >= _DATA_CACHE_MAX:
        _data_cache.clear()
    _data_cache[cache_key] = (time.monotonic(), result)
    return result


def last_n_days(lat: float, lon: float, n: int = 30) -> tuple[list[dict], dict]:
    """Convenience: fetch the last n days ending yesterday."""
    today     = datetime.date.today()
    end_date  = (today - datetime.timedelta(days=1)).isoformat()
    start_date = (today - datetime.timedelta(days=n)).isoformat()
    return fetch_daily(lat, lon, start_date, end_date)
