"""
Tool executor for FarmBot AI assistant.

Makes internal HTTP calls to the existing FFN WSGI endpoints on behalf of
the Gemini model. Reads the base URL from env var AI_SELF_BASE_URL so it
works in both development and production without hard-coding ports.
"""

import os
import json
import logging
import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Base URL of this same Flask server — used to call existing endpoints
_BASE_URL = os.environ.get("AI_SELF_BASE_URL", "http://localhost:8082")
_TIMEOUT_SOIL   = 30   # seconds — soil stats is fast
_TIMEOUT_NDVI   = 120  # seconds — NDVI processes many GeoTIFF files across date range
_TIMEOUT_CIMIS  = 20
_TIMEOUT_MARKET = 30
_TIMEOUT_UC_IPM = 10


def execute_tool(
    tool_name: str,
    tool_args: dict,
    polygon_geojson: dict | None,
) -> dict[str, Any]:
    """
    Dispatch a Gemini function call to the corresponding FFN API endpoint.

    Args:
        tool_name:       Tool name from the model function call
        tool_args:       Arguments the model supplied for the tool call
        polygon_geojson: GeoJSON geometry from the frontend context (may be None
                         for location-only tools like CIMIS/pest/market)

    Returns:
        Parsed JSON result from the endpoint, or an error dict.
    """
    # Polygon-independent tools — dispatch first, no polygon check needed
    try:
        if tool_name == "get_cimis_weather":
            return _get_cimis_weather(tool_args)
        elif tool_name == "get_pest_disease_risk":
            return _get_pest_disease_risk(tool_args)
        elif tool_name == "get_market_prices":
            return _get_market_prices(tool_args)
        elif tool_name == "get_uc_ipm_guidance":
            return _get_uc_ipm_guidance(tool_args)
    except requests.exceptions.Timeout:
        logger.warning("Tool %s timed out", tool_name)
        return {"error": "timeout", "message": f"The {tool_name} request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        logger.error("Tool %s request failed: %s", tool_name, e)
        return {"error": "request_failed", "message": "Could not retrieve data. Please try again."}
    except Exception as e:
        logger.exception("Tool %s unexpected error: %s", tool_name, e)
        return {"error": "unexpected", "message": "An unexpected error occurred."}

    # Polygon-dependent tools — require a drawn field
    if not polygon_geojson:
        return {"error": "no_polygon", "message": "No farm polygon available. Please draw a field on the map first."}

    try:
        if tool_name == "get_soil_stats":
            return _get_soil_stats(polygon_geojson, tool_args)
        elif tool_name == "get_ndvi_timeseries":
            return _get_ndvi_timeseries(polygon_geojson, tool_args)
        elif tool_name == "get_soil_sample_points":
            return _get_soil_sample_points(polygon_geojson, tool_args)
        else:
            return {"error": "unknown_tool", "message": f"Unknown tool: {tool_name}"}
    except requests.exceptions.Timeout:
        logger.warning("Tool %s timed out", tool_name)
        return {"error": "timeout", "message": f"The {tool_name} request timed out. The date range may be too large — try a shorter range (1-2 months)."}
    except requests.exceptions.RequestException as e:
        logger.error("Tool %s request failed: %s", tool_name, e)
        return {"error": "request_failed", "message": "Could not retrieve field data. Please try again."}
    except Exception as e:
        logger.exception("Tool %s unexpected error: %s", tool_name, e)
        return {"error": "unexpected", "message": "An unexpected error occurred."}


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def _get_soil_stats(polygon_geojson: dict, args: dict) -> dict:
    """Call POST /soil/singlepolygon.json"""
    layer = args.get("layer", "ph")
    soildepth = args.get("soildepth", "0-5")

    url = f"{_BASE_URL}/soil/singlepolygon.json"
    params = {"layer": layer, "soildepth": soildepth}
    headers = {"Content-Type": "application/geo+json"}
    body = json.dumps(polygon_geojson)

    resp = requests.post(url, params=params, data=body, headers=headers, timeout=_TIMEOUT_SOIL)
    resp.raise_for_status()
    data = resp.json()

    # Flatten for easier model consumption
    results = data.get("results", {})
    return {
        "layer": layer,
        "depth_cm": soildepth,
        "mean": results.get("mean"),
        "min": results.get("min"),
        "max": results.get("max"),
        "median": results.get("median"),
        "stddev": results.get("stddev"),
        "count": results.get("count"),
    }


def _get_ndvi_timeseries(polygon_geojson: dict, args: dict) -> dict:
    """Call POST /ndvi/singlepolygon.json"""
    from_date = args.get("from_date", "")
    to_date = args.get("to_date", "")

    url = f"{_BASE_URL}/ndvi/singlepolygon.json"
    params = {"from": from_date, "to": to_date}
    headers = {"Content-Type": "application/geo+json"}
    body = json.dumps(polygon_geojson)

    resp = requests.post(url, params=params, data=body, headers=headers, timeout=_TIMEOUT_NDVI)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        return {"date_range": f"{from_date} to {to_date}", "data_points": 0, "message": "No NDVI data available for this period."}

    # Compute summary stats from time series
    values = [r["mean"] for r in results if r.get("mean") is not None]
    avg_ndvi = round(sum(values) / len(values), 3) if values else None
    max_ndvi = round(max(values), 3) if values else None
    min_ndvi = round(min(values), 3) if values else None

    # Include the series (capped at 50 points for context window)
    series = results[-50:] if len(results) > 50 else results

    return {
        "date_range": f"{from_date} to {to_date}",
        "data_points": len(results),
        "average_ndvi": avg_ndvi,
        "max_ndvi": max_ndvi,
        "min_ndvi": min_ndvi,
        "series": series,
    }


# ---------------------------------------------------------------------------
# New location-based tools (no polygon required)
# ---------------------------------------------------------------------------

def _get_cimis_weather(args: dict) -> dict:
    """Fetch CIMIS daily weather/ET data for nearest station to lat/lon."""
    from .cimis_client import fetch_daily

    lat        = float(args.get("lat", 0))
    lon        = float(args.get("lon", 0))
    start_date = args.get("start_date", "")
    end_date   = args.get("end_date", "")

    if not lat or not lon:
        return {"error": "missing_location", "message": "Latitude and longitude are required for CIMIS data."}
    if not start_date or not end_date:
        return {"error": "missing_dates", "message": "start_date and end_date are required."}

    records, station = fetch_daily(lat, lon, start_date, end_date)
    if not records:
        return {"error": "no_data", "message": "No CIMIS data available for this location and date range."}

    # Compute summary stats
    eto_vals   = [r["eto_in"]   for r in records if r["eto_in"]   is not None]
    precip_vals= [r["precip_in"]for r in records if r["precip_in"]is not None]
    tmax_vals  = [r["tmax_f"]   for r in records if r["tmax_f"]   is not None]
    tmin_vals  = [r["tmin_f"]   for r in records if r["tmin_f"]   is not None]
    tavg_vals  = [r["tavg_f"]   for r in records if r["tavg_f"]   is not None]

    def _avg(vals): return round(sum(vals) / len(vals), 2) if vals else None
    def _total(vals): return round(sum(vals), 2) if vals else None

    return {
        "station":       f"{station['city']}, {station['county']} (Station #{station['nbr']})",
        "station_dist_km": station["dist_km"],
        "date_range":    f"{start_date} to {end_date}",
        "days":          len(records),
        "total_eto_in":  _total(eto_vals),
        "total_precip_in": _total(precip_vals),
        "avg_tmax_f":    _avg(tmax_vals),
        "avg_tmin_f":    _avg(tmin_vals),
        "avg_temp_f":    _avg(tavg_vals),
        "daily":         records[-14:],  # last 14 days for context window efficiency
    }


def _get_pest_disease_risk(args: dict) -> dict:
    """Compute GDD-based pest risk and disease window risk for last 30 days."""
    from .cimis_client import last_n_days
    from .pest_risk import compute_risk

    lat  = float(args.get("lat", 0))
    lon  = float(args.get("lon", 0))
    crop = args.get("crop")

    if not lat or not lon:
        return {"error": "missing_location", "message": "Latitude and longitude are required for pest risk."}

    records, station = last_n_days(lat, lon, n=30)
    if not records:
        return {"error": "no_data", "message": "No temperature data available for this location."}

    risk = compute_risk(records, crop=crop)
    risk["station"] = f"{station['city']}, {station['county']} (Station #{station['nbr']})"
    risk["station_dist_km"] = station["dist_km"]
    return risk


def _get_market_prices(args: dict) -> dict:
    """Fetch USDA MARS wholesale market prices for a CA commodity."""
    from .market_client import get_prices

    commodity = args.get("commodity", "")
    if not commodity:
        return {"error": "missing_commodity", "message": "A commodity name is required."}

    return get_prices(commodity)


def _get_uc_ipm_guidance(args: dict) -> dict:
    """Fetch UC IPM pest management guidance for a crop/pest."""
    from .uc_ipm_client import get_ipm_guidance

    crop = args.get("crop", "")
    pest = args.get("pest")

    if not crop:
        return {"error": "missing_crop", "message": "A crop name is required."}

    return get_ipm_guidance(crop, pest)


def _get_soil_sample_points(polygon_geojson: dict, args: dict) -> dict:
    """Call POST /soil/sample.json"""
    num_points = int(args.get("num_points", 7))
    layers = args.get("layers", ["ph", "om"])
    soildepth = args.get("soildepth", "0-5")

    url = f"{_BASE_URL}/soil/sample.json"
    # layers is a repeatable query param
    params = [("num_points", num_points), ("soildepth", soildepth)]
    for layer in layers:
        params.append(("layer", layer))

    headers = {"Content-Type": "application/geo+json"}
    body = json.dumps(polygon_geojson)

    resp = requests.post(url, params=params, data=body, headers=headers, timeout=_TIMEOUT_SOIL)
    resp.raise_for_status()
    data = resp.json()

    points = data.get("results", [])
    return {
        "num_points": len(points),
        "layers_optimized_for": layers,
        "depth_cm": soildepth,
        "sample_locations": [
            {"id": p.get("id"), "lat": p.get("y"), "lon": p.get("x")}
            for p in points
        ],
    }
