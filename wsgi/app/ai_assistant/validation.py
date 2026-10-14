"""
Input validation for FarmBot AI assistant.

Validates untrusted data before it reaches Gemini or the internal FFN APIs:
  - Client-supplied polygon GeoJSON (structure, validity, size, region)
  - Model-supplied tool arguments (types, ranges, dates, enums)

All validators return (value, error). On failure value is None and error is a
short user-safe message that can be returned to the client or the model.
"""

import datetime
import math
import re
from typing import Any

from pyproj import Geod
from shapely.geometry import shape
from shapely.validation import explain_validity

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

# Bounding box covering California and Arizona (the regions FFN has data for)
REGION_MIN_LON, REGION_MAX_LON = -124.6, -108.9
REGION_MIN_LAT, REGION_MAX_LAT = 31.2, 42.1

MAX_POLYGON_VERTICES = 2000
MAX_POLYGON_PARTS = 50
MAX_FARM_AREA_KM2 = 100.0        # ~24,700 acres

NDVI_MIN_DATE = datetime.date(2015, 6, 23)   # Sentinel-2A launch
CIMIS_MIN_DATE = datetime.date(1982, 1, 1)   # CIMIS network start
MAX_NDVI_SPAN_DAYS = 366
MAX_CIMIS_SPAN_DAYS = 366

SOIL_LAYERS = {
    "alpha", "bd", "clay", "hb", "ksat", "lambda",
    "n", "om", "ph", "sand", "silt", "theta_r", "theta_s",
}
SOIL_DEPTHS = {"0-5", "5-15", "15-30", "30-60", "60-100", "100-200"}
SAMPLE_POINT_CHOICES = {5, 7, 10, 12}
MAX_SAMPLE_LAYERS = 5

MAX_TEXT_ARG_LEN = 60
_TEXT_ARG_RE = re.compile(r"^[\w\s\-'(),.&/]+$", re.UNICODE)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_ring(ring: Any) -> str | None:
    if not isinstance(ring, list) or len(ring) < 4:
        return "Each polygon ring needs at least 4 positions."
    for pos in ring:
        if not isinstance(pos, list) or len(pos) < 2 or not all(_is_number(c) for c in pos[:2]):
            return "Polygon coordinates must be [longitude, latitude] numbers."
        lon, lat = pos[0], pos[1]
        if not (REGION_MIN_LON <= lon <= REGION_MAX_LON and REGION_MIN_LAT <= lat <= REGION_MAX_LAT):
            return "The drawn area is outside the supported region (California and Arizona)."
    if ring[0][:2] != ring[-1][:2]:
        return "Polygon rings must be closed."
    return None


def validate_polygon(geojson: Any) -> tuple[dict | None, str | None]:
    """Validate a GeoJSON Polygon/MultiPolygon geometry drawn by the user."""
    if not isinstance(geojson, dict):
        return None, "Invalid field geometry."

    geom_type = geojson.get("type")
    coords = geojson.get("coordinates")
    if geom_type == "Polygon":
        polygons = [coords]
    elif geom_type == "MultiPolygon":
        polygons = coords
    else:
        return None, "Field geometry must be a Polygon or MultiPolygon."

    if not isinstance(polygons, list) or not polygons or len(polygons) > MAX_POLYGON_PARTS:
        return None, "Invalid field geometry."

    vertex_count = 0
    for poly in polygons:
        if not isinstance(poly, list) or not poly:
            return None, "Invalid field geometry."
        for ring in poly:
            err = _check_ring(ring)
            if err:
                return None, err
            vertex_count += len(ring)
            if vertex_count > MAX_POLYGON_VERTICES:
                return None, f"The drawn area is too detailed (max {MAX_POLYGON_VERTICES} points)."

    try:
        geom = shape({"type": geom_type, "coordinates": coords})
    except Exception:
        return None, "Invalid field geometry."
    if geom.is_empty or not geom.is_valid:
        reason = explain_validity(geom) if not geom.is_empty else "empty"
        return None, f"The drawn area is not a valid shape ({reason.split('[')[0].strip()})."

    area_m2 = abs(_GEOD.geometry_area_perimeter(geom)[0])
    if area_m2 / 1e6 > MAX_FARM_AREA_KM2:
        return None, (
            f"The drawn area is too large for FarmBot "
            f"(max {MAX_FARM_AREA_KM2:.0f} km² / ~{MAX_FARM_AREA_KM2 * 247.1:,.0f} acres)."
        )

    # Return a clean copy containing only the validated fields
    return {"type": geom_type, "coordinates": coords}, None


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

def validate_lat_lon(lat: Any, lon: Any) -> tuple[tuple[float, float] | None, str | None]:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None, "Latitude and longitude must be numbers."
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return None, "Latitude and longitude must be numbers."
    if not (REGION_MIN_LAT <= lat_f <= REGION_MAX_LAT and REGION_MIN_LON <= lon_f <= REGION_MAX_LON):
        return None, "Location is outside the supported region (California and Arizona)."
    return (lat_f, lon_f), None


def _parse_iso_date(value: Any) -> datetime.date | None:
    if not isinstance(value, str) or not _ISO_DATE_RE.match(value):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def validate_date_range(
    start: Any,
    end: Any,
    min_date: datetime.date,
    max_span_days: int,
) -> tuple[tuple[str, str] | None, str | None]:
    """Validate an ISO YYYY-MM-DD date range; end is clamped to today."""
    start_d, end_d = _parse_iso_date(start), _parse_iso_date(end)
    if not start_d or not end_d:
        return None, "Dates must be in YYYY-MM-DD format."
    today = datetime.date.today()
    end_d = min(end_d, today)
    if start_d > end_d:
        return None, "Start date must be on or before the end date (and not in the future)."
    if start_d < min_date:
        return None, f"Start date must be on or after {min_date.isoformat()}."
    if (end_d - start_d).days > max_span_days:
        return None, f"Date range is too long (max {max_span_days} days). Try a shorter range."
    return (start_d.isoformat(), end_d.isoformat()), None


def validate_soil_layer(layer: Any) -> tuple[str | None, str | None]:
    if isinstance(layer, str) and layer in SOIL_LAYERS:
        return layer, None
    return None, f"Unsupported soil layer. Use one of: {', '.join(sorted(SOIL_LAYERS))}."


def validate_soil_depth(depth: Any) -> tuple[str | None, str | None]:
    if isinstance(depth, str) and depth in SOIL_DEPTHS:
        return depth, None
    return None, f"Unsupported soil depth. Use one of: {', '.join(sorted(SOIL_DEPTHS, key=lambda d: int(d.split('-')[0])))}."


def validate_num_points(value: Any) -> tuple[int | None, str | None]:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None, "num_points must be a whole number."
    if n not in SAMPLE_POINT_CHOICES:
        return None, f"num_points must be one of {sorted(SAMPLE_POINT_CHOICES)}."
    return n, None


def validate_soil_layers(layers: Any) -> tuple[list[str] | None, str | None]:
    if isinstance(layers, str):
        layers = [layers]
    if not isinstance(layers, list) or not layers:
        return None, "layers must be a non-empty list of soil layers."
    if len(layers) > MAX_SAMPLE_LAYERS:
        return None, f"At most {MAX_SAMPLE_LAYERS} soil layers can be optimized at once."
    clean = []
    for layer in layers:
        value, err = validate_soil_layer(layer)
        if err:
            return None, err
        if value not in clean:
            clean.append(value)
    return clean, None


def validate_text_arg(value: Any, name: str, required: bool = True) -> tuple[str | None, str | None]:
    """Validate a short free-text tool argument such as a crop or commodity name."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return (None, f"A {name} is required.") if required else (None, None)
    if not isinstance(value, str):
        return None, f"{name} must be text."
    value = value.strip()
    if len(value) > MAX_TEXT_ARG_LEN or not _TEXT_ARG_RE.match(value):
        return None, f"Invalid {name}."
    return value, None


def is_https_url(url: Any) -> bool:
    return isinstance(url, str) and url.lower().startswith("https://") and len(url) <= 2048
