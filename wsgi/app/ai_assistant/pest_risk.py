"""
GDD-based pest risk and disease window risk computation.

Each insect pest uses its own biologically-correct base temperature.
Disease models use multi-day temperature + humidity windows (NOT GDD).

Biological notes:
- Powdery mildew: optimal at 60–85°F with moderate/dry RH (40–75%).
  High humidity suppresses conidia germination; moderate humidity + warm
  temps drive sporulation and spread.
- Botrytis/gray mold: favors cool + HIGH humidity (55–75°F, RH > 85%),
  especially with leaf wetness or morning dew.
- Downy mildew: requires cool + very high humidity (50–70°F, RH > 90%).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Insect pest models (GDD-based)
# ---------------------------------------------------------------------------

# base: lower development threshold in °F
# thresholds: [low_end, moderate_end, high_start] GDD accumulation
# crops: list of crops where this pest is most relevant
GDD_MODELS: dict[str, dict] = {
    "navel_orangeworm": {
        "base": 55,
        "thresholds": [200, 500, 900],
        "crops": ["almonds", "walnuts", "pistachios", "figs", "pomegranates"],
        "label": "Navel Orangeworm",
    },
    "codling_moth": {
        "base": 50,
        "thresholds": [250, 600, 1000],
        "crops": ["apples", "pears", "walnuts", "stone fruit"],
        "label": "Codling Moth",
    },
    "thrips": {
        "base": 45,
        "thresholds": [150, 400, 700],
        "crops": ["onions", "tomatoes", "strawberries", "grapes", "peppers", "garlic"],
        "label": "Western Flower Thrips",
    },
    "spider_mites": {
        "base": 55,
        "thresholds": [100, 300, 600],
        "crops": ["almonds", "grapes", "corn", "beans", "strawberries", "cotton"],
        "label": "Spider Mites",
    },
    "aphids": {
        "base": 45,
        "thresholds": [120, 350, 650],
        "crops": ["lettuce", "broccoli", "tomatoes", "peppers", "grains", "potatoes"],
        "label": "Aphids",
    },
}

# ---------------------------------------------------------------------------
# Disease models (weather window-based)
# ---------------------------------------------------------------------------

# For each disease: temp range (°F) and humidity condition that signal risk.
# favorable_days = count of days in the records that match conditions.
DISEASE_MODELS: dict[str, dict] = {
    "powdery_mildew": {
        "temp_min": 60,
        "temp_max": 85,
        "rh_min": 40,   # moderate humidity — conducive range
        "rh_max": 75,   # above 80% RH actually suppresses germination
        "thresholds": [3, 7, 12],  # favorable days → low/moderate/high
        "crops": ["grapes", "strawberries", "cucurbits", "tomatoes", "squash"],
        "label": "Powdery Mildew",
    },
    "botrytis_gray_mold": {
        "temp_min": 55,
        "temp_max": 75,
        "rh_min": 85,   # requires HIGH humidity / leaf wetness
        "rh_max": 100,
        "thresholds": [2, 5, 9],
        "crops": ["strawberries", "lettuce", "grapes", "tomatoes", "peppers"],
        "label": "Botrytis (Gray Mold)",
    },
    "downy_mildew": {
        "temp_min": 50,
        "temp_max": 70,
        "rh_min": 90,   # very high humidity required
        "rh_max": 100,
        "thresholds": [2, 4, 7],
        "crops": ["grapes", "lettuce", "spinach", "cucumbers", "basil"],
        "label": "Downy Mildew",
    },
}


# ---------------------------------------------------------------------------
# Risk level helper
# ---------------------------------------------------------------------------

def _risk_level(value: float, thresholds: list[int]) -> str:
    """Map an accumulated value to low/moderate/high using 3-element thresholds list."""
    low_end, mod_end, _ = thresholds
    if value < low_end:
        return "low"
    if value < mod_end:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# GDD computation
# ---------------------------------------------------------------------------

def _gdd_per_day(tmax_f: float | None, tmin_f: float | None, base: float) -> float:
    """Single-triangle GDD: max(0, (tmax + tmin) / 2 - base)."""
    if tmax_f is None or tmin_f is None:
        return 0.0
    return max(0.0, (tmax_f + tmin_f) / 2.0 - base)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_risk(
    daily_records: list[dict],
    crop: str | None = None,
) -> dict:
    """
    Compute pest and disease risk from CIMIS daily records.

    Args:
        daily_records: List of dicts from cimis_client.fetch_daily —
                       each must have: tmax_f, tmin_f, tavg_f, rh_avg, rh_max
        crop:          Optional crop name to filter pests. If None, all pests returned.

    Returns:
        {
          "pest_risks": {
            "navel_orangeworm": {
              "label": "Navel Orangeworm",
              "base_temp_f": 55,
              "gdd_accumulated": 312.4,
              "risk_level": "moderate",
              "days_analyzed": 30,
            }, ...
          },
          "disease_risks": {
            "powdery_mildew": {
              "label": "Powdery Mildew",
              "favorable_days": 8,
              "days_analyzed": 30,
              "risk_level": "moderate",
            }, ...
          },
          "days_analyzed": 30,
          "crop_filter": "tomatoes" | None,
        }
    """
    crop_lower = crop.lower().strip() if crop else None
    n = len(daily_records)

    # ------------------------------------------------------------------
    # Insect pest GDD accumulation
    # ------------------------------------------------------------------
    pest_results: dict[str, dict] = {}

    for pest_key, model in GDD_MODELS.items():
        # Filter by crop relevance if crop specified
        if crop_lower:
            relevant = any(crop_lower in c or c in crop_lower for c in model["crops"])
            if not relevant:
                continue

        base = model["base"]
        accumulated = sum(
            _gdd_per_day(r.get("tmax_f"), r.get("tmin_f"), base)
            for r in daily_records
        )
        accumulated = round(accumulated, 1)

        pest_results[pest_key] = {
            "label":           model["label"],
            "base_temp_f":     base,
            "gdd_accumulated": accumulated,
            "risk_level":      _risk_level(accumulated, model["thresholds"]),
            "days_analyzed":   n,
            "relevant_crops":  model["crops"],
        }

    # ------------------------------------------------------------------
    # Disease favorable-day counting
    # ------------------------------------------------------------------
    disease_results: dict[str, dict] = {}

    for disease_key, model in DISEASE_MODELS.items():
        if crop_lower:
            relevant = any(crop_lower in c or c in crop_lower for c in model["crops"])
            if not relevant:
                continue

        favorable_days = 0
        for r in daily_records:
            tavg = r.get("tavg_f")
            rh   = r.get("rh_avg")

            if tavg is None or rh is None:
                continue

            temp_ok = model["temp_min"] <= tavg <= model["temp_max"]
            rh_ok   = model["rh_min"]   <= rh   <= model["rh_max"]

            if temp_ok and rh_ok:
                favorable_days += 1

        disease_results[disease_key] = {
            "label":          model["label"],
            "favorable_days": favorable_days,
            "days_analyzed":  n,
            "risk_level":     _risk_level(favorable_days, model["thresholds"]),
            "relevant_crops": model["crops"],
        }

    return {
        "pest_risks":    pest_results,
        "disease_risks": disease_results,
        "days_analyzed": n,
        "crop_filter":   crop,
    }
