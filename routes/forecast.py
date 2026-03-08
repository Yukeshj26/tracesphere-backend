"""
ML Forecast routes — with caching so model trains only once
"""
import warnings
warnings.filterwarnings("ignore")

from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta, timezone
from services.firebase import get_db
from services.ml_model import get_model, SEASONAL_MULTIPLIERS

router = APIRouter()

# ── Cache — avoid retraining on every request ─────────────────────────────────
_cache = {
    "results":     None,
    "summary":     None,
    "analytics":   None,
    "expires_at":  None,
}
CACHE_MINUTES = 5


def _now():
    return datetime.now(timezone.utc)


def _is_cache_valid():
    return (
        _cache["results"] is not None and
        _cache["expires_at"] is not None and
        _now() < _cache["expires_at"]
    )


def _fetch_data():
    db          = get_db()
    assets      = [doc.to_dict() for doc in db.collection("assets").stream()]
    procurement = [doc.to_dict() for doc in db.collection("procurement").stream()]
    return assets, procurement


def _run_and_cache():
    """Fetch data, run ML, store in cache."""
    assets, procurement = _fetch_data()
    model = get_model()

    results   = model.predict(assets, procurement)
    analytics = model.report_analytics(assets, procurement)
    summary   = {
        "critical":      sum(1 for r in results if r["risk"] == "critical"),
        "high":          sum(1 for r in results if r["risk"] == "high"),
        "medium":        sum(1 for r in results if r["risk"] == "medium"),
        "low":           sum(1 for r in results if r["risk"] == "low"),
        "totalBudget":   sum(r["estimatedBudget"] for r in results if r["risk"] != "low"),
        "avgConfidence": round(sum(r["confidence"] for r in results) / len(results), 1) if results else 0,
        "currentSeason": results[0]["seasonLabel"] if results else "Unknown",
        "totalAssets":   len(results),
    }

    _cache["results"]    = results
    _cache["summary"]    = summary
    _cache["analytics"]  = analytics
    _cache["expires_at"] = _now() + timedelta(minutes=CACHE_MINUTES)

    return results, summary, analytics


@router.get("/")
def get_forecast(
    risk:       Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    category:   Optional[str] = Query(None),
):
    if not _is_cache_valid():
        _run_and_cache()

    results = _cache["results"]
    if risk:       results = [r for r in results if r["risk"]       == risk]
    if department: results = [r for r in results if r["department"] == department]
    if category:   results = [r for r in results if r["category"]   == category]

    return {
        "generatedAt": _now().isoformat(),
        "totalAssets": len(_cache["results"]),
        "filtered":    len(results),
        "model":       "RandomForest + GradientBoosting + Ridge Ensemble",
        "cached":      True,
        "results":     results,
    }


@router.get("/summary")
def get_forecast_summary():
    if not _is_cache_valid():
        _run_and_cache()
    return _cache["summary"]


@router.get("/analytics")
def get_analytics():
    if not _is_cache_valid():
        _run_and_cache()
    return _cache["analytics"]


@router.get("/critical")
def get_critical_items():
    if not _is_cache_valid():
        _run_and_cache()
    urgent = [r for r in _cache["results"] if r["risk"] in ("critical", "high")]
    return {"count": len(urgent), "items": urgent}


@router.post("/train")
def retrain_model():
    """Force retrain + clear cache."""
    assets, procurement = _fetch_data()
    model = get_model()
    stats = model.train(assets, procurement, verbose=False)
    model.save("models")
    _cache["results"]    = None
    _cache["expires_at"] = None
    return {"message": "Model retrained, cache cleared", "stats": stats}


@router.post("/refresh")
def refresh_cache():
    """Force re-run forecast without retraining."""
    _run_and_cache()
    return {"message": "Cache refreshed", "totalAssets": len(_cache["results"])}


@router.get("/seasonal-calendar")
def get_seasonal_calendar():
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
    return {"calendar": [
        {"month": months[i], "multiplier": SEASONAL_MULTIPLIERS[i+1],
         "level": "peak"   if SEASONAL_MULTIPLIERS[i+1] >= 1.4 else
                  "high"   if SEASONAL_MULTIPLIERS[i+1] >= 1.2 else
                  "normal" if SEASONAL_MULTIPLIERS[i+1] >= 0.9 else "low"}
        for i in range(12)
    ]}
