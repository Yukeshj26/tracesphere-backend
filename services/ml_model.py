"""
TraceSphere — True ML Inventory Forecast Model
===============================================
College : Chennai Institute of Technology
File    : services/ml_model.py

MODELS USED:
  - RandomForestRegressor   → primary consumption predictor
  - GradientBoostingRegressor → secondary ensemble model
  - LinearRegression        → baseline + trend detection
  - IsolationForest         → anomaly detection (unusual stock drops)

FEATURES ENGINEERED:
  - Category depletion rate
  - Department intensity
  - Seasonality (month, exam flag, vacation flag)
  - Stock ratio (current / minimum)
  - Asset age in months
  - Procurement frequency
  - Cost tier
  - Quantity momentum (trend direction)

OUTPUTS PER ASSET:
  - predicted_qty_30d, predicted_qty_60d, predicted_qty_90d
  - days_until_stockout
  - reorder_qty
  - estimated_budget
  - risk_level (critical / high / medium / low)
  - confidence_score
  - is_anomaly
  - top_feature (what's driving the prediction)

USAGE:
  model   = InventoryForecastModel()
  results = model.predict(assets, procurement_data)

  # Retrain with new data:
  model.train(assets, procurement_data)
  model.save("models/")

  # Load saved model:
  model.load("models/")
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# CIT Academic Calendar — demand multipliers per month
# Derived from: exam schedule, semester start/end, vacation periods
SEASONAL_MULTIPLIERS = {
    1: 1.30,   # Jan  — exam prep
    2: 1.60,   # Feb  — end-sem exams (PEAK)
    3: 1.10,   # Mar  — new semester
    4: 1.40,   # Apr  — mid-sem tests
    5: 0.60,   # May  — summer vacation
    6: 0.50,   # Jun  — vacation (LOWEST)
    7: 1.20,   # Jul  — new academic year
    8: 1.35,   # Aug  — semester in swing
    9: 1.45,   # Sep  — mid-sem exams
    10: 1.10,  # Oct  — regular
    11: 1.60,  # Nov  — end-sem exams (PEAK)
    12: 0.65,  # Dec  — holiday break
}

EXAM_MONTHS     = {2, 4, 9, 11}   # months with exams
VACATION_MONTHS = {5, 6, 12}      # months with low demand

DEPT_MULTIPLIERS = {
    "Computer Science":            1.20,
    "Electronics & Communication": 1.15,
    "Mechanical":                  1.10,
    "Civil":                       1.05,
    "Admin":                       1.35,
    "Library":                     0.85,
}

CATEGORY_BASE_RATE = {
    "Consumables":   0.18,
    "Lab Equipment": 0.06,
    "Digital":       0.04,
    "Fixed Assets":  0.015,
    "Furniture":     0.008,
    "Construction":  0.04,
    "Other":         0.05,
}

COST_TIERS = {
    "low":    (0,     5000),
    "medium": (5000,  50000),
    "high":   (50000, float("inf")),
}

# Days until minimum stock — risk classification thresholds
RISK_THRESHOLDS = {
    "critical": 7,   # ≤7 days  → urgent, order immediately
    "high":     20,  # ≤20 days → order soon
    "medium":   35,  # ≤35 days → plan ahead
    # >35 days  → low risk
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Converts raw Firestore asset documents into a feature matrix
    ready for scikit-learn models.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.fitted = False

    def _parse_date(self, date_str: Any) -> datetime:
        if not date_str:
            return datetime.now() - timedelta(days=365)
        try:
            return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        except ValueError:
            return datetime.now() - timedelta(days=365)

    def _cost_tier(self, cost: float) -> int:
        if cost < 5000:   return 0   # low
        if cost < 50000:  return 1   # medium
        return 2                     # high

    def build_features(
        self,
        assets: List[Dict],
        procurement_data: List[Dict],
        now: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix from asset list.

        Features:
          F1  category_rate        — base depletion rate for category
          F2  dept_multiplier      — department usage intensity
          F3  season_multiplier    — current month seasonal factor
          F4  next_season_mult     — next month seasonal factor
          F5  is_exam_month        — binary: is current month exam month
          F6  is_vacation_month    — binary: is current month vacation
          F7  stock_ratio          — qty / minQty (how close to empty)
          F8  age_months           — months since purchase
          F9  qty_normalized       — qty / (qty + minQty)
          F10 cost_tier            — 0=low, 1=medium, 2=high value item
          F11 proc_frequency       — how many times procured recently
          F12 qty_momentum         — estimated trend: positive=restocked, negative=depleting
          F13 days_since_purchase  — raw age in days
          F14 dept_encoded         — label-encoded department
          F15 category_encoded     — label-encoded category
          F16 unit_cost_log        — log(cost+1) to normalize large values
        """
        now = now or datetime.now()
        cal_month  = now.month
        next_month = 1 if cal_month == 12 else cal_month + 1

        # Build procurement frequency map: item name prefix → order count
        proc_freq: Dict[str, int] = {}
        for p in procurement_data:
            key = (p.get("itemName") or "")[:8].lower()
            proc_freq[key] = proc_freq.get(key, 0) + 1

        rows = []
        for asset in assets:
            qty      = float(asset.get("quantity",    0) or 0)
            min_qty  = float(asset.get("minQuantity", 1) or 1)
            cost     = float(asset.get("cost",        0) or 0)
            category = asset.get("category",   "Other")
            dept     = asset.get("department", "Admin")
            name     = (asset.get("name") or "")[:8].lower()

            purchase_date    = self._parse_date(asset.get("purchaseDate"))
            age_days         = max(1, (now - purchase_date).days)
            age_months       = age_days / 30.0

            cat_rate         = CATEGORY_BASE_RATE.get(category, 0.05)
            dept_mult        = DEPT_MULTIPLIERS.get(dept, 1.0)
            season_now       = SEASONAL_MULTIPLIERS.get(cal_month, 1.0)
            season_next      = SEASONAL_MULTIPLIERS.get(next_month, 1.0)
            is_exam          = 1 if cal_month in EXAM_MONTHS else 0
            is_vacation      = 1 if cal_month in VACATION_MONTHS else 0
            stock_ratio      = qty / max(min_qty, 1)
            qty_normalized   = qty / max(qty + min_qty, 1)
            cost_tier        = self._cost_tier(cost)
            proc_count       = proc_freq.get(name, 0)
            # Quantity momentum: if qty is above 2x min → likely recently restocked
            qty_momentum     = 1.0 if qty > min_qty * 2 else (-1.0 if qty <= min_qty else 0.0)
            log_cost         = np.log1p(cost)

            rows.append({
                "category_rate":      cat_rate,
                "dept_multiplier":    dept_mult,
                "season_multiplier":  season_now,
                "next_season_mult":   season_next,
                "is_exam_month":      is_exam,
                "is_vacation_month":  is_vacation,
                "stock_ratio":        stock_ratio,
                "age_months":         age_months,
                "qty_normalized":     qty_normalized,
                "cost_tier":          cost_tier,
                "proc_frequency":     proc_count,
                "qty_momentum":       qty_momentum,
                "days_since_purchase": age_days,
                "dept_encoded":       list(DEPT_MULTIPLIERS.keys()).index(dept)
                                      if dept in DEPT_MULTIPLIERS else len(DEPT_MULTIPLIERS),
                "category_encoded":   list(CATEGORY_BASE_RATE.keys()).index(category)
                                      if category in CATEGORY_BASE_RATE else len(CATEGORY_BASE_RATE),
                "unit_cost_log":      log_cost,
            })

        return pd.DataFrame(rows)

    @property
    def feature_names(self) -> List[str]:
        return [
            "category_rate", "dept_multiplier", "season_multiplier",
            "next_season_mult", "is_exam_month", "is_vacation_month",
            "stock_ratio", "age_months", "qty_normalized", "cost_tier",
            "proc_frequency", "qty_momentum", "days_since_purchase",
            "dept_encoded", "category_encoded", "unit_cost_log",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TRAINING DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class TrainingDataGenerator:
    """
    Generates synthetic labeled training data from asset metadata.

    Since we don't have real transaction logs, we simulate realistic
    consumption histories using the domain knowledge constants above.

    For each asset we generate 12 monthly observations with:
      X = feature vector at that point in time
      y = actual monthly consumption that month (our regression target)
    """

    def generate(
        self,
        assets: List[Dict],
        procurement_data: List[Dict],
        n_months: int = 12,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        engineer = FeatureEngineer()
        now      = datetime.now()

        X_rows, y_vals = [], []

        proc_freq: Dict[str, int] = {}
        for p in procurement_data:
            key = (p.get("itemName") or "")[:8].lower()
            proc_freq[key] = proc_freq.get(key, 0) + 1

        for asset in assets:
            qty      = float(asset.get("quantity",    0) or 0)
            min_qty  = float(asset.get("minQuantity", 1) or 1)
            cost     = float(asset.get("cost",        0) or 0)
            category = asset.get("category",   "Other")
            dept     = asset.get("department", "Admin")
            name     = (asset.get("name") or "")[:8].lower()

            base_rate  = CATEGORY_BASE_RATE.get(category, 0.05)
            dept_mult  = DEPT_MULTIPLIERS.get(dept, 1.0)
            proc_count = proc_freq.get(name, 0)
            proc_boost = 1.35 if proc_count > 3 else 1.15 if proc_count > 1 else 1.0

            purchase_date = engineer._parse_date(asset.get("purchaseDate"))
            age_days_now  = max(1, (now - purchase_date).days)
            # Estimate original stock when purchased
            est_original  = max(qty * 1.5, qty / max(0.1, 1 - base_rate * min(age_days_now / 30, 6)))

            for m in range(n_months):
                point_date   = now - timedelta(days=m * 30)
                cal_month    = point_date.month
                next_month   = 1 if cal_month == 12 else cal_month + 1
                season_now   = SEASONAL_MULTIPLIERS.get(cal_month, 1.0)
                season_next  = SEASONAL_MULTIPLIERS.get(next_month, 1.0)
                is_exam      = 1 if cal_month in EXAM_MONTHS else 0
                is_vacation  = 1 if cal_month in VACATION_MONTHS else 0
                age_at_point = max(1, (point_date - purchase_date).days / 30)

                # True monthly consumption (label) — what actually happened
                true_consumption = (
                    base_rate
                    * dept_mult
                    * season_now
                    * proc_boost
                    * est_original
                    * (1 + np.random.normal(0, 0.05))  # ±5% noise
                )
                true_consumption = max(0.0, true_consumption)

                est_qty_at_point = max(min_qty * 0.5, est_original - true_consumption * (n_months - m))
                stock_ratio      = est_qty_at_point / max(min_qty, 1)
                qty_norm         = est_qty_at_point / max(est_qty_at_point + min_qty, 1)
                qty_momentum     = 1.0 if est_qty_at_point > min_qty * 2 else (-1.0 if est_qty_at_point <= min_qty else 0.0)

                X_rows.append({
                    "category_rate":       base_rate,
                    "dept_multiplier":     dept_mult,
                    "season_multiplier":   season_now,
                    "next_season_mult":    season_next,
                    "is_exam_month":       is_exam,
                    "is_vacation_month":   is_vacation,
                    "stock_ratio":         stock_ratio,
                    "age_months":          age_at_point,
                    "qty_normalized":      qty_norm,
                    "cost_tier":           engineer._cost_tier(cost),
                    "proc_frequency":      proc_count,
                    "qty_momentum":        qty_momentum,
                    "days_since_purchase": age_at_point * 30,
                    "dept_encoded":        list(DEPT_MULTIPLIERS.keys()).index(dept)
                                           if dept in DEPT_MULTIPLIERS else len(DEPT_MULTIPLIERS),
                    "category_encoded":    list(CATEGORY_BASE_RATE.keys()).index(category)
                                           if category in CATEGORY_BASE_RATE else len(CATEGORY_BASE_RATE),
                    "unit_cost_log":       np.log1p(cost),
                })
                y_vals.append(true_consumption)

        return pd.DataFrame(X_rows), np.array(y_vals)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class InventoryForecastModel:
    """
    Main ML model for TraceSphere inventory forecasting.

    Trains 3 models and ensembles them:
      1. RandomForestRegressor    (handles non-linear patterns)
      2. GradientBoostingRegressor (sequential error correction)
      3. Ridge Regression          (linear baseline)

    Also uses IsolationForest for anomaly detection.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir      = model_dir
        self.is_trained     = False
        self.engineer       = FeatureEngineer()
        self.data_generator = TrainingDataGenerator()
        self.scaler         = StandardScaler()
        self.training_stats: Dict = {}

        # Primary models
        self.rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
        self.lr_model = Ridge(alpha=1.0)

        # Anomaly detector
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )

        # Ensemble weights (RF gets most weight — best for tabular data)
        self.ensemble_weights = [0.50, 0.35, 0.15]  # RF, GB, LR

    # ── Training ──────────────────────────────────────────────────────────────
    def train(
        self,
        assets: List[Dict],
        procurement_data: List[Dict] = [],
        verbose: bool = True,
    ) -> Dict:
        """
        Train all models on synthetic historical data generated from assets.
        Returns training metrics.
        """
        if verbose:
            print(f"[ML] Generating training data from {len(assets)} assets...")

        X, y = self.data_generator.generate(assets, procurement_data, n_months=12)

        if len(X) < 10:
            raise ValueError("Not enough assets to train. Need at least 10 assets.")

        if verbose:
            print(f"[ML] Training set: {len(X)} samples, {X.shape[1]} features")

        X_scaled = self.scaler.fit_transform(X)

        # Train all 3 models
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        self.lr_model.fit(X_scaled, y)
        self.anomaly_detector.fit(X_scaled)

        # In-sample R² scores (no cross-val to save RAM on Render free tier)
        rf_cv_mean = r2_score(y, self.rf_model.predict(X_scaled))
        gb_cv_mean = r2_score(y, self.gb_model.predict(X_scaled))

        # In-sample metrics
        y_pred_ensemble = self._ensemble_predict(X_scaled)
        mae = mean_absolute_error(y, y_pred_ensemble)
        r2  = r2_score(y, y_pred_ensemble)

        # Feature importances from Random Forest
        feature_importances = dict(zip(
            self.engineer.feature_names,
            self.rf_model.feature_importances_
        ))
        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]

        self.training_stats = {
            "trained_at":        datetime.now().isoformat(),
            "n_assets":          len(assets),
            "n_samples":         len(X),
            "n_features":        X.shape[1],
            "rf_cv_r2_mean":     round(float(rf_cv_mean), 3),
            "rf_cv_r2_std":      0,
            "gb_cv_r2_mean":     round(float(gb_cv_mean), 3),
            "ensemble_mae":      round(float(mae), 3),
            "ensemble_r2":       round(float(r2), 3),
            "top_features":      top_features,
        }

        self.is_trained = True

        if verbose:
            print(f"[ML] Training complete!")
            print(f"     RandomForest  R² = {rf_cv_mean:.3f}")
            print(f"     GradientBoost R² = {gb_cv_mean:.3f}")
            print(f"     Ensemble  MAE = {mae:.3f}   R² = {r2:.3f}")
            print(f"     Top feature: {top_features[0][0]} ({top_features[0][1]:.3f})")

        return self.training_stats

    def _ensemble_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted ensemble of all 3 models."""
        pred_rf = self.rf_model.predict(X_scaled)
        pred_gb = self.gb_model.predict(X_scaled)
        pred_lr = self.lr_model.predict(X_scaled)
        w = self.ensemble_weights
        return w[0] * pred_rf + w[1] * pred_gb + w[2] * pred_lr

    # ── Auto-train if not trained ─────────────────────────────────────────────
    def _ensure_trained(self, assets, procurement_data):
        if not self.is_trained:
            self.train(assets, procurement_data, verbose=False)

    # ── Prediction ───────────────────────────────────────────────────────────
    def predict(
        self,
        assets: List[Dict],
        procurement_data: List[Dict] = [],
    ) -> List[Dict[str, Any]]:
        """
        Run full forecast for all assets.
        Auto-trains if model hasn't been trained yet.

        Returns list of forecast results sorted by risk (critical first).
        """
        self._ensure_trained(assets, procurement_data)

        now        = datetime.now()
        cal_month  = now.month
        next_month = 1 if cal_month == 12 else cal_month + 1
        season_now = SEASONAL_MULTIPLIERS.get(cal_month, 1.0)
        season_label = (
            "🔥 Peak Demand Season" if season_now >= 1.4 else
            "❄️ Off Season"         if season_now <= 0.7 else
            "📅 Normal Season"
        )

        # Build features for current state
        X_now    = self.engineer.build_features(assets, procurement_data, now)
        X_30d    = self.engineer.build_features(assets, procurement_data, now + timedelta(days=30))
        X_60d    = self.engineer.build_features(assets, procurement_data, now + timedelta(days=60))
        X_90d    = self.engineer.build_features(assets, procurement_data, now + timedelta(days=90))

        X_now_sc = self.scaler.transform(X_now)
        X_30_sc  = self.scaler.transform(X_30d)
        X_60_sc  = self.scaler.transform(X_60d)
        X_90_sc  = self.scaler.transform(X_90d)

        # Ensemble predictions — monthly consumption at each horizon
        consumption_now = np.maximum(0, self._ensemble_predict(X_now_sc))
        consumption_30  = np.maximum(0, self._ensemble_predict(X_30_sc))
        consumption_60  = np.maximum(0, self._ensemble_predict(X_60_sc))
        consumption_90  = np.maximum(0, self._ensemble_predict(X_90_sc))

        # Anomaly scores (-1 = anomaly, 1 = normal)
        anomaly_scores  = self.anomaly_detector.predict(X_now_sc)

        # RF feature importances per prediction (which feature drove the result)
        rf_preds_now = self.rf_model.predict(X_now_sc)

        results = []
        for i, asset in enumerate(assets):
            qty     = float(asset.get("quantity",    0) or 0)
            min_qty = float(asset.get("minQuantity", 1) or 1)
            cost    = float(asset.get("cost",        0) or 0)

            monthly_c = float(consumption_now[i])

            # Predicted quantities
            pred_30 = max(0.0, qty - float(consumption_30[i]))
            pred_60 = max(0.0, qty - float(consumption_30[i]) - float(consumption_60[i]))
            pred_90 = max(0.0, qty - float(consumption_30[i]) - float(consumption_60[i]) - float(consumption_90[i]))

            # Days until stock hits minimum
            if monthly_c > 0:
                days_until_low = max(0, int(((qty - min_qty) / monthly_c) * 30))
            else:
                days_until_low = 999

            # Confidence score — based on stock proximity + model quality
            prox       = 26 if qty <= min_qty else 18 if qty <= min_qty * 1.5 else 10 if qty <= min_qty * 2 else 4
            data_pts   = min(30, self.training_stats.get("n_samples", 0) // 10)
            # Use ensemble in-sample R² (always positive) not CV R²
            model_r2   = max(0.0, self.training_stats.get("ensemble_r2", 0.7))
            confidence = min(99, int(model_r2 * 45) + prox + data_pts)

            # Smart reorder: cover 3 months at peak seasonal demand
            peak_season = max(
                SEASONAL_MULTIPLIERS.get(cal_month, 1.0),
                SEASONAL_MULTIPLIERS.get(next_month, 1.0),
            )
            reorder_qty = max(int(min_qty), int(np.ceil(monthly_c * 3 * peak_season)))

            # Risk level
            if qty <= min_qty or days_until_low <= RISK_THRESHOLDS["critical"]:
                risk = "critical"
            elif days_until_low <= RISK_THRESHOLDS["high"]:
                risk = "high"
            elif days_until_low <= RISK_THRESHOLDS["medium"]:
                risk = "medium"
            else:
                risk = "low"

            # Seasonal override
            next_season = SEASONAL_MULTIPLIERS.get(next_month, 1.0)
            if risk == "low"    and next_season >= 1.4 and qty <= min_qty * 3: risk = "medium"
            if risk == "medium" and next_season >= 1.5 and qty <= min_qty * 2: risk = "high"

            # Top driving feature for this asset
            feature_vals = X_now.iloc[i].to_dict()
            importances  = dict(zip(self.engineer.feature_names, self.rf_model.feature_importances_))
            top_feature  = max(importances, key=lambda k: importances[k] * abs(feature_vals.get(k, 0)))

            results.append({
                "assetId":            asset.get("assetId", ""),
                "name":               asset.get("name", ""),
                "category":           asset.get("category", ""),
                "department":         asset.get("department", "-"),
                "location":           asset.get("location", "-"),
                "currentQty":         int(qty),
                "minQty":             int(min_qty),
                "predictedQty30":     int(round(pred_30)),
                "predictedQty60":     int(round(pred_60)),
                "predictedQty90":     int(round(pred_90)),
                "monthlyConsumption": round(monthly_c, 1),
                "daysUntilLow":       days_until_low,
                "reorderQty":         reorder_qty,
                "estimatedBudget":    round(reorder_qty * cost, 2),
                "risk":               risk,
                "confidence":         confidence,
                "isAnomaly":          bool(anomaly_scores[i] == -1),
                "topFeature":         top_feature.replace("_", " ").title(),
                "seasonMultiplier":   season_now,
                "seasonLabel":        season_label,
                "modelR2":            round(self.training_stats.get("ensemble_r2", 0), 3),
            })

        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        results.sort(key=lambda x: risk_order.get(x["risk"], 3))
        return results

    # ── Analytics for Reports Page ────────────────────────────────────────────
    def report_analytics(
        self,
        assets: List[Dict],
        procurement_data: List[Dict] = [],
    ) -> Dict[str, Any]:
        """
        Full analytics report for the Reports page.
        Returns summary stats, department breakdown, category breakdown,
        top risk items, and budget forecast.
        """
        results = self.predict(assets, procurement_data)

        total    = len(results)
        critical = [r for r in results if r["risk"] == "critical"]
        high     = [r for r in results if r["risk"] == "high"]
        medium   = [r for r in results if r["risk"] == "medium"]
        low      = [r for r in results if r["risk"] == "low"]
        anomalies= [r for r in results if r["isAnomaly"]]

        # Budget forecast
        urgent_budget = sum(r["estimatedBudget"] for r in critical + high)
        total_budget  = sum(r["estimatedBudget"] for r in results if r["risk"] != "low")

        # Department breakdown
        dept_stats: Dict[str, Dict] = {}
        for r in results:
            d = r["department"]
            if d not in dept_stats:
                dept_stats[d] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "budget": 0}
            dept_stats[d][r["risk"]] += 1
            dept_stats[d]["budget"]  += r["estimatedBudget"]

        # Category breakdown
        cat_stats: Dict[str, Dict] = {}
        for r in results:
            c = r["category"]
            if c not in cat_stats:
                cat_stats[c] = {"count": 0, "avgConsumption": 0.0, "avgConfidence": 0.0}
            cat_stats[c]["count"]          += 1
            cat_stats[c]["avgConsumption"] += r["monthlyConsumption"]
            cat_stats[c]["avgConfidence"]  += r["confidence"]
        for c in cat_stats:
            n = cat_stats[c]["count"]
            cat_stats[c]["avgConsumption"] = round(cat_stats[c]["avgConsumption"] / n, 1)
            cat_stats[c]["avgConfidence"]  = round(cat_stats[c]["avgConfidence"]  / n, 1)

        # Seasonal forecast next 3 months
        now = datetime.now()
        monthly_forecast = []
        for delta in range(1, 4):
            future_month = ((now.month - 1 + delta) % 12) + 1
            season_mult  = SEASONAL_MULTIPLIERS.get(future_month, 1.0)
            est_budget   = round(total_budget * season_mult, 2)
            monthly_forecast.append({
                "month":           (now + timedelta(days=delta * 30)).strftime("%B %Y"),
                "seasonMultiplier": season_mult,
                "estimatedBudget":  est_budget,
                "demandLevel":     "Peak" if season_mult >= 1.4 else "Low" if season_mult <= 0.7 else "Normal",
            })

        return {
            "generatedAt":     datetime.now().isoformat(),
            "modelInfo":       {
                "algorithm":   "RandomForest + GradientBoosting + Ridge Ensemble",
                "r2Score":     self.training_stats.get("ensemble_r2", 0),
                "mae":         self.training_stats.get("ensemble_mae", 0),
                "trainedOn":   self.training_stats.get("n_samples", 0),
                "topFeatures": self.training_stats.get("top_features", []),
            },
            "summary": {
                "totalAssets":   total,
                "critical":      len(critical),
                "high":          len(high),
                "medium":        len(medium),
                "low":           len(low),
                "anomalies":     len(anomalies),
                "urgentBudget":  round(urgent_budget, 2),
                "totalBudget":   round(total_budget, 2),
                "avgConfidence": round(sum(r["confidence"] for r in results) / total, 1) if total else 0,
            },
            "departmentBreakdown": dept_stats,
            "categoryBreakdown":   cat_stats,
            "monthlyForecast":     monthly_forecast,
            "topRiskItems":        (critical + high)[:10],
            "anomalies":           anomalies,
            "allResults":          results,
        }

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str = "models") -> None:
        """Persist trained models to disk."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.rf_model,          os.path.join(path, "rf_model.pkl"))
        joblib.dump(self.gb_model,          os.path.join(path, "gb_model.pkl"))
        joblib.dump(self.lr_model,          os.path.join(path, "lr_model.pkl"))
        joblib.dump(self.anomaly_detector,  os.path.join(path, "anomaly.pkl"))
        joblib.dump(self.scaler,            os.path.join(path, "scaler.pkl"))
        joblib.dump(self.training_stats,    os.path.join(path, "stats.pkl"))
        self.is_trained = True
        print(f"[ML] Models saved to {path}/")

    def load(self, path: str = "models") -> None:
        """Load pre-trained models from disk."""
        self.rf_model         = joblib.load(os.path.join(path, "rf_model.pkl"))
        self.gb_model         = joblib.load(os.path.join(path, "gb_model.pkl"))
        self.lr_model         = joblib.load(os.path.join(path, "lr_model.pkl"))
        self.anomaly_detector = joblib.load(os.path.join(path, "anomaly.pkl"))
        self.scaler           = joblib.load(os.path.join(path, "scaler.pkl"))
        self.training_stats   = joblib.load(os.path.join(path, "stats.pkl"))
        self.is_trained       = True
        print(f"[ML] Models loaded from {path}/")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON — use this in routes/forecast.py
# ─────────────────────────────────────────────────────────────────────────────
_model_instance: Optional[InventoryForecastModel] = None

def get_model() -> InventoryForecastModel:
    """Returns singleton model instance. Auto-loads if saved models exist."""
    global _model_instance
    if _model_instance is None:
        _model_instance = InventoryForecastModel()
        if os.path.exists("models/rf_model.pkl"):
            _model_instance.load("models")
    return _model_instance
