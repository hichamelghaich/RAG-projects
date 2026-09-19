"""Module de forecast sur les KPI numériques (série temporelle mensuelle).

Stratégie de modélisation :
  1. Lissage exponentiel de Holt-Winters (statsmodels) si assez d'historique
     et de variance — capture tendance + saisonnalité.
  2. Repli automatique sur une régression linéaire simple (scikit-learn) si
     Holt-Winters échoue à converger (série trop courte, trop plate, etc.).

Le module expose aussi un petit parseur d'intention pour extraire, à partir
d'une question en langage naturel, la métrique visée et l'horizon de
prévision demandé (utilisé par l'agent pour déclencher cet outil).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src import config


@dataclass
class ForecastResult:
    metric: str
    metric_label: str
    horizon_months: int
    history_dates: List[str]
    history_values: List[float]
    forecast_dates: List[str]
    forecast_values: List[float]
    method: str
    explanation: str = ""


class ForecastEngine:
    def __init__(self, csv_path: str = config.KPI_CSV_PATH):
        self.df = pd.read_csv(csv_path, parse_dates=["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)

    def available_metrics(self) -> List[str]:
        return list(config.FORECAST_METRICS.keys())

    def forecast(self, metric: str, horizon_months: int = config.DEFAULT_FORECAST_HORIZON) -> ForecastResult:
        if metric not in self.df.columns:
            raise ValueError(f"Métrique inconnue : {metric}")

        series = self.df.set_index("date")[metric].astype(float)
        method, values = self._fit_and_predict(series, horizon_months)

        last_date = series.index[-1]
        future_dates = pd.date_range(
            last_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS"
        )

        explanation = self._explain(metric, series, values)

        return ForecastResult(
            metric=metric,
            metric_label=config.FORECAST_METRICS.get(metric, metric),
            horizon_months=horizon_months,
            history_dates=[d.strftime("%Y-%m") for d in series.index],
            history_values=[round(float(v), 3) for v in series.values],
            forecast_dates=[d.strftime("%Y-%m") for d in future_dates],
            forecast_values=[round(float(v), 3) for v in values],
            method=method,
            explanation=explanation,
        )

    def _fit_and_predict(self, series: pd.Series, horizon: int):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            seasonal_periods = 12 if len(series) >= 24 else None
            model = ExponentialSmoothing(
                series.values,
                trend="add",
                seasonal="add" if seasonal_periods else None,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            preds = fitted.forecast(horizon)
            if np.any(np.isnan(preds)):
                raise ValueError("Holt-Winters a produit des NaN.")
            return "holt_winters", list(preds)
        except Exception:
            return self._linear_fallback(series, horizon)

    def _linear_fallback(self, series: pd.Series, horizon: int):
        from sklearn.linear_model import LinearRegression

        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression().fit(x, y)
        future_x = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        preds = model.predict(future_x)
        return "linear_regression_fallback", list(preds)

    def _explain(self, metric: str, series: pd.Series, forecast_values: List[float]) -> str:
        recent = series.values[-6:]
        trend = "en hausse" if recent[-1] > recent[0] else "en baisse" if recent[-1] < recent[0] else "stable"
        label = config.FORECAST_METRICS.get(metric, metric)
        last_val = series.values[-1]
        next_val = forecast_values[0]
        direction = "hausse" if next_val > last_val else "baisse" if next_val < last_val else "stabilité"
        return (
            f"Sur les 6 derniers mois, {label} est globalement {trend}. "
            f"Le modèle projette une {direction} pour le mois suivant "
            f"({last_val:.2f} → {next_val:.2f})."
        )


# --- Parseur d'intention (utilisé par l'agent pour extraire métrique + horizon) ---

_METRIC_KEYWORDS = {
    "cout_par_tonne_usd": ["coût", "cout", "prix par tonne", "usd/tonne", "cost"],
    "volume_transporte_tonnes": ["volume", "tonnage", "tonnes transportées"],
    "delai_livraison_jours": ["délai", "delai", "retard", "livraison"],
    "taux_disponibilite_flotte_pct": ["disponibilité", "disponibilite", "flotte"],
    "incidents_hse": ["incident", "hse", "sécurité", "securite"],
    "stock_intermediaire_tonnes": ["stock", "tampon", "stockage"],
}

_HORIZON_PATTERN = re.compile(
    r"(\d+)\D{0,20}?(mois|month|months)", re.IGNORECASE
)


def extract_metric(question: str) -> Optional[str]:
    q = question.lower()
    for metric, keywords in _METRIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return metric
    return None


def extract_horizon(question: str, default: int = config.DEFAULT_FORECAST_HORIZON) -> int:
    match = _HORIZON_PATTERN.search(question)
    if match:
        try:
            n = int(match.group(1))
            return max(1, min(n, 12))
        except ValueError:
            pass
    return default
