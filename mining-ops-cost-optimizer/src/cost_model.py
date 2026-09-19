"""Modèle de coûts opérationnels : agrégation, décomposition par étape
(waterfall) et analyse de tendance sur la chaîne forage -> sautage ->
chargement -> transport -> énergie."""

from __future__ import annotations

import pandas as pd

from src import config


def load_daily_kpi() -> pd.DataFrame:
    df = pd.read_csv(config.DAILY_KPI_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_equipment_daily() -> pd.DataFrame:
    df = pd.read_csv(config.EQUIPMENT_DAILY_CSV, parse_dates=["date"])
    return df.sort_values(["equipment_id", "date"]).reset_index(drop=True)


def cost_breakdown(df: pd.DataFrame = None, period: str = "all") -> pd.Series:
    """Retourne le coût moyen par tonne pour chaque étape sur la période
    demandée ('all', 'last_30d', 'last_90d')."""
    df = df if df is not None else load_daily_kpi()

    if period == "last_30d":
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    elif period == "last_90d":
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]

    cols = list(config.COST_COLUMNS.values())
    means = df[cols].mean()
    means.index = [config.STAGE_LABELS[s] for s in config.COST_COLUMNS]
    return means.sort_values(ascending=False)


def cost_trend(df: pd.DataFrame = None, freq: str = "W") -> pd.DataFrame:
    """Agrège le coût total et par étape sur une fréquence donnée (par
    défaut hebdomadaire) pour visualiser la tendance."""
    df = df if df is not None else load_daily_kpi()
    cols = list(config.COST_COLUMNS.values()) + ["cout_total_usd_t", "tonnes_produites",
                                                    "disponibilite_flotte_pct", "incidents_hse"]
    agg = df.set_index("date")[cols].resample(freq).mean()
    return agg


def month_over_month_delta(df: pd.DataFrame = None) -> dict:
    """Variation du coût total/tonne entre les 30 derniers jours et les
    30 jours précédents — sert d'alerte simple dans le dashboard."""
    df = df if df is not None else load_daily_kpi()
    last_date = df["date"].max()
    recent = df[df["date"] > last_date - pd.Timedelta(days=30)]["cout_total_usd_t"].mean()
    previous = df[
        (df["date"] <= last_date - pd.Timedelta(days=30)) &
        (df["date"] > last_date - pd.Timedelta(days=60))
    ]["cout_total_usd_t"].mean()
    delta_pct = 100 * (recent - previous) / previous if previous else 0.0
    return {
        "cout_recent_usd_t": round(recent, 2),
        "cout_precedent_usd_t": round(previous, 2),
        "delta_pct": round(delta_pct, 1),
    }


def summary_kpis(df: pd.DataFrame = None) -> dict:
    df = df if df is not None else load_daily_kpi()
    last_30 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    return {
        "tonnes_produites_moy_j": round(last_30["tonnes_produites"].mean()),
        "cout_total_usd_t_moy": round(last_30["cout_total_usd_t"].mean(), 2),
        "disponibilite_flotte_pct_moy": round(last_30["disponibilite_flotte_pct"].mean(), 1),
        "incidents_hse_total_30j": int(last_30["incidents_hse"].sum()),
    }
