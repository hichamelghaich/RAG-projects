"""Simulation (Monte Carlo par ré-échantillonnage historique) du coût et de
la production sur un horizon futur, pour quantifier le risque de dépasser
un budget ou de manquer un objectif de tonnage.

Méthode : bootstrap — on tire au hasard, avec remise, des journées
observées dans l'historique (`daily_kpi.csv`) pour composer des milliers de
scénarios plausibles d'un mois à venir, plutôt que de supposer une loi de
probabilité théorique. C'est une approche simple et robuste, standard en
simulation appliquée, qui ne nécessite aucune hypothèse forte sur la forme
de la distribution des coûts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.cost_model import load_daily_kpi


@dataclass
class SimulationResult:
    horizon_days: int
    n_simulations: int
    cout_p10_usd_t: float
    cout_p50_usd_t: float
    cout_p90_usd_t: float
    tonnage_p10: float
    tonnage_p50: float
    tonnage_p90: float
    proba_depassement_budget: Optional[float]
    proba_sous_objectif: Optional[float]
    samples_cout: np.ndarray
    samples_tonnage: np.ndarray


def run_monte_carlo(
    horizon_days: int = 30,
    n_simulations: int = 5000,
    cost_budget_usd_t: Optional[float] = None,
    tonnage_target: Optional[float] = None,
    period_days: int = 180,
    random_state: int = 42,
) -> SimulationResult:
    """Simule `n_simulations` scénarios d'un horizon de `horizon_days` jours,
    en ré-échantillonnant (avec remise) des journées historiques des
    `period_days` derniers jours. Retourne la distribution du coût moyen
    (USD/tonne) et du tonnage total sur l'horizon, ainsi que, si des seuils
    sont fournis, la probabilité de les dépasser / ne pas les atteindre."""
    rng = np.random.default_rng(random_state)

    kpi = load_daily_kpi()
    recent = kpi[kpi["date"] >= kpi["date"].max() - pd.Timedelta(days=period_days)]

    cost_values = recent["cout_total_usd_t"].to_numpy()
    tonnage_values = recent["tonnes_produites"].to_numpy()
    n_hist = len(cost_values)

    # tirage (n_simulations, horizon_days) d'indices de journées historiques
    draws = rng.integers(0, n_hist, size=(n_simulations, horizon_days))

    sim_cost = cost_values[draws].mean(axis=1)          # coût moyen USD/t sur l'horizon
    sim_tonnage = tonnage_values[draws].sum(axis=1)      # tonnage total sur l'horizon

    cout_p10, cout_p50, cout_p90 = np.percentile(sim_cost, [10, 50, 90])
    ton_p10, ton_p50, ton_p90 = np.percentile(sim_tonnage, [10, 50, 90])

    proba_budget = float(np.mean(sim_cost > cost_budget_usd_t)) if cost_budget_usd_t else None
    proba_shortfall = float(np.mean(sim_tonnage < tonnage_target)) if tonnage_target else None

    return SimulationResult(
        horizon_days=horizon_days,
        n_simulations=n_simulations,
        cout_p10_usd_t=round(float(cout_p10), 2),
        cout_p50_usd_t=round(float(cout_p50), 2),
        cout_p90_usd_t=round(float(cout_p90), 2),
        tonnage_p10=round(float(ton_p10)),
        tonnage_p50=round(float(ton_p50)),
        tonnage_p90=round(float(ton_p90)),
        proba_depassement_budget=round(proba_budget, 3) if proba_budget is not None else None,
        proba_sous_objectif=round(proba_shortfall, 3) if proba_shortfall is not None else None,
        samples_cout=sim_cost,
        samples_tonnage=sim_tonnage,
    )
