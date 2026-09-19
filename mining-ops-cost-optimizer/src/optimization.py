"""Optimisation (recherche opérationnelle) : détermine, pour un objectif de
tonnage donné sur un horizon court (ex. la semaine suivante), la répartition
d'heures d'exploitation la moins coûteuse entre flotte propre et
sous-traitance, étape par étape de la chaîne forage -> sautage ->
chargement -> transport -> énergie.

Modélisé comme un programme linéaire (scipy.optimize.linprog) :
  - variables : heures "flotte propre" et heures "sous-traitance" par étape
  - objectif  : minimiser le coût total (USD)
  - contraintes : throughput x heures >= tonnage cible (par étape) ;
                   heures flotte propre <= capacité disponible (historique)

Le scénario "sans sous-traitance" (flotte propre uniquement, quitte à ne
pas atteindre la cible) sert de comparaison pour chiffrer la valeur de
l'arbitrage flotte propre / sous-traitance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src import config
from src.cost_model import load_daily_kpi, load_equipment_daily


@dataclass
class StageParams:
    stage: str
    cout_propre_usd_h: float
    cout_soustraitance_usd_h: float
    capacite_propre_heures: float
    throughput_t_h: float


@dataclass
class OptimizationResult:
    target_tonnage: float
    horizon_days: int
    stage_allocation: pd.DataFrame
    total_cost_usd: float
    baseline_cost_usd: float
    baseline_shortfall_tonnes: float
    cout_additionnel_pour_cible_usd: float
    cout_marginal_usd_par_tonne: float | None


def _stage_parameters(period_days: int = 30, horizon_days: int = 7) -> Dict[str, StageParams]:
    kpi = load_daily_kpi()
    recent_kpi = kpi[kpi["date"] >= kpi["date"].max() - pd.Timedelta(days=period_days)]

    eq = load_equipment_daily()
    recent_eq = eq[eq["date"] >= eq["date"].max() - pd.Timedelta(days=period_days)]

    n_equipment = recent_eq.groupby("etape")["equipment_id"].nunique()
    avg_hours_per_eq_day = recent_eq.groupby("etape")["heures_operees"].mean()

    params = {}
    for stage in config.OPT_STAGES:
        cost_col = config.COST_COLUMNS[stage]
        avg_cost_per_ton = recent_kpi[cost_col].mean()
        throughput = config.STAGE_THROUGHPUT_T_PER_H[stage]

        cost_own_per_hour = avg_cost_per_ton * throughput
        cost_sub_per_hour = cost_own_per_hour * 1.4  # prime de sous-traitance (+40%)

        capacity_hours = float(n_equipment.get(stage, 0) * horizon_days * avg_hours_per_eq_day.get(stage, 0))

        params[stage] = StageParams(
            stage=stage,
            cout_propre_usd_h=round(cost_own_per_hour, 2),
            cout_soustraitance_usd_h=round(cost_sub_per_hour, 2),
            capacite_propre_heures=round(capacity_hours, 1),
            throughput_t_h=throughput,
        )
    return params


def run_scenario(target_multiplier: float = 1.10, horizon_days: int = 7,
                  period_days: int = 30) -> OptimizationResult:
    """Optimise l'allocation flotte propre / sous-traitance pour atteindre
    `target_multiplier` x la production moyenne récente, sur `horizon_days`
    jours, au moindre coût."""
    kpi = load_daily_kpi()
    recent_kpi = kpi[kpi["date"] >= kpi["date"].max() - pd.Timedelta(days=period_days)]
    avg_daily_tonnage = recent_kpi["tonnes_produites"].mean()
    target_tonnage = avg_daily_tonnage * target_multiplier * horizon_days

    stage_params = _stage_parameters(period_days=period_days, horizon_days=horizon_days)
    stages = config.OPT_STAGES

    # Le sautage n'a pas de flotte propre/sous-traitance dans ce modèle : son
    # coût (marché, ~coût constaté récemment) s'applique au tonnage cible et
    # s'ajoute identiquement aux deux scénarios (n'affecte donc pas l'arbitrage
    # flotte propre / sous-traitance, seulement le coût total affiché).
    avg_cost_sautage = recent_kpi[config.COST_COLUMNS["sautage"]].mean()
    sautage_cost = avg_cost_sautage * target_tonnage

    # variables : [h_propre_1..5, h_sub_1..5]
    n = len(stages)
    c = [stage_params[s].cout_propre_usd_h for s in stages] + \
        [stage_params[s].cout_soustraitance_usd_h for s in stages]

    # contraintes d'inégalité A_ub x <= b_ub : on formule
    # -(throughput_propre * h_propre + throughput_sub * h_sub) <= -target
    # (linprog ne gère que des contraintes "<=")
    A_ub = []
    b_ub = []
    for i, s in enumerate(stages):
        row = [0.0] * (2 * n)
        row[i] = -stage_params[s].throughput_t_h
        row[n + i] = -stage_params[s].throughput_t_h
        A_ub.append(row)
        b_ub.append(-target_tonnage)

    bounds = [(0, stage_params[s].capacite_propre_heures) for s in stages] + \
             [(0, None) for _ in stages]  # sous-traitance non plafonnée (contrainte réaliste : marché externe)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Optimisation infaisable : {res.message}")

    x = res.x
    rows = []
    for i, s in enumerate(stages):
        h_propre = x[i]
        h_sub = x[n + i]
        rows.append({
            "etape": config.STAGE_LABELS[s],
            "heures_flotte_propre": round(h_propre, 1),
            "heures_sous_traitance": round(h_sub, 1),
            "cout_flotte_propre_usd": round(h_propre * stage_params[s].cout_propre_usd_h),
            "cout_sous_traitance_usd": round(h_sub * stage_params[s].cout_soustraitance_usd_h),
            "capacite_propre_heures": stage_params[s].capacite_propre_heures,
            "utilisation_flotte_propre_pct": round(
                100 * h_propre / stage_params[s].capacite_propre_heures, 1
            ) if stage_params[s].capacite_propre_heures else 0.0,
        })

    allocation_df = pd.DataFrame(rows)
    total_cost = float(res.fun) + sautage_cost

    # scénario de référence : tout en flotte propre, plafonné à la capacité
    # (donc sous-production possible) -> chiffre le manque à gagner si on
    # n'a pas recours à la sous-traitance.
    baseline_cost = 0.0
    baseline_min_tonnage = float("inf")
    for s in stages:
        p = stage_params[s]
        h = p.capacite_propre_heures
        baseline_cost += h * p.cout_propre_usd_h
        baseline_min_tonnage = min(baseline_min_tonnage, h * p.throughput_t_h)
    baseline_cost += sautage_cost
    baseline_shortfall = max(0.0, target_tonnage - baseline_min_tonnage)

    extra_cost = total_cost - baseline_cost
    marginal_cost_per_tonne = (extra_cost / baseline_shortfall) if baseline_shortfall > 0 else None

    return OptimizationResult(
        target_tonnage=round(target_tonnage),
        horizon_days=horizon_days,
        stage_allocation=allocation_df,
        total_cost_usd=round(total_cost),
        baseline_cost_usd=round(baseline_cost),
        baseline_shortfall_tonnes=round(baseline_shortfall),
        cout_additionnel_pour_cible_usd=round(extra_cost),
        cout_marginal_usd_par_tonne=round(marginal_cost_per_tonne, 2) if marginal_cost_per_tonne is not None else None,
    )
