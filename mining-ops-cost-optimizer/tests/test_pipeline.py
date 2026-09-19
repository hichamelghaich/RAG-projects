"""Tests de fumée (smoke tests) pour le modèle de coûts, la simulation
Monte Carlo et le module d'optimisation.

Lancer avec : pytest -q
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.cost_model import load_daily_kpi, load_equipment_daily, cost_breakdown, summary_kpis, month_over_month_delta
from src.simulation import run_monte_carlo
from src.optimization import run_scenario


def test_data_files_exist():
    assert os.path.exists(config.DAILY_KPI_CSV), (
        "Lancez `python data/generate_synthetic_data.py` avant les tests."
    )
    assert os.path.exists(config.FLEET_CSV)
    assert os.path.exists(config.EQUIPMENT_DAILY_CSV)


def test_cost_model_breakdown_and_summary():
    kpi = load_daily_kpi()
    assert len(kpi) > 0
    breakdown = cost_breakdown(kpi)
    assert len(breakdown) == len(config.COST_COLUMNS)
    assert (breakdown > 0).all()

    summary = summary_kpis(kpi)
    assert summary["tonnes_produites_moy_j"] > 0
    assert 0 <= summary["disponibilite_flotte_pct_moy"] <= 100

    delta = month_over_month_delta(kpi)
    assert "delta_pct" in delta


def test_equipment_daily_loads():
    eq = load_equipment_daily()
    assert len(eq) > 0
    assert set(["equipment_id", "etape", "heures_operees", "disponible"]).issubset(eq.columns)
    assert eq["disponible"].isin([0, 1]).all()


def test_monte_carlo_simulation_is_consistent():
    kpi = load_daily_kpi()
    budget = kpi["cout_total_usd_t"].mean() * 1.02
    target = kpi["tonnes_produites"].mean() * 30 * 0.95

    sim = run_monte_carlo(horizon_days=30, n_simulations=2000, cost_budget_usd_t=budget, tonnage_target=target)
    assert sim.cout_p10_usd_t <= sim.cout_p50_usd_t <= sim.cout_p90_usd_t
    assert sim.tonnage_p10 <= sim.tonnage_p50 <= sim.tonnage_p90
    assert 0.0 <= sim.proba_depassement_budget <= 1.0
    assert 0.0 <= sim.proba_sous_objectif <= 1.0
    assert len(sim.samples_cout) == 2000


def test_monte_carlo_wider_horizon_narrows_relative_cost_spread():
    short = run_monte_carlo(horizon_days=7, n_simulations=2000)
    long = run_monte_carlo(horizon_days=60, n_simulations=2000)
    short_spread = short.cout_p90_usd_t - short.cout_p10_usd_t
    long_spread = long.cout_p90_usd_t - long.cout_p10_usd_t
    # moyenner sur un horizon plus long réduit la dispersion du coût moyen simulé
    assert long_spread < short_spread


def test_optimization_scenario_is_feasible_and_consistent():
    scenario = run_scenario(target_multiplier=1.10)
    assert scenario.target_tonnage > 0
    assert scenario.total_cost_usd > 0
    assert len(scenario.stage_allocation) == len(config.OPT_STAGES)
    # les heures allouées à la flotte propre ne dépassent jamais la capacité
    over_capacity = scenario.stage_allocation["heures_flotte_propre"] > \
        scenario.stage_allocation["capacite_propre_heures"] + 1e-6
    assert not over_capacity.any()


def test_optimization_higher_target_costs_more():
    low = run_scenario(target_multiplier=1.0)
    high = run_scenario(target_multiplier=1.25)
    assert high.total_cost_usd > low.total_cost_usd
    assert high.target_tonnage > low.target_tonnage
