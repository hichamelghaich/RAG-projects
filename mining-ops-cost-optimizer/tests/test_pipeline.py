"""Tests de fumée (smoke tests) pour le modèle de coûts, la maintenance
prédictive et le module d'optimisation.

Lancer avec : pytest -q
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.cost_model import load_daily_kpi, cost_breakdown, summary_kpis, month_over_month_delta
from src.predictive_maintenance import load_equipment_daily, train_and_evaluate, current_risk_scores
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


def test_predictive_maintenance_trains_and_scores():
    eq = load_equipment_daily()
    result = train_and_evaluate(eq)
    assert result["n_train"] > 0 and result["n_test"] > 0
    assert result["roc_auc"] is None or 0.0 <= result["roc_auc"] <= 1.0

    risk = current_risk_scores(result["pipeline"], eq)
    assert set(risk["risque_panne_7j"].between(0, 1))
    # une ligne par équipement de la flotte
    assert risk["equipment_id"].nunique() == len(risk)


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
