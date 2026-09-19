"""Tests de fumée (smoke tests) pour le pipeline RAG, le forecast et l'agent.

Lancer avec : pytest -q
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import RagPipeline
from src.forecast import ForecastEngine, extract_metric, extract_horizon
from src.agent import LogisticsAgent
from src import config


def test_data_files_exist():
    assert os.path.exists(config.KPI_CSV_PATH), (
        "Lancez `python data/generate_synthetic_data.py` avant les tests."
    )
    assert os.path.isdir(config.REPORTS_DIR)
    assert len(os.listdir(config.REPORTS_DIR)) > 0


def test_rag_pipeline_builds_and_answers():
    rag = RagPipeline()
    rag.build_index()
    assert len(rag.store) > 0

    result = rag.answer("Que dit la procédure de gestion des incidents HSE ?")
    assert result["answer"]
    assert len(result["sources"]) > 0


def test_forecast_returns_expected_horizon():
    fe = ForecastEngine()
    result = fe.forecast("cout_par_tonne_usd", horizon_months=3)
    assert len(result.forecast_values) == 3
    assert len(result.forecast_dates) == 3
    assert result.method in ("holt_winters", "linear_regression_fallback")


def test_intent_extraction():
    assert extract_metric("Quel est le coût par tonne prévu ?") == "cout_par_tonne_usd"
    assert extract_horizon("dans les 5 prochains mois") == 5
    assert extract_horizon("sans mention d'horizon") == config.DEFAULT_FORECAST_HORIZON


def test_agent_routes_forecast_question():
    agent = LogisticsAgent()
    response = agent.ask("Quelle sera l'évolution du coût par tonne dans les 3 prochains mois ?")
    assert "forecast_kpi" in response.tools_used
    assert response.answer


def test_agent_routes_rag_question():
    agent = LogisticsAgent()
    response = agent.ask("Pourquoi le taux de disponibilité de la flotte a-t-il baissé en novembre 2023 ?")
    assert "search_reports" in response.tools_used
    assert response.answer
    assert len(response.sources) > 0


def test_agent_routes_hybrid_question():
    agent = LogisticsAgent()
    response = agent.ask(
        "Quels incidents HSE ont eu lieu récemment et comment va évoluer le "
        "délai de livraison dans les 2 prochains mois ?"
    )
    assert set(response.tools_used) == {"forecast_kpi", "search_reports"}
