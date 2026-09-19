"""Agent léger orchestrant deux outils : recherche documentaire (RAG) et
prévision de KPI (forecast) — pattern "function calling" / IA agentique.

Deux modes de routage :
  - "rule_based" (par défaut, sans clé API) : classifieur simple à base de
    règles/mots-clés qui décide quel(s) outil(s) appeler.
  - "openai_function_calling" (optionnel) : si OPENAI_API_KEY est défini,
    laisse le modèle choisir l'outil à appeler via l'API function calling
    d'OpenAI — démontre le pattern "agentique" standard du marché.

Dans les deux cas, l'agent :
  1. décide quel(s) outil(s) utiliser pour répondre à la question,
  2. appelle le(s) outil(s) correspondant(s) (RAG et/ou forecast),
  3. compose une réponse finale unique avec attribution des sources/méthode.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from src.rag_pipeline import RagPipeline
from src.forecast import ForecastEngine, extract_metric, extract_horizon
from src import config

FORECAST_INTENT_KEYWORDS = [
    "prévi", "previ", "forecast", "prédi", "predi", "dans les prochains",
    "dans les prochain", "mois prochain", "mois suivant", "tendance future",
    "va évoluer", "va evoluer", "projection", "prochains mois",
]

RAG_INTENT_KEYWORDS = [
    "pourquoi", "explique", "expliquer", "qu'est-ce qui", "qu est ce qui",
    "rapport", "incident", "que s'est-il passé", "que s est il passe",
    "procédure", "procedure", "politique", "que dit", "résume", "resume",
]


@dataclass
class AgentResponse:
    question: str
    tools_used: List[str]
    routing_method: str
    answer: str
    sources: List[str]
    forecast_data: Optional[dict] = None


class LogisticsAgent:
    """Agent minimal à deux outils : `search_reports` (RAG) et
    `forecast_kpi` (série temporelle). Le routage détermine s'il faut
    utiliser l'un, l'autre, ou les deux (question hybride)."""

    def __init__(self, rag_pipeline: RagPipeline = None, forecast_engine: ForecastEngine = None):
        self.rag = rag_pipeline or RagPipeline()
        self.forecast_engine = forecast_engine or ForecastEngine()
        self._index_built = False
        self.use_openai_routing = bool(os.environ.get("OPENAI_API_KEY")) and \
            config.LLM_BACKEND == "openai"

    def _ensure_index(self):
        if not self._index_built:
            self.rag.build_index()
            self._index_built = True

    # --- Outils exposés à l'agent (function-calling style) ---

    def tool_search_reports(self, query: str) -> dict:
        self._ensure_index()
        result = self.rag.answer(query)
        return result

    def tool_forecast_kpi(self, metric: str, horizon_months: int) -> dict:
        result = self.forecast_engine.forecast(metric, horizon_months)
        return result.__dict__

    # --- Routage ---

    def _route_rule_based(self, question: str) -> List[str]:
        q = question.lower()
        wants_forecast = any(kw in q for kw in FORECAST_INTENT_KEYWORDS) or extract_metric(question) and any(
            kw in q for kw in ["prochain", "prochains", "futur", "va être", "va etre", "mois"]
        )
        wants_rag = any(kw in q for kw in RAG_INTENT_KEYWORDS)

        # Une question qui nomme une métrique numérique ET un horizon temporel
        # est un signal fort de forecast, même sans mot-clé explicite.
        if extract_metric(question) and re.search(r"\d+\s*mois", q):
            wants_forecast = True

        if wants_forecast and wants_rag:
            return ["forecast_kpi", "search_reports"]
        if wants_forecast:
            return ["forecast_kpi"]
        if wants_rag:
            return ["search_reports"]
        # par défaut : si une métrique est détectée, on tente le forecast,
        # sinon on part sur la recherche documentaire (cas le plus général).
        return ["forecast_kpi"] if extract_metric(question) else ["search_reports"]

    def _route_openai_function_calling(self, question: str) -> List[str]:
        """Utilise l'API function-calling d'OpenAI pour choisir le(s) outil(s).
        N'est appelé que si une clé API est disponible ; en cas d'erreur,
        on retombe sur le routage à base de règles."""
        try:
            from openai import OpenAI
            client = OpenAI()

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_reports",
                        "description": "Recherche dans les rapports opérationnels et "
                                        "documents de référence (RAG) pour répondre à "
                                        "des questions qualitatives (pourquoi, quoi, "
                                        "procédures, incidents passés).",
                        "parameters": {"type": "object", "properties": {
                            "query": {"type": "string"}}, "required": ["query"]},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "forecast_kpi",
                        "description": "Prévoit l'évolution future d'un KPI numérique "
                                        "(coût, volume, délai, disponibilité, incidents, "
                                        "stock) sur un horizon en mois.",
                        "parameters": {"type": "object", "properties": {
                            "metric": {"type": "string", "enum": list(config.FORECAST_METRICS.keys())},
                            "horizon_months": {"type": "integer"},
                        }, "required": ["metric", "horizon_months"]},
                    },
                },
            ]

            resp = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": question}],
                tools=tools,
                tool_choice="auto",
            )
            calls = resp.choices[0].message.tool_calls or []
            names = [c.function.name for c in calls]
            return names or self._route_rule_based(question)
        except Exception as e:
            print(f"[warn] Routage OpenAI indisponible ({e}), repli sur le routage par règles.")
            return self._route_rule_based(question)

    # --- Point d'entrée principal ---

    def ask(self, question: str) -> AgentResponse:
        if self.use_openai_routing:
            tools_to_use = self._route_openai_function_calling(question)
            routing_method = "openai_function_calling"
        else:
            tools_to_use = self._route_rule_based(question)
            routing_method = "rule_based"

        answer_parts = []
        sources: List[str] = []
        forecast_data = None

        if "forecast_kpi" in tools_to_use:
            metric = extract_metric(question) or "cout_par_tonne_usd"
            horizon = extract_horizon(question)
            fresult = self.forecast_engine.forecast(metric, horizon)
            forecast_data = fresult.__dict__
            answer_parts.append(
                f"**Prévision — {fresult.metric_label}** "
                f"(méthode : {fresult.method}, horizon : {horizon} mois)\n"
                f"{fresult.explanation}\n"
                f"Valeurs prévues : "
                + ", ".join(f"{d} → {v}" for d, v in zip(fresult.forecast_dates, fresult.forecast_values))
            )

        if "search_reports" in tools_to_use:
            self._ensure_index()
            rag_result = self.rag.answer(question)
            answer_parts.append(rag_result["answer"])
            sources.extend(rag_result["sources"])

        final_answer = "\n\n---\n\n".join(answer_parts) if answer_parts else \
            "Je n'ai pas pu déterminer comment répondre à cette question."

        return AgentResponse(
            question=question,
            tools_used=tools_to_use,
            routing_method=routing_method,
            answer=final_answer,
            sources=sources,
            forecast_data=forecast_data,
        )
