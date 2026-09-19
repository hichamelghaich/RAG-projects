"""Backends de génération de réponse, interchangeables.

- ExtractiveBackend (par défaut) : ne nécessite aucune clé API. Compose une
  réponse structurée à partir des passages récupérés (extraction +
  synthèse par règles). C'est volontairement le comportement par défaut du
  projet pour qu'il tourne "out of the box".
- OpenAIBackend (optionnel) : si `OPENAI_API_KEY` est définie, génère une
  réponse en langage naturel à partir du contexte récupéré, via l'API
  Chat Completions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.chunking import Chunk


class BaseLLMBackend(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved: List[Chunk]) -> str:
        ...


class ExtractiveBackend(BaseLLMBackend):
    """Synthèse sans LLM : agrège les passages les plus pertinents et les
    présente de façon structurée avec leurs sources. Ce backend garantit
    que le projet est démontrable sans aucune dépendance à une API payante."""

    def generate(self, question: str, retrieved: List[Chunk]) -> str:
        if not retrieved:
            return (
                "Aucun document pertinent n'a été trouvé dans la base "
                "documentaire pour répondre à cette question."
            )

        lines = [
            "Réponse synthétisée à partir des documents les plus pertinents "
            "trouvés dans la base (mode extractif, sans appel LLM) :\n"
        ]
        for i, chunk in enumerate(retrieved, start=1):
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 320:
                snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
            lines.append(f"**[Source {i} — {chunk.source_path}]**\n{snippet}\n")

        return "\n".join(lines)


class OpenAIBackend(BaseLLMBackend):
    """Backend optionnel basé sur l'API OpenAI (Chat Completions).
    N'est utilisé que si RAG_LLM_BACKEND=openai et OPENAI_API_KEY est défini."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, question: str, retrieved: List[Chunk]) -> str:
        context = "\n\n".join(
            f"[Source: {c.source_path}]\n{c.text}" for c in retrieved
        )
        system_prompt = (
            "Tu es un assistant analytique pour une équipe logistique. "
            "Réponds uniquement à partir du contexte fourni, en français, "
            "de façon concise et factuelle. Cite les sources entre crochets. "
            "Si le contexte ne permet pas de répondre, dis-le clairement."
        )
        user_prompt = f"Contexte :\n{context}\n\nQuestion : {question}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content


def get_backend(name: str = "extractive") -> BaseLLMBackend:
    if name == "openai":
        try:
            return OpenAIBackend()
        except Exception as e:  # pas de clé API, package manquant, etc.
            print(f"[warn] Backend OpenAI indisponible ({e}), retour au mode extractif.")
            return ExtractiveBackend()
    return ExtractiveBackend()
