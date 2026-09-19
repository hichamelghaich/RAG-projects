"""Backends d'embedding, interchangeables derrière une interface commune.

Par défaut, le projet utilise un embedding TF-IDF (scikit-learn) : aucune
dépendance lourde, aucune clé API, aucun téléchargement de modèle — il
tourne partout et suffit largement pour un corpus de démonstration.

Deux backends optionnels sont fournis pour montrer comment le pipeline
s'étend vers des embeddings denses "production" :
  - SentenceTransformerEmbeddings (modèle local, nécessite `sentence-transformers`)
  - OpenAIEmbeddings (API, nécessite `OPENAI_API_KEY`)

L'architecture (interface `BaseEmbeddings`) permet de changer de backend
sans toucher au reste du pipeline RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddings(ABC):
    """Interface commune : fit sur un corpus, puis encode des textes en vecteurs denses."""

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Retourne un array (n_texts, dim) en float32."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class TfidfEmbeddings(BaseEmbeddings):
    """Embeddings TF-IDF + réduction de dimension (SVD) pour obtenir des
    vecteurs denses compatibles avec un index FAISS classique.

    C'est le backend par défaut du projet : rapide, déterministe,
    sans dépendance externe autre que scikit-learn.
    """

    def __init__(self, n_components: int = 128, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_df=0.95, min_df=1, ngram_range=(1, 2), sublinear_tf=True
        )
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        n_components = min(self.n_components, max(2, len(corpus) - 1))
        if n_components != self.svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf = self.vectorizer.fit_transform(corpus)
        self.svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings.fit() doit être appelé avant encode().")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf).astype("float32")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    @property
    def dim(self) -> int:
        return self.svd.n_components


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Backend optionnel utilisant un modèle local sentence-transformers.

    Nécessite `pip install sentence-transformers`. Non requis par défaut,
    mais montre comment brancher un vrai modèle d'embedding dense si un
    environnement plus riche (GPU/CPU avec plus de ressources) est disponible.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def fit(self, corpus: List[str]) -> None:
        pass  # modèle pré-entraîné, pas de fit nécessaire

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbeddings(BaseEmbeddings):
    """Backend optionnel utilisant l'API d'embeddings OpenAI.

    Nécessite `OPENAI_API_KEY` dans l'environnement. Non requis par défaut.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self._dim = 1536

    def fit(self, corpus: List[str]) -> None:
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim


def get_default_embeddings() -> BaseEmbeddings:
    return TfidfEmbeddings()
