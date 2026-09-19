"""Wrapper simple autour de FAISS pour l'indexation et la recherche de similarité."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import faiss

from src.chunking import Chunk


class FaissVectorStore:
    """Index FAISS en mémoire (IndexFlatIP -> similarité cosinus sur vecteurs
    normalisés). Suffisant pour un corpus de démonstration (quelques
    centaines de chunks) ; pour un corpus plus large, un index approximatif
    (IVF/HNSW) serait préférable — l'interface resterait identique."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
