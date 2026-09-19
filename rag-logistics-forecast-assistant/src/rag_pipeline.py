"""Pipeline RAG de bout en bout : ingestion -> chunking -> embeddings ->
index FAISS -> retrieval -> génération de réponse."""

from __future__ import annotations

import os
from typing import List

from src import config
from src.chunking import Chunk, chunk_document
from src.embeddings import BaseEmbeddings, get_default_embeddings
from src.vector_store import FaissVectorStore
from src.llm_backend import BaseLLMBackend, get_backend


class RagPipeline:
    def __init__(self, embeddings: BaseEmbeddings = None, llm_backend: BaseLLMBackend = None,
                 chunk_size: int = config.CHUNK_SIZE_CHARS,
                 chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
                 top_k: int = config.TOP_K):
        self.embeddings = embeddings or get_default_embeddings()
        self.llm_backend = llm_backend or get_backend(config.LLM_BACKEND)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.store: FaissVectorStore | None = None

    def _load_corpus_files(self) -> List[tuple]:
        """Retourne une liste de (doc_id, path, texte) pour tous les
        rapports et documents de référence du corpus."""
        files = []
        for folder in (config.REPORTS_DIR, config.REFERENCE_DIR):
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith((".md", ".txt")):
                    continue
                path = os.path.join(folder, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                rel_path = os.path.relpath(path, config.DATA_DIR)
                files.append((fname, rel_path, text))
        return files

    def build_index(self) -> None:
        """Ingère le corpus complet, le découpe en chunks, calcule les
        embeddings et construit l'index vectoriel FAISS."""
        files = self._load_corpus_files()
        if not files:
            raise RuntimeError(
                "Aucun document trouvé. Lancez d'abord "
                "`python data/generate_synthetic_data.py`."
            )

        all_chunks: List[Chunk] = []
        for doc_id, path, text in files:
            all_chunks.extend(
                chunk_document(doc_id, path, text, self.chunk_size, self.chunk_overlap)
            )

        corpus_texts = [c.text for c in all_chunks]
        self.embeddings.fit(corpus_texts)
        vectors = self.embeddings.encode(corpus_texts)

        self.store = FaissVectorStore(dim=self.embeddings.dim)
        self.store.add(all_chunks, vectors)

        print(f"[rag] Index construit : {len(all_chunks)} chunks issus de "
              f"{len(files)} documents (dim={self.embeddings.dim}).")

    def retrieve(self, question: str, top_k: int = None) -> List[Chunk]:
        if self.store is None:
            raise RuntimeError("Index non construit. Appelez build_index() d'abord.")
        top_k = top_k or self.top_k
        query_vec = self.embeddings.encode([question])[0]
        results = self.store.search(query_vec, top_k)
        return [chunk for chunk, score in results]

    def retrieve_with_scores(self, question: str, top_k: int = None):
        if self.store is None:
            raise RuntimeError("Index non construit. Appelez build_index() d'abord.")
        top_k = top_k or self.top_k
        query_vec = self.embeddings.encode([question])[0]
        return self.store.search(query_vec, top_k)

    def answer(self, question: str, top_k: int = None) -> dict:
        retrieved = self.retrieve(question, top_k)
        answer_text = self.llm_backend.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer_text,
            "sources": [c.source_path for c in retrieved],
            "chunks": retrieved,
        }
