"""Découpage (chunking) de documents texte en segments pour l'indexation RAG."""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    doc_id: str
    source_path: str
    chunk_id: str
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Découpe un texte en fenêtres de `chunk_size` caractères avec
    chevauchement `overlap`, en coupant de préférence sur des frontières
    de paragraphe/phrase pour rester lisible."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # essaie de couper sur un saut de ligne ou une fin de phrase proche
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "))
            if cut > chunk_size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc_id: str, source_path: str, text: str,
                    chunk_size: int, overlap: int) -> List[Chunk]:
    pieces = chunk_text(text, chunk_size, overlap)
    return [
        Chunk(doc_id=doc_id, source_path=source_path,
              chunk_id=f"{doc_id}::chunk{i}", text=piece)
        for i, piece in enumerate(pieces)
    ]
