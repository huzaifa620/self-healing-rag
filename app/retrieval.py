"""Retrieval over the frozen index.

Deliberately not a vector database. The index is committed to the repo and
embeddings are L2-normalised at build time, so retrieval is a single matmul and
CI needs no external service, no secrets, and no network for this step. That
makes eval numbers comparable across runs, which is the entire point of the
project.

Ceiling: ~2k chunks. At this size numpy is instant (sub-millisecond) and an ANN
index would be pure overhead. Past ~50k chunks, swap in FAISS or pgvector — the
interface here is two functions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.llm import CostTracker, embed_query
from app.schemas import Chunk

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


@lru_cache(maxsize=1)
def _index() -> tuple[np.ndarray, list[dict]]:
    vecs = np.load(CORPUS / "embeddings.npy")
    with (CORPUS / "chunks.jsonl").open(encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]
    if len(chunks) != vecs.shape[0]:
        raise RuntimeError(
            f"index mismatch: {len(chunks)} chunks vs {vecs.shape[0]} vectors — "
            "re-run corpus/build_corpus.py"
        )
    return vecs, chunks


def search(query: str, cost: CostTracker, k: int = 6) -> list[Chunk]:
    vecs, chunks = _index()
    q = np.asarray(embed_query(query, cost), dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vecs @ q                      # corpus vectors are already normalised
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return [
        Chunk(
            id=chunks[i]["id"],
            text=chunks[i]["text"],
            source=chunks[i]["source"],
            heading=chunks[i]["heading"],
            score=float(scores[i]),
        )
        for i in top
    ]


def format_context(chunks: list[Chunk]) -> str:
    """Render chunks with their ids so the model can cite them by number."""
    return "\n\n".join(f"[{c.id}] ({c.source})\n{c.text}" for c in chunks)
