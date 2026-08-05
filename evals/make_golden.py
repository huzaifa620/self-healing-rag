"""Generate the answerable half of the golden set, then verify it automatically.

Committed so the provenance of golden.jsonl is auditable rather than a mystery
blob. Run once; the output is reviewed by hand before it lands.

Verification built in: a generated question is only kept if retrieval actually
surfaces the chunk it was written from. That drops questions that are ambiguous,
that depend on context the chunk doesn't carry, or that collide with a
better-matching part of the corpus — all of which would show up later as fake
eval failures.

The unanswerable half is hand-written in `UNANSWERABLE` below. Those are the
whole point of the eval and are not worth generating.

    python evals/make_golden.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from app.llm import CostTracker, chat_json
from app.retrieval import search

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT.parent / "corpus" / "chunks.jsonl"
OUT = ROOT / "golden.jsonl"

N_ANSWERABLE = 45
SEED = 7

# Sample only from pages that document *how to use* FastAPI. Without this the
# generator happily writes questions from alternatives.md (a history of other
# frameworks) and the burger analogy in async.md — answerable, but nothing like
# what a real user asks, which makes the eval score less meaningful than it looks.
USAGE_PREFIXES = ("tutorial/", "advanced/", "how-to/", "learn/", "reference/")
USAGE_FILES = {"index.md", "python-types.md", "fastapi-cli.md", "environment-variables.md"}

MAKE_SYSTEM = """You write evaluation questions for a FastAPI documentation assistant.

Given one documentation chunk, write ONE question that:
- a FastAPI user would realistically ask, in their own words
- is answerable using ONLY this chunk
- does not quote the chunk verbatim or mention "the context"/"the docs"
- is specific enough to have one correct answer

Also list 1-3 short key facts the correct answer must contain.

Return JSON: {"question": str, "key_facts": [str]}"""

# Hand-written. Three failure modes the system must refuse:
#   1. a different framework entirely
#   2. a FastAPI-shaped question about something that doesn't exist
#   3. facts the docs never state (numbers a model loves to invent)
UNANSWERABLE = [
    "How do I configure Django settings using django-environ?",
    "How do I register a Django model in the admin site?",
    "What's the syntax for a Flask blueprint with a URL prefix?",
    "How do I use Django's ORM to do a select_related query?",
    "How do I write a Rails ActiveRecord migration?",
    "What is the maximum number of concurrent WebSocket connections FastAPI supports by default?",
    "What is FastAPI's default request timeout in seconds?",
    "How many worker processes does FastAPI spawn by default?",
    "What is the default maximum request body size in FastAPI?",
    "How do I enable FastAPI's built-in rate limiter?",
    "How do I use the @app.cache decorator to cache a response?",
    "How do I configure FastAPI's built-in Redis session backend?",
    "What did FastAPI change in version 0.200.0?",
    "How do I use FastAPI's GraphQL subscriptions support?",
    "What is the license fee for FastAPI Enterprise?",
]


def main() -> None:
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8") if l.strip()]
    # Stratify by source so the set isn't 45 questions about dependencies.
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        src = c["source"]
        if not (src.startswith(USAGE_PREFIXES) or src in USAGE_FILES):
            continue
        if len(c["text"]) > 400:  # thin chunks make vague questions
            by_source.setdefault(src, []).append(c)

    rng = random.Random(SEED)
    sources = sorted(by_source)
    rng.shuffle(sources)

    cost = CostTracker()
    rows: list[dict] = []
    dropped = 0

    for source in sources:
        if len(rows) >= N_ANSWERABLE:
            break
        chunk = rng.choice(by_source[source])
        raw = chat_json(MAKE_SYSTEM, f"CHUNK:\n{chunk['text']}", cost)
        question = (raw.get("question") or "").strip()
        facts = [str(f) for f in raw.get("key_facts", [])][:3]
        if not question or not facts:
            dropped += 1
            continue

        # Verification: does retrieval actually find the chunk this came from?
        hits = search(question, cost, k=8)
        if chunk["id"] not in {h.id for h in hits}:
            dropped += 1
            print(f"  drop (not retrievable): {question[:70]}")
            continue

        rows.append({
            "id": f"a{len(rows):03d}",
            "question": question,
            "answerable": True,
            "source": source,
            "chunk_id": chunk["id"],
            "key_facts": facts,
        })
        print(f"  keep [{len(rows):02d}] {question[:70]}")

    for i, q in enumerate(UNANSWERABLE):
        rows.append({
            "id": f"u{i:03d}", "question": q, "answerable": False,
            "source": "", "chunk_id": None, "key_facts": [],
        })

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\n{len(rows)} rows written ({N_ANSWERABLE} answerable target, {dropped} dropped, "
          f"{len(UNANSWERABLE)} unanswerable) — generation cost ${cost.total_usd:.4f}")
    print(f"REVIEW {OUT} BY HAND before trusting any number that comes out of it.")


if __name__ == "__main__":
    main()
