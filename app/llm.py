"""LLM clients, token accounting, and JSON-mode helpers.

Every call goes through here so cost is measured rather than estimated. The eval
harness reports cost-per-query as a gated metric, which only works if the number
is real.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEN_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512

# USD per 1M tokens. Only used for accounting, so approximate is fine — but keep
# it in one place so a price change is a one-line edit.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
    "text-embedding-3-small": (0.02, 0.0),
}


class CostTracker:
    """Running spend for one request (or one eval run).

    A ceiling of 0 means no ceiling. When a ceiling is set and reached, the graph
    stops retrying and abstains — a bounded-cost failure rather than an unbounded
    retry loop.
    """

    def __init__(self, ceiling_usd: float = 0.0) -> None:
        self.total_usd = 0.0
        self.ceiling_usd = float(ceiling_usd)
        self.calls = 0

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        self.total_usd += (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
        self.calls += 1

    @property
    def exhausted(self) -> bool:
        return self.ceiling_usd > 0 and self.total_usd >= self.ceiling_usd


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set (copy .env.example to .env)")
        _client = OpenAI()
    return _client


def chat_json(
    system: str,
    user: str,
    cost: CostTracker,
    model: str = GEN_MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """One JSON-mode chat call, with usage recorded.

    Returns {} on unparseable output rather than raising — callers treat that as a
    validation failure and retry, which is the same path a schema violation takes.
    """
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if resp.usage:
        cost.add(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


def embed_query(text: str, cost: CostTracker) -> list[float]:
    resp = client().embeddings.create(model=EMBED_MODEL, input=[text], dimensions=EMBED_DIMS)
    if resp.usage:
        cost.add(EMBED_MODEL, resp.usage.prompt_tokens, 0)
    return resp.data[0].embedding
