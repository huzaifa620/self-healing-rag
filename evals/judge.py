"""LLM-as-judge for grounding, relevancy, and key-fact coverage.

The judge should be a *different provider* from the generator. A model family
grading its own output shares its blind spots — it tends to accept fluent,
confidently-wrong answers that it would have produced itself. So Anthropic judges
what OpenAI generated.

If ANTHROPIC_API_KEY is absent the judge falls back to OpenAI and records that in
the results, because an eval that quietly changes methodology is worse than one
that fails loudly. Every results file carries `judge_model` and
`judge_cross_provider` so a number can never be compared across incompatible runs.
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

from app.llm import CostTracker, chat_json

load_dotenv()

ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"

JUDGE_SYSTEM = """You grade a retrieval-augmented answer. Be strict and adversarial.

You get a QUESTION, the CONTEXT that was retrieved, the ANSWER that was given, and
optionally KEY_FACTS the answer was expected to contain.

Grade three INDEPENDENT things. Do not let one influence another.

1. grounded (bool) — decide this using ONLY the QUESTION, CONTEXT and ANSWER.
   true if every substantive claim in the ANSWER is supported by the CONTEXT.
   false if the ANSWER asserts something the CONTEXT does not support, including
   claims that are true of FastAPI generally but absent from this CONTEXT.

   CRITICAL: ignore KEY_FACTS entirely when deciding `grounded`. An answer that is
   brief, or omits a key fact, or is less complete than you would like, is still
   GROUNDED as long as what it does say is supported. Incompleteness is scored by
   facts_covered, never by grounded. A short correct answer is grounded.

2. relevancy (0.0-1.0) — how directly the ANSWER addresses the QUESTION.

3. facts_covered (0.0-1.0) — fraction of KEY_FACTS present in the ANSWER.
   Return 1.0 when KEY_FACTS is empty.

In `note`, if grounded is false, quote the specific unsupported claim. If your only
complaint is that something is missing, grounded must be true.

Return ONLY JSON:
{"grounded": bool, "relevancy": 0.0-1.0, "facts_covered": 0.0-1.0, "note": str}"""


def judge_available() -> tuple[str, bool]:
    """Returns (model, is_cross_provider)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return ANTHROPIC_MODEL, True
    return OPENAI_JUDGE_MODEL, False


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _judge_anthropic(user: str, cost: CostTracker) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        temperature=0.0,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    cost.add(ANTHROPIC_MODEL, resp.usage.input_tokens, resp.usage.output_tokens)
    return _extract_json("".join(b.text for b in resp.content if b.type == "text"))


def judge(question: str, context: str, answer: str, key_facts: list[str], cost: CostTracker) -> dict:
    user = (
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        f"KEY_FACTS:\n{json.dumps(key_facts)}"
    )
    raw = _judge_anthropic(user, cost) if os.getenv("ANTHROPIC_API_KEY") else chat_json(
        JUDGE_SYSTEM, user, cost, model=OPENAI_JUDGE_MODEL
    )

    def num(key: str) -> float:
        try:
            return min(1.0, max(0.0, float(raw.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    return {
        "grounded": bool(raw.get("grounded", False)),
        "relevancy": num("relevancy"),
        "facts_covered": num("facts_covered"),
        "note": str(raw.get("note", ""))[:300],
    }
