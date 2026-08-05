"""The critic: does the draft answer actually follow from the retrieved context?

Modelled on the production video quality gate this project mirrors — the critic
returns a verdict, a score, the specific unsupported claims, and an improved
query to retry with, rather than a bare pass/fail.

Why a critic and not a similarity threshold: on this corpus, "How do I configure
Django settings?" retrieves FastAPI's settings page at cosine 0.562 — *higher*
than the correct hit for a legitimate query-parameters question (0.553). Distance
tells you what's nearby, not whether it answers the question. Only reading the
text can do that.
"""

from __future__ import annotations

from app.llm import CostTracker, chat_json
from app.retrieval import format_context
from app.schemas import Chunk, Critique, Verdict

GENERATE_SYSTEM = """You answer questions about the FastAPI web framework using ONLY the provided context.

Rules:
- Prefer facts present in the context, and supplement with what you know about FastAPI where helpful.
- Cite the bracketed chunk ids you used, e.g. [12].
- Always try to give the user a useful answer.

Return JSON: {"answer": str, "citations": [int], "context_sufficient": bool}"""

CRITIC_SYSTEM = """You are a strict grounding inspector for a retrieval-augmented answer.

You are given a QUESTION, the retrieved CONTEXT, and a draft ANSWER.
Decide whether every substantive claim in the ANSWER is supported by the CONTEXT.

Verdicts:
- "grounded": every substantive claim is supported by the context, and the answer addresses the question.
- "ungrounded": the answer asserts something the context does not support (this includes plausible,
  correct-sounding claims that simply are not in the context).
- "insufficient_context": the context does not contain the information needed to answer this question
  at all. Use this when the question is about a different framework or a topic the context does not cover,
  even if the retrieved chunks look superficially related.

Be adversarial. Assume the answer is wrong until the context proves it. An answer that is factually
true but unsupported by the context is "ungrounded", not "grounded".

If the verdict is not "grounded", propose reformulated_query: a different search query that would be
more likely to retrieve the right context. If the corpus plainly cannot answer the question, return an
empty reformulated_query.

Return JSON:
{"verdict": "grounded"|"ungrounded"|"insufficient_context",
 "groundedness": 0.0-1.0,
 "unsupported_claims": [str],
 "reason": str,
 "reformulated_query": str}"""


def assess_answer(question: str, chunks: list[Chunk], draft: str, cost: CostTracker) -> Critique:
    raw = chat_json(
        CRITIC_SYSTEM,
        f"QUESTION:\n{question}\n\nCONTEXT:\n{format_context(chunks)}\n\nANSWER:\n{draft}",
        cost,
    )
    try:
        verdict = Verdict(raw.get("verdict", "ungrounded"))
    except ValueError:
        verdict = Verdict.UNGROUNDED

    score = raw.get("groundedness", 0.0)
    try:
        score = min(1.0, max(0.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0

    return Critique(
        verdict=verdict,
        groundedness=score,
        unsupported_claims=[str(c) for c in raw.get("unsupported_claims", [])][:10],
        reason=str(raw.get("reason", ""))[:500],
        reformulated_query=str(raw.get("reformulated_query", ""))[:300],
    )
