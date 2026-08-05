"""Typed contracts for the graph, the API, and the evals.

Shape deliberately mirrors the production quality gate this is modelled on
(`assess_clip` -> `ClipAssessment`): a verdict, a score, a machine-readable
rejection reason, and an *improved input* to retry with.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"                      # claims not supported by retrieved context
    INSUFFICIENT_CONTEXT = "insufficient_context"  # retrieval didn't surface the answer at all


class Chunk(BaseModel):
    id: int
    text: str
    source: str
    heading: str = ""
    score: float = 0.0


class Critique(BaseModel):
    """The critic's assessment of one draft answer."""

    verdict: Verdict
    groundedness: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""
    # The critic's suggested retry query. Same role as `improved_prompt` in the
    # video quality gate: the critic doesn't just reject, it says what to try next.
    reformulated_query: str = ""


class Answer(BaseModel):
    """Structured output the generator must produce. Validated, not trusted."""

    answer: str
    citations: list[int] = Field(default_factory=list)  # chunk ids
    # The model's own read on whether the context covered the question. Advisory —
    # the critic decides, not this.
    context_sufficient: bool = True


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Chunk] = Field(default_factory=list)
    verdict: Verdict | None = None
    groundedness: float | None = None
    attempts: int = 1
    queries: list[str] = Field(default_factory=list)  # original + each reformulation
    cost_usd: float = 0.0
    latency_ms: int = 0
    blocked_reason: str | None = None                 # set when guardrails refused the request
