"""The self-healing loop, as a LangGraph state machine.

    retrieve -> generate -> critique --+-- grounded             -> finalise
                                       |-- ungrounded, tries<N  -> reformulate -> retrieve
                                       |-- insufficient context -> abstain
                                       +-- ungrounded, tries=N  -> abstain

That last edge is the promise of the whole service: once retries are exhausted it
abstains rather than shipping its best ungrounded guess. Flip it and the tagline
becomes marketing.
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from pathlib import Path

from app.critic import GENERATE_SYSTEM, assess_answer
from app.llm import CostTracker, chat_json
from app.retrieval import format_context, search
from app.schemas import Answer, AskResponse, Chunk, Critique, Verdict

POLICY = yaml.safe_load((Path(__file__).resolve().parent.parent / "policy.yaml").read_text(encoding="utf-8"))
GRAPH_CFG = POLICY["graph"]
MAX_ATTEMPTS = int(GRAPH_CFG["max_attempts"])
RETRIEVE_K = int(GRAPH_CFG["retrieve_k"])
ABSTAIN_WHEN_EXHAUSTED = bool(GRAPH_CFG["abstain_when_exhausted"])
ABSTAIN_MESSAGE = POLICY["abstain_message"].strip()


def _keep_last(_old, new):
    return new


class State(TypedDict, total=False):
    question: str
    query: Annotated[str, _keep_last]
    queries: list[str]
    chunks: list[Chunk]
    draft: Annotated[Answer | None, _keep_last]
    critique: Annotated[Critique | None, _keep_last]
    attempts: int
    abstained: bool
    cost: CostTracker


def n_retrieve(state: State) -> State:
    chunks = search(state["query"], state["cost"], k=RETRIEVE_K)
    return {"chunks": chunks, "queries": state.get("queries", []) + [state["query"]]}


def n_generate(state: State) -> State:
    raw = chat_json(
        GENERATE_SYSTEM,
        f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{format_context(state['chunks'])}",
        state["cost"],
    )
    try:
        draft = Answer.model_validate(raw)
    except Exception:
        # Schema violation is treated exactly like a bad answer: the critic will
        # reject an empty draft and the loop retries or abstains.
        draft = Answer(answer="", citations=[], context_sufficient=False)

    valid_ids = {c.id for c in state["chunks"]}
    draft.citations = [i for i in draft.citations if i in valid_ids]  # drop invented ids
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1}


def n_critique(state: State) -> State:
    draft = state["draft"]
    if not draft or not draft.answer.strip():
        return {"critique": Critique(
            verdict=Verdict.INSUFFICIENT_CONTEXT, groundedness=0.0,
            reason="generator produced no answer",
        )}
    if POLICY["output"]["require_citations"] and not draft.citations:
        return {"critique": Critique(
            verdict=Verdict.UNGROUNDED, groundedness=0.0,
            reason="answer cited no sources; policy requires citations",
            reformulated_query=state["question"],
        )}
    return {"critique": assess_answer(state["question"], state["chunks"], draft.answer, state["cost"])}


def n_reformulate(state: State) -> State:
    c = state["critique"]
    return {"query": (c.reformulated_query or "").strip() or state["question"]}


def n_abstain(state: State) -> State:
    return {"abstained": True}


def route(state: State) -> str:
    c = state["critique"]
    if c.verdict is Verdict.GROUNDED:
        return "finalise"

    # Both failure verdicts get the same treatment. Routing insufficient_context
    # straight to abstain looks right but isn't: it conflates "the corpus lacks
    # this" with "this retrieval missed it". FastAPI *does* document custom
    # response headers, but the query lexically matched a weak "Custom Headers"
    # caveat section, so the first pass looked like a corpus gap. Reformulation
    # exists for exactly that case.
    #
    # Genuine corpus gaps still abstain immediately, because the critic is
    # instructed to return an empty reformulated_query when the corpus plainly
    # cannot answer — which is what a Django question produces.
    #
    # Retry only if we have attempts left, budget left, and the critic
    # proposed a query we have not already run.
    #
    # That last condition matters: re-running an identical query re-retrieves
    # identical chunks, so the retry cannot change the outcome — it just costs a
    # generate + critique round trip. Both the "no citations" path (which falls
    # back to the original question) and a critic that echoes the question back
    # produce duplicates, so the guard lives here rather than in either caller.
    proposed = c.reformulated_query.strip()
    exhausted = state.get("attempts", 0) >= MAX_ATTEMPTS or state["cost"].exhausted
    already_tried = proposed.lower() in {q.strip().lower() for q in state.get("queries", [])}
    if exhausted or not proposed or already_tried:
        return "abstain" if ABSTAIN_WHEN_EXHAUSTED else "finalise"
    return "reformulate"


def build_graph():
    # Node names are prefixed because LangGraph reserves state keys as names:
    # a node called "critique" collides with the `critique` state field.
    g = StateGraph(State)
    g.add_node("do_retrieve", n_retrieve)
    g.add_node("do_generate", n_generate)
    g.add_node("do_critique", n_critique)
    g.add_node("do_reformulate", n_reformulate)
    g.add_node("do_abstain", n_abstain)

    g.add_edge(START, "do_retrieve")
    g.add_edge("do_retrieve", "do_generate")
    g.add_edge("do_generate", "do_critique")
    g.add_conditional_edges(
        "do_critique", route,
        {"finalise": END, "abstain": "do_abstain", "reformulate": "do_reformulate"},
    )
    g.add_edge("do_reformulate", "do_retrieve")
    g.add_edge("do_abstain", END)
    return g.compile()


GRAPH = build_graph()


def ask(question: str, cost_ceiling_usd: float = 0.0) -> AskResponse:
    started = time.perf_counter()
    cost = CostTracker(ceiling_usd=cost_ceiling_usd)
    final = GRAPH.invoke({
        "question": question, "query": question, "queries": [],
        "attempts": 0, "abstained": False, "cost": cost,
    })

    critique: Critique | None = final.get("critique")
    draft: Answer | None = final.get("draft")
    abstained = bool(final.get("abstained"))
    cited = {c.id for c in final.get("chunks", [])} & set(draft.citations if draft else [])

    return AskResponse(
        answer=ABSTAIN_MESSAGE if abstained else (draft.answer if draft else ABSTAIN_MESSAGE),
        abstained=abstained,
        citations=[] if abstained else [c for c in final.get("chunks", []) if c.id in cited],
        verdict=critique.verdict if critique else None,
        groundedness=critique.groundedness if critique else None,
        attempts=int(final.get("attempts", 0)),
        queries=final.get("queries", []),
        cost_usd=round(cost.total_usd, 6),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
