"""Run the golden set through the service and write a results file.

    python -m evals.run_evals

Metrics, and why these ones:

  hallucination_rate      of the answerable questions the system chose to ANSWER,
                          the fraction the judge found ungrounded. The headline
                          safety number and the primary CI gate.
  false_answer_rate       of the UNANSWERABLE questions, the fraction the system
                          answered anyway. This is the number the whole "refuses
                          to guess" claim rests on.
  false_abstention_rate   of the answerable questions, the fraction it refused.
                          Tracked because it is trivial to score a perfect
                          hallucination rate by abstaining on everything — these
                          two must be read together.
  answer_relevancy        does the answer address the question, judged 0-1.
  key_fact_coverage       did the answer contain the facts the question was written for.
  citation_rate           fraction of answered questions carrying >=1 valid citation.
  latency p50/p95, cost_per_query_usd
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.graph import ask
from app.guardrails import screen
from app.llm import GEN_MODEL, CostTracker
from app.retrieval import format_context
from evals.judge import judge, judge_available

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden.jsonl"
RESULTS = ROOT / "results"
CONCURRENCY = 8


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT.parent, check=True,
        ).stdout.strip()
    except Exception:
        return "nogit"


def run_one(row: dict, cost: CostTracker) -> dict:
    s = screen(row["question"])
    if not s.allowed:
        return {**row, "abstained": True, "blocked": s.reason, "latency_ms": 0,
                "cost_usd": 0.0, "answer": "", "citations": [], "judged": None}

    r = ask(s.text)
    rec = {
        "id": row["id"], "question": row["question"], "answerable": row["answerable"],
        "abstained": r.abstained, "blocked": None, "answer": r.answer,
        "citations": [c.id for c in r.citations], "attempts": r.attempts,
        "verdict": r.verdict.value if r.verdict else None,
        "latency_ms": r.latency_ms, "cost_usd": r.cost_usd, "judged": None,
    }

    # Only answered responses need judging; an abstention is scored by whether it
    # should have abstained, which the dataset already tells us.
    if not r.abstained and r.answer.strip():
        rec["judged"] = judge(
            row["question"], format_context(r.citations), r.answer,
            row.get("key_facts", []), cost,
        )
    return rec


def summarise(recs: list[dict]) -> dict:
    ans = [r for r in recs if r["answerable"]]
    una = [r for r in recs if not r["answerable"]]
    answered_ans = [r for r in ans if not r["abstained"] and r["judged"]]

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 1) if d else 0.0

    def mean(vals: list[float]) -> float:
        return round(statistics.mean(vals), 3) if vals else 0.0

    lats = sorted(r["latency_ms"] for r in recs if r["latency_ms"])
    costs = [r["cost_usd"] for r in recs]

    return {
        "n_total": len(recs), "n_answerable": len(ans), "n_unanswerable": len(una),
        "hallucination_rate": pct(sum(not r["judged"]["grounded"] for r in answered_ans), len(answered_ans)),
        "false_answer_rate": pct(sum(not r["abstained"] for r in una), len(una)),
        "false_abstention_rate": pct(sum(r["abstained"] for r in ans), len(ans)),
        "answer_relevancy": mean([r["judged"]["relevancy"] for r in answered_ans]),
        "key_fact_coverage": mean([r["judged"]["facts_covered"] for r in answered_ans]),
        "citation_rate": pct(sum(bool(r["citations"]) for r in answered_ans), len(answered_ans)),
        "p50_latency_ms": lats[len(lats) // 2] if lats else 0,
        "p95_latency_ms": lats[int(len(lats) * 0.95) - 1] if lats else 0,
        "cost_per_query_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
    }


def main() -> None:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    judge_model, cross = judge_available()
    if not cross:
        print("WARNING: ANTHROPIC_API_KEY unset — judging with the generator's own "
              "provider. Results are weaker; set the key for cross-provider judging.")

    cost = CostTracker()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        recs = list(pool.map(lambda r: run_one(r, cost), rows))

    summary = summarise(recs)
    payload = {
        "sha": git_sha(),
        "gen_model": GEN_MODEL,
        "judge_model": judge_model,
        "judge_cross_provider": cross,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "judge_cost_usd": round(cost.total_usd, 5),
        "summary": summary,
        "records": recs,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{payload['sha']}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (RESULTS / "latest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    width = max(len(k) for k in summary)
    print(f"\n{'metric'.ljust(width)}  value")
    print(f"{'-' * width}  -----")
    for k, v in summary.items():
        print(f"{k.ljust(width)}  {v}")
    print(f"\njudge: {judge_model} (cross-provider: {cross}) · "
          f"wall {payload['wall_seconds']}s · judge cost ${payload['judge_cost_usd']}")


if __name__ == "__main__":
    main()
