"""The merge gate. These assertions are what turn CI red.

Run `python -m evals.run_evals` first; these read evals/results/latest.json.

Absolute thresholds catch "this change made the system unsafe". Regression
thresholds against evals/baseline.json catch "this change made it slower or more
expensive". Both matter — a change that halves the hallucination rate while
tripling latency is still a change someone needs to approve on purpose.

To move a baseline, run the evals on main and copy the summary into
evals/baseline.json in its own commit, so the change is reviewable rather than
buried in a feature diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
LATEST = ROOT / "results" / "latest.json"
BASELINE = ROOT / "baseline.json"

MAX_HALLUCINATION_RATE = 5.0     # % of answered answerable questions judged ungrounded
MAX_FALSE_ANSWER_RATE = 10.0     # % of unanswerable questions answered anyway
MAX_FALSE_ABSTENTION_RATE = 25.0 # % of answerable questions refused
MIN_RELEVANCY = 0.70
MIN_CITATION_RATE = 90.0
MAX_P95_LATENCY_REGRESSION = 1.20   # 20% slower than baseline
MAX_COST_REGRESSION = 1.25          # 25% more expensive than baseline


@pytest.fixture(scope="module")
def summary() -> dict:
    if not LATEST.exists():
        pytest.skip("no results — run `python -m evals.run_evals` first")
    return json.loads(LATEST.read_text(encoding="utf-8"))["summary"]


@pytest.fixture(scope="module")
def baseline() -> dict | None:
    return json.loads(BASELINE.read_text(encoding="utf-8"))["summary"] if BASELINE.exists() else None


def test_hallucination_rate(summary):
    v = summary["hallucination_rate"]
    assert v <= MAX_HALLUCINATION_RATE, (
        f"hallucination rate {v}% exceeds {MAX_HALLUCINATION_RATE}% threshold — blocking merge"
    )


def test_false_answer_rate(summary):
    v = summary["false_answer_rate"]
    assert v <= MAX_FALSE_ANSWER_RATE, (
        f"false-answer rate {v}% on unanswerable questions exceeds "
        f"{MAX_FALSE_ANSWER_RATE}% — the system is guessing"
    )


def test_not_abstaining_on_everything(summary):
    # Without this, abstaining on 100% of questions scores a perfect
    # hallucination rate and a perfect false-answer rate.
    v = summary["false_abstention_rate"]
    assert v <= MAX_FALSE_ABSTENTION_RATE, (
        f"false-abstention rate {v}% exceeds {MAX_FALSE_ABSTENTION_RATE}% — "
        "the system is refusing questions it can answer"
    )


def test_relevancy(summary):
    assert summary["answer_relevancy"] >= MIN_RELEVANCY


def test_citation_rate(summary):
    assert summary["citation_rate"] >= MIN_CITATION_RATE


def test_latency_has_not_regressed(summary, baseline):
    if not baseline:
        pytest.skip("no baseline committed yet")
    ceiling = baseline["p95_latency_ms"] * MAX_P95_LATENCY_REGRESSION
    assert summary["p95_latency_ms"] <= ceiling, (
        f"p95 latency {summary['p95_latency_ms']}ms exceeds "
        f"{ceiling:.0f}ms ({baseline['p95_latency_ms']}ms baseline +20%)"
    )


def test_cost_has_not_regressed(summary, baseline):
    if not baseline:
        pytest.skip("no baseline committed yet")
    ceiling = baseline["cost_per_query_usd"] * MAX_COST_REGRESSION
    assert summary["cost_per_query_usd"] <= ceiling, (
        f"cost/query ${summary['cost_per_query_usd']} exceeds "
        f"${ceiling:.6f} (${baseline['cost_per_query_usd']} baseline +25%)"
    )
