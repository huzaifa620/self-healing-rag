"""Render every results file into one static page for GitHub Pages.

    python -m evals.dashboard

No service, no database, no JS libraries — the whole point of committing results
is that the history is just files in the repo. Inline SVG sparklines are enough
to see a trend.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "docs" / "index.html"

# (key, label, lower_is_better, unit)
SERIES = [
    ("hallucination_rate", "Hallucination rate", True, "%"),
    ("false_answer_rate", "False-answer rate (unanswerable)", True, "%"),
    ("false_abstention_rate", "False-abstention rate", True, "%"),
    ("answer_relevancy", "Answer relevancy", False, ""),
    ("key_fact_coverage", "Key-fact coverage", False, ""),
    ("citation_rate", "Citation rate", False, "%"),
    ("p95_latency_ms", "p95 latency", True, "ms"),
    ("cost_per_query_usd", "Cost per query", True, "$"),
]


def spark(values: list[float], lower_is_better: bool) -> str:
    if len(values) < 2:
        return '<span class="nodata">needs 2+ runs</span>'
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{i / (len(values) - 1) * 100:.1f},{28 - (v - lo) / span * 24:.1f}"
        for i, v in enumerate(values)
    )
    good = (values[-1] <= values[0]) if lower_is_better else (values[-1] >= values[0])
    colour = "#0a7d33" if good else "#b91c1c"
    return (f'<svg viewBox="0 0 100 30" preserveAspectRatio="none" class="spark">'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.5"/></svg>')


def main() -> None:
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name == "latest.json":
            continue
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit("no results in evals/results — run `python -m evals.run_evals`")

    latest = runs[-1]
    rows = []
    for key, label, lower, unit in SERIES:
        vals = [r["summary"][key] for r in runs if key in r["summary"]]
        cur = vals[-1] if vals else 0
        shown = f"${cur}" if unit == "$" else f"{cur}{unit}"
        rows.append(
            f"<tr><td>{label}</td><td class='num'>{shown}</td>"
            f"<td class='sparkcell'>{spark(vals, lower)}</td></tr>"
        )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>self-healing-rag — eval history</title>
<style>
 :root{{color-scheme:light dark;--bg:#fff;--fg:#111;--mut:#666;--line:#e4e4e4}}
 @media(prefers-color-scheme:dark){{:root{{--bg:#0f1115;--fg:#e8e8ea;--mut:#9aa0a6;--line:#2a2d35}}}}
 body{{margin:0;padding:2.5rem 1rem;background:var(--bg);color:var(--fg);
      font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
 main{{max-width:760px;margin:0 auto}} h1{{font-size:1.4rem;margin:0 0 .25rem}}
 p.sub{{color:var(--mut);margin:0 0 1.75rem}}
 table{{width:100%;border-collapse:collapse}}
 th,td{{text-align:left;padding:.6rem .5rem;border-bottom:1px solid var(--line)}}
 .num{{font-variant-numeric:tabular-nums;white-space:nowrap;width:7rem}}
 .sparkcell{{width:120px}} .spark{{width:110px;height:30px;display:block}}
 .nodata{{color:var(--mut);font-size:.8rem}}
 footer{{margin-top:2rem;color:var(--mut);font-size:.85rem}}
</style></head><body><main>
<h1>self-healing-rag — eval history</h1>
<p class="sub">{len(runs)} run(s). Every push runs the golden set and fails the build if the
hallucination rate exceeds 5% or the system answers more than 10% of the questions it should refuse.</p>
<table><thead><tr><th>metric</th><th class="num">latest</th><th>trend</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<footer>
 latest run <code>{latest['sha']}</code> ·
 generator <code>{latest['gen_model']}</code> ·
 judge <code>{latest['judge_model']}</code>
 (cross-provider: {latest['judge_cross_provider']}) ·
 {latest['summary']['n_answerable']} answerable + {latest['summary']['n_unanswerable']} unanswerable questions
</footer>
</main></body></html>
""", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} from {len(runs)} run(s)")


if __name__ == "__main__":
    main()
