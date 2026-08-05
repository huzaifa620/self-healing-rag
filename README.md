# self-healing-rag

**A RAG service that critiques its own answers, refuses to guess, and can't be merged if it regresses.**

Most RAG demos prove someone can wire up a retrieval chain. Almost none prove the thing works — or would notice if it stopped. This one is built around measurement: it answers questions about the FastAPI documentation, grades every answer for grounding before returning it, abstains when the docs don't support an answer, and fails its own CI build if quality drops.

> **[ADD SCREENSHOT HERE]** — a red CI check reading *"hallucination rate 11.2% exceeds 5.0% threshold — blocking merge"*. See [Reproducing the failure](#reproducing-the-failure).

---

## Results

60-question golden set — 45 answerable from the corpus, 15 deliberately unanswerable. Generator `gpt-4o-mini`.

> **Judge caveat, stated up front:** this run was judged by `gpt-4o-mini` — the generator's own model. Cross-provider judging with `claude-haiku-4-5` is implemented and activates whenever `ANTHROPIC_API_KEY` is set, but the numbers below were *not* produced that way, and same-provider judging is the weaker methodology (see [Design decisions](#design-decisions)). Every results file records `judge_model` and `judge_cross_provider` so this can never be ambiguous.

| Metric | Result | Gate |
|---|---|---|
| **Hallucination rate** — answered questions judged ungrounded | **0.0 %** | ≤ 5 % |
| **False-answer rate** — unanswerable questions it answered anyway | **0.0 %** | ≤ 10 % |
| **False-abstention rate** — answerable questions it refused | **0.0 %** | ≤ 25 % |
| Answer relevancy (0–1) | 0.996 | ≥ 0.70 |
| Key-fact coverage (0–1) | 0.864 | — |
| Citation rate | 100 % | ≥ 90 % |
| p50 / p95 latency | 3.7 s / 6.9 s | p95 ≤ baseline +20 % |
| Cost per query | $0.0005 | ≤ baseline +25 % |

Full run: 60 questions in 37 s, about $0.04 all in.

The false-abstention row matters as much as the first one. Abstaining on everything scores a perfect hallucination rate and a perfect false-answer rate, so the gate also fails if the system refuses more than 25 % of questions it can actually answer. Safety metrics are only meaningful next to a usefulness metric.

---

## How it works

```
retrieve ─→ generate ─→ critique ─┬── grounded ────────────→ answer + citations
    ↑                             │
    │                             ├── needs another angle ──→ reformulate ─┐
    │                             │                                        │
    └─────────────────────────────┴── corpus can't answer ──→ ABSTAIN      │
                                      or retries exhausted                 │
    └──────────────────────────────────────────────────────────────────────┘
```

A [LangGraph](https://langchain-ai.github.io/langgraph/) state machine (`app/graph.py`). The retrieve→critique cycle is the point: an answer that fails the grounding check goes back for another retrieval with a reformulated query, and **when retries run out the service abstains rather than shipping its best ungrounded guess**. That edge is the whole product — flip `abstain_when_exhausted` in `policy.yaml` and the tagline becomes marketing.

The critic (`app/critic.py`) returns a verdict, a groundedness score, the specific unsupported claims, and a suggested retry query — so it doesn't just reject, it says what to try next.

### Why a critic instead of a similarity threshold

On this corpus, *"How do I configure Django settings?"* retrieves FastAPI's settings page at cosine **0.562** — **higher** than the correct hit for a legitimate query-parameters question (**0.553**). Distance tells you what's nearby, not whether it answers the question. No threshold separates those two cases. Only reading the text does.

---

## Design decisions

**The corpus is frozen and committed.** FastAPI's docs at a pinned tag (`0.141.1`, commit `95f8322`), chunked and embedded into `corpus/embeddings.npy` (2,052 chunks, 3.8 MB). Retrieval is a single numpy dot product — no vector database, no service to reach, no secrets needed for the retrieval half of CI. When the whole project is about measurement, the retrieval substrate cannot be allowed to move between runs, or the numbers aren't comparable and the gate means nothing. `MANIFEST.json` pins the source commit and a hash of the chunk text; CI verifies it before trusting any result.

**The judge is a different provider from the generator.** OpenAI generates, Anthropic grades. A model family grading its own output shares its blind spots — it tends to accept fluent, confidently-wrong answers of exactly the kind it produces. Every results file records `judge_model` and `judge_cross_provider`, and the runner warns loudly if it has to fall back to same-provider judging, because an eval that quietly changes methodology is worse than one that fails.

**The unanswerable set writes itself.** Ask Django, Flask, and Rails questions of a FastAPI corpus and correct behaviour is unambiguous: abstain. The other unanswerable questions target invented FastAPI features (`@app.cache`, a built-in rate limiter) and facts the docs never state — *"What is FastAPI's default request timeout?"*, *"How many workers does it spawn by default?"* Those are the ones a language model most enjoys making up a number for.

**Grounding is judged against the chunks the answer cited**, not everything retrieved. A stricter bar, and the right one: a reader can only verify what was cited, so an uncited claim is an unverifiable claim.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env      # add OPENAI_API_KEY, and ANTHROPIC_API_KEY for the judge
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. The corpus index is committed, so there is no ingestion step — first answer in under a minute from a clean clone.

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How do I declare a query parameter with a default value?"}'
```

Rebuild the index from source (only needed to change the pin or the chunker):

```bash
python corpus/build_corpus.py      # re-running produces a byte-identical MANIFEST.json
```

---

## The eval harness

```bash
python -m evals.run_evals            # runs the golden set, writes evals/results/<sha>.json
python -m pytest evals/test_gates.py # the merge gate
python -m evals.dashboard            # regenerates docs/index.html
```

`evals/make_golden.py` generates the answerable half from sampled chunks and **verifies each question by checking that retrieval actually finds the chunk it was written from** — 16 of 61 generated questions were dropped that way, because a question the retriever can't reach produces a fake eval failure later. Sampling is restricted to the usage docs (`tutorial/`, `advanced/`, `how-to/`, `reference/`); without that filter the generator writes questions about FastAPI's project history and the burger analogy in `async.md` — answerable, but nothing like what a real user asks.

The 15 unanswerable questions are hand-written. They're the most valuable rows in the file and not worth generating.

### The first run failed its own gate

The initial run came in at **6.7 % hallucination rate** — over the 5 % threshold. Two of the three failures turned out to be judge bugs, not model failures. This answer was marked ungrounded:

> **Q:** How do I start the FastAPI server in production mode?
> **A:** To start the FastAPI server in production mode, use the command `fastapi run`.
> **Judge:** *"...correctly states the command, but does not mention the server's default address or the documentation URL, which are key facts."*

The answer is correct and fully supported. The judge was scoring **completeness** as **grounding**, because a single prompt asked it for both and the criteria bled together. Fixing the prompt — decide `grounded` without reference to `key_facts`, score incompleteness separately — moved the rate to **0.0 %** without touching the threshold or the system under test.

That is the argument for having an eval harness at all: the number that looked like a model failure was a measurement failure, and only an inspectable per-question record made the difference visible.

### Reproducing the failure

To see the gate work, weaken the critic and push:

```bash
git checkout -b regress
# in app/critic.py, replace the CRITIC_SYSTEM verdict rules with "always return grounded"
python -m evals.run_evals && python -m pytest evals/test_gates.py   # fails
```

CI runs the same two commands on every push and posts the metrics table as a PR comment.

---

## Guardrails

`app/guardrails.py` screens input before it reaches the model. PII is redacted (card numbers **Luhn-validated**, so an order id like `1234567812345678` survives intact while a real card doesn't), and prompt-injection attempts are refused without an API call. Policy lives in `policy.yaml` — thresholds, detectors, retry limits, and the abstain message, all editable without touching code.

Output validation lives in the graph: `Answer` is a Pydantic model, citation ids are checked against the retrieved set so invented ones are dropped, and a schema violation becomes an empty draft that the critic rejects — routing into the same retry-or-abstain path as any other bad answer.

There is deliberately **no** LLM-based "is this on topic?" check. The graph already refuses off-topic questions correctly, so a classifier call per request would double latency to re-derive an answer the critic produces for free.

---

## Known limitations

- **Retrieval is a numpy scan.** At 2,052 chunks it's sub-millisecond and an ANN index would be pure overhead. Past roughly 50k chunks this needs FAISS or pgvector; the interface is two functions in `app/retrieval.py`.
- **The judge is noisy on terse answers.** Probing it with a deliberately short but correct answer produced a false "ungrounded". Real generated answers are longer and don't trigger it, so it doesn't affect the reported numbers — but a single judge with a single prompt is the weakest link here. A 3-judge panel with majority voting is the obvious next step.
- **Guardrails are regex heuristics, not classifiers.** They catch the common instruction-override phrasings and will miss obfuscated or multilingual injections.
- **60 questions is a small set.** Enough to catch a real regression, not enough for tight confidence intervals on a 1-in-45 failure. It grows when a real failure is found — that's the actual workflow, not a target to hit up front.
- **One corpus, one language, one framework.** Nothing here has been tested against a corpus where the right answer is spread across many documents.

## What I'd do next

Judge panel with majority voting; per-source retrieval metrics to find which doc areas retrieve badly; caching the critic on unchanged (question, chunks) pairs to cut cost; and a nightly run against the FastAPI `master` branch to catch the day the docs change out from under the pinned index.

---

## Layout

```
app/         graph.py (LangGraph loop) · critic.py · retrieval.py · guardrails.py · llm.py (cost tracking)
corpus/      build_corpus.py + committed chunks.jsonl, embeddings.npy, MANIFEST.json
evals/       make_golden.py · golden.jsonl · run_evals.py · judge.py · test_gates.py · dashboard.py
.github/     evals.yml — the merge gate
```

Built by [Muhammad Huzaifa](https://huzaifa620.github.io). The critic/retry architecture mirrors a video-generation quality gate I built and run in production — same shape, applied to text.
