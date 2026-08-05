"""Build the frozen corpus index from FastAPI's documentation at a pinned tag.

Why a pinned git tag and a committed index, rather than scraping into a vector DB:
this whole project is about *measuring* a RAG system. If the retrieval substrate
moves between runs, eval numbers aren't comparable and the CI gate is meaningless.
So the corpus is byte-reproducible and lives in the repo.

Outputs (all committed):
    corpus/chunks.jsonl     one chunk per line: id, text, source, heading
    corpus/embeddings.npy   float32 [n_chunks, 512], L2-normalised
    corpus/MANIFEST.json    tag, commit sha, params, model, chunks hash

Re-running produces a byte-identical MANIFEST.json. That's the idempotency check.

    python corpus/build_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

REPO = "https://github.com/fastapi/fastapi"
TAG = "0.141.1"
EXPECTED_SHA = "95f8322ee1dcda7ceace7b1c4f6c9915b36d748f"

ROOT = Path(__file__).resolve().parent
SRC = ROOT / ".src"
DOCS = SRC / "docs" / "en" / "docs"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512          # 512 keeps embeddings.npy ~3 MB instead of ~9 MB at 1536
EMBED_BATCH = 100         # the F1GPT loader embedded one chunk per call; this is 100x fewer round trips

CHUNK_CHARS = 800
CHUNK_OVERLAP = 150

# release-notes.md is 687 KB of a 1.4 MB corpus — a changelog that would dominate
# retrieval with version noise. The rest are community/meta pages, not documentation.
EXCLUDE_FILES = {
    "release-notes.md", "fastapi-people.md", "newsletter.md", "external-links.md",
    "translations.md", "translation-banner.md", "help-fastapi.md", "management.md",
    "contributing.md", "_llm-test.md", "benchmarks.md",
}
EXCLUDE_DIRS = {"about", "img", "css", "js"}

INCLUDE_RE = re.compile(r"\{\*\s*(\S+?\.py)(?:\s+[^*]*?)?\s*\*\}")
ANCHOR_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BLANKS_RE = re.compile(r"\n{3,}")


def ensure_clone() -> str:
    """Shallow-clone the pinned tag if absent, and verify we got the expected commit."""
    if not (SRC / ".git").exists():
        print(f"cloning {REPO} @ {TAG} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", TAG, "--quiet", REPO, str(SRC)],
            check=True,
        )
    sha = subprocess.run(
        ["git", "-C", str(SRC), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if sha != EXPECTED_SHA:
        sys.exit(f"tag {TAG} resolved to {sha}, expected {EXPECTED_SHA} — refusing to build")
    return sha


def resolve_includes(text: str, stats: dict) -> str:
    """Inline `{* ../../docs_src/foo.py hl[9] *}` as a fenced code block.

    The code examples are the whole point of these docs — drop them and the corpus
    can't answer "how do I declare a query parameter?", which is most of what
    anyone asks.

    Include paths are relative to `docs/en`, NOT to the markdown file's own
    directory. Getting that wrong resolves nothing and silently produces a
    code-free corpus, so misses are counted and the build fails on a high rate
    rather than shipping a quietly gutted index.
    """
    base = DOCS.parent

    def sub(m: re.Match) -> str:
        target = (base / m.group(1)).resolve()
        try:
            code = target.read_text(encoding="utf-8").strip()
        except OSError:
            stats["missed"] += 1
            return ""
        stats["resolved"] += 1
        return f"\n```python\n{code}\n```\n"

    return INCLUDE_RE.sub(sub, text)


def clean(text: str) -> str:
    text = ANCHOR_RE.sub("", text)          # "# Query Parameters { #query-parameters }"
    text = strip_html_outside_code(text)
    text = re.sub(r"^///.*$", "", text, flags=re.MULTILINE)  # mkdocs admonition fences
    return BLANKS_RE.sub("\n\n", text).strip()


def strip_html_outside_code(text: str) -> str:
    """Remove HTML tags from prose only.

    Running the tag regex over the whole document destroys code examples: the XML
    response sample in advanced/custom-response.md had every `<tag>` deleted,
    leaving a code block that demonstrates nothing. Fenced blocks are passed
    through untouched.
    """
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else HTML_TAG_RE.sub("", line))
    return "\n".join(out)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings, returning (heading, body) pairs.

    Carrying the heading onto every chunk measurably improves retrieval on docs —
    "Query Parameters" is often the only thing tying a code block to its topic.
    """
    parts = re.split(r"^(#{1,3})\s+(.+)$", text, flags=re.MULTILINE)
    if len(parts) == 1:
        return [("", text)]
    out, preamble = [], parts[0].strip()
    if preamble:
        out.append(("", preamble))
    for i in range(1, len(parts), 3):
        heading, body = parts[i + 1].strip(), parts[i + 2].strip()
        if body:
            out.append((heading, body))
    return out


def _tail(text: str, budget: int) -> str:
    """Trailing whole paragraphs (else whole lines) fitting in `budget` chars.

    Slicing the overlap by raw character count cuts mid-word — it produced chunks
    starting "ntent-Type header", which embeds as garbage and retrieves as noise.
    """
    if budget <= 0:
        return ""
    paras = [p for p in text.split("\n\n") if p.strip()]
    kept: list[str] = []
    total = 0
    for p in reversed(paras):
        if total + len(p) > budget and kept:
            break
        kept.insert(0, p)
        total += len(p) + 2
    if kept:
        return "\n\n".join(kept)
    # Single paragraph larger than the budget: fall back to whole trailing lines.
    lines = text.split("\n")
    kept, total = [], 0
    for ln in reversed(lines):
        if total + len(ln) > budget and kept:
            break
        kept.insert(0, ln)
        total += len(ln) + 1
    return "\n".join(kept)


def pack(body: str) -> list[str]:
    """Pack paragraphs into ~CHUNK_CHARS windows, overlapping on whole paragraphs."""
    paras = [p for p in re.split(r"\n\n+", body) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > CHUNK_CHARS:
            chunks.append(cur.strip())
            tail = _tail(cur, CHUNK_OVERLAP)
            cur = f"{tail}\n\n{p}" if tail else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())

    # A single oversized paragraph (usually a long code block) still needs splitting,
    # but split on line boundaries so code stays readable.
    out: list[str] = []
    limit = CHUNK_CHARS * 2
    for c in chunks:
        while len(c) > limit:
            cut = c.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            out.append(c[:cut].strip())
            c = c[cut:].lstrip("\n")
        if c.strip():
            out.append(c.strip())
    return out


def build_chunks() -> list[dict]:
    files = sorted(
        p for p in DOCS.rglob("*.md")
        if p.name not in EXCLUDE_FILES
        and not (set(p.relative_to(DOCS).parts[:-1]) & EXCLUDE_DIRS)
    )
    print(f"{len(files)} markdown files after exclusions")

    stats = {"resolved": 0, "missed": 0}
    chunks: list[dict] = []
    for path in files:
        source = path.relative_to(DOCS).as_posix()
        text = clean(resolve_includes(path.read_text(encoding="utf-8"), stats))
        for heading, body in split_sections(text):
            for piece in pack(body):
                if len(piece.strip()) < 60:
                    continue  # nav stubs and one-liners retrieve as noise
                prefix = f"{source} — {heading}\n\n" if heading else f"{source}\n\n"
                chunks.append({
                    "id": len(chunks),
                    "text": prefix + piece.strip(),
                    "source": source,
                    "heading": heading,
                })

    total = stats["resolved"] + stats["missed"]
    print(f"code includes: {stats['resolved']} resolved, {stats['missed']} missed")
    if total and stats["missed"] / total > 0.05:
        sys.exit(f"{stats['missed']}/{total} includes unresolved — check the base path")
    return chunks


def embed(texts: list[str], client: OpenAI) -> np.ndarray:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch, dimensions=EMBED_DIMS)
        vecs.extend(d.embedding for d in resp.data)
        print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    arr = np.asarray(vecs, dtype=np.float32)
    # L2-normalise at build time so query-time retrieval is one dot product.
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (copy .env.example to .env)")

    sha = ensure_clone()
    chunks = build_chunks()
    print(f"{len(chunks)} chunks")

    chunks_path = ROOT / "chunks.jsonl"
    payload = "\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in chunks) + "\n"
    chunks_path.write_text(payload, encoding="utf-8", newline="\n")

    vecs = embed([c["text"] for c in chunks], OpenAI())
    np.save(ROOT / "embeddings.npy", vecs)

    manifest = {
        "repo": REPO,
        "tag": TAG,
        "commit": sha,
        "n_chunks": len(chunks),
        "n_sources": len({c["source"] for c in chunks}),
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBED_MODEL,
        "embedding_dims": EMBED_DIMS,
        # Hash the text, not the vectors: text is deterministic, and this is what
        # makes re-runs verifiably identical.
        "chunks_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
