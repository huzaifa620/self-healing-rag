"""FastAPI surface: POST /ask, GET /health, and a one-page demo at /."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.graph import ask
from app.guardrails import screen
from app.schemas import AskRequest, AskResponse

STATIC = Path(__file__).resolve().parent / "static"
MANIFEST = json.loads((Path(__file__).resolve().parent.parent / "corpus" / "MANIFEST.json").read_text())

app = FastAPI(
    title="self-healing-rag",
    description="A RAG service that critiques its own answers and refuses to guess.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus": {k: MANIFEST[k] for k in ("tag", "commit", "n_chunks")}}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    s = screen(req.question)
    if not s.allowed:
        return AskResponse(
            answer="This request was blocked before reaching the model.",
            abstained=True,
            blocked_reason=s.reason,
        )
    return ask(s.text)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
