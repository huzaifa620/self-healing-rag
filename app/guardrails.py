"""Input guardrails: PII redaction and prompt-injection screening.

Scope note — there is deliberately no LLM-based "is this on topic?" check. The
graph already refuses off-topic questions correctly (a Django question abstains
in one attempt), so a classifier call per request would double latency to
re-derive an answer the critic already produces for free.

Output-side validation lives in the graph rather than here: `Answer` is a Pydantic
model, invented citation ids are dropped against the retrieved set, and a schema
violation becomes an empty draft that the critic rejects — which routes into the
same retry-or-abstain path as any other bad answer.

These detectors are regex heuristics, not classifiers. See README "Known
limitations" for what that does and doesn't catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

POLICY = yaml.safe_load((Path(__file__).resolve().parent.parent / "policy.yaml").read_text(encoding="utf-8"))
INPUT_CFG = POLICY["input"]

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{0,4}(?!\w)")
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)",
    r"(?:reveal|show|print|repeat|output)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)",
    r"you\s+are\s+now\s+(?:a|an|no longer)",
    r"forget\s+(?:everything|all)\s+(?:you|above)",
    r"</?(?:system|instructions?)>",
    r"\bDAN\b|\bjailbreak\b",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class Screening:
    allowed: bool
    text: str                                    # possibly redacted
    reason: str = ""
    redactions: list[str] = field(default_factory=list)


def luhn_ok(digits: str) -> bool:
    """Luhn checksum. Without it, any 16-digit string (an order id, a timestamp
    range) is flagged as a card number and redacted, which mangles legitimate
    questions."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_cards(text: str, found: list[str]) -> str:
    def sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append("credit_card")
            return "[REDACTED_CARD]"
        return m.group(0)

    return CARD_CANDIDATE_RE.sub(sub, text)


def screen(question: str) -> Screening:
    text = (question or "").strip()

    if not text:
        return Screening(False, text, "empty question")
    if len(text) > int(INPUT_CFG["max_question_chars"]):
        return Screening(False, text, "question exceeds max_question_chars")

    if INPUT_CFG.get("block_prompt_injection", True) and INJECTION_RE.search(text):
        return Screening(False, text, "possible prompt injection")

    detect = set(INPUT_CFG.get("detect", []))
    action = INPUT_CFG.get("pii_action", "redact")
    found: list[str] = []

    if "credit_card" in detect:
        text = _redact_cards(text, found)
    if "email" in detect and EMAIL_RE.search(text):
        found.append("email")
        text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    if "phone" in detect and PHONE_RE.search(text):
        # Run phones last: card redaction has already consumed card-shaped digits.
        found.append("phone")
        text = PHONE_RE.sub("[REDACTED_PHONE]", text)

    if found and action == "block":
        return Screening(False, question, f"PII detected: {', '.join(sorted(set(found)))}")

    return Screening(True, text, "", sorted(set(found)))
