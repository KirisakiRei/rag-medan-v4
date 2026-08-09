"""Validation helpers for AI-generated document and web answers."""

import re
from typing import Tuple


_NOT_FOUND_RE = re.compile(r"^tidak ditemukan(?:[.!]|\s|$)", re.IGNORECASE)
_PROMPT_LEAKAGE_PATTERNS = (
    re.compile(r"\brule\s+\d+\s*:", re.IGNORECASE),
    re.compile(r"\bif\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\baturan\s+ketat\b", re.IGNORECASE),
    re.compile(r"\breferensi\s+teks\b", re.IGNORECASE),
)


def validate_extracted_answer(answer: str) -> Tuple[bool, str]:
    """Return whether an AI extraction can safely be sent as the final answer."""
    normalized_answer = (answer or "").strip()

    if not normalized_answer:
        return False, "empty"

    if _NOT_FOUND_RE.match(normalized_answer):
        return False, "not_found"

    if any(pattern.search(normalized_answer) for pattern in _PROMPT_LEAKAGE_PATTERNS):
        return False, "prompt_leakage"

    return True, ""
