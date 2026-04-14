"""
evidence_checker.py — Lightweight evidence validation for explanations.
"""

from __future__ import annotations

import re
import string


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evidence_in_review(evidence: str, review: str) -> bool:
    """Return True if normalized evidence appears in normalized review."""
    evidence_norm = normalize_text(evidence)
    if not evidence_norm:
        return False
    return evidence_norm in normalize_text(review)


def validate_explanation_items(
    review: str,
    predictions: list[dict],
    items: list[dict],
) -> tuple[list[dict], dict]:
    """
    Enforce explanation guardrails.

    - Drop aspects not predicted by PhoBERT.
    - Restore sentiment if the LLM changed it.
    - Mark invalid evidence as uncertain.
    """
    pred_by_aspect = {p["aspect"]: p for p in predictions}
    valid_items = []
    dropped = 0
    sentiment_restored = 0
    invalid_evidence = 0

    for item in items:
        aspect = str(item.get("aspect", "")).strip()
        if aspect not in pred_by_aspect:
            dropped += 1
            continue

        expected_sentiment = pred_by_aspect[aspect]["sentiment"]
        if item.get("sentiment") != expected_sentiment:
            item["sentiment"] = expected_sentiment
            sentiment_restored += 1

        evidence = str(item.get("evidence", "")).strip()
        if evidence and not evidence_in_review(evidence, review):
            item["evidence"] = ""
            item["evidence_uncertain"] = True
            item["evidence_confidence"] = "low"
            invalid_evidence += 1

        item.setdefault("evidence_uncertain", not bool(item.get("evidence")))
        item.setdefault("evidence_confidence", "low" if item["evidence_uncertain"] else "medium")
        item.setdefault("explanation", "")
        valid_items.append(item)

    return valid_items, {
        "dropped_items": dropped,
        "sentiment_restored": sentiment_restored,
        "invalid_evidence": invalid_evidence,
        "num_predictions": len(predictions),
        "num_items": len(valid_items),
    }
