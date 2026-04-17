
from __future__ import annotations

import json
import re
import time
from typing import Optional

try:
    from .explanation_prompts import build_explanation_prompt
    from .evidence_checker import validate_explanation_items
except ImportError:
    from explanation_prompts import build_explanation_prompt
    from evidence_checker import validate_explanation_items


def _extract_json(raw_text: str) -> Optional[dict]:
    if not raw_text:
        return None
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def explain_review(
    review: str,
    predictions: list[dict],
    llm_client,
    rag_examples: list[dict] | None = None,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> dict:
    if not predictions:
        return {
            "items": [],
            "overall_summary": "Không có aspect nào được PhoBERT phát hiện.",
            "recommended_action": None,
            "validation": {
                "skipped_llm": True,
                "parse_fail": 0,
            },
        }

    messages = build_explanation_prompt(review, predictions, rag_examples)
    raw = llm_client.complete(messages, temperature=temperature, use_cache=use_cache)
    parsed = _extract_json(raw)

    if parsed is None:
        return {
            "items": [],
            "overall_summary": "",
            "recommended_action": None,
            "raw_output_preview": (raw or "")[:300],
            "validation": {
                "skipped_llm": False,
                "parse_fail": 1,
            },
        }

    items = parsed.get("items", [])
    if not isinstance(items, list):
        items = []

    valid_items, validation = validate_explanation_items(review, predictions, items)
    validation["skipped_llm"] = False
    validation["parse_fail"] = 0

    return {
        "items": valid_items,
        "overall_summary": str(parsed.get("overall_summary", "")),
        "recommended_action": parsed.get("recommended_action"),
        "validation": validation,
    }


def explain_batch(
    records: list[dict],
    llm_client,
    rag_examples_by_idx: dict[int, list[dict]] | None = None,
    max_samples: int | None = None,
    sleep_sec: float = 0.5,
) -> tuple[list[dict], dict]:
    n = min(len(records), max_samples) if max_samples is not None else len(records)
    outputs = []
    parse_fails = 0
    invalid_evidence = 0
    sentiment_restored = 0
    skipped_empty = 0

    for i in range(n):
        record = records[i]
        examples = (rag_examples_by_idx or {}).get(record["review_idx"], [])
        explanation = explain_review(
            review=record["review"],
            predictions=record["predictions"],
            llm_client=llm_client,
            rag_examples=examples,
        )
        validation = explanation.get("validation", {})
        parse_fails += int(validation.get("parse_fail", 0))
        invalid_evidence += int(validation.get("invalid_evidence", 0))
        sentiment_restored += int(validation.get("sentiment_restored", 0))
        skipped_empty += int(validation.get("skipped_llm", False))

        outputs.append({**record, "explanation": explanation})

        if sleep_sec > 0 and i < n - 1:
            time.sleep(sleep_sec)

    stats = {
        "num_records": n,
        "parse_fails": parse_fails,
        "parse_fail_rate": parse_fails / n if n else 0.0,
        "invalid_evidence": invalid_evidence,
        "sentiment_restored": sentiment_restored,
        "skipped_empty_predictions": skipped_empty,
    }
    return outputs, stats
