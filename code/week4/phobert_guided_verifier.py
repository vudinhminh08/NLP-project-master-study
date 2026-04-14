"""
phobert_guided_verifier.py — PhoBERT-guided LLM verification.

Goal:
    Keep one trained PhoBERT checkpoint as the main classifier, then use an
    LLM only as a constrained verifier for uncertain cases. This avoids the
    Week 3 failure mode where the LLM had to solve all 34 aspects from scratch,
    and avoids the broad DELETE behavior that hurt ACD in cascade variants.

Default invariant:
    - LLM may ADD weak/rare aspects when PhoBERT predicts absent but assigns
      non-trivial present probability.
    - LLM may correct sentiment for present aspects with high SPC entropy.
    - LLM may NOT delete present aspects unless delete_enabled=True.
    - Any parse failure, invalid label, or missing evidence falls back to PhoBERT.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import (  # noqa: E402
    ASPECT_COLUMNS,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    RARE_ASPECTS,
    ZERO_TRAIN_ASPECTS,
)
from llm_client import LLMClient  # noqa: E402
from rag_retriever import ABSARetriever  # noqa: E402


WEAK_SPC_ASPECTS: list[str] = [
    "HOTEL#MISCELLANEOUS",
    "FACILITIES#COMFORT",
    "FACILITIES#GENERAL",
    "ROOMS#GENERAL",
    "HOTEL#QUALITY",
    "FACILITIES#CLEANLINESS",
    "ROOMS#COMFORT",
    "HOTEL#DESIGN&FEATURES",
    "FACILITIES#DESIGN&FEATURES",
]

ALL_LABELS = {"absent", "positive", "negative", "neutral"}


@dataclass
class GuidedVerifierConfig:
    """Runtime knobs for PhoBERT-guided verification."""

    add_threshold: float = 0.08
    spc_entropy_threshold: float = 0.75
    delete_enabled: bool = False
    delete_max_confidence: float = 0.55
    k_rag: int = 6
    max_candidates_per_review: int = 5
    sleep_sec: float = 0.3
    temperature: float = 0.0
    require_evidence: bool = True
    use_cache: bool = True
    weak_add_aspects: list[str] = field(
        default_factory=lambda: [
            aspect for aspect in RARE_ASPECTS if aspect not in ZERO_TRAIN_ASPECTS
        ]
    )
    weak_spc_aspects: list[str] = field(default_factory=lambda: list(WEAK_SPC_ASPECTS))


def compute_spc_entropy(probs_single: np.ndarray) -> np.ndarray:
    """
    Shannon entropy over normalized positive/negative/neutral probabilities.

    Args:
        probs_single: [34, 4] probabilities, label 0 is absent.
    Returns:
        [34] entropy values in [0, ln(3)].
    """
    eps = 1e-9
    spc_probs = probs_single[:, 1:4]
    spc_sum = spc_probs.sum(axis=1, keepdims=True).clip(min=eps)
    spc_norm = spc_probs / spc_sum
    return -(spc_norm * np.log(spc_norm + eps)).sum(axis=1)


def _top_sentiments(probs_aspect: np.ndarray, top_n: int = 2) -> list[dict]:
    """Return top sentiment labels from probs[1:4]."""
    sentiment_probs = probs_aspect[1:4]
    order = np.argsort(sentiment_probs)[::-1][:top_n]
    return [
        {
            "label": IDX_TO_LABEL[int(idx + 1)],
            "prob": round(float(sentiment_probs[idx]), 4),
        }
        for idx in order
    ]


def select_candidates_for_review(
    probs_single: np.ndarray,
    base_preds_single: np.ndarray,
    config: GuidedVerifierConfig,
) -> list[dict]:
    """
    Select uncertain aspect-level candidates for one review.

    The selector deliberately keeps the LLM surface small: weak aspects only,
    plus a per-review cap. This is the main guard against noisy broad LLM use.
    """
    entropy = compute_spc_entropy(probs_single)
    weak_add = set(config.weak_add_aspects)
    weak_spc = set(config.weak_spc_aspects)
    candidates: list[dict] = []

    for aspect_idx, aspect in enumerate(ASPECT_COLUMNS):
        pred_label_idx = int(base_preds_single[aspect_idx])
        pred_label = IDX_TO_LABEL[pred_label_idx]
        p_absent = float(probs_single[aspect_idx, 0])
        p_present = float(1.0 - p_absent)
        max_conf = float(probs_single[aspect_idx].max())
        top_sent = _top_sentiments(probs_single[aspect_idx], top_n=2)

        if pred_label_idx == 0 and aspect in weak_add and p_present >= config.add_threshold:
            candidates.append(
                {
                    "aspect_idx": aspect_idx,
                    "aspect": aspect,
                    "task": "verify_add",
                    "phobert_label": pred_label,
                    "p_absent": round(p_absent, 4),
                    "p_present": round(p_present, 4),
                    "max_conf": round(max_conf, 4),
                    "spc_entropy": round(float(entropy[aspect_idx]), 4),
                    "top_sentiments": top_sent,
                    "allowed_labels": ["absent", "positive", "negative", "neutral"],
                    "rank_score": p_present + 0.05,
                }
            )

        if (
            pred_label_idx != 0
            and aspect in weak_spc
            and float(entropy[aspect_idx]) >= config.spc_entropy_threshold
        ):
            # Keep the top-2 constraint, but always leave a path to recover
            # neutral because neutral is extremely rare and often under-ranked.
            allowed = [item["label"] for item in top_sent]
            if pred_label not in allowed:
                allowed.append(pred_label)
            if "neutral" not in allowed:
                allowed.append("neutral")
            candidates.append(
                {
                    "aspect_idx": aspect_idx,
                    "aspect": aspect,
                    "task": "verify_sentiment",
                    "phobert_label": pred_label,
                    "p_absent": round(p_absent, 4),
                    "p_present": round(p_present, 4),
                    "max_conf": round(max_conf, 4),
                    "spc_entropy": round(float(entropy[aspect_idx]), 4),
                    "top_sentiments": top_sent,
                    "allowed_labels": allowed,
                    "rank_score": float(entropy[aspect_idx]) / math.log(3),
                }
            )

        if (
            config.delete_enabled
            and pred_label_idx != 0
            and aspect in weak_spc
            and max_conf <= config.delete_max_confidence
        ):
            candidates.append(
                {
                    "aspect_idx": aspect_idx,
                    "aspect": aspect,
                    "task": "verify_delete",
                    "phobert_label": pred_label,
                    "p_absent": round(p_absent, 4),
                    "p_present": round(p_present, 4),
                    "max_conf": round(max_conf, 4),
                    "spc_entropy": round(float(entropy[aspect_idx]), 4),
                    "top_sentiments": top_sent,
                    "allowed_labels": ["absent", pred_label],
                    "rank_score": 1.0 - max_conf,
                }
            )

    deduped = {}
    for cand in candidates:
        key = (cand["aspect"], cand["task"])
        if key not in deduped or cand["rank_score"] > deduped[key]["rank_score"]:
            deduped[key] = cand

    selected = sorted(deduped.values(), key=lambda x: x["rank_score"], reverse=True)
    return selected[: config.max_candidates_per_review]


def _retrieve_aspect_examples(
    processed_text: str,
    aspect: str,
    train_df: pd.DataFrame,
    retriever: ABSARetriever,
    k: int,
) -> list[dict]:
    """Retrieve semantically close examples that contain the target aspect."""
    indices = retriever.retrieve(
        processed_text,
        k=max(k * 8, k),
        aspect_aware=False,
        candidate_pool=max(50, k * 10),
    )
    filtered = [idx for idx in indices if int(train_df.iloc[idx][aspect]) != 0]
    pool = filtered[:k] if filtered else indices[:k]

    examples = []
    for idx in pool:
        row = train_df.iloc[idx]
        label_idx = int(row[aspect])
        examples.append(
            {
                "aspect": aspect,
                "review": str(row.get("Review", row.get("processed_review", ""))),
                "label": IDX_TO_LABEL.get(label_idx, "absent"),
            }
        )
    return examples


def build_guided_verifier_prompt(
    review_text: str,
    processed_text: str,
    candidates: list[dict],
    train_df: pd.DataFrame,
    retriever: ABSARetriever,
    config: GuidedVerifierConfig,
) -> list[dict]:
    """
    Build one compact verifier prompt for all candidates in a review.

    The prompt exposes PhoBERT confidence and restricts valid labels per
    candidate. This turns the LLM into a constrained verifier/reranker.
    """
    examples_by_aspect = {}
    for cand in candidates:
        aspect = cand["aspect"]
        if aspect not in examples_by_aspect:
            examples_by_aspect[aspect] = _retrieve_aspect_examples(
                processed_text=processed_text,
                aspect=aspect,
                train_df=train_df,
                retriever=retriever,
                k=config.k_rag,
            )

    slim_candidates = [
        {
            "aspect": cand["aspect"],
            "task": cand["task"],
            "phobert_label": cand["phobert_label"],
            "p_present": cand["p_present"],
            "max_conf": cand["max_conf"],
            "spc_entropy": cand["spc_entropy"],
            "top_sentiments": cand["top_sentiments"],
            "allowed_labels": cand["allowed_labels"],
        }
        for cand in candidates
    ]

    system_msg = (
        "Bạn là bộ xác minh nhãn ABSA cho review khách sạn tiếng Việt. "
        "PhoBERT đã sinh candidate và xác suất; bạn chỉ được xác minh các "
        "candidate này, không được thêm aspect ngoài danh sách. "
        "Evidence phải là cụm từ xuất hiện trong review gốc. "
        "Chỉ trả JSON hợp lệ."
    )
    user_msg = {
        "review": review_text,
        "processed_review": processed_text,
        "instructions": [
            "Với task verify_add: chọn absent nếu review không nói rõ aspect; nếu chọn positive/negative/neutral thì evidence bắt buộc không rỗng.",
            "Với task verify_sentiment: aspect đã được xem là present; chỉ chọn label trong allowed_labels.",
            "Không suy diễn từ kiến thức ngoài review.",
            "Không trả lời markdown, chỉ JSON.",
        ],
        "phobert_candidates": slim_candidates,
        "rag_examples_by_aspect": examples_by_aspect,
        "output_schema": {
            "decisions": [
                {
                    "aspect": "ASPECT#CATEGORY",
                    "label": "absent|positive|negative|neutral",
                    "evidence": "short phrase from review, empty only for absent",
                    "confidence": "low|medium|high",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
    ]


def _extract_json_object(raw_text: str) -> Optional[dict]:
    """Parse JSON, including common markdown-fenced responses."""
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


def parse_verifier_output(raw_text: str, candidates: list[dict]) -> tuple[list[dict], dict]:
    """
    Parse and validate LLM decisions against the original candidate list.
    """
    candidate_by_aspect = {cand["aspect"]: cand for cand in candidates}
    data = _extract_json_object(raw_text)
    if data is None:
        return [], {"parse_fail": 1, "invalid_items": 0}

    raw_decisions = data.get("decisions", [])
    if isinstance(raw_decisions, dict):
        raw_decisions = [raw_decisions]
    if not isinstance(raw_decisions, list):
        return [], {"parse_fail": 1, "invalid_items": 0}

    decisions = []
    invalid_items = 0
    for item in raw_decisions:
        if not isinstance(item, dict):
            invalid_items += 1
            continue
        aspect = str(item.get("aspect", "")).strip()
        label = str(item.get("label", "")).strip().lower()
        evidence = str(item.get("evidence", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()

        if aspect not in candidate_by_aspect or label not in ALL_LABELS:
            invalid_items += 1
            continue
        if label not in set(candidate_by_aspect[aspect]["allowed_labels"]):
            invalid_items += 1
            continue

        decisions.append(
            {
                "aspect": aspect,
                "label": label,
                "evidence": evidence,
                "confidence": confidence,
            }
        )

    return decisions, {"parse_fail": 0, "invalid_items": invalid_items}


def apply_guided_decisions(
    base_preds_single: np.ndarray,
    candidates: list[dict],
    decisions: list[dict],
    config: GuidedVerifierConfig,
) -> tuple[np.ndarray, dict]:
    """
    Apply guarded LLM decisions.

    Evidence guard is intentionally strict because the target is higher test F1,
    not more LLM activity.
    """
    final_preds = base_preds_single.copy()
    candidate_by_aspect = {cand["aspect"]: cand for cand in candidates}
    applied = []
    skipped = []

    for dec in decisions:
        aspect = dec["aspect"]
        cand = candidate_by_aspect.get(aspect)
        if cand is None:
            skipped.append({"aspect": aspect, "reason": "unknown_aspect"})
            continue

        label = dec["label"]
        evidence = dec.get("evidence", "")
        if config.require_evidence and label != "absent" and len(evidence) < 2:
            skipped.append({"aspect": aspect, "reason": "missing_evidence", "label": label})
            continue

        idx = cand["aspect_idx"]
        old_label_idx = int(final_preds[idx])
        old_label = IDX_TO_LABEL[old_label_idx]

        if cand["task"] == "verify_add":
            if old_label_idx != 0 or label == "absent":
                skipped.append({"aspect": aspect, "reason": "no_add", "label": label})
                continue
            final_preds[idx] = LABEL_TO_IDX[label]
        elif cand["task"] == "verify_sentiment":
            if old_label_idx == 0 or label == "absent":
                skipped.append({"aspect": aspect, "reason": "invalid_spc_label", "label": label})
                continue
            final_preds[idx] = LABEL_TO_IDX[label]
        elif cand["task"] == "verify_delete":
            if not config.delete_enabled or label != "absent":
                skipped.append({"aspect": aspect, "reason": "delete_disabled_or_rejected"})
                continue
            final_preds[idx] = 0
        else:
            skipped.append({"aspect": aspect, "reason": "unknown_task"})
            continue

        new_label = IDX_TO_LABEL[int(final_preds[idx])]
        applied.append(
            {
                "aspect": aspect,
                "task": cand["task"],
                "old": old_label,
                "new": new_label,
                "changed": old_label != new_label,
                "evidence": evidence,
            }
        )

    return final_preds, {"applied": applied, "skipped": skipped}


def guided_verify_single(
    review_text: str,
    processed_text: str,
    probs_single: np.ndarray,
    base_preds_single: np.ndarray,
    train_df: pd.DataFrame,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    config: GuidedVerifierConfig,
) -> tuple[np.ndarray, bool, dict]:
    """Run PhoBERT-guided verification for one review."""
    candidates = select_candidates_for_review(
        probs_single=probs_single,
        base_preds_single=base_preds_single,
        config=config,
    )
    if not candidates:
        return base_preds_single.copy(), False, {
            "candidate_count": 0,
            "applied": [],
            "skipped": [],
            "parse_fail": 0,
            "invalid_items": 0,
        }

    try:
        messages = build_guided_verifier_prompt(
            review_text=review_text,
            processed_text=processed_text,
            candidates=candidates,
            train_df=train_df,
            retriever=retriever,
            config=config,
        )
        raw = llm_client.complete(
            messages,
            temperature=config.temperature,
            use_cache=config.use_cache,
        )
        decisions, parse_stats = parse_verifier_output(raw, candidates)
        final_pred, apply_stats = apply_guided_decisions(
            base_preds_single=base_preds_single,
            candidates=candidates,
            decisions=decisions,
            config=config,
        )
    except Exception as exc:
        return base_preds_single.copy(), False, {
            "candidate_count": len(candidates),
            "candidate_aspects": [c["aspect"] for c in candidates],
            "applied": [],
            "skipped": [],
            "parse_fail": 1,
            "invalid_items": 0,
            "error": str(exc)[:200],
        }

    debug = {
        "candidate_count": len(candidates),
        "candidate_aspects": [c["aspect"] for c in candidates],
        "candidate_tasks": [c["task"] for c in candidates],
        "applied": apply_stats["applied"],
        "skipped": apply_stats["skipped"],
        "parse_fail": parse_stats["parse_fail"],
        "invalid_items": parse_stats["invalid_items"],
        "raw_output_preview": raw[:240] if raw else "",
    }
    return final_pred, True, debug


def run_guided_verifier_on_dataset(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    y_true: np.ndarray,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    config: Optional[GuidedVerifierConfig] = None,
    max_samples: Optional[int] = None,
    return_records: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run verifier over a dataset using precomputed PhoBERT probabilities.

    Returns:
        y_true_out, y_pred_guided, stats
    """
    config = config or GuidedVerifierConfig()
    n_total = min(len(test_df), len(all_probs), len(all_preds), len(y_true))
    n = min(n_total, max_samples) if max_samples is not None else n_total

    final_preds = all_preds[:n].copy()
    llm_call_count = 0
    total_candidates = 0
    total_applied = 0
    total_changed = 0
    add_overrides = 0
    spc_overrides = 0
    delete_overrides = 0
    parse_fails = 0
    invalid_items = 0
    routed_reviews = 0
    sample_records = []
    task_counts = {"verify_add": 0, "verify_sentiment": 0, "verify_delete": 0}
    aspect_override_counts = {aspect: 0 for aspect in ASPECT_COLUMNS}

    for i in range(n):
        row = test_df.iloc[i]
        review_text = str(row.get("Review", ""))
        processed_text = str(row.get("processed_review", review_text))

        pred_i, used_llm, debug = guided_verify_single(
            review_text=review_text,
            processed_text=processed_text,
            probs_single=all_probs[i],
            base_preds_single=all_preds[i],
            train_df=train_df,
            retriever=retriever,
            llm_client=llm_client,
            config=config,
        )
        final_preds[i] = pred_i

        candidate_count = int(debug.get("candidate_count", 0))
        total_candidates += candidate_count
        parse_fails += int(debug.get("parse_fail", 0))
        invalid_items += int(debug.get("invalid_items", 0))
        for task in debug.get("candidate_tasks", []):
            if task in task_counts:
                task_counts[task] += 1

        if used_llm:
            llm_call_count += 1
            routed_reviews += 1
            if config.sleep_sec > 0:
                time.sleep(config.sleep_sec)

        applied = debug.get("applied", [])
        total_applied += len(applied)
        for item in applied:
            if item.get("changed"):
                total_changed += 1
            task = item.get("task")
            aspect = item.get("aspect")
            if task == "verify_add":
                add_overrides += 1
            elif task == "verify_sentiment":
                spc_overrides += 1
            elif task == "verify_delete":
                delete_overrides += 1
            if aspect in aspect_override_counts:
                aspect_override_counts[aspect] += 1

        if return_records:
            sample_records.append(
                {
                    "sample_idx": i,
                    "used_llm": used_llm,
                    "candidate_count": candidate_count,
                    "candidate_aspects": debug.get("candidate_aspects", []),
                    "candidate_tasks": debug.get("candidate_tasks", []),
                    "applied": applied,
                    "skipped": debug.get("skipped", []),
                    "error": debug.get("error"),
                    "raw_output_preview": debug.get("raw_output_preview", ""),
                }
            )

        if (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{n}] "
                f"LLM calls: {llm_call_count} "
                f"({llm_call_count / max(1, i + 1):.0%}) "
                f"| candidates: {total_candidates} "
                f"| changed: {total_changed} "
                f"| parse_fail: {parse_fails}"
            )

    stats = {
        "config": asdict(config),
        "total_samples": n,
        "routed_reviews": routed_reviews,
        "routing_rate": routed_reviews / n if n else 0.0,
        "llm_call_count": llm_call_count,
        "total_candidates": total_candidates,
        "avg_candidates_per_routed_review": (
            total_candidates / routed_reviews if routed_reviews else 0.0
        ),
        "total_applied": total_applied,
        "total_changed": total_changed,
        "add_overrides": add_overrides,
        "spc_overrides": spc_overrides,
        "delete_overrides": delete_overrides,
        "parse_fails": parse_fails,
        "parse_fail_rate": parse_fails / llm_call_count if llm_call_count else 0.0,
        "invalid_items": invalid_items,
        "candidate_task_counts": task_counts,
        "aspect_override_counts": {
            aspect: count for aspect, count in aspect_override_counts.items() if count > 0
        },
    }
    if return_records:
        stats["sample_records"] = sample_records

    return y_true[:n], final_preds, stats
