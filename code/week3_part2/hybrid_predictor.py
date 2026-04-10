"""
hybrid_predictor.py — Hybrid Argmax + Cascade ADD-only predictor.

Kết hợp 3 chiến lược:
    1. Argmax base (ACD mạnh nhất, giữ nguyên)
    2. Cascade ADD-only (chỉ THÊM aspect, KHÔNG sửa/xóa prediction đã có)
    3. Dùng full ABSA prompt của week3 để tận dụng pipeline prompt/parser ổn định

Pipeline cho 1 review:
    Step 1: PhoBERT forward → lấy softmax probs [34, 4] và argmax preds [34]
    Step 2: Với mỗi WEAK_ASPECT mà argmax predict ABSENT (=0):
            → Nếu non_absent_prob = 1 - p(absent) > add_threshold thì cân nhắc gọi LLM
            → Gọi full ABSA prompt với RAG examples
            → Nếu LLM predict aspect đó non-absent → ADD
    Step 3: Giữ nguyên mọi aspect mà base argmax đã predict non-absent

Điểm khác cascade cũ:
    - CHỈ ADD, KHÔNG BAO GIỜ sửa prediction đã non-absent
    - Dùng full ABSA prompt đã ổn định thay vì rerank prompt
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import ASPECT_COLUMNS, WEAK_ASPECTS  # noqa: E402
from utils.helpers import save_json  # noqa: E402
from step4_eval import evaluate_predictions  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from rag_retriever import ABSARetriever  # noqa: E402
from icl_predictor import df_to_examples  # noqa: E402
from prompts import build_prompt, parse_llm_output, labels_dict_to_array  # noqa: E402


WEAK_ASPECT_INDICES = [
    ASPECT_COLUMNS.index(aspect)
    for aspect in WEAK_ASPECTS
    if aspect in ASPECT_COLUMNS
]


def collect_logits_and_preds(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chạy inference, thu thập softmax probabilities + argmax predictions + labels.

    Returns:
        all_probs: np.ndarray [N, 34, 4]
        all_preds: np.ndarray [N, 34]
        y_true:    np.ndarray [N, 34]
    """
    model.eval()
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]  # list of 34 tensors [batch, 4]
            logits_stacked = torch.stack(logits, dim=1)  # [batch, 34, 4]
            probs = F.softmax(logits_stacked, dim=-1)
            preds = logits_stacked.argmax(dim=-1)

            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.numpy())

    return (
        np.concatenate(probs_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


def hybrid_predict_single(
    review_text: str,
    processed_text: str,
    probs_single: np.ndarray,
    argmax_pred: np.ndarray,
    train_df: pd.DataFrame,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    weak_aspects: list[str],
    weak_indices: list[int],
    add_threshold: float = 0.10,
    k: int = 4,
) -> tuple[np.ndarray, bool, dict]:
    """
    Predict 1 review theo chiến lược ADD-only.

    Returns:
        final_pred: [34]
        used_llm: bool
        debug_info: dict
    """
    final_pred = argmax_pred.copy()
    candidates_to_add = []

    for idx, asp_name in zip(weak_indices, weak_aspects):
        if int(argmax_pred[idx]) == 0:
            non_absent_prob = float(1.0 - probs_single[idx, 0])
            if non_absent_prob > add_threshold:
                candidates_to_add.append(
                    {
                        "index": idx,
                        "aspect": asp_name,
                        "non_absent_prob": non_absent_prob,
                    }
                )

    if not candidates_to_add:
        return final_pred, False, {"candidates": 0, "added": []}

    try:
        indices = retriever.retrieve(query=processed_text, k=k, aspect_aware=True)
        examples = df_to_examples(train_df, indices)
        messages = build_prompt(processed_text, examples, include_cot=False)
        raw = llm_client.complete(messages, temperature=0.0, use_cache=True)
        llm_pred_dict = parse_llm_output(raw)
        llm_preds = np.array(labels_dict_to_array(llm_pred_dict), dtype=np.int64)
    except Exception as exc:
        return final_pred, False, {
            "candidates": len(candidates_to_add),
            "added": [],
            "error": str(exc),
        }

    added = []
    for cand in candidates_to_add:
        idx = cand["index"]
        llm_label = int(llm_preds[idx])
        if llm_label != 0:
            final_pred[idx] = llm_label
            added.append(
                {
                    "aspect": cand["aspect"],
                    "llm_label": llm_label,
                    "non_absent_prob": cand["non_absent_prob"],
                }
            )

    return final_pred, True, {
        "candidates": len(candidates_to_add),
        "candidate_aspects": [c["aspect"] for c in candidates_to_add],
        "added": added,
        "raw_output_preview": raw[:200] if "raw" in locals() else "",
    }


def run_hybrid_on_dataset(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    y_true: np.ndarray,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    weak_aspects: list[str],
    weak_indices: list[int],
    add_threshold: float = 0.10,
    k: int = 4,
    sleep_sec: float = 0.5,
    max_samples: Optional[int] = None,
    return_records: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Chạy hybrid predictor trên toàn bộ dataset.

    Returns:
        y_true_out, y_pred_hybrid, stats
    """
    n_total = min(len(test_df), len(all_probs), len(all_preds), len(y_true))
    n = min(n_total, max_samples) if max_samples is not None else n_total

    y_pred_hybrid = all_preds[:n].copy()
    llm_call_count = 0
    total_added = 0
    weak_trigger_counts = {aspect: 0 for aspect in weak_aspects}
    weak_add_counts = {aspect: 0 for aspect in weak_aspects}
    sample_records = []

    for i in range(n):
        row = test_df.iloc[i]
        review_text = str(row.get("Review", ""))
        processed = str(row.get("processed_review", review_text))

        pred, used_llm, debug = hybrid_predict_single(
            review_text=review_text,
            processed_text=processed,
            probs_single=all_probs[i],
            argmax_pred=all_preds[i],
            train_df=train_df,
            retriever=retriever,
            llm_client=llm_client,
            weak_aspects=weak_aspects,
            weak_indices=weak_indices,
            add_threshold=add_threshold,
            k=k,
        )
        y_pred_hybrid[i] = pred

        for aspect in debug.get("candidate_aspects", []):
            if aspect in weak_trigger_counts:
                weak_trigger_counts[aspect] += 1

        if used_llm:
            llm_call_count += 1
            total_added += len(debug.get("added", []))
            for item in debug.get("added", []):
                aspect = item["aspect"]
                if aspect in weak_add_counts:
                    weak_add_counts[aspect] += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        if return_records:
            sample_records.append(
                {
                    "sample_idx": i,
                    "used_llm": used_llm,
                    "candidate_aspects": debug.get("candidate_aspects", []),
                    "added_count": len(debug.get("added", [])),
                    "added": debug.get("added", []),
                    "error": debug.get("error"),
                }
            )

        if (i + 1) % 100 == 0:
            print(
                f"  [{i + 1}/{n}] "
                f"LLM calls: {llm_call_count} "
                f"({llm_call_count / max(1, i + 1):.0%}) "
                f"Total added: {total_added}"
            )

    stats = {
        "total_samples": n,
        "llm_call_count": llm_call_count,
        "trigger_rate": llm_call_count / n if n > 0 else 0.0,
        "total_added": total_added,
        "avg_added_per_trigger": total_added / llm_call_count if llm_call_count > 0 else 0.0,
        "add_threshold": add_threshold,
        "k": k,
        "weak_aspects": weak_aspects,
        "weak_trigger_counts": weak_trigger_counts,
        "weak_add_counts": weak_add_counts,
    }
    if return_records:
        stats["sample_records"] = sample_records

    return y_true[:n], y_pred_hybrid, stats


def grid_search_add_threshold(
    dev_df: pd.DataFrame,
    train_df: pd.DataFrame,
    all_probs_dev: np.ndarray,
    all_preds_dev: np.ndarray,
    y_true_dev: np.ndarray,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    weak_aspects: list[str],
    weak_indices: list[int],
    thresholds: list[float] | None = None,
    k: int = 4,
    sleep_sec: float = 0.3,
    max_samples: int = 200,
    exclude_aspects: Optional[list[str]] = None,
) -> tuple[float, dict]:
    """
    Tune add_threshold trên dev subset.

    Returns:
        best_threshold, results_per_threshold
    """
    thresholds = thresholds or [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    results = {}

    for t in thresholds:
        print(f"\n--- add_threshold = {t} ---")
        y_true_out, y_pred, stats = run_hybrid_on_dataset(
            dev_df,
            train_df,
            all_probs_dev,
            all_preds_dev,
            y_true_dev,
            retriever,
            llm_client,
            weak_aspects,
            weak_indices,
            add_threshold=t,
            k=k,
            sleep_sec=sleep_sec,
            max_samples=max_samples,
            return_records=False,
        )
        metrics = evaluate_predictions(
            y_true_out,
            y_pred,
            title=f"Dev Hybrid ADD-only (t={t})",
            exclude_aspects=exclude_aspects,
        )
        results[t] = {
            "combined_f1": metrics["macro_combined_f1"],
            "acd_f1": metrics["macro_acd_f1"],
            "spc_f1": metrics["macro_spc_f1"],
            "llm_calls": stats["llm_call_count"],
            "total_added": stats["total_added"],
        }
        print(
            f"  Combined F1: {metrics['macro_combined_f1']:.4f}, "
            f"Added: {stats['total_added']}, "
            f"LLM calls: {stats['llm_call_count']}"
        )

    best_t = max(results, key=lambda t: results[t]["combined_f1"])
    print(f"\nBest threshold: {best_t} → Combined F1: {results[best_t]['combined_f1']:.4f}")
    return float(best_t), results


def save_hybrid_outputs(
    metrics: dict,
    stats: dict,
    save_dir: str,
    prefix: str = "hybrid_add_only",
) -> None:
    """Save JSON outputs cho notebook/script tiện dùng lại."""
    os.makedirs(save_dir, exist_ok=True)
    save_json(metrics, os.path.join(save_dir, f"{prefix}_metrics.json"))
    save_json(stats, os.path.join(save_dir, f"{prefix}_stats.json"))
