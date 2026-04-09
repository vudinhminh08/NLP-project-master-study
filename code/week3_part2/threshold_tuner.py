"""
threshold_tuner.py — Per-aspect threshold tuning cho ABSA.

Ý tưởng:
    Thay vì argmax 4-class trực tiếp, tách thành 2 stage:
    Stage 1 (ACD): non_absent_prob = 1 - softmax[0] > threshold? → present/absent
    Stage 2 (SPC): nếu present → argmax(softmax[1:4]) → sentiment

    Tune threshold per-aspect trên dev set → apply lên test set.

Input: raw logits từ PhoBERT (34 heads × 4 classes)
Output: optimized predictions [N, 34]
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))

from utils.constants import ASPECT_COLUMNS, ZERO_TRAIN_ASPECTS  # noqa: E402
from utils.helpers import save_json  # noqa: E402
from step4_eval import evaluate_predictions  # noqa: E402


THRESHOLD_GRID = np.round(np.arange(0.05, 1.00, 0.05), 2)


def collect_logits(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chạy inference và thu raw logits + ground truth labels.

    Returns:
        logits_all: np.ndarray [N, 34, 4]
        y_true:     np.ndarray [N, 34]
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_logits = torch.stack(outputs["logits"], dim=1)  # [B, 34, 4]

            all_logits.append(batch_logits.detach().cpu().numpy())
            all_labels.append(labels)

    logits_all = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return logits_all, y_true


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Stable softmax cho logits [N, 34, 4]."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_with_thresholds(
    probs: np.ndarray,
    thresholds: dict[str, float],
    aspect_columns: list[str] = ASPECT_COLUMNS,
) -> np.ndarray:
    """
    Áp dụng stage-1 ACD threshold + stage-2 SPC argmax.

    Nếu aspect không có threshold thì fallback về argmax 4-class.
    """
    y_pred = probs.argmax(axis=-1).astype(np.int64)

    for aspect_idx, aspect_name in enumerate(aspect_columns):
        if aspect_name not in thresholds:
            continue
        threshold = thresholds[aspect_name]
        non_absent_prob = 1.0 - probs[:, aspect_idx, 0]
        present_mask = non_absent_prob > threshold
        sentiment_pred = probs[:, aspect_idx, 1:4].argmax(axis=-1) + 1

        y_pred[:, aspect_idx] = 0
        y_pred[present_mask, aspect_idx] = sentiment_pred[present_mask]

    return y_pred


def _aspect_combined_f1(y_true_aspect: np.ndarray, y_pred_aspect: np.ndarray) -> dict:
    """
    Combined F1 cho một aspect duy nhất.

    ACD: binary present/absent
    SPC: macro 3-class trên subset y_true != 0
    """
    y_true_bin = (y_true_aspect != 0).astype(int)
    y_pred_bin = (y_pred_aspect != 0).astype(int)
    acd_f1 = f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)

    present_mask = y_true_aspect != 0
    if present_mask.sum() == 0:
        spc_f1 = 0.0
    else:
        spc_f1 = f1_score(
            y_true_aspect[present_mask],
            y_pred_aspect[present_mask],
            average="macro",
            labels=[1, 2, 3],
            zero_division=0,
        )

    return {
        "acd_f1": float(acd_f1),
        "spc_f1": float(spc_f1),
        "combined_f1": float((acd_f1 + spc_f1) / 2.0),
        "support": int(present_mask.sum()),
    }


def find_best_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    aspect_columns: list[str] = ASPECT_COLUMNS,
    exclude_aspects: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Tune threshold riêng cho từng aspect trên dev set.

    Nếu threshold tốt nhất không beat argmax baseline của chính aspect đó,
    aspect sẽ giữ nguyên argmax.
    """
    exclude_set = set(exclude_aspects or [])
    results: dict[str, dict] = {}
    argmax_pred = probs.argmax(axis=-1).astype(np.int64)

    for aspect_idx, aspect_name in enumerate(aspect_columns):
        y_true_aspect = y_true[:, aspect_idx]

        if aspect_name in exclude_set:
            results[aspect_name] = {
                "use_argmax": True,
                "reason": "excluded_aspect",
                "threshold": None,
                "dev_combined_f1": None,
                "baseline_combined_f1": None,
                "support": int((y_true_aspect != 0).sum()),
            }
            continue

        baseline_stats = _aspect_combined_f1(y_true_aspect, argmax_pred[:, aspect_idx])
        best_threshold = None
        best_stats = baseline_stats.copy()
        best_stats["threshold"] = None
        best_stats["use_argmax"] = True

        non_absent_prob = 1.0 - probs[:, aspect_idx, 0]
        sentiment_pred = probs[:, aspect_idx, 1:4].argmax(axis=-1) + 1

        for threshold in THRESHOLD_GRID:
            present_mask = non_absent_prob > threshold
            y_pred_aspect = np.zeros_like(y_true_aspect)
            y_pred_aspect[present_mask] = sentiment_pred[present_mask]
            stats = _aspect_combined_f1(y_true_aspect, y_pred_aspect)
            if stats["combined_f1"] > best_stats["combined_f1"] + 1e-12:
                best_threshold = float(threshold)
                best_stats = stats
                best_stats["threshold"] = best_threshold
                best_stats["use_argmax"] = False

        results[aspect_name] = {
            "use_argmax": best_stats["use_argmax"],
            "threshold": best_threshold,
            "dev_combined_f1": best_stats["combined_f1"],
            "dev_acd_f1": best_stats["acd_f1"],
            "dev_spc_f1": best_stats["spc_f1"],
            "baseline_combined_f1": baseline_stats["combined_f1"],
            "baseline_acd_f1": baseline_stats["acd_f1"],
            "baseline_spc_f1": baseline_stats["spc_f1"],
            "improvement": best_stats["combined_f1"] - baseline_stats["combined_f1"],
            "support": best_stats["support"],
        }

    return results


def apply_thresholds(
    probs: np.ndarray,
    thresholds: dict[str, dict],
    aspect_columns: list[str] = ASPECT_COLUMNS,
) -> np.ndarray:
    """
    Apply tuned thresholds lên probabilities.

    Aspects với `use_argmax=True` hoặc threshold=None sẽ giữ nguyên argmax 4-class.
    """
    threshold_map = {
        aspect: cfg["threshold"]
        for aspect, cfg in thresholds.items()
        if cfg.get("threshold") is not None and not cfg.get("use_argmax", False)
    }
    return predict_with_thresholds(probs, threshold_map, aspect_columns=aspect_columns)


def _summarize_thresholds(thresholds: dict[str, dict]) -> dict:
    """Tạo summary ngắn gọn để log/save."""
    improved = {
        aspect: cfg for aspect, cfg in thresholds.items()
        if not cfg.get("use_argmax", False) and cfg.get("threshold") is not None
    }
    return {
        "num_aspects_total": len(thresholds),
        "num_aspects_tuned": len(improved),
        "tuned_aspects": sorted(improved.keys()),
    }


def run_threshold_tuning(
    model: torch.nn.Module,
    dev_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: Optional[list],
    save_dir: str,
    aspect_columns: list[str] = ASPECT_COLUMNS,
    exclude_aspects: Optional[list[str]] = None,
) -> dict:
    """
    Pipeline đầy đủ:
    1. Collect logits trên dev
    2. Tune per-aspect thresholds trên dev
    3. Collect logits trên test
    4. Apply thresholds lên test
    5. Evaluate baseline argmax vs tuned
    6. Save JSON outputs
    """
    del class_weights  # không cần cho inference raw logits
    exclude_aspects = exclude_aspects or ZERO_TRAIN_ASPECTS
    os.makedirs(save_dir, exist_ok=True)

    dev_logits, y_dev = collect_logits(model, dev_loader, device)
    dev_probs = logits_to_probs(dev_logits)
    dev_argmax = dev_probs.argmax(axis=-1).astype(np.int64)

    thresholds = find_best_thresholds(
        dev_probs,
        y_dev,
        aspect_columns=aspect_columns,
        exclude_aspects=exclude_aspects,
    )
    dev_tuned = apply_thresholds(dev_probs, thresholds, aspect_columns=aspect_columns)

    test_logits, y_test = collect_logits(model, test_loader, device)
    test_probs = logits_to_probs(test_logits)
    test_argmax = test_probs.argmax(axis=-1).astype(np.int64)
    test_tuned = apply_thresholds(test_probs, thresholds, aspect_columns=aspect_columns)

    dev_argmax_metrics = evaluate_predictions(
        y_dev,
        dev_argmax,
        title="Dev — Argmax Baseline",
        exclude_aspects=exclude_aspects,
        save_path=os.path.join(save_dir, "dev_argmax_metrics.json"),
    )
    dev_tuned_metrics = evaluate_predictions(
        y_dev,
        dev_tuned,
        title="Dev — Threshold Tuned",
        exclude_aspects=exclude_aspects,
        save_path=os.path.join(save_dir, "dev_threshold_tuned_metrics.json"),
    )
    test_argmax_metrics = evaluate_predictions(
        y_test,
        test_argmax,
        title="Test — Argmax Baseline",
        exclude_aspects=exclude_aspects,
        save_path=os.path.join(save_dir, "test_argmax_metrics.json"),
    )
    test_tuned_metrics = evaluate_predictions(
        y_test,
        test_tuned,
        title="Test — Threshold Tuned",
        exclude_aspects=exclude_aspects,
        save_path=os.path.join(save_dir, "test_threshold_tuned_metrics.json"),
    )

    summary = {
        "threshold_summary": _summarize_thresholds(thresholds),
        "dev": {
            "argmax_combined_f1": dev_argmax_metrics["macro_combined_f1"],
            "threshold_tuned_combined_f1": dev_tuned_metrics["macro_combined_f1"],
            "improvement": (
                dev_tuned_metrics["macro_combined_f1"]
                - dev_argmax_metrics["macro_combined_f1"]
            ),
        },
        "test": {
            "argmax_combined_f1": test_argmax_metrics["macro_combined_f1"],
            "threshold_tuned_combined_f1": test_tuned_metrics["macro_combined_f1"],
            "improvement": (
                test_tuned_metrics["macro_combined_f1"]
                - test_argmax_metrics["macro_combined_f1"]
            ),
        },
    }

    save_json(thresholds, os.path.join(save_dir, "per_aspect_thresholds.json"))
    save_json(summary, os.path.join(save_dir, "threshold_tuning_summary.json"))
    save_json(
        {
            "argmax": {
                "combined_f1": test_argmax_metrics["macro_combined_f1"],
                "acd_f1": test_argmax_metrics["macro_acd_f1"],
                "spc_f1": test_argmax_metrics["macro_spc_f1"],
            },
            "threshold_tuned": {
                "combined_f1": test_tuned_metrics["macro_combined_f1"],
                "acd_f1": test_tuned_metrics["macro_acd_f1"],
                "spc_f1": test_tuned_metrics["macro_spc_f1"],
            },
        },
        os.path.join(save_dir, "test_comparison.json"),
    )

    return {
        "dev_logits": dev_logits,
        "dev_probs": dev_probs,
        "y_dev": y_dev,
        "dev_argmax": dev_argmax,
        "dev_tuned": dev_tuned,
        "test_logits": test_logits,
        "test_probs": test_probs,
        "y_test": y_test,
        "test_argmax": test_argmax,
        "test_tuned": test_tuned,
        "thresholds": thresholds,
        "summary": summary,
        "dev_argmax_metrics": dev_argmax_metrics,
        "dev_tuned_metrics": dev_tuned_metrics,
        "test_argmax_metrics": test_argmax_metrics,
        "test_tuned_metrics": test_tuned_metrics,
    }
