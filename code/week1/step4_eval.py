"""
step4_eval.py — Evaluation module dùng chung cho cả 3 tầng (tuần 2, 3).

Nguyên tắc tính F1 cho ABSA (theo Huynh et al. 2022 SOTA):
- ACD F1: binary per-aspect (present vs absent), macro average
- SPC F1: 3-class macro trên subset y_true != 0
- Combined F1: average của ACD và SPC

Chạy từ root project:
    python code/week1/step4_eval.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.constants import ASPECT_COLUMNS, RARE_ASPECTS, ZERO_TRAIN_ASPECTS
from utils.helpers import save_json, format_metrics_table


# ─── Core metric computation ─────────────────────────────────────────────────

def compute_aspect_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aspect_columns: list = ASPECT_COLUMNS,
    exclude_aspects: list = None,
) -> dict:
    """
    Tính đầy đủ metrics cho ABSA.

    Args:
        y_true:          shape [N, 34], values 0-3
        y_pred:          shape [N, 34], values 0-3
        aspect_columns:  danh sách 34 aspect names
        exclude_aspects: list aspect names loại khỏi Macro-F1 khi tính tổng.
                         Vẫn tính per-aspect F1, chỉ exclude khi average.
                         Dùng cho ROOM_AMENITIES#PRICES (0 mẫu trong train).

    Returns:
        dict với:
            per_aspect:         dict aspect → {acd_f1, acd_precision, acd_recall,
                                               spc_f1, spc_precision, spc_recall, support}
            macro_acd_f1:       float  (exclude_aspects đã loại)
            macro_spc_f1:       float  (exclude_aspects đã loại)
            macro_combined_f1:  float
            weighted_acd_f1:    float
            excluded_aspects:   list
            num_aspects:        int

    Quan trọng:
        - ACD: binary per aspect (0 vs 1/2/3)
        - SPC: 3-class trên subset y_true != 0 (bỏ absent samples)
        - zero_division=0 cho mọi sklearn calls
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    exclude_set: set = set(exclude_aspects or [])
    per_aspect: dict = {}
    acd_f1s: list = []
    spc_f1s: list = []
    valid_acd: list = []
    valid_spc: list = []

    for i, aspect in enumerate(aspect_columns):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        # ACD: binary (0 vs 1/2/3)
        yt_bin = (yt > 0).astype(int)
        yp_bin = (yp > 0).astype(int)
        acd_f1  = f1_score(yt_bin, yp_bin, average="binary", zero_division=0)
        acd_pre = precision_score(yt_bin, yp_bin, average="binary", zero_division=0)
        acd_rec = recall_score(yt_bin, yp_bin, average="binary", zero_division=0)
        support = int(yt_bin.sum())
        acd_f1s.append(acd_f1)
        if aspect not in exclude_set:
            valid_acd.append(acd_f1)

        # SPC: chỉ trên samples mà y_true != 0
        mask = (yt > 0)
        if mask.sum() == 0:
            spc_f1 = spc_pre = spc_rec = None
        else:
            yt_spc = yt[mask]  # values in {1, 2, 3}
            yp_spc = yp[mask]
            spc_f1  = f1_score(yt_spc, yp_spc, average="macro",
                               labels=[1, 2, 3], zero_division=0)
            spc_pre = precision_score(yt_spc, yp_spc, average="macro",
                                      labels=[1, 2, 3], zero_division=0)
            spc_rec = recall_score(yt_spc, yp_spc, average="macro",
                                   labels=[1, 2, 3], zero_division=0)
            spc_f1s.append(spc_f1)
            if aspect not in exclude_set:
                valid_spc.append(spc_f1)

        per_aspect[aspect] = {
            "acd_f1":        float(acd_f1),
            "acd_precision": float(acd_pre),
            "acd_recall":    float(acd_rec),
            "spc_f1":        float(spc_f1) if spc_f1 is not None else None,
            "spc_precision": float(spc_pre) if spc_pre is not None else None,
            "spc_recall":    float(spc_rec) if spc_rec is not None else None,
            "support":       support,
        }

    macro_acd_f1   = float(np.mean(valid_acd)) if valid_acd else 0.0
    macro_spc_f1   = float(np.mean(valid_spc)) if valid_spc else 0.0
    macro_combined = (macro_acd_f1 + macro_spc_f1) / 2

    # Weighted ACD F1 (weighted by support, dùng tất cả aspects)
    supports = np.array([per_aspect[a]["support"] for a in aspect_columns])
    if supports.sum() > 0:
        weights = supports / supports.sum()
    else:
        weights = np.ones(len(aspect_columns)) / len(aspect_columns)
    weighted_acd_f1 = float(np.average(acd_f1s, weights=weights))

    return {
        "per_aspect":          per_aspect,
        "macro_acd_f1":        macro_acd_f1,
        "macro_spc_f1":        macro_spc_f1,
        "macro_combined_f1":   macro_combined,
        "weighted_acd_f1":     weighted_acd_f1,
        "excluded_aspects":    list(exclude_set),
        "num_aspects":         len(aspect_columns),
    }


# ─── Print + save wrapper ────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Evaluation",
    save_path: str = None,
    exclude_aspects: list = None,
) -> dict:
    """
    Wrapper: compute + print + save.
    Gọi từ tuần 2 và 3.

    Args:
        y_true:          shape [N, 34], values 0-3
        y_pred:          shape [N, 34], values 0-3
        title:           tiêu đề báo cáo
        save_path:       nếu không None, lưu metrics JSON tại đây
        exclude_aspects: list aspect names loại khỏi Macro-F1
                         (mặc định dùng ZERO_TRAIN_ASPECTS nếu None)

    Returns:
        metrics dict từ compute_aspect_f1
    """
    if exclude_aspects is None:
        exclude_aspects = ZERO_TRAIN_ASPECTS
    metrics = compute_aspect_f1(y_true, y_pred, exclude_aspects=exclude_aspects)
    print_evaluation_report(metrics, title)
    if save_path:
        save_json(metrics, save_path)
        print(f"[Saved] {save_path}")
    return metrics


def print_evaluation_report(metrics: dict, title: str = "Evaluation Report") -> None:
    """In báo cáo đẹp ra terminal với ANSI color cho rare aspects."""
    RED   = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(format_metrics_table(
        metrics["per_aspect"],
        metrics["macro_acd_f1"],
        metrics["macro_spc_f1"],
        highlight_rare=RARE_ASPECTS,
    ))
    print(f"\n  Macro ACD F1:      {metrics['macro_acd_f1']:.4f}")
    print(f"  Macro SPC F1:      {metrics['macro_spc_f1']:.4f}")
    print(f"  Combined F1:       {metrics['macro_combined_f1']:.4f}")
    print(f"  Weighted ACD F1:   {metrics['weighted_acd_f1']:.4f}")
    print(f"{'='*70}\n")


# ─── Main (self-test) ────────────────────────────────────────────────────────

def main() -> None:
    print("Running eval metric self-tests...\n")

    # Test A: Perfect prediction = F1 1.0
    rng = np.random.default_rng(42)
    y = rng.integers(0, 4, (200, 34))
    m = compute_aspect_f1(y, y)
    assert abs(m["macro_acd_f1"] - 1.0) < 1e-6, \
        f"Expected macro_acd_f1=1.0, got {m['macro_acd_f1']}"
    assert abs(m["macro_spc_f1"] - 1.0) < 1e-6, \
        f"Expected macro_spc_f1=1.0, got {m['macro_spc_f1']}"
    print(f"[Test A] Perfect prediction:")
    print(f"  ACD F1 = {m['macro_acd_f1']:.4f} (expected 1.0) ✓")
    print(f"  SPC F1 = {m['macro_spc_f1']:.4f} (expected 1.0) ✓")

    # Test B: All-absent prediction = ACD F1 thấp
    y_true = rng.integers(0, 4, (200, 34))
    y_pred = np.zeros((200, 34), dtype=int)
    m2 = compute_aspect_f1(y_true, y_pred)
    assert m2["macro_acd_f1"] < 0.5, \
        f"All-absent pred phải có ACD F1 thấp, got {m2['macro_acd_f1']}"
    print(f"\n[Test B] All-absent prediction:")
    print(f"  ACD F1 = {m2['macro_acd_f1']:.4f} (expected < 0.5) ✓")

    # Test C: In full report
    print("\n[Test C] Full report on perfect prediction:")
    print_evaluation_report(m, title="Test — Perfect Prediction")

    print("✅ Eval metric tests PASSED")
    print(f"   Perfect pred: ACD={m['macro_acd_f1']:.4f}, SPC={m['macro_spc_f1']:.4f}")
    print(f"   All-absent:   ACD={m2['macro_acd_f1']:.4f}")


if __name__ == "__main__":
    main()
