"""
ensemble.py — Ensemble + Threshold Tuning cho ABSA VLSP 2018.

Pipeline:
    1. Load 2 model checkpoints (Gốc cls_only + v2.4 cls_only)
    2. Collect raw logits [N, 34, 4] từ mỗi model trên dev và test set
    3. Average logits → ensemble_logits
    4. Threshold tuning: grid search per-aspect threshold trên dev set
    5. Apply thresholds lên test set, báo cáo kết quả 4 kịch bản

Chạy từ root project:
    python code/week2/ensemble.py

Không cần GPU — inference trên CPU đủ nhanh (~2 phút cho 2600 samples).
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import (
    TRAIN_CONFIG, PHOBERT_MODEL_NAME, ASPECT_COLUMNS, ZERO_TRAIN_ASPECTS
)
from utils.helpers import set_seed, get_device, load_json, save_json
from step2_dataloader import create_dataloaders
from transformers import AutoTokenizer
from model import ABSAPhoBERT
from train import load_class_weights
from step4_eval import evaluate_predictions


# ─── Config ──────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    "orig_cls_only": "outputs/results/week2_results_VNcoreNLP/models_cls_only/best_model.pt",
    "v23_cls_only":  "outputs/results/week2_results_VNCoreNLP_version2.3/models_cls_only/best_model.pt",
}

THRESHOLD_GRID = np.arange(0.30, 0.96, 0.05).tolist()  # [0.30, 0.35, ..., 0.95]


# ─── Core: collect logits ─────────────────────────────────────────────────────

def collect_logits(
    checkpoint_path: str,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Load checkpoint → inference → trả về raw logits và labels.

    Returns:
        logits_arr: np.ndarray [N, 34, 4]  — raw logits trước softmax
        labels_arr: np.ndarray [N, 34]     — ground truth (0-3)
    """
    # Load checkpoint metadata để biết encoder_option
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    encoder_option = cfg.get("encoder_option", "cls_only")

    print(f"  Loading: {checkpoint_path}")
    print(f"  epoch={ckpt['epoch']}, dev_combined_f1={ckpt['combined_f1']:.4f}, "
          f"encoder={encoder_option}, lr={cfg.get('learning_rate')}")

    # Build model với đúng encoder_option
    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=cfg.get("dropout", 0.2),
        encoder_option=encoder_option,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_logits = []   # list of [batch, 34, 4]
    all_labels = []   # list of [batch, 34]

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]          # giữ trên CPU

            out = model(input_ids, attention_mask)    # không truyền labels → không tính loss
            # logits: list of 34 × [batch, 4] → stack → [batch, 34, 4]
            logits_batch = torch.stack(out["logits"], dim=1)  # [batch, 34, 4]

            all_logits.append(logits_batch.cpu().numpy())
            all_labels.append(labels.numpy())

    logits_arr = np.vstack(all_logits)   # [N, 34, 4]
    labels_arr = np.vstack(all_labels)   # [N, 34]
    return logits_arr, labels_arr


# ─── Threshold tuning ─────────────────────────────────────────────────────────

def tune_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: list = THRESHOLD_GRID,
) -> tuple:
    """
    Grid search per-aspect threshold trên dev set.

    Với mỗi aspect i:
        probs = softmax(logits[:, i, :])  → [N, 4]
        p_absent = probs[:, 0]            → [N]

        for t in grid:
            pred = absent nếu p_absent > t, else argmax(probs[:, 1:]) + 1
            tính ACD F1
        t_i* = threshold cho ACD F1 cao nhất

    Returns:
        thresholds: np.ndarray [34]  — ngưỡng tối ưu per aspect
        threshold_f1s: np.ndarray [34]  — ACD F1 tại threshold tối ưu
    """
    from sklearn.metrics import f1_score

    N, num_aspects, num_classes = logits.shape
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()   # [N, 34, 4]
    p_absent = probs[:, :, 0]                                  # [N, 34]

    thresholds   = np.zeros(num_aspects)
    threshold_f1s = np.zeros(num_aspects)

    for i in range(num_aspects):
        yt_bin = (labels[:, i] > 0).astype(int)  # ground truth present/absent

        best_t  = 0.5
        best_f1 = 0.0

        for t in grid:
            # absent nếu p_absent > t
            yp_bin = (p_absent[:, i] <= t).astype(int)   # 1 = present
            f1 = f1_score(yt_bin, yp_bin, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t

        thresholds[i]    = best_t
        threshold_f1s[i] = best_f1

    return thresholds, threshold_f1s


# ─── Apply thresholds → predictions ──────────────────────────────────────────

def apply_thresholds(
    logits: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Dùng per-aspect threshold để convert logits → predictions.

    Args:
        logits:     [N, 34, 4]
        thresholds: [34]

    Returns:
        preds: [N, 34]  values 0-3
    """
    probs    = F.softmax(torch.tensor(logits), dim=-1).numpy()  # [N, 34, 4]
    p_absent = probs[:, :, 0]                                    # [N, 34]

    N, num_aspects = p_absent.shape
    preds = np.zeros((N, num_aspects), dtype=int)

    for i in range(num_aspects):
        t = thresholds[i]
        absent_mask = p_absent[:, i] > t

        # Với samples present: argmax trong 3 classes 1/2/3
        present_pred = np.argmax(probs[:, i, 1:], axis=-1) + 1   # [N], values 1-3
        preds[:, i]  = np.where(absent_mask, 0, present_pred)

    return preds


# ─── Argmax baseline (không threshold) ───────────────────────────────────────

def argmax_preds(logits: np.ndarray) -> np.ndarray:
    """Argmax đơn giản: [N, 34, 4] → [N, 34]."""
    return np.argmax(logits, axis=-1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = torch.device("cpu")   # CPU đủ nhanh cho inference
    print(f"[Device] {device} (ensemble không cần GPU)")

    # === Tokenizer & DataLoaders ===
    print(f"\n[Tokenizer] Loading {PHOBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    config = TRAIN_CONFIG
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_path="data/train_preprocessed.csv",
        dev_path  ="data/dev_preprocessed.csv",
        test_path ="data/test_preprocessed.csv",
        tokenizer=tokenizer,
        batch_size=32,
        max_len=config["max_seq_len"],
        num_workers=0,
        use_preprocessed=True,
    )
    print(f"[Data] dev={len(dev_loader.dataset)} | test={len(test_loader.dataset)}")

    # === Step 1: Collect logits từ mỗi model ===
    print("\n" + "="*60)
    print("STEP 1 — Collect logits từ 2 models")
    print("="*60)

    dev_logits_dict  = {}
    test_logits_dict = {}
    dev_labels = None
    test_labels = None

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n[{model_name}]")
        dl, ll  = collect_logits(ckpt_path, dev_loader, device)
        tl, tll = collect_logits(ckpt_path, test_loader, device)
        dev_logits_dict[model_name]  = dl
        test_logits_dict[model_name] = tl
        if dev_labels is None:
            dev_labels  = ll
            test_labels = tll
        print(f"  dev logits shape:  {dl.shape} | test: {tl.shape}")

    # === Step 2: Ensemble logits (average) ===
    print("\n" + "="*60)
    print("STEP 2 — Average logits (ensemble)")
    print("="*60)

    dev_ensemble  = np.mean(list(dev_logits_dict.values()),  axis=0)  # [N, 34, 4]
    test_ensemble = np.mean(list(test_logits_dict.values()), axis=0)

    print(f"  ensemble dev:  {dev_ensemble.shape}")
    print(f"  ensemble test: {test_ensemble.shape}")

    # === Step 3: Threshold tuning trên dev ===
    print("\n" + "="*60)
    print("STEP 3 — Threshold tuning trên dev set (ensemble logits)")
    print("="*60)

    thresholds, thr_f1s = tune_thresholds(dev_ensemble, dev_labels)

    print(f"  Default threshold (all aspects): 0.50")
    print(f"  Tuned thresholds — mean: {thresholds.mean():.3f} | "
          f"min: {thresholds.min():.3f} | max: {thresholds.max():.3f}")
    changed = [(ASPECT_COLUMNS[i], thresholds[i])
               for i in range(34) if abs(thresholds[i] - 0.5) > 0.09]
    if changed:
        print(f"  Aspects thay đổi nhiều (|t-0.5|>0.09):")
        for asp, t in sorted(changed, key=lambda x: x[1]):
            print(f"    {asp:45s} → t*={t:.2f}")

    # === Step 4: Evaluate 4 kịch bản ===
    print("\n" + "="*60)
    print("STEP 4 — Kết quả 4 kịch bản")
    print("="*60)

    results = {}

    # A: Gốc cls_only baseline (argmax)
    preds_a = argmax_preds(dev_logits_dict["orig_cls_only"])
    m_a_dev = evaluate_predictions(dev_labels, preds_a,
                                   title="A — Gốc cls_only (baseline)",
                                   exclude_aspects=ZERO_TRAIN_ASPECTS)
    preds_a_test = argmax_preds(test_logits_dict["orig_cls_only"])
    m_a_test = evaluate_predictions(test_labels, preds_a_test,
                                    title="A — Gốc cls_only TEST",
                                    exclude_aspects=ZERO_TRAIN_ASPECTS)
    results["A_orig_baseline"] = {
        "dev":  m_a_dev["macro_combined_f1"],
        "test": m_a_test["macro_combined_f1"],
        "desc": "Gốc cls_only, argmax (baseline)"
    }

    # B: Ensemble argmax (không threshold)
    preds_b = argmax_preds(dev_ensemble)
    m_b_dev = evaluate_predictions(dev_labels, preds_b,
                                   title="B — Ensemble argmax",
                                   exclude_aspects=ZERO_TRAIN_ASPECTS)
    preds_b_test = argmax_preds(test_ensemble)
    m_b_test = evaluate_predictions(test_labels, preds_b_test,
                                    title="B — Ensemble argmax TEST",
                                    exclude_aspects=ZERO_TRAIN_ASPECTS)
    results["B_ensemble_argmax"] = {
        "dev":  m_b_dev["macro_combined_f1"],
        "test": m_b_test["macro_combined_f1"],
        "desc": "Ensemble (avg logits), argmax"
    }

    # C: Gốc cls_only + threshold (không ensemble)
    preds_c = apply_thresholds(dev_logits_dict["orig_cls_only"], thresholds)
    m_c_dev = evaluate_predictions(dev_labels, preds_c,
                                   title="C — Gốc cls_only + threshold",
                                   exclude_aspects=ZERO_TRAIN_ASPECTS)
    preds_c_test = apply_thresholds(test_logits_dict["orig_cls_only"], thresholds)
    m_c_test = evaluate_predictions(test_labels, preds_c_test,
                                    title="C — Gốc cls_only + threshold TEST",
                                    exclude_aspects=ZERO_TRAIN_ASPECTS)
    results["C_orig_threshold"] = {
        "dev":  m_c_dev["macro_combined_f1"],
        "test": m_c_test["macro_combined_f1"],
        "desc": "Gốc cls_only + threshold tuning"
    }

    # D: Ensemble + threshold (kỳ vọng tốt nhất)
    preds_d = apply_thresholds(dev_ensemble, thresholds)
    m_d_dev = evaluate_predictions(dev_labels, preds_d,
                                   title="D — Ensemble + threshold",
                                   exclude_aspects=ZERO_TRAIN_ASPECTS)
    preds_d_test = apply_thresholds(test_ensemble, thresholds)
    m_d_test = evaluate_predictions(test_labels, preds_d_test,
                                    title="D — Ensemble + threshold TEST",
                                    exclude_aspects=ZERO_TRAIN_ASPECTS)
    results["D_ensemble_threshold"] = {
        "dev":  m_d_dev["macro_combined_f1"],
        "test": m_d_test["macro_combined_f1"],
        "desc": "Ensemble + threshold tuning (best kỳ vọng)"
    }

    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY — Combined F1 (tất cả kịch bản)")
    print("="*60)
    print(f"  {'Kịch bản':<45} {'Dev':>7} {'Test':>7}")
    print(f"  {'-'*45} {'-'*7} {'-'*7}")
    for key, r in results.items():
        marker = " ← best" if r["test"] == max(v["test"] for v in results.values()) else ""
        print(f"  {r['desc']:<45} {r['dev']:>7.4f} {r['test']:>7.4f}{marker}")
    print(f"\n  Tham chiếu:")
    print(f"  {'v2.4 cls_only (best single model)':<45} {'—':>7} {'0.5583':>7}")
    print(f"  {'SOTA ds4v (Huynh 2022)':<45} {'—':>7} {'0.7732':>7}")

    # === Lưu thresholds và kết quả ===
    os.makedirs("outputs/results/ensemble", exist_ok=True)
    save_json(
        {"thresholds": thresholds.tolist(),
         "threshold_f1s": thr_f1s.tolist(),
         "aspects": ASPECT_COLUMNS},
        "outputs/results/ensemble/thresholds.json"
    )
    save_json(results, "outputs/results/ensemble/ensemble_results.json")
    print(f"\n  Saved: outputs/results/ensemble/")

    return results


if __name__ == "__main__":
    main()
