"""
ensemble.py — Ensemble top-k checkpoints bằng cách average softmax logits.

Cách dùng:
    from ensemble import ensemble_predict_and_evaluate
    metrics, y_pred = ensemble_predict_and_evaluate(ckpt_paths, model_template, ...)

Hoặc chạy trực tiếp:
    python ensemble.py --save-dir outputs/models_lr2e5 --split test
"""

import os
import sys
import argparse
from typing import List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import ZERO_TRAIN_ASPECTS
from utils.helpers import load_json, save_json
from step4_eval import evaluate_predictions


def _collect_logits(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_weights: list,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chạy inference, trả về (logits [N, 34, 4], labels [N, 34])."""
    from train import run_epoch

    model.eval()
    all_logits: list = []
    all_labels: list = []

    weights_on_device = [w.to(device) for w in class_weights]

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out = model(input_ids, attention_mask,
                        labels=labels, class_weights=weights_on_device)

            # out["logits"]: [B, 34, 4]
            logits = out["logits"].cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_logits), np.vstack(all_labels)


def ensemble_predict_and_evaluate(
    ckpt_paths: List[str],
    model_template: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_weights: list,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Tuple[dict, np.ndarray]:
    """
    Load từng checkpoint, collect logits, average softmax, argmax → prediction.

    Args:
        ckpt_paths:     Danh sách đường dẫn đến các .pt checkpoint.
        model_template: Model instance (kiến trúc phải giống nhau giữa các checkpoint).
        dataloader:     DataLoader (test hoặc dev).
        class_weights:  Class weights (chỉ cần cho forward pass tính loss).
        device:         torch.device.
        save_path:      Nếu có, lưu metrics ra JSON.

    Returns:
        (metrics_dict, y_pred_ensemble)
    """
    print(f"\n[Ensemble] {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu")
        print(f"  epoch={ckpt['epoch']}  combined_f1={ckpt['combined_f1']:.4f}  {p}")

    all_softmax: list = []
    y_true_ref:  Optional[np.ndarray] = None

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_template.load_state_dict(ckpt["model_state_dict"])
        model_template.to(device).eval()

        logits, y_true = _collect_logits(model_template, dataloader, class_weights, device)
        # logits: [N, 34, 4]
        softmax = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        all_softmax.append(softmax)

        if y_true_ref is None:
            y_true_ref = y_true

    # Average softmax qua tất cả checkpoints → argmax
    avg_softmax = np.mean(all_softmax, axis=0)   # [N, 34, 4]
    y_pred = np.argmax(avg_softmax, axis=-1)      # [N, 34]

    metrics = evaluate_predictions(
        y_true_ref, y_pred,
        title=f"Ensemble ({len(ckpt_paths)} ckpts) — TEST",
        exclude_aspects=ZERO_TRAIN_ASPECTS,
        save_path=save_path,
    )

    print(f"[Ensemble] ACD F1={metrics['macro_acd_f1']:.4f}  "
          f"SPC F1={metrics['macro_spc_f1']:.4f}  "
          f"Combined F1={metrics['macro_combined_f1']:.4f}")
    return metrics, y_pred


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble top-k checkpoints")
    parser.add_argument("--save-dir",  required=True, help="outputs/models_xxx/")
    parser.add_argument("--encoder",   default=None,  help="concat_4_layers | cls_only")
    parser.add_argument("--split",     default="test", choices=["dev", "test"])
    parser.add_argument("--data-suffix", default="", help="'' | '_sota'")
    args = parser.parse_args()

    from utils.constants import TRAIN_CONFIG, PHOBERT_MODEL_NAME
    from utils.helpers import get_device
    from step2_dataloader import create_dataloaders
    from transformers import AutoTokenizer
    from model import ABSAPhoBERT
    from train import load_class_weights

    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    ds = args.data_suffix
    _, dev_loader, test_loader = create_dataloaders(
        train_path=f"data/train_preprocessed{ds}.csv",
        dev_path  =f"data/dev_preprocessed{ds}.csv",
        test_path =f"data/test_preprocessed{ds}.csv",
        tokenizer=tokenizer,
        batch_size=TRAIN_CONFIG["batch_size"],
        max_len=TRAIN_CONFIG["max_seq_len"],
        num_workers=2,
        use_preprocessed=True,
    )
    loader = test_loader if args.split == "test" else dev_loader

    class_weights = load_class_weights(
        "outputs/eda/class_weights.json",
        weight_clip=TRAIN_CONFIG["weight_clip"],
        device=torch.device("cpu"),
    )

    # Đọc top_k_checkpoints.json
    top_k_path = os.path.join(args.save_dir, "top_k_checkpoints.json")
    if not os.path.exists(top_k_path):
        raise FileNotFoundError(f"Không tìm thấy {top_k_path}")
    top_k_info = load_json(top_k_path)
    ckpt_paths = [item["path"] for item in top_k_info if os.path.exists(item["path"])]

    enc = args.encoder or "concat_4_layers"
    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=TRAIN_CONFIG["dropout"],
        encoder_option=enc,
    ).to(device)

    save_path = os.path.join(args.save_dir, f"ensemble_{args.split}_metrics.json")
    ensemble_predict_and_evaluate(ckpt_paths, model, loader, class_weights, device, save_path)
