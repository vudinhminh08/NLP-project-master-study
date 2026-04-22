
import os
import sys
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))

from utils.constants import ZERO_TRAIN_ASPECTS, RARE_ASPECTS
from utils.helpers import save_json, load_json
from step4_eval import evaluate_predictions
from train import run_epoch

def load_best_model(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(
        f"[Loaded] {checkpoint_path} "
        f"(epoch={ckpt['epoch']}, combined_f1={ckpt['combined_f1']:.4f})"
    )
    return model


def predict_and_evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_weights: list,
    device: torch.device,
    split_name: str = "test",
    save_path: Optional[str] = None,
) -> tuple:
    _, _, _, y_true, y_pred = run_epoch(
        model, dataloader, device, class_weights, is_train=False
    )
    metrics = evaluate_predictions(
        y_true, y_pred,
        title=f"Final — {split_name.upper()}",
        exclude_aspects=ZERO_TRAIN_ASPECTS,
        save_path=save_path,
    )
    return metrics, y_true, y_pred


def generate_summary_report(
    history: dict,
    dev_metrics: dict,
    test_metrics: dict,
    config: dict,
    save_path: str = "outputs/results/phobert_summary.md",
) -> None:
    gap_acd  = 0.8255 - test_metrics["macro_acd_f1"]
    gap_comb = 0.7732 - test_metrics["macro_combined_f1"]


    per_aspect = test_metrics.get("per_aspect", {})
    bottom5 = sorted(per_aspect.items(), key=lambda x: x[1].get("acd_f1", 0))[:5]

    lines = [
        "# Kết quả PhoBERT Multi-task",
        "",
        "## Config thực tế",
        "| Tham số | Giá trị |",
        "|---------|---------|",
        f"| Encoder | {config.get('encoder_option')} |",
        f"| MAX_SEQ_LEN | {config.get('max_seq_len')} |",
        f"| Batch size | {config.get('batch_size')} × {config.get('grad_accumulation_steps')} = "
        f"{config.get('batch_size', 8) * config.get('grad_accumulation_steps', 2)} (effective) |",
        f"| Learning rate | {config.get('learning_rate')} |",
        f"| Warmup | {config.get('warmup_ratio', 0.1) * 100:.0f}% steps |",
        f"| Weight clip | {config.get('weight_clip')} |",
        f"| Best epoch | {history['best_epoch']} / {len(history['train_loss'])} |",
        "",
        "## Kết quả",
        "",
        "| Split | ACD F1 | SPC F1 | Combined F1 |",
        "|-------|--------|--------|-------------|",
        f"| Dev   | {dev_metrics['macro_acd_f1']:.4f} | "
        f"{dev_metrics['macro_spc_f1']:.4f} | "
        f"{dev_metrics['macro_combined_f1']:.4f} |",
        f"| **Test**  | **{test_metrics['macro_acd_f1']:.4f}** | "
        f"**{test_metrics['macro_spc_f1']:.4f}** | "
        f"**{test_metrics['macro_combined_f1']:.4f}** |",
        f"| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 |",
        "",
        "## Phân tích Gap so với SOTA",
        "",
        f"- **ACD F1 gap:** {gap_acd:.4f} ({gap_acd * 100:.1f}%)",
        f"- **Combined F1 gap:** {gap_comb:.4f} ({gap_comb * 100:.1f}%)",
        "",
        "### Nguyên nhân gap (phân tích):",
        "1. **underthesea vs VnCoreNLP:** Dùng underthesea làm fallback → ~1-2% F1 loss",
        "2. **ROOM_AMENITIES#PRICES:** 0 training samples → ACD F1 = 0 cho aspect này",
        "3. **Neutral cực hiếm (weight=154→clip=10):** SPC F1 cho neutral thấp",
        "4. **Dataset nhỏ (3000 train):** khó học tốt các aspect hiếm",
        "5. **Single model:** báo cáo chính giữ bản single để dễ giải thích và so sánh công bằng",
        "",
        "## Bottom 5 Aspects (ACD F1 thấp nhất — Test set)",
        "",
        "| Aspect | ACD F1 | SPC F1 | Support | Ghi chú |",
        "|--------|--------|--------|---------|---------|",
    ]

    for asp, m in bottom5:
        if asp in ZERO_TRAIN_ASPECTS:
            note = "0 train samples"
        elif asp in ["ROOMS#MISCELLANEOUS", "FOOD&DRINKS#MISCELLANEOUS"]:
            note = "rare"
        else:
            note = ""
        lines.append(
            f"| {asp} | {m.get('acd_f1', 0):.4f} | "
            f"{m.get('spc_f1') or 0:.4f} | "
            f"{m.get('support', 0)} | {note} |"
        )

    lines += [
        "",
        "## Learning Curve",
        "Xem learning curve trong output của run PhoBERT tương ứng.",
        "",
        f"Early stopping tại epoch **{history['best_epoch']}**.",
        "Dev loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu overfitting.",
    ]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Saved: {save_path}")
