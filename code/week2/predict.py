"""
predict.py — Load best model → evaluate → generate summary report.

Chạy qua run_experiment.py (không chạy trực tiếp).
"""

import os
import sys
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))

from utils.constants import ZERO_TRAIN_ASPECTS, RARE_ASPECTS
from utils.helpers import save_json, load_json
from step4_eval import evaluate_predictions
from train import run_epoch

# add new (try to fix)
# Thêm vào khoảng dòng 20
def get_smart_preds(logits_list, threshold=0.3):
    """
    Chuyển đổi logits thành dự đoán dựa trên ngưỡng thay vì argmax.
    Nếu xác suất nhãn 0 (Absent) < (1 - threshold), ta ưu tiên chọn nhãn 1, 2, hoặc 3.
    """
    # logits_list: list của 34 tensors, mỗi cái [batch_size, 4]
    # Chuyển về dạng [batch_size, 34, 4]
    stacked_logits = torch.stack(logits_list, dim=1) 
    probs = torch.softmax(stacked_logits, dim=-1) # [batch, 34, 4]
    
    batch_size, num_aspects, _ = probs.shape
    final_preds = np.zeros((batch_size, num_aspects), dtype=int)

    for i in range(batch_size):
        for j in range(num_aspects):
            p = probs[i, j]
            # Nếu xác suất nhãn 'vắng mặt' (0) không đủ cao (ví dụ < 70%)
            # ta ép mô hình chọn nhãn có khả năng nhất trong (Pos, Neg, Neu)
            if p[0] < (1 - threshold): 
                final_preds[i, j] = torch.argmax(p[1:]).item() + 1
            else:
                final_preds[i, j] = 0
    return final_preds

def load_best_model(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load best checkpoint vào model.

    Args:
        checkpoint_path: đường dẫn tới best_model.pt
        model:           ABSAPhoBERT instance (chưa load weights)
        device:          torch device

    Returns:
        model đã load weights, ở eval mode
    """
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
    """
    Inference + compute metrics + optionally save JSON.

    Args:
        model:        ABSAPhoBERT (đã load best weights)
        dataloader:   DataLoader cho split cần evaluate
        class_weights: list of 34 weight tensors
        device:       torch device
        split_name:   tên split để in trong report ("dev" hoặc "test")
        save_path:    nếu không None, lưu metrics JSON

    Returns:
        (metrics dict, y_true np.ndarray [N,34], y_pred np.ndarray [N,34])
    """
    # _, y_true, y_pred = run_epoch(
    #     model, dataloader, device, class_weights, is_train=False
    # )
    # metrics = evaluate_predictions(
    #     y_true, y_pred,
    #     title=f"Final — {split_name.upper()}",
    #     exclude_aspects=ZERO_TRAIN_ASPECTS,
    #     save_path=save_path,
    # )
    # add new (try to fix)
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Lấy logits thô từ model
            outputs = model(input_ids, attention_mask, labels=labels, class_weights=class_weights)
            logits = outputs["logits"] # List của 34 tensors
            
            # Áp dụng ngưỡng thông minh
            preds = get_smart_preds(logits, threshold=0.3)
            
            all_y_true.append(labels.cpu().numpy())
            all_y_pred.append(preds)

    y_true = np.vstack(all_y_true)
    y_pred = np.vstack(all_y_pred)
    return metrics, y_true, y_pred


def generate_summary_report(
    history: dict,
    dev_metrics: dict,
    test_metrics: dict,
    config: dict,
    save_path: str = "outputs/results/week2_summary.md",
) -> None:
    """
    Tạo markdown report cho báo cáo cuối kỳ.

    Bao gồm: config thực tế, kết quả dev/test, gap so với SOTA,
    bottom 5 aspects, nguyên nhân gap, learning curve reference.

    Args:
        history:      dict từ train() — train_loss, dev_f1 theo epoch
        dev_metrics:  dict từ evaluate_predictions trên dev set
        test_metrics: dict từ evaluate_predictions trên test set
        config:       dict config training thực tế
        save_path:    đường dẫn lưu file .md
    """
    gap_acd  = 0.8255 - test_metrics["macro_acd_f1"]
    gap_comb = 0.7732 - test_metrics["macro_combined_f1"]

    # Bottom 5 aspects theo ACD F1
    per_aspect = test_metrics.get("per_aspect", {})
    bottom5 = sorted(per_aspect.items(), key=lambda x: x[1].get("acd_f1", 0))[:5]

    lines = [
        "# Kết quả Tuần 2 — PhoBERT Multi-task",
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
        "4. **Dataset nhỏ (3000 train):** SOTA có thể dùng data augmentation",
        "5. **Single run:** Chưa ensemble nhiều seeds",
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
        "Xem: `outputs/eda/learning_curve.png`",
        "",
        f"Early stopping tại epoch **{history['best_epoch']}**.",
        "Dev loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu overfitting.",
    ]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Saved: {save_path}")
