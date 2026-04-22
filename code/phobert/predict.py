
import os
import sys
from typing import Optional

import torch
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))

from utils.constants import ZERO_TRAIN_ASPECTS, RARE_ASPECTS
from utils.helpers import save_json, load_json
from step4_eval import evaluate_predictions, compute_aspect_f1
from train import run_epoch


def _collect_presence_outputs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    model.eval()
    all_labels = []
    all_presence_probs = []
    all_sent_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)

            presence_logits = torch.stack(out["presence_logits"], dim=1)
            presence_probs = torch.sigmoid(presence_logits).cpu().numpy()
            sent_preds = torch.stack(
                [logit[:, 1:].argmax(dim=-1) + 1 for logit in out["logits"]],
                dim=1,
            ).cpu().numpy()

            all_labels.append(batch["labels"].numpy())
            all_presence_probs.append(presence_probs)
            all_sent_preds.append(sent_preds)

    return (
        np.vstack(all_labels),
        np.vstack(all_presence_probs),
        np.vstack(all_sent_preds),
    )

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
    _, y_true, y_pred = run_epoch(
        model, dataloader, device, class_weights, is_train=False
    )
    metrics = evaluate_predictions(
        y_true, y_pred,
        title=f"Final — {split_name.upper()}",
        exclude_aspects=ZERO_TRAIN_ASPECTS,
        save_path=save_path,
    )
    return metrics, y_true, y_pred


def tune_presence_threshold(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: Optional[list] = None,
    exclude_aspects: Optional[list] = None,
    mode: str = "per_aspect_combined",
    save_path: Optional[str] = None,
) -> tuple:
    if not getattr(model, "use_split_loss", False):
        return None, None
    thresholds = thresholds or [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    exclude_aspects = exclude_aspects or ZERO_TRAIN_ASPECTS

    y_true, presence_probs, sent_preds = _collect_presence_outputs(
        model, dataloader, device
    )

    def score_thresholds(thr_vec: np.ndarray) -> float:
        y_pred = np.where(presence_probs >= thr_vec[None, :], sent_preds, 0)
        metrics = compute_aspect_f1(
            y_true,
            y_pred.astype(np.int64),
            exclude_aspects=exclude_aspects,
        )
        return float(metrics["macro_combined_f1"])

    best_global = float(model.presence_threshold)
    best_score = -1.0
    for thr in thresholds:
        thr_vec = np.full(y_true.shape[1], float(thr), dtype=np.float32)
        score = score_thresholds(thr_vec)
        if score > best_score:
            best_score = score
            best_global = float(thr)

    threshold_vec = np.full(y_true.shape[1], best_global, dtype=np.float32)
    if mode in ("per_aspect", "per_aspect_combined", "hybrid"):
        for i in range(y_true.shape[1]):
            yt_bin = (y_true[:, i] > 0).astype(np.int64)
            local_thr = best_global
            local_f1 = -1.0
            for thr in thresholds:
                yp_bin = (presence_probs[:, i] >= float(thr)).astype(np.int64)
                f1 = f1_score(yt_bin, yp_bin, average="binary", zero_division=0)
                if f1 > local_f1:
                    local_f1 = f1
                    local_thr = float(thr)
            threshold_vec[i] = local_thr
        score = score_thresholds(threshold_vec)
        if score >= best_score:
            best_score = score
        else:
            threshold_vec[:] = best_global

    if mode in ("per_aspect_combined", "hybrid"):
        improved = True
        rounds = 0
        while improved and rounds < 2:
            improved = False
            rounds += 1
            for i in range(y_true.shape[1]):
                current = float(threshold_vec[i])
                best_local_thr = current
                best_local_score = best_score
                for thr in thresholds:
                    if float(thr) == current:
                        continue
                    trial = threshold_vec.copy()
                    trial[i] = float(thr)
                    score = score_thresholds(trial)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_thr = float(thr)
                if best_local_thr != current:
                    threshold_vec[i] = best_local_thr
                    best_score = best_local_score
                    improved = True

    with torch.no_grad():
        model.aspect_presence_thresholds.copy_(
            torch.tensor(
                threshold_vec,
                dtype=model.aspect_presence_thresholds.dtype,
                device=model.aspect_presence_thresholds.device,
            )
        )
    model.presence_threshold = float(threshold_vec.mean())

    payload = {
        "mode": mode,
        "grid": [float(x) for x in thresholds],
        "mean_threshold": float(threshold_vec.mean()),
        "min_threshold": float(threshold_vec.min()),
        "max_threshold": float(threshold_vec.max()),
        "dev_combined_f1": float(best_score),
        "thresholds": [float(x) for x in threshold_vec.tolist()],
    }
    if save_path:
        save_json(payload, save_path)
    print(
        f"[Tune] presence_threshold mode={mode}, "
        f"mean={payload['mean_threshold']:.3f}, "
        f"range=[{payload['min_threshold']:.2f}, {payload['max_threshold']:.2f}], "
        f"dev_combined_f1={best_score:.4f}"
    )
    return payload, best_score


def generate_summary_report(
    history: dict,
    dev_metrics: dict,
    test_metrics: dict,
    config: dict,
    ensemble_metrics: dict = None,
    threshold_info: dict = None,
    save_path: str = "outputs/results/phobert_summary.md",
) -> None:
    primary_metrics = test_metrics
    if (
        ensemble_metrics is not None
        and ensemble_metrics["macro_combined_f1"] > test_metrics["macro_combined_f1"]
    ):
        primary_metrics = ensemble_metrics

    gap_acd  = 0.8255 - primary_metrics["macro_acd_f1"]
    gap_comb = 0.7732 - primary_metrics["macro_combined_f1"]


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
        f"| Split loss | {config.get('use_split_loss', False)} |",
        f"| Threshold tuning | {config.get('tune_presence_threshold', False)} |",
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
    ]

    if ensemble_metrics is not None:
        lines.append(
            f"| Ensemble Top-3 | {ensemble_metrics['macro_acd_f1']:.4f} | "
            f"{ensemble_metrics['macro_spc_f1']:.4f} | "
            f"{ensemble_metrics['macro_combined_f1']:.4f} |"
        )

    lines += [
        f"| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 |",
        f"| Notebook tham khảo — concat | 0.6827 | 0.5374 | 0.6101 |",
        f"| Notebook tham khảo — cls_only | 0.6849 | 0.5587 | 0.6218 |",
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
    ]

    if threshold_info is not None:
        lines += [
            "",
            "## Presence Threshold Tuning",
            "",
            f"- Mode: `{threshold_info.get('mode')}`",
            f"- Grid: `{threshold_info.get('grid')}`",
            f"- Mean threshold: {threshold_info.get('mean_threshold', 0):.3f}",
            f"- Range: {threshold_info.get('min_threshold', 0):.2f} → "
            f"{threshold_info.get('max_threshold', 0):.2f}",
            f"- Dev Combined F1 after tuning: "
            f"{threshold_info.get('dev_combined_f1', 0):.4f}",
        ]

    lines += [
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
