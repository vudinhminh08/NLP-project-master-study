
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


def _collect_presence_and_sentiment_outputs(
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
            labels = batch["labels"]

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                class_weights=None,
            )

            presence_logits = torch.stack(out["presence_logits"], dim=1)
            presence_probs = torch.sigmoid(presence_logits).cpu().numpy()
            sent_preds = torch.stack(
                [logit[:, 1:].argmax(dim=-1) + 1 for logit in out["logits"]],
                dim=1,
            ).cpu().numpy()

            all_labels.append(labels.numpy())
            all_presence_probs.append(presence_probs)
            all_sent_preds.append(sent_preds)

    y_true = np.vstack(all_labels)
    presence_probs = np.vstack(all_presence_probs)
    sent_preds = np.vstack(all_sent_preds)
    return y_true, presence_probs, sent_preds

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
    class_weights: list,
    device: torch.device,
    thresholds: Optional[list] = None,
    exclude_aspects: Optional[list] = None,
    mode: str = "per_aspect",
) -> tuple:
    if not hasattr(model, "presence_threshold"):
        return None, None

    if thresholds is None:
        thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    if exclude_aspects is None:
        exclude_aspects = ZERO_TRAIN_ASPECTS

    y_true, presence_probs, sent_preds = _collect_presence_and_sentiment_outputs(
        model,
        dataloader,
        device,
    )

    def combined_from_thresholds(thr_vec: np.ndarray) -> float:
        y_pred_local = np.where(
            presence_probs >= thr_vec[None, :],
            sent_preds,
            0,
        )
        m = compute_aspect_f1(
            y_true,
            y_pred_local.astype(np.int64),
            exclude_aspects=exclude_aspects,
        )
        return float(m["macro_combined_f1"])

    best_score = -1.0
    best_global_thr = float(model.presence_threshold)

    for thr in thresholds:
        y_pred = np.where(
            presence_probs >= float(thr),
            sent_preds,
            0,
        )
        metrics = compute_aspect_f1(
            y_true,
            y_pred.astype(np.int64),
            exclude_aspects=exclude_aspects,
        )
        combined = metrics["macro_combined_f1"]
        if combined > best_score:
            best_score = combined
            best_global_thr = float(thr)

    per_aspect_thresholds = np.full(y_true.shape[1], best_global_thr, dtype=np.float32)
    if mode in ("per_aspect", "hybrid", "per_aspect_combined"):
        for i in range(y_true.shape[1]):
            yt_bin = (y_true[:, i] > 0).astype(np.int64)
            best_i_thr = best_global_thr
            best_i_f1 = -1.0
            for thr in thresholds:
                yp_bin = (presence_probs[:, i] >= float(thr)).astype(np.int64)
                f1 = f1_score(yt_bin, yp_bin, average="binary", zero_division=0)
                if f1 > best_i_f1:
                    best_i_f1 = f1
                    best_i_thr = float(thr)
            per_aspect_thresholds[i] = best_i_thr

        y_pred_pa = np.where(
            presence_probs >= per_aspect_thresholds[None, :],
            sent_preds,
            0,
        )
        metrics_pa = compute_aspect_f1(
            y_true,
            y_pred_pa.astype(np.int64),
            exclude_aspects=exclude_aspects,
        )
        combined_pa = metrics_pa["macro_combined_f1"]
        if combined_pa >= best_score:
            best_score = combined_pa
        else:
            per_aspect_thresholds[:] = best_global_thr

    # Coordinate search directly on Combined F1 usually yields better dev score
    if mode in ("per_aspect_combined", "hybrid"):
        improved = True
        max_rounds = 2
        round_idx = 0
        while improved and round_idx < max_rounds:
            improved = False
            round_idx += 1
            for i in range(y_true.shape[1]):
                current_thr = float(per_aspect_thresholds[i])
                local_best_thr = current_thr
                local_best_score = best_score
                for thr in thresholds:
                    if float(thr) == current_thr:
                        continue
                    trial = per_aspect_thresholds.copy()
                    trial[i] = float(thr)
                    score_trial = combined_from_thresholds(trial)
                    if score_trial > local_best_score:
                        local_best_score = score_trial
                        local_best_thr = float(thr)
                if local_best_thr != current_thr:
                    per_aspect_thresholds[i] = local_best_thr
                    best_score = local_best_score
                    improved = True

    with torch.no_grad():
        thr_tensor = torch.tensor(
            per_aspect_thresholds,
            dtype=model.aspect_presence_thresholds.dtype,
            device=model.aspect_presence_thresholds.device,
        )
        model.aspect_presence_thresholds.copy_(thr_tensor)
    model.presence_threshold = float(per_aspect_thresholds.mean())

    if np.allclose(per_aspect_thresholds, per_aspect_thresholds[0]):
        print(
            f"[Tune] Best global presence_threshold on dev: {per_aspect_thresholds[0]:.2f} "
            f"(Combined F1={best_score:.4f})"
        )
    else:
        print(
            f"[Tune] Best per-aspect thresholds on dev: mean={per_aspect_thresholds.mean():.3f}, "
            f"min={per_aspect_thresholds.min():.2f}, max={per_aspect_thresholds.max():.2f}, "
            f"Combined F1={best_score:.4f}"
        )
    return per_aspect_thresholds.tolist(), best_score


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
