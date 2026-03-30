"""
train.py — Training loop với early stopping, gradient clipping, LR scheduling.

Fixes so với version cũ:
    1. Gradient accumulation: flush final batch (tránh bỏ gradient của batch cuối)
    2. Mixed Precision (AMP): dùng torch.cuda.amp để tăng tốc 2x trên T4
    3. class_weights được truyền đúng vào model.forward() (fix theo model.py mới)

Chạy từ root project (qua run_experiment.py):
    python code/week2/run_experiment.py                    # concat_4_layers
    python code/week2/run_experiment.py --encoder cls_only # ablation
"""

import os
import sys
import json
import time
from typing import Optional

import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm

# Mixed precision — safe import (fallback nếu PyTorch cũ)
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))

from utils.constants import ASPECT_COLUMNS, ZERO_TRAIN_ASPECTS
from utils.helpers import set_seed, save_json, load_json
from step4_eval import evaluate_predictions


def load_class_weights(
    weights_path: str,
    weight_clip: float = 10.0,
    device: Optional[torch.device] = None,
) -> list:
    """
    Load per-aspect class weights từ class_weights.json (tạo bởi step1_eda.py).
    Clip tại weight_clip để tránh gradient instability (neutral global weight=154).

    Args:
        weights_path: đường dẫn tới outputs/eda/class_weights.json
        weight_clip:  giá trị tối đa cho mỗi weight (mặc định 10.0)
        device:       torch device để đặt tensor lên (None = cpu)

    Returns:
        list of 34 tensors [4] — mỗi tensor là [w0, w1, w2, w3] cho 1 aspect
        w0=absent, w1=positive, w2=negative, w3=neutral
    """
    data = load_json(weights_path)
    per_aspect = data["per_aspect_weights"]
    device = device or torch.device("cpu")

    weights_list = []
    clipped_count = 0

    for aspect in ASPECT_COLUMNS:
        w_dict = per_aspect.get(aspect, {})
        w = []
        for i in range(4):
            raw = float(w_dict.get(str(i), 1.0))  # default=1.0 nếu class không có
            clipped = min(raw, weight_clip)
            if raw > weight_clip:
                clipped_count += 1
            w.append(clipped)
        weights_list.append(torch.tensor(w, dtype=torch.float32, device=device))

    print(f"[Weights] {len(weights_list)} aspects loaded, "
          f"{clipped_count} weight values clipped at {weight_clip}")
    return weights_list


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: list,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    grad_accum: int = 1,
    is_train: bool = True,
    use_amp: bool = False,
    scaler=None,
) -> tuple:
    """
    Chạy 1 epoch train hoặc eval.

    FIX: Gradient accumulation flush final batch — không bỏ gradient batch cuối.
    FIX: AMP support — faster training trên T4 GPU (~2x speedup).

    Args:
        model:         ABSAPhoBERT
        dataloader:    DataLoader (train hoặc dev/test)
        device:        torch device
        class_weights: list of 34 tensors [4] — per-aspect weights
        optimizer:     AdamW optimizer (None khi eval)
        scheduler:     LR scheduler (None khi eval)
        grad_accum:    gradient accumulation steps
        is_train:      True để train, False để eval
        use_amp:       True để dùng mixed precision (chỉ có ích khi GPU)
        scaler:        GradScaler instance (cần thiết khi use_amp=True)

    Returns:
        (mean_loss, y_true, y_pred)
        y_true/y_pred: np.ndarray [N, 34] khi eval, None khi train
    """
    model.train() if is_train else model.eval()
    total_loss  = 0.0
    all_preds:  list = []
    all_labels: list = []

    # Đặt class weights lên đúng device 1 lần (thay vì mỗi batch)
    weights_on_device = [w.to(device) for w in class_weights]

    n_batches = len(dataloader)
    desc = "Train" if is_train else "Eval"

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        if is_train:
            optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # === Forward pass (với AMP nếu được bật) ===
            if use_amp and AMP_AVAILABLE:
                with autocast():
                    out = model(
                        input_ids, attention_mask,
                        labels=labels,
                        class_weights=weights_on_device,
                    )
                    loss = out["loss"]
            else:
                out = model(
                    input_ids, attention_mask,
                    labels=labels,
                    class_weights=weights_on_device,
                )
                loss = out["loss"]

            total_loss += loss.item()

            # === Backward (chỉ khi train) ===
            if is_train:
                scaled_loss = loss / grad_accum

                if use_amp and AMP_AVAILABLE and scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # FIX: flush khi đủ accum steps HOẶC đây là batch cuối cùng
                is_last_batch = (step + 1) == n_batches
                if (step + 1) % grad_accum == 0 or is_last_batch:
                    if use_amp and AMP_AVAILABLE and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # Eval: thu thập predictions
                all_preds.append(out["preds"].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    mean_loss = total_loss / n_batches

    if is_train:
        return mean_loss, None, None

    y_true = np.vstack(all_labels)  # [N, 34]
    y_pred = np.vstack(all_preds)   # [N, 34]
    return mean_loss, y_true, y_pred


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    dev_loader: torch.utils.data.DataLoader,
    class_weights: list,
    device: torch.device,
    config: dict,
    save_dir: str = "outputs/models",
    results_dir: str = "outputs/results",
    use_amp: bool = False,
) -> dict:
    """
    Full training loop với early stopping.

    Monitor: dev combined_f1 (ACD F1 + SPC F1) / 2, exclude ZERO_TRAIN_ASPECTS.
    Save: best checkpoint + training_history.json mỗi epoch (an toàn khi Colab mất kết nối).

    Args:
        model:        ABSAPhoBERT
        train_loader: DataLoader train
        dev_loader:   DataLoader dev
        class_weights: list of 34 weight tensors
        device:       torch device
        config:       dict từ TRAIN_CONFIG (merged với encoder_config.json)
        save_dir:     thư mục lưu best_model.pt
        results_dir:  thư mục lưu training_history.json
        use_amp:      True để dùng Mixed Precision (khuyến nghị khi có GPU)

    Returns:
        history dict: train_loss, dev_loss, dev_acd_f1, dev_spc_f1, dev_combined_f1 theo epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # === Optimizer — single LR group (v2: revert differential LR) ===
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
        eps=1e-8,
    )

    # === Scheduler: warmup 10% + linear decay ===
    # Tính đúng số optimizer steps (sau gradient accumulation)
    steps_per_epoch = max(1, len(train_loader) // config["grad_accumulation_steps"])
    # Cộng thêm 1 nếu có batch cuối không chia hết
    if len(train_loader) % config["grad_accumulation_steps"] != 0:
        steps_per_epoch += 1

    total_steps  = steps_per_epoch * config["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    # === Scheduler: cosine warmup (v2) hoặc linear warmup (v1 fallback) ===
    if config.get("scheduler") == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # === Mixed Precision Scaler ===
    scaler = GradScaler() if (use_amp and AMP_AVAILABLE and device.type == "cuda") else None
    amp_active = scaler is not None

    # === Info ===
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] {n_params:,} trainable parameters")
    print(f"[Scheduler] Total={total_steps} optimizer steps, Warmup={warmup_steps}")
    print(f"[AMP] Mixed precision: {'ON ✓' if amp_active else 'OFF'}")
    print(f"[Config] encoder={config.get('encoder_option')}, "
          f"seq_len={config.get('max_seq_len')}, "
          f"batch={config['batch_size']}×{config['grad_accumulation_steps']}="
          f"{config['batch_size']*config['grad_accumulation_steps']} (effective)")

    # === History ===
    history = {
        "train_loss":       [],
        "dev_loss":         [],
        "dev_acd_f1":       [],
        "dev_spc_f1":       [],
        "dev_combined_f1":  [],
        "best_epoch":       0,
        "best_combined_f1": 0.0,
        "config":           config,
        "use_amp":          amp_active,
    }
    best_f1 = 0.0
    patience = 0

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─'*60}\nEpoch {epoch}/{config['max_epochs']}")

        # --- Train ---
        train_loss, _, _ = run_epoch(
            model, train_loader, device, class_weights,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=config["grad_accumulation_steps"],
            is_train=True,
            use_amp=amp_active, scaler=scaler,
        )

        # --- Eval ---
        dev_loss, y_true, y_pred = run_epoch(
            model, dev_loader, device, class_weights,
            is_train=False,
            use_amp=amp_active, scaler=None,
        )
        metrics  = evaluate_predictions(
            y_true, y_pred,
            title=f"Dev — Epoch {epoch}",
            exclude_aspects=ZERO_TRAIN_ASPECTS,
        )
        combined = metrics["macro_combined_f1"]
        elapsed  = time.time() - t0

        print(
            f"  Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
            f"ACD: {metrics['macro_acd_f1']:.4f} | "
            f"SPC: {metrics['macro_spc_f1']:.4f} | "
            f"Combined: {combined:.4f} | {elapsed:.0f}s"
        )

        # --- Log history ---
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_acd_f1"].append(metrics["macro_acd_f1"])
        history["dev_spc_f1"].append(metrics["macro_spc_f1"])
        history["dev_combined_f1"].append(combined)

        # --- Save best checkpoint ---
        if combined > best_f1:
            best_f1 = combined
            history["best_epoch"]       = epoch
            history["best_combined_f1"] = best_f1
            ckpt = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "combined_f1":      combined,
                "acd_f1":           metrics["macro_acd_f1"],
                "spc_f1":           metrics["macro_spc_f1"],
                "config":           config,
            }
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"  ✅ Best model saved (combined_f1={best_f1:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"  ⚠️  No improvement [{patience}/{config['early_stop_patience']}]")

        # --- Lưu history sau mỗi epoch (safe nếu Colab disconnect) ---
        save_json(history, os.path.join(results_dir, "training_history.json"))

        # --- Early stopping ---
        if patience >= config["early_stop_patience"]:
            print(
                f"\n🛑 Early stopping tại epoch {epoch}. "
                f"Best: epoch={history['best_epoch']}, "
                f"Combined F1={best_f1:.4f}"
            )
            break

    print(
        f"\n✅ Training xong. "
        f"Best Combined F1={best_f1:.4f} @ epoch {history['best_epoch']}"
    )
    return history
