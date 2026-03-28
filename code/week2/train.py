"""
train.py — Training loop với early stopping, gradient clipping, LR scheduling.

Chạy từ root project (qua run_experiment.py):
    python code/week2/run_experiment.py
"""

import os
import sys
import json
import time
from typing import Optional

import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

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
    Clip tại weight_clip để tránh gradient instability với neutral (weight=154).

    Args:
        weights_path: đường dẫn tới outputs/eda/class_weights.json
        weight_clip:  giá trị tối đa cho mỗi weight (mặc định 10.0)
        device:       torch device để đặt tensor lên

    Returns:
        list of 34 tensors [4] — mỗi tensor là [w0, w1, w2, w3] cho 1 aspect
    """
    data = load_json(weights_path)
    per_aspect = data["per_aspect_weights"]
    device = device or torch.device("cpu")

    weights_list = []
    for aspect in ASPECT_COLUMNS:
        w_dict = per_aspect.get(aspect, {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0})
        w = [min(float(w_dict.get(str(i), 1.0)), weight_clip) for i in range(4)]
        weights_list.append(torch.tensor(w, dtype=torch.float32, device=device))

    clipped_count = sum(
        1 for asp in ASPECT_COLUMNS
        for i in range(4)
        if float(per_aspect.get(asp, {}).get(str(i), 0)) > weight_clip
    )
    print(f"[Weights] {len(weights_list)} aspects loaded, "
          f"{clipped_count} values clipped at {weight_clip}")
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
) -> tuple:
    """
    Chạy 1 epoch train hoặc eval.

    Args:
        model:         ABSAPhoBERT
        dataloader:    DataLoader (train hoặc dev/test)
        device:        torch device
        class_weights: list of 34 tensors [4]
        optimizer:     AdamW optimizer (None khi eval)
        scheduler:     LR scheduler (None khi eval)
        grad_accum:    gradient accumulation steps
        is_train:      True để train, False để eval

    Returns:
        (mean_loss, y_true, y_pred)
        y_true/y_pred là np.ndarray [N, 34] khi eval, None khi train
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "Train" if is_train else "Eval"

    with ctx:
        if is_train:
            optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out = model(
                input_ids, attention_mask,
                labels=labels,
                class_weights=class_weights,
            )
            loss = out["loss"]

            if is_train:
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                all_preds.append(out["preds"].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            total_loss += loss.item()

    mean_loss = total_loss / len(dataloader)

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
) -> dict:
    """
    Full training loop với early stopping.

    Monitor: dev combined_f1 (ACD+SPC average), exclude ZERO_TRAIN_ASPECTS.
    Save: best checkpoint + training_history.json mỗi epoch (an toàn khi Colab disconnect).

    Args:
        model:        ABSAPhoBERT
        train_loader: DataLoader train
        dev_loader:   DataLoader dev
        class_weights: list of 34 weight tensors
        device:       torch device
        config:       dict từ TRAIN_CONFIG (có thể override từ encoder_config.json)
        save_dir:     thư mục lưu best_model.pt
        results_dir:  thư mục lưu training_history.json

    Returns:
        history dict với train_loss, dev_loss, dev_f1 theo epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
        eps=1e-8,
    )

    # Setup scheduler: warmup + linear decay
    steps_per_epoch = len(train_loader) // config["grad_accumulation_steps"]
    total_steps  = steps_per_epoch * config["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"[Scheduler] Total={total_steps} steps, Warmup={warmup_steps} steps")
    print(f"[Model] {sum(p.numel() for p in model.parameters()):,} parameters")

    history = {
        "train_loss":       [],
        "dev_loss":         [],
        "dev_acd_f1":       [],
        "dev_spc_f1":       [],
        "dev_combined_f1":  [],
        "best_epoch":       0,
        "best_combined_f1": 0.0,
        "config":           config,
    }
    best_f1 = 0.0
    patience = 0

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─'*60}\nEpoch {epoch}/{config['max_epochs']}")

        # Train
        train_loss, _, _ = run_epoch(
            model, train_loader, device, class_weights,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=config["grad_accumulation_steps"],
            is_train=True,
        )

        # Eval
        dev_loss, y_true, y_pred = run_epoch(
            model, dev_loader, device, class_weights, is_train=False
        )
        metrics = evaluate_predictions(
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

        # Log history
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_acd_f1"].append(metrics["macro_acd_f1"])
        history["dev_spc_f1"].append(metrics["macro_spc_f1"])
        history["dev_combined_f1"].append(combined)

        # Save best checkpoint
        if combined > best_f1:
            best_f1 = combined
            history["best_epoch"] = epoch
            history["best_combined_f1"] = best_f1
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "combined_f1":      combined,
                    "config":           config,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"  ✅ Best model saved (combined_f1={best_f1:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"  ⚠️  No improvement [{patience}/{config['early_stop_patience']}]")

        # Lưu history mỗi epoch (an toàn khi Colab disconnect)
        save_json(history, os.path.join(results_dir, "training_history.json"))

        # Early stopping
        if patience >= config["early_stop_patience"]:
            print(
                f"\n🛑 Early stopping at epoch {epoch}. "
                f"Best epoch={history['best_epoch']}, F1={best_f1:.4f}"
            )
            break

    print(
        f"\n✅ Training done. "
        f"Best Combined F1={best_f1:.4f} @ epoch {history['best_epoch']}"
    )
    return history
