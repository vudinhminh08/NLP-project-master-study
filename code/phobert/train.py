
import os
import sys
import json
import time
from typing import Optional

import torch
import numpy as np
from torch.optim import Adam
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm


try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))

from utils.constants import ASPECT_COLUMNS, ZERO_TRAIN_ASPECTS
from utils.helpers import set_seed, save_json, load_json
from step4_eval import evaluate_predictions


def load_class_weights(
    weights_path: str,
    weight_clip: float = 10.0,
    device: Optional[torch.device] = None,
) -> list:
    data = load_json(weights_path)
    per_aspect = data["per_aspect_weights"]
    device = device or torch.device("cpu")

    weights_list = []
    clipped_count = 0

    for aspect in ASPECT_COLUMNS:
        w_dict = per_aspect.get(aspect, {})
        w = []
        for i in range(4):
            raw = float(w_dict.get(str(i), 1.0))
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
    loss_mode: str = "joint",
) -> tuple:
    model.train() if is_train else model.eval()
    total_loss  = 0.0
    all_preds:  list = []
    all_labels: list = []


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


            if use_amp and AMP_AVAILABLE:
                with autocast():
                    out = model(
                        input_ids, attention_mask,
                        labels=labels,
                        class_weights=weights_on_device,
                        loss_mode=loss_mode,
                    )
                    loss = out["loss"]
            else:
                out = model(
                    input_ids, attention_mask,
                    labels=labels,
                    class_weights=weights_on_device,
                    loss_mode=loss_mode,
                )
                loss = out["loss"]

            total_loss += loss.item()


            if is_train:
                scaled_loss = loss / grad_accum

                if use_amp and AMP_AVAILABLE and scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()


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

                all_preds.append(out["preds"].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    mean_loss = total_loss / n_batches

    if is_train:
        return mean_loss, None, None

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    return mean_loss, y_true, y_pred


def _create_optimizer_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    total_steps: int,
    warmup_ratio: float,
) -> tuple:
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        eps=1e-8,
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )
    return optimizer, scheduler, warmup_steps


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
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    max_epochs = config["max_epochs"]
    acd_warmup_epochs = int(config.get("acd_warmup_epochs", 0))
    acd_warmup_epochs = max(0, min(acd_warmup_epochs, max_epochs))
    phase2_epochs = max_epochs - acd_warmup_epochs

    phase1_steps = len(train_loader) * acd_warmup_epochs
    phase2_steps = len(train_loader) * phase2_epochs

    selection_metric = config.get("selection_metric", "combined_f1").lower()
    if selection_metric not in {"combined_f1", "dev_loss"}:
        raise ValueError("selection_metric phải là 'combined_f1' hoặc 'dev_loss'")

    phase1_lr = float(config.get("phase1_learning_rate", config["learning_rate"]))
    phase2_lr = float(config.get("phase2_learning_rate", config["learning_rate"]))

    active_lr = phase1_lr if acd_warmup_epochs > 0 else phase2_lr
    active_steps = phase1_steps if acd_warmup_epochs > 0 else phase2_steps
    optimizer, scheduler, warmup_steps = _create_optimizer_scheduler(
        model=model,
        learning_rate=active_lr,
        total_steps=active_steps,
        warmup_ratio=config["warmup_ratio"],
    )


    scaler = GradScaler() if (use_amp and AMP_AVAILABLE and device.type == "cuda") else None
    amp_active = scaler is not None


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] {n_params:,} trainable parameters")
    print(f"[Scheduler] Active phase steps={active_steps}, Warmup={warmup_steps}")
    print(f"[LR] phase1={phase1_lr:.2e}, phase2={phase2_lr:.2e}, scheduler=cosine_warmup")
    print(f"[Selection] best checkpoint theo {selection_metric}")
    print(f"[Curriculum] ACD warmup epochs={acd_warmup_epochs}, joint epochs={phase2_epochs}")
    print(f"[AMP] Mixed precision: {'ON' if amp_active else 'OFF'}")
    print(f"[Config] encoder={config.get('encoder_option')}, "
          f"seq_len={config.get('max_seq_len')}, "
          f"batch={config['batch_size']}×{config['grad_accumulation_steps']}="
          f"{config['batch_size']*config['grad_accumulation_steps']} (effective)")


    history = {
        "train_loss":       [],
        "dev_loss":         [],
        "dev_acd_f1":       [],
        "dev_spc_f1":       [],
        "dev_combined_f1":  [],
        "best_epoch":       0,
        "best_dev_loss":    float("inf"),
        "best_combined_f1": 0.0,
        "selection_metric": selection_metric,
        "best_selection_value": None,
        "config":           config,
        "use_amp":          amp_active,
    }
    best_loss = float("inf")
    best_combined = -1.0
    patience  = 0

    for epoch in range(1, config["max_epochs"] + 1):
        stage = "acd_only" if epoch <= acd_warmup_epochs else "joint"
        if epoch == acd_warmup_epochs + 1 and phase2_epochs > 0:
            optimizer, scheduler, warmup_steps = _create_optimizer_scheduler(
                model=model,
                learning_rate=phase2_lr,
                total_steps=phase2_steps,
                warmup_ratio=config["warmup_ratio"],
            )
            print(
                f"\n[Phase Switch] Bắt đầu joint training"
                f" | steps={phase2_steps}, warmup={warmup_steps}, lr={phase2_lr:.2e}"
            )

        t0 = time.time()
        print(f"\n{'─'*60}\nEpoch {epoch}/{config['max_epochs']} ({stage})")


        train_loss, _, _ = run_epoch(
            model, train_loader, device, class_weights,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=config["grad_accumulation_steps"],
            is_train=True,
            use_amp=amp_active, scaler=scaler,
            loss_mode=stage,
        )


        dev_loss, y_true, y_pred = run_epoch(
            model, dev_loader, device, class_weights,
            is_train=False,
            use_amp=amp_active, scaler=None,
            loss_mode="joint",
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


        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_acd_f1"].append(metrics["macro_acd_f1"])
        history["dev_spc_f1"].append(metrics["macro_spc_f1"])
        history["dev_combined_f1"].append(combined)


        improved = False
        if selection_metric == "combined_f1":
            if combined > best_combined:
                best_combined = combined
                improved = True
        else:
            if dev_loss < best_loss:
                improved = True

        if improved:
            best_loss = dev_loss
            best_combined = max(best_combined, combined)
            history["best_epoch"]       = epoch
            history["best_dev_loss"]    = dev_loss
            history["best_combined_f1"] = combined
            history["best_selection_value"] = combined if selection_metric == "combined_f1" else dev_loss
            ckpt = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "dev_loss":         dev_loss,
                "combined_f1":      combined,
                "acd_f1":           metrics["macro_acd_f1"],
                "spc_f1":           metrics["macro_spc_f1"],
                "stage":            stage,
                "selection_metric": selection_metric,
                "config":           config,
            }
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(
                f"  Best model saved (dev_loss={dev_loss:.4f}, combined_f1={combined:.4f}, "
                f"selection={history['best_selection_value']:.4f})"
            )
            patience = 0
        else:
            patience += 1
            print(f"  No improvement [{patience}/{config['early_stop_patience']}]")


        save_json(history, os.path.join(results_dir, "training_history.json"))


        if patience >= config["early_stop_patience"]:
            print(
                f"\nEarly stopping tại epoch {epoch}. "
                f"Best: epoch={history['best_epoch']}, "
                f"dev_loss={best_loss:.4f}, Combined F1={history['best_combined_f1']:.4f}"
            )
            break

    print(
        f"\nTraining xong. "
        f"Best dev_loss={history['best_dev_loss']:.4f}, "
        f"Combined F1={history['best_combined_f1']:.4f}, "
        f"selection_metric={selection_metric}"
        f" @ epoch {history['best_epoch']}"
    )
    return history
