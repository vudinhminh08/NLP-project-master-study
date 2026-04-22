
import os
import sys
import json
import time
import heapq
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


    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
        eps=1e-8,
    )

    total_steps   = len(train_loader) * config["max_epochs"]
    warmup_steps  = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    scaler = GradScaler() if (use_amp and AMP_AVAILABLE and device.type == "cuda") else None
    amp_active = scaler is not None


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] {n_params:,} trainable parameters")
    print(f"[Scheduler] Total={total_steps} optimizer steps, Warmup={warmup_steps}")
    print(f"[LR] Adam lr={config['learning_rate']:.2e}, scheduler=cosine_warmup")
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
        "best_metric":      "macro_combined_f1",
        "config":           config,
        "use_amp":          amp_active,
    }
    best_combined = -float("inf")
    best_loss = float("inf")
    patience  = 0
    top_k = []
    top_k_size = int(config.get("top_k_checkpoints", 3))

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─'*60}\nEpoch {epoch}/{config['max_epochs']}")


        train_loss, _, _ = run_epoch(
            model, train_loader, device, class_weights,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=config["grad_accumulation_steps"],
            is_train=True,
            use_amp=amp_active, scaler=scaler,
        )


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


        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_acd_f1"].append(metrics["macro_acd_f1"])
        history["dev_spc_f1"].append(metrics["macro_spc_f1"])
        history["dev_combined_f1"].append(combined)

        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "dev_loss":         dev_loss,
            "combined_f1":      combined,
            "acd_f1":           metrics["macro_acd_f1"],
            "spc_f1":           metrics["macro_spc_f1"],
            "config":           config,
        }
        ckpt_name = f"checkpoint_epoch_{epoch:03d}.pt"
        ckpt_path = os.path.join(save_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

        ckpt_meta = {
            "epoch": epoch,
            "path": ckpt_path,
            "filename": ckpt_name,
            "combined_f1": combined,
            "acd_f1": metrics["macro_acd_f1"],
            "spc_f1": metrics["macro_spc_f1"],
            "dev_loss": dev_loss,
        }
        heapq.heappush(top_k, (combined, epoch, ckpt_path, ckpt_meta))
        while len(top_k) > top_k_size:
            _, _, stale_path, _ = heapq.heappop(top_k)
            if os.path.exists(stale_path):
                os.remove(stale_path)

        top_k_sorted = [
            item[3] for item in sorted(top_k, key=lambda x: (x[0], x[1]), reverse=True)
        ]
        save_json(
            {"metric": "macro_combined_f1", "top_k": top_k_sorted},
            os.path.join(save_dir, "top_k_checkpoints.json"),
        )

        if combined > best_combined:
            best_combined = combined
            best_loss = dev_loss
            history["best_epoch"]       = epoch
            history["best_dev_loss"]    = best_loss
            history["best_combined_f1"] = combined
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(
                f"  Best model saved "
                f"(combined_f1={combined:.4f}, dev_loss={best_loss:.4f})"
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
        f"Best dev_loss={best_loss:.4f}, Combined F1={history['best_combined_f1']:.4f}"
        f" @ epoch {history['best_epoch']}"
    )
    return history
