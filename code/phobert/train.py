
import os
import sys
import time
from typing import Optional

import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm


try:
    from torch import amp as torch_amp
    AMP_AVAILABLE = True

    def amp_autocast():
        return torch_amp.autocast(device_type="cuda")

    def make_grad_scaler(enabled: bool):
        return torch_amp.GradScaler("cuda", enabled=enabled)

except Exception:
    try:
        from torch.cuda.amp import autocast as cuda_autocast, GradScaler as CudaGradScaler
        AMP_AVAILABLE = True

        def amp_autocast():
            return cuda_autocast()

        def make_grad_scaler(enabled: bool):
            return CudaGradScaler(enabled=enabled)

    except Exception:
        AMP_AVAILABLE = False

        def amp_autocast():
            raise RuntimeError("AMP is not available in this environment")

        def make_grad_scaler(enabled: bool):
            return None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))

from utils.constants import ASPECT_COLUMNS, ZERO_TRAIN_ASPECTS, RARE_ASPECTS
from utils.helpers import set_seed, save_json, load_json
from step4_eval import evaluate_predictions


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict:
        backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            backup[name] = p.detach().clone()
            p.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(backup[name])


def snapshot_state_dict(model: torch.nn.Module) -> dict:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    base_lr = float(config["learning_rate"])
    head_mult = float(config.get("head_lr_mult", 5.0))
    weight_decay = float(config.get("weight_decay", 0.01))
    layerwise_lr_decay = float(config.get("layerwise_lr_decay", 1.0))

    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")

    num_layers = int(getattr(model.phobert.config, "num_hidden_layers", 12))

    def get_encoder_layer_id(param_name: str) -> int:
        # embeddings -> 0, encoder.layer.k -> k+1, others -> top
        if ".embeddings." in param_name:
            return 0
        marker = ".encoder.layer."
        if marker in param_name:
            suffix = param_name.split(marker, 1)[1]
            try:
                layer_idx = int(suffix.split(".", 1)[0])
                return layer_idx + 1
            except Exception:
                pass
        return num_layers + 1

    grouped_params = {}

    def add_param(param: torch.nn.Parameter, lr: float, wd: float) -> None:
        key = (lr, wd)
        if key not in grouped_params:
            grouped_params[key] = []
        grouped_params[key].append(param)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(term in name for term in no_decay_terms)
        is_encoder = name.startswith("phobert.")

        if is_encoder:
            layer_id = get_encoder_layer_id(name)
            if layerwise_lr_decay < 1.0:
                lr_scale = layerwise_lr_decay ** ((num_layers + 1) - layer_id)
            else:
                lr_scale = 1.0
            lr = base_lr * lr_scale
        else:
            lr = base_lr * head_mult

        wd = 0.0 if is_no_decay else weight_decay
        add_param(param, lr, wd)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in grouped_params.items()
    ]

    return AdamW(param_groups, eps=1e-8)


def load_class_weights(
    weights_path: str,
    weight_clip: float = 10.0,
    device: Optional[torch.device] = None,
    rare_mult: float = 1.0,
) -> list:
    data = load_json(weights_path)
    per_aspect = data["per_aspect_weights"]
    device = device or torch.device("cpu")

    weights_list = []
    clipped_count = 0

    for aspect in ASPECT_COLUMNS:
        w_dict = per_aspect.get(aspect, {})
        w = []
        is_rare = aspect in RARE_ASPECTS
        for i in range(4):
            raw = float(w_dict.get(str(i), 1.0))
            if is_rare and rare_mult != 1.0:
                raw = raw * rare_mult
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
    max_grad_norm: float = 1.0,
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
                with amp_autocast():
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                    ema = getattr(model, "_ema", None)
                    if ema is not None:
                        ema.update(model)
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

    optimizer = build_optimizer(model, config)

    grad_accum = max(1, int(config.get("grad_accumulation_steps", 1)))
    steps_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    total_steps   = steps_per_epoch * config["max_epochs"]
    warmup_steps  = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    scaler = make_grad_scaler(True) if (use_amp and AMP_AVAILABLE and device.type == "cuda") else None
    amp_active = scaler is not None

    use_ema = bool(config.get("use_ema", True))
    ema_decay = float(config.get("ema_decay", 0.999))
    if use_ema:
        model._ema = EMA(model, decay=ema_decay)
    else:
        model._ema = None


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] {n_params:,} trainable parameters")
    print(f"[Scheduler] Total={total_steps} optimizer steps, Warmup={warmup_steps}")
    print(
        f"[LR] AdamW encoder_lr={config['learning_rate']:.2e}, "
        f"head_lr={config['learning_rate'] * config.get('head_lr_mult', 5.0):.2e}, "
        f"wd={config.get('weight_decay', 0.01):.2e}, "
        f"llrd={config.get('layerwise_lr_decay', 1.0):.3f}, scheduler=cosine_warmup"
    )
    print(f"[AMP] Mixed precision: {'ON' if amp_active else 'OFF'}")
    print(f"[EMA] {'ON' if use_ema else 'OFF'} (decay={ema_decay:.4f})")
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
        "config":           config,
        "use_amp":          amp_active,
    }
    patience  = 0

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─'*60}\nEpoch {epoch}/{config['max_epochs']}")


        train_loss, _, _ = run_epoch(
            model, train_loader, device, class_weights,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=grad_accum,
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            is_train=True,
            use_amp=amp_active, scaler=scaler,
        )


        if model._ema is not None:
            backup = model._ema.apply_to(model)
        else:
            backup = None

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

        if backup is not None:
            model._ema.restore(model, backup)

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


        # Select checkpoint by Combined F1 on dev (primary metric)
        if combined > history.get("best_combined_f1", 0.0):
            history["best_epoch"]       = epoch
            history["best_dev_loss"]    = dev_loss
            history["best_combined_f1"] = combined
            if model._ema is not None:
                backup_ckpt = model._ema.apply_to(model)
                model_state = snapshot_state_dict(model)
                model._ema.restore(model, backup_ckpt)
            else:
                model_state = snapshot_state_dict(model)

            ckpt = {
                "epoch":            epoch,
                "model_state_dict": model_state,
                "dev_loss":         dev_loss,
                "combined_f1":      combined,
                "acd_f1":           metrics["macro_acd_f1"],
                "spc_f1":           metrics["macro_spc_f1"],
                "config":           config,
            }

            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"  Best model saved (combined_f1={combined:.4f}, dev_loss={dev_loss:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"  No improvement [{patience}/{config['early_stop_patience']}]")


        save_json(history, os.path.join(results_dir, "training_history.json"))


        if patience >= config["early_stop_patience"]:
            print(
                f"\nEarly stopping tại epoch {epoch}. "
                f"Best: epoch={history['best_epoch']}, "
                f"dev_loss={history['best_dev_loss']:.4f}, Combined F1={history['best_combined_f1']:.4f}"
            )
            break

    print(
        f"\nTraining xong. "
        f"Best dev_loss={history['best_dev_loss']:.4f}, Combined F1={history['best_combined_f1']:.4f}"
        f" @ epoch {history['best_epoch']}"
    )
    return history
