import os
import sys
from typing import List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))

from step4_eval import evaluate_predictions
from utils.constants import ZERO_TRAIN_ASPECTS
from utils.helpers import load_json


def load_top_k_checkpoint_paths(top_k_path: str) -> List[str]:
    data = load_json(top_k_path)
    entries = data if isinstance(data, list) else data.get("top_k", [])
    paths = [entry["path"] for entry in entries if entry.get("path")]
    return paths


def ensemble_predict_and_evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    checkpoint_paths: List[str],
    device: torch.device,
    split_name: str = "test",
    save_path: Optional[str] = None,
) -> tuple:
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")

    all_checkpoint_probs = []
    all_checkpoint_presence_probs = []
    split_mode = bool(getattr(model, "use_split_loss", False))
    tuned_thresholds = None
    if split_mode and hasattr(model, "aspect_presence_thresholds"):
        tuned_thresholds = model.aspect_presence_thresholds.detach().clone()
    y_true = None

    for checkpoint_path in checkpoint_paths:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        if tuned_thresholds is not None:
            with torch.no_grad():
                model.aspect_presence_thresholds.copy_(tuned_thresholds.to(device))

        probs_batches = []
        presence_batches = []
        labels_batches = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                out = model(input_ids, attention_mask)
                logits = torch.stack(out["logits"], dim=1)
                if split_mode and "presence_logits" in out:
                    sent_probs = torch.softmax(logits[:, :, 1:], dim=-1)
                    presence_logits = torch.stack(out["presence_logits"], dim=1)
                    presence_probs = torch.sigmoid(presence_logits)
                    probs_batches.append(sent_probs.cpu())
                    presence_batches.append(presence_probs.cpu())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    probs_batches.append(probs.cpu())
                labels_batches.append(batch["labels"].cpu())

        all_checkpoint_probs.append(torch.cat(probs_batches, dim=0))
        if split_mode:
            all_checkpoint_presence_probs.append(torch.cat(presence_batches, dim=0))
        if y_true is None:
            y_true = torch.cat(labels_batches, dim=0).numpy()

        print(
            f"[Ensemble] Loaded {checkpoint_path} "
            f"(epoch={ckpt.get('epoch')}, combined_f1={ckpt.get('combined_f1', 0):.4f})"
        )

    avg_probs = torch.stack(all_checkpoint_probs, dim=0).mean(dim=0)
    if split_mode:
        avg_presence_probs = torch.stack(all_checkpoint_presence_probs, dim=0).mean(dim=0)
        thresholds = model.aspect_presence_thresholds.detach().cpu().view(1, -1)
        present_pred = avg_presence_probs >= thresholds
        sent_pred = avg_probs.argmax(dim=-1) + 1
        y_pred = torch.where(
            present_pred,
            sent_pred,
            torch.zeros_like(sent_pred),
        ).numpy()
    else:
        y_pred = avg_probs.argmax(dim=-1).numpy()

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        title=f"Ensemble Top-K - {split_name.upper()}",
        exclude_aspects=ZERO_TRAIN_ASPECTS,
        save_path=save_path,
    )
    return metrics, y_true, y_pred


def ensemble_from_top_k_file(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    top_k_path: str,
    device: torch.device,
    split_name: str = "test",
    save_path: Optional[str] = None,
) -> tuple:
    checkpoint_paths = load_top_k_checkpoint_paths(top_k_path)
    return ensemble_predict_and_evaluate(
        model,
        dataloader,
        checkpoint_paths,
        device,
        split_name=split_name,
        save_path=save_path,
    )
