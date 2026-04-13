"""
add_spc_predictor.py — Orchestrator: ADD-only + SPC Correction hybrid.

Pipeline:
    1. PhoBERT forward  → probs [N, 34, 4]
    2. ADD-only         → base_preds [N, 34]  (ACD protected, weak aspects added)
    3. SPC correction   → final_preds [N, 34] (uncertain SPC corrected by LLM)

Invariants:
    - ACD: LLM chỉ ADD absent→present, không bao giờ xóa present→absent
    - SPC: LLM chỉ sửa sentiment trong {1,2,3}, không thay đổi 0
    - Fallback PhoBERT cho mọi parse failure
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3_part2"))

from utils.constants import ASPECT_COLUMNS, WEAK_ASPECTS  # noqa: E402
from step4_eval import evaluate_predictions  # noqa: E402
from hybrid_predictor import (  # noqa: E402
    collect_logits_and_preds,
    run_hybrid_on_dataset,
    WEAK_ASPECT_INDICES,
)
from spc_corrector import SPCCorrector  # noqa: E402


class AddSPCPredictor:
    """
    Combined ADD-only + SPC correction predictor.

    Usage:
        predictor = AddSPCPredictor(model, train_df, retriever, llm_client, device)
        y_true, final_preds, stats = predictor.predict(
            test_df, test_loader,
            add_threshold=0.08,
            spc_entropy_threshold=0.8,
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_df: pd.DataFrame,
        retriever,
        llm_client,
        device: torch.device,
        weak_aspects: list[str] = WEAK_ASPECTS,
        weak_indices: list[int] = WEAK_ASPECT_INDICES,
        k_add: int = 4,
        k_spc: int = 4,
        max_spc_per_review: int = 5,
        sleep_sec: float = 0.3,
    ):
        self.model = model
        self.train_df = train_df
        self.retriever = retriever
        self.llm_client = llm_client
        self.device = device
        self.weak_aspects = weak_aspects
        self.weak_indices = weak_indices
        self.k_add = k_add
        self.k_spc = k_spc
        self.max_spc_per_review = max_spc_per_review
        self.sleep_sec = sleep_sec

    def get_phobert_probs(
        self, dataloader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PhoBERT inference → (probs [N,34,4], argmax_preds [N,34], y_true [N,34])."""
        return collect_logits_and_preds(self.model, dataloader, self.device)

    def predict(
        self,
        test_df: pd.DataFrame,
        test_loader: torch.utils.data.DataLoader,
        add_threshold: float = 0.08,
        spc_entropy_threshold: float = 0.8,
        exclude_aspects: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
        run_add_only: bool = True,
        run_spc_correction: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Full predict pipeline.

        Args:
            test_df:               DataFrame with "Review" and "processed_review" columns
            test_loader:           DataLoader for PhoBERT inference
            add_threshold:         ADD-only trigger threshold (non_absent_prob)
            spc_entropy_threshold: SPC correction trigger threshold (SPC entropy)
            exclude_aspects:       Aspects to exclude from evaluation
            max_samples:           Cap number of samples (for dev tuning)
            run_add_only:          Whether to run Component 1 (ADD-only)
            run_spc_correction:    Whether to run Component 2 (SPC correction)

        Returns:
            y_true [N, 34], final_preds [N, 34], stats dict
        """
        # --- Step 1: PhoBERT forward ---
        print("[Step 1] PhoBERT inference...")
        all_probs, argmax_preds, y_true = self.get_phobert_probs(test_loader)

        n = min(len(test_df), len(all_probs))
        if max_samples is not None:
            n = min(n, max_samples)

        texts = test_df["Review"].astype(str).tolist()[:n]
        processed_texts = test_df.get(
            "processed_review", test_df["Review"]
        ).astype(str).tolist()[:n]

        # --- Step 2: ADD-only (Component 1) ---
        if run_add_only:
            print(f"[Step 2] ADD-only (threshold={add_threshold})...")
            y_true_out, base_preds, add_stats = run_hybrid_on_dataset(
                test_df=test_df.iloc[:n].reset_index(drop=True),
                train_df=self.train_df,
                all_probs=all_probs[:n],
                all_preds=argmax_preds[:n],
                y_true=y_true[:n],
                retriever=self.retriever,
                llm_client=self.llm_client,
                weak_aspects=self.weak_aspects,
                weak_indices=self.weak_indices,
                add_threshold=add_threshold,
                k=self.k_add,
                sleep_sec=self.sleep_sec,
                max_samples=n,
            )
        else:
            base_preds = argmax_preds[:n].copy()
            y_true_out = y_true[:n]
            add_stats = {"skipped": True}
            print("[Step 2] ADD-only skipped.")

        # --- Step 3: SPC correction (Component 2) ---
        if run_spc_correction:
            print(f"[Step 3] SPC correction (entropy_threshold={spc_entropy_threshold})...")
            corrector = SPCCorrector(
                train_df=self.train_df,
                retriever=self.retriever,
                llm_client=self.llm_client,
                entropy_threshold=spc_entropy_threshold,
                k=self.k_spc,
                max_per_review=self.max_spc_per_review,
                sleep_sec=self.sleep_sec,
            )
            final_preds, spc_stats = corrector.correct_dataset(
                texts=texts,
                processed_texts=processed_texts,
                base_preds_all=base_preds,
                probs_all=all_probs[:n],
                max_samples=n,
            )
        else:
            final_preds = base_preds
            spc_stats = {"skipped": True}
            print("[Step 3] SPC correction skipped.")

        stats = {
            "add_threshold": add_threshold,
            "spc_entropy_threshold": spc_entropy_threshold,
            "add_stats": add_stats,
            "spc_stats": spc_stats,
        }
        return y_true_out, final_preds, stats

    def tune_spc_threshold(
        self,
        dev_df: pd.DataFrame,
        dev_loader: torch.utils.data.DataLoader,
        add_threshold: float = 0.08,
        entropy_thresholds: Optional[list[float]] = None,
        exclude_aspects: Optional[list[str]] = None,
        max_samples: int = 200,
    ) -> tuple[float, dict]:
        """
        Grid search for best SPC entropy threshold on dev set.

        Returns:
            best_threshold, results_per_threshold
        """
        thresholds = entropy_thresholds or [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
        results = {}

        # Get ADD-only base once (no need to rerun for each threshold)
        print("[Tuning] Running ADD-only base on dev set...")
        all_probs, argmax_preds, y_true = self.get_phobert_probs(dev_loader)
        n = min(len(dev_df), len(all_probs), max_samples)
        texts = dev_df["Review"].astype(str).tolist()[:n]
        processed_texts = dev_df.get(
            "processed_review", dev_df["Review"]
        ).astype(str).tolist()[:n]

        _, base_preds, _ = run_hybrid_on_dataset(
            test_df=dev_df.iloc[:n].reset_index(drop=True),
            train_df=self.train_df,
            all_probs=all_probs[:n],
            all_preds=argmax_preds[:n],
            y_true=y_true[:n],
            retriever=self.retriever,
            llm_client=self.llm_client,
            weak_aspects=self.weak_aspects,
            weak_indices=self.weak_indices,
            add_threshold=add_threshold,
            k=self.k_add,
            sleep_sec=self.sleep_sec,
            max_samples=n,
        )

        for threshold in thresholds:
            print(f"\n--- SPC entropy_threshold = {threshold} ---")
            corrector = SPCCorrector(
                train_df=self.train_df,
                retriever=self.retriever,
                llm_client=self.llm_client,
                entropy_threshold=threshold,
                k=self.k_spc,
                max_per_review=self.max_spc_per_review,
                sleep_sec=self.sleep_sec,
            )
            final_preds, spc_stats = corrector.correct_dataset(
                texts=texts,
                processed_texts=processed_texts,
                base_preds_all=base_preds,
                probs_all=all_probs[:n],
                max_samples=n,
            )
            metrics = evaluate_predictions(
                y_true[:n],
                final_preds,
                title=f"Dev SPC correction (entropy_threshold={threshold})",
                exclude_aspects=exclude_aspects,
            )
            results[threshold] = {
                "combined_f1": metrics["macro_combined_f1"],
                "acd_f1": metrics["macro_acd_f1"],
                "spc_f1": metrics["macro_spc_f1"],
                "routing_rate": spc_stats.get("routing_rate", 0.0),
                "llm_calls": spc_stats.get("total_llm_calls", 0),
                "parse_fail_rate": spc_stats.get("parse_fail_rate", 0.0),
                "total_changed": spc_stats.get("total_changed", 0),
            }
            print(
                f"  Combined F1: {results[threshold]['combined_f1']:.4f} "
                f"| ACD: {results[threshold]['acd_f1']:.4f} "
                f"| SPC: {results[threshold]['spc_f1']:.4f} "
                f"| Routing: {results[threshold]['routing_rate']:.0%} "
                f"| Changed: {results[threshold]['total_changed']}"
            )

        best_t = max(results, key=lambda t: results[t]["combined_f1"])
        print(
            f"\nBest entropy_threshold: {best_t} "
            f"→ Combined F1: {results[best_t]['combined_f1']:.4f}"
        )
        return float(best_t), results
