"""
run_week4_experiment.py — Entrypoint for Week 4: ADD-only + SPC Correction Hybrid.

Chạy 3 variants để có ablation table đầy đủ:
    A) PhoBERT argmax baseline (verify vs reported 0.5543)
    B) ADD-only only (Component 1, verify vs reported 0.5568)
    C) ADD-only + SPC correction (Component 1 + 2, target ~0.57)

Outputs vào outputs/results/week4/:
    baseline_metrics.json
    add_only_metrics.json
    add_spc_metrics.json
    threshold_sweep_results.json
    spc_correction_log.json

Usage (trong notebook hoặc script):
    from code.week4.run_week4_experiment import run_week4
    results = run_week4(config)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3_part2"))

from utils.constants import ZERO_TRAIN_ASPECTS, WEAK_ASPECTS  # noqa: E402
from utils.helpers import save_json  # noqa: E402
from step4_eval import evaluate_predictions  # noqa: E402
from model import ABSAPhoBERT  # noqa: E402
from predict import load_best_model  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from rag_retriever import ABSARetriever  # noqa: E402
from hybrid_predictor import collect_logits_and_preds, WEAK_ASPECT_INDICES  # noqa: E402
from add_spc_predictor import AddSPCPredictor  # noqa: E402
from phobert_guided_verifier import (  # noqa: E402
    GuidedVerifierConfig,
    run_guided_verifier_on_dataset,
)


@dataclass
class Week4Config:
    # Paths
    checkpoint_path: str = "outputs/results/week2_results_VNcoreNLP/models_cls_only/best_model.pt"
    output_dir: str = "outputs/results/week4"
    embeddings_cache: str = "outputs/results/embeddings_cache.npy"

    # LLM
    llm_provider: str = "openai"           # "openai" or "gemini"
    llm_api_key: Optional[str] = None      # None → reads from env

    # ADD-only (Component 1)
    run_add_only_variant: bool = True
    add_threshold: float = 0.08            # best from hybrid grid search

    # SPC correction (Component 2) — tuned on dev, applied on test
    run_add_spc_variant: bool = True
    spc_entropy_threshold: float = 0.8     # starting point, tuned in tune step
    k_spc: int = 4                         # RAG examples for SPC prompt
    max_spc_per_review: int = 5            # cap LLM calls per review

    # Tuning
    tune_on_dev: bool = True
    entropy_thresholds_to_sweep: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    )
    tune_max_samples: int = 200            # dev subset for fast tuning

    # Misc
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
    sleep_sec: float = 0.3                 # between LLM calls
    max_test_samples: Optional[int] = None

    # Variant D — PhoBERT-guided LLM verifier
    run_guided_verifier: bool = True
    guided_enable_add: bool = False
    guided_enable_spc: bool = True
    guided_add_threshold: float = 0.08
    guided_spc_entropy_threshold: float = 0.55
    guided_k_rag: int = 6
    guided_max_candidates_per_review: int = 8
    guided_delete_enabled: bool = False
    guided_require_evidence_for_add: bool = True
    guided_require_evidence_for_spc: bool = False
    guided_apply_label_prior: bool = True
    guided_min_train_count_for_sentiment: int = 2
    guided_allow_neutral_if_train_count_at_least: int = 4
    guided_return_records: bool = True


def run_week4(
    config: Week4Config,
    train_df,
    dev_df,
    test_df,
    dev_loader,
    test_loader,
    class_weights,
) -> dict:
    """
    Main experiment runner.

    Args:
        config:        Week4Config
        train_df:      Train DataFrame (for RAG retrieval)
        dev_df:        Dev DataFrame
        test_df:       Test DataFrame
        dev_loader:    DataLoader for dev set
        test_loader:   DataLoader for test set
        class_weights: List of 34 weight tensors (from week2 setup)

    Returns:
        all_results dict with metrics for each variant
    """
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device_str)
    exclude = ZERO_TRAIN_ASPECTS

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[Setup] Loading PhoBERT checkpoint...")
    model = ABSAPhoBERT(encoder_option="cls_only")
    model = load_best_model(config.checkpoint_path, model, device)

    # ------------------------------------------------------------------
    # Setup RAG + LLM
    # ------------------------------------------------------------------
    print("[Setup] Initializing RAG retriever...")
    retriever = ABSARetriever(cache_path=config.embeddings_cache)
    retriever.fit(train_df)

    print("[Setup] Initializing LLM client...")
    llm_client = LLMClient(
        provider=config.llm_provider,
        api_key=config.llm_api_key,
    )

    # ------------------------------------------------------------------
    # Variant A: PhoBERT argmax baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Variant A] PhoBERT argmax baseline")
    all_probs_test, argmax_preds_test, y_true_test = collect_logits_and_preds(
        model, test_loader, device
    )
    baseline_metrics = evaluate_predictions(
        y_true_test, argmax_preds_test,
        title="Baseline — PhoBERT argmax",
        exclude_aspects=exclude,
        save_path=os.path.join(config.output_dir, "baseline_metrics.json"),
    )
    print(f"  Combined F1: {baseline_metrics['macro_combined_f1']:.4f}")

    # ------------------------------------------------------------------
    # Setup AddSPCPredictor only when old B/C variants are requested.
    # ------------------------------------------------------------------
    predictor = None
    if config.run_add_only_variant or config.run_add_spc_variant:
        predictor = AddSPCPredictor(
            model=model,
            train_df=train_df,
            retriever=retriever,
            llm_client=llm_client,
            device=device,
            weak_aspects=WEAK_ASPECTS,
            weak_indices=WEAK_ASPECT_INDICES,
            k_add=4,
            k_spc=config.k_spc,
            max_spc_per_review=config.max_spc_per_review,
            sleep_sec=config.sleep_sec,
        )

    # ------------------------------------------------------------------
    # Variant B: ADD-only (Component 1 only)
    # ------------------------------------------------------------------
    add_only_metrics = None
    stats_b = {"skipped": True}
    if config.run_add_only_variant:
        print("\n" + "=" * 60)
        print(f"[Variant B] ADD-only (add_threshold={config.add_threshold})")
        y_true_b, preds_b, stats_b = predictor.predict(
            test_df=test_df,
            test_loader=test_loader,
            add_threshold=config.add_threshold,
            spc_entropy_threshold=999.0,   # effectively disabled
            exclude_aspects=exclude,
            max_samples=config.max_test_samples,
            run_add_only=True,
            run_spc_correction=False,
        )
        add_only_metrics = evaluate_predictions(
            y_true_b, preds_b,
            title="Variant B — ADD-only",
            exclude_aspects=exclude,
            save_path=os.path.join(config.output_dir, "add_only_metrics.json"),
        )
        print(f"  Combined F1: {add_only_metrics['macro_combined_f1']:.4f}")
    else:
        print("\n[Variant B] ADD-only skipped by config.")

    # ------------------------------------------------------------------
    # Tune SPC entropy threshold on dev (if requested)
    # ------------------------------------------------------------------
    best_entropy_threshold = config.spc_entropy_threshold

    if config.run_add_spc_variant and config.tune_on_dev:
        print("\n" + "=" * 60)
        print("[Tuning] SPC entropy threshold on dev set...")
        best_entropy_threshold, sweep_results = predictor.tune_spc_threshold(
            dev_df=dev_df,
            dev_loader=dev_loader,
            add_threshold=config.add_threshold,
            entropy_thresholds=config.entropy_thresholds_to_sweep,
            exclude_aspects=exclude,
            max_samples=config.tune_max_samples,
        )
        save_json(
            {"best_threshold": best_entropy_threshold, "sweep": sweep_results},
            os.path.join(config.output_dir, "threshold_sweep_results.json"),
        )
        print(f"  Best entropy_threshold: {best_entropy_threshold}")

    # ------------------------------------------------------------------
    # Variant C: ADD-only + SPC correction
    # ------------------------------------------------------------------
    add_spc_metrics = None
    stats_c = {"skipped": True}
    if config.run_add_spc_variant:
        print("\n" + "=" * 60)
        print(
            f"[Variant C] ADD-only + SPC correction "
            f"(entropy_threshold={best_entropy_threshold})"
        )
        y_true_c, preds_c, stats_c = predictor.predict(
            test_df=test_df,
            test_loader=test_loader,
            add_threshold=config.add_threshold,
            spc_entropy_threshold=best_entropy_threshold,
            exclude_aspects=exclude,
            max_samples=config.max_test_samples,
            run_add_only=True,
            run_spc_correction=True,
        )
        add_spc_metrics = evaluate_predictions(
            y_true_c, preds_c,
            title="Variant C — ADD-only + SPC correction",
            exclude_aspects=exclude,
            save_path=os.path.join(config.output_dir, "add_spc_metrics.json"),
        )
        print(f"  Combined F1: {add_spc_metrics['macro_combined_f1']:.4f}")
    else:
        print("[Variant C] ADD-only + SPC correction skipped by config.")

    # Save SPC correction log
    if "spc_stats" in stats_c and not stats_c["spc_stats"].get("skipped"):
        spc_log = {
            k: v for k, v in stats_c["spc_stats"].items()
            if k != "sample_records"   # exclude verbose per-sample log from summary
        }
        save_json(
            stats_c["spc_stats"],
            os.path.join(config.output_dir, "spc_correction_log.json"),
        )
        print(
            f"  Routing rate: {spc_log.get('routing_rate', 0):.0%} "
            f"| Changed: {spc_log.get('total_changed', 0)} "
            f"| Parse fail: {spc_log.get('parse_fail_rate', 0):.0%}"
        )

    # ------------------------------------------------------------------
    # Variant D: PhoBERT-guided LLM verifier
    # ------------------------------------------------------------------
    guided_metrics = None
    stats_d = {"skipped": True}
    if config.run_guided_verifier:
        print("\n" + "=" * 60)
        print("[Variant D] PhoBERT-guided LLM verifier")
        guided_config = GuidedVerifierConfig(
            add_threshold=config.guided_add_threshold,
            spc_entropy_threshold=config.guided_spc_entropy_threshold,
            enable_add=config.guided_enable_add,
            enable_spc=config.guided_enable_spc,
            delete_enabled=config.guided_delete_enabled,
            k_rag=config.guided_k_rag,
            max_candidates_per_review=config.guided_max_candidates_per_review,
            sleep_sec=config.sleep_sec,
            require_evidence_for_add=config.guided_require_evidence_for_add,
            require_evidence_for_spc=config.guided_require_evidence_for_spc,
            apply_label_prior=config.guided_apply_label_prior,
            min_train_count_for_sentiment=config.guided_min_train_count_for_sentiment,
            allow_neutral_if_train_count_at_least=(
                config.guided_allow_neutral_if_train_count_at_least
            ),
        )
        y_true_d, preds_d, stats_d = run_guided_verifier_on_dataset(
            test_df=test_df,
            train_df=train_df,
            all_probs=all_probs_test,
            all_preds=argmax_preds_test,
            y_true=y_true_test,
            retriever=retriever,
            llm_client=llm_client,
            config=guided_config,
            max_samples=config.max_test_samples,
            return_records=config.guided_return_records,
        )
        guided_metrics = evaluate_predictions(
            y_true_d,
            preds_d,
            title="Variant D — PhoBERT-guided LLM verifier",
            exclude_aspects=exclude,
            save_path=os.path.join(config.output_dir, "guided_verifier_metrics.json"),
        )

        # Keep the summary compact, but save records separately for report examples.
        sample_records = stats_d.pop("sample_records", None)
        save_json(
            stats_d,
            os.path.join(config.output_dir, "guided_verifier_stats.json"),
        )
        if sample_records is not None:
            save_json(
                {"sample_records": sample_records},
                os.path.join(config.output_dir, "guided_verifier_records.json"),
            )

        print(
            f"  Combined F1: {guided_metrics['macro_combined_f1']:.4f} "
            f"| Routing: {stats_d.get('routing_rate', 0):.0%} "
            f"| Changed: {stats_d.get('total_changed', 0)} "
            f"| Parse fail: {stats_d.get('parse_fail_rate', 0):.0%}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_results = {
        "A_baseline": {
            "combined_f1": baseline_metrics["macro_combined_f1"],
            "acd_f1": baseline_metrics["macro_acd_f1"],
            "spc_f1": baseline_metrics["macro_spc_f1"],
        },
        "B_add_only": {
            "combined_f1": add_only_metrics["macro_combined_f1"],
            "acd_f1": add_only_metrics["macro_acd_f1"],
            "spc_f1": add_only_metrics["macro_spc_f1"],
        } if add_only_metrics is not None else {"skipped": True},
        "C_add_spc": {
            "combined_f1": add_spc_metrics["macro_combined_f1"],
            "acd_f1": add_spc_metrics["macro_acd_f1"],
            "spc_f1": add_spc_metrics["macro_spc_f1"],
            "spc_entropy_threshold": best_entropy_threshold,
        } if add_spc_metrics is not None else {"skipped": True},
        "D_guided_verifier": (
            {
                "combined_f1": guided_metrics["macro_combined_f1"],
                "acd_f1": guided_metrics["macro_acd_f1"],
                "spc_f1": guided_metrics["macro_spc_f1"],
                "stats": stats_d,
            }
            if guided_metrics is not None
            else {"skipped": True}
        ),
        "llm_usage": llm_client.get_usage_stats(),
    }
    save_json(
        all_results,
        os.path.join(config.output_dir, "week4_summary.json"),
    )

    print("\n" + "=" * 60)
    print("[Summary]")
    print(f"  A (baseline):    Combined F1 = {all_results['A_baseline']['combined_f1']:.4f}")
    if add_only_metrics is not None:
        print(f"  B (ADD-only):    Combined F1 = {all_results['B_add_only']['combined_f1']:.4f}")
    else:
        print("  B (ADD-only):    skipped")
    if add_spc_metrics is not None:
        print(f"  C (ADD+SPC):     Combined F1 = {all_results['C_add_spc']['combined_f1']:.4f}")
    else:
        print("  C (ADD+SPC):     skipped")
    if guided_metrics is not None:
        print(
            "  D (guided):     "
            f"Combined F1 = {all_results['D_guided_verifier']['combined_f1']:.4f}"
        )

    return all_results
