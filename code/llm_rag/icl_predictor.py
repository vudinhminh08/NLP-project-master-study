"""
icl_predictor.py — In-Context Learning với random example selection.

Ablation: k = 2, 4, 8 examples
Provider: openai, gemini
"""

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from utils.constants import ASPECT_COLUMNS, LABEL_TO_IDX, IDX_TO_LABEL
from utils.helpers import set_seed, save_json
from step4_eval import evaluate_predictions

from prompts import build_prompt, parse_llm_output, labels_dict_to_array
from llm_client import LLMClient


def df_to_examples(df: pd.DataFrame, indices: list[int]) -> list[dict]:
    """
    Chuyển DataFrame rows → list examples cho prompt.

    Returns:
        list of {"review": str, "labels": {aspect: sentiment}}
        Chỉ gồm aspects PRESENT (label != 0)
    """
    examples = []
    for idx in indices:
        row = df.iloc[idx]
        labels = {}
        for asp in ASPECT_COLUMNS:
            val = int(row[asp])
            if val != 0:
                labels[asp] = IDX_TO_LABEL[val]
        examples.append({"review": str(row["processed_review"]), "labels": labels})
    return examples


def predict_icl(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    client: LLMClient,
    k: int = 4,
    seed: int = 42,
    max_samples: int = None,
    sleep_sec: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random few-shot ICL prediction.

    Args:
        test_df:     test DataFrame
        train_df:    train DataFrame (pool to sample examples from)
        client:      LLMClient
        k:           số examples per prompt
        seed:        random seed
        max_samples: giới hạn số test samples (None = tất cả, dùng 100 để test nhanh)

    Returns:
        (y_true [N, 34], y_pred [N, 34])
    """
    set_seed(seed)
    test_df = test_df.head(max_samples) if max_samples else test_df

    y_true_list, y_pred_list = [], []

    for i, (_, test_row) in enumerate(tqdm(test_df.iterrows(),
                                            total=len(test_df),
                                            desc=f"ICL k={k}")):
        # Random sample k examples từ train
        indices = random.sample(range(len(train_df)), k)
        examples = df_to_examples(train_df, indices)

        # Build prompt + call LLM
        messages = build_prompt(str(test_row["processed_review"]), examples)
        raw_output = client.complete(messages)
        pred_dict = parse_llm_output(raw_output)

        # Convert to array
        y_pred_list.append(labels_dict_to_array(pred_dict))

        # Ground truth
        true_arr = [int(test_row[asp]) for asp in ASPECT_COLUMNS]
        y_true_list.append(true_arr)

        import time; time.sleep(sleep_sec)

    return np.array(y_true_list), np.array(y_pred_list)


def run_icl_ablation(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    providers: list[str],
    k_values: list[int],
    api_keys: dict,
    results_dir: str = "outputs/results",
    max_samples: int = None,
    models: dict = None,
    sleep_sec: float = 7.0,
) -> dict:
    """
    Chạy ablation: k=2,4,8 × GPT+Gemini.

    Args:
        providers: ["openai", "gemini"]
        k_values:  [2, 4, 8]
        api_keys:  {"openai": "sk-...", "gemini": "AIza..."}
        models:    {"gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini"} (optional)

    Returns:
        dict tất cả kết quả
    """
    os.makedirs(results_dir, exist_ok=True)
    models = models or {}
    all_results = {}

    for provider in providers:
        client = LLMClient(
            provider=provider,
            api_key=api_keys.get(provider),
            model=models.get(provider),
        )
        for k in k_values:
            exp_name = f"icl_{provider}_k{k}"
            print(f"\n{'─'*50}")
            print(f"Running: {exp_name}")

            y_true, y_pred = predict_icl(
                test_df, train_df, client, k=k, seed=42,
                max_samples=max_samples, sleep_sec=sleep_sec,
            )
            metrics = evaluate_predictions(
                y_true, y_pred,
                title=f"ICL — {provider} k={k}",
                save_path=f"{results_dir}/{exp_name}_metrics.json",
            )
            all_results[exp_name] = metrics

            print(f"  Combined F1: {metrics['macro_combined_f1']:.4f}")

        # Log API usage
        usage = client.get_usage_stats()
        print(f"\n[Usage] {provider}: {usage['total_tokens']} tokens, "
              f"~${usage['estimated_cost_usd']:.3f}")

    return all_results
