"""
rag_predictor.py — Tầng 3: RAG + LLM prediction.

Cùng interface với icl_predictor nhưng dùng retrieval thay vì random.
So sánh fair: cùng LLM, cùng k, cùng prompt format.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))
from utils.constants import ASPECT_COLUMNS
from utils.helpers import set_seed, save_json
from step4_eval import evaluate_predictions

from prompts import build_prompt, parse_llm_output, labels_dict_to_array
from llm_client import LLMClient
from rag_retriever import ABSARetriever
from icl_predictor import df_to_examples


def predict_rag(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    client: LLMClient,
    retriever: ABSARetriever,
    k: int = 4,
    aspect_aware: bool = True,
    max_samples: int = None,
    sleep_sec: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tầng 3: RAG prediction với semantic retrieval.

    Khác icl_predictor duy nhất 1 điểm: cách chọn examples
    → random (ICL) vs semantic retrieval (RAG)
    """
    test_df = test_df.head(max_samples) if max_samples else test_df

    y_true_list, y_pred_list = [], []

    for i, (_, test_row) in enumerate(tqdm(test_df.iterrows(),
                                            total=len(test_df),
                                            desc=f"RAG k={k}")):
        # Retrieve k examples (khác biệt chính so với ICL)
        query = str(test_row["processed_review"])
        indices = retriever.retrieve(
            query=query, k=k, aspect_aware=aspect_aware
        )
        examples = df_to_examples(train_df, indices)

        # Phần còn lại giống hệt ICL
        messages = build_prompt(query, examples)
        raw_output = client.complete(messages)
        pred_dict = parse_llm_output(raw_output)

        y_pred_list.append(labels_dict_to_array(pred_dict))
        y_true_list.append([int(test_row[asp]) for asp in ASPECT_COLUMNS])

        time.sleep(sleep_sec)

    return np.array(y_true_list), np.array(y_pred_list)


def run_rag_ablation(
    test_df, train_df, providers, k_values, api_keys,
    results_dir="outputs/results", max_samples=None,
    models: dict = None,
    sleep_sec: float = 7.0,
) -> dict:
    """Chạy ablation RAG: k=2,4,8 × GPT+Gemini.

    Args:
        models: {"gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini"} (optional)
    """
    os.makedirs(results_dir, exist_ok=True)
    models = models or {}

    # Build retriever 1 lần, dùng cho tất cả experiments
    retriever = ABSARetriever()
    retriever.fit(train_df)

    all_results = {}
    for provider in providers:
        client = LLMClient(provider=provider, api_key=api_keys.get(provider),
                           model=models.get(provider))
        for k in k_values:
            exp_name = f"tier3_{provider}_k{k}"
            print(f"\n{'─'*50}\nRunning: {exp_name}")

            y_true, y_pred = predict_rag(
                test_df, train_df, client, retriever,
                k=k, max_samples=max_samples, sleep_sec=sleep_sec,
            )
            metrics = evaluate_predictions(
                y_true, y_pred,
                title=f"Tầng 3 — {provider} k={k}",
                save_path=f"{results_dir}/{exp_name}_metrics.json",
            )
            all_results[exp_name] = metrics
            print(f"  Combined F1: {metrics['macro_combined_f1']:.4f}")

    return all_results
