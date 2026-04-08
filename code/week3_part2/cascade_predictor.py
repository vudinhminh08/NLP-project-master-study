"""
cascade_predictor.py — Hybrid inference: PhoBERT + LLM cascade.

Luồng:
    review
      -> PhoBERT predict 34 aspects + confidence per head
      -> Nếu có WEAK_ASPECT confidence < threshold
          -> LLM RAG k=4 predict
          -> Override những weak aspects uncertain
      -> Output: final prediction array [34]
"""

import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import ASPECT_COLUMNS, WEAK_ASPECTS  # noqa: E402
from prompts import build_prompt, parse_llm_output, labels_dict_to_array  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from rag_retriever import ABSARetriever  # noqa: E402
from icl_predictor import df_to_examples  # noqa: E402


WEAK_ASPECT_INDICES = [
    ASPECT_COLUMNS.index(aspect) for aspect in WEAK_ASPECTS if aspect in ASPECT_COLUMNS
]


def predict_single_phobert(
    processed_text: str,
    model,
    tokenizer,
    device: torch.device,
    max_len: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict 1 review với PhoBERT.

    Returns:
        preds: [34] argmax per head
        confidences: [34] max softmax score per head
    """
    inputs = tokenizer(
        processed_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask"}
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs["logits"]
    preds = np.array([logit.argmax(dim=-1)[0].item() for logit in logits], dtype=np.int64)
    confidences = np.array(
        [F.softmax(logit, dim=-1)[0].max().item() for logit in logits],
        dtype=np.float32,
    )
    return preds, confidences


def predict_with_cascade(
    review_text: str,
    processed_text: str,
    model,
    tokenizer,
    device: torch.device,
    train_df: pd.DataFrame,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    threshold: float = 0.60,
    k: int = 4,
    sleep_sec: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """
    Cascade prediction cho 1 review.

    Returns:
        final_preds: [34]
        used_llm: True nếu đã trigger LLM
    """
    phobert_preds, confidences = predict_single_phobert(
        processed_text=processed_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    uncertain_weak = [idx for idx in WEAK_ASPECT_INDICES if confidences[idx] < threshold]
    if not uncertain_weak:
        return phobert_preds, False

    try:
        indices = retriever.retrieve(query=processed_text, k=k, aspect_aware=True)
        examples = df_to_examples(train_df, indices)
        messages = build_prompt(processed_text, examples)
        raw = llm_client.complete(messages, temperature=0.0, use_cache=True)
        llm_pred_dict = parse_llm_output(raw)
        llm_preds = np.array(labels_dict_to_array(llm_pred_dict), dtype=np.int64)
        time.sleep(sleep_sec)
    except Exception as exc:
        print(f"  [Cascade] LLM call failed: {exc} -> fallback PhoBERT only")
        return phobert_preds, False

    final_preds = phobert_preds.copy()
    for idx in uncertain_weak:
        if llm_preds[idx] != 0:
            final_preds[idx] = llm_preds[idx]

    return final_preds, True


def run_cascade_on_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    threshold: float = 0.60,
    k: int = 4,
    sleep_sec: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Chạy cascade prediction trên toàn bộ test set.

    Returns:
        y_true: [N, 34]
        y_pred: [N, 34]
        stats: llm_call_count, total, trigger_rate
    """
    y_true_list = []
    y_pred_list = []
    llm_call_count = 0

    for idx, (_, row) in enumerate(test_df.iterrows()):
        review_text = str(row.get("Review", ""))
        processed_text = str(row.get("processed_review", review_text))

        final_preds, used_llm = predict_with_cascade(
            review_text=review_text,
            processed_text=processed_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            train_df=train_df,
            retriever=retriever,
            llm_client=llm_client,
            threshold=threshold,
            k=k,
            sleep_sec=sleep_sec,
        )

        y_true_list.append(np.array([int(row[aspect]) for aspect in ASPECT_COLUMNS], dtype=np.int64))
        y_pred_list.append(final_preds)
        if used_llm:
            llm_call_count += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(test_df)}] LLM calls: {llm_call_count}")

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    stats = {
        "llm_call_count": llm_call_count,
        "total": len(test_df),
        "trigger_rate": llm_call_count / max(1, len(test_df)),
        "threshold": threshold,
        "k": k,
        "weak_aspects": WEAK_ASPECTS,
    }
    return y_true, y_pred, stats
