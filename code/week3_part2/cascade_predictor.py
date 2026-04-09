"""
cascade_predictor.py — Hybrid inference: PhoBERT + LLM cascade.

Ý tưởng:
    PhoBERT predict toàn bộ 34 aspects.
    Nếu confidence thấp trên WEAK_ASPECTS thì gọi LLM + RAG k=4 để override
    có chọn lọc cho đúng các aspects yếu đó.
"""

import os
import sys
import time
from collections import Counter
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
    ASPECT_COLUMNS.index(aspect)
    for aspect in WEAK_ASPECTS
    if aspect in ASPECT_COLUMNS
]


def predict_single_phobert(
    processed_text: str,
    model,
    tokenizer,
    device: torch.device,
    max_len: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict 1 review với PhoBERT.

    Returns:
        preds: [34] argmax per head
        confidences: [34] max softmax score per head
        probabilities: [34, 4] softmax probabilities per head
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
    probabilities = np.stack(
        [F.softmax(logit, dim=-1)[0].detach().cpu().numpy() for logit in logits],
        axis=0,
    )
    preds = np.array(
        [logit.argmax(dim=-1)[0].item() for logit in logits],
        dtype=np.int64,
    )
    confidences = probabilities.max(axis=1).astype(np.float32)
    return preds, confidences, probabilities


def _build_cascade_messages(
    review_text: str,
    examples: list[dict],
    target_aspects: list[str],
    retry_strict: bool = False,
) -> list[dict]:
    """
    Build a more constrained prompt for cascade.

    Compared with the generic week3 prompt, this version:
    - disables CoT to reduce verbose outputs that break JSON parsing
    - asks the LLM to focus only on uncertain weak aspects
    """
    messages = build_prompt(review_text, examples, include_cot=False)
    target_block = ", ".join(target_aspects)
    strict_tail = (
        "CHI TRA VE JSON object thuan, khong giai thich, khong markdown, "
        "khong liet ke buoc suy nghi."
        if retry_strict else
        "Chi tra ve JSON object, khong can giai thich."
    )
    messages[-1]["content"] = (
        f"Review: {review_text}\n"
        f"Chi tap trung vao cac aspect sau: {target_block}\n"
        "Neu review KHONG de cap aspect nao trong danh sach tren thi tra ve {}.\n"
        "Chi dua vao output cac aspect trong danh sach tren va duoc de cap ro rang.\n"
        "Gia tri hop le chi duoc la: positive, negative, neutral. "
        "KHONG duoc dung gia tri absent trong JSON output.\n"
        f"{strict_tail}"
    )
    return messages


def _should_query_llm(
    pred_label: int,
    probs: np.ndarray,
    threshold: float,
    absent_threshold: float,
    min_non_absent_prob: float,
    margin_threshold: float,
) -> bool:
    """
    Decide whether a weak aspect is uncertain enough to escalate to the LLM.

    We use a looser rule for aspects predicted as `absent`, because the weakest
    aspects in this project tend to be missed entirely rather than mislabeled.
    """
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1])
    margin = top1 - top2

    if pred_label == 0:
        absent_prob = float(probs[0])
        non_absent_prob = float(probs[1:].sum())
        return (
            absent_prob < absent_threshold
            or non_absent_prob >= min_non_absent_prob
            or margin < margin_threshold
        )

    return top1 < threshold or margin < margin_threshold


def _query_llm_predictions(
    review_text: str,
    examples: list[dict],
    target_aspects: list[str],
    llm_client: LLMClient,
    num_votes: int = 1,
) -> tuple[dict, list[str]]:
    """
    Query the LLM with a strict JSON-oriented prompt.

    If the first answer cannot be parsed into a useful dict for the target
    aspects, retry once with an even stricter instruction.
    """
    vote_outputs: list[str] = []
    vote_dicts: list[dict] = []

    for vote_idx in range(max(1, num_votes)):
        raw_output = ""
        # Shuffle example order per vote so the LLM sees slightly different context.
        if len(examples) > 1 and num_votes > 1:
            rng = np.random.default_rng(abs(hash((review_text, vote_idx))) % (2**32))
            perm = rng.permutation(len(examples))
            vote_examples = [examples[i] for i in perm]
        else:
            vote_examples = examples

        for retry_strict in (False, True):
            messages = _build_cascade_messages(
                review_text=review_text,
                examples=vote_examples,
                target_aspects=target_aspects,
                retry_strict=retry_strict,
            )
            raw_output = llm_client.complete(messages, temperature=0.0, use_cache=True)
            pred_dict = parse_llm_output(raw_output)
            filtered = {asp: pred_dict[asp] for asp in target_aspects if asp in pred_dict}
            if filtered or raw_output.strip() == "{}":
                vote_dicts.append(filtered)
                vote_outputs.append(raw_output)
                break
        else:
            vote_dicts.append({})
            vote_outputs.append(raw_output)

    merged: dict = {}
    for aspect in target_aspects:
        labels = [pred.get(aspect) for pred in vote_dicts if aspect in pred]
        if not labels:
            continue
        winner, winner_count = Counter(labels).most_common(1)[0]
        # Require only a simple plurality. With num_votes=3 this is either 2/3
        # or 1/3 when the other votes are empty, which is acceptable for this
        # exploratory cascade setting.
        if winner_count >= 1:
            merged[aspect] = winner

    return merged, vote_outputs


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
    absent_threshold: float = 0.90,
    min_non_absent_prob: float = 0.12,
    margin_threshold: float = 0.15,
    override_only_from_absent: bool = True,
    num_votes: int = 1,
    k: int = 4,
    max_len: int = 256,
) -> tuple[np.ndarray, bool, dict]:
    """
    Cascade prediction cho 1 review.

    Returns:
        final_preds: [34]
        used_llm: True nếu trigger LLM
        debug_info: thông tin override/debug
    """
    phobert_preds, confidences, probabilities = predict_single_phobert(
        processed_text=processed_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=max_len,
    )

    uncertain_indices = [
        idx
        for idx in WEAK_ASPECT_INDICES
        if _should_query_llm(
            pred_label=int(phobert_preds[idx]),
            probs=probabilities[idx],
            threshold=threshold,
            absent_threshold=absent_threshold,
            min_non_absent_prob=min_non_absent_prob,
            margin_threshold=margin_threshold,
        )
    ]
    if not uncertain_indices:
        return phobert_preds, False, {
            "uncertain_count": 0,
            "uncertain_aspects": [],
            "overridden_aspects": [],
        }

    try:
        retrieved_indices = retriever.retrieve(
            query=processed_text,
            k=k,
            aspect_aware=True,
        )
        examples = df_to_examples(train_df, retrieved_indices)
        target_aspects = [ASPECT_COLUMNS[idx] for idx in uncertain_indices]
        llm_pred_dict, raw_outputs = _query_llm_predictions(
            review_text=review_text or processed_text,
            examples=examples,
            target_aspects=target_aspects,
            llm_client=llm_client,
            num_votes=num_votes,
        )
        llm_preds = np.array(labels_dict_to_array(llm_pred_dict), dtype=np.int64)
    except Exception as exc:
        return phobert_preds, False, {
            "uncertain_count": len(uncertain_indices),
            "uncertain_aspects": [ASPECT_COLUMNS[idx] for idx in uncertain_indices],
            "overridden_aspects": [],
            "llm_error": str(exc),
        }

    final_preds = phobert_preds.copy()
    overridden = []

    for idx in uncertain_indices:
        aspect_name = ASPECT_COLUMNS[idx]
        phobert_label = int(phobert_preds[idx])
        llm_label = int(llm_preds[idx])

        # Giữ safeguard hiện tại: chỉ override khi LLM khác PhoBERT
        # và dự đoán non-absent để tránh xóa detection của classifier.
        if (
            llm_label != phobert_label
            and llm_label != 0
            and (not override_only_from_absent or phobert_label == 0)
        ):
            final_preds[idx] = llm_label
            overridden.append(
                {
                    "aspect": aspect_name,
                    "phobert": phobert_label,
                    "llm": llm_label,
                    "confidence": float(confidences[idx]),
                    "non_absent_prob": float(probabilities[idx][1:].sum()),
                }
            )

    return final_preds, True, {
        "uncertain_count": len(uncertain_indices),
        "uncertain_aspects": [ASPECT_COLUMNS[idx] for idx in uncertain_indices],
        "overridden_aspects": overridden,
        "llm_returned_empty": len(llm_pred_dict) == 0,
        "raw_output_preview": raw_outputs[0][:200] if raw_outputs else "",
        "num_votes": num_votes,
        "override_only_from_absent": override_only_from_absent,
    }


def run_cascade_on_dataset(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    threshold: float = 0.60,
    absent_threshold: float = 0.90,
    min_non_absent_prob: float = 0.12,
    margin_threshold: float = 0.15,
    override_only_from_absent: bool = True,
    num_votes: int = 1,
    k: int = 4,
    max_len: int = 256,
    sleep_sec: float = 0.5,
    max_samples: Optional[int] = None,
    return_records: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Chạy cascade prediction trên toàn bộ dataset.
    """
    if max_samples is not None:
        test_df = test_df.head(max_samples)

    y_true_list = []
    y_pred_list = []
    llm_call_count = 0
    total_overrides = 0
    empty_llm_responses = 0
    weak_trigger_counts = {aspect: 0 for aspect in WEAK_ASPECTS}
    weak_override_counts = {aspect: 0 for aspect in WEAK_ASPECTS}
    sample_records = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        review_text = str(row.get("Review", ""))
        processed_text = str(row.get("processed_review", review_text))

        final_preds, used_llm, debug = predict_with_cascade(
            review_text=review_text,
            processed_text=processed_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            train_df=train_df,
            retriever=retriever,
            llm_client=llm_client,
            threshold=threshold,
            absent_threshold=absent_threshold,
            min_non_absent_prob=min_non_absent_prob,
            margin_threshold=margin_threshold,
            override_only_from_absent=override_only_from_absent,
            num_votes=num_votes,
            k=k,
            max_len=max_len,
        )

        y_true_list.append(
            np.array([int(row[aspect]) for aspect in ASPECT_COLUMNS], dtype=np.int64)
        )
        y_pred_list.append(final_preds)

        for aspect_name in debug.get("uncertain_aspects", []):
            if aspect_name in weak_trigger_counts:
                weak_trigger_counts[aspect_name] += 1

        if used_llm:
            llm_call_count += 1
            total_overrides += len(debug.get("overridden_aspects", []))
            if debug.get("llm_returned_empty"):
                empty_llm_responses += 1
            for override in debug.get("overridden_aspects", []):
                aspect_name = override["aspect"]
                if aspect_name in weak_override_counts:
                    weak_override_counts[aspect_name] += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        if return_records:
            sample_records.append(
                {
                    "sample_idx": idx,
                    "used_llm": used_llm,
                    "uncertain_aspects": debug.get("uncertain_aspects", []),
                    "override_count": len(debug.get("overridden_aspects", [])),
                    "overridden_aspects": debug.get("overridden_aspects", []),
                    "llm_error": debug.get("llm_error"),
                }
            )

        if (idx + 1) % 100 == 0:
            trigger_rate = llm_call_count / max(1, idx + 1)
            print(
                f"  [{idx + 1}/{len(test_df)}] "
                f"LLM calls: {llm_call_count} ({trigger_rate:.0%}) "
                f"Overrides: {total_overrides}"
            )

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    stats = {
        "total_samples": len(test_df),
        "llm_call_count": llm_call_count,
        "trigger_rate": llm_call_count / max(1, len(test_df)),
        "total_overrides": total_overrides,
        "avg_overrides_per_trigger": (
            total_overrides / llm_call_count if llm_call_count > 0 else 0.0
        ),
        "threshold": threshold,
        "absent_threshold": absent_threshold,
        "min_non_absent_prob": min_non_absent_prob,
        "margin_threshold": margin_threshold,
        "override_only_from_absent": override_only_from_absent,
        "num_votes": num_votes,
        "k": k,
        "weak_aspects": WEAK_ASPECTS,
        "weak_trigger_counts": weak_trigger_counts,
        "weak_override_counts": weak_override_counts,
        "empty_llm_responses": empty_llm_responses,
    }
    if return_records:
        stats["sample_records"] = sample_records
    return y_true, y_pred, stats


def run_cascade_on_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    threshold: float = 0.60,
    absent_threshold: float = 0.90,
    min_non_absent_prob: float = 0.12,
    margin_threshold: float = 0.15,
    override_only_from_absent: bool = True,
    num_votes: int = 1,
    k: int = 4,
    sleep_sec: float = 0.5,
    max_len: int = 256,
    max_samples: Optional[int] = None,
    return_records: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Backward-compatible alias cho notebook cũ.
    """
    return run_cascade_on_dataset(
        test_df=test_df,
        train_df=train_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        retriever=retriever,
        llm_client=llm_client,
        threshold=threshold,
        absent_threshold=absent_threshold,
        min_non_absent_prob=min_non_absent_prob,
        margin_threshold=margin_threshold,
        override_only_from_absent=override_only_from_absent,
        num_votes=num_votes,
        k=k,
        max_len=max_len,
        sleep_sec=sleep_sec,
        max_samples=max_samples,
        return_records=return_records,
    )
