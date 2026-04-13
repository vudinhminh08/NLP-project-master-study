"""
spc_corrector.py — Component 2: SPC Correction for ADD-only + SPC Hybrid.

Chiến lược:
    Với mỗi aspect đã được predict là PRESENT (base_pred != 0):
    - Tính SPC entropy từ probs[i][1:4] (normalized về pos/neg/neu)
    - Nếu entropy > threshold → aspect này PhoBERT uncertain về sentiment
    - Gọi LLM với prompt focused: "Given aspect X is present, sentiment?"
    - LLM output: chỉ nhận positive/negative/neutral (không thay đổi ACD)
    - Fallback về PhoBERT nếu parse fail

Đảm bảo:
    - ACD không bị thay đổi: output chỉ có thể là 1/2/3 (không bao giờ → 0)
    - Parse fail → keep PhoBERT SPC
    - Capped tại max_per_review aspects per review (top entropy)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from rag_retriever import ABSARetriever  # noqa: E402


_SENTIMENT_MAP: dict[str, int] = {"positive": 1, "negative": 2, "neutral": 3}
_ASPECT_DISPLAY = {
    asp: asp.replace("#", " - ").replace("_", " ") for asp in ASPECT_COLUMNS
}


class SPCCorrector:
    """
    Corrects SPC predictions for uncertain PRESENT aspects using LLM.

    Invariant: never changes ACD (absent/present status).
    Only corrects sentiment (1/2/3) of aspects already predicted as present.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        retriever: ABSARetriever,
        llm_client: LLMClient,
        entropy_threshold: float = 0.8,
        k: int = 4,
        max_per_review: int = 5,
        sleep_sec: float = 0.3,
    ):
        self.train_df = train_df
        self.retriever = retriever
        self.llm_client = llm_client
        self.entropy_threshold = entropy_threshold
        self.k = k
        self.max_per_review = max_per_review
        self.sleep_sec = sleep_sec

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_spc_entropy(probs_single: np.ndarray) -> np.ndarray:
        """
        Compute SPC entropy per aspect.

        Args:
            probs_single: [34, 4] softmax probabilities (0=absent,1=pos,2=neg,3=neu)
        Returns:
            entropy: [34] — Shannon entropy over the SPC distribution (probs[1:4])
        """
        eps = 1e-9
        spc_probs = probs_single[:, 1:]                       # [34, 3] pos/neg/neu
        spc_sum = spc_probs.sum(axis=1, keepdims=True).clip(min=eps)
        spc_norm = spc_probs / spc_sum                        # normalize to sum=1
        entropy = -(spc_norm * np.log(spc_norm + eps)).sum(axis=1)  # [34]
        return entropy

    # ------------------------------------------------------------------
    # RAG retrieval (aspect-specific)
    # ------------------------------------------------------------------

    def _retrieve_aspect_examples(
        self, processed_text: str, aspect_idx: int
    ) -> list[dict]:
        """
        Retrieve examples where the specific aspect IS present.
        Filters larger retrieval pool to aspect-positive examples.
        """
        aspect_name = ASPECT_COLUMNS[aspect_idx]
        candidates = self.retriever.retrieve(
            processed_text, k=self.k * 6, aspect_aware=False
        )
        filtered = [
            idx for idx in candidates
            if int(self.train_df.iloc[idx][aspect_name]) != 0
        ]
        pool = filtered if filtered else candidates
        return self._format_examples(aspect_name, pool[: self.k])

    def _format_examples(self, aspect_name: str, indices: list[int]) -> list[dict]:
        """Format examples as {review, sentiment} pairs for the specific aspect."""
        examples = []
        for idx in indices:
            row = self.train_df.iloc[idx]
            label = int(row[aspect_name])
            if label == 0:
                continue
            review = str(row.get("Review", row.get("processed_review", "")))
            examples.append({"review": review, "sentiment": IDX_TO_LABEL[label]})
        return examples

    # ------------------------------------------------------------------
    # Prompt & parse
    # ------------------------------------------------------------------

    def _build_spc_prompt(
        self,
        review_text: str,
        aspect_idx: int,
        examples: list[dict],
    ) -> list[dict]:
        """
        Build simple SPC-only prompt.
        Output: một từ duy nhất — positive / negative / neutral.
        """
        display_name = _ASPECT_DISPLAY[ASPECT_COLUMNS[aspect_idx]]

        system_msg = (
            "Bạn là chuyên gia phân tích cảm xúc review khách sạn tiếng Việt. "
            "Nhiệm vụ: xác định sentiment của một aspect CỤ THỂ đã được xác nhận "
            "là có đề cập trong review. "
            "Chỉ trả lời đúng một trong ba từ: positive / negative / neutral"
        )

        messages: list[dict] = [{"role": "system", "content": system_msg}]

        for ex in examples:
            messages.append({
                "role": "user",
                "content": (
                    f'Review: "{ex["review"]}"\n'
                    f'Aspect [{display_name}] đã được xác nhận là có trong review. '
                    f"Sentiment?"
                ),
            })
            messages.append({"role": "assistant", "content": ex["sentiment"]})

        messages.append({
            "role": "user",
            "content": (
                f'Review: "{review_text}"\n'
                f'Aspect [{display_name}] đã được xác nhận là có trong review. '
                f"Sentiment?"
            ),
        })
        return messages

    def _parse_spc_output(self, text: str) -> Optional[int]:
        """
        Parse LLM output → sentiment int (1/2/3) or None if unparseable.
        Handles both plain text ("positive") and JSON ({"sentiment": "positive"})
        since Gemini uses JSON mode.
        """
        if not text:
            return None
        text_lower = text.strip().lower()

        # Try JSON (Gemini forces JSON output)
        try:
            data = json.loads(text_lower)
            if isinstance(data, dict):
                for v in data.values():
                    s = str(v).strip().lower()
                    if s in _SENTIMENT_MAP:
                        return _SENTIMENT_MAP[s]
            elif isinstance(data, str) and data in _SENTIMENT_MAP:
                return _SENTIMENT_MAP[data]
        except Exception:
            pass

        # Direct match (order matters: check all 3, pick first found)
        for sentiment in ("positive", "negative", "neutral"):
            if sentiment in text_lower:
                return _SENTIMENT_MAP[sentiment]

        return None

    # ------------------------------------------------------------------
    # Correction logic
    # ------------------------------------------------------------------

    def correct_single(
        self,
        review_text: str,
        processed_text: str,
        base_preds: np.ndarray,    # [34] — output from ADD-only or PhoBERT argmax
        probs_single: np.ndarray,  # [34, 4]
    ) -> tuple[np.ndarray, dict]:
        """
        Correct SPC for uncertain present aspects of 1 review.

        Returns:
            corrected_preds: [34]
            debug_info: dict
        """
        final_preds = base_preds.copy()
        entropy = self.compute_spc_entropy(probs_single)

        # Candidates: present aspects with high SPC entropy
        candidates = [
            i for i in range(len(ASPECT_COLUMNS))
            if int(base_preds[i]) != 0 and entropy[i] > self.entropy_threshold
        ]
        # Top-k by entropy (highest = most uncertain)
        candidates = sorted(candidates, key=lambda i: entropy[i], reverse=True)
        candidates = candidates[: self.max_per_review]

        if not candidates:
            return final_preds, {"corrected": [], "parse_fail": 0, "n_candidates": 0}

        corrections = []
        parse_fails = 0

        for idx in candidates:
            aspect_name = ASPECT_COLUMNS[idx]
            try:
                examples = self._retrieve_aspect_examples(processed_text, idx)
                messages = self._build_spc_prompt(review_text, idx, examples)
                raw = self.llm_client.complete(messages, temperature=0.0, use_cache=True)
                parsed = self._parse_spc_output(raw)
            except Exception as exc:
                parse_fails += 1
                corrections.append({
                    "aspect": aspect_name,
                    "status": "error",
                    "error": str(exc)[:120],
                })
                continue

            if parsed is None:
                parse_fails += 1
                corrections.append({
                    "aspect": aspect_name,
                    "status": "parse_fail",
                    "raw_preview": (raw or "")[:80],
                })
                continue

            old_label = int(final_preds[idx])
            final_preds[idx] = parsed  # only 1/2/3 — ACD unchanged
            corrections.append({
                "aspect": aspect_name,
                "old": old_label,
                "new": parsed,
                "changed": parsed != old_label,
                "entropy": round(float(entropy[idx]), 4),
            })

            if self.sleep_sec > 0:
                time.sleep(self.sleep_sec)

        return final_preds, {
            "corrected": corrections,
            "parse_fail": parse_fails,
            "n_candidates": len(candidates),
        }

    def correct_dataset(
        self,
        texts: list[str],
        processed_texts: list[str],
        base_preds_all: np.ndarray,  # [N, 34]
        probs_all: np.ndarray,       # [N, 34, 4]
        max_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Run SPC correction on full dataset.

        Returns:
            corrected_preds_all: [N, 34]
            stats: aggregated statistics
        """
        n = min(len(texts), len(base_preds_all))
        if max_samples is not None:
            n = min(n, max_samples)

        corrected_all = base_preds_all[:n].copy()
        total_llm_calls = 0
        total_parse_fails = 0
        total_changed = 0
        routed_reviews = 0
        sample_records = []

        for i in range(n):
            preds_i, debug = self.correct_single(
                review_text=texts[i],
                processed_text=processed_texts[i],
                base_preds=base_preds_all[i],
                probs_single=probs_all[i],
            )
            corrected_all[i] = preds_i

            n_cand = debug.get("n_candidates", 0)
            total_llm_calls += n_cand
            total_parse_fails += debug.get("parse_fail", 0)
            changed = sum(1 for c in debug.get("corrected", []) if c.get("changed"))
            total_changed += changed
            if n_cand > 0:
                routed_reviews += 1

            sample_records.append({"idx": i, **debug})

            if (i + 1) % 50 == 0:
                print(
                    f"  [{i+1}/{n}] LLM calls: {total_llm_calls} "
                    f"| Changed: {total_changed} "
                    f"| Parse fails: {total_parse_fails} "
                    f"| Routing: {routed_reviews/(i+1):.0%}"
                )

        stats = {
            "total_samples": n,
            "routed_reviews": routed_reviews,
            "routing_rate": routed_reviews / n if n > 0 else 0.0,
            "total_llm_calls": total_llm_calls,
            "total_changed": total_changed,
            "total_parse_fails": total_parse_fails,
            "parse_fail_rate": total_parse_fails / max(1, total_llm_calls),
            "change_rate": total_changed / max(1, total_llm_calls),
            "entropy_threshold": self.entropy_threshold,
            "k": self.k,
            "max_per_review": self.max_per_review,
            "sample_records": sample_records,
        }
        return corrected_all, stats
