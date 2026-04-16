
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_processing"))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL  # noqa: E402


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reviews: list[str],
    max_examples: int = 20,
) -> dict:
    acd_fp = {aspect: 0 for aspect in ASPECT_COLUMNS}
    acd_fn = {aspect: 0 for aspect in ASPECT_COLUMNS}
    spc_err = {aspect: 0 for aspect in ASPECT_COLUMNS}
    examples = []

    n = min(len(y_true), len(y_pred), len(reviews))
    for i in range(n):
        for j, aspect in enumerate(ASPECT_COLUMNS):
            true_label = int(y_true[i, j])
            pred_label = int(y_pred[i, j])
            true_present = true_label != 0
            pred_present = pred_label != 0

            if pred_present and not true_present:
                acd_fp[aspect] += 1
                kind = "acd_false_positive"
            elif true_present and not pred_present:
                acd_fn[aspect] += 1
                kind = "acd_false_negative"
            elif true_present and pred_present and true_label != pred_label:
                spc_err[aspect] += 1
                kind = "spc_error"
            else:
                continue

            if len(examples) < max_examples:
                examples.append(
                    {
                        "review_idx": i,
                        "review": str(reviews[i]),
                        "aspect": aspect,
                        "error_type": kind,
                        "true": IDX_TO_LABEL.get(true_label, "unknown"),
                        "pred": IDX_TO_LABEL.get(pred_label, "unknown"),
                    }
                )

    return {
        "top_acd_false_positive_aspects": _top_counts(acd_fp),
        "top_acd_false_negative_aspects": _top_counts(acd_fn),
        "top_spc_error_aspects": _top_counts(spc_err),
        "representative_errors": examples,
    }


def _top_counts(counts: dict[str, int], k: int = 10) -> list[dict]:
    return [
        {"aspect": aspect, "count": count}
        for aspect, count in sorted(counts.items(), key=lambda x: -x[1])[:k]
        if count > 0
    ]
