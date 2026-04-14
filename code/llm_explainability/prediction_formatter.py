"""
prediction_formatter.py — Convert PhoBERT outputs into explanation inputs.

The LLM explanation layer receives only present aspects. It must not act as a
classifier or add labels outside the PhoBERT prediction.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_processing"))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL  # noqa: E402


def predictions_to_present_items(
    pred_row: list[int] | np.ndarray,
    probs_row: Optional[np.ndarray] = None,
    aspect_columns: list[str] = ASPECT_COLUMNS,
) -> list[dict]:
    """
    Convert one `[34]` prediction row to present-aspect records.

    Args:
        pred_row:  34 labels, 0=absent, 1=positive, 2=negative, 3=neutral.
        probs_row: optional [34, 4] probabilities for confidence display.

    Returns:
        list of {"aspect", "sentiment", "label", "confidence"} records.
    """
    items: list[dict] = []
    for idx, aspect in enumerate(aspect_columns):
        label = int(pred_row[idx])
        if label == 0:
            continue

        item = {
            "aspect": aspect,
            "sentiment": IDX_TO_LABEL.get(label, "unknown"),
            "label": label,
        }
        if probs_row is not None:
            item["confidence"] = round(float(probs_row[idx, label]), 4)
        items.append(item)
    return items


def prediction_matrix_to_records(
    reviews: list[str],
    pred_matrix: np.ndarray,
    probs: Optional[np.ndarray] = None,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Build explanation-ready records for a batch of reviews."""
    n = min(len(reviews), len(pred_matrix))
    if max_samples is not None:
        n = min(n, max_samples)

    records = []
    for i in range(n):
        probs_i = probs[i] if probs is not None else None
        records.append(
            {
                "review_idx": i,
                "review": str(reviews[i]),
                "predictions": predictions_to_present_items(pred_matrix[i], probs_i),
            }
        )
    return records
