
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_processing"))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL


def predictions_to_present_items(
    pred_row: list[int] | np.ndarray,
    probs_row: Optional[np.ndarray] = None,
    aspect_columns: list[str] = ASPECT_COLUMNS,
) -> list[dict]:
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
