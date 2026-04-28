from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSING_DIR = PROJECT_ROOT / "code" / "data_processing"
PHOBERT_DIR = PROJECT_ROOT / "code" / "phobert"
LLM_EXPLAINABILITY_DIR = PROJECT_ROOT / "code" / "llm_explainability"

for path in (DATA_PROCESSING_DIR, PHOBERT_DIR, LLM_EXPLAINABILITY_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model import ABSAPhoBERT  # noqa: E402
from prediction_formatter import predictions_to_present_items  # noqa: E402
from step3_preprocessing import VnCoreNLPSegmenter, preprocess_text  # noqa: E402
from utils.constants import (  # noqa: E402
    ASPECT_COLUMNS,
    IDX_TO_LABEL,
    MAX_SEQ_LEN,
    NUM_ASPECTS,
    NUM_LABELS,
    PHOBERT_MODEL_NAME,
)


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "results" / "phobert_results"
    / "models_cls_only" / "best_model.pt"
)


@dataclass
class PhoBERTPrediction:
    review: str
    processed_review: str
    predictions: list[dict]
    raw_labels: list[int]
    probabilities: list[list[float]]


class PhoBERTService:
    def __init__(
        self,
        checkpoint_path: str | os.PathLike = DEFAULT_CHECKPOINT_PATH,
        vncorenlp_dir: str | os.PathLike | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.vncorenlp_dir = Path(vncorenlp_dir).expanduser().resolve() if vncorenlp_dir else PROJECT_ROOT / "vncorenlp"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint PhoBERT: {self.checkpoint_path}"
            )

        self.segmenter = VnCoreNLPSegmenter(
            vncorenlp_dir=str(self.vncorenlp_dir),
            use_fallback=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
        self.model = self._load_model()

    def _load_model(self) -> ABSAPhoBERT:
        model = ABSAPhoBERT(
            model_name=PHOBERT_MODEL_NAME,
            num_aspects=NUM_ASPECTS,
            num_labels=NUM_LABELS,
            encoder_option="cls_only",
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, review: str) -> PhoBERTPrediction:
        review = str(review).strip()
        if not review:
            raise ValueError("Review không được rỗng.")

        processed_review = preprocess_text(
            review,
            segmenter=self.segmenter,
            do_segment=True,
        )
        encoded = self.tokenizer(
            processed_review,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred_row = output["preds"].detach().cpu().numpy()[0]
            probs_row = np.stack(
                [
                    torch.softmax(logit, dim=-1).detach().cpu().numpy()[0]
                    for logit in output["logits"]
                ],
                axis=0,
            )

        predictions = predictions_to_present_items(
            pred_row=pred_row,
            probs_row=probs_row,
            aspect_columns=ASPECT_COLUMNS,
        )
        return PhoBERTPrediction(
            review=review,
            processed_review=processed_review,
            predictions=predictions,
            raw_labels=[int(x) for x in pred_row.tolist()],
            probabilities=probs_row.round(6).tolist(),
        )


def labels_to_full_table(raw_labels: list[int], probabilities: list[list[float]]) -> list[dict]:
    rows = []
    for idx, aspect in enumerate(ASPECT_COLUMNS):
        label = int(raw_labels[idx])
        rows.append(
            {
                "aspect": aspect,
                "sentiment": IDX_TO_LABEL.get(label, "unknown"),
                "confidence": round(float(probabilities[idx][label]), 4),
            }
        )
    return rows
