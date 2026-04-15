"""
run_explainability.py — Generate LLM explanations for PhoBERT predictions.

This script expects saved PhoBERT predictions. It intentionally does not run a
new LLM classifier; the LLM explains PhoBERT outputs only.

Example:
    python code/llm_explainability/run_explainability.py \
        --predictions_json outputs/results/final_predictions.json \
        --provider openai --max_samples 20
    python code/llm_explainability/run_explainability.py \
        --predictions_json outputs/results/final_predictions.json \
        --provider ollama --max_samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "..", "data_processing"))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "..", "llm_rag"))
sys.path.insert(0, THIS_DIR)

from llm_client import LLMClient  # noqa: E402
from prediction_formatter import prediction_matrix_to_records  # noqa: E402
from llm_explainer import explain_batch  # noqa: E402
from utils.helpers import save_json  # noqa: E402


def load_prediction_matrix(path: str) -> np.ndarray:
    """Load predictions from JSON list or NPY array."""
    if path.endswith(".npy"):
        return np.load(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain PhoBERT ABSA predictions")
    parser.add_argument("--test_csv", default="data/test_preprocessed.csv")
    parser.add_argument("--predictions_json", required=True)
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "ollama"])
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/results/llm_explainability")
    args = parser.parse_args()

    if args.provider == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api_key.")
    elif args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Set GEMINI_API_KEY or pass --api_key.")
    else:
        # Ollama local server does not require API key.
        api_key = args.api_key

    test_df = pd.read_csv(args.test_csv)
    pred_matrix = load_prediction_matrix(args.predictions_json)
    records = prediction_matrix_to_records(
        reviews=test_df["Review"].astype(str).tolist(),
        pred_matrix=pred_matrix,
        max_samples=args.max_samples,
    )

    client = LLMClient(provider=args.provider, api_key=api_key)
    explanations, stats = explain_batch(records, client, max_samples=args.max_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    save_json({"samples": explanations}, os.path.join(args.output_dir, "explanation_samples.json"))
    save_json(stats, os.path.join(args.output_dir, "explanation_quality_report.json"))

    print(f"Saved explanations to {args.output_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
