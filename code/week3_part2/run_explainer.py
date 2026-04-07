"""
run_explainer.py — Entry point Phase 2: Explainability.

Chạy trên Mac local (chỉ cần API key + PhoBERT checkpoint).

Usage:
    python code/week3_part2/run_explainer.py --provider openai --n_samples 20
    python code/week3_part2/run_explainer.py --provider openai \
        --predictions_path outputs/results/week4_augmented/predictions.json --n_samples 20
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL  # noqa: E402
from utils.helpers import set_seed  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from explainer import explain_predictions  # noqa: E402


def load_phobert_predictions(
    predictions_path: str = None,
    test_path: str = "data/test_preprocessed.csv",
    checkpoint_path: str = None,
) -> tuple:
    """
    Load PhoBERT predictions.

    Option 1: Load từ saved predictions file (nhanh, không cần GPU)
    Option 2: Run inference từ checkpoint (cần model + tokenizer)

    Returns:
        (reviews_original, predictions_list)
        reviews_original: list of review texts GỐC
        predictions_list: list of dicts {aspect: sentiment_str}
    """
    test_df = pd.read_csv(test_path)

    if predictions_path and os.path.exists(predictions_path):
        with open(predictions_path, "r") as f:
            saved = json.load(f)

        reviews = test_df["Review"].tolist()
        predictions_list = []

        if isinstance(saved, list) and len(saved) > 0:
            if isinstance(saved[0], list):
                # Array format [sample_idx][aspect_idx] = label_int
                for sample_preds in saved:
                    pred_dict = {}
                    for j, asp in enumerate(ASPECT_COLUMNS):
                        label = int(sample_preds[j])
                        if label > 0:
                            pred_dict[asp] = IDX_TO_LABEL[label]
                    predictions_list.append(pred_dict)
            elif isinstance(saved[0], dict):
                predictions_list = saved

        return reviews, predictions_list

    elif checkpoint_path and os.path.exists(checkpoint_path):
        import torch
        from model import ABSAPhoBERT
        from transformers import AutoTokenizer

        print(f"Loading checkpoint: {checkpoint_path}")
        model = ABSAPhoBERT(
            model_name="vinai/phobert-base-v2",
            encoder_option="cls_only",
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        reviews = test_df["Review"].tolist()
        processed = test_df["processed_review"].tolist()
        predictions_list = []

        with torch.no_grad():
            for text in processed:
                inputs = tokenizer(
                    text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                preds = outputs["preds"][0].cpu().numpy()
                pred_dict = {}
                for j, asp in enumerate(ASPECT_COLUMNS):
                    label = int(preds[j])
                    if label > 0:
                        pred_dict[asp] = IDX_TO_LABEL[label]
                predictions_list.append(pred_dict)

        return reviews, predictions_list

    else:
        raise FileNotFoundError(
            "Cần --predictions_path hoặc --checkpoint_path. "
            "Chạy PhoBERT inference trước hoặc dùng saved predictions."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Explainability for PhoBERT")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--predictions_path", type=str, default=None,
                        help="Path to saved PhoBERT predictions JSON")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to PhoBERT checkpoint .pt")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Số reviews để explain (demo)")
    parser.add_argument("--output_path", type=str,
                        default="outputs/results/week4_explainer_samples.json")
    args = parser.parse_args()

    set_seed(42)

    api_key = args.api_key or os.environ.get(
        "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    )
    if not api_key:
        print(f"Missing API key for {args.provider}.")
        return

    client = LLMClient(provider=args.provider, api_key=api_key)

    reviews, predictions_list = load_phobert_predictions(
        predictions_path=args.predictions_path,
        checkpoint_path=args.checkpoint_path,
    )
    print(f"Loaded {len(reviews)} reviews with predictions")

    # Chọn subset đa dạng (ưu tiên reviews có nhiều aspects)
    n_aspects_per_review = [len(p) for p in predictions_list]
    sorted_indices = sorted(
        range(len(reviews)),
        key=lambda i: n_aspects_per_review[i],
        reverse=True,
    )
    selected = sorted_indices[:args.n_samples]

    print(f"\nExplaining {len(selected)} reviews...")
    results = []

    for idx in selected:
        review = reviews[idx]
        preds = predictions_list[idx]

        if not preds:
            continue

        print(f"\n--- Review {idx} ({len(preds)} aspects) ---")
        print(f"  Text: {review[:100]}...")

        explanations = explain_predictions(review, preds, client)
        results.append({
            "review_idx": idx,
            "review": review,
            "predictions": preds,
            "explanations": explanations,
        })

        time.sleep(1.0)

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} explained reviews → {args.output_path}")

    if results:
        sample = results[0]
        print(f"\n{'=' * 60}")
        print("SAMPLE OUTPUT:")
        print(f"{'=' * 60}")
        print(f"Review: {sample['review'][:200]}")
        print(f"\nPredictions: {sample['predictions']}")
        print("\nExplanations:")
        for expl in sample.get("explanations", []):
            print(f"  {expl.get('aspect', '?')}: {expl.get('explanation', '?')}")


if __name__ == "__main__":
    main()
