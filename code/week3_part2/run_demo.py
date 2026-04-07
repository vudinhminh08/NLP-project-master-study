"""
run_demo.py — Demo end-to-end: nhập review → PhoBERT predict → LLM explain.

Chạy interactive trên terminal hoặc dùng trong notebook.

Usage:
    python code/week3_part2/run_demo.py --provider openai \
        --checkpoint_path outputs/results/week4_augmented/models/best_model.pt
    python code/week3_part2/run_demo.py --provider openai \
        --checkpoint_path outputs/results/week4_augmented/models/best_model.pt \
        --interactive
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from explainer import explain_predictions  # noqa: E402
from retrain import _default_preprocess  # noqa: E402


def demo_single_review(
    review: str,
    model,
    tokenizer,
    llm_client: LLMClient,
    preprocess_fn=None,
    device=None,
) -> dict:
    """
    End-to-end: review text → PhoBERT predict → LLM explain.

    Returns:
        dict với keys: review, predictions, explanations
    """
    import torch

    device = device or torch.device("cpu")

    # 1. Preprocess
    processed = preprocess_fn(review) if preprocess_fn else review

    # 2. Tokenize
    inputs = tokenizer(
        processed,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. PhoBERT predict
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    preds = outputs["preds"][0].cpu().numpy()  # [34]

    # 4. Convert to dict (only present aspects)
    pred_dict = {}
    for j, asp in enumerate(ASPECT_COLUMNS):
        label = int(preds[j])
        if label > 0:
            pred_dict[asp] = IDX_TO_LABEL[label]

    # 5. LLM explain
    explanations = explain_predictions(review, pred_dict, llm_client)

    return {
        "review": review,
        "predictions": pred_dict,
        "explanations": explanations,
    }


def format_output(result: dict) -> str:
    """Format kết quả cho terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"REVIEW: {result['review']}")
    lines.append(f"{'=' * 70}")

    if not result["predictions"]:
        lines.append("  -> Khong phat hien aspect nao.")
        return "\n".join(lines)

    lines.append(f"\nPHAT HIEN {len(result['predictions'])} ASPECTS:\n")

    sentiment_tag = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}

    for expl in result.get("explanations", []):
        aspect = expl.get("aspect", "?")
        sentiment = expl.get("sentiment", "?")
        tag = sentiment_tag.get(sentiment, "[?]")

        lines.append(f"  {tag} {aspect}: {sentiment}")
        if expl.get("evidence"):
            lines.append(f"     Evidence : \"{expl['evidence']}\"")
        if expl.get("explanation"):
            lines.append(f"     Giai thich: {expl['explanation']}")
        if expl.get("action"):
            lines.append(f"     De xuat   : {expl['action']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: Review -> Predict -> Explain")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to PhoBERT .pt checkpoint")
    parser.add_argument("--interactive", action="store_true",
                        help="Che do interactive (nhap review tu terminal)")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from model import ABSAPhoBERT

    api_key = args.api_key or os.environ.get(
        "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    )
    if not api_key:
        print(f"Missing API key for {args.provider}.")
        return

    print("Loading PhoBERT model...")
    model = ABSAPhoBERT(model_name="vinai/phobert-base-v2", encoder_option="cls_only")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    llm_client = LLMClient(provider=args.provider, api_key=api_key)

    print("Ready!\n")

    if args.interactive:
        print("Nhap review (hoac 'quit' de thoat):\n")
        while True:
            review = input(">>> ").strip()
            if review.lower() in ("quit", "exit", "q"):
                break
            if not review:
                continue
            result = demo_single_review(
                review=review,
                model=model,
                tokenizer=tokenizer,
                llm_client=llm_client,
                preprocess_fn=_default_preprocess,
            )
            print(format_output(result))
    else:
        demo_reviews = [
            "Phong rong rai sach se, view dep nhung nhan vien le tan thai do hoi kem.",
            "Khach san cu, noi that xuong cap. Bua sang tam duoc, ko co gi dac biet.",
            "Vi tri rat thuan tien, gan trung tam. Gia hop ly so voi chat luong.",
            "Wifi yeu, minibar gia cat co. Nhung giuong nem rat em, ngu ngon.",
        ]

        for review in demo_reviews:
            result = demo_single_review(
                review=review,
                model=model,
                tokenizer=tokenizer,
                llm_client=llm_client,
                preprocess_fn=_default_preprocess,
            )
            print(format_output(result))
            time.sleep(1.5)


if __name__ == "__main__":
    main()
