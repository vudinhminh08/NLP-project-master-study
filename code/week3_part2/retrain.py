"""
retrain.py — Re-train PhoBERT trên augmented data.

Chiến lược:
1. Load train_preprocessed.csv (3000 gốc)
2. Load augmented_reviews_filtered.json
3. Preprocess augmented reviews (underthesea / VnCoreNLP)
4. Merge thành train_augmented.csv
5. Re-train PhoBERT v2.4 config (best config từ week 2)
6. Evaluate trên dev + test (KHÔNG đổi dev/test)

QUAN TRỌNG:
- Dùng CÙNG eval script (step4_eval.py) để so sánh fair
- Dùng CÙNG config v2.4 (best single model)
- CHỈ thay đổi training data
- Re-training cần GPU — chạy trên Kaggle/Colab
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))

from utils.constants import ASPECT_COLUMNS, LABEL_TO_IDX  # noqa: E402
from utils.helpers import set_seed  # noqa: E402


# ─── Preprocessing helper ────────────────────────────────────────────────────

def _default_preprocess(text: str) -> str:
    """
    Preprocessing mặc định — cùng pipeline với week 1 step3.
    Import từ code/week1/step3_preprocessing.py nếu có.
    Fallback: underthesea word_tokenize.
    """
    try:
        from step3_preprocessing import preprocess_text
        return preprocess_text(text)
    except (ImportError, Exception):
        pass

    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        pass

    # Ultimate fallback — trả về text gốc
    return text


# ─── Dataset preparation ─────────────────────────────────────────────────────

def prepare_augmented_dataset(
    original_train_path: str = "data/train_preprocessed.csv",
    augmented_path: str = "data/augmented_reviews_filtered.json",
    output_path: str = "data/train_augmented.csv",
    preprocess_fn=None,
) -> pd.DataFrame:
    """
    Merge original train + augmented reviews.

    Args:
        preprocess_fn: callable(text) -> processed_text
            Dùng VnCoreNLP/underthesea để segment augmented reviews.
            Nếu None, dùng _default_preprocess.

    Returns:
        DataFrame với cùng format như train_preprocessed.csv
    """
    orig_df = pd.read_csv(original_train_path)
    print(f"Original train: {len(orig_df)} samples")

    with open(augmented_path, "r", encoding="utf-8") as f:
        augmented = json.load(f)
    print(f"Augmented reviews: {len(augmented)} samples")

    if len(augmented) == 0:
        print("No augmented data, returning original")
        return orig_df

    if preprocess_fn is None:
        preprocess_fn = _default_preprocess

    rows = []
    for item in augmented:
        row = {"Review": item["review"]}

        try:
            row["processed_review"] = preprocess_fn(item["review"])
        except Exception as exc:
            print(f"  Preprocess failed: {item['review'][:50]}... → {exc}")
            row["processed_review"] = item["review"]

        for asp in ASPECT_COLUMNS:
            sentiment = item["labels"].get(asp, "absent")
            row[asp] = LABEL_TO_IDX.get(sentiment, 0)

        rows.append(row)

    aug_df = pd.DataFrame(rows)

    # Đảm bảo cùng cột với orig_df
    for col in orig_df.columns:
        if col not in aug_df.columns:
            aug_df[col] = orig_df[col].iloc[0] if len(orig_df) > 0 else ""
    aug_df = aug_df[orig_df.columns]

    print(f"Augmented after preprocessing: {len(aug_df)} samples")

    merged = pd.concat([orig_df, aug_df], ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Merged dataset: {len(merged)} samples → {output_path}")
    print(f"  Original: {len(orig_df)}, Augmented: {len(aug_df)}, "
          f"Ratio: {len(aug_df) / len(orig_df) * 100:.1f}%")

    return merged


# ─── Re-training (cần GPU) ───────────────────────────────────────────────────

def retrain_phobert(
    train_path: str = "data/train_augmented.csv",
    save_dir: str = "outputs/results/week4_augmented",
):
    """
    Re-train PhoBERT với augmented data.
    Dùng CÙNG config v2.4 (best single model từ week 2).

    ⚠️ CẦN GPU — chạy trên Kaggle/Colab.
    """
    import torch
    from transformers import AutoTokenizer
    from model import ABSAPhoBERT
    from train import train as train_fn, load_class_weights

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
    from step2_dataloader import create_dataloaders

    set_seed(42)

    # Config v2.4 — CÙNG config đã cho best result
    config = {
        "learning_rate":           3e-5,
        "warmup_ratio":            0.15,
        "batch_size":              16,
        "grad_accumulation_steps": 1,
        "max_epochs":              35,
        "early_stop_patience":     7,
        "dropout":                 0.2,
        "optimizer":               "AdamW",
        "scheduler":               "cosine_warmup",
        "seed":                    42,
        "max_seq_len":             256,
        "weight_clip":             10.0,
        "encoder_option":          "cls_only",
        "max_grad_norm":           1.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    train_loader, dev_loader, test_loader = create_dataloaders(
        train_path=train_path,
        dev_path="data/dev_preprocessed.csv",
        test_path="data/test_preprocessed.csv",
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_len=config["max_seq_len"],
        num_workers=2,
        use_preprocessed=True,
    )

    model = ABSAPhoBERT(
        model_name="vinai/phobert-base-v2",
        encoder_option=config["encoder_option"],
        dropout=config["dropout"],
    )

    # Dùng class weights từ EDA (augmented ratio nhỏ ~8%, không cần recompute)
    class_weights = load_class_weights(
        weights_path="outputs/eda/class_weights.json",
        weight_clip=config["weight_clip"],
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)
    train_fn(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        class_weights=class_weights,
        device=device,
        config=config,
        save_dir=os.path.join(save_dir, "models"),
        results_dir=save_dir,
    )

    print(f"\nRe-training complete → {save_dir}")
