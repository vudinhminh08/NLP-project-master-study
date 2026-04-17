
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "data_processing")
sys.path.insert(0, WEEK1_DIR)

from utils.constants import ASPECT_COLUMNS, NUM_ASPECTS, ZERO_TRAIN_ASPECTS, LABEL_TO_IDX
from utils.helpers import set_seed
from step4_eval import evaluate_predictions


VI_CLEAN_PATTERN = re.compile(
    r"[^\w\s"
    r"áàảãạăắằẳẵặâấầẩẫậ"
    r"éèẻẽẹêếềểễệ"
    r"íìỉĩị"
    r"óòỏõọôốồổỗộơớờởỡợ"
    r"úùủũụưứừửữự"
    r"ýỳỷỹỵđ]"
)


def _normalize_review_column(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "Review" in df.columns:
        review_col = "Review"
    elif "review" in df.columns:
        review_col = "review"
        print(f"[{split_name}] Found lowercase 'review' column, normalize to 'Review'.")
    else:
        raise ValueError(f"[{split_name}] Missing review column: expected 'Review' or 'review'.")

    if review_col != "Review":
        df = df.copy()
        df["Review"] = df[review_col]

    df["Review"] = df["Review"].fillna("").astype(str)
    return df


def _validate_aspect_columns(df: pd.DataFrame, split_name: str) -> None:
    missing = [col for col in ASPECT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"[{split_name}] Missing {len(missing)} aspect columns: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train.csv")
    dev_path = os.path.join(data_dir, "dev.csv")
    test_path = os.path.join(data_dir, "test.csv")

    for path in (train_path, dev_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    train_df = _normalize_review_column(train_df, "train")
    dev_df = _normalize_review_column(dev_df, "dev")
    test_df = _normalize_review_column(test_df, "test")

    _validate_aspect_columns(train_df, "train")
    _validate_aspect_columns(dev_df, "dev")
    _validate_aspect_columns(test_df, "test")

    print("[Data] train shape:", train_df.shape)
    print("[Data] dev shape:  ", dev_df.shape)
    print("[Data] test shape: ", test_df.shape)

    return train_df, dev_df, test_df


def preprocess_for_tfidf(text: str) -> str:
    text = str(text).lower()
    text = VI_CLEAN_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_svm_per_aspect(
    X_train: Any,
    y_train: np.ndarray,
    aspect_columns: list[str] = ASPECT_COLUMNS,
) -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = []

    for i, aspect in enumerate(aspect_columns):
        y = y_train[:, i]
        unique_classes = np.unique(y)

        if len(unique_classes) <= 1:
            only_class = int(unique_classes[0])
            print(f"  {aspect}: only class={only_class} in train; using majority prediction.")
            models.append(("majority", only_class))
            continue

        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        )
        clf.fit(X_train, y)
        models.append(("svm", clf))

    return models


def predict_all_aspects(
    models: list[tuple[str, Any]],
    X: Any,
    n_samples: int,
) -> np.ndarray:
    y_pred = np.zeros((n_samples, NUM_ASPECTS), dtype=int)
    for i, (model_type, model) in enumerate(models):
        if model_type == "majority":
            y_pred[:, i] = int(model)
        else:
            y_pred[:, i] = model.predict(X)
    return y_pred


def generate_summary(
    dev_metrics: dict,
    test_metrics: dict,
    save_path: str = "outputs/results/svm_baseline_summary.md",
) -> None:
    content = f"""# Ket qua SVM + TF-IDF Baseline

## Config
| Tham so | Gia tri |
|---------|---------|
| Model | LinearSVC (sklearn) |
| Features | TF-IDF, unigram + bigram |
| max_features | 50000 |
| C | 1.0 |
| class_weight | balanced |
| Preprocessing | lowercase + remove special chars |
| Word segmentation | Khong (co tinh giu don gian) |

## Ket qua

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev | {dev_metrics["macro_acd_f1"]:.4f} | {dev_metrics["macro_spc_f1"]:.4f} | {dev_metrics["macro_combined_f1"]:.4f} |
| **Test** | **{test_metrics["macro_acd_f1"]:.4f}** | **{test_metrics["macro_spc_f1"]:.4f}** | **{test_metrics["macro_combined_f1"]:.4f}** |

## So sanh

| Phuong phap | ACD F1 | Combined F1 | Ghi chu |
|-------------|--------|-------------|---------|
| SVM ours | {test_metrics["macro_acd_f1"]:.4f} | {test_metrics["macro_combined_f1"]:.4f} | File nay |
| SA1 VLSP 2018 (Dang et al.) | 0.69 | 0.61 | Tu paper |
| PhoBERT ours | 0.4067 | 0.3309 | Tuan 2 |
| SOTA (Huynh 2022) | 0.8255 | 0.7732 | Upper bound |

## Ghi chu
- SVM khong dung word segmentation -> feature extraction kem hon co the
- class_weight='balanced' la cach xu ly imbalance tuong duong weighted loss
- 34 classifiers doc lap -> khong chia se representation giua aspects (khac PhoBERT multi-task)
"""

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Saved] {save_path}")


def main() -> None:

    _ = CalibratedClassifierCV

    set_seed(42)


    train_df, dev_df, test_df = load_data()


    train_texts = train_df["Review"].apply(preprocess_for_tfidf).tolist()
    dev_texts = dev_df["Review"].apply(preprocess_for_tfidf).tolist()
    test_texts = test_df["Review"].apply(preprocess_for_tfidf).tolist()


    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"[TF-IDF] Vocab size: {len(vectorizer.vocabulary_)}")
    print(f"[TF-IDF] X_train: {X_train.shape}, X_dev: {X_dev.shape}, X_test: {X_test.shape}")


    y_train = train_df[ASPECT_COLUMNS].values.astype(int)
    y_dev = dev_df[ASPECT_COLUMNS].values.astype(int)
    y_test = test_df[ASPECT_COLUMNS].values.astype(int)

    valid_label_ids = set(LABEL_TO_IDX.values())
    for split_name, labels in (("train", y_train), ("dev", y_dev), ("test", y_test)):
        split_values = set(np.unique(labels).tolist())
        if not split_values.issubset(valid_label_ids):
            raise ValueError(f"[{split_name}] Found invalid labels: {sorted(split_values - valid_label_ids)}")


    print("\n" + "-" * 60)
    print("Training 34 SVM classifiers...")
    models = train_svm_per_aspect(X_train, y_train)
    n_svm = sum(1 for m in models if m[0] == "svm")
    n_majority = sum(1 for m in models if m[0] == "majority")
    print(f"Training done - {n_svm} SVMs + {n_majority} majority")


    y_pred_dev = predict_all_aspects(models, X_dev, len(dev_df))
    y_pred_test = predict_all_aspects(models, X_test, len(test_df))


    os.makedirs("outputs/results", exist_ok=True)

    print("\n" + "-" * 60)
    dev_metrics = evaluate_predictions(
        y_dev,
        y_pred_dev,
        title="SVM Baseline - DEV",
        exclude_aspects=ZERO_TRAIN_ASPECTS,
        save_path="outputs/results/svm_baseline_dev_metrics.json",
    )

    print("\n" + "-" * 60)
    test_metrics = evaluate_predictions(
        y_test,
        y_pred_test,
        title="SVM Baseline - TEST",
        exclude_aspects=ZERO_TRAIN_ASPECTS,
        save_path="outputs/results/svm_baseline_test_metrics.json",
    )


    generate_summary(dev_metrics, test_metrics)

    print("\n" + "=" * 60)
    print("SVM Baseline complete!")
    print(f"   Test ACD F1:      {test_metrics['macro_acd_f1']:.4f}")
    print(f"   Test SPC F1:      {test_metrics['macro_spc_f1']:.4f}")
    print(f"   Test Combined F1: {test_metrics['macro_combined_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
