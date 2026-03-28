"""
step1_eda.py — EDA cho ABSA VLSP 2018 Hotel dataset.

Chạy từ root project:
    python code/week1/step1_eda.py

Output:
    outputs/eda/{split}_aspect_presence.png
    outputs/eda/{split}_label_breakdown.png
    outputs/eda/{split}_review_length.png
    outputs/eda/class_weights.json
    outputs/eda/encoder_config.json
"""
import os
import sys
import math
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Đảm bảo import được từ root project
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.constants import (
    ASPECT_COLUMNS,
    RARE_ASPECTS,
    TRAIN_PATH,
    DEV_PATH,
    TEST_PATH,
    EDA_DIR,
    CLASS_WEIGHTS_PATH,
    ENCODER_CONFIG_PATH,
    DEFAULT_ENCODER,
    ENCODER_OPTIONS,
)
from utils.helpers import set_seed, log_versions, save_json, format_metrics_table


# ─── A. Load splits ────────────────────────────────────────────────────────────

def load_splits() -> dict[str, pd.DataFrame]:
    """
    Load train/dev/test CSV.

    Validates:
        - Cột 'Review' (hoặc lowercase variant) tồn tại
        - Đủ 34 cột aspect
        - In shape, missing values, dtypes

    Returns:
        dict split_name → DataFrame
    """
    splits = {}
    paths = {"train": TRAIN_PATH, "dev": DEV_PATH, "test": TEST_PATH}

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} không tồn tại.\n"
                f"Download tại: https://github.com/ds4v/absa-vlsp-2018/tree/main/datasets/vlsp2018_hotel\n"
                f"Đặt vào thư mục data/ của project."
            )

        df = pd.read_csv(path)

        # Normalize column 'Review' (try lowercase if needed)
        if "Review" not in df.columns:
            col_map = {c.lower(): c for c in df.columns}
            if "review" in col_map:
                df = df.rename(columns={col_map["review"]: "Review"})
            else:
                raise ValueError(f"[{name}] Không tìm thấy cột 'Review' trong {path}. Columns: {list(df.columns)}")

        # Validate 34 aspect columns
        missing_aspects = [c for c in ASPECT_COLUMNS if c not in df.columns]
        if missing_aspects:
            raise ValueError(f"[{name}] Thiếu {len(missing_aspects)} aspect columns: {missing_aspects[:5]}...")

        # Print summary
        print(f"\n[Load] {name}: shape={df.shape}")
        missing = df.isnull().sum()
        if missing.any():
            print(f"  Missing values:\n{missing[missing > 0]}")
        else:
            print(f"  No missing values ✓")
        print(f"  Review dtype: {df['Review'].dtype}")
        print(f"  Label sample (first row): {df[ASPECT_COLUMNS].iloc[0].tolist()[:5]}...")

        splits[name] = df

    return splits


# ─── B. Label distribution analysis ──────────────────────────────────────────

def analyze_label_distribution(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Phân tích phân phối nhãn cho 34 aspects.

    Returns:
        DataFrame: index=aspect_name, columns=[count_0, count_1, count_2, count_3, pct_present]
    """
    records = []
    for col in ASPECT_COLUMNS:
        counts = df[col].value_counts().to_dict()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        c3 = counts.get(3, 0)
        total = len(df)
        pct_present = (c1 + c2 + c3) / total * 100
        records.append({
            "aspect":      col,
            "count_0":     c0,
            "count_1":     c1,
            "count_2":     c2,
            "count_3":     c3,
            "pct_present": pct_present,
        })

    dist_df = pd.DataFrame(records).set_index("aspect")

    # Top 5 common / rare
    sorted_df = dist_df.sort_values("pct_present", ascending=False)
    print(f"\n[{split_name}] Top 5 phổ biến nhất:")
    for asp, row in sorted_df.head(5).iterrows():
        print(f"  {asp}: {row['pct_present']:.1f}%")
    print(f"[{split_name}] Top 5 hiếm nhất:")
    for asp, row in sorted_df.tail(5).iterrows():
        print(f"  {asp}: {row['pct_present']:.1f}%")

    # Warn if actual rare aspects differ from constants
    actual_rare = dist_df[dist_df["count_1"] + dist_df["count_2"] + dist_df["count_3"] < 100].index.tolist()
    for asp in actual_rare:
        if asp not in RARE_ASPECTS:
            print(f"[WARN] Aspect '{asp}' có <100 mẫu nhưng không có trong RARE_ASPECTS — cần cập nhật constants.py")
    for asp in RARE_ASPECTS:
        total_present = dist_df.loc[asp, "count_1"] + dist_df.loc[asp, "count_2"] + dist_df.loc[asp, "count_3"]
        if total_present >= 100:
            print(f"[INFO] Aspect '{asp}' thực tế có {total_present} mẫu (>= 100), có thể cập nhật RARE_ASPECTS")

    return dist_df


# ─── C. Plot distributions ────────────────────────────────────────────────────

def plot_aspect_distribution(
    df: pd.DataFrame,
    split_name: str,
    save_dir: str,
) -> None:
    """
    Vẽ và lưu 2 biểu đồ phân phối aspect.

    Biểu đồ 1: {split}_aspect_presence.png — horizontal bar, % presence
    Biểu đồ 2: {split}_label_breakdown.png — stacked horizontal bar, 4 nhãn
    """
    os.makedirs(save_dir, exist_ok=True)

    # Chuẩn bị data
    records = []
    for col in ASPECT_COLUMNS:
        counts = df[col].value_counts().to_dict()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        c3 = counts.get(3, 0)
        total = len(df)
        pct_present = (c1 + c2 + c3) / total * 100
        records.append({"aspect": col, "pct_present": pct_present, "c0": c0, "c1": c1, "c2": c2, "c3": c3})

    plot_df = pd.DataFrame(records).sort_values("pct_present", ascending=True)

    # ── Biểu đồ 1: Aspect presence rate ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 14))
    colors = ["crimson" if asp in RARE_ASPECTS else "steelblue" for asp in plot_df["aspect"]]
    ax.barh(plot_df["aspect"], plot_df["pct_present"], color=colors)
    ax.axvline(x=10, color="gray", linestyle="--", linewidth=1, label="10% threshold")
    ax.set_xlabel("Presence Rate (%)")
    ax.set_title(f"Aspect Presence Rate — {split_name} split")
    ax.legend()

    # Legend for rare aspects
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="normal"),
        Patch(facecolor="crimson", label="rare (<100 samples)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path1 = os.path.join(save_dir, f"{split_name}_aspect_presence.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path1}")

    # ── Biểu đồ 2: Label breakdown (stacked bar) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 14))
    total = len(df)

    lefts = np.zeros(len(plot_df))
    label_colors = {
        "absent":   "lightgrey",
        "positive": "steelblue",
        "negative": "crimson",
        "neutral":  "goldenrod",
    }
    for col_key, label_name, color in [
        ("c0", "absent",   "lightgrey"),
        ("c1", "positive", "steelblue"),
        ("c2", "negative", "crimson"),
        ("c3", "neutral",  "goldenrod"),
    ]:
        vals = plot_df[col_key].values / total * 100
        ax.barh(plot_df["aspect"], vals, left=lefts, label=label_name, color=color)
        lefts += vals

    ax.set_xlabel("Percentage (%)")
    ax.set_title(f"Label Breakdown per Aspect — {split_name} split")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path2 = os.path.join(save_dir, f"{split_name}_label_breakdown.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path2}")


# ─── D. Review length analysis ────────────────────────────────────────────────

def analyze_review_length(df: pd.DataFrame, split_name: str, save_dir: str = EDA_DIR) -> dict:
    """
    Phân tích độ dài review: số từ + số ký tự.
    Gợi ý MAX_SEQ_LEN dựa trên p99 × 1.5, round lên bội số 64.

    Returns:
        dict gồm percentile stats + recommended_max_seq_len
    """
    os.makedirs(save_dir, exist_ok=True)

    word_counts = df["Review"].astype(str).str.split().str.len()
    char_counts = df["Review"].astype(str).str.len()

    percentiles = [25, 50, 75, 90, 95, 99]
    word_pcts = {f"p{p}": int(np.percentile(word_counts, p)) for p in percentiles}
    char_pcts = {f"p{p}": int(np.percentile(char_counts, p)) for p in percentiles}

    p99_words = word_pcts["p99"]
    raw_seq_len = p99_words * 1.5
    recommended = int(math.ceil(raw_seq_len / 64) * 64)

    print(f"\n[{split_name}] Review length stats (words): {word_pcts}")
    print(f"[{split_name}] Review length stats (chars): {char_pcts}")
    print(f"[Gợi ý] MAX_SEQ_LEN = {recommended} (p99 words={p99_words} × 1.5 factor, rounded to 64)")

    # Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(word_counts, bins=50, color="steelblue", edgecolor="white")
    ax1.axvline(p99_words, color="crimson", linestyle="--", label=f"p99={p99_words}")
    ax1.set_title(f"Word Count Distribution — {split_name}")
    ax1.set_xlabel("Number of words")
    ax1.set_ylabel("Count")
    ax1.legend()

    ax2.hist(char_counts, bins=50, color="goldenrod", edgecolor="white")
    ax2.axvline(char_pcts["p99"], color="crimson", linestyle="--", label=f"p99={char_pcts['p99']}")
    ax2.set_title(f"Character Count Distribution — {split_name}")
    ax2.set_xlabel("Number of characters")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{split_name}_review_length.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")

    return {
        "word_percentiles":       word_pcts,
        "char_percentiles":       char_pcts,
        "p99_word_count":         p99_words,
        "recommended_max_seq_len": recommended,
    }


# ─── E. Class imbalance analysis ─────────────────────────────────────────────

def analyze_class_imbalance(df: pd.DataFrame) -> dict:
    """
    Tính class weights theo công thức ds4v:
        majority_count / class_count

    Returns:
        dict với global_weights và per_aspect_weights
    """
    # Global weights
    all_labels = df[ASPECT_COLUMNS].values.flatten()
    counter = Counter(all_labels)
    majority = max(counter.values())
    global_weights = {int(cls): majority / count for cls, count in counter.items()}

    # Per-aspect weights
    per_aspect_weights: dict = {}
    header = f"{'Aspect':<40} {'n_absent':>8} {'n_pos':>6} {'n_neg':>6} {'n_neu':>6} {'w_pos':>6} {'w_neg':>6} {'w_neu':>6}"
    print(f"\n[Class Weights]")
    print(header)
    print("─" * len(header))

    for col in ASPECT_COLUMNS:
        counts = df[col].value_counts().to_dict()
        maj = max(counts.values())
        per_aspect_weights[col] = {int(cls): maj / cnt for cls, cnt in counts.items()}

        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        n2 = counts.get(2, 0)
        n3 = counts.get(3, 0)
        w1 = per_aspect_weights[col].get(1, 0)
        w2 = per_aspect_weights[col].get(2, 0)
        w3 = per_aspect_weights[col].get(3, 0)
        print(f"{col:<40} {n0:>8} {n1:>6} {n2:>6} {n3:>6} {w1:>6.2f} {w2:>6.2f} {w3:>6.2f}")

    result = {
        "global_weights":     {str(k): v for k, v in sorted(global_weights.items())},
        "per_aspect_weights": {col: {str(k): v for k, v in w.items()}
                               for col, w in per_aspect_weights.items()},
        "computed_from":      "train",
        "formula":            "majority_count / class_count",
    }

    save_json(result, CLASS_WEIGHTS_PATH)
    print(f"\n[Saved] {CLASS_WEIGHTS_PATH}")
    return result


# ─── F. Data quality check ───────────────────────────────────────────────────

def check_data_quality(df: pd.DataFrame, split_name: str) -> None:
    """
    Kiểm tra chất lượng dữ liệu:
        - Review trùng lặp
        - Review quá ngắn (<10 ký tự)
        - Review toàn nhãn 0
        - Patterns teencode phổ biến
    """
    import re

    n_total = len(df)
    reviews = df["Review"].astype(str)

    n_dup   = reviews.duplicated().sum()
    n_short = (reviews.str.len() < 10).sum()
    n_all_absent = (df[ASPECT_COLUMNS].sum(axis=1) == 0).sum()

    patterns = {
        r'\bko\b|\bk\b|\bkhg\b': 'không',
        r'\bdc\b|\bđc\b': 'được',
        r'\bsv\b': 'dịch vụ',
        r'\bnv\b|\bntv\b': 'nhân viên',
        r'\boks?\b|\boke\b': 'tốt/okay',
    }
    teencode_hits = {}
    for pattern, meaning in patterns.items():
        count = reviews.str.count(pattern, flags=re.IGNORECASE).sum()
        teencode_hits[meaning] = int(count)

    print(f"\n[Quality Check] {split_name} ({n_total} samples)")
    print(f"  Duplicate reviews:     {n_dup} ({n_dup/n_total*100:.1f}%)")
    print(f"  Short reviews (<10ch): {n_short} ({n_short/n_total*100:.1f}%)")
    print(f"  All-absent labels:     {n_all_absent} ({n_all_absent/n_total*100:.1f}%)")
    print(f"  Teencode patterns found:")
    for meaning, count in teencode_hits.items():
        print(f"    '{meaning}': {count} occurrences")


# ─── G. Save encoder config ──────────────────────────────────────────────────

def save_encoder_config(recommended_max_seq_len: int, split_stats: dict) -> None:
    """
    Lưu encoder config cho tuần 2.

    File: outputs/eda/encoder_config.json
    """
    encoder_info = ENCODER_OPTIONS[DEFAULT_ENCODER]
    config = {
        "recommended_max_seq_len": recommended_max_seq_len,
        "p99_word_count":          split_stats.get("p99_word_count", 0),
        "encoder_option":          DEFAULT_ENCODER,
        "encoder_hidden_size":     encoder_info["hidden_size"],
        "note":                    "concat_4_layers theo SOTA ds4v IEEE 2022",
    }
    save_json(config, ENCODER_CONFIG_PATH)
    print(f"[Saved] {ENCODER_CONFIG_PATH}")


# ─── H. Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    log_versions()
    splits = load_splits()

    for name, df in splits.items():
        print(f"\n{'='*60}")
        print(f"Split: {name.upper()}")
        print(f"{'='*60}")
        dist_df = analyze_label_distribution(df, name)
        plot_aspect_distribution(df, name, EDA_DIR)
        len_stats = analyze_review_length(df, name, EDA_DIR)
        check_data_quality(df, name)

        if name == "train":
            weights = analyze_class_imbalance(df)
            save_encoder_config(len_stats["recommended_max_seq_len"], len_stats)

    print(f"\n✅ EDA hoàn tất. Files đã lưu tại {EDA_DIR}/")
    print(f"   → class_weights.json: dùng cho weighted loss (tuần 2)")
    print(f"   → encoder_config.json: dùng cho model config (tuần 2)")


if __name__ == "__main__":
    main()
