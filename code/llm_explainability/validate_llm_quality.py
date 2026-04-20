"""validate_llm_quality.py — 3 Independent Validation Methods

Mục đích: Cung cấp 3 bằng chứng độc lập rằng LLM trong hướng
PhoBERT + LLM Explanation không bịa nội dung (không hallucinate).

Method 1 — Label Consistency (N=200)
    Claim : LLM không bịa label (aspect/sentiment)
    Metric: % items có sentiment khớp với PhoBERT predictions
    Output: label_consistency_rate (%)

Method 2 — Evidence Groundedness via BERTScore (N=200)
    Claim : LLM không bịa câu trích (evidence)
    Metric: Mean BERTScore F1(evidence, review)
    Backbone: vinai/phobert-base-v2
    Citation: Zhang et al., 2020
    Output: mean_bertscore_f1 (0–1)

Method 3 — Human Rubric (N=30, 2 criteria)
    Claim : LLM không bịa nội dung diễn giải
    Metrics: Mean Faithfulness (0–2), Mean Usefulness (0–2)
    Yêu cầu: Annotation thủ công → rubric_template.csv

CLI usage:
    python validate_llm_quality.py \\
        --samples outputs/results/llm_explainability/explanation_samples.json \\
        [--rubric_csv outputs/results/llm_explainability/rubric_annotated.csv] \\
        [--n 200] [--no_bertscore]
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Optional


# =============================================================================
# Method 1: Label Consistency
# =============================================================================

def compute_label_consistency(samples: list, n: int = 200) -> dict:
    """
    Claim: LLM không bịa label (aspect & sentiment).

    Dùng metadata validation đã lưu trong mỗi sample khi generate:
      - dropped_items      : số items LLM thêm aspect KHÔNG có trong PhoBERT
      - sentiment_restored : số items LLM đặt sentiment SAI → đã bị sửa lại
      - num_items          : số items hợp lệ sau validate

    3 metrics:
      aspect_preservation_rate   = (valid_items) / (valid_items + dropped_spurious) × 100
      sentiment_consistency_rate = (valid_items - sentiment_wrong) / valid_items × 100
      overall_label_accuracy     = (valid_items - sentiment_wrong) / total_generated × 100
    """
    samples_n = samples[:n]
    total_items_valid      = 0
    total_sentiment_wrong  = 0
    total_dropped_spurious = 0
    n_parse_fail = 0

    for s in samples_n:
        val = s.get("explanation", {}).get("validation", {})
        if val.get("parse_fail", 0) or val.get("skipped_llm", False):
            n_parse_fail += 1
            continue
        total_items_valid      += val.get("num_items", 0)
        total_dropped_spurious += val.get("dropped_items", 0)
        total_sentiment_wrong  += val.get("sentiment_restored", 0)

    total_generated = total_items_valid + total_dropped_spurious

    if total_generated == 0:
        return {"error": "Không có items hợp lệ. Kiểm tra explanation_samples.json."}

    aspect_preservation   = total_items_valid / total_generated
    sentiment_consistency = (
        (total_items_valid - total_sentiment_wrong) / total_items_valid
        if total_items_valid > 0 else 0.0
    )
    overall_accuracy = (total_items_valid - total_sentiment_wrong) / total_generated

    return {
        "n_samples":                  len(samples_n),
        "n_parse_fail":               n_parse_fail,
        "total_llm_items_generated":  total_generated,
        "total_items_valid":          total_items_valid,
        "total_dropped_spurious":     total_dropped_spurious,
        "total_sentiment_wrong":      total_sentiment_wrong,
        "aspect_preservation_rate":   round(aspect_preservation   * 100, 1),
        "sentiment_consistency_rate": round(sentiment_consistency  * 100, 1),
        "overall_label_accuracy":     round(overall_accuracy       * 100, 1),
    }


# =============================================================================
# Method 2: BERTScore Evidence Groundedness
# =============================================================================

def compute_bertscore_groundedness(
    samples: list,
    n: int = 200,
    model_type: str = "vinai/phobert-base-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Claim: LLM không bịa câu trích (evidence).

    Công thức: mean BERTScore F1(evidence_i, review_i)
      - evidence_i : chuỗi LLM tự nhận là trích từ review
      - review_i   : review gốc tương ứng
      - Nếu evidence là substring/paraphrase của review → F1 cao (≥ 0.90)
      - Nếu evidence bịa hoàn toàn → F1 thấp hơn đáng kể

    Backbone: vinai/phobert-base-v2 (Zhang et al., 2020 — BERTScore)
    Note: BERTScore dùng subword tokenizer, hoạt động tốt với tiếng Việt
          kể cả khi text chưa qua word segmentation.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError(
            "Thiếu package bert-score.\n"
            "Chạy: pip install bert-score"
        )

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    samples_n     = samples[:n]
    evidence_list = []
    review_list   = []

    for s in samples_n:
        val = s.get("explanation", {}).get("validation", {})
        if val.get("parse_fail", 0) or val.get("skipped_llm", False):
            continue
        review = s.get("review", "")
        items  = s.get("explanation", {}).get("items", [])
        for item in items:
            ev = str(item.get("evidence", "")).strip()
            if ev:
                evidence_list.append(ev)
                review_list.append(review)

    if not evidence_list:
        return {"error": "Không có evidence nào. Kiểm tra explanation_samples.json."}

    if verbose:
        print(f"  Tính BERTScore cho {len(evidence_list)} evidence items")
        print(f"  Model: {model_type} | Device: {device}")

    P, R, F = bert_score_fn(
        cands=evidence_list,
        refs=review_list,
        model_type=model_type,
        lang="vi",
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    return {
        "n_samples":               len(samples_n),
        "n_evidence_items":        len(evidence_list),
        "model_type":              model_type,
        "mean_bertscore_precision": round(float(P.mean()), 4),
        "mean_bertscore_recall":    round(float(R.mean()), 4),
        "mean_bertscore_f1":        round(float(F.mean()), 4),
        "citation":                "Zhang et al. 2020 — BERTScore: Evaluating Text Generation with BERT",
    }


# =============================================================================
# Method 3: Human Rubric
# =============================================================================

RUBRIC_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║         HƯỚNG DẪN ANNOTATION — Human Rubric (N=30)          ║
╚══════════════════════════════════════════════════════════════╝

Điền cột 'faithfulness' và 'usefulness' (giá trị: 0, 1, hoặc 2)
vào file rubric_template.csv, sau đó lưu thành rubric_annotated.csv.

─── Tiêu chí 1: FAITHFULNESS (LLM có bịa không?) ───
  2 = Hoàn toàn trung thực: explanation & evidence đều có căn cứ từ review
  1 = Phần lớn trung thực: có thể hơi mơ hồ, nhưng không sai thực chất
  0 = Hallucination: có thông tin bịa ra không có trong review

─── Tiêu chí 2: USEFULNESS (Khuyến nghị có dùng được không?) ───
  2 = Rất hữu ích: cụ thể, actionable, phù hợp với aspect & sentiment
  1 = Hữu ích một phần: đúng hướng nhưng còn chung chung
  0 = Không hữu ích: không liên quan hoặc quá hiển nhiên

→ Sau khi điền xong, chạy:
   python validate_llm_quality.py \\
       --samples outputs/results/llm_explainability/explanation_samples.json \\
       --rubric_csv outputs/results/llm_explainability/rubric_annotated.csv \\
       --method 3
"""


def generate_rubric_template(
    samples: list,
    n: int = 30,
    save_path: str = "outputs/results/llm_explainability/rubric_template.csv",
    seed: int = 42,
) -> str:
    """
    Tạo CSV template để annotation thủ công (N=30 mẫu đại diện).

    Chiến lược chọn mẫu:
      - Lọc bỏ parse_fail / skipped
      - Ưu tiên 60% samples CÓ recommended_action (để test usefulness)
      - 40% còn lại: samples không có recommended_action
      - Mỗi row = 1 sample (lấy item có evidence dài nhất làm đại diện)
    """
    random.seed(seed)

    valid = [
        s for s in samples
        if not s.get("explanation", {}).get("validation", {}).get("parse_fail", 0)
        and not s.get("explanation", {}).get("validation", {}).get("skipped_llm", False)
        and s.get("explanation", {}).get("items")
    ]

    with_action    = [s for s in valid if s.get("explanation", {}).get("recommended_action")]
    without_action = [s for s in valid if not s.get("explanation", {}).get("recommended_action")]

    n_with    = min(len(with_action),    int(n * 0.6))
    n_without = min(len(without_action), n - n_with)

    selected = (
        random.sample(with_action,    n_with)
        + random.sample(without_action, n_without)
    )
    random.shuffle(selected)

    rows = []
    for rank, s in enumerate(selected, 1):
        expl       = s.get("explanation", {})
        items      = expl.get("items", [])
        rec_action = str(expl.get("recommended_action") or "")
        summary    = str(expl.get("overall_summary") or "")

        # Đại diện = item có evidence dài nhất (rõ ràng nhất để judge)
        rep = max(items, key=lambda x: len(x.get("evidence", "")), default={})

        rows.append({
            "rank":               rank,
            "sample_idx":         s.get("review_idx", ""),
            "review":             s.get("review", "")[:300],
            "aspect":             rep.get("aspect", ""),
            "sentiment":          rep.get("sentiment", ""),
            "evidence":           rep.get("evidence", ""),
            "explanation_item":   rep.get("explanation", ""),
            "overall_summary":    summary[:200],
            "recommended_action": rec_action[:200],
            "faithfulness":       "",   # annotator: 0 / 1 / 2
            "usefulness":         "",   # annotator: 0 / 1 / 2
        })

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(RUBRIC_GUIDE)
    print(f"✓ Template đã lưu: {save_path}")
    print(f"  Tổng mẫu: {len(rows)}  (có recommended_action: {n_with}, không có: {n_without})")
    return save_path


def compute_rubric_scores(annotated_csv: str) -> dict:
    """Đọc CSV đã annotation → tính mean Faithfulness & Usefulness."""
    import pandas as pd

    df = pd.read_csv(annotated_csv, encoding="utf-8-sig")
    df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce")
    df["usefulness"]   = pd.to_numeric(df["usefulness"],   errors="coerce")
    df_clean = df.dropna(subset=["faithfulness", "usefulness"])

    if df_clean.empty:
        return {
            "error": (
                "Chưa có annotation nào. "
                "Hãy điền cột faithfulness và usefulness vào CSV."
            )
        }

    return {
        "n_annotated":        int(len(df_clean)),
        "mean_faithfulness":  round(float(df_clean["faithfulness"].mean()), 2),
        "std_faithfulness":   round(float(df_clean["faithfulness"].std()),  2),
        "mean_usefulness":    round(float(df_clean["usefulness"].mean()),   2),
        "std_usefulness":     round(float(df_clean["usefulness"].std()),    2),
        "pct_fully_faithful": round(float((df_clean["faithfulness"] == 2).mean()) * 100, 1),
        "pct_fully_useful":   round(float((df_clean["usefulness"]   == 2).mean()) * 100, 1),
        "faithfulness_dist":  {
            int(k): int(v)
            for k, v in df_clean["faithfulness"].value_counts().sort_index().items()
        },
        "usefulness_dist": {
            int(k): int(v)
            for k, v in df_clean["usefulness"].value_counts().sort_index().items()
        },
    }


# =============================================================================
# Utility: load samples from JSON
# =============================================================================

def load_samples(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected format in {path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate LLM Explanation Quality — 3 independent methods"
    )
    parser.add_argument(
        "--samples", required=True,
        help="Path to explanation_samples.json"
    )
    parser.add_argument("--n",           type=int, default=200,  help="N cho method 1 & 2")
    parser.add_argument("--method",      choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--rubric_csv",  default=None,
                        help="Annotated rubric CSV (dùng cho Method 3 scoring)")
    parser.add_argument("--output_dir",  default="outputs/results/llm_explainability")
    parser.add_argument("--no_bertscore", action="store_true",
                        help="Bỏ qua Method 2 (BERTScore — cần GPU & bert-score package)")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # ── Method 1 ──
    if args.method in ("1", "all"):
        print("\n" + "=" * 60)
        print("Method 1 — Label Consistency")
        print("=" * 60)
        r1 = compute_label_consistency(samples, n=args.n)
        results["label_consistency"] = r1
        print(json.dumps(r1, ensure_ascii=False, indent=2))

    # ── Method 2 ──
    if args.method in ("2", "all") and not args.no_bertscore:
        print("\n" + "=" * 60)
        print("Method 2 — BERTScore Evidence Groundedness")
        print("=" * 60)
        r2 = compute_bertscore_groundedness(samples, n=args.n)
        results["bertscore_groundedness"] = r2
        print(json.dumps(r2, ensure_ascii=False, indent=2))

    # ── Method 3 ──
    if args.method in ("3", "all"):
        print("\n" + "=" * 60)
        print("Method 3 — Human Rubric")
        print("=" * 60)
        rubric_path = os.path.join(args.output_dir, "rubric_template.csv")
        generate_rubric_template(samples, n=30, save_path=rubric_path)

        if args.rubric_csv and os.path.exists(args.rubric_csv):
            r3 = compute_rubric_scores(args.rubric_csv)
            results["human_rubric"] = r3
            print("\nRubric scores:")
            print(json.dumps(r3, ensure_ascii=False, indent=2))
        else:
            print("\n[INFO] Chưa có annotation. Điền vào rubric_template.csv rồi chạy lại với --rubric_csv.")

    # ── Save combined report ──
    out_path = os.path.join(args.output_dir, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ Validation report saved: {out_path}")

    # ── Print summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY — LLM Non-Hallucination Evidence")
    print("=" * 60)
    if "label_consistency" in results:
        r = results["label_consistency"]
        print(f"  [M1] Label Consistency")
        print(f"       Aspect preservation  : {r.get('aspect_preservation_rate', '?')}%")
        print(f"       Sentiment consistency: {r.get('sentiment_consistency_rate', '?')}%")
        print(f"       Overall label accuracy: {r.get('overall_label_accuracy', '?')}%")
    if "bertscore_groundedness" in results:
        r = results["bertscore_groundedness"]
        print(f"  [M2] BERTScore F1 (evidence ↔ review): {r.get('mean_bertscore_f1', '?')}")
    if "human_rubric" in results:
        r = results["human_rubric"]
        print(f"  [M3] Faithfulness: {r.get('mean_faithfulness', '?')} / 2.0")
        print(f"       Usefulness  : {r.get('mean_usefulness',   '?')} / 2.0")
    print("=" * 60)


if __name__ == "__main__":
    main()
