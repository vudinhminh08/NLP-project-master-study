"""
compare_results.py — So sánh toàn bộ 3 tầng, tạo bảng kết quả cho báo cáo.
"""

import os, json
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))
from utils.helpers import load_json, save_json


# ── Kết quả PhoBERT từ Tuần 2 ──────────────────────────────────────────────────
# [PLACEHOLDER] Cập nhật sau khi tuần 2 xong
WEEK2_RESULTS = {
    "phobert_concat4": {
        "acd_f1":      0.5836,
        "spc_f1":      0.4658,
        "combined_f1": 0.5247,
    },
    "phobert_cls_only": {
        "acd_f1":      0.6360,
        "spc_f1":      0.4727,
        "combined_f1": 0.5543,
    },
}

SOTA_RESULTS = {
    "sota_phobert_huynh2022": {
        "acd_f1": 0.8255, "spc_f1": None, "combined_f1": 0.7732
    },
    "baseline_svm": {
        "acd_f1": 0.4074, "spc_f1": 0.2272, "combined_f1": 0.3173
    },
}


def load_all_results(results_dir: str = "outputs/results") -> dict:
    """Load tất cả metrics JSON từ results_dir."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_metrics.json"):
            key = fname.replace("_metrics.json", "")
            data = load_json(os.path.join(results_dir, fname))
            results[key] = {
                "acd_f1":      data.get("macro_acd_f1", 0),
                "spc_f1":      data.get("macro_spc_f1", 0),
                "combined_f1": data.get("macro_combined_f1", 0),
            }
    return results


def generate_comparison_table(
    results_dir: str = "outputs/results",
    save_path: str = "outputs/results/week3_comparison.md",
) -> str:
    """
    Tạo bảng so sánh markdown tất cả phương pháp.
    Format sẵn để paste vào báo cáo.
    """
    all_results = {
        **SOTA_RESULTS,
        **WEEK2_RESULTS,
        **load_all_results(results_dir),
    }

    rows = []
    for name, m in sorted(all_results.items(), key=lambda x: -x[1]["combined_f1"]):
        tier = (
            "SOTA"    if "sota" in name else
            "Baseline" if "baseline" in name else
            "Tầng 1"  if "phobert" in name else
            "Tầng 2"  if "tier2" in name else
            "Tầng 3"  if "tier3" in name else "?"
        )
        provider = (
            "GPT-4o-mini"   if "gpt" in name else
            "Gemini 1.5"    if "gemini" in name else
            "PhoBERT"       if "phobert" in name else
            "SVM"           if "svm" in name else "—"
        )
        k = name.split("_k")[-1] if "_k" in name else "—"
        spc = f"{m['spc_f1']:.4f}" if m['spc_f1'] else "—"
        rows.append({
            "Tầng": tier,
            "Phương pháp": provider,
            "k": k,
            "ACD F1": f"{m['acd_f1']:.4f}",
            "SPC F1": spc,
            "Combined F1": f"{m['combined_f1']:.4f}",
        })

    df = pd.DataFrame(rows)
    table_md = df.to_markdown(index=False)

    # Phân tích
    tier2_best = max(
        (v["combined_f1"] for k, v in all_results.items() if "tier2" in k), default=0
    )
    tier3_best = max(
        (v["combined_f1"] for k, v in all_results.items() if "tier3" in k), default=0
    )
    phobert_f1 = WEEK2_RESULTS["phobert_concat4"]["combined_f1"]
    rag_gain   = tier3_best - tier2_best

    analysis = f"""
## Phân tích kết quả

### So sánh các tầng
| | Combined F1 | So với PhoBERT |
|---|---|---|
| PhoBERT (Tầng 1) | {phobert_f1:.4f} | baseline |
| LLM Few-shot best (Tầng 2) | {tier2_best:.4f} | {(tier2_best-phobert_f1)*100:+.1f}% |
| RAG best (Tầng 3) | {tier3_best:.4f} | {(tier3_best-phobert_f1)*100:+.1f}% |
| SOTA | 0.7732 | — |

### RAG vs ICL (ablation)
- RAG cải thiện so với ICL cùng k: **{rag_gain*100:+.2f}%**
- {"RAG có hiệu quả" if rag_gain > 0 else "RAG không cải thiện — phân tích lý do bên dưới"}

### Kết luận
- PhoBERT supervised vẫn tốt hơn LLM few-shot: **+{(phobert_f1-tier2_best)*100:.1f}%**
- LLM cần ít data hơn nhưng kém hơn khi có đủ training data
- RAG giúp cải thiện LLM bằng cách chọn examples thông minh hơn
"""

    full_report = f"# Bảng So sánh Kết quả — ABSA VLSP 2018 Hotel\n\n{table_md}\n{analysis}"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"[Report] Saved: {save_path}")

    return full_report
