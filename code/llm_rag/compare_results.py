
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))


REPORT_RESULTS = [
    {
        "method": "SVM + TF-IDF",
        "type": "Traditional baseline",
        "acd_f1": 0.4074,
        "spc_f1": 0.2272,
        "combined_f1": 0.3173,
        "note": "Baseline cổ điển",
    },
    {
        "method": "PhoBERT cls_only + VnCoreNLP",
        "type": "Supervised transformer",
        "acd_f1": 0.6360,
        "spc_f1": 0.4727,
        "combined_f1": 0.5543,
        "note": "Model dự đoán chính, không ensemble",
    },
    {
        "method": "LLM + RAG k=8",
        "type": "LLM few-shot + retrieval",
        "acd_f1": 0.4034,
        "spc_f1": 0.3031,
        "combined_f1": 0.3532,
        "note": "RAG cải thiện LLM nhưng vẫn kém PhoBERT",
    },
    {
        "method": "PhoBERT + LLM explanation",
        "type": "Practical explainability layer",
        "acd_f1": None,
        "spc_f1": None,
        "combined_f1": None,
        "note": "LLM giải thích/evidence, không làm classifier chính",
    },
]


def generate_comparison_table(
    save_path: str = "outputs/results/final_four_direction_comparison.md",
) -> str:
    rows = []
    for item in REPORT_RESULTS:
        rows.append(
            {
                "Method": item["method"],
                "Type": item["type"],
                "ACD F1": _fmt(item["acd_f1"]),
                "SPC F1": _fmt(item["spc_f1"]),
                "Combined F1": _fmt(item["combined_f1"]),
                "Note": item["note"],
            }
        )

    table = pd.DataFrame(rows).to_markdown(index=False)
    analysis = """

## Main Takeaways

1. Traditional SVM is far below transformer-based supervised learning.
2. PhoBERT is the strongest prediction model among retained methods.
3. RAG improves LLM few-shot prompting, but LLM prediction remains much weaker than PhoBERT.
4. The final practical system uses PhoBERT for prediction and LLM for explanation/evidence/error analysis.
"""
    report = f"# Final Method Comparison\n\n{table}{analysis}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] Saved: {save_path}")
    return report


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    generate_comparison_table()
