import os, sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'data_processing'))
from utils.helpers import set_seed

from rag_predictor import run_rag_ablation
from compare_results import generate_comparison_table


def main(api_keys: dict, max_samples: int = None):
    if not api_keys:
        print("[ERROR] Không có API key nào. Set OPENAI_API_KEY hoặc GEMINI_API_KEY.")
        return
    set_seed(42)
    train_df = pd.read_csv("data/train_preprocessed.csv")
    test_df  = pd.read_csv("data/test_preprocessed.csv")

    results = run_rag_ablation(
        test_df=test_df,
        train_df=train_df,
        providers=["openai", "gemini"],
        k_values=[2, 4, 8],
        api_keys=api_keys,
        results_dir="outputs/results",
        max_samples=max_samples,
    )
    print("\nLLM + RAG hoàn tất")
    for exp, m in results.items():
        print(f"  {exp}: Combined F1 = {m['macro_combined_f1']:.4f}")


    generate_comparison_table()
    print("\nBảng so sánh: outputs/results/final_four_direction_comparison.md")


if __name__ == "__main__":
    import os
    api_keys = {}
    if os.environ.get("OPENAI_API_KEY"):
        api_keys["openai"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        api_keys["gemini"] = os.environ["GEMINI_API_KEY"]
    main(api_keys=api_keys)
