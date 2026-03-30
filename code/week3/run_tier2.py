"""Entry point Tầng 2 — Few-shot ICL."""
import os, sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                      # code/week3
sys.path.insert(0, os.path.join(_HERE, '..', 'week1'))         # code/week1
from utils.helpers import set_seed

from icl_predictor import run_icl_ablation


def main(api_keys: dict, max_samples: int = None):
    if not api_keys:
        print("[ERROR] Không có API key nào. Set OPENAI_API_KEY hoặc GEMINI_API_KEY.")
        return
    set_seed(42)
    train_df = pd.read_csv("data/train_preprocessed.csv")
    test_df  = pd.read_csv("data/test_preprocessed.csv")

    results = run_icl_ablation(
        test_df=test_df,
        train_df=train_df,
        providers=["openai", "gemini"],  # bỏ provider nào không có API key
        k_values=[2, 4, 8],
        api_keys=api_keys,
        results_dir="outputs/results",
        max_samples=max_samples,
    )
    print("\n✅ Tầng 2 hoàn tất")
    for exp, m in results.items():
        print(f"  {exp}: Combined F1 = {m['macro_combined_f1']:.4f}")


if __name__ == "__main__":
    import os
    api_keys = {}
    if os.environ.get("OPENAI_API_KEY"):
        api_keys["openai"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        api_keys["gemini"] = os.environ["GEMINI_API_KEY"]
    main(api_keys=api_keys)
