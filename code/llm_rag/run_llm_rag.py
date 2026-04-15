"""Entry point for LLM + RAG experiments."""
import os, sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                      # code/llm_rag
sys.path.insert(0, os.path.join(_HERE, '..', 'data_processing'))         # code/data_processing
from utils.helpers import set_seed

from rag_predictor import run_rag_ablation
from compare_results import generate_comparison_table


def resolve_providers(api_keys: dict) -> list[str]:
    providers = []
    if api_keys.get("openai"):
        providers.append("openai")
    if api_keys.get("gemini"):
        providers.append("gemini")

    # Enable Ollama explicitly via env to avoid accidental local calls.
    if os.environ.get("OLLAMA_ENABLE", "0") == "1":
        providers.append("ollama")

    return providers


def main(api_keys: dict, max_samples: int = None):
    providers = resolve_providers(api_keys)
    if not providers:
        print(
            "[ERROR] Không có provider nào khả dụng. "
            "Set OPENAI_API_KEY/GEMINI_API_KEY hoặc OLLAMA_ENABLE=1."
        )
        return

    set_seed(42)
    train_df = pd.read_csv("data/train_preprocessed.csv")
    test_df  = pd.read_csv("data/test_preprocessed.csv")

    results = run_rag_ablation(
        test_df=test_df,
        train_df=train_df,
        providers=providers,
        k_values=[2, 4, 8],
        api_keys=api_keys,
        results_dir="outputs/results",
        max_samples=max_samples,
    )
    print("\n✅ LLM + RAG hoàn tất")
    for exp, m in results.items():
        print(f"  {exp}: Combined F1 = {m['macro_combined_f1']:.4f}")

    # Tạo bảng so sánh tổng hợp
    generate_comparison_table()
    print("\n📊 Bảng so sánh: outputs/results/final_four_direction_comparison.md")


if __name__ == "__main__":
    import os
    api_keys = {}
    if os.environ.get("OPENAI_API_KEY"):
        api_keys["openai"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        api_keys["gemini"] = os.environ["GEMINI_API_KEY"]
    main(api_keys=api_keys)
