"""
LLM explanation service for PhoBERT predictions.

The LLM receives only PhoBERT predictions and is not allowed to add or modify
aspect/sentiment labels. Guardrails from code/llm_explainability are reused.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_RAG_DIR = PROJECT_ROOT / "code" / "llm_rag"
LLM_EXPLAINABILITY_DIR = PROJECT_ROOT / "code" / "llm_explainability"

for path in (LLM_RAG_DIR, LLM_EXPLAINABILITY_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from llm_client import LLMClient  # noqa: E402
from llm_explainer import explain_review  # noqa: E402


DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "llm_cache" / "app"


def has_openai_api_key(api_key: str | None = None) -> bool:
    return bool((api_key or os.environ.get("OPENAI_API_KEY") or "").strip())


class ExplanationService:
    """Thin wrapper around the existing explanation pipeline."""

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "openai",
        model: str | None = None,
        cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
    ) -> None:
        self.client = LLMClient(
            provider=provider,
            api_key=(api_key or None),
            model=model,
            cache_dir=str(cache_dir),
        )

    def explain(self, review: str, predictions: list[dict]) -> dict:
        return explain_review(
            review=review,
            predictions=predictions,
            llm_client=self.client,
            rag_examples=None,
            temperature=0.0,
            use_cache=True,
        )
