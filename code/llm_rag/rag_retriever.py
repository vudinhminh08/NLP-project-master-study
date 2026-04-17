
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from utils.constants import ASPECT_COLUMNS
from utils.helpers import set_seed


EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

EMBEDDING_MODEL_FALLBACK = "paraphrase-multilingual-MiniLM-L12-v2"


class ABSARetriever:

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        cache_path: str = "outputs/results/embeddings_cache.npy",
        use_faiss: bool = True,
    ):
        self.model_name  = model_name
        self.cache_path  = cache_path
        self.use_faiss   = use_faiss
        self.model       = None
        self.train_embeds = None
        self.train_df    = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"[Retriever] Loaded: {self.model_name}")
        except Exception as e:
            print(f"[WARN] {self.model_name} failed: {e}")
            print(f"[Retriever] Fallback: {EMBEDDING_MODEL_FALLBACK}")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBEDDING_MODEL_FALLBACK)
            except Exception as e2:
                raise RuntimeError(
                    "Failed to load Vietnamese SBERT or multilingual sentence-transformers fallback. "
                    "Please check that sentence-transformers and its model dependencies are installed correctly."
                ) from e2

    def fit(self, train_df: pd.DataFrame, text_col: str = "processed_review"):
        self._load_model()
        self.train_df = train_df.reset_index(drop=True)


        if os.path.exists(self.cache_path):
            self.train_embeds = np.load(self.cache_path)
            print(f"[Retriever] Loaded embeddings cache: {self.train_embeds.shape}")
            assert len(self.train_embeds) == len(train_df), \
                "Cache size mismatch"
            return self


        print(f"[Retriever] Embedding {len(train_df)} train reviews...")
        texts = train_df[text_col].astype(str).tolist()
        self.train_embeds = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )


        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(self.cache_path, self.train_embeds)
        print(f"[Retriever] Saved embeddings: {self.cache_path}")


        if self.use_faiss:
            self._build_faiss_index()

        return self

    def _build_faiss_index(self):
        try:
            import faiss
            dim = self.train_embeds.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.train_embeds.astype(np.float32))
            print(f"[FAISS] Index built: {self.faiss_index.ntotal} vectors")
        except ImportError:
            print("[WARN] FAISS not available, using numpy cosine similarity")
            self.use_faiss = False

    def retrieve(
        self,
        query: str,
        k: int = 4,
        aspect_aware: bool = True,
        candidate_pool: int = 50,
    ) -> list[int]:
        self._load_model()


        query_embed = self.model.encode(
            [query], normalize_embeddings=True
        )[0]


        if self.use_faiss and hasattr(self, 'faiss_index'):
            scores, indices = self.faiss_index.search(
                query_embed.reshape(1, -1).astype(np.float32),
                candidate_pool
            )
            candidates = indices[0].tolist()
        else:

            sims = self.train_embeds @ query_embed
            candidates = np.argsort(sims)[::-1][:candidate_pool].tolist()

        if not aspect_aware or k >= candidate_pool:
            return candidates[:k]


        return self._aspect_aware_select(candidates, k)

    def _aspect_aware_select(self, candidates: list[int], k: int) -> list[int]:
        selected = []
        covered_aspects = set()

        for cand_idx in candidates:
            if len(selected) >= k:
                break


            row = self.train_df.iloc[cand_idx]
            cand_aspects = {
                asp for asp in ASPECT_COLUMNS
                if int(row[asp]) != 0
            }


            new_coverage = len(cand_aspects - covered_aspects)


            if not selected or new_coverage > 0:
                selected.append(cand_idx)
                covered_aspects |= cand_aspects


        for cand_idx in candidates:
            if len(selected) >= k:
                break
            if cand_idx not in selected:
                selected.append(cand_idx)

        return selected[:k]
