"""
rag_retriever.py — Embedding train set + semantic retrieval.

Model embedding: keepitreal/vietnamese-sbert
  - Được train trên tiếng Việt
  - Tốt hơn all-MiniLM-L6-v2 (tiếng Anh) cho domain này

Aspect-aware pool:
  - Sau khi lấy top-K candidates bằng cosine similarity
  - Đảm bảo examples có đa dạng aspects
  - Tránh k examples đều về cùng 1 aspect
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))
from utils.constants import ASPECT_COLUMNS
from utils.helpers import set_seed


EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
# Fallback nếu model trên không load được:
EMBEDDING_MODEL_FALLBACK = "paraphrase-multilingual-MiniLM-L12-v2"


class ABSARetriever:
    """
    Semantic retriever cho ABSA few-shot examples.

    1. Embed toàn bộ train reviews (1 lần, cache kết quả)
    2. Với mỗi test review: tìm k train reviews tương đồng nhất
    3. Aspect-aware filter: đảm bảo k examples phủ đa dạng aspects
    """

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
        """Lazy load sentence transformer model."""
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"[Retriever] Loaded: {self.model_name}")
        except Exception as e:
            print(f"[WARN] {self.model_name} failed: {e}")
            print(f"[Retriever] Fallback: {EMBEDDING_MODEL_FALLBACK}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL_FALLBACK)

    def fit(self, train_df: pd.DataFrame, text_col: str = "processed_review"):
        """
        Embed toàn bộ train set và lưu cache.
        Nếu cache đã tồn tại: load thay vì tính lại.
        """
        self._load_model()
        self.train_df = train_df.reset_index(drop=True)

        # Load cache nếu có
        if os.path.exists(self.cache_path):
            self.train_embeds = np.load(self.cache_path)
            print(f"[Retriever] Loaded embeddings cache: {self.train_embeds.shape}")
            assert len(self.train_embeds) == len(train_df), \
                "Cache size mismatch — xóa cache và chạy lại"
            return self

        # Tính embeddings
        print(f"[Retriever] Embedding {len(train_df)} train reviews...")
        texts = train_df[text_col].astype(str).tolist()
        self.train_embeds = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize để dùng dot product = cosine
        )

        # Lưu cache
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(self.cache_path, self.train_embeds)
        print(f"[Retriever] Saved embeddings: {self.cache_path}")

        # Optional: build FAISS index để tìm kiếm nhanh hơn
        if self.use_faiss:
            self._build_faiss_index()

        return self

    def _build_faiss_index(self):
        """Build FAISS index cho fast retrieval (optional)."""
        try:
            import faiss
            dim = self.train_embeds.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner Product = cosine (after normalize)
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
        """
        Tìm k train indices gần nhất với query.

        Args:
            query:          text review cần tìm examples cho
            k:              số examples cần lấy
            aspect_aware:   nếu True → đảm bảo đa dạng aspects
            candidate_pool: lấy top N candidates trước khi filter

        Returns:
            list of k train DataFrame indices
        """
        self._load_model()

        # Embed query
        query_embed = self.model.encode(
            [query], normalize_embeddings=True
        )[0]  # [dim]

        # Tìm top candidates
        if self.use_faiss and hasattr(self, 'faiss_index'):
            scores, indices = self.faiss_index.search(
                query_embed.reshape(1, -1).astype(np.float32),
                candidate_pool
            )
            candidates = indices[0].tolist()
        else:
            # Numpy fallback
            sims = self.train_embeds @ query_embed  # cosine similarity
            candidates = np.argsort(sims)[::-1][:candidate_pool].tolist()

        if not aspect_aware or k >= candidate_pool:
            return candidates[:k]

        # Aspect-aware selection
        return self._aspect_aware_select(candidates, k)

    def _aspect_aware_select(self, candidates: list[int], k: int) -> list[int]:
        """
        Chọn k examples đảm bảo đa dạng aspects.

        Thuật toán greedy:
        1. Luôn lấy candidate top-1 (giống nhất)
        2. Với mỗi candidate tiếp theo: tính "aspect coverage score"
           = số aspects mới mà candidate này thêm vào pool
        3. Ưu tiên candidates tăng coverage nhiều nhất
        """
        selected = []
        covered_aspects = set()

        for cand_idx in candidates:
            if len(selected) >= k:
                break

            # Lấy aspects của candidate này
            row = self.train_df.iloc[cand_idx]
            cand_aspects = {
                asp for asp in ASPECT_COLUMNS
                if int(row[asp]) != 0
            }

            # Tính new coverage
            new_coverage = len(cand_aspects - covered_aspects)

            # Lấy top-1 luôn luôn; sau đó ưu tiên candidates tăng coverage
            if not selected or new_coverage > 0:
                selected.append(cand_idx)
                covered_aspects |= cand_aspects

        # Fill nếu chưa đủ k
        for cand_idx in candidates:
            if len(selected) >= k:
                break
            if cand_idx not in selected:
                selected.append(cand_idx)

        return selected[:k]
