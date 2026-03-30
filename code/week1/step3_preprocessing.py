"""
step3_preprocessing.py — Preprocessing pipeline cho ABSA VLSP 2018 Hotel.

Pipeline (deterministic + idempotent):
    1. normalize_unicode  — NFC normalization
    2. normalize_whitespace — xóa khoảng trắng thừa
    3. replace_teencode   — từ dài nhất đến ngắn nhất
    4. remove_special_chars — giữ dấu câu cơ bản
    5. segment_words      — VnCoreNLP hoặc underthesea fallback

Chạy từ root project:
    python code/week1/step3_preprocessing.py
"""
import os
import re
import sys
import unicodedata
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.constants import (
    ASPECT_COLUMNS,
    TRAIN_PATH,
    DEV_PATH,
    TEST_PATH,
    TRAIN_PREPROCESSED,
    DEV_PREPROCESSED,
    TEST_PREPROCESSED,
)
from utils.helpers import set_seed, log_versions


# ─── Teencode Dictionary (domain khách sạn) ──────────────────────────────────

TEENCODE_DICT = {
    # Dịch vụ
    "dv":         "dịch vụ",
    "sv":         "dịch vụ",
    "ntv":        "nhân viên",
    "nv":         "nhân viên",
    "nviên":      "nhân viên",
    "letan":      "lễ tân",
    "lt":         "lễ tân",
    "qlý":        "quản lý",
    "ql":         "quản lý",
    # Phòng / tiện nghi
    "phg":        "phòng",
    "phòg":       "phòng",
    "ks":         "khách sạn",
    "ksan":       "khách sạn",
    "kháchsạn":   "khách sạn",
    "bfst":       "bữa sáng",
    "bf":         "bữa sáng",
    "bsáng":      "bữa sáng",
    "res":        "nhà hàng",
    "nhàhàng":    "nhà hàng",
    # Vị trí
    "hn":         "Hà Nội",
    "hcm":        "Hồ Chí Minh",
    "sg":         "Sài Gòn",
    "tphcm":      "Tp Hồ Chí Minh",
    "vt":         "vị trí",
    # Đánh giá tích cực
    "ok":         "tốt",
    "oke":        "tốt",
    "okie":       "tốt",
    "oki":        "tốt",
    "tuyệt":      "tuyệt vời",
    "tuyêt":      "tuyệt vời",
    "hài lòg":    "hài lòng",
    "hàilòng":    "hài lòng",
    "thik":       "thích",
    "thíck":      "thích",
    "ngon":       "ngon",
    # Đánh giá tiêu cực
    "tệ hại":     "tệ hại",
    "tệhại":      "tệ hại",
    "chán":       "chán",
    "bth":        "bình thường",
    "bt":         "bình thường",
    "bthg":       "bình thường",
    # Phủ định
    "k":          "không",
    "ko":         "không",
    "kg":         "không",
    "kh":         "không",
    "khg":        "không",
    "hok":        "không",
    # Động từ / trợ từ
    "đc":         "được",
    "dc":         "được",
    "cx":         "cũng",
    "cg":         "cũng",
    "vs":         "với",
    "nx":         "nữa",
    "r":          "rồi",
    "rồi":        "rồi",
    "đã":         "đã",
    "mk":         "mình",
    "mik":        "mình",
    "mn":         "mọi người",
    "mng":        "mọi người",
    "sp":         "sản phẩm",
    "tc":         "tất cả",
    "trc":        "trước",
    "sau":        "sau",
    "ntn":        "như thế nào",
    "nma":        "nhưng mà",
    "nhưg":       "nhưng",
    "vì":         "vì",
    "đb":         "đặc biệt",
    "hd":         "hướng dẫn",
    "tt":         "thông tin",
    "ck":         "check",
    "ck-in":      "check-in",
    "ckin":       "check-in",
    "ck-out":     "check-out",
    "ckout":      "check-out",
    # Emoji/ký hiệu common (thay bằng mô tả)
    ":)":         "vui",
    ":D":         "rất vui",
    ":(":         "buồn",
    ":'(":        "rất buồn",
}


# ─── Preprocessing functions ─────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC normalization — critical cho tiếng Việt."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Xóa whitespace thừa, normalize newline/tab."""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def replace_teencode(text: str, teencode_dict: dict = TEENCODE_DICT) -> str:
    """
    Thay thế teencode bằng từ chuẩn.
    - Xử lý từ dài nhất trước (tránh partial match)
    - Case-insensitive
    - Dùng word boundary \\b
    """
    # Sort by length desc để tránh "bthg" bị match "bt" trước
    sorted_keys = sorted(teencode_dict.keys(), key=len, reverse=True)
    for key in sorted_keys:
        pattern = r'\b' + re.escape(key) + r'\b'
        text = re.sub(pattern, teencode_dict[key], text, flags=re.IGNORECASE)
    return text


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Xóa ký tự không cần thiết.
    Giữ: chữ cái (kể cả tiếng Việt), số, khoảng trắng.
    Nếu keep_punctuation: giữ thêm .,!?;:-/()
    """
    if keep_punctuation:
        pattern = r'[^\w\s\.,!?;:\-/\(\)]'
    else:
        pattern = r'[^\w\s]'
    return re.sub(pattern, ' ', text, flags=re.UNICODE)


def preprocess_text(
    text: str,
    segmenter: Optional["VnCoreNLPSegmenter"] = None,
    do_segment: bool = True,
) -> str:
    """
    Full pipeline: unicode → whitespace → teencode → special_chars → segment.
    Nếu do_segment=False hoặc segmenter=None: bỏ qua bước segment.
    """
    text = normalize_unicode(str(text))
    text = normalize_whitespace(text)
    text = replace_teencode(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)  # lần 2 sau khi xóa chars
    if do_segment and segmenter is not None:
        text = segmenter.segment(text)
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    segmenter: Optional["VnCoreNLPSegmenter"] = None,
    text_col: str = "Review",
    output_col: str = "processed_review",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply preprocess_text cho toàn bộ DataFrame.
    - Thêm cột output_col (giữ nguyên cột gốc)
    - Nếu cache_path tồn tại: load cache, không chạy lại
    - Lưu cache sau khi xong
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Cache] Load preprocessed từ {cache_path}")
        cached = pd.read_csv(cache_path)
        if output_col in cached.columns:
            df = df.copy()
            df[output_col] = cached[output_col].values
            return df

    tqdm.pandas(desc=f"Preprocessing {text_col}")
    df = df.copy()
    df[output_col] = df[text_col].progress_apply(
        lambda t: preprocess_text(t, segmenter=segmenter)
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[Cache] Saved to {cache_path}")
    return df


# ─── VnCoreNLP Segmenter ─────────────────────────────────────────────────────

class VnCoreNLPSegmenter:
    """
    Wrapper VnCoreNLP với lazy loading + graceful fallback.

    Priority: VnCoreNLP (chính xác hơn) → underthesea (fallback)
    Lý do fallback: VnCoreNLP yêu cầu Java 8+, download model riêng (~200MB).

    Từ ghép được nối bằng '_': "khách_sạn", "nhân_viên", "lễ_tân"
    PhoBERT được train với định dạng này — BẮT BUỘC dùng word segment.
    """

    def __init__(
        self,
        vncorenlp_dir: Optional[str] = None,
        use_fallback: bool = True,
    ) -> None:
        self._segmenter = None
        self._using_fallback = False
        self.vncorenlp_dir = vncorenlp_dir
        self.use_fallback = use_fallback

    def _load(self) -> None:
        """Lazy load — chỉ load lần đầu gọi segment()."""
        if self._segmenter is not None or self._using_fallback:
            return
        try:
            import py_vncorenlp
            # py_vncorenlp đổi cwd sang save_dir; ta phục hồi để tránh ảnh hưởng path tương đối.
            prev_cwd = os.getcwd()
            try:
                if self.vncorenlp_dir:
                    self._segmenter = py_vncorenlp.VnCoreNLP(
                        annotators=["wseg"],
                        save_dir=self.vncorenlp_dir,
                    )
                else:
                    self._segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
            finally:
                os.chdir(prev_cwd)
            print("[Segmenter] VnCoreNLP loaded successfully.")
        except Exception as e:
            if self.use_fallback:
                print(f"[WARN] VnCoreNLP unavailable ({e}). Fallback: underthesea.")
                print("[WARN] Kết quả word-segment có thể kém hơn ~1-2% F1.")
                self._using_fallback = True
            else:
                raise

    def segment(self, text: str) -> str:
        """Segment 1 text, trả về string với từ ghép nối bằng '_'."""
        self._load()
        if self._using_fallback:
            from underthesea import word_tokenize
            return word_tokenize(text, format="text").replace(" ", "_").replace("__", "_")
        result = self._segmenter.word_segment(text)
        if isinstance(result, list):
            return " ".join(result)
        return result

    def segment_batch(self, texts: list[str], batch_size: int = 64) -> list[str]:
        """Segment batch với progress bar tqdm."""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Word segmenting"):
            batch = texts[i:i + batch_size]
            results.extend([self.segment(t) for t in batch])
        return results

    def close(self) -> None:
        """Giải phóng VnCoreNLP process."""
        if self._segmenter and not self._using_fallback:
            try:
                self._segmenter.close()
            except Exception:
                pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    log_versions()

    # Demo pipeline trên 5 review mẫu
    demo_inputs = [
        "Phòng sv rất tệ, nv ko thân thiện dc",
        "ks ok, bfst ngon, vt thuận tiện",
        "Phòng sạch nhưng giá hơi cao :(",
        "khg hài lòng với dv của ntv",
        "Tuyệt vời!!! Sẽ quay lại :D",
    ]

    print("\n" + "=" * 60)
    print("Demo Preprocessing Pipeline (without word segmentation)")
    print("=" * 60)
    for inp in demo_inputs:
        result = preprocess_text(inp, do_segment=False)
        print(f"IN:  {inp}")
        print(f"OUT: {result}")
        print()

    # Preprocess train/dev/test nếu CSV tồn tại
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    segmenter = VnCoreNLPSegmenter(
        vncorenlp_dir=os.path.join(project_root, "vncorenlp"),
        use_fallback=True,
    )

    path_pairs = [
        (TRAIN_PATH, TRAIN_PREPROCESSED, "train"),
        (DEV_PATH,   DEV_PREPROCESSED,   "dev"),
        (TEST_PATH,  TEST_PREPROCESSED,  "test"),
    ]

    for csv_path, cache_path, split_name in path_pairs:
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} không tồn tại — bỏ qua {split_name}")
            continue
        print(f"\n[Preprocessing] {split_name}...")
        df = pd.read_csv(csv_path)
        df = preprocess_dataframe(df, segmenter=segmenter, cache_path=cache_path)
        print(f"  {split_name}: {len(df)} samples preprocessed")
        # Show 2 examples
        for idx in [0, 1]:
            if idx < len(df):
                print(f"  Example {idx}: {df.iloc[idx]['processed_review'][:120]}")

    segmenter.close()
    print("\n✅ Preprocessing hoàn tất.")


if __name__ == "__main__":
    main()
