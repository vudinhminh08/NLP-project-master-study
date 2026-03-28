"""
constants.py — Toàn bộ hằng số của dự án ABSA VLSP 2018 Hotel.
Mọi file khác đều import từ đây, không hardcode lại.
"""

# ─── 34 Aspect Columns (đúng thứ tự cột trong CSV ds4v) ───────────────────────
ASPECT_COLUMNS = [
    "FACILITIES#CLEANLINESS", "FACILITIES#COMFORT", "FACILITIES#DESIGN&FEATURES",
    "FACILITIES#GENERAL", "FACILITIES#MISCELLANEOUS", "FACILITIES#PRICES",
    "FACILITIES#QUALITY",
    "FOOD&DRINKS#MISCELLANEOUS", "FOOD&DRINKS#PRICES", "FOOD&DRINKS#QUALITY",
    "FOOD&DRINKS#STYLE&OPTIONS",
    "HOTEL#CLEANLINESS", "HOTEL#COMFORT", "HOTEL#DESIGN&FEATURES",
    "HOTEL#GENERAL", "HOTEL#MISCELLANEOUS", "HOTEL#PRICES", "HOTEL#QUALITY",
    "LOCATION#GENERAL",
    "ROOMS#CLEANLINESS", "ROOMS#COMFORT", "ROOMS#DESIGN&FEATURES",
    "ROOMS#GENERAL", "ROOMS#MISCELLANEOUS", "ROOMS#PRICES", "ROOMS#QUALITY",
    "ROOM_AMENITIES#CLEANLINESS", "ROOM_AMENITIES#COMFORT",
    "ROOM_AMENITIES#DESIGN&FEATURES", "ROOM_AMENITIES#GENERAL",
    "ROOM_AMENITIES#MISCELLANEOUS", "ROOM_AMENITIES#PRICES",
    "ROOM_AMENITIES#QUALITY",
    "SERVICE#GENERAL",
]

NUM_ASPECTS = len(ASPECT_COLUMNS)  # 34
assert NUM_ASPECTS == 34, "Phải có đúng 34 aspects"

# ─── Label Mapping ─────────────────────────────────────────────────────────────
LABEL_TO_IDX = {"absent": 0, "positive": 1, "negative": 2, "neutral": 3}
IDX_TO_LABEL = {0: "absent", 1: "positive", 2: "negative", 3: "neutral"}
NUM_LABELS   = 4  # 0=absent, 1=pos, 2=neg, 3=neu

# ─── Entity Groups (dùng cho EDA + báo cáo) ───────────────────────────────────
ENTITY_GROUPS = {
    "FACILITIES":     [c for c in ASPECT_COLUMNS if c.startswith("FACILITIES")],
    "FOOD&DRINKS":    [c for c in ASPECT_COLUMNS if c.startswith("FOOD&DRINKS")],
    "HOTEL":          [c for c in ASPECT_COLUMNS if c.startswith("HOTEL#")],
    "LOCATION":       [c for c in ASPECT_COLUMNS if c.startswith("LOCATION")],
    "ROOMS":          [c for c in ASPECT_COLUMNS if c.startswith("ROOMS#")],
    "ROOM_AMENITIES": [c for c in ASPECT_COLUMNS if c.startswith("ROOM_AMENITIES")],
    "SERVICE":        [c for c in ASPECT_COLUMNS if c.startswith("SERVICE")],
}

# ─── Rare Aspects — Cập nhật từ EDA thực tế (6 → 8) ──────────────────────────
# Dùng cho: weighted loss, phân tích lỗi, prompt LLM (liệt kê explicit)
RARE_ASPECTS = [
    "FACILITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",         # 0 mẫu pos/neg/neu trong train!
    "ROOM_AMENITIES#MISCELLANEOUS",  # chỉ có nhãn negative
    "ROOM_AMENITIES#CLEANLINESS",
    "ROOM_AMENITIES#DESIGN&FEATURES",
    "HOTEL#DESIGN&FEATURES",
    "ROOMS#MISCELLANEOUS",           # mới từ EDA: chỉ có nhãn negative
    "FOOD&DRINKS#MISCELLANEOUS",     # mới từ EDA: weight pos=426, neg=597
]

# Aspect không có bất kỳ mẫu nào trong train — exclude khỏi Macro-F1
ZERO_TRAIN_ASPECTS = ["ROOM_AMENITIES#PRICES"]

# ─── Paths (relative từ root project) ─────────────────────────────────────────
DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
EDA_DIR     = "outputs/eda"
MODEL_DIR   = "outputs/models"
RESULTS_DIR = "outputs/results"

TRAIN_PATH  = f"{DATA_DIR}/train.csv"
DEV_PATH    = f"{DATA_DIR}/dev.csv"
TEST_PATH   = f"{DATA_DIR}/test.csv"

# Preprocessed cache (tạo bởi step3_preprocessing.py)
TRAIN_PREPROCESSED = f"{DATA_DIR}/train_preprocessed.csv"
DEV_PREPROCESSED   = f"{DATA_DIR}/dev_preprocessed.csv"
TEST_PREPROCESSED  = f"{DATA_DIR}/test_preprocessed.csv"

# EDA outputs (tạo bởi step1_eda.py, dùng bởi tuần 2)
CLASS_WEIGHTS_PATH  = f"{EDA_DIR}/class_weights.json"
ENCODER_CONFIG_PATH = f"{EDA_DIR}/encoder_config.json"

# ─── PhoBERT Config ────────────────────────────────────────────────────────────
PHOBERT_MODEL_NAME = "vinai/phobert-base"
MAX_SEQ_LEN        = 256   # Sẽ được gợi ý lại bởi EDA (p99 token count)

# ─── Model Architecture Config (QUAN TRỌNG — insight từ ds4v SOTA) ────────────
# Option A: chỉ dùng [CLS] hidden state layer cuối (768 dim)
# Option B: concat 4 hidden layers cuối tại token [CLS] (768×4 = 3072 dim) ← SOTA
# EDA sẽ không thay đổi config này, nhưng training script tuần 2 sẽ đọc nó
ENCODER_OPTIONS = {
    "cls_only":        {"hidden_size": 768,  "num_hidden_layers_concat": 1},
    "concat_4_layers": {"hidden_size": 3072, "num_hidden_layers_concat": 4},
}
DEFAULT_ENCODER = "concat_4_layers"  # Theo SOTA ds4v

# ─── Training Hyperparameters — Cập nhật từ EDA thực tế ───────────────────────
TRAIN_CONFIG = {
    "learning_rate":           2e-5,
    "warmup_ratio":            0.1,       # 10% steps đầu là warmup
    "batch_size":              8,         # giảm từ 16 do seq_len=384, T4 16GB
    "grad_accumulation_steps": 2,         # effective batch = 8×2 = 16
    "max_epochs":              20,
    "early_stop_patience":     3,         # dựa trên Macro-F1 dev set
    "dropout":                 0.2,       # theo ds4v
    "optimizer":               "AdamW",
    "scheduler":               "linear_warmup_decay",
    "seed":                    42,
    "max_seq_len":             384,       # từ encoder_config.json (p99=243 × 1.5)
    "weight_clip":             10.0,      # clip weight neutral=154 → 10.0
    "encoder_option":          "concat_4_layers",
    "max_grad_norm":           1.0,
}
