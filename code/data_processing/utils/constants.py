

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

NUM_ASPECTS = len(ASPECT_COLUMNS)
assert NUM_ASPECTS == 34, "Phải có đúng 34 aspects"


LABEL_TO_IDX = {"absent": 0, "positive": 1, "negative": 2, "neutral": 3}
IDX_TO_LABEL = {0: "absent", 1: "positive", 2: "negative", 3: "neutral"}
NUM_LABELS   = 4


ENTITY_GROUPS = {
    "FACILITIES":     [c for c in ASPECT_COLUMNS if c.startswith("FACILITIES")],
    "FOOD&DRINKS":    [c for c in ASPECT_COLUMNS if c.startswith("FOOD&DRINKS")],
    "HOTEL":          [c for c in ASPECT_COLUMNS if c.startswith("HOTEL#")],
    "LOCATION":       [c for c in ASPECT_COLUMNS if c.startswith("LOCATION")],
    "ROOMS":          [c for c in ASPECT_COLUMNS if c.startswith("ROOMS#")],
    "ROOM_AMENITIES": [c for c in ASPECT_COLUMNS if c.startswith("ROOM_AMENITIES")],
    "SERVICE":        [c for c in ASPECT_COLUMNS if c.startswith("SERVICE")],
}



RARE_ASPECTS = [
    "FACILITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",
    "ROOM_AMENITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#CLEANLINESS",
    "ROOM_AMENITIES#DESIGN&FEATURES",
    "HOTEL#DESIGN&FEATURES",
    "ROOMS#MISCELLANEOUS",
    "FOOD&DRINKS#MISCELLANEOUS",
]



WEAK_ASPECTS = [
    "FACILITIES#MISCELLANEOUS",
    "FOOD&DRINKS#MISCELLANEOUS",
    "ROOMS#MISCELLANEOUS",
    "ROOM_AMENITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",
    "HOTEL#MISCELLANEOUS",
    "FACILITIES#GENERAL",
    "FACILITIES#CLEANLINESS",
    "FACILITIES#COMFORT",
]


ZERO_TRAIN_ASPECTS = ["ROOM_AMENITIES#PRICES"]


DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
EDA_DIR     = "outputs/eda"
MODEL_DIR   = "outputs/models"
RESULTS_DIR = "outputs/results"

TRAIN_PATH  = f"{DATA_DIR}/train.csv"
DEV_PATH    = f"{DATA_DIR}/dev.csv"
TEST_PATH   = f"{DATA_DIR}/test.csv"


TRAIN_PREPROCESSED = f"{DATA_DIR}/train_preprocessed.csv"
DEV_PREPROCESSED   = f"{DATA_DIR}/dev_preprocessed.csv"
TEST_PREPROCESSED  = f"{DATA_DIR}/test_preprocessed.csv"


CLASS_WEIGHTS_PATH  = f"{EDA_DIR}/class_weights.json"
ENCODER_CONFIG_PATH = f"{EDA_DIR}/encoder_config.json"


PHOBERT_V1         = "vinai/phobert-base"
PHOBERT_V2         = "vinai/phobert-base-v2"
PHOBERT_MODEL_NAME = PHOBERT_V2
MAX_SEQ_LEN        = 256





ENCODER_OPTIONS = {
    "cls_only":        {"hidden_size": 768,  "num_hidden_layers_concat": 1},
    "concat_4_layers": {"hidden_size": 3072, "num_hidden_layers_concat": 4},
}
DEFAULT_ENCODER = "concat_4_layers"


TRAIN_CONFIG = {
    "learning_rate":           1e-4,
    "warmup_ratio":            0.15,
    "batch_size":              16,
    "grad_accumulation_steps": 1,
    "max_epochs":              20,
    "early_stop_patience":     7,
    "dropout":                 0.2,
    "optimizer":               "Adam",
    "scheduler":               "cosine_warmup",
    "seed":                    42,
    "max_seq_len":             256,
    "head_lr_mult":            5.0,
    "weight_clip":             10.0,
    "encoder_option":          "cls_only",
    "max_grad_norm":           1.0,
}
