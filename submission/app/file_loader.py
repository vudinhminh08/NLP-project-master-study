from __future__ import annotations

from pathlib import Path

import pandas as pd


REVIEW_TEXT_COLUMN = "review_text"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class ReviewFileError(ValueError):
    pass


def read_review_file(uploaded_file) -> pd.DataFrame:
    filename = getattr(uploaded_file, "name", "")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ReviewFileError("Chỉ hỗ trợ file .csv, .xlsx hoặc .xls.")

    try:
        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as exc:
        raise ReviewFileError(f"Không đọc được file: {exc}") from exc

    return validate_review_dataframe(df)


def validate_review_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if REVIEW_TEXT_COLUMN not in df.columns:
        raise ReviewFileError(
            "File phải có cột bắt buộc 'review_text'. "
            "Ví dụ: review_text"
        )

    cleaned = df[[REVIEW_TEXT_COLUMN]].copy()
    cleaned[REVIEW_TEXT_COLUMN] = cleaned[REVIEW_TEXT_COLUMN].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned[REVIEW_TEXT_COLUMN] != ""].reset_index(drop=True)

    if cleaned.empty:
        raise ReviewFileError("Cột 'review_text' không có review hợp lệ.")

    cleaned.insert(0, "row_number", range(1, len(cleaned) + 1))
    return cleaned


def make_review_options(df: pd.DataFrame) -> list[tuple[int, str]]:
    options = []
    for idx, row in df.iterrows():
        text = str(row[REVIEW_TEXT_COLUMN])
        preview = text[:90] + ("..." if len(text) > 90 else "")
        options.append((idx, f"{int(row['row_number'])}. {preview}"))
    return options
