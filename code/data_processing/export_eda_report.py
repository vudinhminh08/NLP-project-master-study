import os
import sys
import json

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.constants import (
    ASPECT_COLUMNS, RARE_ASPECTS, ENTITY_GROUPS,
    EDA_DIR, CLASS_WEIGHTS_PATH, ENCODER_CONFIG_PATH,
    TRAIN_PATH, DEV_PATH, TEST_PATH,
)

OUTPUT_PATH = "outputs/eda/EDA_Report_ABSA_VLSP2018.docx"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_heading(doc: Document, text: str, level: int):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def add_paragraph(doc: Document, text: str, bold: bool = False, size: int = 11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p


def add_image(doc: Document, img_path: str, width_inches: float = 6.0, caption: str = None):
    if os.path.exists(img_path):
        doc.add_picture(img_path, width=Inches(width_inches))
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(9)
    else:
        doc.add_paragraph(f"[Image not found: {img_path}]")


def styled_table_header(table, header_row: list, bg_hex: str = "2E4057"):
    row = table.rows[0]
    for i, text in enumerate(header_row):
        cell = row.cells[i]
        cell.text = text
        set_cell_bg(cell, bg_hex)
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ─── Main report builder ─────────────────────────────────────────────────────

def build_report():
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df   = pd.read_csv(DEV_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    with open(CLASS_WEIGHTS_PATH, "r", encoding="utf-8") as f:
        class_weights = json.load(f)
    with open(ENCODER_CONFIG_PATH, "r", encoding="utf-8") as f:
        encoder_config = json.load(f)

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ── Trang bìa ────────────────────────────────────────────────────────────
    doc.add_paragraph()
    title = doc.add_heading("BÁO CÁO PHÂN TÍCH DỮ LIỆU (EDA)", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("ABSA VLSP 2018 Hotel — Aspect-Based Sentiment Analysis")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(13)
    sub.runs[0].bold = True

    course = doc.add_paragraph("Môn học: Xử lý Ngôn ngữ Tự nhiên — HUST")
    course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    course.runs[0].font.size = Pt(11)

    doc.add_paragraph()

    # ── 1. Tổng quan dataset ─────────────────────────────────────────────────
    add_heading(doc, "1. Tổng quan Dataset", 1)

    add_paragraph(doc, "Dataset VLSP 2018 Hotel gồm các đánh giá khách sạn tiếng Việt, "
                       "được gán nhãn theo 34 aspect categories với 4 mức cảm xúc: "
                       "Absent (0), Positive (1), Negative (2), Neutral (3).")

    # Bảng tổng quan
    table = doc.add_table(rows=4, cols=5)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    styled_table_header(table, ["Split", "Số mẫu", "Số cột", "Missing", "Duplicate"])

    data_rows = [
        ("Train", str(len(train_df)), "35",
         str(train_df.isnull().sum().sum()),
         str(train_df["Review"].duplicated().sum())),
        ("Dev",   str(len(dev_df)),   "35",
         str(dev_df.isnull().sum().sum()),
         str(dev_df["Review"].duplicated().sum())),
        ("Test",  str(len(test_df)),  "35",
         str(test_df.isnull().sum().sum()),
         str(test_df["Review"].duplicated().sum())),
    ]
    for i, row_data in enumerate(data_rows):
        row = table.rows[i + 1]
        for j, val in enumerate(row_data):
            row.cells[j].text = val
            for para in row.cells[j].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(10)
        bg = "EBF5FB" if i % 2 == 0 else "FDFEFE"
        for j in range(5):
            set_cell_bg(row.cells[j], bg)

    doc.add_paragraph()

    # ── 2. Độ dài review ────────────────────────────────────────────────────
    add_heading(doc, "2. Phân tích Độ dài Review", 1)

    add_paragraph(doc, "Phân tích độ dài review theo số từ và số ký tự giúp xác định "
                       "MAX_SEQ_LEN phù hợp cho PhoBERT tokenizer.")

    # Bảng percentile
    splits_info = [
        ("Train", train_df),
        ("Dev",   dev_df),
        ("Test",  test_df),
    ]

    table2 = doc.add_table(rows=4, cols=8)
    table2.style = "Table Grid"
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER

    styled_table_header(table2,
        ["Split", "p25 (words)", "p50", "p75", "p90", "p95", "p99", "Gợi ý MAX_SEQ_LEN"])

    pcts = [25, 50, 75, 90, 95, 99]
    for i, (name, df) in enumerate(splits_info):
        word_counts = df["Review"].astype(str).str.split().str.len()
        wp = [int(np.percentile(word_counts, p)) for p in pcts]
        p99 = wp[-1]
        import math
        rec = int(math.ceil(p99 * 1.5 / 64) * 64)
        row = table2.rows[i + 1]
        vals = [name] + [str(v) for v in wp] + [str(rec)]
        for j, val in enumerate(vals):
            row.cells[j].text = val
            for para in row.cells[j].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(10)
        bg = "EBF5FB" if i % 2 == 0 else "FDFEFE"
        for j in range(8):
            set_cell_bg(row.cells[j], bg)

    doc.add_paragraph()
    add_paragraph(doc, "Công thức: MAX_SEQ_LEN = p99_words × 1.5, làm tròn lên bội số 64. "
                       "Dựa trên train split → MAX_SEQ_LEN = 384.",
                  bold=False)
    doc.add_paragraph()

    # Biểu đồ độ dài
    for split in ["train", "dev", "test"]:
        img = f"outputs/eda/{split}_review_length.png"
        add_image(doc, img, width_inches=5.8,
                  caption=f"Hình: Phân phối độ dài review — {split.upper()} split")
        doc.add_paragraph()

    # ── 3. Phân phối aspect ─────────────────────────────────────────────────
    add_heading(doc, "3. Phân phối Aspect Presence Rate", 1)

    add_paragraph(doc,
        "Presence rate = tỷ lệ mẫu có nhãn ≠ 0 (có mention aspect đó). "
        "Đường đứt dọc tại 10% là ngưỡng phân biệt aspect phổ biến / hiếm. "
        "Cột màu đỏ = rare aspects.")

    for split in ["train", "dev", "test"]:
        img = f"outputs/eda/{split}_aspect_presence.png"
        add_image(doc, img, width_inches=5.5,
                  caption=f"Hình: Aspect Presence Rate — {split.upper()} split")
        doc.add_paragraph()

    # ── 4. Label breakdown ──────────────────────────────────────────────────
    add_heading(doc, "4. Phân phối Nhãn theo Aspect", 1)

    add_paragraph(doc,
        "Stacked bar chart thể hiện tỷ lệ 4 nhãn (absent/positive/negative/neutral) "
        "cho từng aspect. Majority class là 'absent' (màu xám) — dataset rất mất cân bằng.")

    for split in ["train", "dev", "test"]:
        img = f"outputs/eda/{split}_label_breakdown.png"
        add_image(doc, img, width_inches=5.5,
                  caption=f"Hình: Label Breakdown — {split.upper()} split")
        doc.add_paragraph()

    # ── 5. Class imbalance & Weights ────────────────────────────────────────
    add_heading(doc, "5. Phân tích Class Imbalance & Class Weights", 1)

    add_paragraph(doc,
        "Dataset rất mất cân bằng: class 'absent' chiếm đa số (~70-99% tùy aspect). "
        "Class weights được tính theo công thức SOTA (ds4v IEEE 2022): "
        "weight = majority_count / class_count. "
        "Weights cao = class hiếm → cần penalize mạnh hơn trong loss function.")

    doc.add_paragraph()
    add_paragraph(doc, "Global class weights (train):", bold=True)

    global_w = class_weights["global_weights"]
    label_names = {"0": "Absent", "1": "Positive", "2": "Negative", "3": "Neutral"}

    gtable = doc.add_table(rows=2, cols=4)
    gtable.style = "Table Grid"
    gtable.alignment = WD_TABLE_ALIGNMENT.CENTER
    styled_table_header(gtable, [f"Class {k}: {label_names[k]}" for k in ["0","1","2","3"]])
    row = gtable.rows[1]
    for j, k in enumerate(["0","1","2","3"]):
        row.cells[j].text = f"{global_w[k]:.4f}"
        for para in row.cells[j].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.size = Pt(10)
        set_cell_bg(row.cells[j], "EBF5FB")

    doc.add_paragraph()

    # Top 10 aspects theo class weight (positive)
    add_paragraph(doc, "Top 10 aspects có weight cao nhất cho class Positive (train):", bold=True)

    per_asp = class_weights["per_aspect_weights"]
    w_pos = {asp: float(w.get("1", 0)) for asp, w in per_asp.items()}
    top10 = sorted(w_pos.items(), key=lambda x: x[1], reverse=True)[:10]

    wtable = doc.add_table(rows=11, cols=4)
    wtable.style = "Table Grid"
    wtable.alignment = WD_TABLE_ALIGNMENT.CENTER
    styled_table_header(wtable, ["Aspect", "n_absent", "n_positive", "w_positive"])

    for i, (asp, w) in enumerate(top10):
        row = wtable.rows[i + 1]
        n_absent  = int(train_df[asp].value_counts().get(0, 0))
        n_pos     = int(train_df[asp].value_counts().get(1, 0))
        row.cells[0].text = asp
        row.cells[1].text = str(n_absent)
        row.cells[2].text = str(n_pos)
        row.cells[3].text = f"{w:.2f}"
        bg = "EBF5FB" if i % 2 == 0 else "FDFEFE"
        for j in range(4):
            set_cell_bg(row.cells[j], bg)
            for para in row.cells[j].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(9)

    doc.add_paragraph()

    # ── 6. Rare aspects ──────────────────────────────────────────────────────
    add_heading(doc, "6. Rare Aspects — Thách thức chính", 1)

    add_paragraph(doc,
        "Một số aspects gần như không xuất hiện trong train set, gây khó khăn nghiêm trọng "
        "cho model học pattern. Cần xử lý bằng weighted loss hoặc oversampling.")

    doc.add_paragraph()

    rtable = doc.add_table(rows=len(ASPECT_COLUMNS) + 1, cols=6)
    rtable.style = "Table Grid"
    rtable.alignment = WD_TABLE_ALIGNMENT.CENTER
    styled_table_header(rtable,
        ["Aspect", "n_absent", "n_positive", "n_negative", "n_neutral", "% Present"])

    sorted_by_pct = sorted(
        ASPECT_COLUMNS,
        key=lambda c: (train_df[c] > 0).sum()
    )
    for i, asp in enumerate(sorted_by_pct):
        counts = train_df[asp].value_counts().to_dict()
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        n2 = counts.get(2, 0)
        n3 = counts.get(3, 0)
        pct = (n1 + n2 + n3) / len(train_df) * 100
        row = rtable.rows[i + 1]
        vals = [asp, str(n0), str(n1), str(n2), str(n3), f"{pct:.1f}%"]
        is_rare = asp in RARE_ASPECTS or (n1 + n2 + n3) < 100
        for j, val in enumerate(vals):
            row.cells[j].text = val
            bg = "FADBD8" if is_rare else ("EBF5FB" if i % 2 == 0 else "FDFEFE")
            set_cell_bg(row.cells[j], bg)
            for para in row.cells[j].paragraphs:
                if j > 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(8)
                    if is_rare and j == 0:
                        run.bold = True

    doc.add_paragraph()
    add_paragraph(doc, "Màu đỏ nhạt = rare aspect (< 100 mẫu có nhãn ≠ 0 trong train).",
                  bold=False)

    # ── 7. Encoder config ───────────────────────────────────────────────────
    add_heading(doc, "7. Cấu hình Encoder được chọn", 1)

    add_paragraph(doc,
        "Dựa trên phân tích ds4v/absa-vlsp-2018 (SOTA IEEE 2022), kiến trúc tốt nhất là "
        "concat 4 hidden layers cuối của PhoBERT tại token [CLS]:")

    doc.add_paragraph()

    etable = doc.add_table(rows=6, cols=2)
    etable.style = "Table Grid"
    etable.alignment = WD_TABLE_ALIGNMENT.CENTER
    styled_table_header(etable, ["Tham số", "Giá trị"])

    enc_rows = [
        ("Encoder option",        encoder_config.get("encoder_option", "concat_4_layers")),
        ("Hidden size",           str(encoder_config.get("encoder_hidden_size", 3072)) + " dim (768 × 4)"),
        ("MAX_SEQ_LEN (gợi ý)",   str(encoder_config.get("recommended_max_seq_len", 384))),
        ("p99 word count (train)", str(encoder_config.get("p99_word_count", 243))),
        ("Ghi chú",               encoder_config.get("note", "")),
    ]
    for i, (k, v) in enumerate(enc_rows):
        row = etable.rows[i + 1]
        row.cells[0].text = k
        row.cells[1].text = v
        bg = "EBF5FB" if i % 2 == 0 else "FDFEFE"
        set_cell_bg(row.cells[0], bg)
        set_cell_bg(row.cells[1], bg)
        for j in range(2):
            for para in row.cells[j].paragraphs:
                for run in para.runs:
                    run.font.size = Pt(10)

    doc.add_paragraph()

    # ── 8. Kết luận ─────────────────────────────────────────────────────────
    add_heading(doc, "8. Kết luận & Định hướng Phase PhoBERT", 1)

    conclusions = [
        ("Mất cân bằng nghiêm trọng",
         "Aspect 'ROOM_AMENITIES#PRICES' có 0 mẫu positive/negative trong train. "
         "→ Dùng weighted CrossEntropyLoss với weights từ class_weights.json."),
        ("MAX_SEQ_LEN = 384",
         "p99 của train là 243 từ, nhân 1.5 factor cho word segmentation → 384 tokens. "
         "→ Cập nhật MAX_SEQ_LEN trong constants.py trước khi train."),
        ("Kiến trúc: concat 4 layers",
         "Theo SOTA ds4v IEEE 2022, concat hidden[-4:] tại CLS → 3072 dim. "
         "→ Ablation với 768 dim để so sánh."),
        ("Word segmentation",
         "VnCoreNLP không available → fallback underthesea. "
         "→ Xem xét cài Java 8+ và VnCoreNLP để đạt F1 tốt hơn ~1-2%."),
        ("RARE_ASPECTS cần cập nhật",
         "Thực tế còn FACILITIES#PRICES, FOOD&DRINKS#MISCELLANEOUS, ROOMS#MISCELLANEOUS "
         "cũng < 100 mẫu. → Cập nhật constants.py."),
    ]

    for title_c, detail in conclusions:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(f"{title_c}: ")
        run.bold = True
        run.font.size = Pt(10)
        run2 = p.add_run(detail)
        run2.font.size = Pt(10)

    doc.add_paragraph()
    add_paragraph(doc,
        "Phase PhoBERT: Fine-tune PhoBERT (vinai/phobert-base) với multi-task learning — "
        "34 classification heads song song, weighted loss, warmup + linear decay scheduling.",
        bold=False)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f"Báo cáo đã lưu tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
