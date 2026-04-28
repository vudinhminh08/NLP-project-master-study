from __future__ import annotations

import html

import pandas as pd
import streamlit as st


VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def html_text(value: object, fallback: str = "") -> str:
    text = str(value or fallback).strip()
    return html.escape(text)


def confidence_to_text(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_prediction_table(predictions: list[dict]) -> None:
    if not predictions:
        st.info("PhoBERT không phát hiện aspect nào trong review này.")
        return

    df = pd.DataFrame(predictions)
    df = df[["aspect", "sentiment", "confidence"]]
    df["confidence"] = df["confidence"].map(lambda x: f"{float(x):.4f}")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "aspect": "Aspect",
            "sentiment": "Sentiment",
            "confidence": "Confidence",
        },
    )


def render_explanation(explanation: dict) -> None:
    validation = explanation.get("validation", {})
    if validation.get("skipped_llm"):
        st.info(explanation.get("overall_summary", "Không có aspect nào để giải thích."))
        return

    if validation.get("parse_fail"):
        st.error("LLM không trả về JSON hợp lệ. Hãy thử chạy lại hoặc kiểm tra API.")
        preview = explanation.get("raw_output_preview")
        if preview:
            st.code(preview)
        return

    items = explanation.get("items", [])
    if items:
        render_aspect_cards(items)

    render_summary_action(
        summary=explanation.get("overall_summary"),
        action=explanation.get("recommended_action"),
    )


def render_aspect_cards(items: list[dict]) -> None:
    cards = "".join(aspect_card_html(item) for item in items)
    section_html = (
        '<div class="llm-section-heading">'
        '<div class="llm-section-title">Giải thích theo từng aspect</div>'
        f'<div class="llm-section-count">{len(items)} aspect</div>'
        "</div>"
        f'<div class="aspect-grid">{cards}</div>'
    )
    st.markdown(section_html, unsafe_allow_html=True)


def aspect_card_html(item: dict) -> str:
    sentiment = str(item.get("sentiment", "") or "").strip()
    sentiment_key = sentiment if sentiment in VALID_SENTIMENTS else "unknown"
    evidence = html_text(item.get("evidence"), "Không có bằng chứng rõ trong câu.")
    explanation = html_text(item.get("explanation"), " ")
    confidence = confidence_to_text(item.get("confidence"))
    evidence_uncertain = bool(item.get("evidence_uncertain"))

    confidence_html = (
        f"<div class='aspect-confidence'>Confidence: {html_text(confidence)}</div>"
        if confidence
        else ""
    )
    footnote_html = (
        "<div class='aspect-footnote'>Bằng chứng chưa chắc chắn.</div>"
        if evidence_uncertain
        else ""
    )

    return (
        '<article class="aspect-card">'
        '<div class="aspect-card-head">'
        f'<div class="aspect-card-title">{html_text(item.get("aspect"))}</div>'
        f'<div class="sentiment-pill sentiment-pill-{sentiment_key}">'
        f'{html_text(sentiment, "unknown")}</div>'
        "</div>"
        '<div class="aspect-label">Bằng chứng</div>'
        f'<div class="aspect-evidence">{evidence}</div>'
        '<div class="aspect-label">Giải thích</div>'
        f'<div class="aspect-explanation">{explanation}</div>'
        f"{confidence_html}"
        f"{footnote_html}"
        "</article>"
    )


def render_summary_action(summary: object, action: object) -> None:
    if not summary and not action:
        return

    summary_text = html_text(summary, "Không có tóm tắt.")
    action_text = html_text(action, "Chưa có hành động ưu tiên. Tiếp tục duy trì các điểm đang được đánh giá tốt.")
    summary_html = (
        '<div class="llm-summary-grid">'
        '<section class="llm-summary-panel">'
        '<div class="llm-summary-title">Tóm tắt</div>'
        f'<div class="llm-summary-text">{summary_text}</div>'
        "</section>"
        '<section class="llm-summary-panel">'
        '<div class="llm-summary-title">Gợi ý hành động</div>'
        f'<div class="llm-summary-text">{action_text}</div>'
        "</section>"
        "</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)
