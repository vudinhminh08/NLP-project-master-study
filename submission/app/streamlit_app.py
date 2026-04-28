from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

try:
    from .explanation_service import ExplanationService, has_openai_api_key
    from .file_loader import REVIEW_TEXT_COLUMN, ReviewFileError, make_review_options, read_review_file
    from .phobert_service import DEFAULT_CHECKPOINT_PATH, PhoBERTService
    from .styles import CUSTOM_CSS
    from .ui_components import render_explanation, render_prediction_table
except ImportError:
    from explanation_service import ExplanationService, has_openai_api_key
    from file_loader import REVIEW_TEXT_COLUMN, ReviewFileError, make_review_options, read_review_file
    from phobert_service import DEFAULT_CHECKPOINT_PATH, PhoBERTService
    from styles import CUSTOM_CSS
    from ui_components import render_explanation, render_prediction_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VNCORENLP_DIR = PROJECT_ROOT / "vncorenlp"


st.set_page_config(
    page_title="Hotel Review Insight Assistant",
    page_icon="",
    layout="wide",
)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang load PhoBERT checkpoint...")
def load_phobert_service(checkpoint_path: str, vncorenlp_dir: str) -> PhoBERTService:
    return PhoBERTService(
        checkpoint_path=checkpoint_path,
        vncorenlp_dir=vncorenlp_dir,
    )


def init_settings() -> None:
    st.session_state.setdefault("screen", "start")
    st.session_state.setdefault("checkpoint_path", str(DEFAULT_CHECKPOINT_PATH))
    st.session_state.setdefault("vncorenlp_dir", str(DEFAULT_VNCORENLP_DIR))
    st.session_state.setdefault("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))


def render_header() -> None:
    st.markdown('<div class="app-title">Hotel Review Insight Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'PhoBERT phát hiện khía cạnh và cảm xúc; LLM giải thích bằng chứng để chủ khách sạn dễ ra quyết định.'
        '</div>',
        unsafe_allow_html=True,
    )


def run_analysis(review: str, phobert: PhoBERTService, api_key: str | None) -> None:
    review = str(review).strip()
    if not review:
        st.warning("Vui lòng nhập hoặc chọn một review hợp lệ.")
        return

    with st.spinner("PhoBERT đang phân tích review..."):
        result = phobert.predict(review)

    st.subheader("Review")
    st.write(result.review)

    st.markdown(
        '<div class="metric-strip">'
        f'<div class="metric-chip">Aspect phát hiện: <b>{len(result.predictions)}</b></div>'
        '<div class="metric-chip">Model: <b>PhoBERT cls_only</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Review sau preprocessing bằng VnCoreNLP"):
        st.code(result.processed_review)

    st.subheader("Kết quả PhoBERT")
    render_prediction_table(result.predictions)

    st.subheader("LLM Explanation")
    if not has_openai_api_key(api_key):
        st.warning("Thiếu OPENAI_API_KEY. App chỉ hiển thị prediction PhoBERT và chưa gọi LLM explanation.")
        return

    with st.spinner("LLM đang giải thích prediction của PhoBERT..."):
        explainer = ExplanationService(api_key=api_key)
        explanation = explainer.explain(result.review, result.predictions)
    render_explanation(explanation)


def get_settings() -> tuple[str, str, str]:
    checkpoint_path = st.session_state["checkpoint_path"]
    vncorenlp_dir = st.session_state["vncorenlp_dir"]
    api_key = st.session_state["openai_api_key"]
    return checkpoint_path, vncorenlp_dir, api_key


def render_start_screen() -> None:
    st.markdown(
        """
        <div class="start-spacer"></div>
        <div class="start-title">Hotel Review Insight Assistant</div>
        <div class="start-subtitle">
            Phân tích review khách sạn bằng PhoBERT và giải thích bằng LLM.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([0.35, 0.30, 0.35])
    with center:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state["screen"] = "config"
            st.rerun()


def render_config_screen() -> None:
    render_header()
    st.subheader("Config")
    st.caption("Điền đủ 3 mục rồi lưu để bắt đầu phân tích review.")

    with st.form("config_form"):
        checkpoint_path = st.text_input("PhoBERT checkpoint", value=st.session_state["checkpoint_path"])
        vncorenlp_dir = st.text_input("VnCoreNLP directory", value=st.session_state["vncorenlp_dir"])
        api_key = st.text_input("OpenAI API key", value=st.session_state["openai_api_key"], type="password")
        submitted = st.form_submit_button("Lưu và tiếp tục", type="primary")

    if submitted:
        checkpoint_path = checkpoint_path.strip()
        vncorenlp_dir = vncorenlp_dir.strip()
        api_key = api_key.strip()
        if not checkpoint_path or not vncorenlp_dir or not api_key:
            st.error("Cần điền đủ PhoBERT checkpoint, VnCoreNLP directory và OpenAI API key.")
            return
        st.session_state["checkpoint_path"] = checkpoint_path
        st.session_state["vncorenlp_dir"] = vncorenlp_dir
        st.session_state["openai_api_key"] = api_key
        st.session_state["screen"] = "review"
        st.rerun()


def render_file_input() -> str | None:
    st.markdown("File import phải có cột `review_text`. Các cột khác sẽ bị bỏ qua.")
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        return None
    try:
        df = read_review_file(uploaded_file)
    except ReviewFileError as exc:
        st.error(str(exc))
        st.code('review_text\n"Phòng sạch, nhân viên thân thiện nhưng bữa sáng hơi ít món."')
        return None

    st.success(f"Đã đọc {len(df)} review hợp lệ.")
    selected = st.selectbox(
        "Chọn review để phân tích",
        options=make_review_options(df),
        format_func=lambda item: item[1],
    )
    review = df.loc[selected[0], REVIEW_TEXT_COLUMN]
    st.text_area("Review đã chọn", value=review, height=110, disabled=True)
    return review


def render_review_screen() -> None:
    render_header()
    if st.button("Sửa config"):
        st.session_state["screen"] = "config"
        st.rerun()

    checkpoint_path, vncorenlp_dir, api_key = get_settings()

    try:
        phobert = load_phobert_service(checkpoint_path, vncorenlp_dir)
    except Exception as exc:
        st.error(
            "App yêu cầu PhoBERT thật và VnCoreNLP thật. "
            "Hãy kiểm tra checkpoint, VnCoreNLP, Java và dependencies."
        )
        st.exception(exc)
        return

    input_mode = st.radio("Chọn nguồn review", ["Nhập một review", "Import CSV/XLSX"], horizontal=True)
    if input_mode == "Nhập một review":
        review = st.text_area(
            "Nhập một review khách sạn",
            height=130,
            placeholder="Ví dụ: Phòng sạch, nhân viên thân thiện nhưng bữa sáng hơi ít món.",
        )
    else:
        review = render_file_input()

    if st.button("Phân tích review", type="primary"):
        run_analysis(review or "", phobert, api_key)


def main() -> None:
    init_settings()
    if st.session_state["screen"] == "start":
        render_start_screen()
    elif st.session_state["screen"] == "config":
        render_config_screen()
    else:
        render_review_screen()


if __name__ == "__main__":
    main()
