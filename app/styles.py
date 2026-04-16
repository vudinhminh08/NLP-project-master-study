CUSTOM_CSS = """
<style>
    #MainMenu,
    footer,
    header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        visibility: hidden;
        height: 0;
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 2rem;
    }
    .start-spacer { height: 20vh; }
    .start-title { text-align: center; font-size: 2.4rem; font-weight: 780; }
    .start-subtitle { text-align: center; color: #4b5563; margin: 0.5rem 0 1.5rem; }
    .stButton > button {
        border-radius: 8px;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #4b5563;
        margin-bottom: 1.25rem;
    }
    .metric-strip {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 0.75rem 0 1rem;
    }
    .metric-chip {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        background: #f9fafb;
        color: #111827;
    }
    .sentiment-positive { color: #047857; font-weight: 700; }
    .sentiment-negative { color: #b91c1c; font-weight: 700; }
    .sentiment-neutral { color: #92400e; font-weight: 700; }

    .llm-section-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin: 1.25rem 0 0.8rem;
    }
    .llm-section-title {
        font-size: 1.35rem;
        line-height: 1.2;
        font-weight: 780;
        color: #111827;
    }
    .llm-section-count {
        border: 1px solid #d1d5db;
        border-radius: 999px;
        color: #4b5563;
        font-size: 0.84rem;
        padding: 0.25rem 0.65rem;
        background: #f9fafb;
        white-space: nowrap;
    }
    .aspect-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.2rem;
    }
    .aspect-card {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.9rem;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        min-height: 210px;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }
    .aspect-card-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .aspect-card-title {
        font-size: 1rem;
        font-weight: 780;
        line-height: 1.15;
        color: #111827;
        word-break: break-word;
    }
    .sentiment-pill {
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 780;
        padding: 0.18rem 0.55rem;
        white-space: nowrap;
    }
    .sentiment-pill-positive {
        background: #dcfce7;
        color: #047857;
    }
    .sentiment-pill-negative {
        background: #fee2e2;
        color: #b91c1c;
    }
    .sentiment-pill-neutral {
        background: #fef3c7;
        color: #92400e;
    }
    .sentiment-pill-unknown {
        background: #e5e7eb;
        color: #374151;
    }
    .aspect-label {
        color: #6b7280;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .aspect-evidence {
        border-left: 3px solid #0f766e;
        background: #f0fdfa;
        border-radius: 6px;
        color: #111827;
        font-size: 0.9rem;
        line-height: 1.45;
        margin-bottom: 0.75rem;
        padding: 0.55rem 0.65rem;
    }
    .aspect-explanation {
        color: #1f2937;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0.55rem;
    }
    .aspect-confidence {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: auto;
        padding-top: 0.35rem;
    }
    .aspect-footnote {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }
    .llm-summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1rem;
    }
    .llm-summary-panel {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem;
        min-height: 130px;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }
    .llm-summary-title {
        color: #111827;
        font-size: 1.15rem;
        font-weight: 780;
        margin-bottom: 0.55rem;
    }
    .llm-summary-text {
        color: #1f2937;
        line-height: 1.55;
    }
    @media (max-width: 760px) {
        .llm-summary-grid {
            grid-template-columns: 1fr;
        }
        .llm-section-heading {
            align-items: flex-start;
            flex-direction: column;
        }
    }
</style>
"""
