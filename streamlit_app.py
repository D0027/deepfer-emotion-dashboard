"""
DeepFER — Advanced Streamlit Dashboard
5 panels: Overview & Methodology | Dataset Review | Model Architecture |
          Results & Charts | Live Demo

Run:
    streamlit run streamlit_app.py
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from src import config as C
from src import run_data as RD

# ------------------------------------------------------------------ #
# Page config + global styling ("ultra pro max" theme)
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="DeepFER Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(56,189,248,0.14) 0%, transparent 30%),
        radial-gradient(circle at 85% 25%, rgba(124,58,237,0.16) 0%, transparent 32%),
        radial-gradient(circle at 30% 80%, rgba(56,189,248,0.10) 0%, transparent 30%),
        radial-gradient(circle at 75% 85%, rgba(124,58,237,0.10) 0%, transparent 30%),
        linear-gradient(135deg, #0a0b14 0%, #08090f 50%, #050609 100%);
    background-size: 220% 220%, 220% 220%, 220% 220%, 220% 220%, 100% 100%;
    animation: bgDrift 30s ease-in-out infinite;
}

@keyframes bgDrift {
    0%   { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 0 0; }
    50%  { background-position: 40% 30%, 55% 40%, 35% 65%, 65% 55%, 0 0; }
    100% { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 0 0; }
}


.main .block-container {
    background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 26px 26px;
    animation: gridDrift 45s linear infinite;
}
@keyframes gridDrift {
    from { background-position: 0 0; }
    to   { background-position: 260px 260px; }
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12101f 0%, #0a0b14 100%);
    border-right: 1px solid rgba(140, 110, 255, 0.15);
}

/* ---------- Enhanced Sidebar (Advanced) ---------- */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.55rem 0.8rem;
    border-radius: 12px;
    transition: background 0.2s ease, transform 0.15s ease;
    margin-bottom: 2px;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(56,189,248,0.10));
    transform: translateX(3px);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    border-color: #7c3aed !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(124,58,237,0.25) !important;
    box-shadow: 0 0 8px rgba(124,58,237,0.2);
}

section[data-testid="stSidebar"] code {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(56,189,248,0.12)) !important;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 2px 8px !important;
    border-radius: 6px !important;
}

.sidebar-model-item {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.sidebar-model-item:hover {
    transform: translateX(3px);
    border-color: rgba(74,222,128,0.4);
}

h1, h2, h3 { color: #f4f2ff !important; letter-spacing: -0.01em; }

.hero {
    padding: 2.2rem 2.5rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(56,189,248,0.18));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px -20px rgba(124,58,237,0.55);
    margin-bottom: 1.6rem;
}

.hero {
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(115deg, transparent 40%, rgba(255,255,255,0.06) 50%, transparent 60%);
    animation: heroShimmer 6s ease-in-out infinite;
    pointer-events: none;
}
@keyframes heroShimmer {
    0%   { transform: translateX(-30%); }
    100% { transform: translateX(30%); }
}


.hero h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.3rem; }
.hero p { color: #cfcbe8; font-size: 1.02rem; margin: 0; }

.badge-row { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.badge {
    padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); color: #e5e1ff;
}

.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    transition: transform 0.15s ease, border-color 0.15s ease;
    margin-bottom: 1.2rem;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(124,58,237,0.5); }
.metric-card .label { color: #a9a4cf; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-card .value { color: #ffffff; font-size: 1.9rem; font-weight: 800; margin-top: 0.2rem; }
.metric-card .sub { color: #8f8ab5; font-size: 0.78rem; margin-top: 0.15rem; }

.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

.pipeline-step {
    display: flex; gap: 0.9rem; align-items: flex-start;
    padding: 0.9rem 1rem; border-radius: 14px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.6rem;
}
.pipeline-step .num {
    min-width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg,#7c3aed,#38bdf8); color:white; font-weight:800;
    display:flex; align-items:center; justify-content:center; font-size:0.9rem;
}
.pipeline-step .txt b { color: #f1eefd; }
.pipeline-step .txt span { color: #a9a4cf; font-size: 0.88rem; }

/* ---------- Enhanced Pipeline (Panel 1) ---------- */
.pipeline-wrap {
    position: relative;
    padding-left: 22px;
}
.pipeline-wrap::before {
    content: "";
    position: absolute;
    left: 16px; top: 6px; bottom: 6px;
    width: 2px;
    background: linear-gradient(180deg, #7c3aed, #38bdf8);
    opacity: 0.35;
}
.pipeline-step {
    position: relative;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.pipeline-step:hover {
    transform: translateX(4px);
    border-color: rgba(124,58,237,0.55);
    box-shadow: 0 8px 24px -10px rgba(124,58,237,0.45);
}
.pipeline-step .num {
    box-shadow: 0 0 0 4px rgba(124,58,237,0.12);
}

/* ---------- Enhanced Metric Cards (Panel 1) ---------- */
.metric-card {
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: -40%; right: -40%;
    width: 90px; height: 90px;
    background: radial-gradient(circle, rgba(124,58,237,0.35), transparent 70%);
    border-radius: 50%;
}
.metric-card:hover {
    box-shadow: 0 12px 30px -12px rgba(56,189,248,0.35);
}

/* ---------- Finding Cards (Panel 1) ---------- */
.finding-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
    animation: fadeUp 0.45s ease both;
}
.finding-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56,189,248,0.4);
}
.finding-card .ficon {
    font-size: 1.1rem; margin-right: 0.5rem;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04); border-radius: 12px 12px 0 0; padding: 0.6rem 1.1rem;
    color: #cfcbe8; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.55), rgba(56,189,248,0.35)) !important;
    color: white !important;
}

.footer-note { color:#7c7799; font-size:0.78rem; text-align:center; margin-top:2rem; }

code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ---------- Animations ---------- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0px rgba(124,58,237,0.0); }
    50%      { box-shadow: 0 0 22px rgba(124,58,237,0.35); }
}
@keyframes barGrow {
    from { width: 0%; }
}
@keyframes blinkDot {
    0%, 100% { opacity: 1; } 50% { opacity: 0.35; }
}

.fade-in { animation: fadeUp 0.55s ease both; }
.hero { animation: fadeUp 0.6s ease both, glowPulse 5s ease-in-out infinite; }
.metric-card { animation: fadeUp 0.5s ease both; }
.pipeline-step { animation: fadeUp 0.45s ease both; }

/* ---------- Terminal window ---------- */
.term-window {
    background: #0b0c14;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 18px 45px -18px rgba(0,0,0,0.7);
    margin-bottom: 1.2rem;
    animation: fadeUp 0.5s ease both;
}
.term-titlebar {
    display: flex; align-items: center; gap: 8px;
    padding: 0.55rem 0.9rem;
    background: linear-gradient(180deg, #1c1a2b, #14131f);
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.term-dot { width: 11px; height: 11px; border-radius: 50%; }
.term-dot.red { background: #ff5f57; }
.term-dot.yellow { background: #febc2e; }
.term-dot.green { background: #28c840; }
.term-title {
    margin-left: 8px; color: #9d97c2; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; gap: 6px;
}
.term-live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #4ade80;
    animation: blinkDot 1.4s ease-in-out infinite; display:inline-block;
}
.term-body {
    padding: 1rem 1.2rem; max-height: 380px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.55;
    color: #c9e8c9; white-space: pre-wrap; word-break: break-word;
}
.term-body .tok-acc { color: #7dd3fc; }
.term-body .tok-loss { color: #fca5a5; }
.term-body .tok-best { color: #86efac; font-weight: 600; }
.term-body .tok-lr { color: #fcd34d; }
.term-body::-webkit-scrollbar { width: 8px; }
.term-body::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }


/* ---------- Enhanced Model Architecture (Panel 3) ---------- */
.arch-card {
    position: relative;
    overflow: hidden;
}
.arch-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8);
}
.arch-card h4 {
    display: flex; align-items: center; gap: 0.5rem;
}
.param-chip:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px -12px rgba(124,58,237,0.5);
}
.term-window:hover {
    border-color: rgba(56,189,248,0.3);
}

/* ---------- Summary table box ---------- */
.summary-box {
    background: #0b0c14; border: 1px solid rgba(255,255,255,0.10); border-radius: 14px;
    padding: 1rem 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: #d8d4f0; overflow-x: auto; white-space: pre; margin-bottom: 1rem;
    animation: fadeUp 0.5s ease both;
}

/* ---------- Param stat chips ---------- */
.param-chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 0.8rem 0 1.2rem 0; }
.param-chip {
    flex: 1; min-width: 150px; padding: 0.8rem 1rem; border-radius: 14px;
    background: linear-gradient(145deg, rgba(124,58,237,0.14), rgba(56,189,248,0.08));
    border: 1px solid rgba(255,255,255,0.09);
    animation: fadeUp 0.5s ease both;
}
.param-chip .k { color:#a9a4cf; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;}
.param-chip .v { color:#fff; font-size:1.25rem; font-weight:800; margin-top:2px;}
.param-chip .vsub { color:#8f8ab5; font-size:0.72rem;}

/* ---------- Animated confidence / metric bars ---------- */
.abar-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.55rem; }
.abar-label { width: 90px; font-size:0.82rem; color:#cfcbe8; font-weight:600; text-transform:capitalize; flex-shrink:0; }
.abar-track { flex:1; height: 14px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow:hidden; }
.abar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8);
    animation: barGrow 1.1s cubic-bezier(0.22,1,0.36,1) both;
}
.abar-val { width: 48px; text-align:right; font-size:0.8rem; color:#e5e1ff; font-weight:700; flex-shrink:0; }

/* Classification report table styling */
.cls-table table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.cls-table th {
    text-align:left; color:#a9a4cf; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:0.04em; padding: 0.5rem 0.7rem; border-bottom: 1px solid rgba(255,255,255,0.12);
}
.cls-table td { padding: 0.45rem 0.7rem; color:#e5e1ff; border-bottom: 1px solid rgba(255,255,255,0.05); }
.cls-table tr:hover td { background: rgba(124,58,237,0.08); }

/* ---------- Enhanced Results & Charts (Panel 4) ---------- */
.abar-row {
    transition: transform 0.15s ease;
}
.abar-row:hover {
    transform: translateX(3px);
}
.abar-fill {
    box-shadow: 0 0 12px rgba(56,189,248,0.5);
}
.cls-table {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.verdict-good { text-shadow: 0 0 10px rgba(74,222,128,0.4); }
.verdict-mid  { text-shadow: 0 0 10px rgba(251,191,36,0.4); }
.verdict-bad  { text-shadow: 0 0 10px rgba(248,113,113,0.4); }

.verdict-good { color:#4ade80 !important; font-weight:700; }
.verdict-mid { color:#fbbf24 !important; font-weight:700; }
.verdict-bad { color:#f87171 !important; font-weight:700; }

/* ---------- Dataset gallery ---------- */
.gallery-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 0.5rem; text-align:center;
    transition: transform 0.15s ease, border-color 0.15s ease;
    animation: fadeUp 0.4s ease both;
}
.gallery-card:hover { transform: translateY(-2px); border-color: rgba(124,58,237,0.5); }
.gallery-card img { border-radius: 10px; width: 100%; }
.gallery-card .fname { color:#8f8ab5; font-size:0.65rem; margin-top:0.3rem; word-break:break-all; }

.class-hero {
    display:flex; align-items:center; justify-content:space-between; gap:1rem;
    padding: 1rem 1.3rem; border-radius: 16px; margin-bottom: 1rem;
    background: linear-gradient(120deg, rgba(124,58,237,0.18), rgba(56,189,248,0.08));
    border: 1px solid rgba(255,255,255,0.08); animation: fadeUp 0.5s ease both;
}
.class-hero .name { font-size:1.3rem; font-weight:800; color:#fff; text-transform:capitalize; }
.class-hero .count { color:#a9a4cf; font-size:0.85rem; }

/* ---------- Enhanced Dataset Review (Panel 2) ---------- */
.gallery-card {
    position: relative;
}
.gallery-card img {
    transition: transform 0.25s ease;
}
.gallery-card:hover img {
    transform: scale(1.04);
}
.class-hero {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.class-hero:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px -14px rgba(124,58,237,0.5);
}
.dataset-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.9rem; border-radius: 999px;
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(56,189,248,0.15));
    border: 1px solid rgba(255,255,255,0.1);
    color: #e5e1ff; font-size: 0.82rem; font-weight: 600;
    margin-bottom: 0.8rem;
}

.preload-banner {
    background: linear-gradient(120deg, rgba(74,222,128,0.14), rgba(56,189,248,0.08));
    border: 1px solid rgba(74,222,128,0.35); border-radius: 14px;
    padding: 0.8rem 1.1rem; margin-bottom: 1rem; color:#d7ffe3; font-size:0.88rem;
    animation: fadeUp 0.4s ease both;
}

/* ---------- Enhanced Live Demo (Panel 5) ---------- */
.demo-upload-zone {
    border: 2px dashed rgba(124,58,237,0.35);
    border-radius: 18px;
    padding: 1.5rem;
    background: linear-gradient(145deg, rgba(124,58,237,0.06), rgba(56,189,248,0.03));
    margin-bottom: 1rem;
    text-align: center;
    animation: fadeUp 0.4s ease both;
}
.demo-result-card {
    position: relative;
    overflow: hidden;
    animation: glowPulse 3s ease-in-out infinite;
}
.demo-result-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8, #7c3aed);
    background-size: 200% 100%;
    animation: barGrow 2s linear infinite;
}
.detected-img-wrap {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.3);
    box-shadow: 0 0 30px -8px rgba(56,189,248,0.4);
}
.confidence-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 1rem; border-radius: 999px;
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(56,189,248,0.2));
    border: 1px solid rgba(255,255,255,0.12);
    font-weight: 700; color: #fff; font-size: 0.95rem;
    margin-top: 0.4rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHARTS = C.CHARTS_DIR


def chart_path(name):
    p = os.path.join(CHARTS, name)
    return p if os.path.exists(p) else None


def safe_button(label, **kwargs):
    kwargs.pop("use_container_width", None)
    try:
        return st.button(label, use_container_width=True, **kwargs)
    except TypeError:
        return st.button(label, **kwargs)


def safe_image(img, caption=None, **kwargs):
    """st.image() compatibility shim — older Streamlit versions don't accept
    use_container_width, so fall back gracefully instead of crashing."""
    for kw in ("use_container_width", "use_column_width"):
        try:
            return st.image(img, caption=caption, **{kw: True})
        except TypeError:
            continue
    return st.image(img, caption=caption)


def show_img(name, caption=None):
    p = chart_path(name)
    if p:
        safe_image(Image.open(p), caption=caption)
    else:
        st.info(f"`{name}` not found in outputs/charts/ yet.")


def load_comparison_df():
    for p in [os.path.join(C.OUTPUT_DIR, "model_comparison.csv"), C.COMPARISON_CSV]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def load_class_names():
    if os.path.exists(C.CLASS_NAMES_JSON):
        with open(C.CLASS_NAMES_JSON) as f:
            return json.load(f)
    return C.CLASS_NAMES


import re


def _highlight_log(text):
    """Wrap key tokens in spans for colored terminal syntax highlighting."""
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    escaped = re.sub(r"(accuracy: [\d.]+)", r'<span class="tok-acc">\1</span>', escaped)
    escaped = re.sub(r"(val_accuracy: [\d.]+)", r'<span class="tok-acc">\1</span>', escaped)
    escaped = re.sub(r"(loss: [\d.]+)", r'<span class="tok-loss">\1</span>', escaped)
    escaped = re.sub(r"(learning_rate: [\d.e+-]+)", r'<span class="tok-lr">\1</span>', escaped)
    escaped = re.sub(r"(improved from [^\n,]+|saving model[^\n]*)", r'<span class="tok-best">\1</span>', escaped)
    return escaped


def render_terminal(title, text, live=True, key=""):
    dot = '<span class="term-live-dot"></span> live output' if live else 'log'
    html = f"""<div class="term-window">
      <div class="term-titlebar">
        <span class="term-dot red"></span><span class="term-dot yellow"></span><span class="term-dot green"></span>
        <span class="term-title">{title} &nbsp;·&nbsp; {dot}</span>
      </div>
      <div class="term-body">{_highlight_log(text)}</div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_summary_box(text):
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="summary-box">{escaped}</div>', unsafe_allow_html=True)


def render_param_chips(params):
    st.markdown(f"""<div class="param-chip-row">
        <div class="param-chip"><div class="k">Total Params</div><div class="v">{params['total']}</div><div class="vsub">{params['total_mb']}</div></div>
        <div class="param-chip"><div class="k">Trainable</div><div class="v">{params['trainable']}</div><div class="vsub">{params['trainable_mb']}</div></div>
        <div class="param-chip"><div class="k">Non-Trainable</div><div class="v">{params['non_trainable']}</div><div class="vsub">{params['non_trainable_mb']}</div></div>
    </div>""", unsafe_allow_html=True)


def render_animated_bars(rows_dict):
    """rows_dict: {label: value_0_to_1}"""
    for label, val in rows_dict.items():
        pct = max(0.0, min(1.0, val)) * 100
        st.markdown(f"""<div class="abar-row">
            <div class="abar-label">{label}</div>
            <div class="abar-track"><div class="abar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="abar-val">{pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)


def verdict_class(f1):
    if f1 >= 0.6:
        return "verdict-good"
    if f1 >= 0.35:
        return "verdict-mid"
    return "verdict-bad"


def render_classification_report(rows, overall, macro_avg, weighted_avg, model_name):
    header = "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>"
    body_rows = ""
    for cname, p, r, f1, support in rows:
        vcls = verdict_class(f1)
        body_rows += (f"<tr><td>{cname}</td><td>{p:.2f}</td><td>{r:.2f}</td>"
                      f"<td class='{vcls}'>{f1:.2f}</td><td>{support}</td></tr>")
    body_rows += (f"<tr><td><i>macro avg</i></td><td>{macro_avg[0]:.2f}</td><td>{macro_avg[1]:.2f}</td>"
                  f"<td>{macro_avg[2]:.2f}</td><td>{overall['support']}</td></tr>")
    body_rows += (f"<tr><td><i>weighted avg</i></td><td>{weighted_avg[0]:.2f}</td><td>{weighted_avg[1]:.2f}</td>"
                  f"<td>{weighted_avg[2]:.2f}</td><td>{overall['support']}</td></tr>")
    st.markdown(f'<div class="cls-table"><table>{header}{body_rows}</table></div>', unsafe_allow_html=True)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@st.cache_data(show_spinner=False)
def list_sample_images(train_dir, class_name, n=10):
    """Returns up to n sorted image paths for a class (cached so repeated
    dataset-review renders don't keep hitting the filesystem)."""
    folder = os.path.join(train_dir, class_name)
    if not os.path.isdir(folder):
        return []
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
    return [os.path.join(folder, f) for f in files[:n]]


def build_dataset_zip(data_dir, out_path):
    """Zips the whole data/ folder (train + test) to out_path."""
    import zipfile
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.startswith("."):
                    continue
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, os.path.dirname(data_dir))
                zf.write(full, arcname)
    return out_path


def send_to_live_demo(image_path):
    st.session_state.demo_preload_path = image_path
    st.session_state.pending_nav = "🎥 Live Demo"
    st.rerun()


# ------------------------------------------------------------------ #
# Hero header
# ------------------------------------------------------------------ #
st.markdown("""
<div class="hero">
  <h1>🎭 DeepFER — Facial Emotion Recognition</h1>
  <p>Custom CNN vs. MobileNetV2 Transfer Learning · FER2013 · 7-class emotion classification</p>
  <div class="badge-row">
    <span class="badge">🧠 TensorFlow / Keras</span>
    <span class="badge">📦 TFLite Deployment</span>
    <span class="badge">🎯 7 Emotion Classes</span>
    <span class="badge">⚡ Real-Time Ready</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Sidebar navigation
# ------------------------------------------------------------------ #
st.sidebar.markdown("## 🎭 DeepFER")
st.sidebar.caption("Navigate the project")
NAV_OPTIONS = ["🧭 Overview & Methodology", "📊 Dataset Review", "🏗️ Model Architecture",
               "📈 Results & Charts", "🎥 Live Demo"]
if "nav_panel" not in st.session_state:
    st.session_state.nav_panel = NAV_OPTIONS[0]
# Apply any panel switch requested by a button click (from a previous run),
# BEFORE the radio widget below is instantiated — Streamlit forbids writing
# to a widget-bound session_state key after that widget has already run.
if st.session_state.get("pending_nav"):
    st.session_state.nav_panel = st.session_state.pending_nav
    st.session_state.pending_nav = None
panel = st.sidebar.radio(
    "Panel", NAV_OPTIONS, key="nav_panel", label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("**Models expected in** `models/`")
for f in ["deepfer_custom_cnn_best.keras", "deepfer_mobilenetv2_best.keras",
          "deepfer_deploy_model.tflite", "class_names.json", "model_comparison.csv"]:
    ok = os.path.exists(os.path.join(C.MODELS_DIR, f))
    icon = "✅" if ok else "⬜"
    st.sidebar.markdown(f'<div class="sidebar-model-item">{icon} <code>{f}</code></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# PANEL 1 — Overview & Methodology
# ------------------------------------------------------------------ #
if panel == "🧭 Overview & Methodology":
    col1, col2, col3, col4 = st.columns(4)
    comp = load_comparison_df()
    best_acc = f"{comp['accuracy'].max()*100:.1f}%" if comp is not None else "—"
    n_classes = len(C.CLASS_NAMES)
    for col, label, value, sub in [
        (col1, "Dataset", "FER2013", "48×48 grayscale faces"),
        (col2, "Emotion Classes", str(n_classes), ", ".join(C.CLASS_NAMES[:3]) + "…"),
        (col3, "Best Test Accuracy", best_acc, "across both models"),
        (col4, "Models Trained", "2", "Custom CNN + MobileNetV2"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("### 🧪 Project Pipeline")
    steps = [
        ("1", "Setup & Configuration", "Global config: image size, batch size, epochs, class names, seeds."),
        ("2", "Data Collection & Exploration", "FER2013 (7 classes) — class distribution & sample visualization."),
        ("3", "Preprocessing & Augmentation", "Rescale, class-weighting for imbalance, rotation/zoom/shift/flip/brightness jitter. Two pipelines: 48×48 grayscale (CNN) and 96×96 RGB-replicated (transfer learning)."),
        ("4", "Model 1 — Custom CNN (from scratch)", "Stacked Conv-BN-ReLU blocks with increasing filters, max pooling, dropout, dense classification head."),
        ("5", "Model 2 — Transfer Learning (MobileNetV2)", "ImageNet-pretrained backbone, 2-phase training: frozen head → fine-tune top 30 layers."),
        ("6", "Training", "ModelCheckpoint, EarlyStopping, ReduceLROnPlateau callbacks + class weighting."),
        ("7", "Evaluation", "Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC/AUC — both models, held-out test set."),
        ("8", "Model Comparison", "Side-by-side metric comparison to pick the deployment model."),
        ("9", "Performance Optimization", "Inference latency benchmarking + quantized TFLite export for real-time use."),
        ("10", "Export & Local Deployment", "Exported .keras + .tflite + class_names.json + metrics — used by this local app."),
    ]
    st.markdown('<div class="pipeline-wrap">', unsafe_allow_html=True)
    for num, title, desc in steps:
        st.markdown(f"""<div class="pipeline-step">
            <div class="num">{num}</div>
            <div class="txt"><b>{title}</b><br><span>{desc}</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Key Findings & Challenges")
    findings = [
        ("⚖️", "Class imbalance", '"disgust" is the smallest class in FER2013; addressed via computed class weights, but it typically remains the hardest class (check its row in the confusion matrix).'),
        ("🔍", "Low native resolution (48×48)", "Limits fine detail — augmentation and BatchNorm help generalization."),
        ("⚡", "Transfer learning vs. Custom CNN", "Transfer learning generally converges faster and can generalize better with limited fine-tuning, while the lightweight custom CNN tends to have lower inference latency — compare using the metrics on the Results panel for your specific run."),
        ("📦", "Why MobileNetV2", "Chosen specifically because it's lightweight and fast, directly supporting real-time / performance-optimized deployment."),
    ]
    for icon, title, desc in findings:
        st.markdown(f"""<div class="finding-card">
            <b><span class="ficon">{icon}</span>{title}</b><br>
            <span style="color:#a9a4cf; font-size:0.9rem;">{desc}</span>
        </div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# PANEL 2 — Dataset Review
# ------------------------------------------------------------------ #
elif panel == "📊 Dataset Review":
    st.markdown("### 📊 Dataset Review — FER2013")
    st.markdown("""<div class="panel">
    <b>Dataset:</b> FER2013 — 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise).<br>
    Images are 48×48 grayscale facial crops. Public copies are structured as
    <code>train/&lt;class&gt;/*.jpg</code> and <code>test/&lt;class&gt;/*.jpg</code>, or as a single CSV
    with pixel strings (original Kaggle competition format) — this project supports both.
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Class Distribution", "Sample Images"])

    # ---- TAB 1: Class Distribution only ----
    with tab1:
        show_img("class_distribution.png", "Training set class distribution — note the imbalance across classes.")

    # ---- TAB 2: Sample Images + full dataset browse/gallery/download ----
    with tab2:
        show_img("sample_images.png", "One representative sample per emotion class.")

        st.markdown("### 🔀 Choose Data Source")
        st.markdown(f'<div class="dataset-badge">📌 Currently browsing: <b>{"Train" if st.session_state.get("dataset_review_source","Train")=="Train" else "Test"}</b> split</div>', unsafe_allow_html=True)
        data_source = st.radio(
            "Select which split to browse", ["Train", "Test"],
            horizontal=True, key="dataset_review_source",
        )
        active_dir = C.TRAIN_DIR if data_source == "Train" else C.TEST_DIR
        active_dir_label = "train" if data_source == "Train" else "test"

        has_local_data = os.path.isdir(active_dir) and any(os.scandir(active_dir))

        if has_local_data:
            st.markdown(f"### 📁 Local dataset status — `{active_dir_label}`")
            rows = []
            for cname in C.CLASS_NAMES:
                folder = os.path.join(active_dir, cname)
                n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
                rows.append({"class": cname, f"{active_dir_label}_images": n})
            df = pd.DataFrame(rows)
            fig = px.bar(df, x="class", y=f"{active_dir_label}_images", color="class",
                         title=f"Live count from your local data/{active_dir_label}/ folder", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # ---------------------------------------------------------- #
            # Sample gallery — 10 images per class, click-to-test
            # ---------------------------------------------------------- #
            st.markdown(f"### 🖼️ Browse the Dataset ({data_source}) — 10 samples per class")
            st.caption("Pick any image and send it straight to the Live Demo panel for inference.")

            class_tabs = st.tabs([c.capitalize() for c in C.CLASS_NAMES])
            for cname, ctab in zip(C.CLASS_NAMES, class_tabs):
                with ctab:
                    total_in_class = len(os.listdir(os.path.join(active_dir, cname))) \
                        if os.path.isdir(os.path.join(active_dir, cname)) else 0
                    st.markdown(f"""<div class="class-hero">
                        <div class="name">🎭 {cname}</div>
                        <div class="count">{total_in_class:,} images in data/{active_dir_label}/{cname}/ · showing 10</div>
                    </div>""", unsafe_allow_html=True)

                    samples = list_sample_images(active_dir, cname, n=10)
                    if not samples:
                        st.info(f"No images found in `data/{active_dir_label}/{cname}/`.")
                        continue

                    cols = st.columns(5)
                    for i, img_path in enumerate(samples):
                        with cols[i % 5]:
                            try:
                                thumb = Image.open(img_path)
                            except Exception:
                                continue
                            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                            safe_image(thumb)
                            st.markdown(f'<div class="fname">{os.path.basename(img_path)}</div></div>',
                                        unsafe_allow_html=True)
                            if safe_button("🎥 Test in Live Demo", key=f"test_{active_dir_label}_{cname}_{i}"):
                                send_to_live_demo(img_path)

            # ---------------------------------------------------------- #
            # Full dataset download
            # ---------------------------------------------------------- #
            st.markdown("### 📦 Download the Full Dataset")
            st.markdown("""<div class="panel">
            Package everything in <code>data/train/</code> and <code>data/test/</code> into a single zip
            you can back up or move to another machine.
            </div>""", unsafe_allow_html=True)

            zip_path = os.path.join(C.OUTPUT_DIR, "deepfer_full_dataset.zip")
            col1, col2 = st.columns([1, 2])
            with col1:
                if safe_button("🗜️ Prepare Dataset Zip"):
                    with st.spinner("Zipping data/train/ and data/test/ — this can take a while for the full FER2013 set..."):
                        build_dataset_zip(C.DATA_DIR, zip_path)
                    st.session_state.dataset_zip_ready = True

            if st.session_state.get("dataset_zip_ready") or os.path.exists(zip_path):
                if os.path.exists(zip_path):
                    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                    with col2:
                        with open(zip_path, "rb") as f:
                            data_bytes = f.read()
                        try:
                            st.download_button(
                                f"⬇️ Download deepfer_full_dataset.zip ({size_mb:.1f} MB)",
                                data=data_bytes, file_name="deepfer_full_dataset.zip",
                                mime="application/zip", use_container_width=True,
                            )
                        except TypeError:
                            st.download_button(
                                f"⬇️ Download deepfer_full_dataset.zip ({size_mb:.1f} MB)",
                                data=data_bytes, file_name="deepfer_full_dataset.zip",
                                mime="application/zip",
                            )
        else:
            st.info(f"No local `data/{active_dir_label}/<class>/` images detected — showing the pre-saved chart from the "
                     "original Kaggle run above. Add the FER2013 dataset to `data/train/` and `data/test/` "
                     "to unlock the per-class sample gallery, live counts, and dataset download below.")

# ------------------------------------------------------------------ #
# PANEL 3 — Model Architecture
# ------------------------------------------------------------------ #
elif panel == "🏗️ Model Architecture":
    st.markdown("### 🏗️ Model Architecture")

    arch_tabs = st.tabs(["📐 Architecture", "🧬 Model Summary", "📟 Training Logs", "⚙️ Config & Optimization"])

    # ---- Architecture overview ----
    with arch_tabs[0]:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("""<div class="panel fade-in arch-card">
            <h4>🧩 Model 1 — Custom CNN (from scratch)</h4>
            <p>Input: 48×48×1 grayscale</p>

```
Conv2D(64)  → BN → ReLU   ×2  → MaxPool → Dropout(0.25)
Conv2D(128) → BN → ReLU   ×2  → MaxPool → Dropout(0.25)
Conv2D(256) → BN → ReLU   ×2  → MaxPool → Dropout(0.30)
Flatten → Dense(256) → BN → ReLU → Dropout(0.50)
Dense(7, softmax)
```
<p><b>Optimizer:</b> Adam (lr=1e-3) &nbsp;|&nbsp; <b>Loss:</b> categorical cross-entropy</p>
</div>""", unsafe_allow_html=True)

        with colB:
            st.markdown("""<div class="panel fade-in arch-card">
            <h4>🚀 Model 2 — MobileNetV2 (Transfer Learning)</h4>
            <p>Input: 96×96×3 (upsampled + channel-replicated)</p>

```
MobileNetV2 (ImageNet weights, base frozen initially)
→ GlobalAveragePooling2D
→ Dense(256) → BN → ReLU → Dropout(0.40)
→ Dense(7, softmax)
```
<p><b>Phase 1:</b> train head only (base frozen), Adam 1e-3<br>
<b>Phase 2:</b> unfreeze last 30 layers, fine-tune at Adam 1e-5</p>
</div>""", unsafe_allow_html=True)

    # ---- Real model.summary() outputs ----
    with arch_tabs[1]:
        st.markdown("#### Custom CNN — `model.summary()`")
        render_param_chips(RD.CNN_PARAMS)
        with st.expander("Show full layer-by-layer summary", expanded=False):
            render_summary_box(RD.CNN_SUMMARY_TEXT)

        st.markdown("#### MobileNetV2 Transfer Learning — `model.summary()`")
        render_param_chips(RD.TL_PARAMS)
        with st.expander("Show full layer-by-layer summary", expanded=False):
            render_summary_box(RD.TL_SUMMARY_TEXT)

        st.caption("💡 Notice MobileNetV2 has 2.6M total params but only ~330K trainable — "
                   "the ImageNet backbone (2.26M params) stays frozen except during Phase-2 fine-tuning.")

    # ---- Real training logs, terminal style ----
    with arch_tabs[2]:
        st.markdown("#### Live training run — Custom CNN (40 epochs)")
        render_terminal("bash · custom_cnn_training.log", RD.CNN_TRAINING_LOG)
        st.caption("⏱️ Total training time: **44.0 min** · Best val_accuracy: **64.00%** (epoch 40, restored from epoch 39)")

        st.markdown("#### Live training run — MobileNetV2 (Phase 1, head-only)")
        render_terminal("bash · mobilenetv2_phase1_training.log", RD.TL_TRAINING_LOG)
        st.caption("⏱️ Best val_accuracy this phase: **44.36%** (epoch 7) · learning rate reduced via ReduceLROnPlateau at epoch 10")

        st.markdown("#### Live training run — MobileNetV2 (Phase 2, fine-tuning top 30 layers)")
        render_terminal("bash · mobilenetv2_phase2_finetune.log", RD.TL_TRAINING_LOG_PHASE2)
        st.warning("⚠️ **Finding:** Fine-tuning made things worse here. Val accuracy never surpassed Phase 1's "
                   "44.36% best, and early stopping (monitoring val_loss) restored weights from epoch 1 of "
                   "Phase 2 — a val_accuracy of only 31.54%, well below Phase 1's peak. This points to the "
                   "fine-tuning learning rate being too aggressive for such a small unfrozen layer count on "
                   "48×48-native, low-detail FER2013 images — classic catastrophic-forgetting territory. "
                   "The exported/best-saved model likely reverts to the **Phase 1 checkpoint** for actual use.")

    # ---- Config + optimization ----
    with arch_tabs[3]:
        st.markdown("### ⚙️ Training Configuration")
        cfg_df = pd.DataFrame({
            "Parameter": ["Image size (CNN)", "Image size (Transfer Learning)", "Batch size",
                          "Epochs (CNN)", "Epochs (TL, per phase)", "Classes", "Class weighting",
                          "Augmentation"],
            "Value": [f"{C.IMG_SIZE}×{C.IMG_SIZE} grayscale", f"{C.IMG_SIZE_TL}×{C.IMG_SIZE_TL} RGB",
                      str(C.BATCH_SIZE), str(C.EPOCHS_CNN), str(C.EPOCHS_TL), str(C.NUM_CLASSES),
                      "Balanced (sklearn compute_class_weight)",
                      "Rotation, zoom, shift, h-flip, brightness jitter"]
        })
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)

        st.markdown("### 🚀 Performance Optimization for Deployment")
        st.markdown("""<div class="panel fade-in">
        Post-training, inference latency was benchmarked for both models and the best/lightest one was
        exported as a <b>quantized TFLite model</b> (<code>deepfer_deploy_model.tflite</code>) — this is what
        makes real-time, on-device inference feasible in the Live Demo panel and in local webcam apps.
        </div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# PANEL 4 — Results & Charts
# ------------------------------------------------------------------ #
elif panel == "📈 Results & Charts":
    st.markdown("### 📈 Results & Charts")

    comp = load_comparison_df()
    if comp is not None:
        cols = st.columns(len(comp))
        for col, (_, row) in zip(cols, comp.iterrows()):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">{row.get('model','Model')}</div>
                    <div class="value">{row.get('accuracy',0)*100:.1f}%</div>
                    <div class="sub">Accuracy · F1 {row.get('f1',0):.3f}</div></div>""",
                    unsafe_allow_html=True)

    tabs = st.tabs(["Training Curves", "Confusion Matrices", "ROC Curves", "Metric Comparison", "Classification Reports"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            show_img("training_history_cnn.png", "Custom CNN — accuracy & loss vs. epoch")
        with c2:
            show_img("training_history_tl_head.png", "MobileNetV2 — Phase 1 (frozen head)")
        show_img("training_history_tl_finetune.png", "MobileNetV2 — Phase 2 (fine-tuning)")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            show_img("confusion_matrix_cnn.png", "Custom CNN")
        with c2:
            show_img("confusion_matrix_tl.png", "MobileNetV2 Transfer Learning")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            show_img("roc_curves_cnn.png", "Custom CNN — one-vs-rest ROC")
        with c2:
            show_img("roc_curves_tl.png", "MobileNetV2 TL — one-vs-rest ROC")

    with tabs[3]:
        show_img("metric_comparison.png", "Accuracy / Precision / Recall / F1 — side by side")
        if comp is not None:
            st.dataframe(comp, use_container_width=True, hide_index=True)

        st.markdown('<div style="border-left:3px solid #7c3aed; padding-left:0.8rem; margin:1rem 0 0.6rem;"><b>📊 At a glance</b></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Custom CNN**")
            render_animated_bars({
                "accuracy": RD.CNN_OVERALL["accuracy"], "precision": RD.CNN_OVERALL["precision"],
                "recall": RD.CNN_OVERALL["recall"], "f1": RD.CNN_OVERALL["f1"],
            })
        with c2:
            st.markdown("**MobileNetV2 TL**")
            render_animated_bars({
                "accuracy": RD.TL_OVERALL["accuracy"], "precision": RD.TL_OVERALL["precision"],
                "recall": RD.TL_OVERALL["recall"], "f1": RD.TL_OVERALL["f1"],
            })

    with tabs[4]:
        st.markdown("#### Custom CNN — Classification Report")
        st.markdown(f"""<div class="param-chip-row">
            <div class="param-chip"><div class="k">Accuracy</div><div class="v">{RD.CNN_OVERALL['accuracy']*100:.2f}%</div></div>
            <div class="param-chip"><div class="k">Precision</div><div class="v">{RD.CNN_OVERALL['precision']*100:.2f}%</div></div>
            <div class="param-chip"><div class="k">Recall</div><div class="v">{RD.CNN_OVERALL['recall']*100:.2f}%</div></div>
            <div class="param-chip"><div class="k">F1-score</div><div class="v">{RD.CNN_OVERALL['f1']*100:.2f}%</div></div>
        </div>""", unsafe_allow_html=True)
        render_classification_report(RD.CNN_REPORT_ROWS, RD.CNN_OVERALL, RD.CNN_MACRO_AVG, RD.CNN_WEIGHTED_AVG, "Custom CNN")

        st.markdown("#### MobileNetV2 Transfer Learning — Classification Report")
        st.markdown(f"""<div class="param-chip-row">
            <div class="param-chip"><div class="k">Accuracy</div><div class="v">{RD.TL_OVERALL['accuracy']*100:.2f}%</div></div>
            <div class="param-chip"><div class="k">Precision</div><div class="v">{RD.TL_OVERALL['precision']*100:.2f}%</div></div>
            <div class="param-chip"><div class="k">Recall</div><div class="v">{RD.TL_OVERALL['recall']*100:.2f}%</div></div>
            <div class="param-chip"><div class="k">F1-score</div><div class="v">{RD.TL_OVERALL['f1']*100:.2f}%</div></div>
        </div>""", unsafe_allow_html=True)
        render_classification_report(RD.TL_REPORT_ROWS, RD.TL_OVERALL, RD.TL_MACRO_AVG, RD.TL_WEIGHTED_AVG, "MobileNetV2 TL")

        st.info("🔍 **Custom CNN clearly wins here** — MobileNetV2's fine-tuning phase 2 log wasn't "
                "captured, and its head-only phase (shown in Training Logs) hadn't converged as well, "
                "especially on **sad** (2% recall) and **fear** (10% recall). The Custom CNN generalizes "
                "far better across all 7 classes on this run.")

# ------------------------------------------------------------------ #
# PANEL 5 — Live Demo
# ------------------------------------------------------------------ #
elif panel == "🎥 Live Demo":
    st.markdown("### 🎥 Live Emotion Detection Demo")
    st.markdown("""<div class="panel">
    Upload a photo with a visible face — the app detects the face (OpenCV Haar cascade),
    crops it, and classifies the emotion using your trained Custom CNN.
    </div>""", unsafe_allow_html=True)

    cnn_path = C.CNN_BEST_PATH if os.path.exists(C.CNN_BEST_PATH) else C.CNN_FINAL_PATH
    if not os.path.exists(cnn_path):
        st.error("No custom CNN model found in `models/`. Copy your downloaded "
                  "`deepfer_custom_cnn_best.keras` (or `_final.keras`) there first.")
    else:
        preload_path = st.session_state.get("demo_preload_path")
        pil_image = None

        if preload_path and os.path.exists(preload_path):
            rel = os.path.relpath(preload_path, C.TRAIN_DIR) if preload_path.startswith(C.TRAIN_DIR) else preload_path
            st.markdown(f"""<div class="preload-banner">
                🖼️ Loaded from Dataset Review: <b>{rel}</b>
            </div>""", unsafe_allow_html=True)
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if safe_button("✖ Clear & upload my own"):
                    st.session_state.demo_preload_path = None
                    st.rerun()
            pil_image = Image.open(preload_path).convert("RGB")
        else:
            uploaded = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                pil_image = Image.open(uploaded).convert("RGB")

        if pil_image is not None:
            import cv2
            from tensorflow import keras

            @st.cache_resource
            def _load_model(path):
                return keras.models.load_model(path)

            model = _load_model(cnn_path)
            class_names = load_class_names()
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            img_rgb = np.array(pil_image)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

            img_display = img_rgb.copy()
            if len(faces) == 0:
                face_crop = img_gray
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_crop = img_gray[y:y + h, x:x + w]
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (56, 189, 248), 3)

            face_resized = cv2.resize(face_crop, (C.IMG_SIZE, C.IMG_SIZE))
            face_norm = face_resized.astype("float32") / 255.0
            face_input = face_norm.reshape(1, C.IMG_SIZE, C.IMG_SIZE, 1)

            preds = model.predict(face_input, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_class = class_names[pred_idx]
            confidence = float(preds[pred_idx]) * 100

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="detected-img-wrap">', unsafe_allow_html=True)
                safe_image(img_display, caption="Detected face")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card demo-result-card">
                    <div class="label">Predicted Emotion</div>
                    <div class="value">🎭 {pred_class.upper()}</div>
                    <div class="confidence-badge">⚡ {confidence:.1f}% confidence</div></div>""", unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[float(p) * 100 for p in preds], y=class_names, orientation="h",
                    marker=dict(color=[float(p) for p in preds], colorscale="Viridis")
                ))
                fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0, r=0, t=10, b=0),
                                   xaxis_title="Confidence (%)")
                st.plotly_chart(fig, use_container_width=True)

            if preload_path:
                true_label = os.path.basename(os.path.dirname(preload_path))
                if true_label.lower() == pred_class.lower():
                    st.success(f"✅ Correct! This image comes from the **{true_label}** folder and the model predicted **{pred_class}**.")
                else:
                    st.warning(f"❌ This image comes from the **{true_label}** folder but the model predicted **{pred_class}**.")
        else:
            st.info("Upload an image above, or pick one from Dataset Review's sample gallery, to run live inference.")

st.markdown('<div class="footer-note">DeepFER · Custom CNN + MobileNetV2 Transfer Learning on FER2013 · Built for local & upcoming live deployment</div>', unsafe_allow_html=True)