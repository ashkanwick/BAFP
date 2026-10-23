"""
Streamlit dashboard for the BAFP Persian sentiment capstone.
============================================================

Creative, modern Persian-first UI/UX overhaul with interactive visualisations,
deep metrics analysis, real-time single prediction, and batch inference.

Four pages, switched from the sidebar. Live prediction is the landing page:
  1. پیش‌بینی زنده / Live prediction  -- hero prediction card with emoji verdict,
                                         Plotly confidence gauge & probability distribution,
                                         hazm raw-vs-cleaned comparison, and session history.
  2. داشبورد آماری / Dashboard        -- comprehensive corpus analytics, sentiment donut,
                                         interactive word clouds, token length distributions,
                                         confusion matrix heatmap, and training curves.
  3. پیش‌بینی دسته‌ای / Batch prediction -- process multiple comments via multiline paste
                                         or CSV file upload, real-time progress bar,
                                         sentiment distribution, and Excel-ready export.
  4. راهنما و محدودیت‌ها / Help & limits -- architecture walkthrough, pipeline CLI guide,
                                         and explicit technical limitations.

Machine-specific facts this app is written around
-------------------------------------------------
  * Streamlit 1.59 has no native RTL option, so every Persian string is wrapped
    in an RTL-safe markdown block (:func:`rtl`). Text flows RTL while UI chrome
    remains clean, modern, and responsive.
  * NO Persian font is installed on this machine. tahoma.ttf is present as a
    fallback and Vazirmatn is fetched to assets/fonts/ on first use. The font
    is resolved once by :func:`src.visualize.ensure_font`, which this app
    re-exports through a cached wrapper. Plotly draws Persian correctly on its
    own (the browser shapes it), so chart text is passed raw; only word clouds
    need pre-shaping, handled by :func:`src.visualize.build_wordcloud_image`.
  * The model is CPU-only and ~163M parameters, hence :func:`get_predictor` is
    cached -- reloading it per widget interaction would take tens of seconds.
  * `plotly==6.8.0` / `pandas==3.0.3` are pinned through constraints.txt; this
    file strictly adheres to APIs supported in those versions.

Run
---
    python -m streamlit run app.py
"""

from __future__ import annotations

import html
import io
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# Section 1: Paths and Page Configuration
# ==========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "parsbert-sentiment-3class"
CLEANED_CSV = DATA_DIR / "cleaned_comments.csv"
WIDE_CLEANED_CSV = DATA_DIR / "cleaned_comments_wide.csv"
NEWS_CSV = DATA_DIR / "raw_news_comments.csv"

# First Streamlit call MUST be set_page_config
st.set_page_config(
    page_title="تحلیل احساسات فارسی | Persian Sentiment Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Section 2: Complete Modern CSS Theme (~100 lines)
# ==========================================================================
def _inject_theme() -> None:
    """Inject modern glassmorphism, animated gradients, and RTL styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, sans-serif; }
        .fa-block { direction: rtl; text-align: right; font-size: 1.02rem; line-height: 1.95; font-family: 'Vazirmatn', Tahoma, sans-serif; }
        .fa-block .muted { color: #64748B; font-size: 0.9rem; }
        .stMarkdown [dir="rtl"] { text-align: right; direction: rtl; }
        div[data-baseweb="textarea"], textarea {
            direction: rtl; text-align: right; font-family: 'Vazirmatn', Tahoma, sans-serif !important;
            font-size: 1rem !important; line-height: 1.8 !important; border-radius: 14px !important;
            border: 1px solid #CBD5E1 !important; background: rgba(255, 255, 255, 0.95) !important;
            transition: all 0.25s ease !important;
        }
        div[data-baseweb="textarea"]:focus-within, textarea:focus { border-color: #4F46E5 !important; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.16) !important; }
        h1, h2, h3, h4 { letter-spacing: -0.02em; font-weight: 700; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 4px; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
        @keyframes gradientFlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
        .gradient-title {
            background: linear-gradient(135deg, #1E1B4B 0%, #4F46E5 35%, #0891B2 70%, #059669 100%);
            background-size: 200% 200%; animation: gradientFlow 10s ease infinite;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.15rem; font-weight: 800; line-height: 1.3; margin-bottom: 0.3rem;
        }
        .sidebar-brand {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
            background-size: 200% 200%; animation: gradientFlow 6s ease infinite;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 1.85rem; font-weight: 900; letter-spacing: 0.5px;
        }
        .glass-card {
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            background: rgba(255, 255, 255, 0.82); border: 1px solid rgba(226, 232, 240, 0.85);
            border-radius: 16px; padding: 1.15rem 1.35rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.05), 0 8px 10px -6px rgba(15, 23, 42, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .glass-card:hover { box-shadow: 0 14px 30px -4px rgba(15, 23, 42, 0.08); }
        .metric-card {
            background: rgba(255, 255, 255, 0.88); backdrop-filter: blur(10px);
            border: 1px solid #E2E8F0; border-radius: 14px; padding: 1rem 1.15rem;
            box-shadow: 0 4px 15px -2px rgba(15, 23, 42, 0.04); transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px -3px rgba(15, 23, 42, 0.08); }
        .metric-label { font-size: 0.82rem; color: #64748B; font-weight: 600; }
        .metric-value { font-size: 1.75rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0; line-height: 1.25; }
        .metric-sub { font-size: 0.76rem; color: #94A3B8; }
        .chip {
            display: inline-flex; align-items: center; gap: 4px; padding: 0.26rem 0.72rem;
            border-radius: 9999px; background: #F1F5F9; color: #334155; font-size: 0.82rem;
            font-weight: 500; border: 1px solid #E2E8F0; margin: 0 0.25rem 0.35rem 0;
        }
        .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600; }
        .verdict-hero { direction: rtl; text-align: right; border-radius: 16px; padding: 1.25rem 1.5rem; border-width: 1px; border-style: solid; }
        .verdict-label { font-size: 2.5rem; font-weight: 800; line-height: 1.2; }
        .animated-bar-track { height: 9px; border-radius: 999px; background: rgba(226, 232, 240, 0.85); margin-top: 0.85rem; overflow: hidden; }
        .animated-bar-fill { height: 100%; border-radius: 999px; transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1); }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important; color: #ffffff !important;
            border: none !important; border-radius: 10px !important; padding: 0.55rem 1.4rem !important;
            font-weight: 600 !important; font-size: 0.96rem !important;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35) !important; transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.45) !important; }
        div.stButton > button[kind="secondary"] {
            border-radius: 10px !important; border: 1px solid #CBD5E1 !important; background: #FFFFFF !important;
            color: #334155 !important; transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="secondary"]:hover { border-color: #94A3B8 !important; background: #F8FAFC !important; }
        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #E2E8F0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Keep _inject_css alias for backward compatibility
_inject_css = _inject_theme


# ==========================================================================
# Section 3: RTL and Safety Helpers
# ==========================================================================
def fa(text: str) -> str:
    """Escape user-supplied text before it enters an RTL HTML block.

    The Persian panels are rendered with ``unsafe_allow_html=True``; without this
    a pasted comment containing ``<img onerror=...>`` would execute.
    """
    return html.escape(str(text), quote=False)


def ltr(text: str) -> str:
    """Isolate an LTR run (a date, a filename) inside RTL prose.

    Without the isolates, bidi reordering splits ``2026-09-28 02:54`` around the
    colon and the caption reads as two fragments.
    """
    return f"\u2066{text}\u2069"


def rtl(text: str, muted: bool = False) -> None:
    """Render one RTL-safe Persian markdown block (escaping handled by markdown)."""
    cls = "fa-block muted" if muted else "fa-block"
    st.markdown(f'<div class="{cls}" dir="rtl">{text}</div>', unsafe_allow_html=True)


# ==========================================================================
# Section 4: Font Helpers and Visualization Imports
# ==========================================================================
from src.visualize import (  # noqa: E402
    LABEL_COLORS,
    LABEL_FA,
    LABEL_ORDER,
    build_wordcloud_image,
    ensure_font as _ensure_font,
    top_words,
    word_cloud_text,
)


@st.cache_data(show_spinner="Downloading Persian font...")
def ensure_font() -> tuple[str | None, str]:
    """Streamlit-cached wrapper around :func:`src.visualize.ensure_font`."""
    return _ensure_font()


# ==========================================================================
# Section 5: Data Loading and Predictor
# ==========================================================================
@st.cache_data(show_spinner="Loading corpus...")
def load_cleaned(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "label_name" in frame:
        frame["label_name"] = frame["label_name"].astype("string")
    return frame


@st.cache_data(show_spinner="Loading news corpus...")
def load_news(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


@st.cache_resource(show_spinner="در حال فراخوانی و بارگذاری مدل ParsBERT از حافظه محلی...")
def get_predictor(model_dir: str) -> tuple[Any, str]:
    """Load one fine-tuned predictor once per session, keyed by directory.

    Returns ``(predictor | None, message)`` so a missing checkpoint degrades to a
    clear dashboard message instead of a stack trace.
    """
    directory = Path(model_dir)
    if not directory.exists() or not (directory / "config.json").exists():
        return None, (
            f"No checkpoint in {directory.name}. Run: "
            "python -m src.train --csv data/cleaned_comments.csv"
        )
    try:
        from src.train import SentimentPredictor

        return SentimentPredictor(directory), ""
    except Exception as exc:  # noqa: BLE001 - surface it in the UI
        return None, f"Model load failed: {type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False, ttl=60)
def available_models() -> list[Path]:
    """Every saved checkpoint under models/ that actually has weights."""
    return sorted(
        path for path in MODELS_DIR.glob("parsbert-sentiment-*")
        if (path / "config.json").exists()
    )


@st.cache_data(show_spinner=False, ttl=300)
def load_metrics(model_dir: Path) -> dict[str, Any] | None:
    path = Path(model_dir) / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ==========================================================================
# Section 6: Reusable UI Components
# ==========================================================================
def gradient_header(title: str, subtitle: str = "", badge: str = "") -> None:
    """Render a gradient animated Persian header with optional badge and subtitle."""
    badge_html = f'<span class="chip" style="background:#EEF2FF; color:#4F46E5; border-color:#C7D2FE;">{fa(badge)}</span>' if badge else ""
    st.markdown(
        f'<div style="margin-bottom: 1.15rem; direction: rtl; text-align: right;">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">'
        f'<h1 class="gradient-title" style="margin: 0; padding: 0;">{fa(title)}</h1>{badge_html}</div>'
        f'{"<div class=\"fa-block muted\" style=\"margin-top: 0.35rem;\">" + fa(subtitle) + "</div>" if subtitle else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, subtext: str = "", border_color: str = "#4F46E5", icon: str = "") -> str:
    """Generate HTML for a modern glassmorphic metric card with colored top border."""
    icon_html = f'<span style="font-size: 1.25rem;">{icon}</span>' if icon else ""
    return (
        f'<div class="metric-card" style="border-top: 4px solid {border_color}; direction: rtl; text-align: right;">'
        f'<div style="display: flex; align-items: center; justify-content: space-between;">'
        f'<span class="metric-label">{fa(label)}</span>{icon_html}</div>'
        f'<div class="metric-value">{fa(value)}</div><div class="metric-sub">{fa(subtext)}</div></div>'
    )


def sentiment_badge(label: str) -> str:
    """Return a styled HTML badge corresponding to the sentiment tone."""
    color = LABEL_COLORS.get(label, "#64748B")
    fa_name = LABEL_FA.get(label, label)
    emoji = _EMOJI_MAP.get(label, "•")
    return f'<span class="badge" style="background: {color}18; color: {color}; border: 1px solid {color}44;">{emoji} {fa(fa_name)}</span>'


# ==========================================================================
# Section 7: Plotly Charts
# ==========================================================================
def _probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """Horizontal per-class probabilities, already ordered by :meth:`predict`.

    Preserves exact contract expected by test harness (horizontal bar, 3 classes, padded range).
    """
    names = [LABEL_FA.get(name, name) for name in probabilities]
    values = list(probabilities.values())
    max_val = max(values) if values else 1.0
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": [LABEL_COLORS.get(name, "#888888") for name in probabilities], "line": {"width": 0}},
            text=[f"{value * 100:.1f}%" for value in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=215,
        margin={"l": 10, "r": 50, "t": 10, "b": 10},
        xaxis={"range": [0, max(max_val * 1.35, 0.75)], "visible": False},
        yaxis={"autorange": "reversed", "tickfont": {"size": 13, "family": "Vazirmatn, Tahoma, sans-serif"}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Vazirmatn, Tahoma, sans-serif"},
    )
    return figure


def gauge_confidence_chart(confidence: float, label: str) -> go.Figure:
    """Circular gauge indicator for prediction confidence with gradient steps."""
    color = LABEL_COLORS.get(label, "#4F46E5")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            number={"suffix": "%", "font": {"size": 32, "color": color, "family": "Segoe UI, Tahoma, sans-serif"}},
            title={"text": "درجه اطمینان مدل", "font": {"size": 14, "color": "#64748B", "family": "Vazirmatn, Tahoma, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(241, 245, 249, 0.8)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(226, 232, 240, 0.5)"},
                    {"range": [50, 80], "color": "rgba(203, 213, 225, 0.5)"},
                    {"range": [80, 100], "color": "rgba(148, 163, 184, 0.3)"},
                ],
            },
        )
    )
    fig.update_layout(height=215, margin={"l": 20, "r": 20, "t": 35, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def sentiment_pie(frame: pd.DataFrame) -> go.Figure:
    """Modern donut chart of sentiment distribution."""
    counts = frame["label_name"].value_counts().reindex(LABEL_ORDER).fillna(0)
    figure = go.Figure(
        go.Pie(
            labels=[f"{LABEL_FA.get(n, n)} ({n})" for n in counts.index],
            values=counts.values.tolist(),
            hole=0.52,
            marker={"colors": [LABEL_COLORS.get(n, "#888888") for n in counts.index], "line": {"color": "#FFFFFF", "width": 2}},
            textinfo="label+percent",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>تعداد: %{value:,}<br>سهم: %{percent}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def top_words_bar(words: Counter[str], limit: int = 25) -> go.Figure:
    """Horizontal bar chart for top Persian keywords."""
    pairs = words.most_common(limit)[::-1]
    figure = go.Figure(
        go.Bar(
            x=[count for _, count in pairs],
            y=[word for word, _ in pairs],
            orientation="h",
            marker={"color": [count for _, count in pairs], "colorscale": [[0, "#818CF8"], [1, "#4338CA"]], "line": {"width": 0}},
            text=[f"{count:,}" for _, count in pairs],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:,} تکرار<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=max(360, limit * 20),
        margin={"l": 10, "r": 50, "t": 15, "b": 15}, xaxis={"title": "تعداد تکرار", "gridcolor": "#F1F5F9"},
        yaxis={"autorange": True, "tickfont": {"size": 11}}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def length_histogram(frame: pd.DataFrame) -> go.Figure:
    """Token length distribution with mean reference line."""
    lengths = pd.to_numeric(frame["n_tokens"], errors="coerce").dropna()
    mean_val = float(lengths.mean()) if not lengths.empty else 0.0
    figure = go.Figure(
        go.Histogram(
            x=lengths, nbinsx=40, marker={"color": "#6366F1", "line": {"color": "#4338CA", "width": 1}},
            hovertemplate="بازهٔ توکن: %{x}<br>تعداد نظرات: %{y}<extra></extra>",
        )
    )
    figure.add_vline(x=mean_val, line_dash="dash", line_color="#EF4444", annotation_text=f"میانگین: {mean_val:.1f}", annotation_position="top right")
    figure.update_layout(
        template="plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20}, xaxis={"title": "تعداد توکن در هر نظر", "gridcolor": "#F1F5F9"},
        yaxis={"title": "فراوانی نظرات", "gridcolor": "#F1F5F9"}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def per_class_bar(metrics: dict[str, Any]) -> go.Figure:
    """Grouped bar chart for precision, recall, and F1 per class."""
    test = metrics.get("test") or metrics.get("best_val") or {}
    per_class = test.get("per_class", {})
    rows = [
        {"class": LABEL_FA.get(name, name), "metric": m, "score": values.get(k, 0.0)}
        for name, values in per_class.items() if isinstance(values, dict)
        for m, k in [("Precision (دقت)", "precision"), ("Recall (یادآوری)", "recall"), ("F1-Score", "f1-score")]
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return go.Figure()
    figure = px.bar(
        frame, x="class", y="score", color="metric", barmode="group", range_y=(0, 1.05),
        color_discrete_sequence=["#4F46E5", "#06B6D4", "#10B981"], template="plotly_white",
    )
    figure.update_layout(
        font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="کلاس احساسی", yaxis_title="امتیاز (۰ تا ۱)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def confusion_heatmap(metrics: dict[str, Any]) -> go.Figure:
    """Confusion matrix heatmap."""
    test = metrics.get("test") or metrics.get("best_val") or {}
    matrix = test.get("confusion_matrix")
    if not matrix:
        return go.Figure()
    labels = test.get("labels", LABEL_ORDER)
    names = [LABEL_FA.get(label, label) for label in labels]
    figure = go.Figure(
        go.Heatmap(
            z=matrix, x=names, y=names, colorscale=[[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#312E81"]],
            text=matrix, texttemplate="<b>%{text}</b>", textfont={"size": 14, "color": "black"},
            hovertemplate="واقعی: %{y}<br>پیش‌بینی: %{x}<br>تعداد: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=340,
        margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="پیش‌بینی مدل (Predicted)",
        yaxis_title="برچسب واقعی (Actual)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def training_history_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Plot training loss and validation Macro-F1 across epochs."""
    if not history:
        return go.Figure()
    epochs = [h.get("epoch", idx + 1) for idx, h in enumerate(history)]
    fig = go.Figure()
    traces = [
        ("Train Loss (خطای آموزش)", [h.get("train_loss", 0.0) for h in history], "#EF4444", 3, "solid"),
        ("Val Macro-F1", [h.get("val_macro_f1", 0.0) for h in history], "#4F46E5", 3, "solid"),
        ("Val Accuracy", [h.get("val_accuracy", 0.0) for h in history], "#10B981", 2, "dot"),
    ]
    for name, vals, col, w, dash in traces:
        fig.add_trace(go.Scatter(x=epochs, y=vals, name=name, mode="lines+markers", line={"color": col, "width": w, "dash": dash}, marker={"size": 6}))
    fig.update_layout(
        template="plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20}, xaxis={"title": "دوره (Epoch)", "tickmode": "linear", "dtick": 1, "gridcolor": "#F1F5F9"},
        yaxis={"title": "مقدار معیار", "gridcolor": "#F1F5F9"}, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==========================================================================
# Section 8: Page — Live Prediction (Hero Experience)
# ==========================================================================
_EMOJI_MAP: dict[str, str] = {"positive": "😊", "neutral": "😐", "negative": "😤"}

EXAMPLES: tuple[tuple[str, str], ...] = (
    ("positive", "کیفیتش عالی بود و کاملاً راضی‌ام، ارسالش هم سریع بود"),
    ("negative", "به‌شدت بد بود و اصلاً ارزش خرید ندارد"),
    ("neutral", "معمولی است؛ نه خیلی خوب نه خیلی بد"),
)


def _verdict_card(label: str, label_fa: str, confidence: float) -> None:
    """Hero verdict card with sentiment emoji, label, confidence, and animated bar."""
    emoji = _EMOJI_MAP.get(label, "🔍")
    colour = LABEL_COLORS.get(label, "#5B6B7F")
    conf_pct = confidence * 100
    st.markdown(
        f'<div class="verdict-hero fade-in" style="border-color: {colour}55; background: linear-gradient(135deg, {colour}14 0%, rgba(255,255,255,0.92) 100%); box-shadow: 0 10px 25px -5px {colour}22;">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">'
        f'<span style="font-size: 3rem; line-height: 1; filter: drop-shadow(0 2px 5px rgba(0,0,0,0.12));">{emoji}</span>'
        f'<span class="badge" style="background: {colour}22; color: {colour}; font-weight: 700; border: 1px solid {colour}44; font-size: 0.85rem;">{fa(label.upper())}</span></div>'
        f'<div class="verdict-label" style="color: {colour};">{fa(label_fa)}</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.8rem; direction: rtl; text-align: right;">'
        f'<span style="color: #475569; font-size: 0.95rem; font-weight: 600;">درجه اطمینان مدل:</span>'
        f'<span style="color: {colour}; font-size: 1.5rem; font-weight: 800; font-family: monospace;">{conf_pct:.1f}%</span></div>'
        f'<div class="animated-bar-track"><div class="animated-bar-fill" style="width: {conf_pct:.1f}%; background: linear-gradient(90deg, {colour} 0%, {colour}dd 100%);"></div></div></div>',
        unsafe_allow_html=True,
    )


def _model_status(model_dir: Path) -> None:
    """Chips describing the selected checkpoint, plus a reload escape hatch."""
    metrics = load_metrics(model_dir)
    meta_path = model_dir / "training_meta.json"
    base_name = "ParsBERT"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_name = Path(str(meta.get("base_model", ""))).name or base_name
        except (json.JSONDecodeError, OSError):
            pass
    chips = [f"چک‌پوینت: {model_dir.name}", f"پایه: {base_name}"]
    if metrics:
        score = (metrics.get("test") or metrics.get("best_val") or {}).get("macro_f1")
        if isinstance(score, (int, float)):
            chips.append(f"macro-F1: {score:.3f}")
        if metrics.get("train_rows"):
            chips.append(f"{metrics['train_rows']:,} نمونهٔ آموزش")

    chips_html = "".join(f'<span class="chip">{fa(chip)}</span>' for chip in chips)
    st.markdown(f'<div style="margin-bottom: 0.5rem; direction: rtl;">{chips_html}</div>', unsafe_allow_html=True)

    if metrics:
        updated = datetime.fromtimestamp((model_dir / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        st.caption(f"اگر آموزشی در جریان باشد این اعداد مربوط به پیش از پایان آن است · آخرین نوشتن metrics.json: {ltr(updated)}")

    if st.button("🔄 بازخوانی مدل", key="reload_model_btn", help="از سرِ نو بارگذاری می‌کند تا وزن‌های تازهٔ آموزش جای وزن‌های کش‌شده بنشیند"):
        get_predictor.clear()
        st.rerun()


def _use_example(example: str) -> None:
    """Callback: fill the text area before rerun."""
    st.session_state["user_text"] = example


def _clear_panel() -> None:
    """Callback: reset text area and active prediction."""
    st.session_state["user_text"] = ""
    st.session_state.pop("prediction", None)


def page_prediction(model_dir: Path) -> None:
    """Hero Live Prediction page with interactive controls, verdict cards, and history."""
    gradient_header(
        "پیش‌بینی زندهٔ احساسات فارسی",
        "متن دلخواه یا نظر مشتری را بنویسید؛ مدل سه‌کلاسهٔ ParsBERT آن را به مثبت / خنثی / منفی "
        "طبقه‌بندی کرده و خط لوله نرمال‌سازی hazm را به تصویر می‌کشد.",
        badge="استنتاج برخط",
    )

    _model_status(model_dir)

    st.markdown('<div style="margin-top: 0.8rem; margin-bottom: 0.3rem; font-weight: 600; color: #475569; direction: rtl; text-align: right;">نمونه‌های آماده برای بررسی سریع:</div>', unsafe_allow_html=True)
    pills = st.columns(len(EXAMPLES))
    for column, (tone, example) in zip(pills, EXAMPLES):
        with column:
            st.button(
                f"{_EMOJI_MAP.get(tone, '')} نمونهٔ {LABEL_FA[tone]}",
                key=f"example_{tone}",
                help=example,
                on_click=_use_example,
                args=(example,),
                use_container_width=True,
            )

    text = st.text_area(
        "متن نظر (فارسی)",
        height=140,
        key="user_text",
        placeholder="مثلاً: کیفیت ساخت دستگاه بسیار بالاست و عملکرد روانی دارد، فقط ارسال با دو روز تأخیر انجام شد.",
    )
    stripped = (text or "").strip()

    char_count = len(stripped)
    word_count = len(stripped.split())
    st.markdown(
        f'<div style="display: flex; justify-content: flex-start; gap: 8px; margin-top: -6px; margin-bottom: 12px; direction: rtl;">'
        f'<span class="chip" style="font-size: 0.78rem;">📝 <b>{char_count:,}</b> نویسه</span>'
        f'<span class="chip" style="font-size: 0.78rem;">🔤 <b>{word_count:,}</b> واژه</span></div>',
        unsafe_allow_html=True,
    )

    action_left, action_right = st.columns([1, 4])
    ran = action_left.button("پیش‌بینی کن", type="primary", disabled=not stripped, use_container_width=True)
    action_right.button("پاک کردن", on_click=_clear_panel, disabled=not stripped and "prediction" not in st.session_state)

    if ran:
        predictor, message = get_predictor(str(model_dir))
        if predictor is None:
            st.error(message)
            return
        started = time.perf_counter()
        result = predictor.predict(stripped, top_k=3)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["input"] = stripped
        st.session_state["prediction"] = result

        if "history" not in st.session_state:
            st.session_state["history"] = []
        hist_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "text": stripped,
            "label": result["label"],
            "label_fa": result["label_fa"],
            "confidence": result["confidence"],
            "elapsed_ms": result["elapsed_ms"],
        }
        st.session_state["history"].insert(0, hist_entry)
        st.session_state["history"] = st.session_state["history"][:10]

    prediction = st.session_state.get("prediction")
    if prediction is None:
        rtl("هنوز متنی پیش‌بینی نشده است. یکی از دکمه‌های نمونهٔ بالا را انتخاب کنید یا متن دلخواه خود را بنویسید.", muted=True)
        return
    if prediction["label"] is None:
        st.warning(prediction["note"] or "متن قابل پردازش نبود.")
        return

    # Visual Presentation of Prediction
    col_verdict, col_charts = st.columns([1.1, 1.4])
    with col_verdict:
        _verdict_card(prediction["label"], prediction["label_fa"], prediction["confidence"])
        st.caption(f"⏱️ زمان استنتاج: {prediction['elapsed_ms']} میلی‌ثانیه بر روی پردازنده مرکزی (CPU) · مدل: {model_dir.name}")

    with col_charts:
        tab_prob, tab_gauge = st.tabs(["📊 توزیع احتمالات کلاس‌ها", "🎯 سنجه اطمینان (Gauge)"])
        with tab_prob:
            st.plotly_chart(_probability_chart(prediction["probabilities"]), width="stretch")
        with tab_gauge:
            st.plotly_chart(gauge_confidence_chart(prediction["confidence"], prediction["label"]), width="stretch")

    # Raw vs Cleaned Comparison
    cleaned = prediction["cleaned"]
    with st.expander("🔍 مقایسه متن خام و متن نرمال‌شده توسط خط‌لوله (hazm)", expanded=True):
        col_raw, col_clean = st.columns(2)
        with col_raw:
            st.markdown(
                f'<div class="glass-card fa-block" dir="rtl" style="min-height: 110px; border-top: 3px solid #64748B;">'
                f'<div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="font-weight: 700; color: #475569;">متن ورودی خام</span>'
                f'<span class="badge" style="background: #F1F5F9; color: #475569;">{len(prediction["input"])} نویسه</span></div>'
                f'<div style="color: #1E293B; word-break: break-word;">{fa(prediction["input"])}</div></div>',
                unsafe_allow_html=True,
            )
        with col_clean:
            st.markdown(
                f'<div class="glass-card fa-block" dir="rtl" style="min-height: 110px; border-top: 3px solid #4F46E5;">'
                f'<div style="display: flex; justify-content: space-between; margin-bottom: 6px;"><span style="font-weight: 700; color: #4338CA;">متن نرمال‌شده (hazm)</span>'
                f'<span class="badge" style="background: #EEF2FF; color: #4338CA;">{len(cleaned)} نویسه</span></div>'
                f'<div style="color: #1E293B; word-break: break-word;">{fa(cleaned)}</div></div>',
                unsafe_allow_html=True,
            )

        remaining_zwnj = cleaned.count(chr(0x200C))
        if remaining_zwnj == 0:
            st.markdown(
                '<div class="glass-card" style="margin-top: 10px; padding: 0.75rem 1rem; border-right: 4px solid #059669; background: #ECFDF5; direction: rtl; text-align: right;">'
                '<span style="color: #059669; font-weight: 700; margin-left: 6px;">✓ وضعیت نیم‌فاصله (ZWNJ):</span>'
                '<span style="color: #065F46; font-size: 0.92rem;">صفر نیم‌فاصله باقی‌مانده است — قاعدهٔ تبدیل نیم‌فاصله به فاصله کامل با موفقیت اجرا شد.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="glass-card" style="margin-top: 10px; padding: 0.75rem 1rem; border-right: 4px solid #DC2626; background: #FEF2F2; direction: rtl; text-align: right;">'
                f'<span style="color: #DC2626; font-weight: 700; margin-left: 6px;">✗ هشدار نیم‌فاصله:</span>'
                f'<span style="color: #991B1B; font-size: 0.92rem;">{remaining_zwnj} مورد نیم‌فاصله در ورودی مدل باقی مانده است.</span></div>',
                unsafe_allow_html=True,
            )

    # Session Prediction History Timeline
    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"🕒 تاریخچه پیش‌بینی‌های این نشست ({len(history)} مورد اخیر)", expanded=False):
            h_left, h_right = st.columns([4, 1])
            with h_right:
                if st.button("پاک کردن تاریخچه", key="clear_history_btn", use_container_width=True):
                    st.session_state["history"] = []
                    st.rerun()
            history_html_parts = []
            for item in history:
                color = LABEL_COLORS.get(item["label"], "#64748B")
                history_html_parts.append(
                    f'<div class="glass-card" style="margin-bottom: 8px; padding: 0.7rem 1rem; border-right: 4px solid {color}; direction: rtl; text-align: right;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
                    f'<span style="font-weight: 700; color: {color}; font-size: 0.95rem;">'
                    f'<span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: {color}; margin-left: 6px;"></span>'
                    f'{fa(item["label_fa"])} ({item["confidence"]*100:.1f}%)</span>'
                    f'<span style="font-size: 0.8rem; color: #94A3B8; font-family: monospace;">{ltr(item["timestamp"])} · {item["elapsed_ms"]}ms</span></div>'
                    f'<div style="font-size: 0.9rem; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{fa(item["text"])}</div></div>'
                )
            st.markdown("".join(history_html_parts), unsafe_allow_html=True)


# ==========================================================================
# Section 9: Page — Dashboard
# ==========================================================================
def page_dashboard(
    frame: pd.DataFrame | None,
    news: pd.DataFrame | None,
    model_dir: Path,
    corpus_name: str,
) -> None:
    """Comprehensive analytical dashboard with corpus stats, model evaluation, and news."""
    gradient_header(
        "داشبورد تحلیلی و آماری احساسات",
        "نمای یکپارچه از پیکرهٔ نظرات فارسی دیجی‌کالا، واژگان پرتکرار، ابر کلمات و ارزیابی جامع "
        "مدل عمیق ParsBERT همراه با ماتریس درهم‌ریختگی و نمودارهای دورهٔ آموزش.",
        badge=f"پیکره: {corpus_name}",
    )

    if frame is None:
        st.warning(
            "هیچ پیکرهٔ پاک‌سازی‌شده‌ای یافت نشد. لطفاً مراحل زیر را اجرا فرمایید:\n\n"
            "```bash\n"
            "python -m src.scraper --source digikala --out data/raw_digikala_comments.csv\n"
            "python -m src.preprocess --in data/raw_digikala_comments.csv --out data/cleaned_comments.csv\n"
            "python -m src.train --csv data/cleaned_comments.csv --epochs 3\n"
            "python -m src.visualize\n"
            "```"
        )
        return

    labelled = frame[frame["label_name"].notna()]
    metrics = load_metrics(model_dir)

    # 5 Modern Top Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(metric_card("کل نظرات", f"{len(frame):,}", "پیکره انتخاب‌شده", "#4F46E5", "💬"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("نظرات برچسب‌دار", f"{len(labelled):,}", "سه کلاس احساسی", "#059669", "🏷️"), unsafe_allow_html=True)
    with m3:
        prod_count = int(pd.to_numeric(frame.get("product_id"), errors="coerce").nunique()) if "product_id" in frame.columns else "n/a"
        st.markdown(metric_card("تنوع کالاها", f"{prod_count:,}" if isinstance(prod_count, int) else prod_count, "دیجی‌کالا", "#0891B2", "📦"), unsafe_allow_html=True)
    with m4:
        mean_tok = float(pd.to_numeric(labelled.get("n_tokens"), errors="coerce").mean()) if not labelled.empty else 0.0
        st.markdown(metric_card("میانگین توکن‌ها", f"{mean_tok:.1f}", "طول هر نظر", "#D97706", "📏"), unsafe_allow_html=True)
    with m5:
        macro_f1 = f"{metrics['test']['macro_f1']:.3f}" if metrics and metrics.get("test") and "macro_f1" in metrics["test"] else "n/a"
        st.markdown(metric_card("Macro-F1 آزمون", macro_f1, "معیار اصلی ارزیابی", "#7C3AED", "🏆"), unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    tab_corpus, tab_model, tab_news = st.tabs(["📊 پیکره و توزیع داده‌ها (Corpus)", "🧠 ارزیابی مدل عمیق (Model)", "📰 نظرات خبری بدون برچسب (News)"])

    with tab_corpus:
        mix_left, mix_right = st.columns([1, 1])
        with mix_left:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">توزیع فراوانی احساسات</div>', unsafe_allow_html=True)
            st.plotly_chart(sentiment_pie(labelled), width="stretch")
        with mix_right:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">توزیع طول نظرات (تعداد واژگان/توکن‌ها)</div>', unsafe_allow_html=True)
            st.plotly_chart(length_histogram(labelled), width="stretch")

        st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)
        scope = st.radio("فیلتر واژگان بر اساس احساس:", ["همه", *LABEL_FA.values()], horizontal=True)
        if scope == "همه":
            subset = labelled
        else:
            english = {v: k for k, v in LABEL_FA.items()}[scope]
            subset = labelled[labelled["label_name"] == english]

        text_column = "eda_text" if "eda_text" in subset.columns else "clean_text"
        words = top_words(subset[text_column].fillna("").astype(str), limit=200)

        chart_left, chart_right = st.columns([1, 1])
        with chart_left:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">پرکاربردترین واژه‌های نظرات</div>', unsafe_allow_html=True)
            st.plotly_chart(top_words_bar(words, limit=20), width="stretch")
        with chart_right:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">ابر کلمات فارسی (Word Cloud)</div>', unsafe_allow_html=True)
            font_path, font_note = ensure_font()
            if words:
                image = build_wordcloud_image(word_cloud_text(words), font_path)
                if image is not None:
                    st.image(image, width="stretch")
            st.caption(f"قلم نمایشی: {font_note}")

        st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 6px;">پیش‌نمایش تعاملی نمونه‌ای از نظرات پیکره</div>', unsafe_allow_html=True)
        preview_cols = [c for c in ["raw_text", "clean_text", "label_name", "rate", "split"] if c in labelled.columns]
        preview = labelled[preview_cols].copy()
        rename_map = {"raw_text": "متن خام", "clean_text": "متن تمیزشده", "label_name": "برچسب احساس", "rate": "امتیاز (ستاره)", "split": "بخش داده"}
        st.dataframe(preview.rename(columns=rename_map).sample(min(300, len(preview)), random_state=42), width="stretch", height=320)

    with tab_model:
        if metrics is None:
            st.info("فایل metrics.json برای این مدل یافت نشد — با اجرای دستور `python -m src.train` مدل را آموزش دهید.")
        else:
            download = metrics.get("download", {})
            st.markdown(
                f'<div class="glass-card" style="margin-bottom: 1rem; border-right: 4px solid #4F46E5; direction: rtl; text-align: right;">'
                f'<div style="font-weight: 700; color: #1E293B; margin-bottom: 4px;">مشخصات و متادیتای آموزش مدل:</div>'
                f'<div style="color: #475569; font-size: 0.92rem; line-height: 1.8;">پایه: <code>{fa(str(metrics.get("base_model", "ParsBERT")))}</code> · '
                f'آموزش روی <b>{metrics.get("train_rows", 0):,}</b> نمونه، اعتبارسنجی <b>{metrics.get("val_rows", 0):,}</b>، آزمون <b>{metrics.get("test_rows", 0):,}</b> · '
                f'مدت زمان کل: <b>{metrics.get("total_minutes", "n/a")}</b> دقیقه روی <code>{fa(str(metrics.get("device", "cpu")))}</code>.<br>'
                f'دانلود مستقیم وزن‌ها: <b>{download.get("mb", 0)}</b> مگابایت در <b>{download.get("seconds", 0)}</b> ثانیه از منبع <code>{fa(str(download.get("source", "direct")))}</code>.</div></div>',
                unsafe_allow_html=True,
            )

            best, test = metrics.get("best_val") or {}, metrics.get("test") or {}
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card("دقت اعتبارسنجی (Val Acc)", f"{best.get('accuracy', float('nan')):.3f}", "بهترین اپوک", "#4F46E5"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Macro-F1 اعتبارسنجی", f"{best.get('macro_f1', float('nan')):.3f}", "معیار توازن کلاس‌ها", "#0891B2"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("دقت آزمون (Test Acc)", f"{test.get('accuracy', float('nan')):.3f}", "روی داده ارزیابی نهایی", "#059669"), unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("Macro-F1 آزمون نهایی", f"{test.get('macro_f1', float('nan')):.3f}", "معیار قطعی مدل", "#7C3AED"), unsafe_allow_html=True)

            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            eval_left, eval_right = st.columns([1, 1])
            with eval_left:
                st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">دقت (Precision)، یادآوری (Recall) و F1 به تفکیک کلاس</div>', unsafe_allow_html=True)
                st.plotly_chart(per_class_bar(metrics), width="stretch")
            with eval_right:
                st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">ماتریس درهم‌ریختگی (Confusion Matrix)</div>', unsafe_allow_html=True)
                st.plotly_chart(confusion_heatmap(metrics), width="stretch")

            # Training History line chart
            history_data = metrics.get("history", [])
            if history_data:
                st.markdown('<div class="fa-block" style="font-weight:700; margin-top: 1rem; margin-bottom: 4px;">روند کاهش خطا و بهبود F1 در طول اپوک‌های آموزش</div>', unsafe_allow_html=True)
                st.plotly_chart(training_history_chart(history_data), width="stretch")
                with st.expander("جدول جزئیات عددی سابقه آموزش (Training Log Data)", expanded=False):
                    st.dataframe(pd.DataFrame(history_data), width="stretch")

            neutral = (test.get("per_class", {}) or {}).get("neutral", {})
            if isinstance(neutral, dict) and neutral.get("recall", 1.0) < 0.4:
                st.markdown(
                    '<div class="glass-card" style="margin-top: 1rem; border-right: 4px solid #D97706; background: #FFFBEB; direction: rtl; text-align: right;">'
                    '<div style="color: #B45309; font-weight: 700; margin-bottom: 4px;">⚠️ تحلیل یادآوری کلاس خنثی:</div>'
                    '<div style="color: #92400E; font-size: 0.92rem; line-height: 1.8;">میزان Recall کلاس خنثی پایین‌تر از سایر کلاس‌هاست. علت فنی: برچسب‌ها از ستارهٔ ۱ تا ۵ فروشگاه استخراج شده‌اند و '
                    'امتیاز ۳ ستاره در نظرات واقعی کاربران به‌شدت نادر است (حدود ۱۴ برابر کمتر از ۵ ستاره). به همین دلیل، Macro-F1 معیار معتبر اصلی برای ارزیابی این سامانه در نظر گرفته شده است.</div></div>',
                    unsafe_allow_html=True,
                )

    with tab_news:
        if news is None:
            st.info("پیکرهٔ نظرات خبری یافت نشد — می‌توانید آن را با دستور `python -m src.scraper --source news --out data/raw_news_comments.csv` جمع‌آوری کنید.")
        else:
            rtl(
                f"{len(news):,} نظر جمع‌آوری‌شده از سایت‌های خبری (ورزش۳). این نظرات بدون امتیاز ستاره هستند "
                "و به عنوان دادهٔ خارج از دامنه (Out-of-Domain) برای تحلیل زبان‌شناختی و تست توان تعمیم مدل کاربرد دارند.",
                muted=True,
            )
            votes = pd.to_numeric(news.get("likes"), errors="coerce").fillna(0)
            dislikes = pd.to_numeric(news.get("dislikes"), errors="coerce").fillna(0)
            n1, n2, n3 = st.columns(3)
            with n1:
                st.markdown(metric_card("تعداد خبرها", f"{news.get('news_id', pd.Series()).nunique():,}", "منبع ورزش۳", "#4F46E5", "📰"), unsafe_allow_html=True)
            with n2:
                st.markdown(metric_card("مجموع پسندها (Likes)", f"{int(votes.sum()):,}", "تعامل مثبت کاربران", "#059669", "👍"), unsafe_allow_html=True)
            with n3:
                st.markdown(metric_card("مجموع ناپسندها (Dislikes)", f"{int(dislikes.sum()):,}", "تعامل منفی کاربران", "#DC2626", "👎"), unsafe_allow_html=True)

            body_column = "body" if "body" in news.columns else "clean_text"
            words = top_words(news[body_column].fillna("").astype(str), limit=60)
            st.markdown('<div class="fa-block" style="font-weight:700; margin-top: 1rem; margin-bottom: 4px;">واژگان شاخص نظرات ورزشی</div>', unsafe_allow_html=True)
            st.plotly_chart(top_words_bar(words, limit=18), width="stretch")

            news_cols = [c for c in ("comment_id", "news_title", "body", "likes", "dislikes", "created_at") if c in news.columns]
            st.dataframe(news[news_cols].head(200), width="stretch", height=280)

            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            if st.button("🔮 پیش‌بینی احساسات ۲۰ نظر اخیر خبری با ParsBERT", key="predict_news_btn"):
                predictor, message = get_predictor(str(model_dir))
                if predictor is None:
                    st.error(message)
                else:
                    sample = news.dropna(subset=[body_column]).head(20)
                    with st.spinner("در حال استنتاج روی نظرات خبری..."):
                        results = []
                        for row in sample[body_column]:
                            pred = predictor.predict(str(row))
                            results.append({
                                "متن نظر": str(row)[:110] + "...",
                                "احساس پیش‌بینی‌شده": pred["label_fa"],
                                "کلاس": pred["label"],
                                "درجه اطمینان": f"{pred['confidence'] * 100:.1f}%",
                            })
                        res_frame = pd.DataFrame(results)
                        st.dataframe(res_frame, width="stretch")


# ==========================================================================
# Section 10: Page — Batch Prediction (NEW)
# ==========================================================================
def page_batch(model_dir: Path) -> None:
    """Batch prediction page supporting multiline text paste or CSV file upload."""
    gradient_header(
        "پیش‌بینی دسته‌ای نظرات | Batch Prediction",
        "مجموعه‌ای از نظرات را به صورت متنی یا از طریق بارگذاری فایل CSV وارد کنید تا مدل "
        "احساسات تک‌تک آن‌ها را استخراج کند و خروجی ساختاریافته را تحویل دهد.",
        badge="پردازش گروهی",
    )

    mode = st.radio("روش ورود داده:", ["✏️ ورود متن چندخطی (Paste)", "📁 بارگذاری فایل CSV (Upload)"], horizontal=True)

    predictor, message = get_predictor(str(model_dir))
    if predictor is None:
        st.error(message)
        return

    texts_to_predict: list[str] = []

    if mode == "✏️ ورود متن چندخطی (Paste)":
        sample_batch_text = (
            "کیفیت ساخت و متریال واقعاً عالی بود، خیلی راضی هستم\n"
            "اصلاً پیشنهاد نمی‌کنم، بعد از یک روز خراب شد و گارانتی هم پاسخگو نبود\n"
            "کالای معمولی است و کارایی متوسطی نسبت به قیمتش دارد\n"
            "ارسال به موقع انجام شد اما بسته‌بندی پاره بود\n"
            "طراحی فوق‌العاده زیبا و کاربری بسیار راحتی دارد"
        )
        col_txt_hdr, col_btn = st.columns([4, 1.2])
        with col_btn:
            if st.button("درج نمونه‌های پیش‌فرض", key="fill_batch_samples", use_container_width=True):
                st.session_state["batch_paste_area"] = sample_batch_text

        input_text = st.text_area(
            "نظرات را وارد کنید (هر نظر در یک سطر):",
            value=st.session_state.get("batch_paste_area", ""),
            height=180,
            key="batch_paste_input",
            placeholder="هر سطر نمایانگر یک نظر است...",
        )
        if input_text.strip():
            texts_to_predict = [line.strip() for line in input_text.splitlines() if line.strip()]
            st.caption(f"تعداد نظرات شناسایی‌شده: {len(texts_to_predict):,} سطر")

    else:
        uploaded_file = st.file_uploader("فایل CSV نظرات را انتخاب کنید:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except Exception:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")

            st.success(f"فایل بارگذاری شد: {len(df_upload):,} سطر و {len(df_upload.columns)} ستون")
            text_candidates = [c for c in df_upload.columns if df_upload[c].dtype == "object"]
            default_idx = 0
            for idx, c in enumerate(text_candidates):
                if any(k in c.lower() for k in ["body", "comment", "text", "clean", "متن", "نظر"]):
                    default_idx = idx
                    break

            selected_col = st.selectbox("ستون متن نظرات:", text_candidates, index=default_idx)
            max_rows = min(len(df_upload), 200)
            sample_size = st.slider(
                "حداکثر تعداد نظرات برای پردازش در این مرحله:",
                min_value=1,
                max_value=max_rows,
                value=min(max_rows, 50),
                help="جهت حفظ سرعت روی پردازنده مرکزی (CPU)، تعداد نظرات قابل تنظیم است.",
            )
            texts_to_predict = df_upload[selected_col].dropna().astype(str).head(sample_size).tolist()

    st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    run_batch = st.button("🚀 اجرای پیش‌بینی دسته‌ای", type="primary", disabled=not texts_to_predict)

    if run_batch and texts_to_predict:
        total = len(texts_to_predict)
        progress_bar = st.progress(0, text="در حال آماده‌سازی و استنتاج مدل...")
        results = []
        started = time.perf_counter()

        for idx, item in enumerate(texts_to_predict):
            pred = predictor.predict(item)
            results.append({
                "ردیف": idx + 1,
                "متن نظر": item,
                "احساس": pred["label_fa"] or "نامشخص",
                "کلاس لاتین": pred["label"] or "unknown",
                "درجه اطمینان": f"{pred['confidence'] * 100:.1f}%",
                "confidence_val": pred["confidence"],
                "raw_label": pred["label"],
            })
            progress_bar.progress((idx + 1) / total, text=f"پردازش نظر {idx + 1} از {total}...")

        elapsed = round(time.perf_counter() - started, 2)
        progress_bar.empty()

        res_df = pd.DataFrame(results)
        st.session_state["batch_results"] = res_df
        st.session_state["batch_elapsed"] = elapsed

    batch_res = st.session_state.get("batch_results")
    if batch_res is not None and not batch_res.empty:
        elapsed = st.session_state.get("batch_elapsed", 0.0)
        st.markdown(
            f'<div class="glass-card" style="margin-top: 1rem; margin-bottom: 1rem; border-right: 4px solid #059669; direction: rtl; text-align: right;">'
            f'<span style="font-weight: 700; color: #059669;">✓ پردازش دسته‌ای با موفقیت پایان یافت:</span>'
            f'<span style="color: #334155; font-size: 0.95rem;"> <b>{len(batch_res):,}</b> نظر در مدت <b>{elapsed}</b> ثانیه بر روی CPU ارزیابی شد.</span></div>',
            unsafe_allow_html=True,
        )

        counts = batch_res["raw_label"].value_counts()
        pos_cnt = counts.get("positive", 0)
        neu_cnt = counts.get("neutral", 0)
        neg_cnt = counts.get("negative", 0)
        total_cnt = len(batch_res)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(metric_card("کل پردازش‌شده", f"{total_cnt:,}", f"{elapsed} ثانیه", "#4F46E5", "📋"), unsafe_allow_html=True)
        with b2:
            st.markdown(metric_card("نظرات مثبت", f"{pos_cnt:,}", f"{pos_cnt / total_cnt * 100:.1f}% سهم", LABEL_COLORS["positive"], "😊"), unsafe_allow_html=True)
        with b3:
            st.markdown(metric_card("نظرات خنثی", f"{neu_cnt:,}", f"{neu_cnt / total_cnt * 100:.1f}% سهم", LABEL_COLORS["neutral"], "😐"), unsafe_allow_html=True)
        with b4:
            st.markdown(metric_card("نظرات منفی", f"{neg_cnt:,}", f"{neg_cnt / total_cnt * 100:.1f}% سهم", LABEL_COLORS["negative"], "😤"), unsafe_allow_html=True)

        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        col_table, col_pie = st.columns([1.6, 1])
        with col_pie:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">توزیع احساسات این دسته</div>', unsafe_allow_html=True)
            pie_counts = pd.Series([neg_cnt, neu_cnt, pos_cnt], index=["negative", "neutral", "positive"])
            pie_fig = go.Figure(
                go.Pie(
                    labels=[LABEL_FA[name] for name in pie_counts.index],
                    values=pie_counts.values.tolist(),
                    hole=0.48,
                    marker={"colors": [LABEL_COLORS[name] for name in pie_counts.index]},
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b>: %{value} نظر (%{percent})<extra></extra>",
                )
            )
            pie_fig.update_layout(
                template="plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=260,
                margin={"l": 10, "r": 10, "t": 10, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(pie_fig, width="stretch")

        with col_table:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom: 4px;">جدول نتایج تفکیکی</div>', unsafe_allow_html=True)
            display_cols = ["ردیف", "متن نظر", "احساس", "درجه اطمینان"]
            st.dataframe(batch_res[display_cols], width="stretch", height=260)

        csv_io = io.BytesIO()
        export_df = batch_res[["ردیف", "متن نظر", "احساس", "کلاس لاتین", "درجه اطمینان"]]
        export_df.to_csv(csv_io, index=False, encoding="utf-8-sig")
        csv_bytes = csv_io.getvalue()

        st.download_button(
            label="📥 دانلود نتایج دسته‌ای به عنوان فایل CSV (سازگار با اکسل فارسی)",
            data=csv_bytes,
            file_name=f"sentiment_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==========================================================================
# Section 10.5: Page — Product Advisor (Digikala URL Analyzer)
# ==========================================================================
def page_advisor(model_dir: Path) -> None:
    """Analyze Digikala product comments live and generate buy/skip recommendation."""
    gradient_header(
        "مشاور هوشمند خرید کالا | Digikala Product Advisor",
        "لینک یا کد کالای دیجی‌کالا را وارد کنید تا نظرات واقعی خریداران مستقیماً استخراج شده و با مدل هوش مصنوعی ParsBERT تحلیل شوند.",
        badge="مشاوره خرید کالا",
    )

    from src.product_advisor import (
        analyze_digikala_product,
        RecommendationVerdict,
    )

    # Quick sample links
    col_in, col_samples = st.columns([2.6, 1.4])
    with col_samples:
        st.markdown('<div class="fa-block" style="font-size:0.85rem; font-weight:600; margin-bottom:4px;">کالاهای نمونه برای تست سریع:</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        if s1.button("گوشی Galaxy A56", key="sample_a56", use_container_width=True):
            st.session_state["product_url_input"] = "https://www.digikala.com/product/dkp-18575915/"
        if s2.button("هدفون بلوتوثی P47", key="sample_p47", use_container_width=True):
            st.session_state["product_url_input"] = "https://www.digikala.com/product/dkp-5112620/"

    with col_in:
        prod_input = st.text_input(
            "آدرس یا شناسه کالای دیجی‌کالا (URL یا کد dkp):",
            value=st.session_state.get("product_url_input", ""),
            placeholder="مثال: https://www.digikala.com/product/dkp-18575915/ یا 18575915",
            key="product_url_text_input",
        )

    c_cnt, c_btn = st.columns([1.5, 1])
    with c_cnt:
        comments_limit = st.slider(
            "تعداد نظرات برای استخراج و تحلیل:",
            min_value=5,
            max_value=60,
            value=25,
            step=5,
            help="نظرات از جدیدترین نظرات مشتریان دیجی‌کالا خوانده می‌شوند.",
        )
    with c_btn:
        st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        analyze_btn = st.button("🚀 استخراج و تحلیل ارزش خرید", type="primary", use_container_width=True, disabled=not prod_input.strip())

    if analyze_btn and prod_input.strip():
        with st.spinner("در حال واکشی اطلاعات و نظرات از دیجی‌کالا و استنتاج احساسات توسط ParsBERT..."):
            try:
                report = analyze_digikala_product(prod_input.strip(), max_comments=comments_limit, model_dir=model_dir)
                st.session_state["advisor_report"] = report
            except Exception as e:
                st.error(f"خطا در پردازش کالا: {e}")

    report = st.session_state.get("advisor_report")
    if report:
        prod = report.product
        rec = report.recommendation

        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        p_col1, p_col2 = st.columns([1, 2.5])
        with p_col1:
            if prod.get("image_url"):
                st.image(prod["image_url"], width=220)
            else:
                st.markdown('<div style="font-size: 4rem; text-align: center;">📦</div>', unsafe_allow_html=True)
        with p_col2:
            st.markdown(
                f'<div class="glass-card" style="direction: rtl; text-align: right;">'
                f'<div style="font-size: 1.25rem; font-weight: 800; color: #1E293B; margin-bottom: 6px;">{fa(prod.get("title_fa", ""))}</div>'
                f'<div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">برند: <b>{fa(prod.get("brand", "نامشخص"))}</b> | شناسه کالا: <code>dkp-{prod.get("product_id")}</code></div>'
                + (f'<div style="font-size: 1.2rem; font-weight: 700; color: #059669; margin-bottom: 6px;">💰 قیمت: {prod.get("price_toman"):,} تومان</div>' if prod.get("price_toman") else '')
                + (f'<div style="color: #334155; font-size: 0.88rem;">⭐ رضایت کلی دیجی‌کالا: {prod.get("rating_score")}٪ (از {prod.get("rating_count"):,} رأی)</div>' if prod.get("rating_score") else '')
                + f'<div style="margin-top: 8px;"><a href="{prod.get("url")}" target="_blank" style="color: #4F46E5; text-decoration: none; font-size: 0.88rem;">🔗 مشاهده کالا در دیجی‌کالا</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        verdict_color = "#059669" if "🟢" in rec.verdict else ("#D97706" if "🟡" in rec.verdict else "#DC2626")
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="verdict-hero fade-in" style="background: {verdict_color}10; border-color: {verdict_color}44;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<div class="verdict-label" style="color: {verdict_color};">{fa(rec.verdict)}</div>'
            f'<div style="font-size: 1.4rem; font-weight: 800; color: {verdict_color};">امتیاز ارزش خرید: {rec.score_out_of_100:.1f} / ۱۰۰</div>'
            f'</div>'
            f'<div style="font-size: 1.05rem; color: #1E293B; line-height: 1.8; margin-top: 0.75rem;">{fa(rec.summary_text_fa)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("نظرات تحلیل‌شده", f"{report.analyzed_count}", f"در {report.elapsed_seconds} ثانیه", "#4F46E5", "💬"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("احساسات مثبت", f"{rec.positive_ratio * 100:.1f}٪", "سهم نظرات مثبت", "#059669", "😊"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("احساسات منفی", f"{rec.negative_ratio * 100:.1f}٪", "سهم نظرات منفی", "#DC2626", "😤"), unsafe_allow_html=True)
        with m4:
            buyer_stat = f"{rec.buyer_positive_ratio * 100:.1f}٪" if rec.buyer_comments_count else "نامشخص"
            st.markdown(metric_card("رضایت خریداران قطعی", buyer_stat, f"از {rec.buyer_comments_count} خریدار", "#D97706", "🛒"), unsafe_allow_html=True)

        if rec.top_advantages or rec.top_disadvantages:
            adv_col, dis_col = st.columns(2)
            with adv_col:
                if rec.top_advantages:
                    st.markdown('<div class="fa-block" style="font-weight:700; color:#059669; margin-top:1rem; margin-bottom:6px;">👍 نقاط قوت پربسامد گزارش‌شده:</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip" style="background:#ECFDF5; color:#047857; border-color:#A7F3D0;">✓ {fa(a)}</span>' for a in rec.top_advantages)
                    st.markdown(chips, unsafe_allow_html=True)
            with dis_col:
                if rec.top_disadvantages:
                    st.markdown('<div class="fa-block" style="font-weight:700; color:#DC2626; margin-top:1rem; margin-bottom:6px;">👎 نقاط ضعف و ایرادات مکرر:</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip" style="background:#FEF2F2; color:#B91C1C; border-color:#FECACA;">✗ {fa(d)}</span>' for d in rec.top_disadvantages)
                    st.markdown(chips, unsafe_allow_html=True)

        if rec.reasons_fa:
            with st.expander("📌 دلایل شاخص تصمیم‌گیری مدل (Decision Factors)", expanded=True):
                reasons_html = "".join(f'<div class="fa-block" style="margin-bottom: 4px;">• {fa(r)}</div>' for r in rec.reasons_fa)
                st.markdown(reasons_html, unsafe_allow_html=True)

        with st.expander(f"📋 مشاهده تمام {report.analyzed_count} نظر تحلیل‌شده با برچسب احساسات", expanded=False):
            table_rows = [
                {
                    "ردیف": idx + 1,
                    "متن نظر": c["body"],
                    "احساس": c["sentiment_fa"],
                    "خریدار": "بله" if c.get("is_buyer") else "خیر",
                    "امتیاز کاربر": f"{c.get('rate') or '-'} ستاره",
                    "درجه اطمینان": f"{c['confidence'] * 100:.1f}٪",
                }
                for idx, c in enumerate(report.comments_sample)
            ]
            st.dataframe(pd.DataFrame(table_rows), width="stretch", height=280)


# ==========================================================================
# Section 11: Page — Help & Limits
# ==========================================================================
def page_help() -> None:
    """Documentation, architecture workflow, CLI guide, and explicit technical limits."""
    gradient_header(
        "راهنمای سامانه و محدودیت‌های فنی",
        "مستندات معماری، خط‌لوله داده، فرامین خط فرمان و بیان شفاف محدودیت‌های مدل و داده‌ها.",
        badge="مستندات مهندسی",
    )

    rtl(
        "این سامانه برای ارزیابی احساسات سه کلاسه (مثبت / خنثی / منفی) زبان فارسی بر بستر ParsBERT پیاده‌سازی شده "
        "و داده‌های آموزشی آن از نظرات دارای ستاره در دیجی‌کالا و نظرات خبری بدون برچسب ورزش۳ استخراج شده است.",
        muted=True,
    )

    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    c_pipe, c_limits = st.columns([1.1, 1.2])

    with c_pipe:
        st.markdown(
            '<div class="glass-card" style="margin-bottom: 1rem; border-top: 4px solid #4F46E5;">'
            '<div style="font-weight: 700; color: #1E293B; font-size: 1.1rem; margin-bottom: 8px; direction: rtl; text-align: right;">🛠️ فرامین خط لوله (Pipeline CLI)</div>'
            '<div style="font-size: 0.9rem; color: #475569; direction: rtl; text-align: right; margin-bottom: 8px;">تمامی مراحل از جمع‌آوری تا آموزش به صورت ماژول‌های مستقل پایتون قابل اجرا هستند:</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
```bash
# ۱. تست سلامت خزشگرها
python -m src.scraper --smoke

# ۲. خزش نظرات برچسب‌دار دیجی‌کالا و نظرات خبری
python -m src.scraper --source digikala --target 5000 --out data/raw_digikala_comments.csv
python -m src.scraper --source news --target 4000 --out data/raw_news_comments.csv

# ۳. پیش‌پردازش و پاک‌سازی hazm (قاعده تبدیل ZWNJ به فاصله)
python -m src.preprocess --in data/raw_digikala_comments.csv --out data/cleaned_comments.csv

# ۴. تولید ارقام آماری و ابر کلمات
python -m src.visualize --csv data/cleaned_comments.csv --out assets/figures

# ۵. فاین‌تیون مدل ParsBERT با توابع زیان وزن‌دار
python -m src.train --smoke
python -m src.train --csv data/cleaned_comments.csv --epochs 3

# ۶. اجرای داشبورد وب Streamlit
python -m streamlit run app.py
```
            """
        )

    with c_limits:
        st.markdown(
            '<div class="glass-card" style="margin-bottom: 1rem; border-top: 4px solid #D97706;">'
            '<div style="font-weight: 700; color: #1E293B; font-size: 1.1rem; margin-bottom: 8px; direction: rtl; text-align: right;">⚠️ محدودیت‌های فنی و واقعیت‌های مهندسی</div></div>',
            unsafe_allow_html=True,
        )

        limits = [
            ("نظارت ضعیف (Weak Supervision)", "برچسب‌های احساسات مستقیماً از امتیازهای ۱ تا ۵ ستاره مشتق شده‌اند (۱-۲ منفی، ۳ خنثی، ۴-۵ مثبت). این برچسب‌گذاری تقریب ضعیف است نه برچسب‌گذاری دستی توسط کارشناس زبان‌شناس."),
            ("عدم توازن شدید کلاس خنثی", "در رفتارهای خرید واقعی، نظرات ۳ ستاره حدود ۱۴ برابر کمتر از نظرات ۵ ستاره هستند. به همین علت یادآوری (Recall) کلاس خنثی پایین‌تر است و معیار کلیدی پروژه Macro-F1 است، نه صرفاً Accuracy سطحی."),
            ("فقدان دیتاست سه‌کلاسه استاندارد", "دیتاست‌های در دسترس در Hugging Face (مثل SnappFood mirrors و Digikala dumps) عمدتاً دودویی (Binary) هستند. به همین علت پیکرهٔ سه کلاسه مستقیماً خزش و تولید شد."),
            ("عدم وجود امتیاز در نظرات خبری", "نظرات سایت‌های خبری مانند ورزش۳ فاقد ستارهٔ رضایت هستند و صرفاً برای تحلیل واژگانی و آزمون استنتاج خارج از دامنه استفاده می‌شوند."),
            ("محدودیت سخت‌افزاری پردازنده (CPU-only)", "این محیط فاقد GPU انویدیا است. مدل ۱۶۳ میلیون پارامتری ParsBERT با حداکثر بهینه‌سازی بارگذاری و کش استنتاج می‌شود."),
        ]

        limits_html = "".join(
            f'<div class="glass-card" style="margin-bottom: 8px; padding: 0.75rem 1rem; border-right: 4px solid #CBD5E1; direction: rtl; text-align: right;">'
            f'<div style="font-weight: 700; color: #1E293B; font-size: 0.95rem; margin-bottom: 3px;">• {fa(title)}</div>'
            f'<div style="font-size: 0.88rem; color: #475569; line-height: 1.7;">{fa(desc)}</div></div>'
            for title, desc in limits
        )
        st.markdown(limits_html, unsafe_allow_html=True)


# ==========================================================================
# Section 12: Main Entry Point and Navigation
# ==========================================================================
def _existing(paths: list[Path]) -> list[Path]:
    """Filter a list of paths to only those that exist on disk."""
    return [path for path in paths if path.exists()]


def main() -> None:
    """Main Streamlit application entry point."""
    _inject_theme()

    with st.sidebar:
        st.markdown(
            '<div style="padding: 0.4rem 0 1rem 0; text-align: center;">'
            '<div class="sidebar-brand">BAFP</div>'
            '<div style="font-size: 0.84rem; color: #64748B; font-weight: 600; margin-top: 3px;">تحلیل هوشمند احساسات فارسی</div></div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "صفحه",
            ["Live prediction", "Product advisor", "Dashboard", "Batch prediction", "Help & limits"],
            index=0,
            format_func=lambda p: {
                "Live prediction": "🔮 پیش‌بینی زنده",
                "Product advisor": "🛍️ مشاور خرید کالا",
                "Dashboard": "📊 داشبورد آماری",
                "Batch prediction": "📋 پیش‌بینی دسته‌ای",
                "Help & limits": "ℹ️ راهنما و محدودیت‌ها",
            }[p],
        )
        st.divider()

        # Corpus Selection
        corpora = _existing([CLEANED_CSV, WIDE_CLEANED_CSV])
        corpus_choice = (
            st.selectbox("پیکره داده (Corpus)", [path.name for path in corpora], index=len(corpora) - 1)
            if corpora else None
        )

        # Checkpoint Selection
        checkpoint_names = [path.name for path in available_models()]
        chosen = (
            st.selectbox(
                "مدل ذخیره‌شده (Checkpoint)",
                checkpoint_names,
                index=checkpoint_names.index(DEFAULT_MODEL_DIR.name) if DEFAULT_MODEL_DIR.name in checkpoint_names else 0,
                help="parsbert-sentiment-3class: پیکره ۱۶ کالا؛ -wide: پیکره ۵۳ کالا",
            )
            if checkpoint_names else None
        )

        st.divider()
        rtl("پیکره: نظرات دیجی‌کالا (برچسب‌دار) و ورزش۳ (بدون برچسب)", muted=True)

        # About Footer
        st.markdown(
            '<div class="glass-card" style="margin-top: 1.2rem; padding: 0.85rem; font-size: 0.8rem; color: #64748B; direction: rtl; text-align: right;">'
            '<div style="font-weight: 700; color: #334155; margin-bottom: 4px;">مشخصات فنی سامانه:</div>'
            '<div>مدل: ParsBERT (163M)</div><div>سخت‌افزار: ویندوز (CPU-only)</div><div>ابزارها: Streamlit 1.59 · Hazm</div></div>',
            unsafe_allow_html=True,
        )

    model_dir = MODELS_DIR / chosen if chosen else DEFAULT_MODEL_DIR
    frame = load_cleaned(Path(DATA_DIR / corpus_choice)) if corpus_choice else None
    news = load_news(NEWS_CSV)

    if page == "Dashboard":
        page_dashboard(frame, news, model_dir, corpus_choice or "none")
    elif page == "Live prediction":
        page_prediction(model_dir)
    elif page == "Product advisor":
        page_advisor(model_dir)
    elif page == "Batch prediction":
        page_batch(model_dir)
    else:
        page_help()


if __name__ == "__main__":
    main()
