"""
Streamlit dashboard for the BAFP Persian sentiment capstone — v2 Redesign.
========================================================================

Modern Persian-first RTL dashboard with light/dark themes, glassmorphism,
micro-animations, and optimised inference paths.

Pages (sidebar radio):
  1. پیش‌بینی زنده  / Live prediction  — hero verdict, gauge, raw-vs-cleaned
  2. مشاور خرید     / Product advisor  — Digikala URL → sentiment verdict
  3. داشبورد آماری / Dashboard        — corpus analytics & model evaluation
  4. پیش‌بینی دسته‌ای / Batch          — batched inference (predict_batch)
  5. راهنما          / Help & limits   — architecture & limitations

Design decisions
----------------
* Streamlit 1.59 has no native RTL → a global CSS pass flips the app shell
  (sidebar moves to the right edge; columns, tabs and radios flow
  right-to-left), and every Persian block is wrapped in ``dir="rtl"`` with
  ``Vazirmatn`` and bidi isolates for LTR fragments. Multi-fact notes render
  one line per fact (``.fa-lines``) instead of mid-dot-glued run-ons.
* No Persian font on the host → Vazirmatn is lazy-fetched to assets/fonts/.
  Plotly draws Persian natively (browser shaping); only word-clouds need
  pre-shaping via ``src.visualize.build_wordcloud_image``.
* Model is ~163M params, CPU-only → ``get_predictor`` is ``cache_resource``,
  batch page uses ``predict_batch`` (single forward pass per 32 texts).
* All charts use Plotly 6.8 / pandas 3.0 APIs only (via constraints.txt).

Run:
    python -m streamlit run app.py
"""

from __future__ import annotations

import base64
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

st.set_page_config(
    page_title="تحلیل احساسات فارسی | BAFP",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_KEY = "bafp_theme"


def _get_theme() -> str:
    return st.session_state.get(THEME_KEY, "light")


def _toggle_theme() -> None:
    cur = st.session_state.get(THEME_KEY, "light")
    new = "dark" if cur == "light" else "light"
    st.session_state[THEME_KEY] = new
    # Persist in the URL so a page reload / shared link keeps the theme
    try:
        st.query_params["theme"] = new
    except Exception:
        pass


# ==========================================================================
# Section 2: Theme — CSS variables, light/dark, animations
# ==========================================================================
def _inject_theme() -> None:
    """Inject the full design system. Light is default; dark overrides via variables."""
    theme = _get_theme()
    is_dark = theme == "dark"

    # Color tokens interpolated into CSS — avoids Streamlit theme repaint flicker
    bg = "#0B1220" if is_dark else "#F8FAFC"
    surface = "#151E32" if is_dark else "#FFFFFF"
    surface_2 = "#1E2A45" if is_dark else "#F1F5F9"
    border = "#24304E" if is_dark else "#E2E8F0"
    border_strong = "#2E3D5E" if is_dark else "#CBD5E1"
    text = "#E2E8F0" if is_dark else "#0F172A"
    text_muted = "#94A3B8" if is_dark else "#64748B"
    text_faint = "#64748B" if is_dark else "#94A3B8"
    # Primary palette stays vibrant in both modes
    primary = "#6366F1"
    primary_2 = "#8B5CF6"
    # Gradient title: the style block is re-injected on every theme change, so
    # bake the gradient in directly — a [data-theme] ancestor hook never
    # matches because the marker div is a sibling of the content, not its
    # ancestor.
    title_gradient = (
        "linear-gradient(135deg, #A5B4FC 0%, #818CF8 35%, #22D3EE 70%, #34D399 100%)"
        if is_dark
        else "linear-gradient(135deg, #1E1B4B 0%, #6366F1 35%, #06B6D4 70%, #10B981 100%)"
    )

    # Streamlit chrome (labels, tabs, selects, expanders, uploader, inputs) is
    # light-native: when dark mode is active, restyle every widget container so
    # nothing renders as dark-text-on-dark-background. Light mode only gets a
    # link-color polish.
    if is_dark:
        streamlit_overrides = """
        /* ---- Dark coverage for native Streamlit widgets ---- */
        .stApp, [data-testid="stAppViewContainer"] { color: #E2E8F0; }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: #CBD5E1; }
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p { color: #E2E8F0 !important; }
        [data-testid="stCaption"], .stCaption, p.caption { color: #A3B0C6 !important; }
        label { color: #CBD5E1; }
        a { color: #818CF8; }
        div.stButton > button[kind="secondary"] { background: #1E2A45 !important; color: #E2E8F0 !important; border-color: #2E3D5E !important; }
        div.stButton > button[kind="secondary"]:hover { background: #24304E !important; border-color: #6366F1 !important; }
        div.stButton > button[kind="secondary"] p { color: #E2E8F0 !important; }
        /* Tooltip-wrapped buttons (help=…) dodge the kind selectors — pin them
           by testid so they never fall back to the browser's auto theme */
        button[data-testid="stBaseButton-secondary"] { background: #1E2A45 !important; color: #E2E8F0 !important; border: 1px solid #2E3D5E !important; border-radius: var(--radius-sm) !important; font-weight: 500 !important; }
        button[data-testid="stBaseButton-secondary"]:hover { background: #24304E !important; border-color: #6366F1 !important; color: #E2E8F0 !important; }
        button[data-testid="stBaseButton-secondary"] p { color: #E2E8F0 !important; }
        button[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important; color: #fff !important; border: none !important; }
        button[data-testid="stBaseButton-primary"] p { color: #fff !important; }
        code, pre { background: #0F172A !important; color: #E2E8F0 !important; border-radius: 8px; }
        [data-testid="stExpander"] { background: #151E32; border: 1px solid #24304E; border-radius: 14px; }
        [data-testid="stExpander"] summary { color: #E2E8F0; }
        [data-testid="stExpander"] svg { fill: #94A3B8; }
        [data-baseweb="tab-list"] { border-color: #24304E !important; gap: 4px; }
        [data-baseweb="tab"] { color: #94A3B8 !important; }
        [data-baseweb="tab"]:hover { color: #E2E8F0 !important; }
        [data-baseweb="tab"][aria-selected="true"] { color: #A5B4FC !important; }
        [data-baseweb="tab-highlight"] { background: #6366F1 !important; border-radius: 4px; }
        /* Selectbox: 1.59 renders selects as a react-aria ComboBox — the old
           data-baseweb="select" hooks match nothing, leaving a white control
           with dark text. Keep the legacy selectors alongside. */
        .react-aria-ComboBox > div, div[data-baseweb="select"] > div { background: #151E32 !important; border-color: #24304E !important; color: #E2E8F0 !important; }
        .react-aria-ComboBox input { background: transparent !important; color: #E2E8F0 !important; caret-color: #E2E8F0 !important; }
        .react-aria-ComboBox button { color: #94A3B8 !important; }
        div[data-baseweb="select"] > div > div, div[data-baseweb="select"] svg, .react-aria-ComboBox svg { color: #94A3B8 !important; fill: #94A3B8; }
        /* Open dropdown: the floating listbox has no opaque surface of its own
           and inherits dark option text — paint the panel and options. */
        [data-testid="stSelectboxVirtualDropdown"] { background: #1E2A45 !important; border: 1px solid #2E3D5E !important; border-radius: 10px !important; box-shadow: var(--shadow-md) !important; overflow: hidden; }
        [data-testid="stSelectboxVirtualDropdown"] [role="option"],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"] > div { color: #E2E8F0 !important; }
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"] > div,
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] > div { background: rgba(99,102,241,0.28) !important; }
        [data-baseweb="popover"], [data-baseweb="menu"] { background: #1E2A45 !important; border-color: #2E3D5E !important; }
        [data-baseweb="menu"] li { color: #CBD5E1 !important; }
        [data-baseweb="menu"] li:hover { background: #24304E !important; }
        [data-testid="stFileUploaderDropzone"] { background: #151E32 !important; border: 1.5px dashed #2E3D5E !important; color: #94A3B8 !important; border-radius: 14px; }
        [data-testid="stFileUploaderDropzone"] button { background: #1E2A45 !important; color: #E2E8F0 !important; border-color: #2E3D5E !important; }
        [data-testid="stRadio"] label p, [data-testid="stRadio"] label { color: #CBD5E1 !important; }
        [data-testid="stCheckbox"] label p { color: #CBD5E1 !important; }
        [data-testid="stTextInput"] input { background: #151E32 !important; color: #E2E8F0 !important; border-color: #24304E !important; border-radius: 10px; }
        [data-testid="stTextInput"] label p { color: #E2E8F0 !important; }
        [data-testid="stNumberInput"] input { background: #151E32 !important; color: #E2E8F0 !important; }
        [data-testid="stDivider"], [data-testid="stMain"] hr { background: #24304E !important; border: none !important; }
        [data-testid="stSpinner"] { color: #CBD5E1; }
        [data-testid="stMultiInput"] input { background: #151E32 !important; color: #E2E8F0 !important; }
        [data-baseweb="tag"], [data-testid="stMultiInput"] div[data-baseweb="tag"] { background: #1E2A45 !important; color: #CBD5E1 !important; }
        /* ---- Dark accent recolor: Streamlit's stock primary is red; the design
           system is indigo. Mirror the light-mode widget recolor with the
           brighter dark-palette indigo. ---- */
        /* Radio nav: selected dot — scoped to the circle only, the generic depth
           selector would paint the label's markdown container too (round-6 bug) */
        [data-testid="stRadioOption"][data-selected="true"] > div > div > div:not([data-testid="stMarkdownContainer"]) { background-color: #818CF8 !important; }
        [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] { background-color: transparent !important; }
        /* Unselected radio circles: lift them off the dark background */
        [data-testid="stRadioOption"] > div > div > div:not([data-testid="stMarkdownContainer"]) { border-color: #475569 !important; }
        /* Slider: native theming (config.toml [theme] base=dark, primaryColor
           #6366F1) fills the track proportionally — the old full-track
           gradient override painted the whole rail indigo regardless of value.
           Only the thumb-value text color stays pinned. */
        [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] div,
        [data-testid="stSliderThumbValue"] p { color: #A5B4FC !important; border-color: transparent !important; }
        /* Tabs: selection indicator */
        [data-testid="stTabs"] .react-aria-SelectionIndicator { background-color: #6366F1 !important; border-radius: 999px; }
        """
    else:
        streamlit_overrides = """
        a { color: #4F46E5; }
        /* Light-mode primary recolor: Streamlit's stock primary is red while the
           design system is indigo. Pin the native widgets to indigo so nothing
           renders red. */
        /* Radio nav: selected dot — scoped to the circle container only; the
           generic depth selector also matched the label's markdown container
           and painted the whole nav row solid blue */
        [data-testid="stRadioOption"][data-selected="true"] > div > div > div:not([data-testid="stMarkdownContainer"]) { background-color: #6366F1 !important; }
        [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] { background-color: transparent !important; }
        /* Tabs: selected label + underline indicator */
        [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
        [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p { color: #4F46E5 !important; }
        [data-testid="stTabs"] .react-aria-SelectionIndicator { background-color: #6366F1 !important; border-radius: 999px; }
        /* Slider: native theming (config.toml [theme] primaryColor #6366F1)
           fills the track proportionally — the old full-track gradient
           override painted the whole rail indigo regardless of value. Only
           the thumb-value text color stays pinned. */
        [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] div,
        [data-testid="stSliderThumbValue"] p { color: #4F46E5 !important; border-color: transparent !important; }
        /* Buttons: every variant pinned to the light palette (baseweb auto-themes
           secondary buttons dark on dark-OS systems, dodging the kind selectors) */
        button[data-testid="stBaseButton-secondary"] {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
        }
        button[data-testid="stBaseButton-secondary"]:hover {
            background: var(--surface-2) !important;
            border-color: #6366F1 !important;
            color: var(--text) !important;
        }
        button[data-testid="stBaseButton-secondary"] p { color: var(--text) !important; }
        button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            color: #fff !important;
            border: none !important;
        }
        button[data-testid="stBaseButton-primary"] p { color: #fff !important; }
        /* Buttons: settle on press so the hover lift doesn't read as a jump */
        div.stButton > button[kind="primary"]:active,
        button[data-testid="stBaseButton-primary"]:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 8px rgba(99,102,241,0.32) !important;
        }
        """

    st.markdown(
        f"""
        <style>
        /* Vazirmatn is served LOCALLY via Streamlit static serving
           (.streamlit/config.toml -> static/fonts/). The old Google Fonts
           @import rendered differently (or timed out on first paint) wherever
           fonts.googleapis.com is blocked or slow. One variable file covers
           weights 100-900. */
        @font-face {{
            font-family: 'Vazirmatn';
            /* Streamlit mounts ./static under /app/static — a bare /static
               path falls through to the SPA HTML instead of the font. */
            src: url('/app/static/fonts/Vazirmatn-Variable.ttf') format('truetype');
            font-weight: 100 900;
            font-style: normal;
            font-display: swap;
        }}

        :root {{
            --bg: {bg};
            --surface: {surface};
            --surface-2: {surface_2};
            --border: {border};
            --border-strong: {border_strong};
            --text: {text};
            --text-muted: {text_muted};
            --text-faint: {text_faint};
            --primary: {primary};
            --primary-2: {primary_2};
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --shadow-sm: 0 4px 16px rgba(15,23,42,0.06);
            --shadow-md: 0 10px 28px rgba(15,23,42,0.08);
            --shadow-lg: 0 18px 40px rgba(15,23,42,0.12);
        }}

        html, body, [class*="css"] {{
            font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, sans-serif;
        }}
        .stApp {{
            background: var(--bg);
            transition: background 0.35s ease;
        }}
        /* Main container breathing room */
        .block-container {{
            padding-top: 1.7rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 7px; height: 7px; }}
        ::-webkit-scrollbar-track {{ background: var(--surface-2); border-radius: 4px; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--text-faint); }}

        /* ---- Global RTL shell: sidebar at the right edge, RTL flow ---- */
        [data-testid="stAppViewContainer"] {{ flex-direction: row-reverse; }}
        [data-testid="stMain"], [data-testid="stSidebarContent"], [data-testid="stSidebarContentContainer"] {{ direction: rtl; }}
        /* Sidebar header keeps LTR metrics; flip it so the collapse control
           hugs the INNER edge (towards the content) and the arrow points
           toward the right-hand sidebar. */
        [data-testid="stSidebarHeader"] {{ direction: rtl; }}
        [data-testid="stSidebarCollapseButton"] svg {{ transform: scaleX(-1); }}
        /* Collapsed state: the floating expand control must hug the RIGHT edge
           (where the sidebar went), and the in-sidebar collapse control — which
           Streamlit parks off-screen in LTR — ends up dead under the toolbar
           once flipped, so hide it. */
        [data-testid="stExpandSidebarButton"] {{
            position: fixed; top: 16px; right: 19px; left: auto; z-index: 9999995;
        }}
        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] {{ display: none; }}
        /* Streamlit's sidebar RESIZE HANDLE: an 8px col-resize strip glued to
           the sidebar's edge. It hijacks the pointer and re-opens the closed
           sidebar as a hover preview — remove it entirely; expand/collapse uses
           the arrow controls instead. */
        [data-testid="stSidebar"] [class*="eelgd2m3"] {{ display: none !important; }}
        [data-testid="stSidebar"] div[style*="col-resize"] {{ display: none !important; }}
        /* Stock expand/collapse controls are theme-naive (a glaring white blob
           in dark mode) — restyle them with the app's own tokens. */
        [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button {{
            background: var(--surface) !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        [data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapseButton"] svg {{
            color: var(--text-muted) !important; fill: var(--text-muted) !important;
        }}
        [data-testid="stExpandSidebarButton"]:hover, [data-testid="stSidebarCollapseButton"] button:hover {{
            border-color: var(--primary) !important;
            box-shadow: var(--shadow-md) !important;
        }}
        [data-testid="stExpandSidebarButton"]:hover svg, [data-testid="stSidebarCollapseButton"] button:hover svg {{
            color: var(--primary) !important; fill: var(--primary) !important;
        }}
        /* Header band should never flash a foreign color in dark mode */
        [data-testid="stHeader"] {{ background: transparent !important; }}
        /* Latin-only surfaces stay LTR */
        pre, code, [data-testid="stCodeBlock"], [data-testid="stTextInput"] input,
        .react-aria-ComboBox input {{
            direction: ltr; text-align: left;
        }}
        /* Native captions and alerts follow the Persian direction */
        [data-testid="stCaption"], .stCaption,
        [data-testid="stAlert"], [data-baseweb="notification"] {{ direction: rtl; text-align: right; }}
        /* Sliders keep LTR track geometry (min at left, max at right), so their
           tick labels and thumb value must follow the track — the inherited RTL
           direction flips the labels and they end up swapped. */
        [data-testid="stSliderTickBar"], [data-testid="stSliderThumbValue"] {{ direction: ltr; }}
        /* One fact per line — no mid-dot glued run-on sentences */
        .fa-lines > div {{ margin-bottom: 0.34rem; }}
        .fa-lines > div:last-child {{ margin-bottom: 0; }}

        /* Persian blocks */
        .fa-block {{ direction: rtl; text-align: right; font-size: 1.02rem; line-height: 1.95; font-family: 'Vazirmatn', Tahoma, sans-serif; color: var(--text); }}
        .fa-block.muted {{ color: var(--text-muted); font-size: 0.92rem; }}
        .fa-block.faint {{ color: var(--text-faint); font-size: 0.85rem; }}

        /* Textarea — rounded, RTL, focus ring */
        div[data-baseweb="textarea"], textarea {{
            direction: rtl; text-align: right;
            font-family: 'Vazirmatn', Tahoma, sans-serif !important;
            font-size: 1rem !important; line-height: 1.9 !important;
            border-radius: var(--radius-md) !important;
            border: 1.5px solid var(--border) !important;
            background: var(--surface) !important;
            color: var(--text) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}
        div[data-baseweb="textarea"]:focus-within, textarea:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
        }}

        /* Select & inputs */
        div[data-baseweb="select"] > div, .react-aria-ComboBox > div {{
            border-radius: var(--radius-sm) !important;
            border-color: var(--border) !important;
            background: var(--surface) !important;
        }}
        /* Select value: a px font-size keeps the filename inside the
           fixed-width pill even when a browser "large font" setting scales
           every rem-sized text (rem 18+ overflowed and clipped the value). */
        .react-aria-ComboBox input {{ font-size: 13px; }}

        /* Headings */
        h1, h2, h3, h4 {{ letter-spacing: -0.02em; font-weight: 700; color: var(--text); }}

        /* Animations */
        @keyframes gradientFlow {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
        @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
        @keyframes shimmer {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(100%); }} }}
        @keyframes pulseDot {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:0.6; transform:scale(0.85); }} }}
        /* Opacity-only entrance: a translate animation replays on every
           Streamlit rerun and makes the page look like it jumps on click. */
        .fade-in {{ animation: fadeIn 0.35s ease forwards; }}
        .stagger-1 {{ animation-delay: 0.06s; opacity:0; }}
        .stagger-2 {{ animation-delay: 0.12s; opacity:0; }}
        .stagger-3 {{ animation-delay: 0.18s; opacity:0; }}

        /* Gradient title */
        .gradient-title {{
            background: {title_gradient};
            background-size: 200% 200%; animation: gradientFlow 9s ease infinite;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.0rem; font-weight: 800; line-height: 1.35; margin: 0;
        }}
        .sidebar-brand {{
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #3B82F6 100%);
            background-size: 200% 200%; animation: gradientFlow 6s ease infinite;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 1.9rem; font-weight: 900; letter-spacing: 0.5px;
        }}

        /* Cards — glassmorphism */
        .glass-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.15rem 1.35rem;
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }}
        .glass-card:hover {{ box-shadow: var(--shadow-md); border-color: var(--border-strong); }}
        .glass-card.accent-top {{ border-top: 3px solid var(--primary); }}

        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow-sm);
            transition: transform 0.22s cubic-bezier(0.16,1,0.3,1), box-shadow 0.22s ease;
            position: relative; overflow: hidden;
        }}
        .metric-card::before {{
            content: ''; position: absolute; top:0; left:0; right:0; height: 3px;
            background: var(--metric-accent, var(--primary));
            opacity: 0.9;
        }}
        .metric-card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-md); }}
        .metric-label {{ font-size: 0.78rem; color: var(--text-muted); font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase; }}
        .metric-value {{ font-size: 1.7rem; font-weight: 800; color: var(--text); margin: 0.15rem 0; line-height: 1.2; }}
        .metric-sub {{ font-size: 0.78rem; color: var(--text-faint); }}

        /* Chips & badges */
        .chip {{
            display: inline-flex; align-items: center; gap: 5px;
            padding: 0.28rem 0.75rem; border-radius: 9999px;
            background: var(--surface-2); color: var(--text-muted);
            font-size: 0.82rem; font-weight: 500;
            border: 1px solid var(--border); margin: 0 0.28rem 0.38rem 0;
        }}
        .chip.primary {{ background: rgba(99,102,241,0.12); color: #6366F1; border-color: rgba(99,102,241,0.22); }}
        .badge {{ display: inline-block; padding: 0.22rem 0.62rem; border-radius: 7px; font-size: 0.78rem; font-weight: 600; }}

        /* Verdict hero */
        .verdict-hero {{
            direction: rtl; text-align: right;
            border-radius: var(--radius-lg); padding: 1.35rem 1.5rem;
            border: 1px solid; position: relative; overflow: hidden;
        }}
        .verdict-hero::after {{
            content: ''; position: absolute; inset: 0;
            background: radial-gradient(600px 200px at 90% -20%, rgba(99,102,241,0.08), transparent 60%);
            pointer-events: none;
        }}
        .verdict-label {{ font-size: 2.3rem; font-weight: 800; line-height: 1.2; }}
        .animated-bar-track {{ height: 10px; border-radius: 999px; background: var(--surface-2); margin-top: 0.9rem; overflow: hidden; position: relative; }}
        .animated-bar-fill {{ height: 100%; border-radius: 999px; transition: width 0.8s cubic-bezier(0.16,1,0.3,1); position: relative; overflow: hidden; }}
        .animated-bar-fill::after {{
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
            animation: shimmer 1.8s ease-in-out infinite;
        }}
        .pulse-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background: #10B981; animation: pulseDot 1.8s ease-in-out infinite; vertical-align: middle; margin-left: 6px; }}

        /* Theme toggle pill */
        .theme-pill {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 0.35rem 0.85rem; border-radius: 999px;
            background: var(--surface-2); border: 1px solid var(--border);
            font-size: 0.82rem; color: var(--text-muted); cursor: pointer;
        }}

        /* Buttons */
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            color: #fff !important; border: none !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.58rem 1.4rem !important; font-weight: 600 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        }}
        div.stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px) !important; box-shadow: 0 7px 22px rgba(99,102,241,0.42) !important;
        }}
        div.stButton > button[kind="secondary"] {{
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--border-strong) !important;
            background: var(--surface) !important; color: var(--text) !important;
            transition: all 0.18s ease !important;
        }}
        div.stButton > button[kind="secondary"]:hover {{
            border-color: var(--primary) !important; background: var(--surface-2) !important;
        }}

        /* DataFrame rounding */
        div[data-testid="stDataFrame"] {{ border-radius: var(--radius-md); overflow: hidden; border: 1px solid var(--border); }}

        /* Tabs */
        button[data-baseweb="tab"] {{ font-family: 'Vazirmatn', Tahoma, sans-serif !important; }}

        /* Empty state */
        .empty-state {{
            text-align: center; padding: 2.2rem 1.5rem;
            border: 1.5px dashed var(--border-strong);
            border-radius: var(--radius-lg); background: var(--surface);
            color: var(--text-muted);
        }}
        .empty-state .emoji {{ font-size: 2.2rem; margin-bottom: 0.4rem; }}

        /* Section divider */
        .section-divider {{ height: 1px; background: var(--border); margin: 1.2rem 0; border: none; }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{ background: var(--surface) !important; border-left: 1px solid var(--border); }}
        [data-testid="stSidebar"] .block-container {{ padding-top: 0.8rem !important; }}
        /* Sidebar widget labels: guaranteed contrast in BOTH themes */
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] label p {{ color: var(--text) !important; font-weight: 600; }}

        /* Hide Streamlit chrome cruft (menu, footer, cloud Deploy button) */
        #MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ display: none !important; }}

        {streamlit_overrides}
        </style>
        <div data-theme="{theme}" style="display:none;"></div>
        """,
        unsafe_allow_html=True,
    )


_inject_css = _inject_theme  # backward compat alias


# ==========================================================================
# Section 3: RTL and Safety Helpers
# ==========================================================================
# Latin digits / percent -> Persian digits / Persian percent sign, for display.
_FA_DIGIT_TABLE = str.maketrans("0123456789%", "۰۱۲۳۴۵۶۷۸۹٪")


def fa_num(text: Any) -> str:
    """Render digits (and %) of a value with Persian glyphs — display only.

    CSV exports, timestamps inside ``ltr()`` isolates and monospace technical
    values intentionally keep Latin digits.
    """
    return str(text).translate(_FA_DIGIT_TABLE)


def fa(text: str) -> str:
    """Escape user-supplied text before it enters an RTL HTML block."""
    return html.escape(str(text), quote=False)


def ltr(text: str) -> str:
    """Isolate an LTR run (date, filename) inside RTL prose with bidi isolates."""
    return f"\u2066{text}\u2069"


def rtl(text: str, muted: bool = False) -> None:
    """Render one RTL-safe Persian markdown block."""
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


@st.cache_data(show_spinner="در حال آماده‌سازی قلم فارسی...")
def ensure_font() -> tuple[str | None, str]:
    return _ensure_font()


@st.cache_data(show_spinner=False)
def cached_wordcloud_image(text: str, font_path: str | None, is_dark: bool = False) -> bytes | None:
    """Word-cloud as PNG bytes. Encoding here (instead of letting Streamlit
    marshal the PIL object) keeps the media identical across environments —
    the .venv install served a broken image through Streamlit's own PIL
    pipeline. Raw bytes also keep the element payload small so page switches
    stay fast."""
    image = build_wordcloud_image(text, font_path, is_dark=is_dark)
    if image is None:
        return None
    if image.width > 900:
        image = image.resize((900, round(image.height * 900 / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def get_cached_top_words(corpus_file: str, scope: str, limit: int = 200) -> Counter[str]:
    """Frequent words keyed by (corpus, sentiment scope) — hashing thousands of
    texts on every rerun re-introduced seconds of server time."""
    frame = load_cleaned(DATA_DIR / corpus_file)
    if frame is None:
        return Counter()
    labelled = frame[frame["label_name"].notna()]
    if scope != "همه":
        english = {v: k for k, v in LABEL_FA.items()}[scope]
        labelled = labelled[labelled["label_name"] == english]
    text_column = "eda_text" if "eda_text" in labelled.columns else "clean_text"
    return top_words(labelled[text_column].fillna("").astype(str), limit=limit)


@st.cache_data(show_spinner=False)
def get_cached_news_top_words(limit: int = 60) -> Counter[str]:
    news = load_news(NEWS_CSV)
    if news is None:
        return Counter()
    text_column = "body" if "body" in news.columns else "clean_text"
    return top_words(news[text_column].fillna("").astype(str), limit=limit)


# ==========================================================================
# Section 5: Data Loading and Predictor
# ==========================================================================
@st.cache_data(show_spinner="در حال بارگذاری پیکره...")
def load_cleaned(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "label_name" in frame:
        frame["label_name"] = frame["label_name"].astype("string")
    return frame


@st.cache_data(show_spinner="در حال بارگذاری نظرات خبری...")
def load_news(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


@st.cache_resource(show_spinner="در حال بارگذاری مدل ParsBERT...")
def get_predictor(model_dir: str) -> tuple[Any, str]:
    """Load one fine-tuned predictor per session, keyed by directory."""
    directory = Path(model_dir)
    if not directory.exists() or not (directory / "config.json").exists():
        return None, (
            f"چک‌پوینتی در {directory.name} یافت نشد. اجرا کنید: "
            "python -m src.train --csv data/cleaned_comments.csv"
        )
    try:
        from src.train import SentimentPredictor

        return SentimentPredictor(directory), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"خطا در بارگذاری مدل: {type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False, ttl=60)
def available_models() -> list[Path]:
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
    badge_html = (
        f'<span class="chip primary" style="font-weight:700;"><span class="pulse-dot"></span>{fa(badge)}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div style="margin-bottom: 1.15rem; direction: rtl; text-align: right;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">'
        f'<h1 class="gradient-title">{fa(title)}</h1>{badge_html}</div>'
        f'{"<div class=\"fa-block muted\" style=\"margin-top:0.35rem;\">" + fa(subtitle) + "</div>" if subtitle else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, subtext: str = "", border_color: str = "#6366F1", icon: str = "") -> str:
    icon_html = f'<span style="font-size:1.25rem;">{icon}</span>' if icon else ""
    return (
        f'<div class="metric-card" style="--metric-accent:{border_color}; direction:rtl; text-align:right;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between;">'
        f'<span class="metric-label">{fa(label)}</span>{icon_html}</div>'
        f'<div class="metric-value">{fa(fa_num(value))}</div><div class="metric-sub">{fa(fa_num(subtext))}</div></div>'
    )


def sentiment_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#64748B")
    fa_name = LABEL_FA.get(label, label)
    emoji = _EMOJI_MAP.get(label, "•")
    return f'<span class="badge" style="background:{color}18; color:{color}; border:1px solid {color}44;">{emoji} {fa(fa_name)}</span>'


def empty_state(emoji: str, title: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="empty-state fade-in" dir="rtl">'
        f'<div class="emoji">{emoji}</div>'
        f'<div style="font-weight:700; color:var(--text); font-size:1.05rem;">{fa(title)}</div>'
        f'{"<div style=\"margin-top:0.35rem; font-size:0.9rem; color:var(--text-muted);\">" + fa(hint) + "</div>" if hint else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# Section 7: Plotly Charts
# ==========================================================================
def _probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """Horizontal per-class probabilities — keeps the 3-class padded-range contract."""
    names = [LABEL_FA.get(name, name) for name in probabilities]
    values = list(probabilities.values())
    max_val = max(values) if values else 1.0
    # Theme-aware grid / text
    is_dark = _get_theme() == "dark"
    grid = "#24304E" if is_dark else "#F1F5F9"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    figure = go.Figure(
        go.Bar(
            x=values, y=names, orientation="h",
            marker={"color": [LABEL_COLORS.get(n, "#888888") for n in probabilities], "line": {"width": 0}},
            text=[f"{v*100:.1f}%" for v in values], textposition="outside",
            textfont={"color": text_col, "size": 12},
            hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white" if not is_dark else "plotly_dark",
        height=215, margin={"l": 10, "r": 50, "t": 10, "b": 10},
        xaxis={"range": [0, max(max_val * 1.35, 0.75)], "visible": False},
        yaxis={"autorange": "reversed", "tickfont": {"size": 13, "family": "Vazirmatn, Tahoma, sans-serif", "color": text_col}, "gridcolor": grid},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Vazirmatn, Tahoma, sans-serif", "color": text_col},
    )
    return figure


def gauge_confidence_chart(confidence: float, label: str) -> go.Figure:
    color = LABEL_COLORS.get(label, "#6366F1")
    is_dark = _get_theme() == "dark"
    tick_col = "#94A3B8" if is_dark else "#CBD5E1"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            number={"suffix": "%", "font": {"size": 30, "color": color, "family": "JetBrains Mono, Tahoma, sans-serif"}},
            title={"text": "درجه اطمینان مدل", "font": {"size": 13, "color": "#94A3B8", "family": "Vazirmatn, Tahoma, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": tick_col},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(241,245,249,0.0)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(226,232,240,0.35)"},
                    {"range": [50, 80], "color": "rgba(203,213,225,0.4)"},
                    {"range": [80, 100], "color": "rgba(148,163,184,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=215, margin={"l": 20, "r": 20, "t": 35, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": text_col})
    return fig


def sentiment_pie(frame: pd.DataFrame) -> go.Figure:
    counts = frame["label_name"].value_counts().reindex(LABEL_ORDER).fillna(0)
    is_dark = _get_theme() == "dark"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    figure = go.Figure(
        go.Pie(
            labels=[f"{LABEL_FA.get(n, n)} ({n})" for n in counts.index],
            values=counts.values.tolist(), hole=0.52,
            marker={"colors": [LABEL_COLORS.get(n, "#888888") for n in counts.index], "line": {"color": "#FFFFFF" if not is_dark else "#1E293B", "width": 2}},
            textinfo="label+percent", textposition="inside",
            textfont={"family": "Vazirmatn, Tahoma, sans-serif", "size": 12},
            hovertemplate="<b>%{label}</b><br>تعداد: %{value:,}<br>سهم: %{percent}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white" if not is_dark else "plotly_dark",
        font={"family": "Vazirmatn, Tahoma, sans-serif", "color": text_col},
        height=320, margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def top_words_bar(words: Counter[str], limit: int = 25) -> go.Figure:
    pairs = words.most_common(limit)[::-1]
    is_dark = _get_theme() == "dark"
    grid = "#24304E" if is_dark else "#F1F5F9"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    figure = go.Figure(
        go.Bar(
            x=[count for _, count in pairs], y=[word for word, _ in pairs], orientation="h",
            marker={"color": [count for _, count in pairs], "colorscale": [[0, "#818CF8"], [1, "#4338CA"]], "line": {"width": 0}},
            text=[f"{count:,}" for _, count in pairs], textposition="outside",
            textfont={"color": text_col, "size": 11},
            hovertemplate="<b>%{y}</b>: %{x:,} تکرار<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white" if not is_dark else "plotly_dark",
        font={"family": "Vazirmatn, Tahoma, sans-serif", "color": text_col},
        height=max(360, limit * 20), margin={"l": 10, "r": 50, "t": 15, "b": 15},
        xaxis={"title": "تعداد تکرار", "gridcolor": grid}, yaxis={"autorange": True, "tickfont": {"size": 11}, "gridcolor": grid},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def length_histogram(frame: pd.DataFrame) -> go.Figure:
    lengths = pd.to_numeric(frame["n_tokens"], errors="coerce").dropna()
    mean_val = float(lengths.mean()) if not lengths.empty else 0.0
    is_dark = _get_theme() == "dark"
    grid = "#24304E" if is_dark else "#F1F5F9"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    figure = go.Figure(
        go.Histogram(x=lengths, nbinsx=40, marker={"color": "#6366F1", "line": {"color": "#4338CA", "width": 1}}, hovertemplate="بازهٔ توکن: %{x}<br>تعداد نظرات: %{y}<extra></extra>")
    )
    figure.add_vline(x=mean_val, line_dash="dash", line_color="#EF4444", annotation_text=f"میانگین: {mean_val:.1f}", annotation_position="top right")
    figure.update_layout(
        template="plotly_white" if not is_dark else "plotly_dark",
        font={"family": "Vazirmatn, Tahoma, sans-serif", "color": text_col},
        height=320, margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"title": "تعداد توکن در هر نظر", "gridcolor": grid}, yaxis={"title": "فراوانی نظرات", "gridcolor": grid},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def per_class_bar(metrics: dict[str, Any]) -> go.Figure:
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
    is_dark = _get_theme() == "dark"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    figure = px.bar(frame, x="class", y="score", color="metric", barmode="group", range_y=(0, 1.05), color_discrete_sequence=["#6366F1", "#06B6D4", "#10B981"], template="plotly_dark" if is_dark else "plotly_white")
    figure.update_layout(font={"family": "Vazirmatn, Tahoma, sans-serif", "color": text_col}, height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="کلاس احساسی", yaxis_title="امتیاز (۰ تا ۱)", legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def confusion_heatmap(metrics: dict[str, Any]) -> go.Figure:
    test = metrics.get("test") or metrics.get("best_val") or {}
    matrix = test.get("confusion_matrix")
    if not matrix:
        return go.Figure()
    is_dark = _get_theme() == "dark"
    labels = test.get("labels", LABEL_ORDER)
    names = [LABEL_FA.get(label, label) for label in labels]
    figure = go.Figure(go.Heatmap(z=matrix, x=names, y=names, colorscale=[[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#312E81"]], text=matrix, texttemplate="<b>%{text}</b>", textfont={"size": 14, "color": "#E2E8F0" if is_dark else "black"}, hovertemplate="واقعی: %{y}<br>پیش‌بینی: %{x}<br>تعداد: %{z}<extra></extra>"))
    figure.update_layout(template="plotly_dark" if is_dark else "plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif"}, height=340, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="پیش‌بینی مدل (Predicted)", yaxis_title="برچسب واقعی (Actual)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def training_history_chart(history: list[dict[str, Any]]) -> go.Figure:
    if not history:
        return go.Figure()
    is_dark = _get_theme() == "dark"
    grid = "#24304E" if is_dark else "#F1F5F9"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    epochs = [h.get("epoch", idx + 1) for idx, h in enumerate(history)]
    fig = go.Figure()
    traces = [
        ("Train Loss (خطای آموزش)", [h.get("train_loss", 0.0) for h in history], "#EF4444", 3, "solid"),
        ("Val Macro-F1", [h.get("val_macro_f1", 0.0) for h in history], "#6366F1", 3, "solid"),
        ("Val Accuracy", [h.get("val_accuracy", 0.0) for h in history], "#10B981", 2, "dot"),
    ]
    for name, vals, col, w, dash in traces:
        fig.add_trace(go.Scatter(x=epochs, y=vals, name=name, mode="lines+markers", line={"color": col, "width": w, "dash": dash}, marker={"size": 6}))
    fig.update_layout(template="plotly_white" if not is_dark else "plotly_dark", font={"family": "Vazirmatn, Tahoma, sans-serif", "color": text_col}, height=320, margin={"l": 20, "r": 20, "t": 20, "b": 20}, xaxis={"title": "دوره (Epoch)", "tickmode": "linear", "dtick": 1, "gridcolor": grid}, yaxis={"title": "مقدار معیار", "gridcolor": grid}, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ==========================================================================
# Section 8: Page — Live Prediction (Hero)
# ==========================================================================
_EMOJI_MAP: dict[str, str] = {"positive": "😊", "neutral": "😐", "negative": "😤"}

EXAMPLES: tuple[tuple[str, str], ...] = (
    ("positive", "کیفیتش عالی بود و کاملاً راضی‌ام، ارسالش هم سریع بود"),
    ("negative", "به‌شدت بد بود و اصلاً ارزش خرید ندارد"),
    ("neutral", "معمولی است؛ نه خیلی خوب نه خیلی بد"),
)


def _confidence_level(confidence: float) -> tuple[str, str]:
    """Human-readable confidence tier + its colour (shared by card and captions)."""
    if confidence >= 0.90:
        return "اطمینان بسیار بالا", "#10B981"
    if confidence >= 0.75:
        return "اطمینان بالا", "#059669"
    if confidence >= 0.55:
        return "اطمینان متوسط", "#F59E0B"
    return "اطمینان پایین — احتمال خط بالاست", "#EF4444"


def _verdict_card(label: str, label_fa: str, confidence: float) -> None:
    emoji = _EMOJI_MAP.get(label, "🔍")
    colour = LABEL_COLORS.get(label, "#5B6B7F")
    conf_pct = confidence * 100
    level_text, level_colour = _confidence_level(confidence)
    st.markdown(
        f'<div class="verdict-hero fade-in" style="border-color:{colour}55; background: linear-gradient(135deg, {colour}14 0%, var(--surface) 100%); box-shadow: 0 12px 30px -8px {colour}22;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.5rem;">'
        f'<span style="font-size:3rem; line-height:1; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.14));">{emoji}</span>'
        f'<span class="badge" style="background:{colour}20; color:{colour}; border:1px solid {colour}40; font-weight:700; font-size:0.85rem; border-radius:999px;">{fa(label_fa)}</span></div>'
        f'<div class="verdict-label" style="color:{colour};">{fa(label_fa)}</div>'
        f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:0.85rem; direction:rtl; text-align:right;">'
        f'<span style="color:var(--text-muted); font-size:0.95rem; font-weight:600;">درجه اطمینان مدل</span>'
        f'<span style="color:{colour}; font-size:1.55rem; font-weight:800; font-family: JetBrains Mono, monospace;">{fa_num(f"{conf_pct:.1f}")}%</span></div>'
        f'<div class="animated-bar-track"><div class="animated-bar-fill" style="width:{conf_pct:.1f}%; background: linear-gradient(90deg, {colour} 0%, {colour}dd 100%);"></div></div>'
        f'<div style="margin-top:10px;"><span class="badge" style="background:{level_colour}18; color:{level_colour}; border:1px solid {level_colour}40;">{fa(level_text)}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _chip(label: str, value: str) -> str:
    """One status chip: RTL label with the value as a directional island."""
    direction = ' dir="ltr"' if value.isascii() else ""
    return f'<span class="chip">{fa(label)}: <b{direction}>{fa(value)}</b></span>'


def _fa_label(pred: dict[str, Any]) -> str:
    """Persian class name for a prediction dict (checkpoints may return English)."""
    return LABEL_FA.get(pred.get("label"), pred.get("label_fa") or "")


def _model_status(model_dir: Path) -> None:
    metrics = load_metrics(model_dir)
    meta_path = model_dir / "training_meta.json"
    base_name = "ParsBERT"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_name = Path(str(meta.get("base_model", ""))).name or base_name
        except (json.JSONDecodeError, OSError):
            pass
    chips: list[tuple[str, str]] = [("چک‌پوینت", model_dir.name), ("پایه", base_name)]
    if metrics:
        score = (metrics.get("test") or metrics.get("best_val") or {}).get("macro_f1")
        if isinstance(score, (int, float)):
            chips.append(("macro-F1", f"{score:.3f}"))
        if metrics.get("train_rows"):
            chips.append(("نمونهٔ آموزش", fa_num(f"{metrics['train_rows']:,}")))
    chips_html = "".join(_chip(label, value) for label, value in chips)
    st.markdown(f'<div style="margin-bottom:0.55rem; direction:rtl;">{chips_html}</div>', unsafe_allow_html=True)
    if metrics:
        updated = datetime.fromtimestamp((model_dir / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        st.markdown(
            '<div class="fa-block faint fa-lines" style="line-height:1.9;">'
            f'<div>آخرین به‌روزرسانی معیارها: {ltr(updated)}</div>'
            '<div>اگر آموزشی در جریان باشد، این اعداد مربوط به پیش از پایان آن است.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    if st.button("🔄 بازخوانی مدل", key="reload_model_btn", help="وزن‌های تازهٔ آموزش را جای وزن‌های کش‌شده می‌نشاند"):
        get_predictor.clear()
        st.rerun()


def _use_example(example: str) -> None:
    st.session_state["user_text"] = example


def _clear_panel() -> None:
    st.session_state["user_text"] = ""
    st.session_state.pop("prediction", None)


def page_prediction(model_dir: Path) -> None:
    gradient_header(
        "پیش‌بینی زندهٔ احساسات فارسی",
        "متن دلخواه یا نظر مشتری را بنویسید؛ مدل سه‌کلاسهٔ ParsBERT آن را به مثبت / خنثی / منفی "
        "طبقه‌بندی کرده و خط لوله نرمال‌سازی hazm را به تصویر می‌کشد.",
        badge="استنتاج برخط",
    )
    _model_status(model_dir)

    st.markdown('<div style="margin-top:0.8rem; margin-bottom:0.3rem; font-weight:600; color:var(--text-muted); direction:rtl; text-align:right;">نمونه‌های آماده برای بررسی سریع:</div>', unsafe_allow_html=True)
    pills = st.columns(len(EXAMPLES))
    for column, (tone, example) in zip(pills, EXAMPLES):
        with column:
            st.button(f"{_EMOJI_MAP.get(tone,'')} نمونهٔ {LABEL_FA[tone]}", key=f"example_{tone}", help=example, on_click=_use_example, args=(example,), use_container_width=True)

    text = st.text_area("متن نظر (فارسی)", height=145, key="user_text", placeholder="مثلاً: کیفیت ساخت دستگاه بسیار بالاست و عملکرد روانی دارد، فقط ارسال با دو روز تأخیر انجام شد.")
    stripped = (text or "").strip()
    char_count = len(stripped)
    word_count = len(stripped.split())
    # Live counts — shown as chips
    st.markdown(
        f'<div style="display:flex; justify-content:flex-start; gap:8px; margin-top:-6px; margin-bottom:12px; direction:rtl;">'
        f'<span class="chip" style="font-size:0.78rem;">📝 <b>{fa_num(char_count)}</b> نویسه</span>'
        f'<span class="chip" style="font-size:0.78rem;">🔤 <b>{fa_num(word_count)}</b> واژه</span>'
        f'{"<span class=\"chip\" style=\"font-size:0.78rem; background:#FEF2F2; color:#DC2626; border-color:#FECACA;\">⚠️ متن کوتاه</span>" if 0 < word_count < 3 else ""}'
        f'</div>',
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
        # Some checkpoints return an English label_fa — always show the Persian name
        result["label_fa"] = LABEL_FA.get(result["label"], result.get("label_fa", ""))
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["input"] = stripped
        st.session_state["prediction"] = result
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].insert(0, {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "text": stripped, "label": result["label"], "label_fa": result["label_fa"],
            "confidence": result["confidence"], "elapsed_ms": result["elapsed_ms"],
        })
        st.session_state["history"] = st.session_state["history"][:10]
        if result.get("label") and result["confidence"] > 0.88:
            try:
                st.toast(f"پیش‌بینی: {result['label_fa']} ({result['confidence']*100:.0f}٪ اطمینان)", icon="✅")
            except Exception:
                pass

    prediction = st.session_state.get("prediction")
    if prediction is None:
        empty_state("💬", "هنوز متنی پیش‌بینی نشده است", "یکی از نمونه‌های بالا را انتخاب کنید یا متن دلخواه خود را بنویسید.")
        return
    if prediction["label"] is None:
        st.warning(prediction["note"] or "متن قابل پردازش نبود.")
        return

    col_verdict, col_charts = st.columns([1.1, 1.4])
    with col_verdict:
        _verdict_card(prediction["label"], prediction["label_fa"], prediction["confidence"])
        st.markdown(
            '<div class="fa-block faint fa-lines" style="margin-top:0.4rem;">'
            f'<div>⏱ زمان استنتاج: {fa_num(prediction["elapsed_ms"])} میلی‌ثانیه روی CPU</div>'
            f'<div>🧩 مدل: {ltr(model_dir.name)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_charts:
        tab_prob, tab_gauge = st.tabs(["📊 توزیع احتمالات", "🎯 سنجه اطمینان"])
        with tab_prob:
            st.plotly_chart(_probability_chart(prediction["probabilities"]), width="stretch")
        with tab_gauge:
            st.plotly_chart(gauge_confidence_chart(prediction["confidence"], prediction["label"]), width="stretch")

    cleaned = prediction["cleaned"]
    with st.expander("🔍 مقایسه متن خام و متن نرمال‌شده (hazm)", expanded=True):
        col_raw, col_clean = st.columns(2)
        with col_raw:
            st.markdown(
                f'<div class="glass-card fa-block" dir="rtl" style="min-height:112px; border-top:3px solid #64748B;">'
                f'<div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span style="font-weight:700; color:var(--text-muted);">متن ورودی خام</span>'
                f'<span class="badge" style="background:var(--surface-2); color:var(--text-muted);">{len(prediction["input"])} نویسه</span></div>'
                f'<div style="color:var(--text); word-break:break-word;">{fa(prediction["input"])}</div></div>',
                unsafe_allow_html=True,
            )
        with col_clean:
            st.markdown(
                f'<div class="glass-card fa-block" dir="rtl" style="min-height:112px; border-top:3px solid #6366F1;">'
                f'<div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span style="font-weight:700; color:#6366F1;">متن نرمال‌شده (hazm)</span>'
                f'<span class="badge" style="background:#EEF2FF; color:#4338CA;">{len(cleaned)} نویسه</span></div>'
                f'<div style="color:var(--text); word-break:break-word;">{fa(cleaned)}</div></div>',
                unsafe_allow_html=True,
            )
        remaining_zwnj = cleaned.count(chr(0x200C))
        if remaining_zwnj == 0:
            st.markdown(
                '<div class="glass-card" style="margin-top:10px; padding:0.75rem 1rem; border-right:4px solid #10B981; background:#ECFDF5; direction:rtl; text-align:right;">'
                '<span style="color:#059669; font-weight:700; margin-left:6px;">✓ نیم‌فاصله (ZWNJ):</span>'
                '<span style="color:#065F46; font-size:0.92rem;">صفر نیم‌فاصله باقی‌مانده — تبدیل ZWNJ به فاصله با موفقیت اجرا شد.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="glass-card" style="margin-top:10px; padding:0.75rem 1rem; border-right:4px solid #EF4444; background:#FEF2F2; direction:rtl; text-align:right;">'
                f'<span style="color:#DC2626; font-weight:700; margin-left:6px;">✗ هشدار نیم‌فاصله:</span>'
                f'<span style="color:#991B1B; font-size:0.92rem;">{remaining_zwnj} مورد نیم‌فاصله در ورودی مدل باقی مانده است.</span></div>',
                unsafe_allow_html=True,
            )

    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"🕒 تاریخچه این نشست ({len(history)} مورد اخیر)", expanded=False):
            h_left, h_right = st.columns([4, 1])
            with h_right:
                if st.button("پاک کردن تاریخچه", key="clear_history_btn", use_container_width=True):
                    st.session_state["history"] = []
                    st.rerun()
            parts = []
            for item in history:
                color = LABEL_COLORS.get(item["label"], "#64748B")
                parts.append(
                    f'<div class="glass-card" style="margin-bottom:8px; padding:0.7rem 1rem; border-right:4px solid {color}; direction:rtl; text-align:right;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                    f'<span style="font-weight:700; color:{color}; font-size:0.95rem;">'
                    f'<span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:{color}; margin-left:6px;"></span>'
                    f'{fa(item["label_fa"])} ({item["confidence"]*100:.1f}%)</span>'
                    f'<span dir="ltr" style="font-size:0.8rem; color:var(--text-faint); font-family: JetBrains Mono, monospace;">{item["timestamp"]}&emsp;{item["elapsed_ms"]}ms</span></div>'
                    f'<div style="font-size:0.9rem; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{fa(item["text"])}</div></div>'
                )
            st.markdown("".join(parts), unsafe_allow_html=True)


# ==========================================================================
# Section 9: Page — Dashboard
# ==========================================================================
def page_dashboard(frame: pd.DataFrame | None, news: pd.DataFrame | None, model_dir: Path, corpus_name: str) -> None:
    gradient_header(
        "داشبورد تحلیلی و آماری احساسات",
        "نمای یکپارچه از پیکرهٔ فارسی دیجی‌کالا، واژگان پرتکرار، ابر کلمات و ارزیابی جامع "
        "مدل عمیق ParsBERT همراه با ماتریس درهم‌ریختگی و نمودارهای دورهٔ آموزش.",
        badge=f"پیکره: {corpus_name}",
    )
    if frame is None:
        st.warning("هیچ پیکرهٔ پاک‌سازی‌شده‌ای یافت نشد. مراحل زیر را اجرا کنید:\n\n```bash\npython -m src.scraper --source digikala --out data/raw_digikala_comments.csv\npython -m src.preprocess --in data/raw_digikala_comments.csv --out data/cleaned_comments.csv\npython -m src.train --csv data/cleaned_comments.csv --epochs 3\npython -m src.visualize\n```")
        return

    labelled = frame[frame["label_name"].notna()]
    metrics = load_metrics(model_dir)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(metric_card("کل نظرات", f"{len(frame):,}", "پیکره انتخاب‌شده", "#6366F1", "💬"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("نظرات برچسب‌دار", f"{len(labelled):,}", "سه کلاس احساسی", "#10B981", "🏷️"), unsafe_allow_html=True)
    with m3:
        prod_count = int(pd.to_numeric(frame.get("product_id"), errors="coerce").nunique()) if "product_id" in frame.columns else "n/a"
        st.markdown(metric_card("تنوع کالاها", f"{prod_count:,}" if isinstance(prod_count, int) else prod_count, "دیجی‌کالا", "#06B6D4", "📦"), unsafe_allow_html=True)
    with m4:
        mean_tok = float(pd.to_numeric(labelled.get("n_tokens"), errors="coerce").mean()) if not labelled.empty else 0.0
        st.markdown(metric_card("میانگین توکن‌ها", f"{mean_tok:.1f}", "طول هر نظر", "#F59E0B", "📏"), unsafe_allow_html=True)
    with m5:
        macro_f1 = f"{metrics['test']['macro_f1']:.3f}" if metrics and metrics.get("test") and "macro_f1" in metrics["test"] else "n/a"
        st.markdown(metric_card("Macro-F1 آزمون", macro_f1, "معیار اصلی", "#8B5CF6", "🏆"), unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    # Radio-based sections instead of st.tabs: tabs stream EVERY section's
    # charts and tables on each rerun (even hidden ones), which drags the rerun
    # out for seconds and keeps the previous page's elements on screen during
    # page switches. Only the active section is rendered now.
    section = st.radio(
        "بخش داشبورد:",
        ["📊 پیکره و توزیع داده‌ها", "🧠 ارزیابی مدل عمیق", "📰 نظرات خبری (بدون برچسب)"],
        horizontal=True,
        key="dash_section",
    )
    st.markdown('<div style="margin-top:0.6rem;"></div>', unsafe_allow_html=True)

    if section == "📊 پیکره و توزیع داده‌ها":
        mix_left, mix_right = st.columns([1, 1])
        with mix_left:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">توزیع فراوانی احساسات</div>', unsafe_allow_html=True)
            st.plotly_chart(sentiment_pie(labelled), width="stretch")
        with mix_right:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">توزیع طول نظرات (توکن)</div>', unsafe_allow_html=True)
            st.plotly_chart(length_histogram(labelled), width="stretch")

        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        scope = st.radio("فیلتر واژگان بر اساس احساس:", ["همه", *LABEL_FA.values()], horizontal=True)
        if scope == "همه":
            subset = labelled
        else:
            english = {v: k for k, v in LABEL_FA.items()}[scope]
            subset = labelled[labelled["label_name"] == english]
        text_column = "eda_text" if "eda_text" in subset.columns else "clean_text"
        words = get_cached_top_words(corpus_name, scope, limit=200)
        is_dark = _get_theme() == "dark"
        chart_left, chart_right = st.columns([1, 1])
        with chart_left:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">پرکاربردترین واژه‌ها</div>', unsafe_allow_html=True)
            st.plotly_chart(top_words_bar(words, limit=20), width="stretch")
        with chart_right:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">ابر کلمات فارسی (Word Cloud)</div>', unsafe_allow_html=True)
            font_path, font_note = ensure_font()
            if words:
                png_bytes = cached_wordcloud_image(word_cloud_text(words), font_path, is_dark=is_dark)
                if png_bytes:
                    # Inline data-URI instead of st.image: the /media pipeline
                    # crashes under some installs (gzip middleware ASGI error,
                    # broken image), the inline form serves everywhere.
                    png_b64 = base64.b64encode(png_bytes).decode("ascii")
                    st.markdown(
                        f'<div dir="rtl" style="text-align:center;">'
                        f'<img src="data:image/png;base64,{png_b64}" alt="ابر کلمات فارسی" '
                        'style="max-width:100%; height:auto; border-radius:12px;" /></div>',
                        unsafe_allow_html=True,
                    )
            st.caption(f"قلم نمایشی: {font_note}")

        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:6px;">پیش‌نمایش تعاملی نمونه‌ای از پیکره</div>', unsafe_allow_html=True)
        preview_cols = [c for c in ["raw_text", "clean_text", "label_name", "rate", "split"] if c in labelled.columns]
        # Sample FIRST, then truncate — slicing 6402 long texts on every rerun
        # costs server time that keeps stale page elements alive during switches
        preview = labelled[preview_cols].sample(min(150, len(labelled)), random_state=42).copy()
        for col in preview.columns:
            if preview[col].dtype == "object" or str(preview[col].dtype) == "string":
                preview[col] = preview[col].astype(str).str.slice(0, 140)
        rename_map = {"raw_text": "متن خام", "clean_text": "متن تمیزشده", "label_name": "برچسب احساس", "rate": "امتیاز (ستاره)", "split": "بخش داده"}
        st.dataframe(preview.rename(columns=rename_map), width="stretch", height=320)

    elif section == "🧠 ارزیابی مدل عمیق":
        if metrics is None:
            st.info("فایل metrics.json برای این مدل یافت نشد — با `python -m src.train` مدل را آموزش دهید.")
        else:
            download = metrics.get("download", {})
            m_base = fa(str(metrics.get("base_model", "ParsBERT")))
            m_train = fa_num(f"{metrics.get('train_rows', 0):,}")
            m_val = fa_num(f"{metrics.get('val_rows', 0):,}")
            m_test = fa_num(f"{metrics.get('test_rows', 0):,}")
            m_minutes = fa_num(metrics.get("total_minutes", "n/a"))
            m_device = fa(str(metrics.get("device", "cpu")))
            dl_mb = fa_num(download.get("mb", 0))
            dl_sec = fa_num(download.get("seconds", 0))
            dl_src = fa(str(download.get("source", "direct")))
            meta_rows = [
                ("مدل پایه", f"<code>{m_base}</code>"),
                ("نمونه‌های آموزش", f"<b>{m_train}</b>"),
                ("نمونه‌های اعتبارسنجی", f"<b>{m_val}</b>"),
                ("نمونه‌های آزمون", f"<b>{m_test}</b>"),
                ("مدت آموزش", f"<b>{m_minutes}</b> دقیقه روی <code>{m_device}</code>"),
                ("دانلود وزن‌ها", f"<b>{dl_mb}</b> مگابایت در <b>{dl_sec}</b> ثانیه ({dl_src})"),
            ]
            rows_html = "".join(f"<div>{fa(label)}: {value}</div>" for label, value in meta_rows)
            st.markdown(
                '<div class="glass-card" style="margin-bottom:1rem; border-right:4px solid #6366F1; direction:rtl; text-align:right;">'
                '<div style="font-weight:700; color:var(--text); margin-bottom:6px;">مشخصات و متادیتای آموزش مدل</div>'
                f'<div class="fa-lines" style="color:var(--text-muted); font-size:0.92rem; line-height:1.85;">{rows_html}</div></div>',
                unsafe_allow_html=True,
            )
            best, test = metrics.get("best_val") or {}, metrics.get("test") or {}
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card("دقت اعتبارسنجی (Val Acc)", f"{best.get('accuracy', float('nan')):.3f}", "بهترین اپوک", "#6366F1"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Macro-F1 اعتبارسنجی", f"{best.get('macro_f1', float('nan')):.3f}", "توازن کلاس‌ها", "#06B6D4"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("دقت آزمون (Test Acc)", f"{test.get('accuracy', float('nan')):.3f}", "داده ارزیابی نهایی", "#10B981"), unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("Macro-F1 آزمون نهایی", f"{test.get('macro_f1', float('nan')):.3f}", "معیار قطعی مدل", "#8B5CF6"), unsafe_allow_html=True)
            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
            eval_left, eval_right = st.columns([1, 1])
            with eval_left:
                st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">دقت، یادآوری و F1 به تفکیک کلاس</div>', unsafe_allow_html=True)
                st.plotly_chart(per_class_bar(metrics), width="stretch")
            with eval_right:
                st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">ماتریس درهم‌ریختگی (Confusion Matrix)</div>', unsafe_allow_html=True)
                st.plotly_chart(confusion_heatmap(metrics), width="stretch")
            history_data = metrics.get("history", [])
            if history_data:
                st.markdown('<div class="fa-block" style="font-weight:700; margin-top:1rem; margin-bottom:4px;">روند کاهش خطا و بهبود F1 در طول اپوک‌ها</div>', unsafe_allow_html=True)
                st.plotly_chart(training_history_chart(history_data), width="stretch")
                with st.expander("جدول جزئیات عددی سابقه آموزش", expanded=False):
                    st.dataframe(pd.DataFrame(history_data), width="stretch")
            neutral = (test.get("per_class", {}) or {}).get("neutral", {})
            if isinstance(neutral, dict) and neutral.get("recall", 1.0) < 0.4:
                st.markdown(
                    '<div class="glass-card" style="margin-top:1rem; border-right:4px solid #F59E0B; background:#FFFBEB; direction:rtl; text-align:right;">'
                    '<div style="color:#B45309; font-weight:700; margin-bottom:6px;">⚠️ تحلیل یادآوری کلاس خنثی</div>'
                    '<div class="fa-lines" style="color:#92400E; font-size:0.92rem; line-height:1.85;">'
                    '<div>میزان Recall کلاس خنثی پایین‌تر از سایر کلاس‌هاست.</div>'
                    '<div>علت فنی: برچسب‌ها از ستارهٔ ۱ تا ۵ فروشگاه استخراج شده‌اند و امتیاز ۳ ستاره در نظرات واقعی کاربران به‌شدت نادر است (حدود ۱۴ برابر کمتر از ۵ ستاره).</div>'
                    '<div>به همین دلیل، Macro-F1 معیار اصلی ارزیابی در نظر گرفته شده است.</div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

    else:
        if news is None:
            st.info("پیکرهٔ نظرات خبری یافت نشد — با `python -m src.scraper --source news --out data/raw_news_comments.csv` جمع‌آوری کنید.")
        else:
            rtl(f"{len(news):,} نظر جمع‌آوری‌شده از سایت‌های خبری (ورزش۳). این نظرات بدون امتیاز ستاره هستند و به عنوان دادهٔ خارج از دامنه برای تحلیل زبان‌شناختی کاربرد دارند.", muted=True)
            votes = pd.to_numeric(news.get("likes"), errors="coerce").fillna(0)
            dislikes = pd.to_numeric(news.get("dislikes"), errors="coerce").fillna(0)
            n1, n2, n3 = st.columns(3)
            with n1:
                st.markdown(metric_card("تعداد خبرها", f"{news.get('news_id', pd.Series()).nunique():,}", "منبع ورزش۳", "#6366F1", "📰"), unsafe_allow_html=True)
            with n2:
                st.markdown(metric_card("مجموع پسندها", f"{int(votes.sum()):,}", "تعامل مثبت", "#10B981", "👍"), unsafe_allow_html=True)
            with n3:
                st.markdown(metric_card("مجموع ناپسندها", f"{int(dislikes.sum()):,}", "تعامل منفی", "#EF4444", "👎"), unsafe_allow_html=True)
            body_column = "body" if "body" in news.columns else "clean_text"
            words = get_cached_news_top_words(limit=60)
            st.markdown('<div class="fa-block" style="font-weight:700; margin-top:1rem; margin-bottom:4px;">واژگان شاخص نظرات ورزشی</div>', unsafe_allow_html=True)
            st.plotly_chart(top_words_bar(words, limit=18), width="stretch")
            news_cols = [c for c in ("comment_id", "news_title", "body", "likes", "dislikes", "created_at") if c in news.columns]
            news_view = news[news_cols].head(60).copy()
            for col in ("news_title", "body"):
                if col in news_view.columns:
                    news_view[col] = news_view[col].astype(str).str.slice(0, 140)
            st.dataframe(news_view, width="stretch", height=280)
            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
            if st.button("🔮 پیش‌بینی احساسات ۲۰ نظر اخیر خبری با ParsBERT", key="predict_news_btn"):
                predictor, message = get_predictor(str(model_dir))
                if predictor is None:
                    st.error(message)
                else:
                    sample = news.dropna(subset=[body_column]).head(20)
                    with st.spinner("در حال استنتاج روی نظرات خبری..."):
                        bodies = sample[body_column].astype(str).tolist()
                        # Batched path when available
                        if hasattr(predictor, "predict_batch"):
                            preds = predictor.predict_batch(bodies)  # type: ignore
                        else:
                            preds = [predictor.predict(t) for t in bodies]
                        results = [
                            {"متن نظر": str(row)[:110] + "...", "احساس پیش‌بینی‌شده": _fa_label(p), "کلاس": p["label"], "درجه اطمینان": f"{p['confidence']*100:.1f}%"}
                            for row, p in zip(sample[body_column], preds)
                        ]
                        st.dataframe(pd.DataFrame(results), width="stretch")


# ==========================================================================
# Section 10: Page — Batch Prediction (optimised: batched inference)
# ==========================================================================
def page_batch(model_dir: Path) -> None:
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
        input_text = st.text_area("نظرات را وارد کنید (هر نظر در یک سطر):", value=st.session_state.get("batch_paste_area", ""), height=180, key="batch_paste_input", placeholder="هر سطر نمایانگر یک نظر است...")
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
            # pandas 3.0 gives CSV string columns the new `str` dtype — testing
            # only "object" (pandas 2.x) left text_candidates empty and crashed
            # the page with KeyError: None on the column selectbox.
            text_candidates = [
                c for c in df_upload.columns
                if df_upload[c].dtype == "object" or str(df_upload[c].dtype) in ("str", "string")
            ]
            default_idx = 0
            for idx, c in enumerate(text_candidates):
                if any(k in c.lower() for k in ["body", "comment", "text", "clean", "متن", "نظر"]):
                    default_idx = idx
                    break
            selected_col = st.selectbox("ستون متن نظرات:", text_candidates, index=default_idx)
            max_rows = min(len(df_upload), 200)
            sample_size = st.slider("حداکثر تعداد نظرات برای پردازش:", min_value=1, max_value=max_rows, value=min(max_rows, 50), help="جهت حفظ سرعت روی CPU، تعداد قابل تنظیم است.")
            texts_to_predict = df_upload[selected_col].dropna().astype(str).head(sample_size).tolist()

    st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
    run_batch = st.button("🚀 اجرای پیش‌بینی دسته‌ای", type="primary", disabled=not texts_to_predict)

    if run_batch and texts_to_predict:
        total = len(texts_to_predict)
        progress_bar = st.progress(0, text="در حال استنتاج دسته‌ای با ParsBERT...")
        started = time.perf_counter()

        # Optimised path: single batched forward pass per 32 items
        use_batch = hasattr(predictor, "predict_batch")
        if use_batch:
            try:
                # Chunked to keep progress bar alive for large batches
                chunk_size = 32
                all_preds: list[dict[str, Any]] = []
                for chunk_start in range(0, total, chunk_size):
                    chunk = texts_to_predict[chunk_start: chunk_start + chunk_size]
                    preds = predictor.predict_batch(chunk)  # type: ignore[union-attr]
                    all_preds.extend(preds)
                    done = min(chunk_start + chunk_size, total)
                    progress_bar.progress(done / total, text=f"پردازش {fa_num(done)} از {fa_num(total)} نظر...")
                    time.sleep(0.02)  # let Streamlit flush
                elapsed = round(time.perf_counter() - started, 2)
                results = [
                    {"ردیف": idx + 1, "متن نظر": txt, "احساس": _fa_label(p) or "نامشخص", "کلاس لاتین": p["label"] or "unknown", "درجه اطمینان": f"{p['confidence']*100:.1f}%", "confidence_val": p["confidence"], "raw_label": p["label"]}
                    for idx, (txt, p) in enumerate(zip(texts_to_predict, all_preds))
                ]
                st.session_state["batch_results"] = pd.DataFrame(results)
                st.session_state["batch_elapsed"] = elapsed
                progress_bar.empty()
                try:
                    st.toast(f"✅ {total} نظر در {elapsed} ثانیه تحلیل شد", icon="🎉")
                except Exception:
                    pass
            except Exception as exc:
                progress_bar.empty()
                st.error(f"خطا در استنتاج دسته‌ای: {exc}")
        else:
            # Fallback: per-item loop with progress
            results = []
            for idx, item in enumerate(texts_to_predict):
                pred = predictor.predict(item)
                results.append({"ردیف": idx + 1, "متن نظر": item, "احساس": _fa_label(pred) or "نامشخص", "کلاس لاتین": pred["label"] or "unknown", "درجه اطمینان": f"{pred['confidence']*100:.1f}%", "confidence_val": pred["confidence"], "raw_label": pred["label"]})
                progress_bar.progress((idx + 1) / total, text=f"پردازش نظر {idx+1} از {total}...")
            elapsed = round(time.perf_counter() - started, 2)
            progress_bar.empty()
            st.session_state["batch_results"] = pd.DataFrame(results)
            st.session_state["batch_elapsed"] = elapsed

    batch_res = st.session_state.get("batch_results")
    if batch_res is None or batch_res.empty:
        empty_state("📋", "هنوز پردازش دسته‌ای انجام نشده است", "نظرات را در کادر متنی وارد یا فایل CSV بارگذاری کنید و دکمهٔ اجرا را بزنید.")
        return
    if batch_res is not None and not batch_res.empty:
        elapsed = st.session_state.get("batch_elapsed", 0.0)
        n_done = fa_num(f"{len(batch_res):,}")
        batched_active = hasattr(get_predictor(str(model_dir))[0], "predict_batch")
        st.markdown(
            '<div class="glass-card fa-lines" style="margin-top:1rem; margin-bottom:1rem; border-right:4px solid #10B981; direction:rtl; text-align:right;">'
            '<div><span style="font-weight:700; color:#059669;">✓ پردازش دسته‌ای با موفقیت پایان یافت</span></div>'
            f'<div><span style="color:var(--text-muted); font-size:0.95rem;"><b>{n_done}</b> نظر در <b>{fa_num(elapsed)}</b> ثانیه روی CPU ارزیابی شد.</span></div>'
            + ('<div><span style="color:var(--text-muted); font-size:0.95rem;">استنتاج دسته‌ای (batched) فعال بود.</span></div>' if batched_active else '')
            + '</div>',
            unsafe_allow_html=True,
        )
        counts = batch_res["raw_label"].value_counts()
        pos_cnt = int(counts.get("positive", 0))
        neu_cnt = int(counts.get("neutral", 0))
        neg_cnt = int(counts.get("negative", 0))
        total_cnt = len(batch_res)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(metric_card("کل پردازش‌شده", f"{total_cnt:,}", f"{elapsed} ثانیه", "#6366F1", "📋"), unsafe_allow_html=True)
        with b2:
            st.markdown(metric_card("نظرات مثبت", f"{pos_cnt:,}", f"{pos_cnt/total_cnt*100:.1f}% سهم", LABEL_COLORS["positive"], "😊"), unsafe_allow_html=True)
        with b3:
            st.markdown(metric_card("نظرات خنثی", f"{neu_cnt:,}", f"{neu_cnt/total_cnt*100:.1f}% سهم", LABEL_COLORS["neutral"], "😐"), unsafe_allow_html=True)
        with b4:
            st.markdown(metric_card("نظرات منفی", f"{neg_cnt:,}", f"{neg_cnt/total_cnt*100:.1f}% سهم", LABEL_COLORS["negative"], "😤"), unsafe_allow_html=True)
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        col_table, col_pie = st.columns([1.6, 1])
        with col_pie:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">توزیع احساسات این دسته</div>', unsafe_allow_html=True)
            is_dark = _get_theme() == "dark"
            pie_counts = pd.Series([neg_cnt, neu_cnt, pos_cnt], index=["negative", "neutral", "positive"])
            pie_fig = go.Figure(go.Pie(labels=[LABEL_FA[n] for n in pie_counts.index], values=pie_counts.values.tolist(), hole=0.48, marker={"colors": [LABEL_COLORS[n] for n in pie_counts.index]}, textinfo="label+percent", hovertemplate="<b>%{label}</b>: %{value} نظر (%{percent})<extra></extra>"))
            pie_fig.update_layout(template="plotly_dark" if is_dark else "plotly_white", font={"family": "Vazirmatn, Tahoma, sans-serif", "color": "#E2E8F0" if is_dark else "#0F172A"}, height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pie_fig, width="stretch")
        with col_table:
            st.markdown('<div class="fa-block" style="font-weight:700; margin-bottom:4px;">جدول نتایج تفکیکی</div>', unsafe_allow_html=True)
            fa_labels = ["مثبت", "خنثی", "منفی"]
            chosen_labels = st.multiselect("فیلتر نمایش:", fa_labels, default=fa_labels, key="batch_filter_labels")
            view = batch_res if not chosen_labels else batch_res[batch_res["احساس"].isin(chosen_labels)].copy()
            if view.empty:
                st.info("هیچ ردیفی با این فیلتر مطابقت ندارد.")
            else:
                view = view.assign(**{"اطمینان": (view["confidence_val"] * 100).round(1)})
                st.dataframe(
                    view[["ردیف", "متن نظر", "احساس", "اطمینان"]],
                    width="stretch",
                    height=260,
                    hide_index=True,
                    column_config={
                        "اطمینان": st.column_config.ProgressColumn("درجه اطمینان", min_value=0, max_value=100, format="%.0f%%"),
                        "متن نظر": st.column_config.TextColumn("متن نظر", width="medium"),
                        "احساس": st.column_config.TextColumn("احساس", width="small"),
                    },
                )
        csv_io = io.BytesIO()
        export_df = batch_res[["ردیف", "متن نظر", "احساس", "کلاس لاتین", "درجه اطمینان"]]
        export_df.to_csv(csv_io, index=False, encoding="utf-8-sig")
        st.download_button(label="📥 دانلود نتایج دسته‌ای (CSV سازگار با اکسل فارسی)", data=csv_io.getvalue(), file_name=f"sentiment_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)


# ==========================================================================
# Section 10.5: Page — Product Advisor (Digikala URL Analyzer)
# ==========================================================================
def page_advisor(model_dir: Path) -> None:
    gradient_header(
        "مشاور هوشمند خرید کالا | Digikala Product Advisor",
        "لینک یا کد کالای دیجی‌کالا را وارد کنید تا نظرات واقعی خریداران مستقیماً استخراج شده و با مدل ParsBERT تحلیل شوند.",
        badge="مشاوره خرید",
    )
    from src.product_advisor import analyze_digikala_product

    col_in, col_samples = st.columns([2.6, 1.4])
    with col_samples:
        st.markdown('<div class="fa-block" style="font-size:0.85rem; font-weight:600; margin-bottom:4px;">کالاهای نمونه برای تست سریع:</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        if s1.button("گوشی Galaxy A56", key="sample_a56", use_container_width=True):
            st.session_state["product_url_input"] = "https://www.digikala.com/product/dkp-18575915/"
        if s2.button("هدفون بلوتوثی P47", key="sample_p47", use_container_width=True):
            st.session_state["product_url_input"] = "https://www.digikala.com/product/dkp-5112620/"
    with col_in:
        prod_input = st.text_input("آدرس یا شناسه کالای دیجی‌کالا (URL یا کد dkp):", value=st.session_state.get("product_url_input", ""), placeholder="مثال: https://www.digikala.com/product/dkp-18575915/ یا 18575915", key="product_url_text_input")

    c_cnt, c_btn = st.columns([1.5, 1])
    with c_cnt:
        comments_limit = st.slider("تعداد نظرات برای استخراج و تحلیل:", min_value=5, max_value=60, value=25, step=5, help="نظرات از جدیدترین نظرات مشتریان دیجی‌کالا خوانده می‌شوند.")
    with c_btn:
        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        analyze_btn = st.button("🚀 استخراج و تحلیل ارزش خرید", type="primary", use_container_width=True, disabled=not prod_input.strip())

    if analyze_btn and prod_input.strip():
        with st.spinner("در حال واکشی اطلاعات و نظرات از دیجی‌کالا و استنتاج با ParsBERT..."):
            try:
                cached_predictor, _ = get_predictor(str(model_dir))
                report = analyze_digikala_product(
                    prod_input.strip(),
                    max_comments=comments_limit,
                    model_dir=model_dir,
                    predictor=cached_predictor,
                )
                st.session_state["advisor_report"] = report
                try:
                    st.toast("تحلیل کالا با موفقیت انجام شد", icon="✅")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"خطا در پردازش کالا: {e}")

    report = st.session_state.get("advisor_report")
    if not report:
        empty_state("🛍️", "هنوز کالایی تحلیل نشده است", "لینک دیجی‌کالا را وارد کنید یا یکی از کالاهای نمونه را انتخاب کنید.")
        return

    if report:
        prod = report.product
        rec = report.recommendation
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        p_col1, p_col2 = st.columns([1, 2.5])
        with p_col1:
            if prod.get("image_url"):
                st.image(prod["image_url"], width=220)
            else:
                st.markdown('<div style="font-size:4rem; text-align:center;">📦</div>', unsafe_allow_html=True)
        with p_col2:
            price_val = prod.get("price_toman")
            rating_score = prod.get("rating_score")
            rating_count = prod.get("rating_count")
            st.markdown(
                f'<div class="glass-card" style="direction:rtl; text-align:right;">'
                f'<div style="font-size:1.25rem; font-weight:800; color:var(--text); margin-bottom:6px;">{fa(prod.get("title_fa",""))}</div>'
                f'<div class="fa-lines" style="color:var(--text-muted); font-size:0.9rem; margin-bottom:8px;">'
                f'<div>برند: <b>{fa(prod.get("brand","نامشخص"))}</b></div>'
                f'<div>شناسه کالا: <code>dkp-{prod.get("product_id")}</code></div></div>'
                + (f'<div style="font-size:1.2rem; font-weight:700; color:#10B981; margin-bottom:6px;">💰 قیمت: {fa_num(f"{price_val:,}")} تومان</div>' if price_val else '')
                + (f'<div style="color:var(--text); font-size:0.88rem;">⭐ رضایت کلی دیجی‌کالا: {fa_num(rating_score)}٪ (از {fa_num(f"{rating_count:,}")} رأی)</div>' if rating_score else '')
                + f'<div style="margin-top:8px;"><a href="{prod.get("url")}" target="_blank" style="color:#6366F1; text-decoration:none; font-size:0.88rem;">🔗 مشاهده کالا در دیجی‌کالا</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        verdict_color = "#10B981" if "🟢" in rec.verdict else ("#F59E0B" if "🟡" in rec.verdict else "#EF4444")
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="verdict-hero fade-in" style="background:{verdict_color}10; border-color:{verdict_color}44;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
            f'<div class="verdict-label" style="color:{verdict_color};">{fa(rec.verdict)}</div>'
            f'<div style="font-size:1.35rem; font-weight:800; color:{verdict_color};">امتیاز: {fa_num(f"{rec.score_out_of_100:.1f}")} از ۱۰۰</div>'
            f'</div>'
            f'<div style="font-size:1.05rem; color:var(--text); line-height:1.8; margin-top:0.75rem;">{fa(rec.summary_text_fa)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("نظرات تحلیل‌شده", f"{report.analyzed_count}", f"در {report.elapsed_seconds} ثانیه", "#6366F1", "💬"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("احساسات مثبت", f"{rec.positive_ratio*100:.1f}٪", "سهم نظرات مثبت", "#10B981", "😊"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("احساسات منفی", f"{rec.negative_ratio*100:.1f}٪", "سهم نظرات منفی", "#EF4444", "😤"), unsafe_allow_html=True)
        with m4:
            buyer_stat = f"{rec.buyer_positive_ratio*100:.1f}٪" if rec.buyer_comments_count else "نامشخص"
            st.markdown(metric_card("رضایت خریداران قطعی", buyer_stat, f"از {rec.buyer_comments_count} خریدار", "#F59E0B", "🛒"), unsafe_allow_html=True)

        # Donut: sentiment mix of the analyzed comments
        total_c = max(rec.total_comments, 1)
        mix_counts = {
            "positive": round(rec.positive_ratio * total_c),
            "neutral": round(rec.neutral_ratio * total_c),
            "negative": round(rec.negative_ratio * total_c),
        }
        donut = go.Figure(
            go.Pie(
                labels=[LABEL_FA[k] for k in mix_counts],
                values=list(mix_counts.values()),
                hole=0.62,
                marker={"colors": [LABEL_COLORS[k] for k in mix_counts]},
                textinfo="value+percent",
                textfont={"family": "Vazirmatn, Tahoma, sans-serif", "size": 12},
                hovertemplate="<b>%{label}</b>: %{value} نظر (%{percent})<extra></extra>",
            )
        )
        donut.add_annotation(
            text=f'<b>{fa_num(f"{rec.score_out_of_100:.0f}")}</b><span style="font-size:0.7em;color:#94A3B8">/۱۰۰</span>',
            x=0.5, y=0.5, showarrow=False,
            font={"size": 26, "family": "Vazirmatn, Tahoma, sans-serif"},
        )
        donut.update_layout(
            template="plotly_dark" if _get_theme() == "dark" else "plotly_white",
            font={"family": "Vazirmatn, Tahoma, sans-serif"},
            height=250, margin={"l": 10, "r": 10, "t": 30, "b": 10},
            title={"text": "ترکیب احساسات نظرات این کالا", "font": {"size": 13, "family": "Vazirmatn, Tahoma, sans-serif"}, "x": 0.5, "xanchor": "center"},
            showlegend=True, legend={"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(donut, width="stretch")
        if rec.top_advantages or rec.top_disadvantages:
            adv_col, dis_col = st.columns(2)
            with adv_col:
                if rec.top_advantages:
                    st.markdown('<div class="fa-block" style="font-weight:700; color:#10B981; margin-top:1rem; margin-bottom:6px;">👍 نقاط قوت پربسامد:</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip" style="background:#ECFDF5; color:#047857; border-color:#A7F3D0;">✓ {fa(a)}</span>' for a in rec.top_advantages)
                    st.markdown(chips, unsafe_allow_html=True)
            with dis_col:
                if rec.top_disadvantages:
                    st.markdown('<div class="fa-block" style="font-weight:700; color:#EF4444; margin-top:1rem; margin-bottom:6px;">👎 نقاط ضعف مکرر:</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip" style="background:#FEF2F2; color:#B91C1C; border-color:#FECACA;">✗ {fa(d)}</span>' for d in rec.top_disadvantages)
                    st.markdown(chips, unsafe_allow_html=True)
        if rec.reasons_fa:
            with st.expander("📌 دلایل تصمیم‌گیری مدل", expanded=True):
                reasons_html = "".join(f'<div class="fa-block" style="margin-bottom:4px;">• {fa(r)}</div>' for r in rec.reasons_fa)
                st.markdown(reasons_html, unsafe_allow_html=True)
        with st.expander(f"📋 مشاهده تمام {report.analyzed_count} نظر تحلیل‌شده", expanded=False):
            table_rows = [
                {"ردیف": idx+1, "متن نظر": c["body"], "احساس": c["sentiment_fa"], "خریدار": "بله" if c.get("is_buyer") else "خیر", "امتیاز کاربر": f"{c.get('rate') or '-'} ستاره", "درجه اطمینان": f"{c['confidence']*100:.1f}٪"}
                for idx, c in enumerate(report.comments_sample)
            ]
            st.dataframe(pd.DataFrame(table_rows), width="stretch", height=280)


# ==========================================================================
# Section 11: Page — Help & Limits
# ==========================================================================
def page_help() -> None:
    gradient_header("راهنمای سامانه و محدودیت‌های فنی", "مستندات معماری، خط‌لوله داده، فرامین خط فرمان و بیان شفاف محدودیت‌ها.", badge="مستندات مهندسی")
    rtl("این سامانه برای ارزیابی احساسات سه کلاسه (مثبت / خنثی / منفی) زبان فارسی بر بستر ParsBERT پیاده‌سازی شده و داده‌های آموزشی آن از نظرات دارای ستاره در دیجی‌کالا و نظرات خبری بدون برچسب ورزش۳ استخراج شده است.", muted=True)
    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    c_pipe, c_limits = st.columns([1.1, 1.2])
    with c_pipe:
        # Pipeline flow strip: جمع‌آورد → پیش‌پردازش → آموزش → ارزیابی → داشبورد
        steps = (
            ("۱", "جمع‌آوری", "src/scraper.py"),
            ("۲", "پیش‌پردازش", "src/preprocess.py"),
            ("۳", "آموزش", "src/train.py"),
            ("۴", "ارزیابی", "metrics.json"),
            ("۵", "داشبورد", "app.py"),
        )
        parts = []
        for idx, (num, title, module) in enumerate(steps):
            parts.append(
                f'<span class="chip primary" style="font-size:0.8rem;"><b>{fa(num)}</b> {fa(title)} <span style="font-size:0.68rem; color:var(--text-faint);" dir="ltr">{fa(module)}</span></span>'
            )
            if idx < len(steps) - 1:
                parts.append('<span style="color:var(--text-faint); font-size:0.85rem;">←</span>')
        st.markdown(
            '<div class="glass-card" style="margin-bottom:1rem; direction:rtl; text-align:right;">'
            '<div style="font-weight:700; color:var(--text); font-size:1rem; margin-bottom:10px;">🔀 جریان کلی خط لوله</div>'
            f'<div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">{"".join(parts)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="glass-card accent-top" style="margin-bottom:1rem; direction:rtl; text-align:right;">'
            '<div style="font-weight:700; color:var(--text); font-size:1.1rem; margin-bottom:8px;">🛠️ فرامین خط لوله (Pipeline CLI)</div>'
            '<div style="font-size:0.9rem; color:var(--text-muted); margin-bottom:8px;">تمامی مراحل از جمع‌آوری تا آموزش به صورت ماژول‌های مستقل قابل اجرا هستند:</div></div>',
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

# ۳. پیش‌پردازش hazm (قاعده تبدیل ZWNJ به فاصله)
python -m src.preprocess --in data/raw_digikala_comments.csv --out data/cleaned_comments.csv

# ۴. تولید ارقام آماری و ابر کلمات
python -m src.visualize --csv data/cleaned_comments.csv --out assets/figures

# ۵. فاین‌تیون ParsBERT با زیان وزن‌دار
python -m src.train --smoke
python -m src.train --csv data/cleaned_comments.csv --epochs 3

# ۶. اجرای داشبورد Streamlit
python -m streamlit run app.py
```
            """
        )
    with c_limits:
        st.markdown(
            '<div class="glass-card" style="margin-bottom:1rem; border-top:3px solid #F59E0B; direction:rtl; text-align:right;">'
            '<div style="font-weight:700; color:var(--text); font-size:1.1rem; margin-bottom:8px;">⚠️ محدودیت‌های فنی و واقعیت‌های مهندسی</div></div>',
            unsafe_allow_html=True,
        )
        limits = [
            ("نظارت ضعیف (Weak Supervision)", "برچسب‌های احساسات از امتیازهای ۱ تا ۵ ستاره مشتق شده‌اند (۱-۲ منفی، ۳ خنثی، ۴-۵ مثبت). این برچسب‌گذاری تقریب ضعیف است نه برچسب‌گذاری دستی توسط کارشناس."),
            ("عدم توازن شدید کلاس خنثی", "نظرات ۳ ستاره حدود ۱۴ برابر کمتر از ۵ ستاره هستند. به همین علت یادآوری (Recall) کلاس خنثی پایین‌تر است و Macro-F1 معیار کلیدی است."),
            ("فقدان دیتاست سه‌کلاسه استاندارد", "دیتاست‌های Hugging Face عمدتاً دودویی هستند. پیکرهٔ سه کلاسه مستقیماً خزش و تولید شد."),
            ("عدم وجود امتیاز در نظرات خبری", "نظرات ورزش۳ فاقد ستاره هستند و صرفاً برای تحلیل واژگانی و آزمون خارج از دامنه استفاده می‌شوند."),
            ("محدودیت سخت‌افزاری CPU-only", "این محیط فاقد GPU است. مدل ۱۶۳M پارامتری با حداکثر بهینه‌سازی بارگذاری و کش استنتاج می‌شود."),
        ]
        limits_html = "".join(
            f'<div class="glass-card" style="margin-bottom:8px; padding:0.75rem 1rem; border-right:4px solid var(--border-strong); direction:rtl; text-align:right;">'
            f'<div style="font-weight:700; color:var(--text); font-size:0.95rem; margin-bottom:3px;">• {fa(t)}</div>'
            f'<div style="font-size:0.88rem; color:var(--text-muted); line-height:1.7;">{fa(d)}</div></div>'
            for t, d in limits
        )
        st.markdown(limits_html, unsafe_allow_html=True)


# ==========================================================================
# Section 12: Main Entry Point and Navigation
# ==========================================================================
def _existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def main() -> None:
    # Restore theme from ?theme=dark before first paint
    if THEME_KEY not in st.session_state:
        try:
            st.session_state[THEME_KEY] = st.query_params.get("theme", "light")
        except Exception:
            st.session_state[THEME_KEY] = "light"
    _inject_theme()
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.4rem 0 0.6rem 0; text-align:center;">'
            '<div class="sidebar-brand">BAFP</div>'
            '<div style="font-size:0.84rem; color:var(--text-muted); font-weight:600; margin-top:3px;">تحلیل هوشمند احساسات فارسی</div></div>',
            unsafe_allow_html=True,
        )
        # Theme toggle — compact, no rerun flash
        col_theme_label, col_theme_btn = st.columns([2, 1])
        with col_theme_label:
            st.markdown(f'<div style="font-size:0.85rem; color:var(--text-muted); padding-top:6px;" dir="rtl">{"🌙 حالت تیره" if _get_theme()=="light" else "☀️ حالت روشن"}</div>', unsafe_allow_html=True)
        with col_theme_btn:
            icon = "🌙" if _get_theme() == "light" else "☀️"
            if st.button(icon, key="theme_toggle_btn", help="تغییر تم روشن/تیره"):
                _toggle_theme()
                st.rerun()

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
        corpora = _existing([CLEANED_CSV, WIDE_CLEANED_CSV])
        # Short Persian display labels (format_func keeps the filename as the
        # value): long ASCII filenames overflow the fixed sidebar pill on
        # systems that inflate text — Windows text scaling and browser
        # minimum-font-size override even px sizes — truncating to
        # "cleaned_comment…".
        corpus_fa = {"cleaned_comments.csv": "اصلی (base)", "cleaned_comments_wide.csv": "گسترده (wide)"}
        corpus_choice = (
            st.selectbox(
                "پیکره داده (Corpus)",
                [path.name for path in corpora],
                index=len(corpora) - 1,
                format_func=lambda name: corpus_fa.get(name, name),
                help="cleaned_comments.csv: اصلی؛ cleaned_comments_wide.csv: گسترده",
            )
            if corpora else None
        )
        checkpoint_names = [path.name for path in available_models()]
        ckpt_fa = {"parsbert-sentiment-3class": "ParsBERT (اصلی)", "parsbert-sentiment-3class-wide": "ParsBERT (گسترده)"}
        chosen = (
            st.selectbox(
                "مدل ذخیره‌شده (Checkpoint)",
                checkpoint_names,
                index=checkpoint_names.index(DEFAULT_MODEL_DIR.name) if DEFAULT_MODEL_DIR.name in checkpoint_names else 0,
                format_func=lambda name: ckpt_fa.get(name, name.replace("parsbert-sentiment-", "")),
                help="parsbert-sentiment-3class: پیکره ۱۶ کالا؛ -wide: پیکره ۵۳ کالا",
            )
            if checkpoint_names else None
        )
        st.divider()
        rtl("پیکره: نظرات دیجی‌کالا (برچسب‌دار) و ورزش۳ (بدون برچسب)", muted=True)
        # About Footer — theme-aware
        st.markdown(
            '<div class="glass-card" style="margin-top:1.1rem; padding:0.85rem; font-size:0.8rem; color:var(--text-muted); direction:rtl; text-align:right;">'
            '<div style="font-weight:700; color:var(--text); margin-bottom:6px;">مشخصات فنی سامانه</div>'
            '<div class="fa-lines" style="line-height:1.9;">'
            '<div>مدل: ParsBERT (<span dir="ltr">163M</span> پارامتر)</div>'
            '<div>وزن‌دهی کلاس متوازن</div>'
            '<div>سخت‌افزار: فقط CPU</div>'
            '<div>استنتاج دسته‌ای</div>'
            '<div>ابزارها: <span dir="ltr">Streamlit · Hazm · Plotly</span></div>'
            '</div>'
            '<div style="margin-top:8px; font-size:0.75rem; color:var(--text-faint);">طراحی: BAFP v2 (تم روشن/تیره، RTL کامل)</div></div>',
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
