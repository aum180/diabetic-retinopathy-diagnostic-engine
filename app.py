import sys
import os
import io
import time
import base64
import requests
from datetime import datetime
import streamlit as st
from PIL import Image
import numpy as np

# Ensure root workspace directory is in python path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

# Internal prediction engine import with fallback safety
try:
    from src.predict import run_prediction_pipeline
    HAS_LOCAL_PREDICT = True
    IMPORT_ERROR = None
except Exception as e:
    HAS_LOCAL_PREDICT = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. Page Configuration & Theme
# ==============================================================================
st.set_page_config(
    page_title="RETINA•AI | Clinical Diagnostic Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000/predict")

# ==============================================================================
# 2. Modern Glassmorphic Medical Styling
# ==============================================================================
def inject_custom_styles():
    st.markdown(
        """
        <style>
        /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Root Reset & Typography */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #e2e8f0;
        }

        /* App Background */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #0d1933 0%, #080d1a 60%, #050811 100%);
            background-attachment: fixed;
        }

        /* Professional Executive Slate-Blue Sidebar */
        [data-testid="stSidebar"], 
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #0e1a30 0%, #11213d 50%, #0a1426 100%) !important;
            border-right: 1px solid rgba(56, 189, 248, 0.25) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5) !important;
        }

        /* High-Contrast Sidebar Typography - Guarantee Maximum Readability */
        [data-testid="stSidebar"] * {
            color: #f1f5f9 !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #e2e8f0 !important;
        }

        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #38bdf8 !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
        }

        /* Sidebar Form Field Labels */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label p,
        [data-testid="stSidebar"] label span {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            letter-spacing: 0.01em !important;
        }

        /* Sidebar Captions & Secondary Text */
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] small {
            color: #94a3b8 !important;
            font-size: 0.82rem !important;
            line-height: 1.45 !important;
        }

        /* Sidebar Selectbox & Inputs */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #172544 !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #ffffff !important;
        }

        /* Sidebar Cards for Modular Clinical Hierarchy */
        .sidebar-section-card {
            background: rgba(23, 37, 68, 0.65);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 12px;
            padding: 16px 14px 10px 14px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }

        .sidebar-section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            color: #38bdf8 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Sidebar Divider */
        [data-testid="stSidebar"] hr {
            border-color: rgba(56, 189, 248, 0.18) !important;
            margin: 1.2rem 0 !important;
        }

        /* Top HUD Status Bar */
        .hud-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .hud-badge-group {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .hud-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        .hud-pill-active {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        .hud-pill-info {
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }

        .hud-pill-tag {
            background: rgba(148, 163, 184, 0.1);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 10px #10b981;
            animation: pulseAnimation 1.8s infinite ease-in-out;
        }

        @keyframes pulseAnimation {
            0% { transform: scale(0.95); opacity: 0.7; }
            50% { transform: scale(1.25); opacity: 1; box-shadow: 0 0 14px #34d399; }
            100% { transform: scale(0.95); opacity: 0.7; }
        }

        /* Hero Header Card */
        .hero-container {
            position: relative;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(20, 30, 55, 0.6) 100%);
            border-radius: 20px;
            padding: 30px 32px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            margin-bottom: 28px;
            overflow: hidden;
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -60px;
            right: -60px;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.35rem;
            font-weight: 800;
            background: linear-gradient(120deg, #ffffff 30%, #38bdf8 80%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 18px;
            line-height: 1.5;
            max-width: 860px;
        }

        /* Clinical Cards */
        .clinical-card {
            background: rgba(15, 23, 42, 0.75);
            border-radius: 16px;
            padding: 22px;
            border: 1px solid rgba(56, 189, 248, 0.18);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
            margin-bottom: 20px;
            transition: border-color 0.3s ease, transform 0.2s ease;
        }

        .clinical-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
        }

        .card-header-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
        }

        /* Scan Viewport Container */
        .scan-wrapper {
            position: relative;
            width: 100%;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(56, 189, 248, 0.28);
            background: #020617;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        }

        /* Animated Laser Radar Line */
        .laser-beam {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent 0%, #38bdf8 30%, #00f2fe 50%, #38bdf8 70%, transparent 100%);
            box-shadow: 0 0 12px #00f2fe, 0 0 24px #38bdf8;
            animation: radarSweep 2.4s infinite ease-in-out;
            z-index: 20;
            pointer-events: none;
        }

        @keyframes radarSweep {
            0% { top: 1%; opacity: 0.8; }
            50% { top: 96%; opacity: 1; }
            100% { top: 1%; opacity: 0.8; }
        }

        /* Diagnosis Alert Banners */
        .diag-banner-dr {
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(159, 18, 57, 0.28) 100%);
            border: 1px solid rgba(244, 63, 94, 0.4);
            border-radius: 14px;
            padding: 20px 24px;
            margin-top: 15px;
            box-shadow: 0 8px 30px rgba(225, 29, 72, 0.18);
        }

        .diag-banner-healthy {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
            border: 1px solid rgba(52, 211, 153, 0.4);
            border-radius: 14px;
            padding: 20px 24px;
            margin-top: 15px;
            box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
        }

        .diag-title-dr {
            color: #fb7185;
            font-family: 'Outfit', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .diag-title-healthy {
            color: #34d399;
            font-family: 'Outfit', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Metric Tiles */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }

        .metric-box {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 14px 16px;
            transition: all 0.2s ease;
        }

        .metric-box:hover {
            border-color: rgba(56, 189, 248, 0.3);
            background: rgba(20, 32, 58, 0.85);
        }

        .metric-label {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 4px;
        }

        .metric-val {
            font-size: 1.4rem;
            font-weight: 700;
            color: #f1f5f9;
            font-family: 'Outfit', sans-serif;
        }

        /* Buttons Styling */
        .stButton>button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(37, 99, 235, 0.3) 100%) !important;
            color: #f8fafc !important;
            transition: all 0.25s ease !important;
        }

        .stButton>button:hover {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.4) !important;
            transform: translateY(-1px) !important;
        }

        /* Heatmap Legend Bar */
        .legend-bar {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #0000ff 0%, #00ffff 25%, #00ff00 50%, #ffff00 75%, #ff0000 100%);
            margin: 8px 0;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
        }

        /* High-Contrast Professional Medical Expanders */
        [data-testid="stExpander"],
        details {
            background: linear-gradient(180deg, #0e1a30 0%, #11213d 100%) !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
            margin-bottom: 16px !important;
            overflow: hidden !important;
        }

        /* Expander Summary / Header */
        [data-testid="stExpander"] summary,
        details summary {
            background: rgba(22, 36, 64, 0.75) !important;
            color: #38bdf8 !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            padding: 14px 18px !important;
            border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stExpander"] summary:hover,
        details summary:hover {
            color: #7dd3fc !important;
            background: rgba(30, 48, 86, 0.85) !important;
        }

        [data-testid="stExpander"] summary svg,
        details summary svg {
            fill: #38bdf8 !important;
            color: #38bdf8 !important;
        }

        /* Expander Body Content Container */
        [data-testid="stExpander"] div[role="region"],
        [data-testid="stExpander"] [data-testid="stExpanderDetails"],
        details > div:not(summary) {
            background: rgba(11, 20, 38, 0.95) !important;
            padding: 20px 22px !important;
            color: #f1f5f9 !important;
        }

        /* All Text inside expander guaranteed maximum readability */
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span,
        [data-testid="stExpander"] li,
        [data-testid="stExpander"] div {
            color: #e2e8f0 !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
        }

        [data-testid="stExpander"] strong,
        [data-testid="stExpander"] b {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        [data-testid="stExpander"] h1,
        [data-testid="stExpander"] h2,
        [data-testid="stExpander"] h3,
        [data-testid="stExpander"] h4 {
            color: #38bdf8 !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            margin-top: 10px !important;
            margin-bottom: 10px !important;
        }

        [data-testid="stExpander"] ul {
            padding-left: 20px !important;
            margin-bottom: 10px !important;
        }

        [data-testid="stExpander"] li {
            margin-bottom: 8px !important;
            color: #e2e8f0 !important;
        }

        /* Styled Protocol Action Cards */
        .protocol-card {
            background: rgba(22, 36, 64, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-left: 4px solid #38bdf8;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        .protocol-card-dr {
            background: rgba(225, 29, 72, 0.12);
            border: 1px solid rgba(244, 63, 94, 0.3);
            border-left: 4px solid #f43f5e;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        .protocol-title {
            color: #38bdf8 !important;
            font-weight: 700;
            font-size: 0.98rem;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .protocol-title-dr {
            color: #fb7185 !important;
            font-weight: 700;
            font-size: 0.98rem;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .protocol-desc {
            color: #f1f5f9 !important;
            font-size: 0.91rem;
            line-height: 1.5;
        }

        /* Medical Regulatory Disclaimer Callout Box */
        .disclaimer-callout {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-left: 4px solid #f59e0b;
            border-radius: 10px;
            padding: 16px 20px;
            margin-top: 16px;
        }

        .disclaimer-title {
            color: #fbbf24 !important;
            font-weight: 700;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }

        .disclaimer-text {
            color: #fde68a !important;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        /* High-Contrast Inline Code & Tags (Removes white vague boxes) */
        code, [data-testid="stMarkdown"] code {
            background: rgba(22, 36, 64, 0.9) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            padding: 3px 8px !important;
            border-radius: 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
            font-size: 0.88em !important;
        }

        /* Dedicated Clinical Case Header Card */
        .case-header-card {
            background: linear-gradient(90deg, rgba(23, 37, 68, 0.85) 0%, rgba(14, 24, 44, 0.6) 100%);
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-left: 4px solid #38bdf8;
            border-radius: 12px;
            padding: 14px 20px;
            margin: 24px 0 16px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }

        .case-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.05em;
        }

        .case-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.01em;
        }

        .case-source-pill {
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            background: rgba(148, 163, 184, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.2);
            letter-spacing: 0.04em;
        }

        /* High-Contrast Clinical Navigation Tabs */
        [data-testid="stTabs"] div[data-baseweb="tab-list"] {
            gap: 12px !important;
            background: transparent !important;
            border-bottom: 2px solid rgba(56, 189, 248, 0.25) !important;
            padding-bottom: 2px !important;
            margin-bottom: 20px !important;
        }

        /* All Tabs Base Styling */
        [data-testid="stTabs"] button[data-baseweb="tab"],
        [data-testid="stTabs"] [role="tab"] {
            height: auto !important;
            padding: 12px 24px !important;
            border-radius: 10px 10px 0 0 !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1rem !important;
            letter-spacing: 0.01em !important;
            transition: all 0.25s ease !important;
        }

        /* INACTIVE / UNSELECTED TABS (High visibility & contrast) */
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="false"],
        [data-testid="stTabs"] [role="tab"][aria-selected="false"] {
            background: rgba(22, 36, 64, 0.8) !important;
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
            border-bottom: none !important;
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="false"] *,
        [data-testid="stTabs"] [role="tab"][aria-selected="false"] * {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }

        /* Hover on Inactive Tabs */
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="false"]:hover,
        [data-testid="stTabs"] [role="tab"][aria-selected="false"]:hover {
            background: rgba(30, 48, 86, 0.95) !important;
            border-color: rgba(56, 189, 248, 0.5) !important;
            color: #ffffff !important;
            transform: translateY(-1px) !important;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="false"]:hover * {
            color: #ffffff !important;
        }

        /* ACTIVE / SELECTED TABS */
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, rgba(14, 165, 233, 0.22) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
            border: 1px solid #38bdf8 !important;
            border-bottom: 3px solid #00f2fe !important;
            color: #38bdf8 !important;
            font-weight: 700 !important;
            box-shadow: 0 -4px 16px rgba(56, 189, 248, 0.25) !important;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] *,
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] * {
            color: #38bdf8 !important;
            font-weight: 700 !important;
        }

        /* Hide default Streamlit tab highlight underline */
        [data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_styles()

# ==============================================================================
# 3. Clinical HUD Header & Telemetry
# ==============================================================================
st.markdown(
    """
    <div class="hud-banner">
        <div class="hud-badge-group">
            <span class="hud-pill hud-pill-active">
                <span class="pulse-dot"></span> ML Inference Engine Online
            </span>
            <span class="hud-pill hud-pill-info">
                👁️ EfficientNetB1 Backbone
            </span>
            <span class="hud-pill hud-pill-tag">
                Grad-CAM XAI Layer
            </span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #64748b;">
            DECISION SUPPORT PLATFORM • v2.4
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">RETINA•AI Clinical Diagnostic Suite</h1>
        <p class="hero-subtitle">
            Next-generation automated ophthalmic screening engine powered by deep convolutional networks and 
            Explainable AI (Grad-CAM). Provides immediate lesion localization and risk stratification for Diabetic Retinopathy.
        </p>
        <div style="display: flex; gap: 16px; flex-wrap: wrap;">
            <div style="background: rgba(14, 165, 233, 0.1); border-left: 3px solid #38bdf8; padding: 10px 16px; border-radius: 6px; font-size: 0.88rem; color: #cbd5e1; flex: 1; min-width: 280px;">
                <b style="color: #38bdf8;">Standard RGB Pipeline:</b> Recommended for standard clinical fundus photographs with uniform illumination.
            </div>
            <div style="background: rgba(168, 85, 247, 0.1); border-left: 3px solid #a855f7; padding: 10px 16px; border-radius: 6px; font-size: 0.88rem; color: #cbd5e1; flex: 1; min-width: 280px;">
                <b style="color: #c084fc;">Ben Graham Contrast Enhancement:</b> Employs circular boundary crop and local Gaussian color normalization to accentuate microaneurysms.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 4. Sidebar Controls & Model Customization
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.25); margin-bottom: 16px;">
            <div style="font-size: 1.25rem; font-weight: 800; font-family: 'Outfit', sans-serif; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <span style="color: #38bdf8; font-size: 1.4rem;">👁️</span> CONTROL PANEL
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-top: 3px; letter-spacing: 0.05em;">
                CLINICAL AI WORKSTATION
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="sidebar-section-card">
            <div class="sidebar-section-title">🎛️ Diagnostic Configuration</div>
        """,
        unsafe_allow_html=True
    )
    model_choice = st.selectbox(
        "Select Deep Learning Backbone",
        options=["b1_rgb", "b1_bengraham"],
        index=0,
        format_func=lambda x: "EfficientNetB1 (Standard RGB)" if x == "b1_rgb" else "EfficientNetB1 (Ben Graham Contrast)"
    )
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 8px; line-height: 1.4;">
            Active Strategy: <b style="color: #38bdf8;">{}</b>
        </div>
        </div>
        """.format("Standard RGB Preprocessing" if model_choice == "b1_rgb" else "Ben Graham Contrast Normalization"),
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-section-card">
            <div class="sidebar-section-title">⚙️ Visualizer Controls</div>
        """,
        unsafe_allow_html=True
    )
    show_scanner = st.toggle("Laser Radar Scanning Overlay", value=True)
    show_metrics_breakdown = st.toggle("Detailed Probability Breakdown", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-section-card">
            <div class="sidebar-section-title">📋 System & Pipeline Health</div>
        """,
        unsafe_allow_html=True
    )
    if HAS_LOCAL_PREDICT:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                <span style="color: #34d399; font-weight: 700; font-size: 0.85rem;">● LOCAL INFERENCE READY</span>
                <div style="color: #cbd5e1; font-size: 0.75rem; margin-top: 2px;">TensorFlow 2.21 + Keras 3 Active</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                <span style="color: #fbbf24; font-weight: 700; font-size: 0.85rem;">⚠️ FALLBACK MODE</span>
                <div style="color: #cbd5e1; font-size: 0.75rem; margin-top: 2px;">{IMPORT_ERROR[:35]}...</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; line-height: 1.5; margin-top: 6px;">
            • Model: EfficientNet-B1<br>
            • Resolution: 224×224 px<br>
            • Saliency: Grad-CAM XAI
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 5. Input Mode Selector (Sample Demo Cases vs Upload)
# ==============================================================================
input_tab1, input_tab2 = st.tabs(["🧪 Clinical Demo Library (Instant)", "📂 Upload Custom Fundus Scans"])

sample_dir = os.path.join(WORKSPACE_ROOT, "Test_Streamlit")
demo_cases = {
    "Case 1: Diabetic Retinopathy (Severe Microaneurysms)": "DR_1_IDRiD_0003.jpg",
    "Case 2: Diabetic Retinopathy (Exudates & Hemorrhages)": "DR_2_IDRiD_0004.jpg",
    "Case 3: Diabetic Retinopathy (Sample 1)": "DR_1.jpg",
    "Case 4: Healthy Retina (No Pathological DR)": "NO_DRIDRiD_0001.jpg",
    "Case 5: Healthy Retina (Clear Macula)": "NO_DR_2IDRiD_0002.jpg",
    "Case 6: Healthy Retina (Sample 1)": "NO_DR_1.jpg"
}

active_items = []

with input_tab1:
    st.markdown(
        """
        <div style="font-size: 0.92rem; color: #94a3b8; margin-bottom: 12px;">
            Select pre-verified clinical fundus scans to evaluate the deep learning diagnostic model and Grad-CAM activation in real-time.
        </div>
        """, 
        unsafe_allow_html=True
    )
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_demo_key = st.selectbox("Choose Verified Retinal Case:", list(demo_cases.keys()))
    with col_btn:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        analyze_demo_clicked = st.button("🔬 Ingest & Analyze Scan", use_container_width=True)

    if selected_demo_key:
        demo_filename = demo_cases[selected_demo_key]
        demo_filepath = os.path.join(sample_dir, demo_filename)
        if os.path.exists(demo_filepath):
            with open(demo_filepath, "rb") as f:
                demo_bytes = f.read()
            active_items.append({
                "name": f"Demo: {selected_demo_key.split(':')[0]} ({demo_filename})",
                "bytes": demo_bytes,
                "is_demo": True
            })

with input_tab2:
    st.markdown(
        """
        <div style="font-size: 0.92rem; color: #94a3b8; margin-bottom: 12px;">
            Upload one or multiple retinal fundus images in JPG, JPEG, or PNG format for automatic batch inference.
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_files = st.file_uploader(
        "Drop fundus photographs here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        # Override demo if user specifically uploaded custom files
        active_items = []
        for uf in uploaded_files:
            active_items.append({
                "name": uf.name,
                "bytes": uf.getvalue(),
                "type": uf.type,
                "is_demo": False
            })

# ==============================================================================
# 6. Inference Pipeline & Diagnostic Execution
# ==============================================================================
if active_items:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 14px; margin-bottom: 20px; padding: 10px 16px; background: rgba(15, 23, 42, 0.6); border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2);">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; font-family: 'Outfit', sans-serif;">
                📂 Active Worklist: {len(active_items)} Ingested Scan{'s' if len(active_items) > 1 else ''} Ready for Diagnostic Evaluation
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                STATUS: READY FOR INFERENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for idx, item in enumerate(active_items, start=1):
        image_name = item["name"]
        image_bytes = item["bytes"]
        is_demo = item.get("is_demo", False)

        st.markdown(
            f"""
            <div class="case-header-card">
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <span class="case-badge">CASE #{idx}</span>
                    <span class="case-title">{image_name}</span>
                </div>
                <div class="case-source-pill">
                    SOURCE: {'VERIFIED CLINICAL DEMO' if is_demo else 'PATIENT SCAN UPLOAD'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2-Column Clinical Layout
        col_scan, col_xai = st.columns(2)

        with col_scan:
            st.markdown(
                """
                <div class="card-header-title">
                    <span>📷 Fundus Scan & Laser Radar</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Scan Display with Live Laser Scanner
            st.markdown(
                f"""
                <div class="scan-wrapper">
                    { '<div class="laser-beam"></div>' if show_scanner else '' }
                """,
                unsafe_allow_html=True
            )
            st.image(image_bytes, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Inference Processing
        with st.spinner("Executing Convolutional Inference & Grad-CAM Analysis..."):
            start_time = time.time()
            data_processed = False
            label, confidence, gradcam_img, raw_val = None, 0.0, None, 0.5

            # 1. First attempt: FastAPI microservice
            try:
                files = {"file": (image_name, image_bytes, "image/jpeg")}
                params = {"model_key": model_choice}
                response = requests.post(FASTAPI_URL, files=files, params=params, timeout=3)
                if response.status_code == 200:
                    res_data = response.json()
                    label = res_data.get("prediction")
                    confidence = float(res_data.get("confidence", 0))
                    gradcam_b64 = res_data.get("gradcam_base64")
                    if gradcam_b64:
                        g_bytes = base64.b64decode(gradcam_b64)
                        gradcam_img = Image.open(io.BytesIO(g_bytes))
                    raw_val = float(res_data.get("raw_value", 0.5))
                    data_processed = True
            except Exception:
                data_processed = False

            # 2. Local Fallback Execution
            if not data_processed and HAS_LOCAL_PREDICT:
                try:
                    pipeline_res = run_prediction_pipeline(image_bytes, model_key=model_choice)
                    label = pipeline_res.get("label")
                    confidence = float(pipeline_res.get("confidence", 0))
                    gradcam_img = pipeline_res.get("gradcam_image")
                    raw_val = float(pipeline_res.get("raw_value", 0.5))
                    data_processed = True
                except Exception as ex:
                    st.error(f"Inference Engine Exception: {ex}")

            latency_ms = (time.time() - start_time) * 1000

        # Display Grad-CAM XAI
        with col_xai:
            st.markdown(
                """
                <div class="card-header-title">
                    <span>🔥 Grad-CAM Explainable AI Activation</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if data_processed and gradcam_img is not None:
                st.markdown('<div class="scan-wrapper">', unsafe_allow_html=True)
                st.image(gradcam_img, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Heatmap Temperature Legend
                st.markdown(
                    """
                    <div style="margin-top: 10px;">
                        <div class="legend-bar"></div>
                        <div class="legend-labels">
                            <span>🔵 Baseline / Normal Tissue</span>
                            <span>🟢 Moderate Salience</span>
                            <span>🔴 High Focal Pathology</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Heatmap activation unavailable for this scan.")

        # Clinical Decision Support Banner
        if data_processed:
            is_dr = (label == "DR")
            conf_pct = confidence * 100 if confidence <= 1.0 else confidence

            if is_dr:
                st.markdown(
                    f"""
                    <div class="diag-banner-dr">
                        <div class="diag-title-dr">
                            <span>🚨 DIAGNOSTIC FINDING: DIABETIC RETINOPATHY (DR) DETECTED</span>
                        </div>
                        <div style="color: #fecdd3; font-size: 0.98rem; line-height: 1.5; margin-top: 6px;">
                            The deep neural network identified pathological vascular changes indicative of Diabetic Retinopathy.
                            Significant activation weights are localized in the highlighted focal regions.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="diag-banner-healthy">
                        <div class="diag-title-healthy">
                            <span>✅ DIAGNOSTIC FINDING: NO DIABETIC RETINOPATHY (NO DR)</span>
                        </div>
                        <div style="color: #a7f3d0; font-size: 0.98rem; line-height: 1.5; margin-top: 6px;">
                            Retinal fundus vascular morphology appears intact with uniform macular reflection.
                            No focal microvascular abnormalities or exudates detected above baseline threshold.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Telemetry Metrics Grid
            st.markdown(
                f"""
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Model Confidence</div>
                        <div class="metric-val" style="color: {'#fb7185' if is_dr else '#34d399'};">{conf_pct:.2f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Risk Category</div>
                        <div class="metric-val" style="color: {'#f43f5e' if is_dr else '#10b981'};">{'HIGH RISK' if is_dr else 'ROUTINE'}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Inference Latency</div>
                        <div class="metric-val">{latency_ms:.0f} ms</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Processing Backbone</div>
                        <div class="metric-val" style="font-size: 1.05rem; padding-top: 5px;">{'Ben Graham' if model_choice == 'b1_bengraham' else 'Standard RGB'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Recommended Clinical Next Steps
            with st.expander("🩺 Recommended Clinical Protocol & Next Steps", expanded=is_dr):
                if is_dr:
                    st.markdown(
                        """
                        <div class="protocol-card-dr">
                            <div class="protocol-title-dr">👨‍⚕️ Specialist Referral</div>
                            <div class="protocol-desc">Urgent referral to a vitreoretinal specialist / comprehensive ophthalmologist for formal clinical dilated fundoscopy.</div>
                        </div>
                        <div class="protocol-card-dr">
                            <div class="protocol-title-dr">🔬 Ancillary Imaging</div>
                            <div class="protocol-desc">Optical Coherence Tomography (OCT) to rule out Diabetic Macular Edema (DME), and Fluorescein Angiography (FA) if neovascularization is suspected.</div>
                        </div>
                        <div class="protocol-card-dr">
                            <div class="protocol-title-dr">💉 Glycemic & Systemic Control</div>
                            <div class="protocol-desc">Coordinate with primary care provider to optimize HbA1c glycemic index, blood pressure, lipid profile, and renal function.</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="protocol-card">
                            <div class="protocol-title">👁️ Routine Surveillance</div>
                            <div class="protocol-desc">Schedule standard annual dilated retinal examination with primary eye care specialist.</div>
                        </div>
                        <div class="protocol-card">
                            <div class="protocol-title">📋 Patient Education & Lifestyle</div>
                            <div class="protocol-desc">Advise patient on maintaining tight glycemic index control, balanced diet, and promptly reporting sudden visual acuity changes or floaters.</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("<hr style='border: 1px solid rgba(56, 189, 248, 0.1); margin: 30px 0;'>", unsafe_allow_html=True)
else:
    st.info("👈 Select a clinical case from the **Demo Library** tab or **Upload custom fundus scans** above to begin automated screening.")

# ==============================================================================
# 7. Medical Educational Accordion & Regulatory Disclaimer
# ==============================================================================
with st.expander("ℹ️ Scientific Background & Regulatory Disclaimer"):
    st.markdown(
        """
        <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8; font-family: 'Outfit', sans-serif; margin-bottom: 12px;">
            🔬 Scientific Methodology & Neural Network Architecture
        </div>
        <div class="protocol-card">
            <div class="protocol-title">🧠 Convolutional Backbone Architecture</div>
            <div class="protocol-desc"><b>EfficientNet-B1</b> transfer learning architecture fine-tuned specifically on annotated clinical fundus photographs for high sensitivity Diabetic Retinopathy classification.</div>
        </div>
        <div class="protocol-card">
            <div class="protocol-title">🔥 Explainable AI (Grad-CAM) Activation</div>
            <div class="protocol-desc"><b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> calculates gradients of the diagnostic class score with respect to the feature map activations of the final convolutional layer (<code style="background: rgba(15,23,42,0.8); color: #38bdf8; padding: 2px 6px; border-radius: 4px;">top_activation</code>), providing focal spatial localization of pathological microaneurysms, hemorrhages, and exudates.</div>
        </div>
        <div class="disclaimer-callout">
            <div class="disclaimer-title">
                ⚠️ INVESTIGATIONAL CLINICAL SUPPORT TOOL
            </div>
            <div class="disclaimer-text">
                This artificial intelligence application is designed strictly for research, triage assistance, and clinical decision support. Final diagnostic and therapeutic decisions must always be established by certified ophthalmologists and qualified medical professionals.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )