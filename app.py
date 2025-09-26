# streamlit run app.py
import os, io, zipfile, glob, json, tempfile, time, pathlib
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load   # <<< استعمل joblib فقط

# Optional matplotlib (fallback if unavailable)
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    plt = None
    HAVE_MPL = False

# ================== CONFIG ==================
REPO_DIR = pathlib.Path(__file__).parent
MODEL_PATH = str(REPO_DIR / "rf_model_new.pkl")   # لازم الملف يبقى مرفوع جنب app.py
REQUIRED_FEATURES = ["distinct_B", "successful_sms"]
CHUNK_ROWS = 2_000_000
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.9978"))

st.set_page_config(page_title="Anomaly Scoring — Production", layout="wide")

# ================== STYLES ==================
st.markdown(
    """
    <style>
      .app-title {font-size: 1.9rem; font-weight: 800; margin: 0.4rem 0 0.2rem 0}
      .subtle {color: #6b7280}
      .stMetric {border-radius: 16px; padding: 6px 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08)}
      div[data-testid="stSidebar"] .sidebar-title {font-weight:700; font-size:1.05rem;}
    </style>
    <div class="app-title">🚨 Anomaly/Fraud Scoring — Production</div>
    <div class="subtle">Clean UI · English only ·</div>
    """,
    unsafe_allow_html=True,
)

# ================== MODEL LOAD ==================
@st.cache_resource
def load_model_and_threshold():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found. Upload rf_model_new.pkl next to app.py.")
        st.stop()

    try:
        model = load(MODEL_PATH)   # joblib فقط
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    th = DEFAULT_THRESHOLD
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                th = float(json.load(f).get("threshold", DEFAULT_THRESHOLD))
        except Exception:
            pass

    classes = getattr(model, "classes_", None)
    if classes is not None and list(classes) != [0, 1]:
        st.warning(f"⚠️ model.classes_ = {classes} (expected [0, 1]).")
    return model, th

model, default_th = load_model_and_threshold()
