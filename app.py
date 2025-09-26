# app.py
import os, json, pickle, pathlib
import streamlit as st
import pandas as pd
import numpy as np

# ================== CONFIG ==================
REPO_DIR = pathlib.Path(__file__).parent
MODEL_PATH = str(REPO_DIR / "rf_model_new.pkl")   # خلي الملف ده جنب app.py
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.9978"))

st.set_page_config(page_title="🚨 Anomaly/Fraud Scoring — Production", layout="wide")

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
def _load_model(path: str):
    """جرب joblib لو موجود، لو مش موجود استعمل pickle"""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            return pickle.load(fh)

@st.cache_resource
def load_model_and_threshold():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()

    model = _load_model(MODEL_PATH)

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

# ================== RUN ==================
model, default_th = load_model_and_threshold()
st.success("✅ Model loaded successfully!")

# ========== مثال بسيط على input ==========
uploaded = st.file_uploader("📂 Upload CSV for scoring", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("📊 Preview:", df.head())

    try:
        preds = model.predict(df)
        st.write("✅ Predictions:", preds[:10])
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
