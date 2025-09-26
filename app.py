# streamlit run app.py
import os, json, pathlib, pickle
import streamlit as st
import pandas as pd
import numpy as np

# Optional matplotlib (fallback if unavailable)
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    plt = None
    HAVE_MPL = False

# ================== CONFIG ==================
REPO_DIR = pathlib.Path(__file__).parent
MODEL_PATH = str(REPO_DIR / "rf_model_new.pkl")   # مكان الموديل
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
import joblib   # <<< الاستدعاء الصحيح

def _load_model(path: str):
    """Load model with joblib (fallback to pickle if needed)."""
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as e:
            st.error(f"❌ Failed to load model at {path}. Error: {e}")
            st.stop()

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

# حمل الموديل والثريشولد
model, default_th = load_model_and_threshold()

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider(
    "Anomaly Threshold", 0.90, 0.9999, default_th, step=0.0001
)

uploaded_file = st.sidebar.file_uploader("Upload CSV for Scoring", type=["csv"])

# ================== MAIN ==================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    X = df[REQUIRED_FEATURES].values
    probs = model.predict_proba(X)[:, 1]
    df["anomaly_score"] = probs
    df["anomaly_flag"] = (probs >= threshold).astype(int)

    st.success("✅ Scoring completed!")
    st.dataframe(df.head(20))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Scored CSV", csv, "scored.csv", "text/csv")

elif not uploaded_file:
    st.info("⬅️ Upload a CSV file from the sidebar to start scoring.")
