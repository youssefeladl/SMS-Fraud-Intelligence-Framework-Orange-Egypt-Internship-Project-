# streamlit run app.py
import os, io, zipfile, glob, json, tempfile, time, pathlib, pickle
import streamlit as st
import pandas as pd
import numpy as np

# ================== CONFIG ==================
REPO_DIR = pathlib.Path(__file__).parent
MODEL_PATH = str(REPO_DIR / "rf_model_cloud_5.pkl")   # نسخة cloud الجديدة
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
def _load_model(path: str):
    """Load Pickle-based model (saved via export_model.py)."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        st.error(f"❌ Failed to load model at {path}: {e}")
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

model, default_th = load_model_and_threshold()

# ================== HELPERS ==================
def ensure_clean(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.lower() in {
        "isolationforest_labels", "iso_label", "isolation_forest_labels"
    }]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def _build_X_for_model(df: pd.DataFrame, base_feats: list, model):
    n_expected = getattr(model, "n_features_in_", len(base_feats))

    missing = [c for c in base_feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    X = df[base_feats].astype(float)

    if n_expected == len(base_feats):
        return X

    if n_expected == 3:
        zero_col = pd.Series(0.0, index=df.index, name="msgs_per_recipient")
        X = pd.concat([X, zero_col], axis=1).astype(float)
        return X

    raise ValueError(
        f"Model expects {n_expected} features, but app is configured for {len(base_feats)}."
    )

def score_df(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = ensure_clean(df)
    try:
        X = _build_X_for_model(df, REQUIRED_FEATURES, model)
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["anomaly_score"] = proba
    out["label"] = (out["anomaly_score"] >= threshold).astype(int)
    return out

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Inputs</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload CSV / Parquet / ZIP (multi-file OK)",
        type=["csv", "parquet", "zip"],
        accept_multiple_files=True
    )
    th = st.slider("Threshold (higher ⇒ fewer anomalies)", 0.50, 1.00, float(default_th), 0.0001)
    run = st.button("▶️ Run scoring")

st.caption("Features used: distinct_B + successful_sms only.")
st.divider()

# ================== RUN SCORING ==================
if run:
    with st.spinner("Scoring…"):
        if uploads:
            dfs = []
            for f in uploads:
                df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_parquet(f)
                dfs.append(score_df(model, df, th))
            scored = pd.concat(dfs, ignore_index=True)
        else:
            st.info("Upload files first."); st.stop()

    if scored.empty:
        st.stop()

    # KPIs
    total_rate = scored["label"].mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total rows", f"{len(scored):,}")
    k2.metric("Threshold", f"{th:.4f}")
    k3.metric("# anomalies (raw)", f"{(scored['label']==1).sum():,}")
    k4.metric("Anomaly rate (raw)", f"{total_rate:.2f}%")

    # Downloads
    st.download_button(
        "⬇️ Download SCORED data",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_full.csv", mime="text/csv"
    )

    st.success("✅ Scoring finished.")
else:
    st.info("Upload files, set threshold, then press Run scoring.")
