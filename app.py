# streamlit run app.py
import os, io, zipfile, glob, json, tempfile, time, pathlib, pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG ==================
REPO_DIR = pathlib.Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", str(REPO_DIR / "rf_model.joblib"))
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

# ================== MODEL FETCH (optional via secrets) ==================
def ensure_model_local():
    """
    If MODEL_PATH exists locally, use it.
    Otherwise, try to download from st.secrets["MODEL_URL"] into /tmp.
    """
    global MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    try:
        model_url = st.secrets.get("MODEL_URL", None)
    except Exception:
        model_url = None

    if not model_url:
        st.error(
            "Model not found locally and no MODEL_URL provided in secrets. "
            "Either upload rf_model.joblib alongside app.py or set MODEL_URL in Secrets."
        )
        st.stop()

    # Lazy import to avoid ModuleNotFoundError if not needed
    try:
        import requests  # noqa
    except Exception:
        st.error(
            "Missing dependency: requests. Add 'requests>=2.31' to requirements.txt "
            "or upload the model file locally."
        )
        st.stop()

    from pathlib import Path
    import requests  # type: ignore

    tmp_path = Path("/tmp/rf_model.joblib")
    if not tmp_path.exists():
        try:
            r = requests.get(model_url, timeout=120)
            r.raise_for_status()
            tmp_path.write_bytes(r.content)
        except Exception as e:
            st.error(f"Failed to download model from MODEL_URL: {e}")
            st.stop()

    MODEL_PATH = str(tmp_path)
    return MODEL_PATH

def _load_model(path: str):
    """
    Try joblib first; if unavailable or fails, try pickle.
    """
    # Try joblib (lazy import)
    try:
        import joblib  # noqa
        return joblib.load(path)  # type: ignore
    except Exception as e_joblib:
        # Fallback to pickle
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as e_pickle:
            st.error(
                "Failed to load model. "
                "Install joblib (add 'joblib>=1.3' to requirements.txt) "
                "or provide a pickle-compatible model file."
            )
            st.stop()

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model_and_threshold():
    ensure_model_local()
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}")
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
        st.warning(f"model.classes_ = {classes} (expected [0, 1]).")
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

    # model must implement predict_proba with classes_ = [0,1]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["anomaly_score"] = proba
    out["label"] = (out["anomaly_score"] >= threshold).astype(int)
    return out

# --- Safe remove (Windows) ---
def safe_unlink(p, retries=5, delay=0.2):
    for _ in range(retries):
        try:
            os.remove(p); return
        except PermissionError:
            time.sleep(delay)
    try:
        os.remove(p)
    except Exception:
        pass

# --- Readers ---
def read_csv_in_chunks(path):
    with open(path, "rb") as fh:
        reader = pd.read_csv(fh, chunksize=CHUNK_ROWS, sep=None, engine="python")
        for chunk in reader:
            yield chunk
        try:
            reader.close()
        except Exception:
            pass

def read_any_path(path):
    low = path.lower()
    if low.endswith(".csv"):
        yield from read_csv_in_chunks(path)
    elif low.endswith(".parquet"):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            yield pd.read_parquet(io.BytesIO(data))
        except Exception as e:
            st.error(
                "Failed to read Parquet. Add 'pyarrow>=14.0' to requirements.txt "
                "or provide CSV instead."
            )
            return
    else:
        st.warning(f"Skip {os.path.basename(path)} (CSV/Parquet only)")

def process_uploaded_files(model, files, threshold: float) -> pd.DataFrame:
    parts = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                for member in z.namelist():
                    mlow = member.lower()
                    if mlow.endswith((".csv", ".parquet")):
                        with z.open(member) as mf:
                            data = mf.read()
                            if mlow.endswith(".csv"):
                                df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
                            else:
                                try:
                                    df = pd.read_parquet(io.BytesIO(data))
                                except Exception:
                                    st.error(
                                        "Failed to read Parquet inside ZIP. "
                                        "Add 'pyarrow>=14.0' to requirements.txt or use CSV."
                                    )
                                    continue
                            scored = score_df(model, df, threshold)
                            if not scored.empty:
                                parts.append(scored)
        else:
            ext = os.path.splitext(name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                for piece in read_any_path(tmp_path):
                    scored = score_df(model, piece, threshold)
                    if not scored.empty:
                        parts.append(scored)
            finally:
                safe_unlink(tmp_path)

    if not parts:
        st.error("No valid data found. Expected: distinct_B and successful_sms.")
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

def process_folder(model, folder_path: str, threshold: float) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, "**/*.*"), recursive=True)
    wanted = [p for p in files if p.lower().endswith((".csv", ".parquet"))]
    parts = []
    for p in wanted:
        for piece in read_any_path(p):
            scored = score_df(model, piece, threshold)
            if not scored.empty:
                parts.append(scored)
    if not parts:
        st.error("No valid data found in folder.")
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Inputs</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload CSV / Parquet / ZIP (multi-file OK)",
        type=["csv", "parquet", "zip"],
        accept_multiple_files=True
    )
    folder = st.text_input("Or server folder path (for very large data)")  # local use only
    th = st.slider("Threshold (higher ⇒ fewer anomalies)", 0.50, 1.00, float(default_th), 0.0001)
    run = st.button("▶️ Run scoring")

st.caption("Features used: distinct_B + successful_sms only.")
st.divider()

# ================== RUN SCORING ==================
if run:
    with st.spinner("Scoring…"):
        if uploads:
            scored = process_uploaded_files(model, uploads, th)
        elif folder.strip():
            if os.path.isdir(folder):
                scored = process_folder(model, folder.strip(), th)
            else:
                st.error("Invalid folder path."); st.stop()
        else:
            st.info("Upload files or provide a folder path."); st.stop()

    if scored.empty:
        st.stop()

    # ===== KPIs =====
    total_rate = scored["label"].mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total rows", f"{len(scored):,}")
    k2.metric("Threshold", f"{th:.4f}")
    k3.metric("# anomalies (raw)", f"{(scored['label']==1).sum():,}")
    k4.metric("Anomaly rate (raw)", f"{total_rate:.2f}%")

    st.divider()

    # ===== DOWNLOADS =====
    raw_anoms = scored[scored["label"] == 1].copy()
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download RAW anomalies (model only)",
            data=raw_anoms.to_csv(index=False).encode("utf-8"),
            file_name="anomalies_raw.csv", mime="text/csv"
        )
    with d2:
        st.download_button(
            "⬇️ Download SCORED data (all rows)",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_full.csv", mime="text/csv"
        )

    # ===== GROUP BY SENDER =====
    st.subheader("By Sender (RAW anomalies)")
    if "sending_party_hash" in raw_anoms.columns and not raw_anoms.empty:
        agg = (
            raw_anoms.groupby("sending_party_hash", dropna=False)
            .agg(
                n_rows=("label", "count"),
                worst_score=("anomaly_score", "min"),
                total_sms=("successful_sms", "sum"),
                total_distinct=("distinct_B", "sum")
            )
            .sort_values(["worst_score", "total_sms"], ascending=[True, False])
            .reset_index()
        )
        st.dataframe(agg.head(50))
        st.download_button(
            "⬇️ Download anomalies by sender",
            data=agg.to_csv(index=False).encode("utf-8"),
            file_name="anomalies_by_sender.csv", mime="text/csv"
        )
    else:
        st.info("No 'sending_party_hash' or no anomalies found.")

    # ===== TABLE & CHARTS =====
    t1, t2 = st.tabs(["📄 RAW anomalies (sample)", "📊 Charts"])
    with t1:
        st.dataframe(raw_anoms.head(200))
    with t2:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(scored["anomaly_score"], bins=60)
        ax1.set_title("Anomaly score distribution (all rows)")
        st.pyplot(fig1)

    st.success("✅ Scoring finished.")
else:
    st.info("Upload files or provide a folder path, set the threshold, and press Run scoring.")
