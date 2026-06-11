"""
app.py – Streamlit ECG Classifier Interface
============================================
Run with:  streamlit run app.py

Folder structure expected:
    project/
    ├── app.py
    ├── inference.py
    ├── requirements.txt
    ├── checkpoints/
    │   ├── baseline_cnn_best.pt
    │   └── cnn_bilstm_best.pt
    ├── part1_load_preprocess.py
    ├── part2_label_mapping.py
    ├── part5_baseline_cnn.py
    ├── part6_cnn_bilstm.py
    └── part7_train.py
"""

import json
import numpy as np
import streamlit as st

from inference import (
    load_model,
    load_from_csv,
    load_from_wfdb,
    predict,
    compute_gradcam,
    plot_ecg,
    plot_probabilities,
    plot_attention,
    build_report,
    SUPERCLASSES,
    CLASS_COLORS,
    LEAD_NAMES,
)

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "ECG Classifier",
    page_icon   = "🫀",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 600; }
    .positive  { color: #E8694C; }
    .negative  { color: #4C9BE8; }
    .section-header {
        font-size: 16px; font-weight: 600;
        color: #ccc; margin: 1.2rem 0 0.5rem;
        border-bottom: 1px solid #333; padding-bottom: 6px;
    }
    div[data-testid="stFileUploader"] { border: 1px dashed #444; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🫀 ECG Classifier")
    st.caption("PTB-XL 5-class superclass detection")
    st.divider()

    # ── Model selection ────────────────────────────────────────────
    st.markdown("### Model")
    model_name = st.radio(
        "Select model",
        options=["cnn_bilstm", "baseline_cnn"],
        format_func=lambda x: "CNN + BiLSTM (recommended)" if x == "cnn_bilstm"
                               else "Baseline CNN (faster)",
        label_visibility="collapsed",
    )

    # ── Decision threshold ─────────────────────────────────────────
    st.markdown("### Decision threshold")
    threshold = st.slider(
        "Threshold", min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Probability above this → class predicted positive",
    )

    # ── XAI options ────────────────────────────────────────────────
    st.markdown("### Explainability (XAI)")
    show_gradcam  = st.checkbox("Grad-CAM heatmap on ECG", value=True)
    show_attn     = st.checkbox("Temporal attention map", value=True,
                                disabled=(model_name != "cnn_bilstm"),
                                help="Only available for CNN-BiLSTM model")
    gradcam_class = None
    if show_gradcam:
        gradcam_class = st.selectbox(
            "Grad-CAM target class",
            options=SUPERCLASSES,
            index=0,
            help="Which class's gradient to visualise on the ECG leads",
        )

    st.divider()

    # ── File upload ────────────────────────────────────────────────
    st.markdown("### Upload ECG")
    upload_format = st.radio(
        "File format",
        options=["CSV", "WFDB (.hea + .dat)"],
        horizontal=True,
    )

    signal = None
    if upload_format == "CSV":
        st.caption("12 columns (leads) × up to 5000 rows (time steps). No header row.")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if csv_file:
            try:
                signal = load_from_csv(csv_file)
                st.success(f"✓ Loaded  shape: {signal.shape}")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    else:
        st.caption("Upload both .hea and .dat files from the same record.")
        hea_file = st.file_uploader("Upload .hea", type=["hea"])
        dat_file = st.file_uploader("Upload .dat", type=["dat"])
        if hea_file and dat_file:
            try:
                signal = load_from_wfdb(hea_file.read(), dat_file.read())
                st.success(f"✓ Loaded  shape: {signal.shape}")
            except Exception as e:
                st.error(f"Error loading WFDB: {e}")

    st.divider()
    st.caption("Model checkpoints loaded from `./checkpoints/`")
    st.caption(f"Device: {'GPU 🚀' if __import__('torch').cuda.is_available() else 'CPU'}")


# ──────────────────────────────────────────────────────────────────
# LOAD MODEL (cached — only runs once per model selection)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def get_model(name: str):
    return load_model(name)

try:
    model = get_model(model_name)
except FileNotFoundError:
    st.error(
        f"❌ Checkpoint not found: `checkpoints/{model_name}_best.pt`\n\n"
        "Make sure you've trained the model and the checkpoint exists."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────────────────────────
if signal is None:
    # ── Landing / instructions ──────────────────────────────────────
    st.title("ECG Superclass Classifier")
    st.markdown("""
    Upload a 12-lead ECG recording in the sidebar to get predictions for the
    **5 PTB-XL diagnostic superclasses**:
    """)

    cols = st.columns(5)
    descs = {
        "NORM": "Normal ECG",
        "MI"  : "Myocardial Infarction",
        "STTC": "ST/T-wave Change",
        "CD"  : "Conduction Disturbance",
        "HYP" : "Hypertrophy",
    }
    for col, sc in zip(cols, SUPERCLASSES):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{descs[sc]}</div>"
                f"<div class='metric-value' style='color:{CLASS_COLORS[sc]}'>{sc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### How to prepare your CSV")
    st.code(
        "# Export from any ECG software as:\n"
        "# 12 columns (I, II, III, aVR, aVL, aVF, V1–V6)\n"
        "# up to 5000 rows (10 seconds at 500 Hz)\n"
        "# No header, values in mV\n\n"
        "# Convert a PTB-XL WFDB record to CSV:\n"
        "import wfdb, numpy as np\n"
        "rec = wfdb.rdrecord('path/to/record')\n"
        "np.savetxt('ecg.csv', rec.p_signal, delimiter=',')",
        language="python",
    )
    st.stop()


# ── Signal loaded — run inference ───────────────────────────────────
results  = predict(model, signal, threshold=threshold)
probs    = results["probs"]
preds    = results["preds"]
tensor   = results["tensor"]

positive = [sc for i, sc in enumerate(SUPERCLASSES) if preds[i]]
negative = [sc for i, sc in enumerate(SUPERCLASSES) if not preds[i]]

# ── Header summary ──────────────────────────────────────────────────
st.title("Prediction Results")
pos_str = "  ·  ".join(
    [f"<span style='color:{CLASS_COLORS[sc]};font-weight:700'>{sc}</span>"
     for sc in positive]
) if positive else "<span style='color:#4C9BE8'>None (all negative)</span>"

st.markdown(
    f"**Positive classes detected:** {pos_str}",
    unsafe_allow_html=True,
)

# ── Top metric cards ────────────────────────────────────────────────
st.markdown("<div class='section-header'>Per-class probabilities</div>",
            unsafe_allow_html=True)

cols = st.columns(5)
for col, sc, prob, pred in zip(cols, SUPERCLASSES, probs, preds):
    cls = "positive" if pred else "negative"
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{sc}</div>"
        f"<div class='metric-value {cls}'>{prob:.2f}</div>"
        f"<div style='font-size:11px;color:#666;margin-top:4px'>"
        f"{'✓ Positive' if pred else '— Negative'}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Probability bar chart ───────────────────────────────────────────
fig_probs = plot_probabilities(probs, preds, threshold)
st.pyplot(fig_probs, use_container_width=True)

# ── Grad-CAM ────────────────────────────────────────────────────────
cam = None
if show_gradcam and gradcam_class is not None:
    class_idx = SUPERCLASSES.index(gradcam_class)
    with st.spinner(f"Computing Grad-CAM for {gradcam_class}…"):
        try:
            cam = compute_gradcam(model, tensor, class_idx, model_name)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")
            cam = None

# ── ECG waveform plot ───────────────────────────────────────────────
st.markdown("<div class='section-header'>12-Lead ECG waveform</div>",
            unsafe_allow_html=True)
if cam is not None:
    st.caption(f"Grad-CAM overlay: regions highlighted for class **{gradcam_class}** (hot = high saliency)")

fig_ecg = plot_ecg(signal, cam=cam,
                   title=f"12-Lead ECG  {'— Grad-CAM: ' + gradcam_class if cam is not None else ''}")
st.pyplot(fig_ecg, use_container_width=True)

# ── Attention map ────────────────────────────────────────────────────
if show_attn and model_name == "cnn_bilstm":
    st.markdown("<div class='section-header'>Temporal self-attention map</div>",
                unsafe_allow_html=True)
    st.caption("Shows which time windows the BiLSTM attends to when making its prediction.")
    with st.spinner("Extracting attention weights…"):
        try:
            model(tensor)   # forward pass to populate attention cache
            attn = model.get_attention_weights()
            if attn is not None:
                attn_np = attn.squeeze(0).cpu().numpy()
                fig_attn = plot_attention(attn_np)
                st.pyplot(fig_attn, use_container_width=True)
        except Exception as e:
            st.warning(f"Attention map unavailable: {e}")

# ── Export / download ────────────────────────────────────────────────
st.markdown("<div class='section-header'>Export</div>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)

report = build_report(probs, preds, model_name, threshold)
with col_a:
    st.download_button(
        label     = "⬇ Download JSON report",
        data      = json.dumps(report, indent=2),
        file_name = "ecg_prediction.json",
        mime      = "application/json",
    )

with col_b:
    raw_csv = "\n".join(
        [",".join(SUPERCLASSES)] +
        [",".join(str(round(float(p), 4)) for p in probs)]
    )
    st.download_button(
        label     = "⬇ Download probabilities CSV",
        data      = raw_csv,
        file_name = "ecg_probabilities.csv",
        mime      = "text/csv",
    )

# ── Raw prediction table ─────────────────────────────────────────────
with st.expander("Show raw prediction details"):
    import pandas as pd
    tbl = pd.DataFrame({
        "Class"      : SUPERCLASSES,
        "Probability": [round(float(p), 4) for p in probs],
        "Logit"      : [round(float(l), 4) for l in results["logits"]],
        "Prediction" : ["Positive ✓" if p else "Negative" for p in preds],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.json(report)
