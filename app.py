import io
import os
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Streamlit frontend for the existing Flask drought backend
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quantum Drought Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("DROUGHT_API_URL", "http://localhost:5000").rstrip("/")

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(32, 160, 130, .18), transparent 28%),
            radial-gradient(circle at 88% 12%, rgba(65, 105, 225, .16), transparent 25%),
            #0b1020;
        color: #edf2f7;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111a2e 0%, #0d1425 100%);
        border-right: 1px solid rgba(148,163,184,.18);
    }
    .hero {
        padding: 1.7rem 1.8rem;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 22px;
        background: linear-gradient(125deg, rgba(17,94,89,.36), rgba(30,41,59,.72) 55%, rgba(49,46,129,.28));
        margin-bottom: 1.3rem;
    }
    .eyebrow {
        color: #5eead4; text-transform: uppercase; letter-spacing: .16em;
        font-size: .75rem; font-weight: 700;
    }
    .hero h1 { margin: .35rem 0 .55rem 0; font-size: 2.25rem; color: #f8fafc; }
    .hero p { color: #cbd5e1; max-width: 850px; margin-bottom: 0; }
    .metric-card {
        background: rgba(15,23,42,.75);
        border: 1px solid rgba(148,163,184,.2);
        border-radius: 16px; padding: 1rem 1.1rem;
    }
    .small-label { color: #94a3b8; font-size: .82rem; }
    .big-value { color: #f8fafc; font-size: 1.55rem; font-weight: 750; margin-top: .2rem; }
    .section-title { color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin: .4rem 0 .8rem 0; }
    .hint { color: #94a3b8; font-size: .9rem; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(94,234,212,.5); border-radius: 16px; padding: .65rem;
        background: rgba(15,23,42,.45);
    }
    .stButton > button {
        border-radius: 12px; border: 1px solid rgba(94,234,212,.35);
        background: linear-gradient(90deg, #0f766e, #2563eb);
        color: white; font-weight: 700; min-height: 2.8rem;
    }
    .stButton > button:hover { border-color: #5eead4; color: white; }
    div[data-testid="stImage"] img { border-radius: 14px; }
    hr { border-color: rgba(148,163,184,.18); }
</style>
""", unsafe_allow_html=True)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "heatmap_bytes" not in st.session_state:
    st.session_state.heatmap_bytes = None
if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None
if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

with st.sidebar:
    st.markdown("## 🌱 Quantum Drought AI")
    st.caption("Satellite image analysis")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Image Analysis", "About the System"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Backend status**")
    if st.button("Check connection", use_container_width=True):
        try:
            response = requests.get(API_BASE, timeout=4)
            st.info(f"Server responded with HTTP {response.status_code}.")
        except requests.RequestException:
            st.error("Cannot reach Flask API. Start backend.py on port 5000.")
    st.caption(f"API endpoint: `{API_BASE}`")
    st.caption("Upload a satellite image to run the existing backend models.")

if page == "About the System":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Project overview</div>
      <h1>Quantum-assisted drought detection</h1>
      <p>The application checks whether an uploaded image is accepted as a satellite image,
      extracts quantum features, and predicts whether the region appears drought-affected
      or healthy using the configured backend model.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="small-label">Input</div><div class="big-value">Satellite image</div><p class="hint">PNG, JPG or JPEG</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="small-label">Feature extraction</div><div class="big-value">4-qubit circuit</div><p class="hint">2 × 2 pixel windows</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="small-label">Prediction classes</div><div class="big-value">Drought / Healthy</div><p class="hint">Backend model output</p></div>', unsafe_allow_html=True)
    st.info("The frontend displays the result returned by your Flask API. It does not retrain or change your models.")
    st.stop()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Satellite intelligence · Quantum feature extraction</div>
  <h1>Quantum Drought Intelligence</h1>
  <p>Upload a satellite image to screen the input, predict drought conditions, and optionally
  generate a quantum heatmap overlay. The interface uses your existing Flask endpoints.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.markdown('<div class="section-title">01 / Upload satellite image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a satellite image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="The backend currently resizes prediction images to 64 × 64 pixels.",
    )
    if uploaded:
        raw_bytes = uploaded.getvalue()
        st.session_state.last_image_bytes = raw_bytes
        st.session_state.last_image_name = uploaded.name
        try:
            preview = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            st.image(preview, caption=f"Input image · {uploaded.name}", use_container_width=True)
            st.caption(f"Original dimensions: {preview.width} × {preview.height}")
        except Exception:
            st.error("This file could not be previewed as an image.")

    run_prediction = st.button("Run drought analysis", use_container_width=True, disabled=uploaded is None)
    make_heatmap = st.button("Generate quantum heatmap", use_container_width=True, disabled=uploaded is None)

    if run_prediction and uploaded:
        with st.spinner("Checking image and running quantum-assisted prediction..."):
            try:
                response = requests.post(
                    f"{API_BASE}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=180,
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"error": f"Backend returned a non-JSON response (HTTP {response.status_code})."}
                if response.ok:
                    st.session_state.analysis_result = payload
                else:
                    st.session_state.analysis_result = {"error": payload.get("error", payload.get("message", f"HTTP {response.status_code}"))}
                st.session_state.heatmap_bytes = None
            except requests.RequestException as exc:
                st.session_state.analysis_result = {"error": f"Could not connect to the Flask backend: {exc}"}

    if make_heatmap and uploaded:
        with st.spinner("Generating heatmap..."):
            try:
                response = requests.post(
                    f"{API_BASE}/heatmap",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=180,
                )
                if response.ok and response.headers.get("content-type", "").startswith("image/"):
                    st.session_state.heatmap_bytes = response.content
                else:
                    try:
                        message = response.json().get("error", "Heatmap generation failed.")
                    except ValueError:
                        message = f"Heatmap endpoint returned HTTP {response.status_code}."
                    st.error(message)
            except requests.RequestException as exc:
                st.error(f"Could not connect to the Flask backend: {exc}")

with right:
    st.markdown('<div class="section-title">02 / Analysis results</div>', unsafe_allow_html=True)
    result = st.session_state.analysis_result
    if result:
        if result.get("error"):
            st.error(result["error"])
        elif result.get("result") == "Error":
            st.warning(result.get("message", "The uploaded image was rejected by the image screening model."))
        else:
            label = result.get("result", "Unknown")
            confidence = result.get("confidence", "N/A")
            if label.lower() == "drought":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="small-label">Predicted class</div><div class="big-value">{label}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="small-label">Model confidence</div><div class="big-value">{confidence}</div></div>', unsafe_allow_html=True)

            drought_prob = result.get("drought_prob")
            healthy_prob = result.get("healthy_prob")
            if drought_prob is not None and healthy_prob is not None:
                st.markdown("**Class probabilities**")
                st.progress(max(0.0, min(1.0, float(drought_prob))), text=f"Drought · {float(drought_prob) * 100:.1f}%")
                st.progress(max(0.0, min(1.0, float(healthy_prob))), text=f"Healthy · {float(healthy_prob) * 100:.1f}%")
            st.caption("Prediction reflects the currently loaded model and its training data; it is not a substitute for field assessment.")
    else:
        st.markdown("""
        <div class="metric-card">
          <div class="small-label">Waiting for analysis</div>
          <div class="big-value">No result yet</div>
          <p class="hint">Upload an image and select “Run drought analysis”. Your prediction and class probabilities will appear here.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">03 / Quantum heatmap</div>', unsafe_allow_html=True)
    if st.session_state.heatmap_bytes:
        st.image(st.session_state.heatmap_bytes, caption="Quantum feature heatmap overlay", use_container_width=True)
        st.download_button(
            "Download heatmap",
            data=st.session_state.heatmap_bytes,
            file_name="quantum_heatmap.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.markdown('<div class="metric-card"><p class="hint">Generate a heatmap to visualize the backend’s quantum scan overlay. This is a feature visualization, not a pixel-level ground-truth drought map.</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Quantum Drought Intelligence · Streamlit frontend connected to the existing Flask API")
