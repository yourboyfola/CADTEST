import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile
import os
import zipfile
import io
from PIL import Image

st.set_page_config(
    page_title="LungScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
.hero { background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
        border-bottom: 1px solid #1e293b; padding: 3rem 2rem 2rem; text-align: center; }
.hero-title { font-size: 3rem; font-weight: 700; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem; }
.hero-sub { color: #64748b; font-size: 1rem; font-weight: 400;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem; }
.hero-desc { color: #94a3b8; font-size: 1rem; max-width: 560px;
    margin: 0 auto; line-height: 1.7; }
.stage-badge { display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 999px; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
.badge-s1 { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb44; }
.badge-s2 { background: #1a1f35; color: #a78bfa; border: 1px solid #7c3aed44; }
.metric-card { background: #0f172a; border: 1px solid #1e293b;
    border-radius: 12px; padding: 1.25rem 1.5rem; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; color: #f1f5f9;
    line-height: 1; margin-bottom: 0.25rem; }
.metric-label { font-size: 0.75rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em; }
.region-card { background: #0f172a; border-radius: 10px;
    padding: 1rem 1.25rem; margin-bottom: 0.5rem; }
.region-malignant { border-left: 3px solid #ef4444; }
.region-benign { border-left: 3px solid #22c55e; }
.region-title { font-size: 0.85rem; font-weight: 600;
    color: #e2e8f0; margin-bottom: 0.25rem; }
.region-detail { font-size: 0.75rem; color: #64748b;
    font-family: 'JetBrains Mono', monospace; }
.diag-malignant { color: #ef4444; font-weight: 600; }
.diag-benign { color: #22c55e; font-weight: 600; }
.section-header { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b; }
.stButton > button { background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white; border: none; border-radius: 8px; padding: 0.75rem 2rem;
    font-weight: 600; font-size: 0.95rem; width: 100%; }
.warning-box { background: #1a1205; border: 1px solid #854d0e44;
    border-radius: 8px; padding: 0.75rem 1rem; color: #fbbf24;
    font-size: 0.85rem; margin-top: 1rem; }
.info-pill { display: inline-block; background: #0f172a;
    border: 1px solid #1e293b; border-radius: 6px; padding: 0.2rem 0.6rem;
    font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


STAGE1_GDRIVE_ID = "1wksEgR9bfUFC0C7tcVuO8gfnD6VKKbNK"
STAGE2_GDRIVE_ID = "1HBC-lm7fnZ2MNE7V1glhfD3BeBJPG3Ll"


def download_from_gdrive(file_id, dest_path):
    import requests
    if os.path.exists(dest_path):
        return
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url, params={"confirm": token, "id": file_id}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_models():
    import tensorflow as tf

    class EfficientNetPreprocess(tf.keras.layers.Layer):
        def call(self, x):
            return tf.keras.applications.efficientnet.preprocess_input(x)
        def get_config(self):
            return super().get_config()

    s1_path = "/tmp/stage1_FULL.keras"
    s2_path = "/tmp/stage2_FULL.keras"

    with st.spinner("Downloading Stage 1 model (first run only)..."):
        download_from_gdrive(STAGE1_GDRIVE_ID, s1_path)
    with st.spinner("Downloading Stage 2 model (first run only)..."):
        download_from_gdrive(STAGE2_GDRIVE_ID, s2_path)

    custom_objects = {"EfficientNetPreprocess": EfficientNetPreprocess}
    det_model = tf.keras.models.load_model(s1_path, custom_objects=custom_objects)
    cls_model = tf.keras.models.load_model(s2_path, custom_objects=custom_objects)
    return det_model, cls_model


def load_dicom_from_zip(zip_file, img_size=224):
    import pydicom
    slices_display, slices_model, fnames = [], [], []

    with zipfile.ZipFile(zip_file, 'r') as z:
        dcm_names = sorted([n for n in z.namelist()
                            if n.lower().endswith('.dcm')
                            and not os.path.basename(n).startswith('.')])
        for name in dcm_names:
            with z.open(name) as f:
                with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
            try:
                dcm = pydicom.dcmread(tmp_path)
                img = dcm.pixel_array.astype(np.float32)
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
                img_win = np.clip(img, -1000, 400)
                img_display = ((img_win + 1000) / 1400 * 255).astype(np.uint8)
                img_display = cv2.resize(img_display, (img_size, img_size))
                slices_display.append(img_display)
                img_model = cv2.resize(
                    ((img_win + 1000) / 1400 * 255),
                    (img_size, img_size)
                ).astype(np.float32)
                slices_model.append(np.stack([img_model]*3, axis=-1))
                fnames.append(os.path.basename(name))
            finally:
                os.unlink(tmp_path)

    return fnames, np.array(slices_display), np.array(slices_model, dtype=np.float32)


def group_consecutive(indices, gap=2):
    if len(indices) == 0: return []
    groups, current = [], [indices[0]]
    for i in indices[1:]:
        if i - current[-1] <= gap: current.append(i)
        else: groups.append(current); current = [i]
    groups.append(current)
    return groups


def make_bar_chart(det_scores, threshold, nodule_groups):
    fig, ax = plt.subplots(figsize=(16, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    colors = ['#ef4444' if s >= threshold else '#1e40af' for s in det_scores]
    ax.bar(range(len(det_scores)), det_scores, color=colors,
           edgecolor='none', alpha=0.9, width=0.8)
    ax.axhline(threshold, color='#f8fafc', linestyle='--',
               linewidth=1.2, label=f'Threshold ({threshold})', alpha=0.6)
    for group in nodule_groups:
        ax.axvspan(group[0]-0.5, group[-1]+0.5, alpha=0.07, color='#ef4444')
    ax.set_xlabel("Slice Index", color='#64748b', fontsize=9)
    ax.set_ylabel("Detection Score", color='#64748b', fontsize=9)
    ax.set_title("Stage 1 — Nodule Detection Scores Across All CT Slices",
                 color='#e2e8f0', fontsize=11, fontweight='600', pad=12)
    ax.set_xlim(-1, len(det_scores)); ax.set_ylim(0, 1.05)
    ax.tick_params(colors='#475569', labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#1e293b')
    red_p  = mpatches.Patch(color='#ef4444', alpha=0.9, label='Candidate slice')
    blue_p = mpatches.Patch(color='#1e40af', alpha=0.9, label='Normal slice')
    ax.legend(handles=[red_p, blue_p], fontsize=8, facecolor='#0f172a',
              edgecolor='#1e293b', labelcolor='#94a3b8')
    plt.tight_layout()
    return fig


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-sub">University of Lagos · Final Year Project</div>
    <div class="hero-title">LungScan AI</div>
    <div class="hero-desc">
        A two-stage deep learning pipeline for automated pulmonary nodule
        detection and malignancy classification in helical CT scans.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="metric-card">
        <div class="stage-badge badge-s1">Stage 1</div>
        <div class="metric-value">EfficientNet‑B0</div>
        <div class="metric-label">Nodule Detection · AUC 0.84</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <div class="stage-badge badge-s2">Stage 2</div>
        <div class="metric-value">EfficientNet‑B2</div>
        <div class="metric-label">Malignancy Classification · 89.6% Acc</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <div class="stage-badge badge-s1">Dataset</div>
        <div class="metric-value">LUNA16</div>
        <div class="metric-label">888 Unique Patient CT Scans</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Upload CT Scan</div>', unsafe_allow_html=True)

uploaded_zip = st.file_uploader(
    "Upload a ZIP file containing all DICOM (.dcm) files from a single CT scan",
    type=["zip"],
    help="Zip all .dcm files from one patient scan folder and upload here"
)

threshold = st.slider("Detection threshold", 0.3, 0.9, 0.5, 0.05)
run_btn = st.button("🔬 Run Analysis", disabled=uploaded_zip is None)

if uploaded_zip and not run_btn:
    st.markdown(f"""<div class="warning-box">
        ZIP file loaded — click <strong>Run Analysis</strong> to begin.
    </div>""", unsafe_allow_html=True)

if run_btn and uploaded_zip:
    with st.spinner("Loading models..."):
        try:
            det_model, cls_model = load_models()
        except Exception as e:
            st.error(f"Could not load models: {e}")
            st.stop()

    with st.spinner("Extracting and processing DICOM slices..."):
        try:
            fnames, vol_display, vol_model = load_dicom_from_zip(uploaded_zip)
        except Exception as e:
            st.error(f"Error reading DICOM files: {e}")
            st.stop()

    st.success(f"Loaded {len(fnames)} slices successfully.")

    with st.spinner("Running Stage 1 detection..."):
        det_scores    = det_model.predict(vol_model, batch_size=16, verbose=0).ravel()
        candidate_idx = np.where(det_scores >= threshold)[0]
        nodule_groups = group_consecutive(candidate_idx, gap=2)

    with st.spinner("Running Stage 2 classification..."):
        region_results = []
        for i, group in enumerate(nodule_groups):
            best_idx  = group[np.argmax(det_scores[group])]
            patch     = vol_model[best_idx:best_idx+1]
            mal_score = float(cls_model.predict(patch, verbose=0)[0][0])
            diagnosis = "Malignant" if mal_score >= 0.5 else "Benign"
            region_results.append({
                "region": i+1, "slices": group, "best_slice": best_idx,
                "det_score": float(det_scores[group].max()),
                "mal_score": mal_score, "diagnosis": diagnosis
            })

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    mal_count = sum(1 for r in region_results if r['diagnosis'] == 'Malignant')
    ben_count = len(region_results) - mal_count

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(fnames)}</div>
            <div class="metric-label">Total Slices</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(candidate_idx)}</div>
            <div class="metric-label">Candidate Slices</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ef4444">{mal_count}</div>
            <div class="metric-label">Malignant Regions</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#22c55e">{ben_count}</div>
            <div class="metric-label">Benign Regions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Stage 1 — Detection Score Distribution</div>',
                unsafe_allow_html=True)
    fig = make_bar_chart(det_scores, threshold, nodule_groups)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    if region_results:
        st.markdown('<div class="section-header">Stage 2 — Region Classification</div>',
                    unsafe_allow_html=True)
        left, right = st.columns([1, 2])

        with left:
            for res in region_results:
                cls = "region-malignant" if res['diagnosis'] == 'Malignant' else "region-benign"
                diag_cls = "diag-malignant" if res['diagnosis'] == 'Malignant' else "diag-benign"
                st.markdown(f"""
                <div class="region-card {cls}">
                    <div class="region-title">Region {res['region']}
                        <span class="{diag_cls}" style="float:right">{res['diagnosis']}</span>
                    </div>
                    <div class="region-detail">
                        Slices {res['slices'][0]+1}–{res['slices'][-1]+1} &nbsp;|&nbsp;
                        Det: {res['det_score']:.3f} &nbsp;|&nbsp;
                        Mal: {res['mal_score']:.3f}
                    </div>
                </div>""", unsafe_allow_html=True)

        with right:
            show = region_results[:6]
            cols = st.columns(min(3, len(show)))
            for idx, res in enumerate(show):
                with cols[idx % 3]:
                    pil_img = Image.fromarray(vol_display[res['best_slice']]).convert("RGB")
                    color = "#ef4444" if res['diagnosis'] == 'Malignant' else "#22c55e"
                    st.image(pil_img, use_container_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:-0.5rem">
                        <span style="font-size:0.7rem; color:{color}; font-weight:600">
                            {res['diagnosis']} · {res['mal_score']:.3f}
                        </span><br>
                        <span class="info-pill">Slice {res['best_slice']+1}</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box" style="background:#0f1a0f; border-color:#16653144; color:#86efac">
        ⚕️ <strong>Clinical Note:</strong> This system is an assistive diagnostic tool only.
        All findings must be reviewed and confirmed by a qualified radiologist before any
        clinical decision is made.
    </div>
    """, unsafe_allow_html=True)

elif not uploaded_zip:
    st.markdown("""
    <div style="border: 2px dashed #1e293b; border-radius: 16px; padding: 3rem;
                text-align: center; background: #0f172a;">
        <div style="font-size:2.5rem; margin-bottom:1rem">🫁</div>
        <div style="color:#475569; font-size:0.9rem">
            Upload a <strong style="color:#64748b">.zip file</strong> containing
            all DICOM files from a single CT scan
        </div>
    </div>
    """, unsafe_allow_html=True)
