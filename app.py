import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile
import os
import zipfile
from PIL import Image

st.set_page_config(
    page_title="NODAC — Nodule Detection and Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F4F6F9;
    color: #1A2332;
}

.stApp { background-color: #F4F6F9; }

/* ── Header ── */
.nodac-header {
    background: #ffffff;
    border-bottom: 1px solid #D8E0EA;
    padding: 0;
    margin-bottom: 0;
}

.nodac-header-inner {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px 32px 20px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}

.nodac-wordmark {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    color: #0B4F8A;
    line-height: 1;
}

.nodac-wordmark span {
    color: #00897B;
}

.nodac-fullname {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 400;
    color: #6B7A8D;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 4px;
}

.nodac-byline {
    text-align: right;
    font-family: 'Inter', sans-serif;
}

.nodac-author {
    font-size: 0.82rem;
    font-weight: 500;
    color: #1A2332;
}

.nodac-meta {
    font-size: 0.72rem;
    color: #8A96A3;
    margin-top: 2px;
}

.nodac-uni {
    font-size: 0.7rem;
    color: #0B4F8A;
    font-weight: 500;
    margin-top: 2px;
    letter-spacing: 0.02em;
}

/* ── Stage ribbon ── */
.stage-ribbon {
    background: #0B4F8A;
    padding: 10px 32px;
    display: flex;
    gap: 48px;
    align-items: center;
}

.stage-item {
    display: flex;
    align-items: center;
    gap: 10px;
}

.stage-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #7EB8E8;
    letter-spacing: 0.12em;
}

.stage-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #ffffff;
}

.stage-metric {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #A8D5F5;
    margin-left: 6px;
}

.stage-divider {
    width: 1px;
    height: 24px;
    background: #1E6FAD;
}

/* ── Main content ── */
.nodac-main {
    max-width: 1200px;
    margin: 32px auto;
    padding: 0 32px;
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #0B4F8A;
    padding-bottom: 8px;
    border-bottom: 2px solid #0B4F8A;
    margin-bottom: 16px;
    display: inline-block;
}

/* ── Upload card ── */
.upload-card {
    background: #ffffff;
    border: 1px solid #D8E0EA;
    border-radius: 4px;
    padding: 28px 32px;
    margin-bottom: 24px;
}

.upload-instruction {
    font-size: 0.85rem;
    color: #4A5568;
    margin-bottom: 16px;
    line-height: 1.6;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #D8E0EA;
    border-radius: 4px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    background: #0B4F8A;
}

.metric-card.accent-teal::before { background: #00897B; }
.metric-card.accent-red::before  { background: #C0392B; }
.metric-card.accent-green::before { background: #27AE60; }

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #1A2332;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-val.red  { color: #C0392B; }
.metric-val.green { color: #27AE60; }

.metric-lbl {
    font-size: 0.72rem;
    color: #8A96A3;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Results panel ── */
.results-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    margin-top: 24px;
}

.region-panel {
    background: #ffffff;
    border: 1px solid #D8E0EA;
    border-radius: 4px;
    overflow: hidden;
}

.region-panel-header {
    background: #F0F4F8;
    border-bottom: 1px solid #D8E0EA;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #6B7A8D;
    text-transform: uppercase;
}

.region-row {
    border-bottom: 1px solid #EDF1F5;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: default;
}

.region-row:last-child { border-bottom: none; }

.region-left { flex: 1; }

.region-id {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6B7A8D;
    margin-bottom: 2px;
}

.region-slices {
    font-size: 0.8rem;
    font-weight: 500;
    color: #1A2332;
}

.region-scores {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #8A96A3;
    margin-top: 2px;
}

.region-badge {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 2px;
}

.badge-mal {
    background: #FDECEA;
    color: #C0392B;
    border: 1px solid #F5C0BA;
}

.badge-ben {
    background: #E8F5E9;
    color: #27AE60;
    border: 1px solid #A8D5AA;
}

.image-panel {
    background: #ffffff;
    border: 1px solid #D8E0EA;
    border-radius: 4px;
    overflow: hidden;
}

.image-panel-header {
    background: #F0F4F8;
    border-bottom: 1px solid #D8E0EA;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #6B7A8D;
    text-transform: uppercase;
}

.image-grid {
    padding: 16px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.image-item { text-align: center; }

.image-caption {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #8A96A3;
    margin-top: 6px;
}

.image-diag {
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 2px;
}

.image-diag.mal { color: #C0392B; }
.image-diag.ben { color: #27AE60; }

/* ── Disclaimer ── */
.disclaimer {
    background: #EBF4FB;
    border: 1px solid #B3D4EE;
    border-left: 3px solid #0B4F8A;
    border-radius: 2px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #2C5F8A;
    margin-top: 24px;
    line-height: 1.6;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: #0B4F8A !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: background 0.15s !important;
}

.stButton > button:hover {
    background: #0A4070 !important;
}

div[data-testid="stFileUploadDropzone"] {
    background: #F8FAFC !important;
    border: 1.5px dashed #B0BFCC !important;
    border-radius: 3px !important;
}

.stSlider > div { padding-top: 4px !important; }

[data-testid="stSlider"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    color: #4A5568 !important;
}

.stSuccess, .stError, .stWarning {
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────
STAGE1_GDRIVE_ID = "1wksEgR9bfUFC0C7tcVuO8gfnD6VKKbNK"
STAGE2_GDRIVE_ID = "1HBC-lm7fnZ2MNE7V1glhfD3BeBJPG3Ll"


def download_from_gdrive(file_id, dest_path):
    import requests
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        return
    session = requests.Session()
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "authuser": "0", "confirm": "t"}
    response = session.get(url, params=params, stream=True)
    total = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(65536):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    if total < 1_000_000:
        os.remove(dest_path)
        raise ValueError(f"Download failed ({total} bytes) for {file_id}")


@st.cache_resource
def load_models():
    import tensorflow as tf

    class EfficientNetPreprocess(tf.keras.layers.Layer):
        def call(self, x):
            return tf.keras.applications.efficientnet.preprocess_input(x)
        def get_config(self):
            return super().get_config()

    custom_objects = {"EfficientNetPreprocess": EfficientNetPreprocess}

    # Try local first (Git LFS), then Drive
    s1_local = os.path.join(os.path.dirname(__file__), "stage1_FULL.keras")
    s2_local = os.path.join(os.path.dirname(__file__), "stage2_FULL.keras")

    if os.path.exists(s1_local) and os.path.getsize(s1_local) > 1_000_000:
        s1_path, s2_path = s1_local, s2_local
    else:
        s1_path, s2_path = "/tmp/stage1_FULL.keras", "/tmp/stage2_FULL.keras"
        with st.spinner("Initialising models (first run only)..."):
            download_from_gdrive(STAGE1_GDRIVE_ID, s1_path)
            download_from_gdrive(STAGE2_GDRIVE_ID, s2_path)

    det = tf.keras.models.load_model(s1_path, custom_objects=custom_objects)
    cls = tf.keras.models.load_model(s2_path, custom_objects=custom_objects)
    return det, cls


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
                win = np.clip(img, -1000, 400)
                disp = ((win + 1000) / 1400 * 255).astype(np.uint8)
                disp = cv2.resize(disp, (img_size, img_size))
                slices_display.append(disp)
                mod = cv2.resize(((win + 1000) / 1400 * 255), (img_size, img_size)).astype(np.float32)
                slices_model.append(np.stack([mod]*3, axis=-1))
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
    fig, ax = plt.subplots(figsize=(16, 3))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#F8FAFC')

    # Subtle grid — medical graph paper feel
    ax.yaxis.grid(True, color='#E2E8F0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    colors = ['#C0392B' if s >= threshold else '#7EB8E8' for s in det_scores]
    ax.bar(range(len(det_scores)), det_scores, color=colors,
           edgecolor='none', alpha=0.9, width=0.75, zorder=2)
    ax.axhline(threshold, color='#0B4F8A', linestyle='--',
               linewidth=1.0, label=f'Detection threshold ({threshold})', zorder=3)

    for group in nodule_groups:
        ax.axvspan(group[0]-0.5, group[-1]+0.5, alpha=0.06, color='#C0392B', zorder=1)

    ax.set_xlabel("Slice index", color='#8A96A3', fontsize=8,
                  fontfamily='monospace', labelpad=8)
    ax.set_ylabel("Detection score", color='#8A96A3', fontsize=8,
                  fontfamily='monospace', labelpad=8)
    ax.set_title("Stage 1 — Nodule Detection Probability · All CT Slices",
                 color='#1A2332', fontsize=9, fontweight='600',
                 fontfamily='sans-serif', pad=10, loc='left')
    ax.set_xlim(-1, len(det_scores))
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='#8A96A3', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#D8E0EA')
        spine.set_linewidth(0.5)

    red_p  = mpatches.Patch(color='#C0392B', alpha=0.9, label='Candidate slice')
    blue_p = mpatches.Patch(color='#7EB8E8', alpha=0.9, label='Non-candidate slice')
    ax.legend(handles=[red_p, blue_p], fontsize=7.5, facecolor='#ffffff',
              edgecolor='#D8E0EA', labelcolor='#4A5568',
              loc='upper right', framealpha=0.95)
    plt.tight_layout(pad=1.2)
    return fig


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="nodac-header">
  <div class="nodac-header-inner">
    <div>
      <div class="nodac-wordmark">NOD<span>AC</span></div>
      <div class="nodac-fullname">Nodule Detection and Classification System</div>
    </div>
    <div class="nodac-byline">
      <div class="nodac-author">DA-SILVA Anthony Oluwafemi &nbsp;·&nbsp; 190410008</div>
      <div class="nodac-meta">B.Sc Biomedical Engineering &nbsp;·&nbsp; Supervisor: Prof. O. Adeleye</div>
      <div class="nodac-uni">University of Lagos, Akoka</div>
    </div>
  </div>
</div>

<div class="stage-ribbon">
  <div class="stage-item">
    <div>
      <div class="stage-num">STAGE 01</div>
      <div class="stage-label">Nodule Detection <span class="stage-metric">EfficientNet-B0 · AUC 0.84</span></div>
    </div>
  </div>
  <div class="stage-divider"></div>
  <div class="stage-item">
    <div>
      <div class="stage-num">STAGE 02</div>
      <div class="stage-label">Malignancy Classification <span class="stage-metric">EfficientNet-B2 · 89.6% Acc</span></div>
    </div>
  </div>
  <div class="stage-divider"></div>
  <div class="stage-item">
    <div>
      <div class="stage-num">TRAINING DATA</div>
      <div class="stage-label">LUNA16 + IQ-OTH/NCCD <span class="stage-metric">888 Patient CT Scans</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────
st.markdown('<div class="nodac-main">', unsafe_allow_html=True)
st.markdown('<div class="section-label">CT Scan Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown("""<p class="upload-instruction">
Upload a single <strong>.zip</strong> file containing all DICOM (.dcm) slices from one patient CT scan.
The system will automatically sort slices by filename and process them sequentially through both pipeline stages.
</p>""", unsafe_allow_html=True)

uploaded_zip = st.file_uploader(
    "Select ZIP file",
    type=["zip"],
    label_visibility="collapsed"
)

col_thresh, col_btn = st.columns([3, 1])
with col_thresh:
    threshold = st.slider(
        "Detection threshold",
        min_value=0.30, max_value=0.90,
        value=0.50, step=0.05,
        help="Slices scoring above this value are forwarded to Stage 2 classification"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Analysis →", disabled=uploaded_zip is None)

st.markdown('</div>', unsafe_allow_html=True)

# ── Inference ─────────────────────────────────────────────────
if run_btn and uploaded_zip:
    with st.spinner("Loading models..."):
        try:
            det_model, cls_model = load_models()
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.stop()

    with st.spinner("Reading DICOM slices..."):
        try:
            fnames, vol_display, vol_model = load_dicom_from_zip(uploaded_zip)
        except Exception as e:
            st.error(f"DICOM read error: {e}")
            st.stop()

    st.success(f"{len(fnames)} slices loaded successfully.")

    with st.spinner("Stage 1 — Running nodule detection..."):
        det_scores    = det_model.predict(vol_model, batch_size=16, verbose=0).ravel()
        candidate_idx = np.where(det_scores >= threshold)[0]
        nodule_groups = group_consecutive(candidate_idx, gap=2)

    with st.spinner("Stage 2 — Classifying candidate regions..."):
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

    mal_count = sum(1 for r in region_results if r['diagnosis'] == 'Malignant')
    ben_count = len(region_results) - mal_count

    # Metrics
    st.markdown('<div class="section-label" style="margin-top:24px">Analysis Results</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val">{len(fnames)}</div>
        <div class="metric-lbl">Total Slices</div>
      </div>
      <div class="metric-card accent-teal">
        <div class="metric-val">{len(candidate_idx)}</div>
        <div class="metric-lbl">Candidate Slices</div>
      </div>
      <div class="metric-card accent-red">
        <div class="metric-val red">{mal_count}</div>
        <div class="metric-lbl">Malignant Regions</div>
      </div>
      <div class="metric-card accent-green">
        <div class="metric-val green">{ben_count}</div>
        <div class="metric-lbl">Benign Regions</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart
    st.markdown('<div class="section-label">Stage 1 — Detection Score Distribution</div>',
                unsafe_allow_html=True)
    fig = make_bar_chart(det_scores, threshold, nodule_groups)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Region breakdown + images
    if region_results:
        st.markdown('<div class="section-label" style="margin-top:24px">Stage 2 — Region Classification</div>',
                    unsafe_allow_html=True)

        left, right = st.columns([1, 1.8])

        with left:
            rows_html = ""
            for res in region_results:
                badge = (f'<span class="region-badge badge-mal">Malignant</span>'
                         if res['diagnosis'] == 'Malignant'
                         else f'<span class="region-badge badge-ben">Benign</span>')
                rows_html += f"""
                <div class="region-row">
                  <div class="region-left">
                    <div class="region-id">REGION {res['region']:02d}</div>
                    <div class="region-slices">Slices {res['slices'][0]+1} – {res['slices'][-1]+1}</div>
                    <div class="region-scores">Det {res['det_score']:.3f} &nbsp;·&nbsp; Mal {res['mal_score']:.3f}</div>
                  </div>
                  {badge}
                </div>"""
            st.markdown(f"""
            <div class="region-panel">
              <div class="region-panel-header">Detected Regions · {len(region_results)} Total</div>
              {rows_html}
            </div>""", unsafe_allow_html=True)

        with right:
            show = region_results[:6]
            cols = st.columns(min(3, len(show)))
            for idx, res in enumerate(show):
                with cols[idx % 3]:
                    pil_img = Image.fromarray(vol_display[res['best_slice']]).convert("RGB")
                    st.image(pil_img, use_container_width=True)
                    diag_cls = "mal" if res['diagnosis'] == 'Malignant' else "ben"
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:4px">
                      <div class="image-diag {diag_cls}">{res['diagnosis']}</div>
                      <div class="image-caption">Slice {res['best_slice']+1} · Score {res['mal_score']:.3f}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <strong>Clinical Notice:</strong> NODAC is a computer-aided detection tool intended to assist,
      not replace, clinical radiological assessment. All findings must be reviewed and confirmed
      by a qualified radiologist before any clinical decision is made. Final diagnosis remains
      under physician supervision.
    </div>""", unsafe_allow_html=True)

elif not uploaded_zip:
    st.markdown("""
    <div style="background:#ffffff; border:1px solid #D8E0EA; border-radius:4px;
                padding:48px 32px; text-align:center; margin-top:8px;">
      <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                  letter-spacing:0.14em; color:#B0BFCC; text-transform:uppercase;
                  margin-bottom:12px;">Awaiting Input</div>
      <div style="font-size:0.9rem; color:#8A96A3; font-family:'Inter',sans-serif;
                  line-height:1.7;">
        Upload a <strong style="color:#0B4F8A">.zip file</strong> containing DICOM slices
        from a single patient CT scan to begin analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
