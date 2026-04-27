import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "codebase", "BEST_MODELS",
                          "best_0944_final_wiou_bifpn_100ep.pt")

# Use DirectShow on Windows for reliable camera enumeration
CAM_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

@st.cache_data(ttl=60, show_spinner=False)
def scan_cameras(max_check: int = 8):
    """Return list of (index, label) for cameras that actually open and stream."""
    found = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, CAM_BACKEND)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                label = f"Camera {i}"
                if i == 0:
                    label += "  (default / external)"
                elif i == 1:
                    label += "  (integrated webcam)"
                found.append((i, label))
        else:
            cap.release()
    return found if found else [(0, "Camera 0")]

st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

.stApp { background: #0f172a; color: #f1f5f9; }

[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155;
}

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.25rem;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #22c55e;
    margin: 0 0 0.25rem;
    line-height: 1.2;
}
.app-header p { color: #94a3b8; margin: 0; font-size: 0.875rem; }

/* ── Badges ── */
.badge-row { display: flex; gap: 0.5rem; margin-top: 0.75rem; flex-wrap: wrap; }
.badge {
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid;
}
.badge-green { background: #052e1620; color: #86efac; border-color: #22c55e50; }
.badge-blue  { background: #0c1e3820; color: #93c5fd; border-color: #3b82f650; }
.badge-slate { background: #1e293b;   color: #94a3b8; border-color: #334155; }

/* ── Stat cards ── */
.stats-row {
    display: flex;
    gap: 0.75rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 90px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value { font-size: 1.8rem; font-weight: 700; line-height: 1; }
.stat-label { font-size: 0.75rem; color: #64748b; margin-top: 0.25rem; }
.stat-green { color: #22c55e; }
.stat-red   { color: #ef4444; }
.stat-blue  { color: #60a5fa; }
.stat-slate { color: #e2e8f0; }

/* ── Detection list ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1rem 0 0.5rem;
}
.det-item {
    background: #1e293b;
    border: 1px solid #2d3f55;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.det-tag-bio    { background: #14532d; color: #86efac; padding: 2px 9px; border-radius: 20px; font-weight: 700; font-size: 0.75rem; white-space: nowrap; }
.det-tag-nonbio { background: #7f1d1d; color: #fca5a5; padding: 2px 9px; border-radius: 20px; font-weight: 700; font-size: 0.75rem; white-space: nowrap; }
.conf-bar-bg { flex: 1; background: #334155; border-radius: 4px; height: 5px; overflow: hidden; }
.conf-fill-bio    { height: 5px; background: #22c55e; border-radius: 4px; }
.conf-fill-nonbio { height: 5px; background: #ef4444; border-radius: 4px; }
.conf-pct { color: #94a3b8; font-size: 0.82rem; font-variant-numeric: tabular-nums; white-space: nowrap; }

/* ── Image label ── */
.img-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

/* ── Empty state ── */
.empty-state {
    background: #1e293b;
    border: 2px dashed #334155;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    color: #475569;
    margin-top: 0.5rem;
}
.empty-icon  { font-size: 2.5rem; margin-bottom: 0.5rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #64748b; }
.empty-sub   { font-size: 0.83rem; margin-top: 0.25rem; }

/* ── No detection notice ── */
.no-det {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    color: #64748b;
    text-align: center;
    font-size: 0.875rem;
}

/* ── Sidebar model card ── */
.model-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.9rem 1rem;
}
.model-card-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.82rem;
    color: #94a3b8;
}
.model-row:last-child { border-bottom: none; }
.model-val { font-weight: 600; color: #22c55e; }
.model-val-blue { font-weight: 600; color: #60a5fa; }

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #1e293b !important;
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #22c55e !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s;
}
.stDownloadButton > button:hover {
    border-color: #22c55e !important;
    color: #22c55e !important;
}

/* ── Webcam live stats (vertical) ── */
.live-stats { display: flex; flex-direction: column; gap: 0.6rem; }

/* ── Toggle / radio tweaks ── */
[data-testid="stRadio"] label { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

with st.spinner("Loading model…"):
    model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:0.1rem'>♻️ Waste Detector</div>"
        "<div style='font-size:0.75rem;color:#475569;margin-bottom:1rem'>YOLO11s · BiFPN · WIoU</div>",
        unsafe_allow_html=True,
    )

    mode = st.radio("Input Mode", ["📷 Upload Image", "🎥 Webcam"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.78rem;font-weight:600;color:#94a3b8;margin-bottom:0.5rem'>DETECTION SETTINGS</div>",
                unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", 0.10, 0.90, 0.30, 0.05,
                            help="Minimum score to show a detection")
    iou_thresh  = st.slider("IoU (NMS)",  0.10, 0.90, 0.45, 0.05,
                            help="Non-max suppression overlap threshold")

    fast_mode = False
    cam_index = 0
    if mode == "🎥 Webcam":
        st.markdown("---")
        st.markdown("<div style='font-size:0.78rem;font-weight:600;color:#94a3b8;margin-bottom:0.5rem'>WEBCAM SETTINGS</div>",
                    unsafe_allow_html=True)
        fast_mode = st.checkbox("⚡ Fast Mode (320 px)",
                                help="Halves input resolution — ~4× faster on CPU / low-end GPU")

        available_cams = scan_cameras()
        cam_labels = [label for _, label in available_cams]
        cam_indices = [idx for idx, _ in available_cams]

        if len(available_cams) == 0:
            st.warning("No cameras detected.")
        else:
            # Default to integrated webcam (index 1) if available, else first found
            default_pos = 1 if len(cam_indices) > 1 else 0
            chosen = st.selectbox("Camera", cam_labels, index=default_pos)
            cam_index = cam_indices[cam_labels.index(chosen)]

        if st.button("🔄 Refresh camera list", use_container_width=True):
            scan_cameras.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="model-card">
      <div class="model-card-title">Model Performance</div>
      <div class="model-row"><span>Val mAP@0.5</span>  <span class="model-val">94.39%</span></div>
      <div class="model-row"><span>Test mAP@0.5</span> <span class="model-val">85.46%</span></div>
      <div class="model-row"><span>Precision</span>    <span class="model-val">91.55%</span></div>
      <div class="model-row"><span>Recall</span>       <span class="model-val">89.63%</span></div>
      <div class="model-row"><span>GPU Speed</span>    <span class="model-val">4.0 ms</span></div>
      <div class="model-row"><span>Parameters</span>   <span class="model-val-blue">9.4 M</span></div>
      <div class="model-row"><span>Model Size</span>   <span class="model-val-blue">19.2 MB</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>♻️ Smart Waste Segregation System</h1>
  <p>Real-time biodegradable / non-biodegradable waste detection &nbsp;·&nbsp; Adamas University MCA 2026</p>
  <div class="badge-row">
    <span class="badge badge-green">94.39% Val mAP@0.5</span>
    <span class="badge badge-green">85.46% Test mAP@0.5</span>
    <span class="badge badge-blue">YOLO11s + BiFPN + WIoU</span>
    <span class="badge badge-slate">Bio &amp; Non-Bio</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_inference(img_bgr, imgsz=640):
    t0 = time.perf_counter()
    results = model.predict(img_bgr, conf=conf_thresh, iou=iou_thresh,
                            imgsz=imgsz, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result = results[0]
    return result, result.plot(line_width=2), elapsed_ms


def parse_detections(result):
    bio, nonbio = [], []
    if result.boxes and len(result.boxes):
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            (bio if label == "Bio" else nonbio).append(conf)
    return bio, nonbio


def stats_row_html(bio, nonbio, elapsed_ms):
    total      = len(bio) + len(nonbio)
    bio_pct    = f"{len(bio)/total*100:.0f}%" if total else "—"
    nonbio_pct = f"{len(nonbio)/total*100:.0f}%" if total else "—"
    return f"""
<div class="stats-row">
  <div class="stat-card">
    <div class="stat-value stat-slate">{total}</div>
    <div class="stat-label">Detected</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-green">{len(bio)}</div>
    <div class="stat-label">Bio &nbsp;<span style="color:#334155">{bio_pct}</span></div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-red">{len(nonbio)}</div>
    <div class="stat-label">Non-Bio &nbsp;<span style="color:#334155">{nonbio_pct}</span></div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-blue">{elapsed_ms:.0f}<span style="font-size:0.9rem;font-weight:400"> ms</span></div>
    <div class="stat-label">Infer. Time</div>
  </div>
</div>
"""


def detection_list_html(bio, nonbio):
    items = sorted(
        [("Bio", c) for c in bio] + [("Non Bio", c) for c in nonbio],
        key=lambda x: -x[1],
    )
    if not items:
        return '<div class="no-det">No waste items detected — try lowering the confidence threshold.</div>'
    rows = ""
    for label, conf in items:
        tag_cls  = "det-tag-bio"    if label == "Bio" else "det-tag-nonbio"
        fill_cls = "conf-fill-bio"  if label == "Bio" else "conf-fill-nonbio"
        rows += f"""
<div class="det-item">
  <span class="{tag_cls}">{label}</span>
  <div class="conf-bar-bg"><div class="{fill_cls}" style="width:{conf*100:.0f}%"></div></div>
  <span class="conf-pct">{conf:.1%}</span>
</div>"""
    return f'<div class="section-label">Detections (sorted by confidence)</div>{rows}'


# ── Image Upload ──────────────────────────────────────────────────────────────
if mode == "📷 Upload Image":
    uploaded = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Cap very large images at 1280 px to speed up inference on low-end hardware
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1280:
            scale   = 1280 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        col_in, col_out = st.columns(2, gap="medium")
        with col_in:
            st.markdown('<div class="img-label">Input Image</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Running detection…"):
            result, annotated, elapsed_ms = run_inference(img_bgr)

        with col_out:
            st.markdown('<div class="img-label">Detection Result</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        bio, nonbio = parse_detections(result)
        st.markdown(stats_row_html(bio, nonbio, elapsed_ms), unsafe_allow_html=True)
        st.markdown(detection_list_html(bio, nonbio), unsafe_allow_html=True)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        st.download_button(
            "⬇️ Download Annotated Image",
            buf.tobytes(),
            "waste_detection_result.jpg",
            "image/jpeg",
        )

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📁</div>
          <div class="empty-title">Upload a waste image to begin</div>
          <div class="empty-sub">JPG · PNG · BMP · WebP &nbsp;|&nbsp; Auto-resizes images larger than 1280 px</div>
        </div>
        """, unsafe_allow_html=True)

# ── Webcam ────────────────────────────────────────────────────────────────────
else:
    imgsz = 320 if fast_mode else 640

    col_stream, col_info = st.columns([3, 1], gap="medium")
    with col_stream:
        st.markdown('<div class="img-label">Live Feed</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown('<div class="img-label">Live Stats</div>', unsafe_allow_html=True)

    frame_ph = col_stream.empty()
    stats_ph = col_info.empty()

    run_cam = st.toggle("▶ Start Webcam")

    if run_cam:
        cap = cv2.VideoCapture(cam_index, CAM_BACKEND)
        if not cap.isOpened():
            st.error(f"Cannot open Camera {cam_index}. Use 🔄 Refresh camera list and try another.")
        else:
            stop_btn  = st.button("⏹ Stop")
            prev_time = time.perf_counter()

            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                result, annotated, elapsed_ms = run_inference(frame, imgsz=imgsz)
                bio, nonbio = parse_detections(result)
                total = len(bio) + len(nonbio)

                now       = time.perf_counter()
                fps       = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               use_container_width=True)

                bio_pct    = f"{len(bio)/total*100:.0f}%" if total else "—"
                nonbio_pct = f"{len(nonbio)/total*100:.0f}%" if total else "—"

                stats_ph.markdown(f"""
<div class="live-stats">
  <div class="stat-card">
    <div class="stat-value stat-slate">{total}</div>
    <div class="stat-label">Detected</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-green">{len(bio)}</div>
    <div class="stat-label">Bio &nbsp;{bio_pct}</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-red">{len(nonbio)}</div>
    <div class="stat-label">Non-Bio &nbsp;{nonbio_pct}</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-blue">{elapsed_ms:.0f}<span style="font-size:0.85rem"> ms</span></div>
    <div class="stat-label">Infer. Time</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-slate">{fps:.1f}<span style="font-size:0.85rem"> fps</span></div>
    <div class="stat-label">Frame Rate</div>
  </div>
</div>
""", unsafe_allow_html=True)

            cap.release()
            frame_ph.empty()
            stats_ph.empty()
