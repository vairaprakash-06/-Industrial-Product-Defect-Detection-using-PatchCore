"""Streamlit Web Application for Industrial Product Defect Detection using PatchCore.

Features:
- Industrial Inspection Dashboard with modern dark theme and responsive layout.
- Real-time PatchCore inference (Towards Total Recall in Industrial Anomaly Detection, CVPR 2022).
- Multi-category support (bottle, metal_nut, capsule, pill, screw, etc.).
- Image upload (PNG, JPG, JPEG) and 1-click curated test sample gallery.
- Side-by-side visual inspection: Original, Heatmap, Overlay with Bounding Contours, and Binary Defect Mask.
- Download result images (PNG overlay & heatmap) + structured JSON/TXT compliance reports.
- Comprehensive Dataset Evaluation view with Image & Pixel AUROC, Confusion Matrix, and Subtype breakdown.
- Session Inspection History with timestamped records and transient in-memory notice.
"""

import sys
import json
import time
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import cv2
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import DefectInspector
from src.download_mvtec import check_dataset_exists
from src.evaluate import evaluate_patchcore
from config import SUPPORTED_CATEGORIES, DEFAULT_CATEGORY


# Configure Streamlit Page
st.set_page_config(
    page_title="Industrial Defect Detection | PatchCore AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark industrial glassmorphism theme)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(11, 15, 25, 1) 0%, rgba(18, 24, 38, 1) 90%);
        color: #e2e8f0;
    }
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .badge-normal {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 8px 20px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    .badge-anomaly {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 8px 20px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .confusion-matrix-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        text-align: center;
    }
    .confusion-matrix-table th, .confusion-matrix-table td {
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px;
    }
    .cm-header {
        background: rgba(51, 65, 85, 0.6);
        font-weight: 600;
        color: #e2e8f0;
    }
    .cm-tp {
        background: rgba(16, 185, 129, 0.25);
        color: #34d399;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .cm-tn {
        background: rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .cm-fp, .cm-fn {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "inspection_history" not in st.session_state:
    st.session_state["inspection_history"] = []

if "session_stats" not in st.session_state:
    st.session_state["session_stats"] = {
        "total_inspected": 0,
        "normal_count": 0,
        "defect_count": 0,
    }

if "last_inspected_id" not in st.session_state:
    st.session_state["last_inspected_id"] = None


@st.cache_resource(show_spinner=False)
def load_inspector(category: str, device: str = "auto") -> DefectInspector:
    """Loads and caches the DefectInspector for the chosen product category."""
    return DefectInspector(category=category, device=device)


def check_category_status(category: str) -> dict:
    """Returns existence status of weights, dataset, and metrics for a category."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / category / f"patchcore_{category}.pt"
    metrics_path = PROJECT_ROOT / "checkpoints" / category / "metrics.json"
    data_path = PROJECT_ROOT / "data" / "mvtec" / category

    data_status = check_dataset_exists(data_path)
    has_weights = ckpt_path.is_file()
    has_metrics = metrics_path.is_file()

    metrics = None
    if has_metrics:
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    return {
        "has_weights": has_weights,
        "ckpt_path": ckpt_path,
        "has_dataset": data_status["exists"],
        "dataset_info": data_status,
        "metrics": metrics,
    }


def record_inspection(
    image_name: str,
    category: str,
    score: float,
    threshold: float,
    is_anomaly: bool,
    latency_ms: float,
    coverage_pct: float,
):
    """Logs inspection record to session state history."""
    record_id = f"{image_name}_{score:.4f}"
    if st.session_state["last_inspected_id"] == record_id:
        return  # Avoid duplicate increments on page rerenders

    st.session_state["last_inspected_id"] = record_id
    prediction = "DEFECTIVE" if is_anomaly else "NORMAL"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state["inspection_history"].append({
        "timestamp": now_str,
        "image_name": image_name,
        "category": category,
        "prediction": prediction,
        "anomaly_score": round(score, 4),
        "threshold": round(threshold, 4),
        "defect_coverage_pct": round(coverage_pct, 2),
        "latency_ms": round(latency_ms, 1),
    })

    st.session_state["session_stats"]["total_inspected"] += 1
    if is_anomaly:
        st.session_state["session_stats"]["defect_count"] += 1
    else:
        st.session_state["session_stats"]["normal_count"] += 1


def image_to_png_bytes(rgb_array: np.ndarray) -> bytes:
    """Encodes RGB numpy array into PNG bytes for Streamlit download."""
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", bgr_array)
    if not success:
        # Fallback using PIL
        pil_img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    return buffer.tobytes()


def main():
    # Sidebar Configuration & Telemetry
    with st.sidebar:
        st.markdown("### ⚙️ Inspection Configuration")
        selected_category = st.selectbox(
            "Select Product Category:",
            options=SUPPORTED_CATEGORIES,
            index=0,
            help="Choose an industrial manufactured product type.",
        )

        cat_status = check_category_status(selected_category)

        # Status Badges
        st.markdown("---")
        st.markdown("#### 📦 System Status")
        if cat_status["has_weights"]:
            st.success(f"✅ Weights ready ({selected_category})")
        else:
            st.error(f"❌ Weights missing ({selected_category})")

        if cat_status["has_dataset"]:
            train_count = cat_status["dataset_info"].get("train_count", 0)
            test_count = cat_status["dataset_info"].get("test_count", 0)
            st.info(f"📂 Dataset ready ({train_count} train / {test_count} test)")
        else:
            st.warning(f"⚠️ Dataset missing ({selected_category})")

        st.markdown("---")
        st.markdown("#### 🎛️ Visualization Settings")
        overlay_alpha = st.slider("Heatmap Blending Alpha", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        colormap_choice = st.selectbox(
            "Heatmap Palette",
            options=["JET", "TURBO", "INFERNO", "VIRIDIS"],
            index=0,
        )
        colormap_dict = {
            "JET": cv2.COLORMAP_JET,
            "TURBO": cv2.COLORMAP_TURBO,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        }

        # Session Telemetry in Sidebar
        st.markdown("---")
        st.markdown("#### ⏱️ Current Session Telemetry")
        stats = st.session_state["session_stats"]
        st.write(f"• **Inspected:** `{stats['total_inspected']}` images")
        st.write(f"• **Normal:** `{stats['normal_count']}` passed")
        st.write(f"• **Defective:** `{stats['defect_count']}` anomalies")

        if st.button("🗑️ Reset Session History", use_container_width=True):
            st.session_state["inspection_history"] = []
            st.session_state["session_stats"] = {
                "total_inspected": 0,
                "normal_count": 0,
                "defect_count": 0,
            }
            st.session_state["last_inspected_id"] = None
            st.rerun()

    # Main Header Card
    st.markdown(f"""
    <div class="header-card">
        <div class="header-title">🔍 Industrial Product Defect Detection</div>
        <div class="header-subtitle">
            Deep Learning Inspection System powered by <b>PatchCore</b> (Roth et al., CVPR 2022) |
            Active Category: <b style="color: #38bdf8;">{selected_category.upper()}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Dashboard
    stats = st.session_state["session_stats"]
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_inspected']}</div>
            <div class="stat-label">Session Inspected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #34d399;">{stats['normal_count']}</div>
            <div class="stat-label">Conformant (Normal)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #f87171;">{stats['defect_count']}</div>
            <div class="stat-label">Anomalies Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        rate = (stats['defect_count'] / stats['total_inspected'] * 100) if stats['total_inspected'] > 0 else 0.0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #c084fc;">{rate:.1f}%</div>
            <div class="stat-label">Session Defect Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Primary Navigation Tabs
    tab_inspect, tab_eval, tab_history, tab_docs = st.tabs([
        "🔬 Image Inspection",
        "📊 Dataset & Model Evaluation",
        "📜 Session History",
        "📘 Architecture & Methodology",
    ])

    # =========================================================================
    # TAB 1: IMAGE INSPECTION
    # =========================================================================
    with tab_inspect:
        if not cat_status["has_weights"]:
            st.error(
                f"### ⚠️ No Trained PatchCore Model Found for '{selected_category}'\n\n"
                f"The checkpoint file was not found at `{cat_status['ckpt_path']}`.\n\n"
                f"**To train and evaluate the model for '{selected_category}', run the following commands in PowerShell:**\n"
                f"```powershell\n"
                f"# 1. Download or verify MVTec AD dataset\n"
                f"python src/download_mvtec.py --category {selected_category}\n\n"
                f"# 2. Train PatchCore memory bank on normal samples only\n"
                f"python src/train.py --category {selected_category}\n\n"
                f"# 3. Run validation evaluation to calculate AUROC and optimal threshold\n"
                f"python src/evaluate.py --category {selected_category}\n"
                f"```\n\n"
                f"Once trained, refresh this page to begin visual inspection."
            )
        else:
            try:
                inspector = load_inspector(selected_category)
            except Exception as e:
                st.error(f"Failed to load PatchCore model: {e}")
                return

            # Threshold Sensitivity Expander
            default_thresh = inspector.threshold
            with st.expander("🛠️ Decision Threshold Calibration", expanded=False):
                st.markdown(
                    "The decision threshold ($\tau$) separates nominal products from defective ones. "
                    "The calibrated threshold is computed on the validation set to maximize F1-score. "
                    "You may temporarily adjust it below to test tolerance sensitivity:"
                )
                t_col1, t_col2 = st.columns([3, 1])
                with t_col1:
                    active_threshold = st.slider(
                        "Anomaly Score Decision Threshold:",
                        min_value=max(0.1, default_thresh * 0.3),
                        max_value=default_thresh * 2.5,
                        value=float(default_thresh),
                        step=0.01,
                        help="Scores at or above this value trigger an Anomaly Alert.",
                    )
                with t_col2:
                    st.metric("Calibrated Optimal Threshold", f"{default_thresh:.4f}")

            # Sub-tabs for Image Input: Upload vs Gallery
            subtab_upload, subtab_gallery = st.tabs(["📤 Upload Product Image", "🧪 Test Sample Gallery"])

            input_image = None
            image_source_name = None

            with subtab_upload:
                uploaded_file = st.file_uploader(
                    "Upload an industrial manufacturing product image for inspection:",
                    type=["png", "jpg", "jpeg"],
                    help="Supported formats: PNG, JPG, JPEG (RGB)",
                )
                if uploaded_file is not None:
                    try:
                        input_image = Image.open(uploaded_file).convert("RGB")
                        image_source_name = uploaded_file.name
                    except Exception as ex:
                        st.error(f"Error opening uploaded image: {ex}")

            with subtab_gallery:
                st.markdown("Select a verified test sample to immediately run PatchCore inference:")
                sample_options = {}

                # Curated samples
                curated_dir = PROJECT_ROOT / "samples"
                if curated_dir.is_dir():
                    for s_img in sorted(curated_dir.glob("*.png")):
                        label_tag = "NORMAL" if "normal" in s_img.name.lower() else "DEFECT"
                        sample_options[f"[{label_tag}] {s_img.name}"] = s_img

                # Category test folder
                cat_data = PROJECT_ROOT / "data" / "mvtec" / selected_category / "test"
                if cat_data.is_dir():
                    for defect_dir in sorted(cat_data.iterdir()):
                        if defect_dir.is_dir():
                            imgs = list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))
                            for img_p in imgs[:2]:
                                key_name = f"[{defect_dir.name.upper()}] {img_p.name}"
                                if key_name not in sample_options:
                                    sample_options[key_name] = img_p

                if sample_options:
                    chosen_sample = st.selectbox("Choose a test sample:", options=list(sample_options.keys()))
                    selected_sample_path = sample_options[chosen_sample]

                    g_col1, g_col2 = st.columns([1, 4])
                    with g_col1:
                        st.image(str(selected_sample_path), width=110, caption="Preview")
                    with g_col2:
                        st.write(f"**Sample File:** `{selected_sample_path.name}`")
                        if st.button("🚀 Inspect This Sample", type="primary", use_container_width=False):
                            input_image = Image.open(selected_sample_path).convert("RGB")
                            image_source_name = f"{selected_sample_path.parent.name}/{selected_sample_path.name}"
                            st.session_state["active_sample"] = selected_sample_path

                    if "active_sample" in st.session_state and input_image is None and uploaded_file is None:
                        stored_path = st.session_state["active_sample"]
                        if stored_path.is_file():
                            input_image = Image.open(stored_path).convert("RGB")
                            image_source_name = f"{stored_path.parent.name}/{stored_path.name}"
                else:
                    st.info(f"No test samples found in `{cat_data}`. Please use the Upload tab.")

            # Run Inspection & Render Results
            if input_image is not None:
                st.markdown("---")
                st.markdown(f"### 📋 Inspection Analysis: `{image_source_name}`")

                with st.spinner("Extracting multi-scale patch embeddings and calculating memory distances..."):
                    result = inspector.inspect_image(
                        image_input=input_image,
                        custom_threshold=active_threshold,
                        alpha=overlay_alpha,
                        colormap=colormap_dict[colormap_choice],
                    )

                score = result["anomaly_score"]
                is_anomaly = result["is_anomaly"]
                thresh = result["threshold"]
                latency_ms = result["inference_time_ms"]
                coverage_pct = result["defect_coverage_pct"]

                # Record in session history
                record_inspection(
                    image_name=image_source_name,
                    category=selected_category,
                    score=score,
                    threshold=thresh,
                    is_anomaly=is_anomaly,
                    latency_ms=latency_ms,
                    coverage_pct=coverage_pct,
                )

                # Verdict & Telemetry Row
                v_col, m1, m2, m3, m4 = st.columns([2, 1, 1, 1, 1])

                with v_col:
                    if is_anomaly:
                        st.markdown(
                            '<div class="badge-anomaly">🚨 DEFECTIVE (ANOMALY)</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("Product exceeds nominal tolerance threshold.")
                    else:
                        st.markdown(
                            '<div class="badge-normal">✅ NORMAL (PASSED)</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("Product conforms to nominal specifications.")

                with m1:
                    delta_val = score - thresh
                    st.metric(
                        label="Anomaly Score",
                        value=f"{score:.4f}",
                        delta=f"{delta_val:+.4f}",
                        delta_color="inverse",
                    )
                    st.caption("Raw Euclidean distance")

                with m2:
                    st.metric(label="Decision Threshold", value=f"{thresh:.4f}")
                    st.caption("Calibrated cut-off")

                with m3:
                    st.metric(label="Defect Surface Area", value=f"{coverage_pct:.2f}%")
                    st.caption("Pixels above threshold")

                with m4:
                    st.metric(label="Inference Latency", value=f"{latency_ms:.1f} ms")
                    st.caption("Feature search time")

                # Honest Metric Disclaimer (Mandatory Requirement)
                st.info(
                    "ℹ️ **Metric Interpretation:** The anomaly score represents the maximum Euclidean distance "
                    "between query patch embeddings and nominal patterns in the memory bank. "
                    "It is an uncalibrated geometric distance, not a probability percentage."
                )

                # Side-by-Side Visualizations
                st.markdown("#### 🔬 Visual Defect Localization")
                v1, v2, v3, v4 = st.columns(4)

                with v1:
                    st.markdown("**1. Original Product**")
                    st.image(result["original_image"], use_container_width=True)

                with v2:
                    st.markdown("**2. Anomaly Heatmap**")
                    st.image(result["heatmap_colored"], use_container_width=True)

                with v3:
                    st.markdown("**3. Defect Overlay & Contours**")
                    st.image(result["overlay"], use_container_width=True)

                with v4:
                    st.markdown("**4. Defect Binary Mask**")
                    st.image(result["defect_mask"], use_container_width=True)

                # Download Options Section
                st.markdown("---")
                st.markdown("#### 💾 Export Inspection Results & Certificate")
                d_col1, d_col2, d_col3, d_col4 = st.columns(4)

                # 1. Overlay Image Download
                with d_col1:
                    overlay_bytes = image_to_png_bytes(result["overlay"])
                    stem = Path(image_source_name).stem
                    st.download_button(
                        label="📥 Download Overlay Image (PNG)",
                        data=overlay_bytes,
                        file_name=f"{stem}_defect_overlay.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                # 2. Heatmap Image Download
                with d_col2:
                    heatmap_bytes = image_to_png_bytes(result["heatmap_colored"])
                    st.download_button(
                        label="📥 Download Heatmap (PNG)",
                        data=heatmap_bytes,
                        file_name=f"{stem}_anomaly_heatmap.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                # 3. JSON Audit Report Download
                report_data = result["report"]
                with d_col3:
                    report_json_str = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="📥 Download Audit Report (JSON)",
                        data=report_json_str,
                        file_name=f"inspection_report_{report_data['inspection_metadata']['report_id']}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                # 4. Text Certificate Download
                with d_col4:
                    cert_text = f"""=======================================================
INDUSTRIAL PRODUCT INSPECTION CERTIFICATE
=======================================================
Report ID:        {report_data['inspection_metadata']['report_id']}
Timestamp:        {report_data['inspection_metadata']['timestamp']}
Product Category: {report_data['inspection_metadata']['product_category']}
Image File:       {report_data['inspection_metadata']['image_filename']}
Resolution:       {report_data['inspection_metadata']['image_resolution']}
-------------------------------------------------------
ASSESSMENT:
Status:           {report_data['model_assessment']['status']}
Anomaly Score:    {report_data['model_assessment']['uncalibrated_anomaly_score']}
Threshold:        {report_data['model_assessment']['decision_threshold']}
Defect Coverage:  {report_data['model_assessment']['defect_surface_coverage_pct']}%
Inference Time:   {report_data['model_assessment']['inference_latency_ms']} ms
-------------------------------------------------------
DISCLAIMER:
{report_data['disclaimer']}
=======================================================
"""
                    st.download_button(
                        label="📥 Download Certificate (TXT)",
                        data=cert_text,
                        file_name=f"certificate_{report_data['inspection_metadata']['report_id']}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                with st.expander("🔍 View Machine-Readable Audit Record (JSON)"):
                    st.json(report_data)

            else:
                st.info("👆 Please upload an image or choose a sample from the test gallery to begin visual inspection.")

    # =========================================================================
    # TAB 2: DATASET & MODEL EVALUATION
    # =========================================================================
    with tab_eval:
        st.markdown(f"### 📊 Benchmark Evaluation: `{selected_category.upper()}`")
        st.markdown(
            "Evaluation metrics are computed rigorously against the official MVTec Anomaly Detection benchmark. "
            "Normal training images are used solely to build the memory bank, while test sets containing both "
            "nominal and defective parts are used for evaluation."
        )

        metrics = cat_status["metrics"]

        # Run Evaluation Button
        eval_col1, eval_col2 = st.columns([2, 1])
        with eval_col1:
            st.markdown("You can run the full test evaluation on all test images directly from this interface:")
        with eval_col2:
            if st.button("⚡ Run Full Test Set Evaluation", type="primary", use_container_width=True):
                if not cat_status["has_weights"]:
                    st.error("Cannot evaluate: Model checkpoint is missing. Please train first.")
                elif not cat_status["has_dataset"]:
                    st.error("Cannot evaluate: Dataset missing. Please download first.")
                else:
                    with st.spinner(f"Evaluating all test images for '{selected_category}'..."):
                        try:
                            eval_res = evaluate_patchcore(category=selected_category, calc_pixel_auroc=True)
                            st.success("✅ Evaluation completed successfully! Metrics updated.")
                            cat_status["metrics"] = eval_res
                            metrics = eval_res
                        except Exception as ex:
                            st.error(f"Evaluation error: {ex}")

        st.markdown("---")

        if metrics:
            # Top Metrics Cards
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric(
                    label="Image-Level AUROC",
                    value=f"{metrics.get('image_auroc', 0.0)*100:.2f}%",
                    help="Area Under Receiver Operating Characteristic Curve across all test images.",
                )
            with e2:
                pix_val = metrics.get('pixel_auroc')
                if pix_val != "N/A" and isinstance(pix_val, (int, float)):
                    st.metric(label="Pixel-Level AUROC", value=f"{pix_val*100:.2f}%")
                else:
                    st.metric(label="Pixel-Level AUROC", value="N/A")
            with e3:
                st.metric(label="Optimal Decision Threshold (τ)", value=f"{metrics.get('optimal_threshold', 0.0):.4f}")
            with e4:
                st.metric(label="F1-Score", value=f"{metrics.get('f1_score', 0.0):.4f}")

            # Confusion Matrix & Classification Stats
            st.markdown("#### 🧮 Confusion Matrix & Classification Report")
            cm = metrics.get("confusion_matrix", {})
            tp = cm.get("tp", 0)
            fp = cm.get("fp", 0)
            tn = cm.get("tn", 0)
            fn = cm.get("fn", 0)

            c_matrix_col, c_stats_col = st.columns([1, 1])

            with c_matrix_col:
                st.markdown(f"""
                <table class="confusion-matrix-table">
                    <thead>
                        <tr>
                            <th class="cm-header">Ground Truth \\ Predicted</th>
                            <th class="cm-header">Predicted Normal</th>
                            <th class="cm-header">Predicted Defective</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="cm-header"><b>Actual Normal</b></td>
                            <td class="cm-tn">TN = {tn}</td>
                            <td class="{"cm-fp" if fp > 0 else "cm-tn"}">FP = {fp}</td>
                        </tr>
                        <tr>
                            <td class="cm-header"><b>Actual Defective</b></td>
                            <td class="{"cm-fn" if fn > 0 else "cm-tn"}">FN = {fn}</td>
                            <td class="cm-tp">TP = {tp}</td>
                        </tr>
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
                st.caption("Confusion matrix computed using optimal decision threshold.")

            with c_stats_col:
                st.markdown(f"""
                - **Precision:** `{metrics.get('precision', 0.0)*100:.2f}%` ($TP / (TP + FP)$)
                - **Recall / Sensitivity:** `{metrics.get('recall', 0.0)*100:.2f}%` ($TP / (TP + FN)$)
                - **Specificity:** `{metrics.get('specificity', 0.0)*100:.2f}%` ($TN / (TN + FP)$)
                - **Total Evaluated Images:** `{metrics.get('test_samples', 0)}` ({metrics.get('normal_samples', 0)} normal, {metrics.get('anomalous_samples', 0)} anomalous)
                - **Mean Inference Latency:** `{metrics.get('average_inference_ms', 0.0):.2f} ms` / image
                """)

            # Breakdown by Defect Subtype
            defect_breakdown = metrics.get("defect_breakdown", {})
            if defect_breakdown:
                st.markdown("#### 🔬 Performance by Defect Subtype")
                rows = []
                for d_name, d_stat in defect_breakdown.items():
                    rows.append({
                        "Subtype": d_name.upper(),
                        "Samples (n)": d_stat.get("sample_count", 0),
                        "Accuracy": f"{d_stat.get('accuracy', 0.0)*100:.1f}%",
                        "Mean Anomaly Score": f"{d_stat.get('mean_anomaly_score', 0.0):.4f}",
                        "Score Range [Min, Max]": f"[{d_stat.get('min_score', 0.0):.4f}, {d_stat.get('max_score', 0.0):.4f}]",
                    })
                st.dataframe(rows, use_container_width=True)

            st.caption(
                "⚠️ **Methodology Note:** Benchmark results depend on image quality, illumination, "
                "backbone features, and coreset sampling fraction. Metrics reported above represent real "
                "evaluation executions and are never fabricated."
            )
        else:
            st.info(
                f"No evaluation metrics recorded yet for '{selected_category}'. "
                f"Click 'Run Full Test Set Evaluation' above or run `python -m src.evaluate --category {selected_category}`."
            )

    # =========================================================================
    # TAB 3: SESSION HISTORY
    # =========================================================================
    with tab_history:
        st.markdown("### 📜 Session Inspection History")
        st.markdown(
            "Maintains a sequential audit log of every product analyzed during the current browser session."
        )

        # Notice on Transient Session State (Mandatory Requirement)
        st.warning(
            "🔒 **Data Privacy & Storage Notice:** This history is maintained in transient browser memory "
            "for the active session only. It is not saved to an external database. Uploaded images are "
            "processed in-memory and are never permanently stored without explicit configuration."
        )

        history = st.session_state["inspection_history"]
        if len(history) > 0:
            st.dataframe(
                history,
                column_config={
                    "timestamp": "Timestamp",
                    "image_name": "Image File",
                    "category": "Category",
                    "prediction": "Prediction",
                    "anomaly_score": "Score",
                    "threshold": "Threshold",
                    "defect_coverage_pct": "Defect Area (%)",
                    "latency_ms": "Latency (ms)",
                },
                use_container_width=True,
            )

            # Export history as JSON
            history_json = json.dumps(history, indent=2)
            st.download_button(
                label="📥 Export Session Audit Trail (JSON)",
                data=history_json,
                file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.info("No images have been inspected yet in this session. Inspect an image in Tab 1 to build the history.")

    # =========================================================================
    # TAB 4: ARCHITECTURE & DOCUMENTATION
    # =========================================================================
    with tab_docs:
        st.markdown("### 📘 System Architecture & PatchCore Methodology")
        st.markdown("""
        #### 1. Multi-Scale Feature Extraction
        Industrial defects appear across diverse spatial scales—from hairline scratches to broad surface contaminations.
        PatchCore utilizes a pretrained convolutional backbone (e.g., **ResNet-18**):
        - Extracts intermediate features from `layer2` (high-resolution spatial context) and `layer3` (semantic representations).
        - Applies local average pooling ($p=3$, stride $1$) to aggregate neighborhood information.
        - Bilinearly aligns resolutions and concatenates feature channels ($D = 128 + 256 = 384$).

        #### 2. Greedy Minimax k-Center Coreset Selection
        Across nominal training images, hundreds of thousands of patch representations are generated.
        To ensure real-time inference without sacrificing recall, PatchCore applies **Minimax Facility Location**:
        - Accelerated by Johnson-Lindenstrauss random projection ($d_{\\text{proj}}=128$).
        - Iteratively selects the patch that maximizes the distance to the nearest existing selected point:
        $$idx^* = \\arg\\max_{i \\notin S} \\min_{j \\in S} \\|p_i - p_j\\|_2$$
        - Compresses the memory bank by 98% (retaining 2% representative patch tokens) while retaining full anomaly recall.

        #### 3. Anomaly Scoring & Spatial Heatmaps
        At inference time:
        - Query patches are compared to the memory bank via nearest-neighbor Euclidean distance.
        - The image anomaly score is the maximum patch distance:
        $$S_{\\text{image}} = \\max_{p \\in P} \\min_{m \\in \\mathcal{M}} \\|p - m\\|_2$$
        - Patch scores are mapped back to spatial coordinates, smoothed with a 2D Gaussian filter ($\\sigma=4.0$),
          and bilinearly interpolated to the original camera resolution.
        - Regions exceeding threshold $\\tau$ are segmented and contoured.
        """)


if __name__ == "__main__":
    main()
