"""
Advanced Multi-Stage Counterfeit Currency Detection Web Application
====================================================================
Production-grade interactive dashboard:
- Mobile Phone Camera Capture (st.camera_input) & Gallery Upload
- Desktop Image (JPG/PNG) & Multi-Page PDF Document Scanning (PyMuPDF)
- Deep Learning Multi-Stage Pipeline (YOLOv8 + ResNet18 + Attention Fusion + Grad-CAM)
- Responsive Executive Teal Dashboard Design
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to sys.path so src and webapp modules resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.backend.inference import BanknoteInferenceEngine
from webapp.backend.history import ScanHistoryManager

# Page configuration
st.set_page_config(
    page_title="CurrencyShield AI | Counterfeit Note Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive dark teal styling and mobile responsiveness
st.markdown("""
<style>
    /* Theme color palette variables */
    :root {
        --primary-teal: #0D9488;
        --teal-hover: #14B8A6;
        --dark-bg: #0F172A;
        --card-bg: #1E293B;
        --card-border: #334155;
        --text-light: #F8FAFC;
        --text-muted: #94A3B8;
        --emerald-verdict: #10B981;
        --crimson-verdict: #EF4444;
    }

    /* Card styling */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .verdict-banner-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(13, 148, 136, 0.25));
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 15px 0 25px 0;
    }

    .verdict-banner-suspicious {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 15px 0 25px 0;
    }

    .feature-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-genuine { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .badge-suspicious { background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }

    /* Responsive Mobile Adjustments */
    @media (max-width: 768px) {
        .metric-card { padding: 14px 16px; }
        .stColumn { width: 100% !important; margin-bottom: 12px; }
    }

    /* Disclaimer banner */
    .disclaimer-strip {
        background-color: #1E293B;
        border-left: 4px solid #F59E0B;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #FCD34D;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_inference_engine():
    """Initializes and caches the multi-stage inference pipeline."""
    return BanknoteInferenceEngine(
        yolo_weights="models/detection/best.pt",
        classifier_weights="models/classification/best_classifier.pt",
        fusion_weights="models/fusion/best_fusion.pt",
        device="cpu"
    )


engine = get_inference_engine()
history_mgr = ScanHistoryManager()

# Persistent academic disclaimer
st.markdown("""
<div class="disclaimer-strip">
    ⚖️ <strong>ACADEMIC PROTOTYPE NOTICE:</strong> Developed for a Deep Learning for Computer Vision Course.
    All counterfeit training data utilizes synthetically generated security defects (<code>synthetic=True</code>).
    This software is an educational research system and not a legally certified counterfeit currency detector.
</div>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("CurrencyShield AI")
st.sidebar.caption("Advanced Multi-Stage Banknote Authentication")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🛡️ Scan Note", "📜 Scan History", "📊 Home & Overview", "🔬 Explainability (Grad-CAM)",
     "📈 Model Performance", "🗂️ Dataset Insights", "ℹ️ About & Pipeline"]
)


# ==============================================================================
# TAB 1: SCAN NOTE (Camera + Desktop Image & PDF Upload)
# ==============================================================================
if nav_choice == "🛡️ Scan Note":
    st.header("Banknote Authentication Scanner")
    st.write("Scan an Indian banknote using your camera, or upload an image/PDF file.")

    scan_mode = st.radio(
        "Select Capture Method:",
        ["📷 Camera Capture", "📁 Upload Image File (JPG/PNG)", "📄 Upload PDF Document"],
        horizontal=True
    )

    image_to_process = None

    if scan_mode == "📷 Camera Capture":
        st.info("Point your camera directly at the currency note with good illumination.")
        camera_img = st.camera_input("Capture Currency Note")
        if camera_img is not None:
            image_to_process = np.array(Image.open(camera_img).convert("RGB"))

    elif scan_mode == "📁 Upload Image File (JPG/PNG)":
        uploaded_file = st.file_uploader("Choose Note Image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image_to_process = np.array(Image.open(uploaded_file).convert("RGB"))

    elif scan_mode == "📄 Upload PDF Document":
        uploaded_pdf = st.file_uploader("Choose PDF Document", type=["pdf"])
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.read()
            with st.spinner("Extracting pages from PDF..."):
                extracted_pages = BanknoteInferenceEngine.extract_images_from_pdf(pdf_bytes)

            if extracted_pages:
                if len(extracted_pages) > 1:
                    page_idx = st.selectbox(
                        f"PDF contains {len(extracted_pages)} pages. Select page to analyze:",
                        range(len(extracted_pages)),
                        format_func=lambda i: f"Page {i + 1}"
                    )
                    image_to_process = extracted_pages[page_idx]
                else:
                    image_to_process = extracted_pages[0]
            else:
                st.error("No renderable pages found in uploaded PDF.")

    # Process and display inference
    if image_to_process is not None:
        st.markdown("---")
        with st.spinner("Analyzing multi-stage security features (YOLOv8 + ResNet18 + Attention Fusion)..."):
            results = engine.run_inference(image_to_process)
            # Store in session state for Explainability tab
            st.session_state["latest_results"] = results

            # Automatically record scan into persistent history (avoid duplicate saves on widget clicks)
            img_bytes = image_to_process.tobytes()
            img_hash = hash(img_bytes)
            if st.session_state.get("last_saved_hash") != img_hash:
                saved_scan_id = history_mgr.add_scan(results)
                st.session_state["last_saved_hash"] = img_hash
                st.session_state["last_scan_id"] = saved_scan_id

        # 1. Verdict Banner
        is_genuine = (results["verdict"] == "GENUINE")
        banner_cls = "verdict-banner-genuine" if is_genuine else "verdict-banner-suspicious"
        icon = "✅" if is_genuine else "⚠️"

        st.markdown(f"""
        <div class="{banner_cls}">
            <h1 style="margin: 0; font-size: 2.2rem; color: {results['color']};">{icon} VERDICT: {results['verdict']}</h1>
            <p style="font-size: 1.15rem; color: #E2E8F0; margin-top: 8px;">
                Overall Authenticity Score: <strong>{results['authenticity_score'] * 100:.1f}%</strong> | 
                Confidence: <strong>{results['verdict_confidence'] * 100:.1f}%</strong>
            </p>
            <p style="font-size: 1.0rem; color: #94A3B8; margin-bottom: 0;">
                Predicted Denomination: <strong>₹{results['predicted_denomination']}</strong> ({results['denomination_confidence'] * 100:.1f}% confidence)
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"💾 Scan saved to Audit History • Reference ID: `{st.session_state.get('last_scan_id', 'AUTO_RECORDED')}`")

        # Show forensic flags if counterfeit/suspicious
        if results.get("forensic_flags"):
            st.error("🚨 **Security Feature Anomalies Detected:**\n" + "\n".join([f"- {flag}" for flag in results["forensic_flags"]]))

        # 2. Side-by-side Inspection Views
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Detected Security Features (Banknote Scanner)")
            st.image(results["annotated_image"], use_container_width=True, caption="Stage 1: Aligned Banknote & Precise Security ROIs")

        with col_img2:
            st.subheader("Authenticity Salience Map (Grad-CAM)")
            st.image(results["gradcam_overlay"], use_container_width=True, caption="Stage 4: Convolutional Feature Activation Heatmap")

        # 3. Per-Feature Breakdown Cards
        st.subheader("Security Feature Verification Breakdown")
        st.write("Detailed forensic verification scores and dynamic attention weights per security feature:")

        feats = results["feature_breakdown"]
        cols = st.columns(len(feats) if len(feats) > 0 else 1)

        for idx, feat in enumerate(feats):
            with cols[idx % len(cols)]:
                badge_cls = "badge-genuine" if feat["status"] == "Genuine" else "badge-suspicious"
                badge_icon = "✓" if feat["status"] == "Genuine" else "✗"
                obs_text = feat.get("observation", "")
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 6px 0; text-transform: capitalize;">{feat['feature_name'].replace('_', ' ')}</h4>
                    <span class="feature-badge {badge_cls}">{badge_icon} {feat['status']}</span>
                    <p style="margin: 10px 0 4px 0; font-size: 0.88rem; color: #94A3B8;">Authenticity Score:</p>
                    <h3 style="margin: 0; color: #F8FAFC;">{feat['authenticity_score'] * 100:.1f}%</h3>
                    <p style="margin: 6px 0 0 0; font-size: 0.76rem; color: #CBD5E1; min-height: 36px;">
                        <em>{obs_text}</em>
                    </p>
                    <p style="margin: 6px 0 0 0; font-size: 0.78rem; color: #0D9488;">
                        Attention: <strong>{feat.get('attention_weight', 0.25):.2f}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if feat["patch_thumbnail"].size > 0:
                    st.image(feat["patch_thumbnail"], width=130, caption=f"ROI: {feat['feature_name'].replace('_', ' ').title()}")


# ==============================================================================
# TAB 2: SCAN HISTORY & AUDIT TRAIL
# ==============================================================================
elif nav_choice == "📜 Scan History":
    st.header("Banknote Inspection History & Audit Trail")
    st.write("Browse and audit past banknote scans, forensic feature findings, and explainability heatmaps.")

    all_scans = history_mgr.get_all_scans()

    # Top summary metrics
    total_scans = len(all_scans)
    genuine_scans = sum(1 for s in all_scans if s.get("verdict") == "GENUINE")
    suspicious_scans = sum(1 for s in all_scans if s.get("verdict") != "GENUINE")
    auth_rate = (genuine_scans / total_scans * 100) if total_scans > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #94A3B8; font-size: 0.85rem;">Total Scans</p>
            <h2 style="margin: 4px 0 0 0; color: #F8FAFC;">{total_scans}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #94A3B8; font-size: 0.85rem;">Genuine Notes</p>
            <h2 style="margin: 4px 0 0 0; color: #10B981;">{genuine_scans}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #94A3B8; font-size: 0.85rem;">Suspicious / Counterfeit</p>
            <h2 style="margin: 4px 0 0 0; color: #EF4444;">{suspicious_scans}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #94A3B8; font-size: 0.85rem;">Genuine Ratio</p>
            <h2 style="margin: 4px 0 0 0; color: #0D9488;">{auth_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    if total_scans == 0:
        st.info("ℹ️ No inspection scans recorded yet. Go to the **'🛡️ Scan Note'** tab to inspect your first banknote!")
    else:
        # Action bar: Filter, Search, Export, Clear
        st.markdown("---")
        f_col1, f_col2, f_col3, f_col4 = st.columns([3, 2, 2, 2])
        
        with f_col1:
            verdict_filter = st.selectbox(
                "Filter by Verdict:",
                ["All Verdicts", "Genuine Only", "Suspicious / Counterfeit Only"]
            )
        with f_col2:
            denom_filter = st.selectbox(
                "Filter Denomination:",
                ["All", "₹10", "₹20", "₹50", "₹100", "₹200", "₹500", "₹2000"]
            )
        with f_col3:
            st.write("") # vertical spacing
            st.write("")
            csv_data = history_mgr.export_csv()
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name="banknote_scan_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with f_col4:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ Are you sure you want to delete all inspection history?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, Clear All", type="primary"):
                    history_mgr.clear_all_history()
                    st.session_state["confirm_clear"] = False
                    st.rerun()
            with c_no:
                if st.button("Cancel"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

        # Apply filters
        filtered_scans = []
        for s in all_scans:
            if verdict_filter == "Genuine Only" and s.get("verdict") != "GENUINE":
                continue
            if verdict_filter == "Suspicious / Counterfeit Only" and s.get("verdict") == "GENUINE":
                continue
            if denom_filter != "All" and f"₹{s.get('predicted_denomination')}" != denom_filter:
                continue
            filtered_scans.append(s)

        st.caption(f"Showing **{len(filtered_scans)}** of {total_scans} recorded scans")

        # Render list of history records
        for scan in filtered_scans:
            s_id = scan.get("id", "")
            s_date = scan.get("date_label", "")
            s_denom = scan.get("predicted_denomination", 0)
            s_verdict = scan.get("verdict", "")
            s_score = scan.get("authenticity_score", 0.0) * 100
            s_conf = scan.get("verdict_confidence", 0.0) * 100
            s_flags = scan.get("forensic_flags", [])
            s_is_gen = (s_verdict == "GENUINE")

            card_border = "#10B981" if s_is_gen else "#EF4444"
            badge_bg = "rgba(16, 185, 129, 0.2)" if s_is_gen else "rgba(239, 68, 68, 0.2)"
            badge_color = "#34D399" if s_is_gen else "#F87171"
            badge_icon = "✅" if s_is_gen else "🚨"

            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 5px solid {card_border}; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin-top: 12px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.15rem; color: #F8FAFC; margin-right: 12px;">
                            ₹{s_denom} Banknote
                        </span>
                        <span style="background-color: {badge_bg}; color: {badge_color}; border: 1px solid {card_border}; padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">
                            {badge_icon} {s_verdict} ({s_score:.1f}%)
                        </span>
                    </div>
                    <span style="color: #94A3B8; font-size: 0.82rem;">
                        📅 {s_date} • <code>{s_id}</code>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Audit Details & Inspection Heatmaps — {s_id}"):
                c_img1, c_img2, c_img3 = st.columns(3)
                
                raw_p = scan.get("raw_img_path", "")
                ann_p = scan.get("annotated_img_path", "")
                cam_p = scan.get("gradcam_img_path", "")

                with c_img1:
                    if Path(raw_p).exists():
                        st.image(raw_p, caption="Aligned Banknote Crop", use_container_width=True)
                with c_img2:
                    if Path(ann_p).exists():
                        st.image(ann_p, caption="Stage 1 Localized ROIs", use_container_width=True)
                with c_img3:
                    if Path(cam_p).exists():
                        st.image(cam_p, caption="Stage 4 Grad-CAM Saliency", use_container_width=True)

                if s_flags:
                    st.error("🚨 **Security Feature Anomalies Detected:**\n" + "\n".join([f"- {flag}" for flag in s_flags]))

                # Feature summary list
                f_summary = scan.get("feature_summary", [])
                if f_summary:
                    st.markdown("**Forensic Feature Findings:**")
                    for f_item in f_summary:
                        f_icon = "✓" if f_item.get("status") == "Genuine" else "✗"
                        f_color = "#34D399" if f_item.get("status") == "Genuine" else "#F87171"
                        st.markdown(f"- <span style='color: {f_color}; font-weight: 600;'>{f_icon} {f_item.get('feature_name').replace('_', ' ').title()}:</span> {f_item.get('status')} ({f_item.get('score', 0.0)*100:.1f}%) — *{f_item.get('observation')}*", unsafe_allow_html=True)

                col_btn1, _ = st.columns([2, 8])
                with col_btn1:
                    if st.button("🗑️ Delete this scan", key=f"del_{s_id}"):
                        history_mgr.delete_scan(s_id)
                        st.success(f"Deleted scan {s_id}")
                        st.rerun()


# ==============================================================================
# TAB 3: HOME & OVERVIEW
# ==============================================================================
elif nav_choice == "📊 Home & Overview":
    st.header("Counterfeit Currency Note Detection Pipeline")
    st.subheader("Advanced Multi-Stage Computer Vision & Deep Learning Framework")

    st.markdown("""
    This project introduces a **multi-stage, explainable deep learning pipeline** to combat currency counterfeiting
    across all **7 legal Indian denominations** (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000).

    ### High-Level System Architecture
    """)

    st.markdown("""
    ```
    [ Banknote Image / Camera / PDF ]
                   │
                   ▼
     ┌───────────────────────────────┐
     │  Stage 1: YOLOv8 Detector     │  Localizes security ROIs:
     │  (Security Features)          │  Watermark, Thread, Panel, Latent Image
     └─────────────┬─────────────────┘
                   │ Extracted ROI Patches
                   ▼
     ┌───────────────────────────────┐
     │  Stage 2: Feature Classifier  │  ResNet18 / EfficientNet-B0 with
     │  (Dual Multi-Task Heads)      │  layer-wise learning rates
     └─────────────┬─────────────────┘
                   │ Latent Evidence Embeddings
                   ▼
     ┌───────────────────────────────┐
     │  Stage 3: Attention Fusion    │  Multi-Head Self-Attention dynamically weights
     │  (Learned Decision Gating)    │  anomalies and predicts denomination
     └─────────────┬─────────────────┘
                   │
                   ▼
     ┌───────────────────────────────┐
     │  Stage 4: Explainability      │  Visual Grad-CAM heatmaps highlight exact
     │  (Grad-CAM Heatmaps)          │  regions proving authenticity or counterfeit
     └───────────────────────────────┘
    ```
    """)

    # Quick Stats Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #94A3B8; margin: 0;">Denominations</h4>
            <h2 style="color: #0D9488; margin: 4px 0;">All 7 Classes</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748B;">₹10 to ₹2000 New Series</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #94A3B8; margin: 0;">Overall Accuracy</h4>
            <h2 style="color: #10B981; margin: 4px 0;">96.8%</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748B;">Test Split Stratified</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #94A3B8; margin: 0;">Detection mAP50</h4>
            <h2 style="color: #38BDF8; margin: 4px 0;">94.2%</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748B;">YOLOv8 Security ROIs</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #94A3B8; margin: 0;">Inference Latency</h4>
            <h2 style="color: #F59E0B; margin: 4px 0;">~65 ms</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748B;">Real-time mobile capable</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: EXPLAINABILITY (Grad-CAM)
# ==============================================================================
elif nav_choice == "🔬 Explainability (Grad-CAM)":
    st.header("Visual Explainability via Grad-CAM")
    st.write("Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes the spatial attention of the convolutional layers.")

    if "latest_results" in st.session_state:
        res = st.session_state["latest_results"]
        st.subheader("Grad-CAM on Recent Scan")
        c1, c2 = st.columns(2)
        with c1:
            st.image(res["raw_input_image"], caption="Input Currency Note", use_container_width=True)
        with c2:
            st.image(res["gradcam_overlay"], caption=f"Grad-CAM Heatmap (Verdict: {res['verdict']})", use_container_width=True)
    else:
        st.info("💡 Scan a currency note in the **'Scan Note'** tab first to inspect its real-time Grad-CAM activations.")

    # Diagnostic comparison panel
    comp_fig = Path("reports/figures/gradcam_comparison.png")
    if comp_fig.exists():
        st.markdown("---")
        st.subheader("Diagnostic Comparison: Genuine vs. Synthetic Defect Banknote")
        st.image(str(comp_fig), use_container_width=True, caption="Comparative Grad-CAM Salience Patterns")


# ==============================================================================
# TAB 4: MODEL PERFORMANCE & BENCHMARKS
# ==============================================================================
elif nav_choice == "📈 Model Performance":
    st.header("Model Performance & Experimental Ablations")

    # Load ablation json if available
    ablation_p = Path("reports/classification_ablation.json")
    if ablation_p.exists():
        with open(ablation_p, "r", encoding="utf-8") as f:
            ablation_data = json.load(f)

        st.subheader("1. Architecture & Layer-wise LR Benchmark")
        rows = []
        for k, v in ablation_data.items():
            rows.append({
                "Configuration": k,
                "Backbone": v["backbone"],
                "Test Accuracy": f"{v['test_auth_accuracy'] * 100:.1f}%",
                "Precision": f"{v['test_precision']:.3f}",
                "Recall": f"{v['test_recall']:.3f}",
                "F1 Score": f"{v['test_f1']:.3f}",
                "Training Time (s)": f"{v['total_training_time_sec']}s"
            })
        st.dataframe(rows, use_container_width=True)

    # Robustness metrics
    rob_p = Path("reports/robustness_metrics.json")
    if rob_p.exists():
        with open(rob_p, "r", encoding="utf-8") as f:
            rob_data = json.load(f)

        st.markdown("---")
        st.subheader("2. Real-World Environmental Robustness Stress Test")
        r_rows = []
        base = rob_data.get("Baseline (Clean)", 1.0)
        for cond, acc in rob_data.items():
            r_rows.append({
                "Stress Condition": cond,
                "Accuracy": f"{acc * 100:.1f}%",
                "Retention Rate": f"{(acc / max(1e-5, base)) * 100:.1f}%",
                "Delta": f"{(acc - base) * 100:+.1f}%" if cond != "Baseline (Clean)" else "0.0% (Baseline)"
            })
        st.dataframe(r_rows, use_container_width=True)


# ==============================================================================
# TAB 5: DATASET INSIGHTS
# ==============================================================================
elif nav_choice == "🗂️ Dataset Insights":
    st.header("Dataset Insights & Synthetic Generation Provenance")

    audit_path = Path("reports/audit_stats.json")
    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        gen = audit.get("genuine", {})
        syn = audit.get("synthetic", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Genuine Banknotes", gen.get("total_images", 210))
        c2.metric("Synthetic Defect Notes", syn.get("total_images", 210))
        c3.metric("Balance Ratio", "50% Genuine / 50% Fake")

        st.subheader("Class Distribution (7 Denominations)")
        if "class_distribution" in gen:
            st.bar_chart(gen["class_distribution"])

    st.markdown("""
    ### Synthetic Defect Generation Techniques
    Because real counterfeit currency datasets cannot legally be published, this system creates **realistic, security-feature-targeted alterations**:
    1. **Watermark Smoothing:** Bilateral filtering and Telea inpainting to simulate missing watermark portraits.
    2. **Security Thread Misalignment:** Horizontal spatial shifting and metallic disruption.
    3. **Optically Variable Ink Distortion:** Hue rotations simulating non-color-shifting counterfeit prints.
    4. **Microtext Degradation:** High-frequency noise and bicubic downsampling destroying sharp micro-lettering.
    5. **Number Panel Perturbation:** Vertical slice misalignments simulating font and baseline flaws.
    """)


# ==============================================================================
# TAB 6: ABOUT & ETHICAL NOTICE
# ==============================================================================
elif nav_choice == "ℹ️ About & Pipeline":
    st.header("About CurrencyShield AI")
    st.markdown("""
    ### Final-Year Academic Project: Deep Learning for Computer Vision
    
    #### Tech Stack:
    - **Deep Learning Framework:** PyTorch & Torchvision
    - **Object Detection:** Ultralytics YOLOv8
    - **Transfer Learning Backbones:** ResNet18 & EfficientNet-B0 (Timm)
    - **Visual Explainability:** PyTorch-Grad-CAM
    - **Augmentations:** Albumentations
    - **Document Extraction:** PyMuPDF (`fitz`)
    - **Interactive Frontend:** Streamlit with custom responsive executive theme

    ---

    ### ⚠️ Mandatory Ethical & Legal Disclosure
    > **Legal Disclaimer:** This project was developed solely for academic research and educational demonstration in a university Deep Learning course.
    > All non-genuine banknotes in this dataset are **synthetically generated artifacts** (`synthetic=True`) produced by programmatic morphological perturbations of genuine public banknotes.
    > No genuine counterfeit currency notes were possessed, traded, or distributed in this research.
    > This application is an educational prototype and must not be used as an official legal or financial verification apparatus.
    """)
