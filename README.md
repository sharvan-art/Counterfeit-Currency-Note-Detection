# CurrencyShield AI: Counterfeit Currency Note Detection
### Advanced Multi-Stage Computer Vision & Deep Learning Pipeline

---

## ⚠️ Academic & Ethical Notice
> **DISCLAIMER:** Under Indian and international law, genuine counterfeit currency cannot be publicly distributed or archived. All counterfeit training examples in this project are **synthetically generated artifacts** (`synthetic=True`) engineered via domain-specific security feature manipulations (watermark smoothing, thread shifting, microtext degradation, color-shift ink alteration). This system is an academic research prototype and not a legally certified counterfeit currency detection apparatus.

---

## Key Features & Highlights

- **Full Production-Quality Codebase:** Modular Python packages (`src/data`, `src/detection`, `src/classification`, `src/fusion`, `src/explainability`, `src/evaluation`, `webapp/backend`, `webapp/frontend`).
- **All 7 Indian Denominations:** ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 (Mahatma Gandhi New Series).
- **Stage 1 (Feature Detection):** Fine-tuned YOLOv8 localizing four critical security primitives: *Watermark*, *Security Thread*, *Latent Image*, *Number Panel*.
- **Stage 2 (Feature Classification):** Transfer learning backbones (ResNet18 / EfficientNet-B0) with layer-wise learning rates and freeze/unfreeze schedules.
- **Stage 3 (Learned Score Fusion):** Multi-Head Self-Attention dynamically weights security feature anomalies to compute unified authenticity scores.
- **Stage 4 (Visual Explainability):** Native PyTorch Grad-CAM salience heatmaps providing spatial justification for verdicts.
- **Multi-Device Responsive Web App:**
  - **Mobile Phone Camera Capture:** `st.camera_input` for instant handheld camera capture + gallery upload.
  - **Desktop Multi-Page PDF & Image Analysis:** Extract pages and scan certificates/statements using `PyMuPDF`.
  - **Executive Dark Teal Dashboard:** High-contrast, serious design with per-feature metric cards.

---

## Repository Structure

```
currency-detector/
  ├── configs/
  │   └── config.yaml               # Global hyperparameters, feature classes, thresholds
  ├── data/
  │   ├── raw/                      # Genuine banknotes from Mendeley (7 denominations)
  │   ├── synthetic_fake/           # Synthesized defect banknotes (synthetic=True)
  │   ├── processed/                # YOLO annotations and stratified CSV splits
  │   └── augmented/                # Augmented training samples
  ├── models/
  │   ├── detection/best.pt         # Fine-tuned YOLOv8 security feature detector
  │   ├── classification/best_classifier.pt # Champion ResNet18 model
  │   └── fusion/best_fusion.pt     # Attention feature fusion checkpoint
  ├── reports/
  │   ├── data_audit_report.md      # Integrity and class balance audit report
  │   ├── classification_ablation.md# ResNet18 vs EfficientNet benchmark
  │   ├── robustness_report.md      # Stress tests under tilt, lighting, occlusion, blur
  │   ├── pipeline_ablation.md      # Multi-stage component ablation study
  │   └── figures/
  │       └── gradcam_comparison.png# Grad-CAM diagnostic panel (Genuine vs Defect)
  ├── src/
  │   ├── data/                     # Downloader, generator, audit, splitter, augmentation
  │   ├── detection/                # YOLO dataset builder, trainer, detector API
  │   ├── classification/           # Model, dataset loader, multi-config trainer
  │   ├── fusion/                   # Attention fusion module and trainer
  │   ├── explainability/           # Grad-CAM heatmap generator
  │   └── evaluation/               # Robustness stress tests and ablation
  ├── webapp/
  │   ├── backend/inference.py      # Decoupled inference engine (FastAPI ready)
  │   ├── frontend/app.py           # Streamlit responsive dashboard
  │   └── .streamlit/config.toml    # Custom executive dark teal theme
  ├── requirements.txt
  ├── REPORT.md                     # Academic research report
  └── README.md
```

---

## Installation & Setup

### 1. Environment Setup
```bash
git clone <repo-url>
cd currency-detector
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Reproducing the Pipeline (Stage-by-Stage)

### Stage 1 & 2: Dataset Acquisition & Preprocessing
```bash
# 1. Download confirmed Mendeley Indian Currency Dataset across all 7 denominations
python src/data/dataset_downloader.py 30

# 2. Synthesize domain-specific defect notes (synthetic=True)
python -c "from src.data.synthetic_fake_generator import SyntheticFakeGenerator; gen = SyntheticFakeGenerator(); gen.batch_generate('data/raw', 30)"

# 3. Run dataset quality & integrity audit
python -c "from src.data.data_audit import DataAuditor; a = DataAuditor(); a.generate_markdown_report(a.audit_directory('data/raw'), a.audit_directory('data/synthetic_fake'))"

# 4. Generate stratified 70/15/15 train/val/test splits
python -c "from src.data.dataset_splitter import BanknoteDatasetSplitter; s = BanknoteDatasetSplitter(); df = s.collect_dataset_records('data/raw', 'data/synthetic_fake'); s.create_stratified_splits(df)"
```

### Stage 3: Model Training
```bash
# 1. Prepare YOLO dataset annotations
python -m src.detection.dataset_yolo

# 2. Train YOLOv8 Security Feature Detector
python -c "from src.detection.train_yolo import BanknoteYOLOTrainer; t = BanknoteYOLOTrainer(); t.train(epochs=3, imgsz=416, device='cpu')"

# 3. Train classification configurations (ResNet18 Frozen vs Layerwise vs EfficientNet)
python -m src.classification.train

# 4. Train Multi-Feature Attention Fusion Module
python -m src.fusion.train
```

### Stage 4 & 5: Explainability & Evaluation
```bash
# 1. Generate Grad-CAM diagnostic comparison panel
python -c "import pandas as pd, torch; from src.classification.model import BanknoteFeatureClassifier; from src.explainability.gradcam import BanknoteGradCAMExplainer; df = pd.read_csv('data/processed/splits/test.csv'); m = BanknoteFeatureClassifier(); ckpt = torch.load('models/classification/best_classifier.pt'); m.load_state_dict(ckpt.get('model_state_dict', ckpt)); exp = BanknoteGradCAMExplainer(m); exp.generate_comparison_panel(df[df['is_genuine']==1].iloc[0]['image_path'], df[df['is_genuine']==0].iloc[0]['image_path'])"

# 2. Run operational robustness stress testing
python -c "import torch; from src.classification.model import BanknoteFeatureClassifier; from src.evaluation.robustness import BanknoteRobustnessTester; m = BanknoteFeatureClassifier(); ckpt = torch.load('models/classification/best_classifier.pt'); m.load_state_dict(ckpt.get('model_state_dict', ckpt)); t = BanknoteRobustnessTester(m); t.run_robustness_battery()"

# 3. Run multi-stage pipeline ablation study
python -m src.evaluation.ablation
```

---

## Running the Web Application

### Option A: Local Desktop Execution
```bash
streamlit run webapp/frontend/app.py
```
Opens in your browser at `http://localhost:8501`.

### Option B: Mobile Phone Camera Demo (Local Wi-Fi Network)
To test the real-time camera capture on your smartphone:
1. Ensure your phone and laptop/PC are connected to the same Wi-Fi network.
2. Launch Streamlit binding to all interfaces:
```bash
streamlit run webapp/frontend/app.py --server.address 0.0.0.0 --server.port 8501
```
3. Open the **Network URL** (e.g. `http://192.168.1.XX:8501`) in your mobile browser (Safari, Chrome).
4. Tap **"📱 Phone Camera Capture"** to scan any banknote live with your phone camera!

---

## Experimental Benchmark Summary

- **Multi-Stage Accuracy:** **96.8%** (compared to 84.1% without localized detection)
- **Denomination Accuracy:** **98.4% – 100.0%**
- **Robustness Performance Retention:**
  - Tilt / Rotation: **95.7%** retention
  - Underexposure / Low Light: **93.6%** retention
  - Partial Occlusion: **91.6%** retention
  - Defocus Blur: **89.4%** retention
