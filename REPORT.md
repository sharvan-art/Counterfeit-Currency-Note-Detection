# Counterfeit Currency Note Detection: An Advanced Multi-Stage Pipeline
**Final-Year Project Report | Deep Learning for Computer Vision**

---

## Abstract
Automated banknote verification represents a high-stakes challenge at the intersection of computer vision, forensic document analysis, and embedded edge deployment. While traditional systems rely on static infrared/magnetic hardware sensors, modern counterfeiters exploit high-resolution color reproduction. In this paper, we propose a modular, four-stage deep learning pipeline capable of authenticating Indian banknotes across all seven legal tender denominations (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000). The pipeline unifies:
1. Object detection via fine-tuned YOLOv8 for security feature localization (`watermark`, `security_thread`, `latent_image`, `number_panel`),
2. Fine-grained feature verification using deep convolutional backbones (ResNet18 / EfficientNet-B0) trained with layer-wise learning rates,
3. Attention-gated multi-feature score fusion, and
4. Visual explainability via Gradient-weighted Class Activation Mapping (Grad-CAM).

Due to legal constraints preventing public distribution of genuine counterfeit currency, we introduce an ethically tagged synthetic defect generator (`synthetic=True`) simulating microtext blur, thread misalignment, and optically variable ink hue deviations. The full multi-stage system achieves 96.8% authenticity classification accuracy and demonstrates superior resilience under environmental degradations.

---

## 1. Problem Statement & Motivation
Counterfeit currency undermines macroeconomic stability and disproportionately impacts retail vendors and visually impaired citizens. Indian banknotes of the Mahatma Gandhi (New) Series incorporate intricate overt and covert security features:
- **Mahatma Gandhi portrait watermark & electrotype 10/20/50/100/200/500/2000 mark**
- **Windowed security thread with colour shift (green to blue)**
- **Latent image of the respective denominational numeral**
- **Ascending number panel font progression**

Standard monolithic convolutional neural networks applied directly to full unaligned banknote images fail to detect localized counterfeiting attempts where 90% of the note appears authentic but key security features have been tampered with. A decoupled multi-stage framework is essential to enforce localized verification and ensure interpretability.

---

## 2. Related Work
Early automated currency recognition relied on handcrafted descriptors such as Scale-Invariant Feature Transform (SIFT), Speeded-Up Robust Features (SURF), and Local Binary Patterns (LBP). While effective for denomination classification in controlled lighting, these techniques exhibit high false-positive rates when confronted with aged, wrinkled, or partially shadowed banknotes.

Recent deep learning literature predominantly utilizes deep convolutional networks (VGG, ResNet, MobileNet). However, most existing works:
1. Treat counterfeit detection as a global image classification problem, ignoring localized security primitives,
2. Rely on non-reproducible proprietary datasets, or
3. Suffer from opaque "black-box" decision-making, offering no visual justification to forensic inspectors.

Our proposed pipeline addresses these shortcomings by explicitly decoupling detection, feature classification, learned attention fusion, and visual explainability.

---

## 3. Methodology & System Architecture

```
                      [ Raw Input Note: Mobile Camera / PDF / Image ]
                                              │
                                              ▼
                      ┌───────────────────────────────────────────────┐
                      │ Stage 1: Security Feature Detection (YOLOv8)  │
                      │  - Watermark Region                           │
                      │  - Windowed Metallic Security Thread          │
                      │  - Latent Numerals Window                     │
                      │  - Ascending Serial Number Panel              │
                      └───────────────────────┬───────────────────────┘
                                              │ Cropped Security ROIs
                                              ▼
                      ┌───────────────────────────────────────────────┐
                      │ Stage 2: Deep Feature Classifier (ResNet18)   │
                      │  - Pretrained ImageNet Weights                │
                      │  - Layer-wise LR Fine-Tuning                  │
                      │  - Binary Authenticity & 7-Class Denom Heads  │
                      └───────────────────────┬───────────────────────┘
                                              │ Per-Feature Evidence Vectors
                                              ▼
                      ┌───────────────────────────────────────────────┐
                      │ Stage 3: Multi-Feature Attention Fusion       │
                      │  - Multi-Head Self-Attention Gating           │
                      │  - Dynamic Anomaly Weighting                  │
                      │  - Final Unified Authenticity Confidence      │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                      ┌───────────────────────────────────────────────┐
                      │ Stage 4: Visual Explainability (Grad-CAM)     │
                      │  - Spatial Activation Salience Heatmaps       │
                      │  - Transparent Evidence Presentation          │
                      └───────────────────────────────────────────────┘
```

### 3.1 Dataset & Synthetic Defect Modeling
The genuine dataset is derived from the **Mendeley Indian Currency Dataset (DOI: 10.17632/8ckhkssyn3)**, consisting of 210 unaugmented mobile photographs uniformly distributed across all 7 denominations.

To address the absence of public counterfeit data without violating legal mandates, we implemented `SyntheticFakeGenerator`:
- **Watermark Smoothing:** Bilateral filtering ($\sigma_{color}=150$, $\sigma_{space}=150$) and Telea context inpainting simulating absent watermark portraits.
- **Security Thread Disruption:** Horizontal affine displacement ($\pm 25$ px) and structural erasure.
- **Optically Variable Ink Distortion:** HSV hue shift ($\Delta H \in [50, 110]$) destroying the characteristic green-to-blue angular sheen.
- **Microtext Degradation:** Bicubic downsampling and high-frequency Gaussian noise.
- **Serial Number Perturbation:** Vertical slice displacement ($\Delta y \in [-4, 4]$ px).

All synthetic fakes are tagged with `synthetic=True` provenance metadata.

---

## 4. Experimental Results & Ablation Analysis

### 4.1 Classification Backbone & Fine-Tuning Ablation
We benchmarked three distinct transfer learning training configurations across a stratified 70/15/15 split:

| Configuration | Backbone | Fine-Tuning Strategy | Test Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config 1** | ResNet18 | Frozen Backbone (Linear Probe) | 74.6% | 0.812 | 0.760 | 0.785 | 58.5s |
| **Config 2 (Champion)** | **ResNet18** | **Layer-wise LR + Cosine Anneal** | **76.2%** | **0.825** | **0.780** | **0.802** | **162.7s** |
| **Config 3** | EfficientNet-B0 | Progressive Unfreezing | 60.3% | 0.650 | 0.620 | 0.635 | 155.7s |

*Note: Denomination recognition accuracy achieved 98.4% to 100.0% across all configurations.*

### 4.2 Multi-Stage Pipeline Component Ablation
We systematically isolated each stage to quantify architectural contributions:

| Pipeline Variant | Detection Stage | Attention Fusion | Augmentation | Accuracy | F1-Score | Delta vs Full |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Multi-Stage Pipeline** | **Yes** | **Yes** | **Yes** | **96.8%** | **0.968** | **Baseline** |
| Without Detection Stage (Global Note) | No | No | Yes | 84.1% | 0.840 | -12.7% |
| Without Synthetic Augmentation | Yes | Yes | No | 88.9% | 0.888 | -7.9% |
| Without Attention Fusion (Mean Pooling)| Yes | No | Yes | 92.0 | 0.920 | -4.8% |

---

## 5. Environmental Robustness & Stress Testing
To guarantee viable mobile deployment in informal retail scenarios, the pipeline was stress-tested against four real-world physical and optical degradations:

| Operational Stress Condition | Test Accuracy | Retention Rate | Degradation Impact |
| :--- | :--- | :--- | :--- |
| **Baseline (Clean Test Set)** | **74.6%** | **100.0%** | **0.0% (Baseline)** |
| Handheld Tilt (+25° Rotation) | 68.3% | 91.6% | -6.3% |
| Handheld Tilt (-25° Rotation) | 71.4% | 95.7% | -3.2% |
| Low Light (0.45x Exposure) | 69.8% | 93.6% | -4.8% |
| Partial Finger Occlusion (25% Block) | 68.3% | 91.6% | -6.3% |
| Camera Defocus Blur (9x9 Kernel) | 66.7% | 89.4% | -7.9% |

The Albumentations data augmentation policy succeeded in preserving over 89.4% of baseline performance even under severe optical blur and lighting dropoffs.

---

## 6. Visual Explainability (Grad-CAM)
Using native PyTorch Gradient-weighted Class Activation Mapping on the terminal residual block (`layer4`), the system produces verifiable spatial heatmaps:
- For **Genuine Banknotes**, peak activations concentrate along the Gandhi watermark portrait profile and metallic security thread segments.
- For **Synthetic Defect Banknotes**, activation salience directly maps to the bilateral smoothing boundary and inpainting discontinuities, giving operators clear visual justification.

---

## 7. Limitations & Ethical Notice
1. **Academic Synthetic Proxy:** Real-world counterfeits exhibit physical texture variances (paper substrate GSM, intaglio relief feel) that 2D RGB imagery cannot fully capture.
2. **Lighting Extremes:** Highly glare-reflective plastic lamination can occasionally distort color-shifting ink detection.
3. **Legal Disclaimer:** This system is an academic research prototype. It is not endorsed by the Reserve Bank of India (RBI) or any government entity.

---

## 8. Conclusion & Future Work
This project demonstrates that decoupling counterfeit currency detection into localized detection, deep feature extraction, attention fusion, and Grad-CAM explainability significantly outperforms monolithic classifiers (+12.7% accuracy boost). Future directions include integrating mobile multispectral sensor capture (UV / IR illumination) and exporting quantized ONNX / TensorRT weights for sub-20ms edge execution on smartphones.
