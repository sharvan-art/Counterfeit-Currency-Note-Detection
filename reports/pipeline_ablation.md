# Multi-Stage Pipeline Component Ablation Study

Systematic ablation assessing the impact of security feature localization, attention fusion, and synthetic augmentation:

| Pipeline Configuration | Detection Stage | Attention Fusion | Augmentation | Accuracy | F1-Score | Performance Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Multi-Stage Pipeline (YOLO + ResNet + Fusion + Aug)** | Yes | Yes | Yes | 96.8% | 0.968 | Baseline (Full) |
| **Ablation 1: Without Detection Stage (Global Note CNN)** | No | No | Yes | 84.1% | 0.840 | -12.7% |
| **Ablation 2: Without Synthetic Augmentation** | Yes | Yes | No | 88.9% | 0.888 | -7.9% |
| **Ablation 3: Without Attention Fusion (Simple Mean Pooling)** | Yes | No | Yes | 92.0% | 0.920 | -4.8% |

### Key Takeaways
1. **Detection Stage Impact (+12.7% accuracy):** Localizing specific security regions (watermark, thread, number panel) is essential. Without it, global background texture overwhelms micro-defects.
2. **Synthetic Handheld Augmentation (+7.9% accuracy):** Robustness to lighting shifts and partial finger occlusion prevents overfitting to clean static datasets.
3. **Attention-Gated Fusion (+4.8% accuracy):** Giving higher dynamic weight to defective or suspicious patches prevents false-positive masking.