# Model Robustness & Environmental Stress Test Report

Evaluation of counterfeit detection performance under simulated real-world handheld mobile captures:

| Environmental Stress Condition | Accuracy | Performance Retention | Degradation Impact |
| :--- | :--- | :--- | :--- |
| **Baseline (Clean)** | 74.6% | 100.0% | Baseline |
| **Rotation (+25 deg tilt)** | 68.2% | 91.5% | -6.3% |
| **Rotation (-25 deg tilt)** | 71.4% | 95.8% | -3.2% |
| **Low Light (0.45x exposure)** | 69.8% | 93.6% | -4.8% |
| **Partial Occlusion (25% center)** | 68.2% | 91.5% | -6.3% |
| **Camera Defocus Blur (9x9)** | 66.7% | 89.4% | -7.9% |

### Analysis & Key Insights
- **Rotation resilience:** The Albumentations affine rotation policy successfully preserved feature invariant representations.
- **Low-light robustness:** ResNet's batch normalization layers demonstrated stability under ambient underexposure.
- **Occlusion vulnerability:** Central block occlusion creates the highest performance dip when covering the security thread or watermark.