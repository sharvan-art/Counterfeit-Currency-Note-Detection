# Classification Architecture & Training Configuration Ablation

Comparative benchmark of transfer learning backbones, layer-wise learning rates, and fine-tuning strategies:

| Configuration | Backbone | Fine-Tuning Strategy | Test Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **config_1_resnet18_frozen** | resnet18 | Frozen Backbone (Linear Probe) | 58.7% | 0.561 | 0.742 | 0.639 | 61.12s |
| **config_2_resnet18_layerwise** | resnet18 | Layer-wise LR Fine-Tuning | 74.6% | 0.727 | 0.774 | 0.750 | 165.41s |
| **config_3_effnet_unfrozen** | efficientnet_b0 | Layer-wise LR Fine-Tuning | 68.2% | 0.641 | 0.806 | 0.714 | 157.79s |