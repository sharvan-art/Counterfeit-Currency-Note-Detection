"""
Grad-CAM Explainability Module
===============================
Native PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Computes activation heatmaps for convolutional layers (e.g. ResNet18 layer4),
highlighting salient security feature patterns (watermark, security thread, etc.).

Outputs visual diagnostic comparison panels for genuine vs synthetic defect notes.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.classification.model import BanknoteFeatureClassifier


class BanknoteGradCAMExplainer:
    """Robust, native PyTorch Grad-CAM generator."""

    def __init__(self, model: BanknoteFeatureClassifier, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Hook target layer
        if self.model.backbone_name == "resnet18":
            # layer4 is self.model.backbone[7]
            self.target_layer = self.model.backbone[7]
        else:
            self.target_layer = self.model.backbone[-1]

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def explain_image(
        self,
        rgb_image: np.ndarray,
        target_category: int = 1  # 1 = Genuine, 0 = Suspicious
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Computes Grad-CAM heatmap for input banknote.
        
        Returns:
            cam_overlay: RGB uint8 image with jet heatmap overlay
            raw_mask: (H, W) float32 activation map
            scores: Model prediction dictionary
        """
        h, w = rgb_image.shape[:2]
        resized = cv2.resize(rgb_image, (224, 224))
        norm_img = np.float32(resized) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_tensor = (norm_img - mean) / std
        input_tensor = torch.from_numpy(norm_tensor.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        out = self.model(input_tensor)
        auth_logits = out["auth_logits"]
        probs = torch.softmax(auth_logits, dim=1)[0].detach().cpu().numpy()
        denom_probs = torch.softmax(out["denom_logits"], dim=1)[0].detach().cpu().numpy()

        # Target class backpropagation
        score = auth_logits[0, target_category]
        score.backward()

        # Gradients GAP (global average pooling)
        grads = self.gradients.detach()  # (1, C, H_f, W_f)
        acts = self.activations.detach()  # (1, C, H_f, W_f)

        pooled_grads = torch.mean(grads, dim=[0, 2, 3])  # (C,)
        for i in range(acts.shape[1]):
            acts[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.sum(acts, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        # Upscale heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Colorize with Jet colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Blend overlay (0.55 image + 0.45 heatmap)
        cam_overlay = cv2.addWeighted(rgb_image, 0.55, colormap_rgb, 0.45, 0)

        return cam_overlay, heatmap_resized, {
            "fake_prob": float(probs[0]),
            "genuine_prob": float(probs[1]),
            "pred_label": int(np.argmax(probs)),
            "pred_denom": int(np.argmax(denom_probs))
        }

    def generate_comparison_panel(
        self,
        genuine_img_path: str,
        fake_img_path: str,
        output_path: str = "reports/figures/gradcam_comparison.png"
    ) -> str:
        """Creates a side-by-side diagnostic visualization comparing Genuine vs. Synthetic Defect Note."""
        gen_bgr = cv2.imread(genuine_img_path)
        fake_bgr = cv2.imread(fake_img_path)

        if gen_bgr is None or fake_bgr is None:
            raise FileNotFoundError("Could not read genuine or fake comparison image.")

        gen_rgb = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB)
        fake_rgb = cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)

        gen_cam, _, gen_scores = self.explain_image(gen_rgb, target_category=1)
        fake_cam, _, fake_scores = self.explain_image(fake_rgb, target_category=0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Genuine Note
        axes[0, 0].imshow(gen_rgb)
        axes[0, 0].set_title("Genuine Note Input\nGround Truth: Genuine (1.0)", fontsize=12, fontweight="bold", color="darkgreen")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(gen_cam)
        axes[0, 1].set_title(f"Grad-CAM Salience (Genuine Verification)\nAuthenticity Confidence: {gen_scores['genuine_prob']*100:.1f}%",
                             fontsize=12, fontweight="bold", color="darkgreen")
        axes[0, 1].axis("off")

        # Fake Note
        axes[1, 0].imshow(fake_rgb)
        axes[1, 0].set_title("Synthetic Defect Note Input (synthetic=True)\nGround Truth: Defect/Suspicious", fontsize=12, fontweight="bold", color="darkred")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(fake_cam)
        axes[1, 1].set_title(f"Grad-CAM Salience (Defect Evidence Localization)\nSuspicious Score: {fake_scores['fake_prob']*100:.1f}%",
                             fontsize=12, fontweight="bold", color="darkred")
        axes[1, 1].axis("off")

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Grad-CAM comparison panel saved to: {out_p}")
        return str(out_p)


if __name__ == "__main__":
    net = BanknoteFeatureClassifier(backbone_name="resnet18", pretrained=False)
    explainer = BanknoteGradCAMExplainer(net)
    dummy = np.zeros((300, 600, 3), dtype=np.uint8)
    cam, _, sc = explainer.explain_image(dummy)
    print("Native GradCAM Explainer verified! Cam shape:", cam.shape)
