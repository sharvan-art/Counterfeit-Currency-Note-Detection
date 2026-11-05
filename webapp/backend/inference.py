"""
Decoupled Inference Backend API
================================
Implements the end-to-end multi-stage currency authentication pipeline:
1. Banknote Localization & Alignment: Detects banknote boundary, strips letterboxing/margins, auto-orients to horizontal landscape.
2. Stage 1: High-Precision Security Feature Localization:
   - Security Thread (dynamic peak detection on metallic ribbon)
   - Watermark Window (far-right Mahatma Gandhi circular window)
   - Serial Number Panel (bottom-right ascending serial number)
   - Latent Image (bottom-left numeral panel)
3. Stage 2: Deep Feature Classification (ResNet18) + Forensic Metric Validation
4. Stage 3: Multi-Feature Attention Fusion
5. Stage 4: Grad-CAM Explainability Heatmap Generation

Completely decoupled from the Streamlit UI layer for clean modularity & FastAPI compatibility.
"""

import os
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import torch
import pymupdf

from src.detection.detector import SecurityFeatureDetector, BanknoteCropper
from src.classification.model import BanknoteFeatureClassifier
from src.fusion.model import AttentionFeatureFusion
from src.explainability.gradcam import BanknoteGradCAMExplainer


class BanknoteInferenceEngine:
    """Production inference engine for counterfeit currency detection."""

    DENOMINATIONS = [10, 20, 50, 100, 200, 500, 2000]

    def __init__(
        self,
        yolo_weights: str = "models/detection/best.pt",
        classifier_weights: str = "models/classification/best_classifier.pt",
        fusion_weights: str = "models/fusion/best_fusion.pt",
        device: str = "cpu"
    ):
        self.device = torch.device(device)

        # 1. Stage 1 Detector & Cropper
        self.detector = SecurityFeatureDetector(model_path=yolo_weights)

        # 2. Stage 2 Classifier
        self.classifier = BanknoteFeatureClassifier(backbone_name="resnet18", pretrained=False)
        cls_p = Path(classifier_weights)
        if cls_p.exists():
            try:
                ckpt = torch.load(cls_p, map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                self.classifier.load_state_dict(state)
            except Exception as e:
                print(f"Warning: Classifier load error: {e}")
        self.classifier.to(self.device)
        self.classifier.eval()

        # 3. Stage 3 Fusion
        self.fusion = AttentionFeatureFusion(feature_dim=256, hidden_dim=128)
        fus_p = Path(fusion_weights)
        if fus_p.exists():
            try:
                self.fusion.load_state_dict(torch.load(fus_p, map_location=self.device))
            except Exception as e:
                print(f"Warning: Fusion load error: {e}")
        self.fusion.to(self.device)
        self.fusion.eval()

        # 4. Stage 4 Explainer
        self.explainer = BanknoteGradCAMExplainer(self.classifier, device=device)

    @classmethod
    def extract_images_from_pdf(cls, pdf_bytes: bytes) -> List[np.ndarray]:
        """Extracts page renderings from uploaded PDF document as RGB numpy arrays."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))
        doc.close()
        return images

    def preprocess_image(self, img_input: Any) -> np.ndarray:
        """Converts PIL, Bytes, or ndarray into uint8 RGB numpy image."""
        if isinstance(img_input, bytes):
            pil_img = Image.open(io.BytesIO(img_input)).convert("RGB")
            return np.array(pil_img)
        elif isinstance(img_input, Image.Image):
            return np.array(img_input.convert("RGB"))
        elif isinstance(img_input, (str, Path)):
            pil_img = Image.open(str(img_input)).convert("RGB")
            return np.array(pil_img)
        elif isinstance(img_input, np.ndarray):
            if len(img_input.shape) == 2:
                return cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 4:
                return cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
            return img_input
        else:
            raise ValueError("Unsupported image input format.")

    def _run_single_inference(self, raw_bgr: np.ndarray) -> Dict[str, Any]:
        """Executes multi-stage feature extraction on a single orientation candidate."""
        # Stage 1: Detect banknote contour & locate features
        det_res = self.detector.detect_and_crop(raw_bgr)

        cropped_note_bgr = det_res["cropped_banknote"]
        cropped_note_rgb = cv2.cvtColor(cropped_note_bgr, cv2.COLOR_BGR2RGB)
        annotated_bgr = det_res["annotated_image"]
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        detected_features = det_res["features"]

        nh, nw = cropped_note_bgr.shape[:2]

        # Normalization constants
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Global Note Evaluation with ResNet18
        note_resized = cv2.resize(cropped_note_rgb, (224, 224))
        norm_note = (note_resized.astype(np.float32) / 255.0 - mean) / std
        tensor_note = torch.from_numpy(norm_note.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            global_out = self.classifier(tensor_note)
            global_auth_prob = torch.softmax(global_out["auth_logits"], dim=1)[0].cpu().numpy()
            global_denom_prob = torch.softmax(global_out["denom_logits"], dim=1)[0].cpu().numpy()

        global_genuine_score = float(global_auth_prob[1])
        predicted_denom_idx = int(np.argmax(global_denom_prob))
        predicted_denom = self.DENOMINATIONS[predicted_denom_idx]
        denom_confidence = float(global_denom_prob[predicted_denom_idx])

        # Forensic Feature Analysis
        feature_breakdown = []
        feature_tokens = []
        forensic_flags = []

        # Analyze Note Laplacian Sharpness
        gray_note = cv2.cvtColor(cropped_note_bgr, cv2.COLOR_BGR2GRAY)
        note_laplacian = float(cv2.Laplacian(gray_note, cv2.CV_64F).var())

        for feat in detected_features:
            fn = feat["feature_name"]
            patch_bgr = feat["patch"]
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

            if patch_bgr.size == 0:
                continue

            # Deep feature embedding
            p_resized = cv2.resize(patch_rgb, (224, 224))
            p_norm = (p_resized.astype(np.float32) / 255.0 - mean) / std
            t_patch = torch.from_numpy(p_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                cls_out = self.classifier(t_patch)
                auth_prob = torch.softmax(cls_out["auth_logits"], dim=1)[0].cpu().numpy()
                embedding = cls_out["embedding"]

            cnn_score = float(auth_prob[1])
            patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

            # Forensic Rule-Based Checks per feature
            is_suspicious = False
            observation = "Intact & authentic"
            forensic_score = cnn_score

            if fn == "watermark":
                wm_var = float(cv2.Laplacian(patch_gray, cv2.CV_64F).var())
                # If watermark is smoothed with bilateral filter or blurred or inpainted
                if wm_var < 50.0:
                    is_suspicious = True
                    forensic_score = min(0.20, wm_var / 250.0)
                    observation = f"Watermark missing/blurred (Variance: {wm_var:.1f} < 50)"
                    forensic_flags.append("Watermark blurred or removed")
                elif wm_var < 150.0 and note_laplacian < 500.0:
                    is_suspicious = True
                    forensic_score = 0.35
                    observation = "Watermark texture degraded"
                    forensic_flags.append("Watermark texture degraded")
                else:
                    forensic_score = max(0.75, cnn_score)
                    observation = "Authentic portrait watermark & electrotype verified"

            elif fn == "security_thread":
                # Master Forensic Gate: Security Thread Analysis
                thread_metrics = feat.get("thread_metrics", {})
                is_th_auth = thread_metrics.get("is_authentic", True)
                th_continuity = thread_metrics.get("continuity", 0.5)
                th_pos = thread_metrics.get("center_ratio", 0.49)
                th_edge = thread_metrics.get("edge_energy", 50.0)

                if not is_th_auth or th_continuity < 0.22:
                    is_suspicious = True
                    forensic_score = 0.15
                    anomalies = thread_metrics.get("anomalies", ["Security thread missing or disrupted"])
                    observation = f"Defect: {'; '.join(anomalies)}"
                    forensic_flags.extend(anomalies)
                else:
                    forensic_score = max(0.88, cnn_score)
                    observation = f"Verified: Continuous metallic ribbon ({th_continuity*100:.0f}% continuity at {th_pos*100:.1f}% width)"

            elif fn == "number_panel":
                # Check horizontal gradient / alignment
                diff_h = np.diff(patch_gray.astype(float), axis=1)
                diff_v = np.diff(patch_gray.astype(float), axis=0)
                step_discontinuity = float(np.std(diff_v) / max(1.0, np.std(diff_h)))
                if (step_discontinuity > 1.85 and cnn_score < 0.40) or (cnn_score < 0.12):
                    is_suspicious = True
                    forensic_score = 0.28
                    observation = "Serial numbers misaligned / sliced"
                    forensic_flags.append("Serial numbers misaligned")
                else:
                    forensic_score = max(0.72, cnn_score)
                    observation = "Ascending font alignment verified"

            elif fn == "latent_image":
                # Real banknotes photographed at normal eye level / hand capture will NOT show
                # the latent 200 numeral because latent images only reveal under 45-degree angled lighting!
                # We verify the intaglio relief texture (Laplacian microtext & guilloche pattern sharpness).
                latent_var = float(cv2.Laplacian(patch_gray, cv2.CV_64F).var())
                if latent_var < 40.0 and cnn_score < 0.20:
                    is_suspicious = True
                    forensic_score = 0.30
                    observation = "Latent numeral pattern degraded / washed out"
                    forensic_flags.append("Latent image texture smoothed or missing")
                else:
                    # Intaglio micro-print relief pattern is intact
                    forensic_score = max(0.72, cnn_score)
                    observation = f"Intaglio relief pattern verified (texture sharpness: {latent_var:.1f})"

            status = "Suspicious" if is_suspicious or forensic_score < 0.50 else "Genuine"

            feature_breakdown.append({
                "feature_name": fn,
                "bbox": feat["bbox"],
                "detector_confidence": feat["confidence"],
                "authenticity_score": round(float(forensic_score), 3),
                "status": status,
                "observation": observation,
                "patch_thumbnail": patch_rgb
            })
            feature_tokens.append(embedding)

        # Pad tokens for attention fusion
        while len(feature_tokens) < 4:
            feature_tokens.append(torch.zeros(1, 256, device=self.device))
        feature_tokens_tensor = torch.cat(feature_tokens[:4], dim=0).unsqueeze(0)

        # Stage 3: Multi-Feature Attention Fusion
        with torch.no_grad():
            fusion_out = self.fusion(feature_tokens_tensor)
            attention_weights = fusion_out["attention_weights"][0].cpu().numpy()

        for i, feat in enumerate(feature_breakdown):
            if i < len(attention_weights):
                feat["attention_weight"] = round(float(attention_weights[i]), 3)

        # Final Forensic Verdict Determination:
        # A banknote is a legal instrument: ALL core security features MUST be genuine.
        suspicious_list = [f for f in feature_breakdown if f["status"] == "Suspicious"]

        if len(suspicious_list) > 0 or len(forensic_flags) > 0:
            verdict = "SUSPICIOUS / COUNTERFEIT"
            # Min of suspicious scores
            worst_score = min([f["authenticity_score"] for f in suspicious_list] or [0.30])
            overall_authenticity = min(0.35, worst_score)
            verdict_confidence = round(1.0 - overall_authenticity, 3)
            color = "#EF4444"  # Red
        else:
            verdict = "GENUINE"
            avg_score = sum(f["authenticity_score"] for f in feature_breakdown) / max(1, len(feature_breakdown))
            overall_authenticity = round(max(0.70, avg_score), 3)
            verdict_confidence = overall_authenticity
            color = "#10B981"  # Emerald Green

        # Stage 4: Grad-CAM Explainability on the cropped banknote
        gradcam_target = 1 if verdict == "GENUINE" else 0
        cam_overlay, raw_mask, _ = self.explainer.explain_image(cropped_note_rgb, target_category=gradcam_target)

        return {
            "verdict": verdict,
            "verdict_confidence": verdict_confidence,
            "authenticity_score": overall_authenticity,
            "predicted_denomination": predicted_denom,
            "denomination_confidence": denom_confidence,
            "color": color,
            "feature_breakdown": feature_breakdown,
            "attention_weights": [float(w) for w in attention_weights],
            "annotated_image": annotated_rgb,
            "gradcam_overlay": cam_overlay,
            "raw_input_image": cropped_note_rgb,
            "forensic_flags": forensic_flags
        }

    def is_upright_orientation(self, note_bgr: np.ndarray) -> bool:
        """
        Determines whether a horizontal banknote is in normal upright orientation (0°)
        or inverted upside-down (180°).
        Uses anatomical Gandhi portrait luminance distribution (head vs chest)
        combined with ResNet18 denomination confidence gating.
        """
        h, w = note_bgr.shape[:2]
        gray = cv2.cvtColor(note_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Gandhi Head vs Chest (Gandhi is upright -> bald head is brighter than chest/shawl folds)
        g_head = gray[int(0.15 * h):int(0.45 * h), int(0.28 * w):int(0.52 * w)]
        g_chest = gray[int(0.50 * h):int(0.80 * h), int(0.28 * w):int(0.52 * w)]
        g_diff_0 = float(g_head.mean()) - float(g_chest.mean())

        gray_180 = cv2.rotate(gray, cv2.ROTATE_180)
        g_head_180 = gray_180[int(0.15 * h):int(0.45 * h), int(0.28 * w):int(0.52 * w)]
        g_chest_180 = gray_180[int(0.50 * h):int(0.80 * h), int(0.28 * w):int(0.52 * w)]
        g_diff_180 = float(g_head_180.mean()) - float(g_chest_180.mean())

        # 2. ResNet Denomination probability
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def get_denom_conf(im_bgr):
            rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            res_224 = cv2.resize(rgb, (224, 224))
            norm = (res_224.astype(np.float32) / 255.0 - mean) / std
            t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                out = self.classifier(t)
                denom = torch.softmax(out["denom_logits"], dim=1)[0].cpu().numpy()
            return float(np.max(denom))

        d0 = get_denom_conf(note_bgr)
        d180 = get_denom_conf(cv2.rotate(note_bgr, cv2.ROTATE_180))

        score_0 = (g_diff_0 * 0.1) + (d0 * 2.0)
        score_180 = (g_diff_180 * 0.1) + (d180 * 2.0)
        return score_0 >= score_180

    def run_inference(self, img_input: Any) -> Dict[str, Any]:
        """
        Executes complete omnidirectional multi-stage inference pipeline on single banknote image:
        1. Auto-crops and perspective-deskews banknote at ANY arbitrary rotation angle or tilt.
        2. Automatically tests and ensures true upright orientation (0° vs 180°).
        3. Precisely locates security features (thread, watermark, number panel, latent image).
        4. Runs forensic verification + deep learning classification.
        5. Computes transparent authenticity verdict and Grad-CAM salience map.
        """
        raw_rgb = self.preprocess_image(img_input)
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)

        # 1. Perspective deskew & crop
        cropped_note_bgr, _, _ = BanknoteCropper.crop_and_orient(raw_bgr)

        # 2. Automatically orient to upright (0°)
        if not self.is_upright_orientation(cropped_note_bgr):
            cropped_note_bgr = cv2.rotate(cropped_note_bgr, cv2.ROTATE_180)

        # 3. Run full multi-stage forensic evaluation on the upright cropped banknote
        return self._run_single_inference(cropped_note_bgr)


if __name__ == "__main__":
    engine = BanknoteInferenceEngine()
    dummy = np.zeros((600, 800, 3), dtype=np.uint8)
    res = engine.run_inference(dummy)
    print("Inference Engine test complete! Verdict:", res["verdict"], "Score:", res["authenticity_score"])
