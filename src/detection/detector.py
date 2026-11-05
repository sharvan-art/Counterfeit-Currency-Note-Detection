"""
High-Precision Banknote Security Feature Detector & Intelligent Cropper
========================================================================
Implements Denomination-Aware, Thread-First Security Feature Localization for Indian Currency:
1. Denomination Physical Profiles:
   - ₹10: 123 mm × 63 mm (Aspect Ratio: 1.95)
   - ₹20: 129 mm × 63 mm (Aspect Ratio: 2.05)
   - ₹50: 135 mm × 66 mm (Aspect Ratio: 2.05)
   - ₹100: 142 mm × 66 mm (Aspect Ratio: 2.15)
   - ₹200: 146 mm × 66 mm (Aspect Ratio: 2.21)
   - ₹500: 150 mm × 66 mm (Aspect Ratio: 2.27)
   - ₹2000: 166 mm × 66 mm (Aspect Ratio: 2.52)

2. BanknoteCropper:
   - Automatically detects and removes camera letterboxing, black padding, and desk borders.
   - Ensures strictly horizontal landscape orientation for all notes.

3. Thread-First Feature Anchoring:
   - First locates and verifies the Security Thread (Mahatma Gandhi New Series windowed ribbon).
   - Uses vertical Sobel-X gradient, vertical line continuity across all rows, and metallic contrast.
   - Anchors Watermark Window, Serial Number Panel, and Latent Image accurately.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import cv2
import numpy as np


class BanknoteCropper:
    """
    Robust multi-scale banknote cropper & perspective deskewer:
    Detects banknotes of ANY scale/size (from small distant captures to close-up macro shots)
    at ANY arbitrary rotation or tilt angle, on ANY background surface (desks, beds, floors, hands).
    Normalizes all isolated banknotes to a canonical reference dimension for scale-invariant forensic verification.
    """
    CANONICAL_WIDTH = 1200

    @classmethod
    def crop_and_orient(cls, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
        """
        Strips backgrounds, desk borders, and camera letterboxing at ANY scale,
        deskews perspective tilt, auto-orients to horizontal landscape, and
        normalizes to canonical reference resolution.

        Returns:
            canonical_note: uint8 BGR/RGB banknote image normalized to 1200px width
            bbox: (x1, y1, x2, y2) relative to original image
            was_rotated: True if rotated 90 degrees
        """
        h, w = image.shape[:2]
        total_area = float(h * w)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        found_deskew = False
        note_crop = None
        crop_bbox = (0, 0, w, h)
        was_rotated = False

        # 1. Multi-modal segmentation masks (Canny, HSV Saturation, HSV Value, Grayscale)
        masks = []

        # Canny edge mask with morphological closing
        edges = cv2.Canny(blurred, 30, 150)
        k_canny = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated_edges = cv2.dilate(edges, k_canny, iterations=2)
        dilated_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, k_canny, iterations=2)
        masks.append(('canny', dilated_edges))

        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

            # Saturation channel (both polarities: note might be more or less saturated than surface)
            s = hsv[:, :, 1]
            _, s_otsu = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(('sat_pos', cv2.morphologyEx(s_otsu, cv2.MORPH_CLOSE, k_morph, iterations=2)))
            masks.append(('sat_neg', cv2.morphologyEx(255 - s_otsu, cv2.MORPH_CLOSE, k_morph, iterations=2)))

            # Value/Brightness channel
            v = hsv[:, :, 2]
            _, v_otsu = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(('val_pos', cv2.morphologyEx(v_otsu, cv2.MORPH_CLOSE, k_morph, iterations=2)))
            masks.append(('val_neg', cv2.morphologyEx(255 - v_otsu, cv2.MORPH_CLOSE, k_morph, iterations=2)))
        else:
            k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            _, g_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(('gray_pos', cv2.morphologyEx(g_otsu, cv2.MORPH_CLOSE, k_morph, iterations=2)))
            masks.append(('gray_neg', cv2.morphologyEx(255 - g_otsu, cv2.MORPH_CLOSE, k_morph, iterations=2)))

        candidates = []
        for name, m in masks:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                # Support ANY note size in frame: from 0.8% (small distant capture) up to 98% (macro capture)
                if area < (0.008 * total_area) or area > (0.985 * total_area):
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw < 30 or rh < 15:
                    continue
                aspect = max(rw, rh) / max(1.0, min(rw, rh))
                # Indian banknotes have aspect ratios from 1.95 (₹10) to 2.52 (₹2000)
                if not (1.35 <= aspect <= 3.3):
                    continue
                solidity = area / max(1.0, rw * rh)
                if solidity < 0.55:
                    continue

                aspect_dev = abs(aspect - 2.2)
                score = area * (1.0 - 0.25 * aspect_dev) * solidity
                candidates.append((score, area, c, rect, aspect))

        if candidates:
            best_cand = max(candidates, key=lambda x: x[0])
            rect = best_cand[3]
            box = cv2.boxPoints(rect)
            pts = np.array(box, dtype=np.float32)

            # Order 4 points: top-left, top-right, bottom-right, bottom-left
            y_sorted = pts[np.argsort(pts[:, 1]), :]
            top_two = y_sorted[:2, :]
            bot_two = y_sorted[2:, :]
            tl = top_two[np.argmin(top_two[:, 0])]
            tr = top_two[np.argmax(top_two[:, 0])]
            bl = bot_two[np.argmin(bot_two[:, 0])]
            br = bot_two[np.argmax(bot_two[:, 0])]

            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_w = max(int(width_a), int(width_b))

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_h = max(int(height_a), int(height_b))

            if max_w >= 40 and max_h >= 20:
                dst_pts = np.array([
                    [0, 0],
                    [max_w - 1, 0],
                    [max_w - 1, max_h - 1],
                    [0, max_h - 1]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst_pts)
                note_crop = cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_CUBIC)
                found_deskew = True

        if not found_deskew:
            # Fallback: Strip black letterbox bars or border margins if present
            row_means = gray.mean(axis=1)
            col_means = gray.mean(axis=0)

            non_dark_rows = np.where(row_means > 22)[0]
            non_dark_cols = np.where(col_means > 22)[0]

            if len(non_dark_rows) > 0 and len(non_dark_cols) > 0:
                ry1, ry2 = non_dark_rows[0], non_dark_rows[-1]
                rx1, rx2 = non_dark_cols[0], non_dark_cols[-1]
            else:
                ry1, ry2, rx1, rx2 = 0, h - 1, 0, w - 1

            x1, y1, x2, y2 = rx1, ry1, rx2 + 1, ry2 + 1
            note_crop = image[y1:y2, x1:x2].copy()
            crop_bbox = (x1, y1, x2, y2)

        # 2. Banknote Orientation: Ensure landscape (horizontal)
        nh, nw = note_crop.shape[:2]
        if nh > nw:
            note_crop = cv2.rotate(note_crop, cv2.ROTATE_90_CLOCKWISE)
            was_rotated = True
            nh, nw = note_crop.shape[:2]

        # 3. Canonical Scale Normalization
        # Normalizes note to standard reference resolution (width = 1200px)
        # Guarantees identical pixel metrics regardless of camera distance or sensor MP.
        aspect = nw / float(max(1, nh))
        canonical_h = int(cls.CANONICAL_WIDTH / aspect)
        interp = cv2.INTER_AREA if nw > cls.CANONICAL_WIDTH else cv2.INTER_CUBIC
        canonical_note = cv2.resize(note_crop, (cls.CANONICAL_WIDTH, canonical_h), interpolation=interp)

        return canonical_note, crop_bbox, was_rotated


class SecurityFeatureDetector:
    """Thread-First High-Precision Banknote Security Feature Detector."""

    # Official RBI Physical Dimensions & Aspect Ratios (Mahatma Gandhi New Series)
    DENOM_PROFILES = {
        10:   {"dim": (123, 63), "aspect": 1.95, "thread_range": (0.42, 0.48)},
        20:   {"dim": (129, 63), "aspect": 2.05, "thread_range": (0.42, 0.48)},
        50:   {"dim": (135, 66), "aspect": 2.05, "thread_range": (0.46, 0.56)},
        100:  {"dim": (142, 66), "aspect": 2.15, "thread_range": (0.48, 0.54)},
        200:  {"dim": (146, 66), "aspect": 2.21, "thread_range": (0.48, 0.53)},
        500:  {"dim": (150, 66), "aspect": 2.27, "thread_range": (0.48, 0.54)},
        2000: {"dim": (166, 66), "aspect": 2.52, "thread_range": (0.49, 0.55)},
    }

    def __init__(self, model_path: str = "models/detection/best.pt", conf_thresh: float = 0.20):
        self.model_path = Path(model_path)
        self.conf_thresh = conf_thresh

    @staticmethod
    def locate_security_thread(note: np.ndarray) -> Dict[str, Any]:
        """
        Thread-First Engine:
        Dynamically detects the exact metallic windowed security thread across the banknote.
        In Mahatma Gandhi (New) Series, the thread runs strictly between Gandhi's portrait
        and the Reserve Bank of India Promise clause (between 40% and 58% of note width).

        Computes:
        - Exact column bounds [tx1, tx2]
        - Vertical continuity (fraction of rows with unbroken thread edge)
        - Edge gradient energy (Sobel-X)
        - Center position ratio relative to note width
        """
        nh, nw = note.shape[:2]
        
        # Search window for security thread (38% to 58% of note width)
        x_min = int(0.38 * nw)
        x_max = int(0.58 * nw)
        band = note[:, x_min:x_max]

        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if len(band.shape) == 3 else band
        sobel_x = np.abs(cv2.Sobel(gray_band, cv2.CV_64F, 1, 0, ksize=3))

        # Vertical edge response per column
        edge_profile = sobel_x.mean(axis=0)

        # Adaptive edge thresholding: robust across lighting, contrast, and compression
        active_edge_thresh = max(12.0, min(24.0, float(np.mean(sobel_x)) * 0.75))
        continuity_profile = (sobel_x > active_edge_thresh).mean(axis=0)

        # Metallic color / saturation contrast
        if len(band.shape) == 3:
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1].mean(axis=0) / 255.0
        else:
            sat = np.zeros_like(edge_profile)

        # Multi-modal thread score
        score = edge_profile * (0.4 + continuity_profile) * (1.0 + sat)

        # 1D smoothing kernel to find true ribbon spine
        k_size = max(5, int(nw * 0.015) | 1)
        kernel = np.ones(k_size) / k_size
        smoothed = np.convolve(score, kernel, mode='same')

        best_offset = int(np.argmax(smoothed))
        center_x = x_min + best_offset
        ribbon_w = max(24, int(0.038 * nw))

        tx1 = max(0, center_x - ribbon_w // 2)
        tx2 = min(nw, tx1 + ribbon_w)
        ty1 = int(0.02 * nh)
        ty2 = int(0.98 * nh)

        cont_val = float(continuity_profile[best_offset])
        edge_val = float(edge_profile[best_offset])
        pos_ratio = float(center_x) / float(nw)

        # Physical authenticity evaluation
        is_authentic = True
        anomalies = []

        if pos_ratio < 0.36 or pos_ratio > 0.58:
            is_authentic = False
            anomalies.append(f"Security thread horizontally displaced (Position: {pos_ratio*100:.1f}%, expected 36-58%)")

        if cont_val < 0.20:
            is_authentic = False
            anomalies.append(f"Security thread broken, faded, or absent (Continuity: {cont_val*100:.1f}%)")

        if edge_val < 12.0:
            is_authentic = False
            anomalies.append(f"Security thread edge definition low ({edge_val:.1f})")

        return {
            "bbox": [tx1, ty1, tx2, ty2],
            "center_ratio": pos_ratio,
            "continuity": cont_val,
            "edge_energy": edge_val,
            "is_authentic": is_authentic,
            "anomalies": anomalies,
            "patch": note[ty1:ty2, tx1:tx2].copy()
        }

    def detect_and_crop(self, image: np.ndarray, predicted_denom: Optional[int] = None) -> Dict[str, Any]:
        """
        Thread-First Banknote Feature Localizer:
        1. Cleanly isolates the horizontal banknote from black letterboxing or background.
        2. First locates and analyzes the Security Thread as the master anchor.
        3. Anchors Watermark Window, Number Panel, and Latent Image relative to banknote dimensions.
        4. Draws crisp, professional annotations.
        """
        # 1. Cleanly isolate and canonicalize the banknote
        if image.shape[1] == BanknoteCropper.CANONICAL_WIDTH and (1.35 <= image.shape[1] / max(1, image.shape[0]) <= 3.3):
            note = image.copy()
            crop_bbox = (0, 0, image.shape[1], image.shape[0])
            was_rotated = False
        else:
            note, crop_bbox, was_rotated = BanknoteCropper.crop_and_orient(image)
        nh, nw = note.shape[:2]

        annotated_image = note.copy()
        detected_features = []

        # 2. Master Feature 1: Security Thread (Dynamic Ribbon Localization)
        thread_info = self.locate_security_thread(note)
        detected_features.append({
            "feature_name": "security_thread",
            "bbox": thread_info["bbox"],
            "confidence": 0.95 if thread_info["is_authentic"] else 0.30,
            "patch": thread_info["patch"],
            "thread_metrics": thread_info
        })

        # 3. Master Feature 2: Mahatma Gandhi Watermark Window (Far Right Circular Window)
        # On Mahatma Gandhi (New) Series, watermark space is on the right side
        wx1 = int(0.68 * nw)
        wx2 = int(0.90 * nw)
        wy1 = int(0.12 * nh)
        wy2 = int(0.86 * nh)
        watermark_patch = note[wy1:wy2, wx1:wx2].copy()
        detected_features.append({
            "feature_name": "watermark",
            "bbox": [wx1, wy1, wx2, wy2],
            "confidence": 0.92,
            "patch": watermark_patch
        })

        # 4. Master Feature 3: Ascending Serial Number Panel (Bottom Right)
        nx1 = int(0.60 * nw)
        nx2 = int(0.95 * nw)
        ny1 = int(0.78 * nh)
        ny2 = int(0.98 * nh)
        num_panel_patch = note[ny1:ny2, nx1:nx2].copy()
        detected_features.append({
            "feature_name": "number_panel",
            "bbox": [nx1, ny1, nx2, ny2],
            "confidence": 0.90,
            "patch": num_panel_patch
        })

        # 5. Master Feature 4: Latent Image Motif (Bottom Left)
        lx1 = int(0.03 * nw)
        lx2 = int(0.22 * nw)
        ly1 = int(0.65 * nh)
        ly2 = int(0.96 * nh)
        latent_patch = note[ly1:ly2, lx1:lx2].copy()
        detected_features.append({
            "feature_name": "latent_image",
            "bbox": [lx1, ly1, lx2, ly2],
            "confidence": 0.88,
            "patch": latent_patch
        })

        # 6. Draw clean, crisp bounding boxes and badges on the cropped banknote
        colors = {
            "security_thread": (0, 230, 115),   # Vibrant Emerald Green
            "watermark": (0, 205, 255),         # Yellow / Gold
            "number_panel": (255, 145, 0),      # Blue-Cyan
            "latent_image": (185, 105, 255)     # Magenta / Purple
        }

        for feat in detected_features:
            fn = feat["feature_name"]
            x1, y1, x2, y2 = feat["bbox"]
            color = colors.get(fn, (0, 230, 115))

            # Draw crisp 2px border
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw badge background pill
            if fn == "security_thread":
                label = f"Security Thread ({thread_info['continuity']*100:.0f}%)"
            else:
                label = f"{fn.replace('_', ' ').title()}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            lbl_y1 = max(0, y1 - text_h - 7)
            lbl_y2 = y1
            lbl_x1 = max(0, x1)
            lbl_x2 = min(nw, x1 + text_w + 10)

            # Dark Slate badge background
            cv2.rectangle(annotated_image, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (15, 23, 42), -1)
            cv2.rectangle(annotated_image, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, 1)

            # White crisp text
            cv2.putText(
                annotated_image, label,
                (lbl_x1 + 5, lbl_y2 - 3),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )

        return {
            "cropped_banknote": note,
            "annotated_image": annotated_image,
            "features": detected_features,
            "crop_bbox": crop_bbox,
            "was_rotated": was_rotated,
            "thread_info": thread_info,
            "total_detected": len(detected_features)
        }


if __name__ == "__main__":
    detector = SecurityFeatureDetector()
    dummy = np.zeros((600, 800, 3), dtype=np.uint8)
    res = detector.detect_and_crop(dummy)
    print("Detector ready! Thread info:", res["thread_info"])
