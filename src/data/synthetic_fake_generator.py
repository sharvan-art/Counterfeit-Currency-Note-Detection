"""
Synthetic Fake Note Generator
==============================
Applies domain-specific, security-feature-targeted alterations to genuine Indian banknote images.

IMPORTANT LEGAL & ETHICAL NOTICE:
All altered images produced by this script are explicitly tagged as synthetic fakes
(`synthetic=True`, `defect_type=...`). These are generated exclusively for computer vision
training and academic research. They do NOT represent genuine counterfeit currency.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class BanknoteSecurityCoordinates:
    """
    Normalized approximate coordinate bounding boxes [ymin, xmin, ymax, xmax]
    for typical Indian currency notes (Mahatma Gandhi New Series).
    """
    FEATURE_REGIONS = {
        "watermark": (0.12, 0.68, 0.86, 0.90),        # Gandhi watermark window on right
        "security_thread": (0.02, 0.46, 0.98, 0.54),  # Windowed security thread
        "latent_image": (0.65, 0.03, 0.96, 0.22),     # Latent image bottom-left
        "number_panel": (0.78, 0.60, 0.98, 0.95),     # Bottom right ascending number panel
        "color_shift_patch": (0.40, 0.52, 0.65, 0.68) # Rupee symbol color-shifting ink
    }

    @classmethod
    def get_pixel_box(cls, feature: str, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        ymin, xmin, ymax, xmax = cls.FEATURE_REGIONS.get(feature, (0.2, 0.2, 0.8, 0.8))
        return int(ymin * img_h), int(xmin * img_w), int(ymax * img_h), int(xmax * img_w)


class SyntheticFakeGenerator:
    """
    Generates synthetic defect banknote images targeting specific security features.
    """

    def __init__(self, output_dir: str = "data/synthetic_fake", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)

    def blur_watermark(self, image: np.ndarray) -> np.ndarray:
        """Simulate missing or poorly printed watermark by excessive smoothing / inpainting."""
        h, w = image.shape[:2]
        y1, x1, y2, x2 = BanknoteSecurityCoordinates.get_pixel_box("watermark", h, w)
        img_out = image.copy()
        roi = img_out[y1:y2, x1:x2]
        
        # Heavy bilateral filter to erase fine portrait watermarks
        blurred_roi = cv2.bilateralFilter(roi, d=25, sigmaColor=150, sigmaSpace=150)
        # Apply Gaussian blur overlay
        blurred_roi = cv2.GaussianBlur(blurred_roi, (21, 21), 0)
        img_out[y1:y2, x1:x2] = blurred_roi
        return img_out

    def remove_watermark(self, image: np.ndarray) -> np.ndarray:
        """Completely replace the watermark region with local background average."""
        h, w = image.shape[:2]
        y1, x1, y2, x2 = BanknoteSecurityCoordinates.get_pixel_box("watermark", h, w)
        img_out = image.copy()
        
        # Inpaint or texture fill using surrounding background
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        img_out = cv2.inpaint(img_out, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return img_out

    def shift_security_thread(self, image: np.ndarray) -> np.ndarray:
        """Misalign or shift the security thread horizontally or distort its continuity."""
        h, w = image.shape[:2]
        from src.detection.detector import SecurityFeatureDetector
        th_info = SecurityFeatureDetector.locate_security_thread(image)
        x1, y1, x2, y2 = th_info["bbox"]
        img_out = image.copy()
        thread_roi = img_out[y1:y2, x1:x2].copy()
        
        # Shift substantially outside valid window (into Gandhi portrait or promise clause)
        shift_pixels = random.randint(45, 80) * random.choice([-1, 1])
        new_x1 = max(0, min(w - (x2 - x1), x1 + shift_pixels))
        new_x2 = new_x1 + (x2 - x1)
        
        # Inpaint original thread location
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, max(0, x1-2):min(w, x2+2)] = 255
        img_out = cv2.inpaint(img_out, mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

        # Paste shifted thread
        img_out[y1:y2, new_x1:new_x2] = cv2.addWeighted(
            img_out[y1:y2, new_x1:new_x2], 0.2, thread_roi, 0.8, 0
        )
        return img_out

    def remove_security_thread(self, image: np.ndarray) -> np.ndarray:
        """Erase the metallic security thread using local context inpainting."""
        h, w = image.shape[:2]
        from src.detection.detector import SecurityFeatureDetector
        th_info = SecurityFeatureDetector.locate_security_thread(image)
        x1, y1, x2, y2 = th_info["bbox"]
        img_out = image.copy()
        
        # Inpaint full thread ribbon cleanly
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, max(0, x1-4):min(w, x2+4)] = 255
        img_out = cv2.inpaint(img_out, mask, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
        # Apply slight texture blur to remove any remaining metallic traces
        roi = img_out[y1:y2, max(0, x1-4):min(w, x2+4)]
        if roi.size > 0:
            img_out[y1:y2, max(0, x1-4):min(w, x2+4)] = cv2.GaussianBlur(roi, (7, 7), 0)
        return img_out

    def alter_color_shift_patch(self, image: np.ndarray) -> np.ndarray:
        """Distort optically variable ink by shifting hue to unnatural static green/brown."""
        h, w = image.shape[:2]
        y1, x1, y2, x2 = BanknoteSecurityCoordinates.get_pixel_box("color_shift_patch", h, w)
        img_out = image.copy()
        patch = img_out[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Shift hue channel significantly to destroy genuine green-to-blue shift
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(50, 110)) % 180
        # Flatten saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.6, 0, 255)
        altered_patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img_out[y1:y2, x1:x2] = altered_patch
        return img_out

    def degrade_microtext(self, image: np.ndarray) -> np.ndarray:
        """Degrade high-resolution micro-lettering using downsampling, blur, and noise."""
        h, w = image.shape[:2]
        # Microtext commonly runs along borders and portrait margins
        scale = random.uniform(0.2, 0.35)
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        # Add high-frequency Gaussian noise
        noise = np.random.normal(0, 12, degraded.shape).astype(np.float32)
        degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return degraded

    def misalign_number_panel(self, image: np.ndarray) -> np.ndarray:
        """Disrupt font progression and alignment in the ascending serial number panel."""
        h, w = image.shape[:2]
        y1, x1, y2, x2 = BanknoteSecurityCoordinates.get_pixel_box("number_panel", h, w)
        img_out = image.copy()
        panel = img_out[y1:y2, x1:x2].copy()
        
        # Break into vertical slices simulating individual misaligned digits
        slices = np.array_split(panel, 6, axis=1)
        misaligned_slices = []
        for s in slices:
            dy = random.randint(-4, 4)
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            shifted = cv2.warpAffine(s, M, (s.shape[1], s.shape[0]), borderMode=cv2.BORDER_REFLECT)
            misaligned_slices.append(shifted)
        
        reconstructed = np.hstack(misaligned_slices)
        img_out[y1:y2, x1:x2] = reconstructed
        return img_out

    def generate_fake(
        self,
        image_path: str,
        denomination: int,
        defects: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generates a synthetic fake banknote image from a genuine source image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")

        # Ensure horizontal landscape orientation
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        available_defects = {
            "blur_watermark": self.blur_watermark,
            "remove_watermark": self.remove_watermark,
            "shift_security_thread": self.shift_security_thread,
            "remove_security_thread": self.remove_security_thread,
            "alter_color_shift_patch": self.alter_color_shift_patch,
            "degrade_microtext": self.degrade_microtext,
            "misalign_number_panel": self.misalign_number_panel
        }

        if not defects:
            # Pick 1 to 3 random defect types
            num_defects = random.randint(1, 3)
            defects = random.sample(list(available_defects.keys()), num_defects)

        result_img = img.copy()
        applied_defects = []
        for defect in defects:
            if defect in available_defects:
                result_img = available_defects[defect](result_img)
                applied_defects.append(defect)

        metadata = {
            "source_image": os.path.basename(image_path),
            "denomination": denomination,
            "is_genuine": False,
            "synthetic": True,
            "defects_applied": applied_defects,
            "legal_notice": "Synthetically altered for academic research. Not real counterfeit currency."
        }
        return result_img, metadata

    def batch_generate(
        self,
        genuine_dir: str,
        count_per_denomination: int = 100
    ) -> List[Dict]:
        """
        Processes a genuine dataset folder and generates synthetic fakes.
        """
        all_metadata = []
        p = Path(genuine_dir)
        
        for denom_folder in p.iterdir():
            if not denom_folder.is_dir():
                continue
            
            try:
                denom_val = int(''.join(filter(str.isdigit, denom_folder.name)))
            except ValueError:
                continue

            images = list(denom_folder.glob("*.jpg")) + list(denom_folder.glob("*.png"))
            if not images:
                continue

            target_dir = self.output_dir / f"denom_{denom_val}"
            target_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count_per_denomination):
                src_img_path = random.choice(images)
                fake_img, meta = self.generate_fake(str(src_img_path), denom_val)
                
                out_filename = f"synthetic_fake_{denom_val}_{i:04d}.jpg"
                out_path = target_dir / out_filename
                cv2.imwrite(str(out_path), fake_img)
                
                meta["saved_path"] = str(out_path)
                meta["filename"] = out_filename
                all_metadata.append(meta)

        # Save synthetic catalog
        catalog_path = self.output_dir / "synthetic_fakes_catalog.json"
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2)

        return all_metadata


if __name__ == "__main__":
    generator = SyntheticFakeGenerator()
    print("SyntheticFakeGenerator initialized successfully.")
