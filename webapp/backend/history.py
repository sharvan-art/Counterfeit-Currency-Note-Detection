"""
Scan History Manager
====================
Manages persistent storage, retrieval, filtering, and export of banknote inspection scans.
Saves scan metadata to data/scan_history/history.json and inspection images to data/scan_history/images/.
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ScanHistoryManager:
    """Handles persistence and retrieval of historical banknote scans."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = PROJECT_ROOT / "data" / "scan_history"
        else:
            self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.history_file = self.base_dir / "history.json"

        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._save_records([])

    def _load_records(self) -> List[Dict[str, Any]]:
        """Reads all records from the JSON history file."""
        if not self.history_file.exists():
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history file: {e}")
            return []

    def _save_records(self, records: List[Dict[str, Any]]) -> None:
        """Writes records list to the JSON history file."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing history file: {e}")

    def add_scan(self, inference_result: Dict[str, Any]) -> str:
        """
        Saves a new scan inspection result and its images into persistent history.
        
        Returns:
            scan_id (str): Unique identifier for this scan session.
        """
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_label = now.strftime("%d %b %Y, %I:%M %p")
        scan_id = f"SCAN_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Save images
        raw_img_path = self.images_dir / f"{scan_id}_raw.jpg"
        annotated_img_path = self.images_dir / f"{scan_id}_annotated.jpg"
        gradcam_img_path = self.images_dir / f"{scan_id}_gradcam.jpg"

        # Raw / Cropped note
        if "raw_input_image" in inference_result and inference_result["raw_input_image"] is not None:
            raw_rgb = inference_result["raw_input_image"]
            cv2.imwrite(str(raw_img_path), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))

        # Annotated note
        if "annotated_image" in inference_result and inference_result["annotated_image"] is not None:
            ann_rgb = inference_result["annotated_image"]
            cv2.imwrite(str(annotated_img_path), cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR))

        # Grad-CAM heatmap overlay
        if "gradcam_overlay" in inference_result and inference_result["gradcam_overlay"] is not None:
            cam_rgb = inference_result["gradcam_overlay"]
            cv2.imwrite(str(gradcam_img_path), cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR))

        # Build feature summary
        feature_summary = []
        for feat in inference_result.get("feature_breakdown", []):
            feature_summary.append({
                "feature_name": feat.get("feature_name", ""),
                "status": feat.get("status", "Unknown"),
                "score": float(feat.get("authenticity_score", 0.0)),
                "observation": feat.get("observation", "")
            })

        record = {
            "id": scan_id,
            "timestamp": timestamp_str,
            "date_label": date_label,
            "verdict": inference_result.get("verdict", "UNKNOWN"),
            "authenticity_score": float(inference_result.get("authenticity_score", 0.0)),
            "verdict_confidence": float(inference_result.get("verdict_confidence", 0.0)),
            "predicted_denomination": int(inference_result.get("predicted_denomination", 0)),
            "denomination_confidence": float(inference_result.get("denomination_confidence", 0.0)),
            "color": inference_result.get("color", "#10B981"),
            "forensic_flags": inference_result.get("forensic_flags", []),
            "feature_summary": feature_summary,
            "raw_img_path": str(raw_img_path),
            "annotated_img_path": str(annotated_img_path),
            "gradcam_img_path": str(gradcam_img_path)
        }

        records = self._load_records()
        records.insert(0, record)  # Newest first
        self._save_records(records)
        return scan_id

    def get_all_scans(self) -> List[Dict[str, Any]]:
        """Returns all recorded scans, ordered newest first."""
        return self._load_records()

    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific scan by ID."""
        records = self._load_records()
        for r in records:
            if r["id"] == scan_id:
                return r
        return None

    def delete_scan(self, scan_id: str) -> bool:
        """Deletes a specific scan record and its associated images."""
        records = self._load_records()
        filtered = []
        found = False
        for r in records:
            if r["id"] == scan_id:
                found = True
                # Remove associated images
                for key in ["raw_img_path", "annotated_img_path", "gradcam_img_path"]:
                    p = Path(r.get(key, ""))
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass
            else:
                filtered.append(r)

        if found:
            self._save_records(filtered)
        return found

    def clear_all_history(self) -> None:
        """Wipes all scan history and image files."""
        records = self._load_records()
        for r in records:
            for key in ["raw_img_path", "annotated_img_path", "gradcam_img_path"]:
                p = Path(r.get(key, ""))
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        self._save_records([])

    def export_csv(self) -> str:
        """Exports the scan history summary as a CSV string."""
        records = self._load_records()
        if not records:
            return "Scan ID,Date,Denomination,Verdict,Authenticity Score (%),Confidence (%),Anomalies\n"

        rows = []
        for r in records:
            rows.append({
                "Scan ID": r.get("id"),
                "Date": r.get("date_label"),
                "Denomination (Rs.)": f"₹{r.get('predicted_denomination')}",
                "Verdict": r.get("verdict"),
                "Authenticity Score (%)": round(r.get("authenticity_score", 0.0) * 100, 1),
                "Confidence (%)": round(r.get("verdict_confidence", 0.0) * 100, 1),
                "Anomalies": "; ".join(r.get("forensic_flags", [])) if r.get("forensic_flags") else "None"
            })
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)


if __name__ == "__main__":
    mgr = ScanHistoryManager()
    print("ScanHistoryManager initialized. Total existing scans:", len(mgr.get_all_scans()))
