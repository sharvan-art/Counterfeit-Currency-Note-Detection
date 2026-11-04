"""
YOLOv8 Security Feature Detection Trainer
==========================================
Fine-tunes YOLOv8 on localized banknote security features:
- Watermark
- Security Thread
- Latent Image
- Number Panel

Computes and saves mAP50, mAP50-95, and precision/recall metrics.
Saves model weights to `models/detection/best.pt`.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


class BanknoteYOLOTrainer:
    """Trainer and evaluator for YOLOv8 banknote security feature detector."""

    def __init__(
        self,
        yaml_config_path: str = "data/processed/yolo/dataset.yaml",
        model_name: str = "yolov8n.pt",
        save_dir: str = "models/detection"
    ):
        self.yaml_config_path = Path(yaml_config_path)
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        epochs: int = 15,
        imgsz: int = 640,
        batch_size: int = 16,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Fine-tune YOLOv8 model."""
        from ultralytics import YOLO

        print(f"Loading pretrained model: {self.model_name}...")
        model = YOLO(self.model_name)

        print(f"Starting YOLOv8 training for {epochs} epochs on {device}...")
        results = model.train(
            data=str(self.yaml_config_path.resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=str(self.save_dir),
            name="train_run",
            exist_ok=True,
            verbose=True,
            plots=True
        )

        # Validate on test split
        print("Evaluating on test split...")
        val_results = model.val(
            data=str(self.yaml_config_path.resolve()),
            split="test",
            device=device
        )

        metrics = {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "epochs": epochs,
            "img_size": imgsz
        }

        # Copy best model to standard location
        trained_best = self.save_dir / "train_run" / "weights" / "best.pt"
        target_best = self.save_dir / "best.pt"
        if trained_best.exists():
            import shutil
            shutil.copy2(trained_best, target_best)
            print(f"Saved primary model checkpoint to: {target_best}")

        # Save metrics JSON
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "detection_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Detection training complete! Test mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
        return metrics


if __name__ == "__main__":
    trainer = BanknoteYOLOTrainer()
    print("BanknoteYOLOTrainer ready.")
