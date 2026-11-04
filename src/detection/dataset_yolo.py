"""
YOLO Dataset Preparation Module
================================
Converts banknote dataset and security feature annotations into standard YOLOv8 format:
- Classes:
    0: watermark
    1: security_thread
    2: latent_image
    3: number_panel
- Generates normalized [x_center, y_center, width, height] label files
- Creates dataset.yaml configuration file for YOLOv8 training
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
import yaml

from src.data.synthetic_fake_generator import BanknoteSecurityCoordinates

CLASS_NAMES = ["watermark", "security_thread", "latent_image", "number_panel"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class BanknoteYOLOFormatter:
    """Formats banknote images and security feature coordinates for YOLOv8 training."""

    def __init__(
        self,
        splits_csv_dir: str = "data/processed/splits",
        yolo_root: str = "data/processed/yolo"
    ):
        self.splits_csv_dir = Path(splits_csv_dir)
        self.yolo_root = Path(yolo_root)

    @staticmethod
    def convert_box_to_yolo(ymin: float, xmin: float, ymax: float, xmax: float) -> Tuple[float, float, float, float]:
        """Convert [ymin, xmin, ymax, xmax] (normalized) to [x_center, y_center, width, height]."""
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + (width / 2.0)
        y_center = ymin + (height / 2.0)
        return (
            max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, width)),
            max(0.0, min(1.0, height))
        )

    def generate_yolo_annotations_for_split(self, split_name: str) -> int:
        """Processes images in a split and writes YOLO images and label text files."""
        csv_file = self.splits_csv_dir / f"{split_name}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Split file missing: {csv_file}")

        df = pd.read_csv(csv_file)
        split_images_dir = self.yolo_root / "images" / split_name
        split_labels_dir = self.yolo_root / "labels" / split_name

        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for _, row in df.iterrows():
            src_img_path = Path(row["image_path"])
            if not src_img_path.exists():
                continue

            # Standardize filename
            dst_img_name = f"{row['denomination']}_{'fake' if row['is_synthetic'] else 'gen'}_{src_img_path.name}"
            dst_img_path = split_images_dir / dst_img_name
            label_path = split_labels_dir / f"{dst_img_path.stem}.txt"

            shutil.copy2(src_img_path, dst_img_path)

            # Write bounding box annotations for all 4 security features
            label_lines = []
            for feat_name, class_id in CLASS_TO_ID.items():
                if feat_name in BanknoteSecurityCoordinates.FEATURE_REGIONS:
                    ymin, xmin, ymax, xmax = BanknoteSecurityCoordinates.FEATURE_REGIONS[feat_name]
                    xc, yc, w, h = self.convert_box_to_yolo(ymin, xmin, ymax, xmax)
                    label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            count += 1

        print(f"Split '{split_name}': Processed {count} images with YOLO annotations.")
        return count

    def create_dataset_yaml(self) -> Path:
        """Creates dataset.yaml configuration file for Ultralytics YOLOv8."""
        yaml_path = self.yolo_root / "dataset.yaml"
        abs_yolo_root = self.yolo_root.resolve().as_posix()

        config = {
            "path": abs_yolo_root,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            "nc": len(CLASS_NAMES)
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"Created YOLO dataset yaml configuration at: {yaml_path}")
        return yaml_path

    def prepare_all_splits(self):
        """Prepares train, val, and test splits along with dataset.yaml."""
        for split in ["train", "val", "test"]:
            self.generate_yolo_annotations_for_split(split)
        return self.create_dataset_yaml()


if __name__ == "__main__":
    formatter = BanknoteYOLOFormatter()
    yaml_config = formatter.prepare_all_splits()
    print("YOLO dataset preparation complete.")
