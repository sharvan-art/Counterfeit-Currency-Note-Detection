"""
Data Augmentation Pipeline
===========================
Defines realistic photographic variability transforms using Albumentations.
Strictly applied ONLY to the training split to guarantee zero test leakage:
- Slight affine rotation (-15 to 15 deg) & perspective skew (mimicking handheld phone capture)
- Ambient lighting / brightness / contrast adjustments
- Hand / finger partial occlusion (CoarseDropout)
- Camera motion blur / Gaussian blur / sensor noise
"""

from typing import Dict, Any, Optional
import numpy as np


class BanknoteAugmenter:
    """Provides training augmentation and test preprocessing transforms."""

    def __init__(self, img_size: int = 640):
        self.img_size = img_size
        self._train_transform = None
        self._val_transform = None

    def get_train_transforms(self):
        """Constructs Albumentations transform pipeline for training split."""
        if self._train_transform is not None:
            return self._train_transform

        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            self._train_transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                # Simulating casual mobile phone capture angles
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.08, 0.08),
                    rotate=(-15, 15),
                    p=0.7
                ),
                A.Perspective(scale=(0.04, 0.09), keep_size=True, p=0.5),
                
                # Environmental illumination variations
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
                
                # Optical and camera sensor imperfections
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.4),
                
                # Partial finger/wallet occlusion simulation
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(int(self.img_size * 0.04), int(self.img_size * 0.12)),
                    hole_width_range=(int(self.img_size * 0.04), int(self.img_size * 0.12)),
                    p=0.4
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        except ImportError:
            import torchvision.transforms as T
            self._train_transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return self._train_transform

    def get_val_transforms(self):
        """Constructs deterministic validation/test preprocessing transforms."""
        if self._val_transform is not None:
            return self._val_transform

        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            self._val_transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        except ImportError:
            import torchvision.transforms as T
            self._val_transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return self._val_transform


if __name__ == "__main__":
    augmenter = BanknoteAugmenter()
    print("BanknoteAugmenter module ready.")
