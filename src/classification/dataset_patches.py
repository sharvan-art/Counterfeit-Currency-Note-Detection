"""
Banknote Dataset Loader & PyTorch Dataset
==========================================
Loads stratified banknote splits, prepares tensors, and applies Albumentations
transforms for train split and deterministic resizing for validation/test splits.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.augmentation import BanknoteAugmenter

DENOMINATION_MAP = {10: 0, 20: 1, 50: 2, 100: 3, 200: 4, 500: 5, 2000: 6}
INV_DENOMINATION_MAP = {v: k for k, v in DENOMINATION_MAP.items()}


class BanknoteClassificationDataset(Dataset):
    """PyTorch Dataset for banknote authenticity and denomination classification."""

    def __init__(
        self,
        csv_file: str,
        is_train: bool = True,
        img_size: int = 224
    ):
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Missing CSV split file: {csv_file}")

        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train
        self.img_size = img_size

        augmenter = BanknoteAugmenter(img_size=img_size)
        self.transform = augmenter.get_train_transforms() if is_train else augmenter.get_val_transforms()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        denom = int(row["denomination"])
        is_genuine = int(row["is_genuine"])

        # Load RGB image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback black image if corrupted/missing
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transform
        transformed = self.transform(image=img)
        tensor_img = transformed["image"]

        denom_class = DENOMINATION_MAP.get(denom, 0)

        return {
            "image": tensor_img,
            "auth_label": torch.tensor(is_genuine, dtype=torch.long),
            "denom_label": torch.tensor(denom_class, dtype=torch.long),
            "path": img_path
        }


def create_dataloaders(
    splits_dir: str = "data/processed/splits",
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, val, and test DataLoaders."""
    splits_path = Path(splits_dir)

    train_ds = BanknoteClassificationDataset(splits_path / "train.csv", is_train=True, img_size=img_size)
    val_ds = BanknoteClassificationDataset(splits_path / "val.csv", is_train=False, img_size=img_size)
    test_ds = BanknoteClassificationDataset(splits_path / "test.csv", is_train=False, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_ld, val_ld, test_ld = create_dataloaders()
    sample = next(iter(train_ld))
    print("Dataloader verified! Batch image tensor:", sample["image"].shape)
    print("Auth labels:", sample["auth_label"].shape, "Denom labels:", sample["denom_label"].shape)
