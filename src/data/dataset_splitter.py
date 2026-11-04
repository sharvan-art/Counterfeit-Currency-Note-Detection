"""
Dataset Stratified Splitter
============================
Performs a stratified 70 / 15 / 15 train / val / test split across:
- 7 Denominations (10, 20, 50, 100, 200, 500, 2000)
- Authenticity Class (Genuine vs. Synthetic Fake)

Exports clean reproducible CSV catalogs to `data/processed/splits/`.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class BanknoteDatasetSplitter:
    """Stratified splitter for multimodal banknote dataset."""

    def __init__(
        self,
        splits_dir: str = "data/processed/splits",
        random_state: int = 42
    ):
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def collect_dataset_records(
        self,
        genuine_dir: str = "data/raw",
        synthetic_dir: str = "data/synthetic_fake"
    ) -> pd.DataFrame:
        """Indexes all images and extracts denomination + authenticity labels."""
        records = []
        
        # 1. Genuine images
        gen_path = Path(genuine_dir)
        if gen_path.exists():
            for root, _, files in os.walk(gen_path):
                folder_name = Path(root).name
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        denom_digits = ''.join(filter(str.isdigit, folder_name))
                        denom = int(denom_digits) if denom_digits else 0
                        records.append({
                            "image_path": str(Path(root) / file),
                            "filename": file,
                            "denomination": denom,
                            "is_genuine": 1,
                            "is_synthetic": 0,
                            "strata": f"gen_{denom}"
                        })

        # 2. Synthetic fake images
        syn_path = Path(synthetic_dir)
        if syn_path.exists():
            for root, _, files in os.walk(syn_path):
                folder_name = Path(root).name
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        denom_digits = ''.join(filter(str.isdigit, folder_name))
                        denom = int(denom_digits) if denom_digits else 0
                        records.append({
                            "image_path": str(Path(root) / file),
                            "filename": file,
                            "denomination": denom,
                            "is_genuine": 0,
                            "is_synthetic": 1,
                            "strata": f"fake_{denom}"
                        })

        df = pd.DataFrame(records)
        return df

    def create_stratified_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into 70/15/15 stratified on the joint strata column.
        """
        assert round(train_ratio + val_ratio + test_ratio, 2) == 1.0, "Ratios must sum to 1.0"

        # Check for single-instance classes to avoid stratify failure
        class_counts = df["strata"].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        if rare_classes:
            print(f"Warning: Strata with < 2 samples detected: {rare_classes}. Stratifying with fallback.")
            strat_col = None
        else:
            strat_col = df["strata"]

        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=self.random_state,
            stratify=strat_col
        )

        # Second split: Val vs Test (50/50 of temp)
        temp_strat_col = temp_df["strata"] if strat_col is not None else None
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.random_state,
            stratify=temp_strat_col
        )

        train_df = train_df.assign(split="train")
        val_df = val_df.assign(split="val")
        test_df = test_df.assign(split="test")

        # Save to CSV
        train_path = self.splits_dir / "train.csv"
        val_path = self.splits_dir / "val.csv"
        test_path = self.splits_dir / "test.csv"
        all_splits_path = self.splits_dir / "all_splits.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        full_df.to_csv(all_splits_path, index=False)

        print(f"Stratified Splits Saved successfully:")
        print(f" - Train: {len(train_df)} samples -> {train_path}")
        print(f" - Val:   {len(val_df)} samples -> {val_path}")
        print(f" - Test:  {len(test_df)} samples -> {test_path}")

        return train_df, val_df, test_df


if __name__ == "__main__":
    splitter = BanknoteDatasetSplitter()
    print("BanknoteDatasetSplitter module ready.")
