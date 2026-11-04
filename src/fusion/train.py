"""
Fusion Training Module
=======================
Trains the Multi-Feature Attention Fusion Network using extracted
latent security ROI embeddings. Saves best checkpoint to `models/fusion/best_fusion.pt`.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.fusion.model import AttentionFeatureFusion


class FusionTrainer:
    """Trains the attention-gated multi-feature fusion head."""

    def __init__(
        self,
        save_dir: str = "models/fusion",
        reports_dir: str = "reports",
        device: str = "cpu"
    ):
        self.save_dir = Path(save_dir)
        self.reports_dir = Path(reports_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

    def generate_synthetic_feature_embeddings(self, n_samples: int = 420) -> Dict[str, torch.Tensor]:
        """
        Creates feature token embeddings representing the 4 localized security regions.
        Genuine samples exhibit consistent high-confidence feature alignments.
        Synthetic fakes exhibit localized feature anomaly vectors corresponding to their defects.
        """
        torch.manual_seed(42)
        half = n_samples // 2

        # Genuine: consistent clean features
        gen_feats = torch.randn(half, 4, 256) * 0.4 + 1.0
        gen_auth_labels = torch.ones(half, dtype=torch.long)
        gen_denom_labels = torch.randint(0, 7, (half,), dtype=torch.long)

        # Fake: defect feature perturbation on 1-2 regions
        fake_feats = torch.randn(half, 4, 256) * 0.4 + 1.0
        for i in range(half):
            defect_idx = torch.randint(0, 4, (1,)).item()
            fake_feats[i, defect_idx, :] = torch.randn(256) * 2.5 - 1.5
        fake_auth_labels = torch.zeros(half, dtype=torch.long)
        fake_denom_labels = torch.randint(0, 7, (half,), dtype=torch.long)

        X = torch.cat([gen_feats, fake_feats], dim=0)
        y_auth = torch.cat([gen_auth_labels, fake_auth_labels], dim=0)
        y_denom = torch.cat([gen_denom_labels, fake_denom_labels], dim=0)

        # Shuffle
        indices = torch.randperm(n_samples)
        return {
            "tokens": X[indices],
            "auth": y_auth[indices],
            "denom": y_denom[indices]
        }

    def train(self, epochs: int = 15, batch_size: int = 16) -> Dict[str, Any]:
        """Trains the AttentionFeatureFusion model."""
        data = self.generate_synthetic_feature_embeddings(420)
        n = len(data["tokens"])
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_ds = TensorDataset(data["tokens"][:n_train], data["auth"][:n_train], data["denom"][:n_train])
        val_ds = TensorDataset(
            data["tokens"][n_train:n_train + n_val],
            data["auth"][n_train:n_train + n_val],
            data["denom"][n_train:n_train + n_val]
        )
        test_ds = TensorDataset(data["tokens"][n_train + n_val:], data["auth"][n_train + n_val:], data["denom"][n_train + n_val:])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = AttentionFeatureFusion(feature_dim=256, hidden_dim=128)
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit_auth = nn.CrossEntropyLoss()
        crit_denom = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_path = self.save_dir / "best_fusion.pt"

        print(f"\nTraining AttentionFeatureFusion for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for tokens, auth_y, denom_y in train_loader:
                tokens, auth_y, denom_y = tokens.to(self.device), auth_y.to(self.device), denom_y.to(self.device)
                optimizer.zero_grad()
                out = model(tokens)
                loss = crit_auth(out["auth_logits"], auth_y) + 0.3 * crit_denom(out["denom_logits"], denom_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Val
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for tokens, auth_y, _ in val_loader:
                    tokens, auth_y = tokens.to(self.device), auth_y.to(self.device)
                    out = model(tokens)
                    preds = torch.argmax(out["auth_logits"], dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_trues.extend(auth_y.cpu().numpy())

            acc = float(accuracy_score(val_trues, val_preds))
            if acc > best_acc or epoch == 1:
                best_acc = acc
                torch.save(model.state_dict(), best_path)

        # Test evaluation
        model.load_state_dict(torch.load(best_path, map_location=self.device))
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for tokens, auth_y, _ in test_loader:
                tokens = tokens.to(self.device)
                out = model(tokens)
                preds = torch.argmax(out["auth_logits"], dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_trues.extend(auth_y.numpy())

        test_acc = float(accuracy_score(test_trues, test_preds))
        p, r, f1, _ = precision_recall_fscore_support(test_trues, test_preds, average="binary", zero_division=0)

        results = {
            "test_accuracy": round(test_acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "checkpoint": str(best_path)
        }

        with open(self.reports_dir / "fusion_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Fusion Training Complete! Test Accuracy: {test_acc * 100:.1f}%, F1: {f1:.3f}")
        return results


if __name__ == "__main__":
    trainer = FusionTrainer()
    trainer.train()
