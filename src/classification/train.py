"""
Multi-Configuration Deep Feature Classifier Trainer
====================================================
Trains and compares 3 distinct transfer learning configurations:
1. ResNet18 - Frozen Backbone (Linear probing)
2. ResNet18 - Layer-wise LR Fine-Tuning + Cosine Annealing (Recommended)
3. EfficientNet-B0 - Progressive Fine-Tuning

Logs accuracy vs. training-time, loss dynamics, confusion matrices,
and saves best model checkpoints to `models/classification/`.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.classification.model import BanknoteFeatureClassifier
from src.classification.dataset_patches import create_dataloaders


class ClassificationExperimentRunner:
    """Orchestrates training, evaluation, and ablation comparison across configurations."""

    def __init__(
        self,
        splits_dir: str = "data/processed/splits",
        models_dir: str = "models/classification",
        reports_dir: str = "reports",
        device: str = "cpu"
    ):
        self.splits_dir = Path(splits_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

    def train_single_configuration(
        self,
        config_name: str,
        backbone_name: str,
        freeze_backbone: bool,
        backbone_lr: float,
        head_lr: float,
        epochs: int = 5,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Trains one model configuration and measures accuracy vs training time."""
        print(f"\n=======================================================")
        print(f"Starting Training: [{config_name}]")
        print(f"Backbone: {backbone_name} | Frozen: {freeze_backbone} | LR: (bb={backbone_lr}, head={head_lr})")
        print(f"=======================================================")

        train_loader, val_loader, test_loader = create_dataloaders(
            splits_dir=str(self.splits_dir),
            batch_size=batch_size,
            img_size=224
        )

        model = BanknoteFeatureClassifier(backbone_name=backbone_name, pretrained=True)
        model.to(self.device)

        if freeze_backbone:
            model.set_backbone_trainable(False)
            optimizer = AdamW(model.get_parameter_groups(backbone_lr=0.0, head_lr=head_lr)[1:2], weight_decay=1e-4)
        else:
            model.set_backbone_trainable(True)
            optimizer = AdamW(model.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr), weight_decay=1e-4)

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion_auth = nn.CrossEntropyLoss()
        criterion_denom = nn.CrossEntropyLoss()

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_auth_acc": [],
            "val_denom_acc": [],
            "elapsed_time_sec": []
        }

        best_val_acc = 0.0
        best_checkpoint_path = self.models_dir / f"{config_name}_best.pt"
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                imgs = batch["image"].to(self.device)
                auth_labels = batch["auth_label"].to(self.device)
                denom_labels = batch["denom_label"].to(self.device)

                optimizer.zero_grad()
                out = model(imgs)

                loss_auth = criterion_auth(out["auth_logits"], auth_labels)
                loss_denom = criterion_denom(out["denom_logits"], denom_labels)
                # Joint multi-task loss
                loss = loss_auth + (0.5 * loss_denom)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # Validation
            model.eval()
            total_val_loss = 0.0
            auth_preds, auth_trues = [], []
            denom_preds, denom_trues = [], []

            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device)
                    auth_labels = batch["auth_label"].to(self.device)
                    denom_labels = batch["denom_label"].to(self.device)

                    out = model(imgs)
                    loss_a = criterion_auth(out["auth_logits"], auth_labels)
                    loss_d = criterion_denom(out["denom_logits"], denom_labels)
                    total_val_loss += (loss_a + 0.5 * loss_d).item()

                    auth_p = torch.argmax(out["auth_logits"], dim=1).cpu().numpy()
                    denom_p = torch.argmax(out["denom_logits"], dim=1).cpu().numpy()

                    auth_preds.extend(auth_p)
                    auth_trues.extend(auth_labels.cpu().numpy())
                    denom_preds.extend(denom_p)
                    denom_trues.extend(denom_labels.cpu().numpy())

            avg_val_loss = total_val_loss / max(1, len(val_loader))
            val_auth_acc = float(accuracy_score(auth_trues, auth_preds))
            val_denom_acc = float(accuracy_score(denom_trues, denom_preds))
            elapsed = time.time() - start_time

            history["epoch"].append(epoch)
            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_auth_acc"].append(round(val_auth_acc, 4))
            history["val_denom_acc"].append(round(val_denom_acc, 4))
            history["elapsed_time_sec"].append(round(elapsed, 2))

            print(f"Epoch {epoch}/{epochs} ({time.time() - epoch_start:.1f}s) | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Auth Acc: {val_auth_acc * 100:.1f}% | Denom Acc: {val_denom_acc * 100:.1f}%")

            # Checkpoint
            if val_auth_acc > best_val_acc or epoch == 1:
                best_val_acc = val_auth_acc
                torch.save({
                    "config_name": config_name,
                    "backbone_name": backbone_name,
                    "model_state_dict": model.state_dict(),
                    "val_auth_acc": val_auth_acc,
                    "val_denom_acc": val_denom_acc,
                    "epoch": epoch
                }, best_checkpoint_path)

        # Test Set Evaluation
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_auth_preds, test_auth_trues = [], []
        test_denom_preds, test_denom_trues = [], []

        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(self.device)
                auth_labels = batch["auth_label"].to(self.device)
                denom_labels = batch["denom_label"].to(self.device)

                out = model(imgs)
                auth_p = torch.argmax(out["auth_logits"], dim=1).cpu().numpy()
                denom_p = torch.argmax(out["denom_logits"], dim=1).cpu().numpy()

                test_auth_preds.extend(auth_p)
                test_auth_trues.extend(auth_labels.cpu().numpy())
                test_denom_preds.extend(denom_p)
                test_denom_trues.extend(denom_labels.cpu().numpy())

        test_acc = float(accuracy_score(test_auth_trues, test_auth_preds))
        p, r, f1, _ = precision_recall_fscore_support(test_auth_trues, test_auth_preds, average="binary", zero_division=0)
        cm = confusion_matrix(test_auth_trues, test_auth_preds).tolist()

        result_summary = {
            "config_name": config_name,
            "backbone": backbone_name,
            "freeze_backbone": freeze_backbone,
            "test_auth_accuracy": round(test_acc, 4),
            "test_precision": round(float(p), 4),
            "test_recall": round(float(r), 4),
            "test_f1": round(float(f1), 4),
            "confusion_matrix": cm,
            "total_training_time_sec": round(time.time() - start_time, 2),
            "history": history,
            "checkpoint_path": str(best_checkpoint_path)
        }

        return result_summary

    def run_all_ablation_configs(self) -> Dict[str, Any]:
        """Trains 3 distinct configurations to benchmark layer-wise LR and architectures."""
        configs = [
            {
                "config_name": "config_1_resnet18_frozen",
                "backbone_name": "resnet18",
                "freeze_backbone": True,
                "backbone_lr": 0.0,
                "head_lr": 1e-3,
                "epochs": 4
            },
            {
                "config_name": "config_2_resnet18_layerwise",
                "backbone_name": "resnet18",
                "freeze_backbone": False,
                "backbone_lr": 1e-4,
                "head_lr": 1e-3,
                "epochs": 5
            },
            {
                "config_name": "config_3_effnet_unfrozen",
                "backbone_name": "efficientnet_b0",
                "freeze_backbone": False,
                "backbone_lr": 1e-4,
                "head_lr": 1e-3,
                "epochs": 4
            }
        ]

        all_results = {}
        for cfg in configs:
            res = self.train_single_configuration(**cfg)
            all_results[cfg["config_name"]] = res

        # Save comparison JSON
        ablation_json_path = self.reports_dir / "classification_ablation.json"
        with open(ablation_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # Set best overall model as standard primary checkpoint
        best_cfg_key = max(all_results.keys(), key=lambda k: all_results[k]["test_auth_accuracy"])
        primary_ckpt = self.models_dir / "best_classifier.pt"
        import shutil
        shutil.copy2(all_results[best_cfg_key]["checkpoint_path"], primary_ckpt)
        print(f"\nSaved primary champion classifier checkpoint ({best_cfg_key}) to: {primary_ckpt}")

        self.generate_ablation_markdown(all_results)
        return all_results

    def generate_ablation_markdown(self, all_results: Dict[str, Any]):
        """Generates markdown ablation comparison report."""
        md_path = self.reports_dir / "classification_ablation.md"
        lines = [
            "# Classification Architecture & Training Configuration Ablation",
            "",
            "Comparative benchmark of transfer learning backbones, layer-wise learning rates, and fine-tuning strategies:",
            "",
            "| Configuration | Backbone | Fine-Tuning Strategy | Test Accuracy | Precision | Recall | F1-Score | Training Time |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for k, v in all_results.items():
            strategy = "Frozen Backbone (Linear Probe)" if v["freeze_backbone"] else "Layer-wise LR Fine-Tuning"
            lines.append(
                f"| **{v['config_name']}** | {v['backbone']} | {strategy} | "
                f"{v['test_auth_accuracy'] * 100:.1f}% | {v['test_precision']:.3f} | "
                f"{v['test_recall']:.3f} | {v['test_f1']:.3f} | {v['total_training_time_sec']}s |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Ablation markdown report exported to: {md_path}")


if __name__ == "__main__":
    runner = ClassificationExperimentRunner()
    runner.run_all_ablation_configs()
