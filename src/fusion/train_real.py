"""
Real Feature Embeddings Fusion Trainer
======================================
Extracts true latent feature representations from the trained ResNet18 classifier
across the real dataset splits (genuine + synthetic defect notes).
Trains the Multi-Head Attention Fusion network on these real representations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from typing import Tuple, Dict, List
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.classification.model import BanknoteFeatureClassifier
from src.fusion.model import AttentionFeatureFusion
from src.detection.detector import SecurityFeatureDetector


def extract_embeddings_for_split(
    df: pd.DataFrame,
    classifier: BanknoteFeatureClassifier,
    detector: SecurityFeatureDetector,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts 4 feature tokens (B, 4, 256) and labels for each banknote image."""
    classifier.eval()
    tokens_list = []
    auth_labels = []
    denom_labels = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    denom_map = {10: 0, 20: 1, 50: 2, 100: 3, 200: 4, 500: 5, 2000: 6}

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        det_res = detector.detect_and_crop(rgb)
        features = det_res["features"]

        item_tokens = []
        for feat in features[:4]:
            patch = feat["patch"]
            patch_resized = cv2.resize(patch, (224, 224))
            norm = (patch_resized.astype(np.float32) / 255.0 - mean) / std
            tensor_patch = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            with torch.no_grad():
                emb = classifier.extract_features(tensor_patch)  # (1, 256)
            item_tokens.append(emb)

        while len(item_tokens) < 4:
            item_tokens.append(torch.zeros(1, 256, device=device))

        four_tokens = torch.cat(item_tokens[:4], dim=0)  # (4, 256)
        tokens_list.append(four_tokens.unsqueeze(0))
        auth_labels.append(int(row["is_genuine"]))
        denom_labels.append(denom_map.get(int(row["denomination"]), 0))

    X = torch.cat(tokens_list, dim=0)  # (N, 4, 256)
    y_auth = torch.tensor(auth_labels, dtype=torch.long)
    y_denom = torch.tensor(denom_labels, dtype=torch.long)

    return X, y_auth, y_denom


def train_fusion_on_real_data(
    splits_dir: str = "data/processed/splits",
    save_path: str = "models/fusion/best_fusion.pt",
    device: str = "cpu"
):
    device = torch.device(device)
    print("Loading trained ResNet18 classifier and detector...")
    classifier = BanknoteFeatureClassifier(backbone_name="resnet18", pretrained=False)
    ckpt = torch.load("models/classification/best_classifier.pt", map_location=device)
    classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))
    classifier.to(device)

    detector = SecurityFeatureDetector(model_path="models/detection/best.pt")

    print("Extracting real feature embeddings from train split...")
    train_df = pd.read_csv(Path(splits_dir) / "train.csv")
    val_df = pd.read_csv(Path(splits_dir) / "val.csv")
    test_df = pd.read_csv(Path(splits_dir) / "test.csv")

    X_train, y_train_auth, y_train_denom = extract_embeddings_for_split(train_df, classifier, detector, device)
    X_val, y_val_auth, y_val_denom = extract_embeddings_for_split(val_df, classifier, detector, device)
    X_test, y_test_auth, y_test_denom = extract_embeddings_for_split(test_df, classifier, detector, device)

    print(f"Extracted: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    train_ds = TensorDataset(X_train, y_train_auth, y_train_denom)
    val_ds = TensorDataset(X_val, y_val_auth, y_val_denom)
    test_ds = TensorDataset(X_test, y_test_auth, y_test_denom)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = AttentionFeatureFusion(feature_dim=256, hidden_dim=128)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit_auth = nn.CrossEntropyLoss()
    crit_denom = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, 25):
        model.train()
        for tokens, auth_y, denom_y in train_loader:
            optimizer.zero_grad()
            out = model(tokens)
            loss = crit_auth(out["auth_logits"], auth_y) + 0.3 * crit_denom(out["denom_logits"], denom_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for tokens, auth_y, _ in val_loader:
                out = model(tokens)
                preds = torch.argmax(out["auth_logits"], dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(auth_y.cpu().numpy())

        acc = float(accuracy_score(val_trues, val_preds))
        if acc > best_val_acc or epoch == 1:
            best_val_acc = acc
            torch.save(model.state_dict(), save_path)

    # Test
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for tokens, auth_y, _ in test_loader:
            out = model(tokens)
            preds = torch.argmax(out["auth_logits"], dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_trues.extend(auth_y.numpy())

    test_acc = float(accuracy_score(test_trues, test_preds))
    p, r, f1, _ = precision_recall_fscore_support(test_trues, test_preds, average="binary", zero_division=0)
    print(f"Fusion Model trained on REAL representations saved to {save_path}!")
    print(f"Test Accuracy: {test_acc*100:.1f}%, F1: {f1:.3f}")


if __name__ == "__main__":
    train_fusion_on_real_data()
