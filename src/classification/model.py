"""
Feature Classifier Architecture (ResNet18 & EfficientNet-B0)
=============================================================
Deep multi-task network verifying authenticity (Genuine vs Suspicious)
and predicting denomination across localized security patches or whole banknotes.

Key architectural features:
- Pretrained ImageNet representations
- Layer-wise learning rate control
- Configurable freezing/unfreezing schedules
- Batch normalization & adaptive dropout regularization
- Dual task heads: Authenticity (Binary) + Denomination (7-class)
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
import timm


class BanknoteFeatureClassifier(nn.Module):
    """Deep feature classifier with dual authenticity and denomination heads."""

    DENOMINATIONS = [10, 20, 50, 100, 200, 500, 2000]
    NUM_DENOMS = len(DENOMINATIONS)

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.35,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.dropout_rate = dropout_rate

        # 1. Feature Extractor Backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            in_features = resnet.fc.in_features
            # Remove original FC
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.target_cam_layer = self.backbone[-2]  # layer4 for Grad-CAM
        elif backbone_name == "efficientnet_b0":
            # Using timm or torchvision
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            effnet = models.efficientnet_b0(weights=weights)
            in_features = effnet.classifier[1].in_features
            self.backbone = effnet.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.target_cam_layer = self.backbone[-1]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.in_features = in_features

        # 2. Shared Latent Embedding Projection
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

        # 3. Authenticity Head (Binary: 0=Counterfeit/Synthetic Fake, 1=Genuine)
        self.auth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(64, 2)
        )

        # 4. Denomination Head (7 Classes)
        self.denom_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(64, self.NUM_DENOMS)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial pooled feature vectors from backbone."""
        if self.backbone_name == "resnet18":
            feats = self.backbone(x)
        else:
            feats = self.backbone(x)
            feats = self.pool(feats)
        return self.shared_fc(feats)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding authenticity logits, denomination logits,
        and latent embedding vectors for fusion.
        """
        latent = self.extract_features(x)
        auth_logits = self.auth_head(latent)
        denom_logits = self.denom_head(latent)

        return {
            "auth_logits": auth_logits,
            "denom_logits": denom_logits,
            "embedding": latent
        }

    def set_backbone_trainable(self, trainable: bool = True):
        """Freezes or unfreezes backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable
        print(f"[{self.backbone_name}] Backbone set to trainable={trainable}")

    def get_parameter_groups(self, backbone_lr: float, head_lr: float):
        """Constructs parameter groups for layer-wise learning rates."""
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.shared_fc.parameters()) +
            list(self.auth_head.parameters()) +
            list(self.denom_head.parameters())
        )

        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "heads"}
        ]


if __name__ == "__main__":
    net = BanknoteFeatureClassifier(backbone_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print("ResNet18 Model Verified. Output shapes:")
    print("  auth_logits:", out["auth_logits"].shape)
    print("  denom_logits:", out["denom_logits"].shape)
    print("  embedding:", out["embedding"].shape)
