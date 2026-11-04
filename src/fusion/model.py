"""
Multi-Feature Learned Attention Fusion Module
==============================================
Aggregates per-feature localized evidence vectors (watermark, security thread,
latent image, number panel) using self-attention and MLP gating to predict:
1. Overall Note Authenticity Score [0.0 = Fake, 1.0 = Genuine]
2. Per-feature Attention Weights (Explainable dynamic contribution)
3. Denomination Prediction
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFeatureFusion(nn.Module):
    """Attention-gated fusion network for multi-security-feature aggregation."""

    FEATURE_NAMES = ["watermark", "security_thread", "latent_image", "number_panel"]
    NUM_FEATURES = len(FEATURE_NAMES)
    NUM_DENOMS = 7

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 1. Feature Projector for each security ROI
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Multi-Head Self-Attention across the 4 security features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Dynamic Attention Importance Scoring Head
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # 4. Fused Authenticity Classification Head
        self.auth_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # 5. Denomination Head
        self.denom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_DENOMS)
        )

    def forward(self, feature_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feature_tokens: Tensor of shape (batch_size, num_features, feature_dim)
            
        Returns:
            auth_logits: (batch_size, 2)
            auth_prob: (batch_size, 2) - Softmax probabilities
            denom_logits: (batch_size, 7)
            attention_weights: (batch_size, num_features) - Normalized importance per feature
        """
        # Project tokens
        projected = self.projector(feature_tokens)  # (B, N, hidden_dim)

        # Apply multi-head self-attention
        attn_out, _ = self.self_attention(projected, projected, projected)
        context = projected + attn_out  # Residual connection

        # Compute dynamic importance weights across features
        raw_importance = self.importance_scorer(context).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(raw_importance, dim=-1)  # (B, N)

        # Weighted pooling into a single banknote representation vector
        weighted_context = torch.bmm(attn_weights.unsqueeze(1), context).squeeze(1)  # (B, hidden_dim)

        # Classification
        auth_logits = self.auth_classifier(weighted_context)
        denom_logits = self.denom_classifier(weighted_context)

        return {
            "auth_logits": auth_logits,
            "auth_prob": F.softmax(auth_logits, dim=-1),
            "denom_logits": denom_logits,
            "denom_prob": F.softmax(denom_logits, dim=-1),
            "attention_weights": attn_weights,
            "fused_embedding": weighted_context
        }


if __name__ == "__main__":
    fusion = AttentionFeatureFusion(feature_dim=256, hidden_dim=128)
    dummy_input = torch.randn(4, 4, 256)
    out = fusion(dummy_input)
    print("AttentionFeatureFusion module verified!")
    print("  auth_prob shape:", out["auth_prob"].shape)
    print("  attention_weights shape:", out["attention_weights"].shape)
