"""
Channel-wise Attention Fusion Module (CAFM) for SentinelFuseNet.
Adaptively recalibrates SAR and optical cross-modal feature contributions.

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai
"""

import torch
import torch.nn as nn


class AttentionFusionModule(nn.Module):
    """
    Channel-wise Attention Fusion Module (CAFM).

    Concatenates optical and SAR feature vectors, computes per-channel
    attention weights via a two-layer MLP with Sigmoid activation, applies
    the weights via Hadamard product, and projects to a fused embedding.

    Architecture:
        F_cat   = concat(F_opt, F_sar)            B × 1024
        A       = Sigmoid(W2 · ReLU(W1 · F_cat))  B × 1024
        F_att   = F_cat ⊙ A                        B × 1024
        F_fused = Dropout(ReLU(BN(W3 · F_att)))   B × 512

    Args:
        in_features  (int): Concatenated feature dimension (default 1024)
        mid_features (int): Attention bottleneck dimension  (default 256)
        out_features (int): Fused output dimension          (default 512)
    """

    def __init__(self, in_features: int = 1024,
                 mid_features: int = 256,
                 out_features: int = 512):
        super(AttentionFusionModule, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(inplace=True),
            nn.Linear(mid_features, in_features),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, f_opt: torch.Tensor,
                f_sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_opt: Optical features  [B, 512]
            f_sar: SAR features      [B, 512]

        Returns:
            f_fused: Fused embedding [B, 512]
        """
        f_cat             = torch.cat([f_opt, f_sar], dim=1)  # B × 1024
        attention_weights = self.attention(f_cat)              # B × 1024
        f_att             = f_cat * attention_weights          # B × 1024
        f_fused           = self.fusion(f_att)                 # B × 512
        return f_fused
