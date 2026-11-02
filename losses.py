"""
Multi-task loss functions for SentinelFuseNet.

L_total = λ1·L_cls + λ2·L_det + λ3·L_seg
with λ1 = λ2 = λ3 = 1.0 (equal task weighting).

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for SentinelFuseNet.

    Sums three task losses with equal scalar weights (λ=1.0):
        L_cls — CrossEntropyLoss  (10-class land-use classification)
        L_det — MSELoss           (YOLO bounding-box regression)
        L_seg — BCEWithLogitsLoss (binary airstrip segmentation)

    Args:
        lambda_cls (float): Weight for classification loss (default 1.0)
        lambda_det (float): Weight for detection loss      (default 1.0)
        lambda_seg (float): Weight for segmentation loss   (default 1.0)
    """

    def __init__(self, lambda_cls: float = 1.0,
                 lambda_det: float = 1.0,
                 lambda_seg: float = 1.0):
        super(MultiTaskLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_det = lambda_det
        self.lambda_seg = lambda_seg

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_det = nn.MSELoss()
        self.criterion_seg = nn.BCEWithLogitsLoss()

    def forward(self,
                pred_cls: torch.Tensor, y_cls: torch.Tensor,
                pred_det: torch.Tensor, y_det: torch.Tensor,
                pred_seg: torch.Tensor, y_seg: torch.Tensor):
        """
        Args:
            pred_cls: Classification logits  [B, 10]
            y_cls:    Class labels           [B]
            pred_det: Detection predictions  [B, 5]
            y_det:    YOLO bbox targets      [B, 5]
            pred_seg: Segmentation logits    [B, 1, 128, 128]
            y_seg:    Binary mask targets    [B, 1, 128, 128]

        Returns:
            total_loss: Weighted sum of the three task losses
            loss_cls:   Classification loss component
            loss_det:   Detection loss component
            loss_seg:   Segmentation loss component
        """
        loss_cls = self.criterion_cls(pred_cls, y_cls)
        loss_det = self.criterion_det(pred_det, y_det)
        loss_seg = self.criterion_seg(pred_seg, y_seg)

        total_loss = (self.lambda_cls * loss_cls +
                      self.lambda_det * loss_det +
                      self.lambda_seg * loss_seg)

        return total_loss, loss_cls, loss_det, loss_seg
