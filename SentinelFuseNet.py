"""
SentinelFuseNet: Dual-Encoder Attention Fusion Network
for Simultaneous Detection, Segmentation and Classification
of Heterogeneous Sentinel Imagery.

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AttentionFusionModule(nn.Module):
    """
    Channel-wise Attention Fusion Module.
    Adaptively recalibrates cross-modal feature contributions
    from optical and SAR encoder streams.
    """
    def __init__(self, in_features=1024, mid_features=256, out_features=512):
        super(AttentionFusionModule, self).__init__()

        # Channel-wise attention
        self.attention = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(inplace=True),
            nn.Linear(mid_features, in_features),
            nn.Sigmoid()
        )

        # Projection to fused representation
        self.fusion = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, f_opt, f_sar):
        # Concatenate optical and SAR features
        f_cat = torch.cat([f_opt, f_sar], dim=1)  # B x 1024

        # Compute attention weights
        attention_weights = self.attention(f_cat)   # B x 1024

        # Apply attention via Hadamard product
        f_att = f_cat * attention_weights           # B x 1024

        # Project to fused representation
        f_fused = self.fusion(f_att)               # B x 512

        return f_fused


class ClassificationHead(nn.Module):
    """
    Classification head for 10-class EuroSAT land-use classification.
    """
    def __init__(self, in_features=512, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class DetectionHead(nn.Module):
    """
    Detection head for YOLO-format bounding box regression.
    Output: (class, x_c, y_c, w, h) — 5-dimensional vector.
    """
    def __init__(self, in_features=512, out_features=5):
        super(DetectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        return self.head(x)


class SegmentationHead(nn.Module):
    """
    Segmentation head with 7 transposed convolutional layers.
    Progressively upsamples from 1x1 to 128x128 binary mask.
    """
    def __init__(self, in_channels=512):
        super(SegmentationHead, self).__init__()
        self.decoder = nn.Sequential(
            # 1x1 -> 2x2
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 2x2 -> 4x4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # Reshape for spatial upsampling
        x = x.view(x.size(0), 512, 1, 1)
        return self.decoder(x)


class SentinelFuseNet(nn.Module):
    """
    SentinelFuseNet: Unified Dual-Encoder Attention Fusion Network
    for simultaneous classification, detection, and segmentation
    of heterogeneous Sentinel-1 SAR and Sentinel-2 optical imagery.

    Architecture:
        - Optical Encoder: ResNet-18 (ImageNet pretrained)
        - SAR Encoder:     ResNet-18 (ImageNet pretrained)
        - Fusion Module:   Channel-wise Attention Fusion
        - Task Heads:      Classification + Detection + Segmentation

    Total Parameters: 26,457,280
    """

    def __init__(self, num_classes=10, pretrained=True):
        super(SentinelFuseNet, self).__init__()

        # ── Optical Encoder (Sentinel-2) ──
        optical_backbone = models.resnet18(pretrained=pretrained)
        self.optical_encoder = nn.Sequential(
            *list(optical_backbone.children())[:-1]  # Remove FC layer
        )

        # ── SAR Encoder (Sentinel-1) ──
        sar_backbone = models.resnet18(pretrained=pretrained)
        self.sar_encoder = nn.Sequential(
            *list(sar_backbone.children())[:-1]      # Remove FC layer
        )

        # ── Channel-wise Attention Fusion Module ──
        self.fusion = AttentionFusionModule(
            in_features=1024,
            mid_features=256,
            out_features=512
        )

        # ── Task-Specific Heads ──
        self.cls_head = ClassificationHead(
            in_features=512,
            num_classes=num_classes
        )
        self.det_head = DetectionHead(
            in_features=512,
            out_features=5
        )
        self.seg_head = SegmentationHead(
            in_channels=512
        )

    def forward(self, x_opt, x_sar):
        """
        Forward pass through SentinelFuseNet.

        Args:
            x_opt: Optical image batch  [B, 3, 128, 128]
            x_sar: SAR image batch      [B, 3, 128, 128]

        Returns:
            y_cls: Classification logits  [B, 10]
            y_det: Detection predictions  [B, 5]
            y_seg: Segmentation mask      [B, 1, 128, 128]
        """
        # ── Step 1: Dual Encoder Feature Extraction ──
        f_opt = self.optical_encoder(x_opt)   # B x 512 x 1 x 1
        f_sar = self.sar_encoder(x_sar)       # B x 512 x 1 x 1

        # Flatten spatial dimensions
        f_opt = f_opt.view(f_opt.size(0), -1) # B x 512
        f_sar = f_sar.view(f_sar.size(0), -1) # B x 512

        # ── Step 2: Cross-Sensor Attention Fusion ──
        f_fused = self.fusion(f_opt, f_sar)   # B x 512

        # ── Step 3: Task-Specific Prediction ──
        y_cls = self.cls_head(f_fused)        # B x 10
        y_det = self.det_head(f_fused)        # B x 5
        y_seg = self.seg_head(f_fused)        # B x 1 x 128 x 128

        return y_cls, y_det, y_seg


def get_model(num_classes=10, pretrained=True):
    """Initialize and return SentinelFuseNet model."""
    model = SentinelFuseNet(
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick model test
    model = get_model()
    print(f'SentinelFuseNet initialized')
    print(f'Total parameters: {count_parameters(model):,}')

    # Test forward pass
    x_opt = torch.randn(2, 3, 128, 128)
    x_sar = torch.randn(2, 3, 128, 128)
    y_cls, y_det, y_seg = model(x_opt, x_sar)
    print(f'Classification output: {y_cls.shape}')
    print(f'Detection output:      {y_det.shape}')
    print(f'Segmentation output:   {y_seg.shape}')
