"""
Ablation Study — Table 8 of the paper.

Six variants (V1–V6) progressively build up the full SentinelFuseNet:

    V1  ResNet-18 Optical only (single encoder, classification head)
    V2  ResNet-18 SAR only     (single encoder, classification head)
    V3  Dual-encoder, simple concatenation fusion (no CAFM)
    V4  Dual-encoder + CAFM, classification head only
    V5  Dual-encoder + CAFM, classification + detection heads
    V6  Dual-encoder + CAFM, all three heads  ← Full SentinelFuseNet

Expected WTP values from the paper (Table 8):
    V1=39.04%  V2=2.99%  V3=67.04%  V4=52.12%  V5=58.85%  V6=73.72%

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai

Usage:
    python scripts/ablation.py \
        --preprocessed /path/to/preprocessed \
        --output_path  results/ablation
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.losses import MultiTaskLoss
from utils.metrics import (boundary_f1_score, detection_iou,
                           compute_miou, multi_task_consistency_score,
                           weighted_task_performance)

SEED       = 42
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-4
STEP_SIZE  = 10
GAMMA      = 0.5

# ─────────────────────────────────────────────
# Variant model builders
# ─────────────────────────────────────────────

import torchvision.models as tv_models


class _Encoder(nn.Module):
    """Shared ResNet-18 encoder: extracts 512-d global pool feature."""
    def __init__(self, pretrained=True):
        super().__init__()
        base = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained
            else None)
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)   # B × 512


class _ClsHead(nn.Module):
    def __init__(self, in_features=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class _DetHead(nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        self.fc = nn.Linear(in_features, 5)

    def forward(self, x):
        return self.fc(x)


class _SegHead(nn.Module):
    def __init__(self, in_features=512, img_size=128):
        super().__init__()
        self.fc = nn.Linear(in_features, img_size * img_size)
        self.img_size = img_size

    def forward(self, x):
        b = x.size(0)
        return self.fc(x).view(b, 1, self.img_size, self.img_size)


class _CAFM(nn.Module):
    """Lightweight CAFM (same as attention.py)."""
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(True),
            nn.Linear(256, 1024), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512),
            nn.ReLU(True), nn.Dropout(0.3))

    def forward(self, f_opt, f_sar):
        f_cat = torch.cat([f_opt, f_sar], dim=1)
        return self.fusion(f_cat * self.attention(f_cat))


def build_variant(variant: str, num_classes: int = 10,
                  img_size: int = 128, pretrained: bool = True):
    """
    Returns (model_dict, use_opt, use_sar, use_det, use_seg).
    model_dict keys: 'opt_enc', 'sar_enc', 'cafm', 'cls', 'det', 'seg'
    (absent keys are None).
    """
    enc_opt = _Encoder(pretrained) if variant != 'V2' else None
    enc_sar = _Encoder(pretrained) if variant != 'V1' else None

    fused_dim = 512   # single encoder or concat-then-CAFM
    if variant == 'V3':
        # Simple concatenation → project to 512
        concat_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True))
    else:
        concat_proj = None

    cafm = _CAFM() if variant in ('V4', 'V5', 'V6') else None

    cls_head = _ClsHead(fused_dim, num_classes)
    det_head = _DetHead(fused_dim)   if variant in ('V5', 'V6') else None
    seg_head = _SegHead(fused_dim,
                        img_size)    if variant == 'V6'         else None

    modules = {}
    if enc_opt:     modules['enc_opt']     = enc_opt
    if enc_sar:     modules['enc_sar']     = enc_sar
    if concat_proj: modules['concat_proj'] = concat_proj
    if cafm:        modules['cafm']        = cafm
    modules['cls'] = cls_head
    if det_head:    modules['det']         = det_head
    if seg_head:    modules['seg']         = seg_head

    return modules


class VariantModel(nn.Module):
    """Wraps a variant dict and provides a unified forward()."""
    def __init__(self, modules_dict, variant):
        super().__init__()
        self.variant = variant
        for name, mod in modules_dict.items():
            setattr(self, name, mod)

    def forward(self, x_opt, x_sar):
        if self.variant == 'V1':
            f = self.enc_opt(x_opt)
        elif self.variant == 'V2':
            f = self.enc_sar(x_sar)
        elif self.variant == 'V3':
            fo = self.enc_opt(x_opt)
            fs = self.enc_sar(x_sar)
            f  = self.concat_proj(torch.cat([fo, fs], dim=1))
        else:  # V4, V5, V6
            fo = self.enc_opt(x_opt)
            fs = self.enc_sar(x_sar)
            f  = self.cafm(fo, fs)

        pred_cls = self.cls(f)
        pred_det = self.det(f) if hasattr(self, 'det') else None
        pred_seg = self.seg(f) if hasattr(self, 'seg') else None
        return pred_cls, pred_det, pred_seg


# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────

def load_splits(preprocessed):
    def _load(split):
        opt   = np.load(os.path.join(preprocessed,
                                     f'{split}_eurosat_images.npy'))
        lbl   = np.load(os.path.join(preprocessed,
                                     f'{split}_eurosat_labels.npy'))
        s1    = np.load(os.path.join(preprocessed,
                                     f'{split}_s1_images.npy'))
        bbox  = np.load(os.path.join(preprocessed,
                                     f'{split}_s1_bboxes.npy'))
        masks = np.load(os.path.join(preprocessed,
                                     f'{split}_s1_masks.npy'))
        return opt, lbl, s1, bbox, masks

    return _load('train'), _load('val'), _load('test')


def to_loader(opt, lbl, s1, bbox, masks, batch_size, shuffle):
    opt_t   = torch.FloatTensor(opt).permute(0, 3, 1, 2)
    lbl_t   = torch.LongTensor(lbl)
    s1_t    = torch.FloatTensor(s1).permute(0, 3, 1, 2)
    bbox_t  = torch.FloatTensor(bbox)
    masks_t = torch.FloatTensor(masks).permute(0, 3, 1, 2)
    ds = TensorDataset(opt_t, lbl_t, s1_t, bbox_t, masks_t)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)


# ─────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────

def train_variant(model, loader, device, variant):
    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler    = GradScaler()

    model.train()
    for _ in range(EPOCHS):
        for x_opt, y_cls, x_sar, y_det, y_seg in loader:
            x_opt = x_opt.to(device); x_sar = x_sar.to(device)
            y_cls = y_cls.to(device); y_det = y_det.to(device)
            y_seg = y_seg.to(device)

            optimizer.zero_grad()
            with autocast():
                pc, pd, ps = model(x_opt, x_sar)
                # Only compute losses for heads this variant has
                lc = nn.CrossEntropyLoss()(pc, y_cls)
                ld = nn.MSELoss()(pd, y_det)     if pd is not None \
                     else torch.tensor(0., device=device)
                ls = nn.BCEWithLogitsLoss()(ps, y_seg) if ps is not None \
                     else torch.tensor(0., device=device)
                loss = lc + ld + ls
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()


def evaluate_variant(model, loader, device, variant):
    model.eval()
    img_size = 128
    cls_true, cls_pred = [], []
    det_ious, seg_mious, seg_bf1s = [], [], []

    with torch.no_grad():
        for x_opt, y_cls, x_sar, y_det, y_seg in loader:
            x_opt = x_opt.to(device); x_sar = x_sar.to(device)
            pc, pd, ps = model(x_opt, x_sar)

            _, pred_c = pc.max(1)
            cls_true.extend(y_cls.numpy())
            cls_pred.extend(pred_c.cpu().numpy())

            if pd is not None:
                for dp, dt in zip(pd.cpu().numpy(), y_det.numpy()):
                    det_ious.append(detection_iou(dp[1:], dt[1:]))

            if ps is not None:
                seg_bin = (torch.sigmoid(ps) > 0.5).float().cpu().numpy()
                for sb, st in zip(seg_bin, y_seg.numpy()):
                    seg_mious.append(
                        compute_miou(sb[0].astype(int),
                                     st[0].astype(int)))
                    seg_bf1s.append(
                        boundary_f1_score(sb[0].astype(bool),
                                          st[0].astype(bool)))

    acc     = accuracy_score(cls_true, cls_pred) * 100
    det_iou = np.mean(det_ious) * 100 if det_ious else 0.0
    miou    = np.mean(seg_mious) * 100 if seg_mious else 0.0

    wtp = weighted_task_performance(
        acc / 100, det_iou / 100, miou / 100) * 100

    return {
        'Accuracy':      round(acc, 2),
        'Detection_IoU': round(det_iou, 2),
        'mIoU':          round(miou, 2),
        'WTP':           round(wtp, 2),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

VARIANT_LABELS = {
    'V1': 'ResNet-18 Optical only',
    'V2': 'ResNet-18 SAR only',
    'V3': 'Dual-encoder, concat fusion',
    'V4': 'Dual-encoder + CAFM, cls only',
    'V5': 'Dual-encoder + CAFM, cls+det',
    'V6': 'Full SentinelFuseNet (cls+det+seg)',
}

PAPER_WTP = {'V1': 39.04, 'V2': 2.99, 'V3': 67.04,
             'V4': 52.12, 'V5': 58.85, 'V6': 73.72}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("Loading data splits...")
    train_data, _, test_data = load_splits(args.preprocessed)
    tr_loader  = to_loader(*train_data, BATCH_SIZE, shuffle=True)
    test_loader = to_loader(*test_data, BATCH_SIZE, shuffle=False)

    os.makedirs(args.output_path, exist_ok=True)
    all_results = {}

    for v in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        print(f"\n{'='*55}")
        print(f"Training {v}: {VARIANT_LABELS[v]}")
        print('='*55)

        modules = build_variant(v)
        model   = VariantModel(modules, v).to(device)
        train_variant(model, tr_loader, device, v)
        metrics = evaluate_variant(model, test_loader, device, v)
        metrics['variant'] = v
        metrics['description'] = VARIANT_LABELS[v]
        metrics['paper_WTP'] = PAPER_WTP[v]
        all_results[v] = metrics

        print(f"  Accuracy    : {metrics['Accuracy']:.2f}%")
        print(f"  Detection IoU: {metrics['Detection_IoU']:.2f}%")
        print(f"  mIoU        : {metrics['mIoU']:.2f}%")
        print(f"  WTP         : {metrics['WTP']:.2f}%  "
              f"(paper: {PAPER_WTP[v]:.2f}%)")

    print(f"\n{'='*55}")
    print("Ablation Summary (Table 8)")
    print(f"{'Variant':<6} {'Description':<38} {'WTP':>7}  {'Paper':>7}")
    print('-'*60)
    for v, res in all_results.items():
        print(f"{v:<6} {res['description']:<38} "
              f"{res['WTP']:>6.2f}%  {res['paper_WTP']:>6.2f}%")

    out_file = os.path.join(args.output_path, 'ablation_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ablation study (V1–V6) for SentinelFuseNet')
    parser.add_argument('--preprocessed', required=True)
    parser.add_argument('--output_path',  default='results/ablation')
    args = parser.parse_args()
    main(args)
