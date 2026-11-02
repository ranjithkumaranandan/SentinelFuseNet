"""
Baseline Comparison — Table 7 of the paper.

Five baseline models evaluated under the same train/test split:

    B1  ResNet-18 Optical (classification only)  WTP = 39.04%
    B2  ResNet-18 SAR     (classification only)  WTP =  2.99%
    B3  U-Net             (segmentation only)    WTP = 44.61%
    B4  YOLOv5            (detection only)       WTP = 37.46%
    B5  Single-Task MTL   (no fusion)            WTP = 62.58%

WTP = 0.4 × Accuracy + 0.3 × Detection-IoU + 0.3 × mIoU
(Tasks not supported by a baseline contribute 0 to WTP.)

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai

Usage:
    python scripts/baseline.py \
        --preprocessed /path/to/preprocessed \
        --output_path  results/baselines
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
from utils.metrics import (detection_iou, compute_miou, boundary_f1_score,
                           weighted_task_performance)

SEED       = 42
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-4
STEP_SIZE  = 10
GAMMA      = 0.5

# ─────────────────────────────────────────────
# Baseline model definitions
# ─────────────────────────────────────────────

import torchvision.models as tv_models


class ResNet18Cls(nn.Module):
    """ResNet-18 for 10-class land-use classification (B1 / B2)."""
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        base = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained
            else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.classifier(feat)


class UNetBaseline(nn.Module):
    """
    Simplified UNet for binary segmentation (B3 — U-Net baseline).
    Encoder: 4 down-sampling blocks.
    Decoder: 4 up-sampling blocks with skip connections.
    Input / output: 128 × 128.
    """
    def __init__(self, in_ch=3):
        super().__init__()

        def _block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc),
                nn.ReLU(True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc),
                nn.ReLU(True))

        self.enc1 = _block(in_ch, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _block(64,  128);   self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _block(128, 256);   self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _block(256, 512);   self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = _block(512, 1024)

        self.up4   = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4  = _block(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3  = _block(512,  256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2  = _block(256,  128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1  = _block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x);   e1p = self.pool1(e1)
        e2 = self.enc2(e1p); e2p = self.pool2(e2)
        e3 = self.enc3(e2p); e3p = self.pool3(e3)
        e4 = self.enc4(e3p); e4p = self.pool4(e4)
        b  = self.bottleneck(e4p)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


class YOLOBaseline(nn.Module):
    """
    Lightweight YOLO-style detector (B4).
    ResNet-18 backbone → FC head predicting (class, xc, yc, w, h).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        base = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained
            else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head     = nn.Linear(512, 5)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


class SingleTaskMTL(nn.Module):
    """
    Single-Task MTL baseline (B5): three separate ResNet-18 encoders,
    one per task, without any fusion.
    """
    def __init__(self, num_classes=10, img_size=128, pretrained=True):
        super().__init__()

        def _enc():
            base = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.DEFAULT if pretrained
                else None)
            return nn.Sequential(*list(base.children())[:-1])

        self.enc_cls = _enc()
        self.enc_det = _enc()
        self.enc_seg = _enc()

        self.cls_head = nn.Linear(512, num_classes)
        self.det_head = nn.Linear(512, 5)
        self.seg_head = nn.Sequential(
            nn.Linear(512, img_size * img_size))
        self.img_size = img_size

    def forward(self, x):
        b = x.size(0)
        pc = self.cls_head(self.enc_cls(x).flatten(1))
        pd = self.det_head(self.enc_det(x).flatten(1))
        ps = self.seg_head(
            self.enc_seg(x).flatten(1)
        ).view(b, 1, self.img_size, self.img_size)
        return pc, pd, ps


# ─────────────────────────────────────────────
# Helpers
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
    return _load('train'), _load('test')


def to_loader(opt, lbl, s1, bbox, masks, batch_size, shuffle):
    opt_t   = torch.FloatTensor(opt).permute(0, 3, 1, 2)
    lbl_t   = torch.LongTensor(lbl)
    s1_t    = torch.FloatTensor(s1).permute(0, 3, 1, 2)
    bbox_t  = torch.FloatTensor(bbox)
    masks_t = torch.FloatTensor(masks).permute(0, 3, 1, 2)
    ds = TensorDataset(opt_t, lbl_t, s1_t, bbox_t, masks_t)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)


def _step(model, optimizer, scaler, criterion, batch, device, mode):
    x_opt, y_cls, x_sar, y_det, y_seg = [t.to(device) for t in batch]
    optimizer.zero_grad()
    with autocast():
        if mode == 'cls_opt':
            out  = model(x_opt)
            loss = criterion(out, y_cls)
        elif mode == 'cls_sar':
            out  = model(x_sar)
            loss = criterion(out, y_cls)
        elif mode == 'seg':
            out  = model(x_sar)
            loss = nn.BCEWithLogitsLoss()(out, y_seg)
        elif mode == 'det':
            out  = model(x_sar)
            loss = nn.MSELoss()(out, y_det)
        elif mode == 'mtl':
            pc, pd, ps = model(x_opt)
            loss = (nn.CrossEntropyLoss()(pc, y_cls) +
                    nn.MSELoss()(pd, y_det) +
                    nn.BCEWithLogitsLoss()(ps, y_seg))
    scaler.scale(loss).backward()
    scaler.step(optimizer); scaler.update()


def train_model(model, loader, device, mode):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler    = GradScaler()
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            _step(model, optimizer, scaler, criterion,
                  batch, device, mode)
        scheduler.step()


# ─────────────────────────────────────────────
# Evaluation per baseline
# ─────────────────────────────────────────────

@torch.no_grad()
def eval_cls(model, loader, device, use_sar=False):
    model.eval()
    true_l, pred_l = [], []
    for x_opt, y_cls, x_sar, *_ in loader:
        x = x_sar.to(device) if use_sar else x_opt.to(device)
        _, p = model(x).max(1)
        true_l.extend(y_cls.numpy()); pred_l.extend(p.cpu().numpy())
    acc = accuracy_score(true_l, pred_l) * 100
    return {'Accuracy': round(acc, 2),
            'Detection_IoU': 0.0, 'mIoU': 0.0,
            'WTP': round(weighted_task_performance(acc/100, 0, 0)*100, 2)}


@torch.no_grad()
def eval_seg(model, loader, device):
    model.eval()
    mious, bf1s = [], []
    for _, _, x_sar, _, y_seg in loader:
        x_sar = x_sar.to(device)
        out   = model(x_sar)
        seg_b = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
        for sb, st in zip(seg_b, y_seg.numpy()):
            mious.append(compute_miou(sb[0].astype(int),
                                      st[0].astype(int)))
            bf1s.append(boundary_f1_score(sb[0].astype(bool),
                                          st[0].astype(bool)))
    miou = np.mean(mious) * 100
    return {'Accuracy': 0.0, 'Detection_IoU': 0.0,
            'mIoU':    round(miou, 2),
            'BF1':     round(np.mean(bf1s)*100, 2),
            'WTP': round(weighted_task_performance(0, 0, miou/100)*100, 2)}


@torch.no_grad()
def eval_det(model, loader, device):
    model.eval()
    det_ious = []
    for _, _, x_sar, y_det, _ in loader:
        x_sar = x_sar.to(device)
        out   = model(x_sar).cpu().numpy()
        for dp, dt in zip(out, y_det.numpy()):
            det_ious.append(detection_iou(dp[1:], dt[1:]))
    diou = np.mean(det_ious) * 100
    return {'Accuracy': 0.0,
            'Detection_IoU': round(diou, 2), 'mIoU': 0.0,
            'WTP': round(
                weighted_task_performance(0, diou/100, 0)*100, 2)}


@torch.no_grad()
def eval_mtl(model, loader, device):
    model.eval()
    cls_t, cls_p = [], []
    det_ious, seg_mious = [], []
    for x_opt, y_cls, _, y_det, y_seg in loader:
        x_opt = x_opt.to(device)
        pc, pd, ps = model(x_opt)
        _, p = pc.max(1)
        cls_t.extend(y_cls.numpy()); cls_p.extend(p.cpu().numpy())
        for dp, dt in zip(pd.cpu().numpy(), y_det.numpy()):
            det_ious.append(detection_iou(dp[1:], dt[1:]))
        sb = (torch.sigmoid(ps) > 0.5).float().cpu().numpy()
        for s, st in zip(sb, y_seg.numpy()):
            seg_mious.append(compute_miou(s[0].astype(int),
                                          st[0].astype(int)))
    acc  = accuracy_score(cls_t, cls_p) * 100
    diou = np.mean(det_ious)  * 100
    miou = np.mean(seg_mious) * 100
    wtp  = weighted_task_performance(acc/100, diou/100, miou/100) * 100
    return {'Accuracy': round(acc, 2),
            'Detection_IoU': round(diou, 2),
            'mIoU': round(miou, 2),
            'WTP':  round(wtp, 2)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

PAPER_WTP = {'B1': 39.04, 'B2': 2.99, 'B3': 44.61,
             'B4': 37.46, 'B5': 62.58}

BASELINE_LABELS = {
    'B1': 'ResNet-18 Optical',
    'B2': 'ResNet-18 SAR',
    'B3': 'U-Net (SAR segmentation)',
    'B4': 'YOLO Detector (SAR)',
    'B5': 'Single-Task MTL (no fusion)',
}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("Loading data splits...")
    train_data, test_data = load_splits(args.preprocessed)
    tr_loader   = to_loader(*train_data, BATCH_SIZE, shuffle=True)
    test_loader = to_loader(*test_data,  BATCH_SIZE, shuffle=False)

    os.makedirs(args.output_path, exist_ok=True)
    all_results = {}

    configs = [
        ('B1', ResNet18Cls(pretrained=True),   'cls_opt', eval_cls,
         {'use_sar': False}),
        ('B2', ResNet18Cls(pretrained=True),   'cls_sar', eval_cls,
         {'use_sar': True}),
        ('B3', UNetBaseline(),                 'seg',     eval_seg, {}),
        ('B4', YOLOBaseline(pretrained=True),  'det',     eval_det, {}),
        ('B5', SingleTaskMTL(pretrained=True), 'mtl',     eval_mtl, {}),
    ]

    for bname, model, mode, eval_fn, eval_kw in configs:
        print(f"\n{'='*55}")
        print(f"Training {bname}: {BASELINE_LABELS[bname]}")
        print('='*55)
        model = model.to(device)
        train_model(model, tr_loader, device, mode)
        metrics = eval_fn(model, test_loader, device, **eval_kw)
        metrics['baseline']    = bname
        metrics['description'] = BASELINE_LABELS[bname]
        metrics['paper_WTP']   = PAPER_WTP[bname]
        all_results[bname]     = metrics
        print(f"  WTP = {metrics['WTP']:.2f}%  "
              f"(paper: {PAPER_WTP[bname]:.2f}%)")

    print(f"\n{'='*55}")
    print("Baseline Summary (Table 7)")
    print(f"{'ID':<4} {'Description':<38} {'WTP':>7}  {'Paper':>7}")
    print('-'*60)
    for b, res in all_results.items():
        print(f"{b:<4} {res['description']:<38} "
              f"{res['WTP']:>6.2f}%  {res['paper_WTP']:>6.2f}%")

    out_file = os.path.join(args.output_path, 'baseline_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline evaluation (B1–B5) for SentinelFuseNet')
    parser.add_argument('--preprocessed', required=True)
    parser.add_argument('--output_path',  default='results/baselines')
    args = parser.parse_args()
    main(args)
