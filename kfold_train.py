"""
5-Fold Stratified Cross-Validation on EuroSAT (Classification Head).

Matches Section 6.4 of the paper:
    "5-fold stratified cross-validation on EuroSAT yielded
     98.32% ± 0.17% classification accuracy"

Only classification accuracy is reported per fold; the model is
trained end-to-end (all three task heads) on each fold's train split
so the backbone learns from the full multi-task objective.

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai

Usage:
    python scripts/kfold_train.py \
        --eurosat_path  /path/to/EuroSAT/2750 \
        --s1aad_path    /path/to/S1-AAD \
        --preprocessed  /path/to/preprocessed \
        --output_path   results/kfold
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SentinelFuseNet import get_model
from utils.losses import MultiTaskLoss

SEED        = 42
N_FOLDS     = 5
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-4
STEP_SIZE   = 10
GAMMA       = 0.5


def load_full_eurosat(preprocessed: str):
    """
    Re-assemble the full EuroSAT train+val+test from saved .npy files
    so the k-fold uses all available EuroSAT data.
    """
    splits = ['train', 'val', 'test']
    imgs, lbls = [], []
    for sp in splits:
        imgs.append(np.load(os.path.join(preprocessed,
                                         f'{sp}_eurosat_images.npy')))
        lbls.append(np.load(os.path.join(preprocessed,
                                         f'{sp}_eurosat_labels.npy')))
    return np.concatenate(imgs), np.concatenate(lbls)


def load_s1_train(preprocessed: str):
    """Load the S1-AAD training split for the detection/seg heads."""
    imgs  = np.load(os.path.join(preprocessed, 'train_s1_images.npy'))
    bbox  = np.load(os.path.join(preprocessed, 'train_s1_bboxes.npy'))
    masks = np.load(os.path.join(preprocessed, 'train_s1_masks.npy'))
    return imgs, bbox, masks


def make_s1_loader(s1_imgs, s1_bbox, s1_masks, batch_size):
    """Infinite S1 iterator for synchronised multi-task batching."""
    s1_t    = torch.FloatTensor(s1_imgs).permute(0, 3, 1, 2)
    bbox_t  = torch.FloatTensor(s1_bbox)
    masks_t = torch.FloatTensor(s1_masks).permute(0, 3, 1, 2)
    ds = TensorDataset(s1_t, bbox_t, masks_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      drop_last=True)


def train_fold(fold_idx, train_idx, val_idx,
               euro_imgs, euro_lbls,
               s1_imgs, s1_bbox, s1_masks,
               device):
    """Train one fold and return validation accuracy."""
    # EuroSAT tensors for this fold
    opt_tr  = torch.FloatTensor(euro_imgs[train_idx]).permute(0, 3, 1, 2)
    lbl_tr  = torch.LongTensor(euro_lbls[train_idx])
    opt_val = torch.FloatTensor(euro_imgs[val_idx]).permute(0, 3, 1, 2)
    lbl_val = torch.LongTensor(euro_lbls[val_idx])

    tr_ds  = TensorDataset(opt_tr, lbl_tr)
    val_ds = TensorDataset(opt_val, lbl_val)
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE,
                            shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False)

    s1_loader = make_s1_loader(s1_imgs, s1_bbox, s1_masks, BATCH_SIZE)
    s1_iter   = iter(s1_loader)

    model     = get_model(num_classes=10, pretrained=True).to(device)
    criterion = MultiTaskLoss(lambda_cls=1.0,
                              lambda_det=1.0,
                              lambda_seg=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler    = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        for x_opt, y_cls in tr_loader:
            try:
                x_sar, y_det, y_seg = next(s1_iter)
            except StopIteration:
                s1_iter = iter(s1_loader)
                x_sar, y_det, y_seg = next(s1_iter)

            x_opt  = x_opt.to(device)
            x_sar  = x_sar.to(device)
            y_cls  = y_cls.to(device)
            y_det  = y_det.to(device)
            y_seg  = y_seg.to(device)

            optimizer.zero_grad()
            with autocast():
                pred_cls, pred_det, pred_seg = model(x_opt, x_sar)
                loss, _, _, _ = criterion(
                    pred_cls, y_cls,
                    pred_det, y_det,
                    pred_seg, y_seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    # Validation — classification accuracy only (SAR not needed)
    model.eval()
    dummy_sar = torch.zeros(
        BATCH_SIZE, 3,
        opt_val.shape[2], opt_val.shape[3], device=device)
    all_true, all_pred = [], []
    with torch.no_grad():
        for x_opt, y_cls in val_loader:
            bs   = x_opt.size(0)
            xopt = x_opt.to(device)
            xsar = dummy_sar[:bs]
            pred_cls, _, _ = model(xopt, xsar)
            _, pred = pred_cls.max(1)
            all_true.extend(y_cls.numpy())
            all_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(all_true, all_pred) * 100
    print(f"  Fold {fold_idx + 1}/{N_FOLDS}  Val Acc = {acc:.2f}%")
    return acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading EuroSAT (all splits)...")
    euro_imgs, euro_lbls = load_full_eurosat(args.preprocessed)
    print(f"  Total EuroSAT samples: {len(euro_imgs)}")

    print("Loading S1-AAD training split...")
    s1_imgs, s1_bbox, s1_masks = load_s1_train(args.preprocessed)

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=SEED)
    fold_accs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(euro_imgs, euro_lbls)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} "
              f"(train={len(tr_idx)}, val={len(val_idx)}) ---")
        acc = train_fold(fold_idx, tr_idx, val_idx,
                         euro_imgs, euro_lbls,
                         s1_imgs, s1_bbox, s1_masks,
                         device)
        fold_accs.append(acc)

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n=== 5-Fold CV Results ===")
    print(f"Fold accuracies : {[round(a, 2) for a in fold_accs]}")
    print(f"Mean ± Std      : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  (Paper reports 98.32% ± 0.17%)")

    results = {
        'fold_accuracies': [round(a, 2) for a in fold_accs],
        'mean_accuracy':   round(mean_acc, 2),
        'std_accuracy':    round(std_acc, 4),
        'n_folds':         N_FOLDS,
        'epochs_per_fold': EPOCHS,
    }
    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, 'kfold_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='5-fold CV for SentinelFuseNet')
    parser.add_argument('--preprocessed', required=True,
                        help='Path to preprocessed .npy files')
    parser.add_argument('--output_path',  default='results/kfold')
    args = parser.parse_args()
    main(args)
