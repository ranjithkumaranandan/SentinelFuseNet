"""
SentinelFuseNet Training Script.
Trains the model with multi-task loss on EuroSAT + S1-AAD datasets.

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai

Usage:
    python scripts/train.py \
        --data_path /path/to/preprocessed \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-4 \
        --seed 42
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SentinelFuseNet import get_model


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data(data_path):
    """Load preprocessed numpy arrays."""
    eurosat_images = np.load(
        os.path.join(data_path, 'eurosat_images.npy'))
    eurosat_labels = np.load(
        os.path.join(data_path, 'eurosat_labels.npy'))
    s1_images      = np.load(
        os.path.join(data_path, 's1_images.npy'))
    s1_bboxes      = np.load(
        os.path.join(data_path, 's1_bboxes.npy'))
    s1_masks       = np.load(
        os.path.join(data_path, 's1_masks.npy'))
    return eurosat_images, eurosat_labels, s1_images, \
           s1_bboxes, s1_masks


def create_dataloaders(eurosat_images, eurosat_labels,
                       s1_images, s1_bboxes, s1_masks,
                       batch_size=32, seed=42):
    """Create stratified train/val/test dataloaders."""
    N = len(eurosat_images)
    indices = np.arange(N)

    # Stratified split: 70/15/15
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30,
        stratify=eurosat_labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=eurosat_labels[temp_idx], random_state=seed)

    # Verify zero overlap
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    def make_loader(idx, shuffle=True):
        opt  = torch.FloatTensor(
            eurosat_images[idx]).permute(0, 3, 1, 2)
        lbl  = torch.LongTensor(eurosat_labels[idx])
        s1   = torch.FloatTensor(
            s1_images[idx % len(s1_images)]).permute(0, 3, 1, 2)
        bbox = torch.FloatTensor(
            s1_bboxes[idx % len(s1_bboxes)])
        mask = torch.FloatTensor(
            s1_masks[idx % len(s1_masks)]).unsqueeze(1)
        ds = TensorDataset(opt, s1, lbl, bbox, mask)
        return DataLoader(ds, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0)

    return (make_loader(train_idx, shuffle=True),
            make_loader(val_idx, shuffle=False),
            make_loader(test_idx, shuffle=False),
            test_idx)


def train_one_epoch(model, loader, optimizer,
                    criterion_cls, criterion_det,
                    criterion_seg, scaler, device):
    """Train for one epoch with AMP."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x_opt, x_sar, y_cls, y_det, y_seg in loader:
        x_opt = x_opt.to(device)
        x_sar = x_sar.to(device)
        y_cls = y_cls.to(device)
        y_det = y_det.to(device)
        y_seg = y_seg.to(device)

        optimizer.zero_grad()

        with autocast():
            pred_cls, pred_det, pred_seg = model(x_opt, x_sar)
            loss_cls = criterion_cls(pred_cls, y_cls)
            loss_det = criterion_det(pred_det, y_det)
            loss_seg = criterion_seg(pred_seg, y_seg)
            loss     = loss_cls + loss_det + loss_seg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = pred_cls.max(1)
        correct += predicted.eq(y_cls).sum().item()
        total   += y_cls.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion_cls,
             criterion_det, criterion_seg, device):
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for x_opt, x_sar, y_cls, y_det, y_seg in loader:
            x_opt = x_opt.to(device)
            x_sar = x_sar.to(device)
            y_cls = y_cls.to(device)
            y_det = y_det.to(device)
            y_seg = y_seg.to(device)

            pred_cls, pred_det, pred_seg = model(x_opt, x_sar)
            loss_cls = criterion_cls(pred_cls, y_cls)
            loss_det = criterion_det(pred_det, y_det)
            loss_seg = criterion_seg(pred_seg, y_seg)
            loss     = loss_cls + loss_det + loss_seg

            total_loss += loss.item()
            _, predicted = pred_cls.max(1)
            correct += predicted.eq(y_cls).sum().item()
            total   += y_cls.size(0)

    return total_loss / len(loader), correct / total


def main(args):
    set_seed(args.seed)

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    print('Loading data...')
    eurosat_images, eurosat_labels, \
    s1_images, s1_bboxes, s1_masks = load_data(args.data_path)

    # Dataloaders
    train_loader, val_loader, test_loader, _ = \
        create_dataloaders(
            eurosat_images, eurosat_labels,
            s1_images, s1_bboxes, s1_masks,
            batch_size=args.batch_size,
            seed=args.seed)

    # Model
    model = get_model(num_classes=10, pretrained=True).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_det = nn.MSELoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scaler    = GradScaler()

    # Training loop
    best_val_loss = float('inf')
    os.makedirs(args.output_path, exist_ok=True)
    history = []

    print(f'Training for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer,
            criterion_cls, criterion_det,
            criterion_seg, scaler, device)

        val_loss, val_acc = validate(
            model, val_loader,
            criterion_cls, criterion_det,
            criterion_seg, device)

        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f'Epoch [{epoch:02d}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc*100:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc*100:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_path,
                                    'best_model.pth'))
            print(f'  ✓ Best model saved at epoch {epoch}')

    # Save training history
    with open(os.path.join(args.output_path,
                           'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete.')
    print(f'Best model saved to: '
          f'{os.path.join(args.output_path, "best_model.pth")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SentinelFuseNet')
    parser.add_argument('--data_path', type=str,
                        required=True,
                        help='Path to preprocessed data')
    parser.add_argument('--output_path', type=str,
                        default='results',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
