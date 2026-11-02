"""
SentinelFuseNet Evaluation Script.
Evaluates a trained model on the test split and reports all 9 metrics.

Metrics reported:
    Traditional  : Accuracy, Precision, Recall, F1-Score, mIoU
    Established  : BF1, Detection IoU
    Study-specific: MTCS, WTP, FGI

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai

Usage:
    python scripts/test.py \
        --data_path  /path/to/preprocessed \
        --model_path results/best_model.pth \
        --output_path results
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SentinelFuseNet import get_model
from utils.metrics import (boundary_f1_score, multi_task_consistency_score,
                           weighted_task_performance, fusion_gain_index,
                           detection_iou, compute_miou)


def load_test_data(data_path: str, batch_size: int = 32):
    """Load test split arrays and return a DataLoader."""
    opt   = np.load(os.path.join(data_path, 'test_eurosat_images.npy'))
    lbl   = np.load(os.path.join(data_path, 'test_eurosat_labels.npy'))
    s1    = np.load(os.path.join(data_path, 'test_s1_images.npy'))
    bbox  = np.load(os.path.join(data_path, 'test_s1_bboxes.npy'))
    masks = np.load(os.path.join(data_path, 'test_s1_masks.npy'))

    opt_t   = torch.FloatTensor(opt).permute(0, 3, 1, 2)
    lbl_t   = torch.LongTensor(lbl)
    s1_t    = torch.FloatTensor(s1).permute(0, 3, 1, 2)
    bbox_t  = torch.FloatTensor(bbox)
    masks_t = torch.FloatTensor(masks).permute(0, 3, 1, 2)

    ds = TensorDataset(opt_t, s1_t, lbl_t, bbox_t, masks_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def evaluate(model, loader, device):
    """Run inference and collect all predictions."""
    model.eval()
    all_cls_true, all_cls_pred = [], []
    all_det_pred, all_det_true = [], []
    all_seg_pred, all_seg_true = [], []

    with torch.no_grad():
        for x_opt, x_sar, y_cls, y_det, y_seg in loader:
            x_opt = x_opt.to(device)
            x_sar = x_sar.to(device)

            pred_cls, pred_det, pred_seg = model(x_opt, x_sar)

            _, cls_pred = pred_cls.max(1)
            all_cls_true.extend(y_cls.numpy())
            all_cls_pred.extend(cls_pred.cpu().numpy())

            all_det_pred.extend(pred_det.cpu().numpy())
            all_det_true.extend(y_det.numpy())

            seg_bin = (torch.sigmoid(pred_seg) > 0.5).float()
            all_seg_pred.extend(seg_bin.cpu().numpy())
            all_seg_true.extend(y_seg.numpy())

    return (np.array(all_cls_true), np.array(all_cls_pred),
            np.array(all_det_pred), np.array(all_det_true),
            np.array(all_seg_pred), np.array(all_seg_true))


def compute_all_metrics(cls_true, cls_pred,
                        det_pred, det_true,
                        seg_pred, seg_true):
    """Compute all 9 metrics matching the paper."""
    # Traditional
    acc  = accuracy_score(cls_true, cls_pred) * 100
    prec = precision_score(cls_true, cls_pred,
                           average='weighted', zero_division=0) * 100
    rec  = recall_score(cls_true, cls_pred,
                        average='weighted', zero_division=0) * 100
    f1   = f1_score(cls_true, cls_pred,
                    average='weighted', zero_division=0) * 100

    # mIoU (segmentation)
    miou_vals = []
    for sp, st in zip(seg_pred, seg_true):
        miou_vals.append(
            compute_miou(sp[0].astype(int), st[0].astype(int)))
    miou = np.mean(miou_vals) * 100

    # BF1 (established)
    bf1_vals = []
    for sp, st in zip(seg_pred, seg_true):
        bf1_vals.append(
            boundary_f1_score(sp[0].astype(bool), st[0].astype(bool)))
    bf1 = np.mean(bf1_vals) * 100

    # Detection IoU (established)
    det_iou_vals = []
    for dp, dt in zip(det_pred, det_true):
        det_iou_vals.append(detection_iou(dp[1:], dt[1:]))
    det_iou = np.mean(det_iou_vals) * 100

    # Study-specific
    mtcs = multi_task_consistency_score(
        acc / 100, det_iou / 100, miou / 100) * 100
    wtp  = weighted_task_performance(
        acc / 100, det_iou / 100, miou / 100) * 100

    return {
        'Accuracy':      round(acc,  2),
        'Precision':     round(prec, 2),
        'Recall':        round(rec,  2),
        'F1-Score':      round(f1,   2),
        'mIoU':          round(miou, 2),
        'BF1':           round(bf1,  2),
        'Detection_IoU': round(det_iou, 2),
        'MTCS':          round(mtcs, 2),
        'WTP':           round(wtp,  2),
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading test data...')
    loader = load_test_data(args.data_path, args.batch_size)

    print('Loading model...')
    model = get_model(num_classes=10, pretrained=False).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device))

    print('Running evaluation...')
    (cls_true, cls_pred,
     det_pred, det_true,
     seg_pred, seg_true) = evaluate(model, loader, device)

    metrics = compute_all_metrics(
        cls_true, cls_pred,
        det_pred, det_true,
        seg_pred, seg_true)

    # FGI: best unimodal WTP (U-Net baseline, WTP=44.61%)
    # Update wtp_best_unimodal if you rerun baselines
    wtp_best_unimodal = 44.61
    metrics['FGI'] = round(
        fusion_gain_index(metrics['WTP'], wtp_best_unimodal), 2)

    print("\n=== SentinelFuseNet Test Results ===")
    print(f"{'Metric':<20} {'Value':>8}")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k:<20} {v:>7.2f}%")

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, 'test_results.json')
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate SentinelFuseNet on test split')
    parser.add_argument('--data_path',   required=True)
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--output_path', default='results')
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()
    main(args)
