"""
Novel Evaluation Metrics for SentinelFuseNet.
Implements FGI, BF1, MTCS, WTP, and Detection IoU.

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai
"""

import numpy as np
import torch


def fusion_gain_index(acc_fused, acc_opt, acc_sar):
    """
    Fusion Gain Index (FGI):
    Quantifies percentage improvement of cross-sensor fusion
    over the best-performing single-sensor baseline.

    FGI = (Acc_fused - max(Acc_opt, Acc_sar)) /
           max(Acc_opt, Acc_sar) × 100

    Args:
        acc_fused: Accuracy of fused model
        acc_opt:   Accuracy of optical-only baseline
        acc_sar:   Accuracy of SAR-only baseline

    Returns:
        FGI score (percentage)
    """
    best_unimodal = max(acc_opt, acc_sar)
    fgi = ((acc_fused - best_unimodal) / best_unimodal) * 100
    return fgi


def boundary_f1_score(pred_mask, gt_mask, threshold=2):
    """
    Boundary F1-Score (BF1):
    Computes F1-Score exclusively at segmentation mask
    boundary pixels for stringent delineation assessment.

    Args:
        pred_mask: Predicted binary mask [H, W]
        gt_mask:   Ground truth binary mask [H, W]
        threshold: Boundary pixel dilation (pixels)

    Returns:
        BF1 score
    """
    from scipy.ndimage import binary_dilation

    # Extract boundary pixels
    pred_boundary = binary_dilation(pred_mask) ^ pred_mask
    gt_boundary   = binary_dilation(gt_mask) ^ gt_mask

    # Dilate boundaries for tolerance
    pred_dilated = binary_dilation(pred_boundary,
                                   iterations=threshold)
    gt_dilated   = binary_dilation(gt_boundary,
                                   iterations=threshold)

    # Compute boundary precision and recall
    precision_b = np.sum(pred_boundary & gt_dilated) / \
                  (np.sum(pred_boundary) + 1e-8)
    recall_b    = np.sum(gt_boundary & pred_dilated) / \
                  (np.sum(gt_boundary) + 1e-8)

    # BF1
    if precision_b + recall_b == 0:
        return 0.0
    bf1 = 2 * precision_b * recall_b / (precision_b + recall_b)
    return bf1


def multi_task_consistency_score(acc, det_iou, miou):
    """
    Multi-Task Consistency Score (MTCS):
    Measures balance of performance across all three tasks.
    Penalizes architectures that excel at one task at the
    expense of others.

    MTCS = 1 - σ(Acc, Det-IoU, mIoU) / μ(Acc, Det-IoU, mIoU)

    Args:
        acc:     Classification accuracy
        det_iou: Detection IoU
        miou:    Segmentation mIoU

    Returns:
        MTCS score (higher = more balanced)
    """
    scores = np.array([acc, det_iou, miou])
    mean   = np.mean(scores)
    std    = np.std(scores)
    mtcs   = 1 - (std / (mean + 1e-8))
    return mtcs


def weighted_task_performance(accuracy, miou, f1_score):
    """
    Weighted Task Performance (WTP):
    Composite score aggregating all three task performances.

    WTP = 0.4 × Accuracy + 0.3 × mIoU + 0.3 × F1-Score

    Args:
        accuracy:  Classification accuracy
        miou:      Segmentation mIoU
        f1_score:  Classification F1-Score

    Returns:
        WTP score
    """
    wtp = 0.4 * accuracy + 0.3 * miou + 0.3 * f1_score
    return wtp


def detection_iou(pred_bbox, gt_bbox):
    """
    Detection IoU:
    Computes Intersection-over-Union between predicted
    and ground-truth YOLO bounding boxes.

    Args:
        pred_bbox: Predicted bbox [x_c, y_c, w, h] (normalized)
        gt_bbox:   Ground truth bbox [x_c, y_c, w, h] (normalized)

    Returns:
        IoU score
    """
    # Convert center format to corner format
    pred_x1 = pred_bbox[0] - pred_bbox[2] / 2
    pred_y1 = pred_bbox[1] - pred_bbox[3] / 2
    pred_x2 = pred_bbox[0] + pred_bbox[2] / 2
    pred_y2 = pred_bbox[1] + pred_bbox[3] / 2

    gt_x1 = gt_bbox[0] - gt_bbox[2] / 2
    gt_y1 = gt_bbox[1] - gt_bbox[3] / 2
    gt_x2 = gt_bbox[0] + gt_bbox[2] / 2
    gt_y2 = gt_bbox[1] + gt_bbox[3] / 2

    # Intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    inter_area = max(0, inter_x2 - inter_x1) * \
                 max(0, inter_y2 - inter_y1)

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area   = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / (union_area + 1e-8)
    return iou


def compute_miou(pred_mask, gt_mask, num_classes=2):
    """
    Compute mean Intersection-over-Union for segmentation.

    Args:
        pred_mask:   Predicted mask [H, W]
        gt_mask:     Ground truth mask [H, W]
        num_classes: Number of classes

    Returns:
        mIoU score
    """
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls   = (gt_mask == cls)
        intersection = np.sum(pred_cls & gt_cls)
        union        = np.sum(pred_cls | gt_cls)
        if union == 0:
            continue
        iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0.0
