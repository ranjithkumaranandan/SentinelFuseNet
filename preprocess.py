"""
Data preprocessing for SentinelFuseNet.

Prepares EuroSAT (Sentinel-2 optical) and S1-AAD (Sentinel-1 SAR)
datasets into numpy arrays ready for training.

EuroSAT  : 27,000 images × 10 land-use classes, 64×64 RGB JPEG
S1-AAD   : 1,040 SAR images with YOLO bounding boxes and binary masks

Partitioning:
    EuroSAT : 70% train / 15% val / 15% test  (stratified by class)
    S1-AAD  : 70% train / 15% val / 15% test  (independent, random seed 42)
              → 728 train / 156 val / 156 test

Both datasets are resized to 128 × 128 pixels before saving.
S1-AAD SAR images are replicated to 3-channel to match ResNet-18 input.

Author: Ranjith Kumar Anandan
Institution: SRM Institute of Science and Technology, Chennai

Usage:
    python datasets/preprocess.py \
        --eurosat_path /path/to/EuroSAT/2750 \
        --s1aad_path   /path/to/S1-AAD \
        --output_path  /path/to/preprocessed
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]
IMG_SIZE   = 128
SEED       = 42
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15


def load_eurosat(root: str):
    """
    Load EuroSAT RGB images from the flat 10-class folder structure.

    Args:
        root: Path to EuroSAT/2750 directory (contains one folder per class)

    Returns:
        images: float32 array [N, 128, 128, 3], normalised to [0, 1]
        labels: int64 array  [N]
    """
    images, labels = [], []
    for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
        cls_dir = Path(root) / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")
        for img_path in sorted(cls_dir.glob('*.jpg')):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(cls_idx)
    print(f"EuroSAT loaded: {len(images)} images, "
          f"{len(set(labels))} classes")
    return np.array(images), np.array(labels, dtype=np.int64)


def load_s1aad(root: str):
    """
    Load S1-AAD Sentinel-1 SAR images, bounding boxes, and masks.

    Expects directory layout:
        <root>/images/    *.jpg  — grayscale SAR patches
        <root>/labels/    *.txt  — YOLO format: class xc yc w h (normalised)
        <root>/masks/     *.png  — binary segmentation masks

    Args:
        root: Path to S1-AAD root directory

    Returns:
        images: float32 [N, 128, 128, 3], normalised, 3-channel
        bboxes: float32 [N, 5]  — (class, xc, yc, w, h) normalised
        masks:  float32 [N, 128, 128, 1], binary [0,1]
    """
    img_dir  = Path(root) / 'images'
    lbl_dir  = Path(root) / 'labels'
    mask_dir = Path(root) / 'masks'

    images, bboxes, masks = [], [], []
    for img_path in sorted(img_dir.glob('*.jpg')):
        stem = img_path.stem

        # SAR image → grayscale → 3-channel replication
        img = Image.open(img_path).convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=-1)   # H×W×3
        images.append(arr)

        # YOLO bounding box (first annotation only)
        lbl_path = lbl_dir / f'{stem}.txt'
        if lbl_path.exists():
            with open(lbl_path) as f:
                line = f.readline().strip().split()
            bbox = [float(v) for v in line[:5]]
        else:
            bbox = [0.0, 0.5, 0.5, 1.0, 1.0]
        bboxes.append(bbox)

        # Binary segmentation mask
        mask_path = mask_dir / f'{stem}.png'
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((IMG_SIZE, IMG_SIZE),
                               Image.NEAREST)
            mask_arr = (np.array(mask, dtype=np.float32) / 255.0)
            mask_arr = (mask_arr > 0.5).astype(np.float32)
        else:
            mask_arr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        masks.append(mask_arr[..., np.newaxis])

    print(f"S1-AAD loaded: {len(images)} images")
    return (np.array(images),
            np.array(bboxes, dtype=np.float32),
            np.array(masks, dtype=np.float32))


def split_and_save(eurosat_images, eurosat_labels,
                   s1_images, s1_bboxes, s1_masks,
                   output_path: str):
    """
    Stratified split of EuroSAT; independent split of S1-AAD.
    Saves train/val/test numpy arrays to output_path.
    """
    os.makedirs(output_path, exist_ok=True)
    N_e = len(eurosat_images)
    N_s = len(s1_images)

    # EuroSAT stratified split
    idx_e = np.arange(N_e)
    tr_e, tmp_e = train_test_split(
        idx_e, test_size=VAL_FRAC + TEST_FRAC,
        stratify=eurosat_labels, random_state=SEED)
    val_e, te_e = train_test_split(
        tmp_e, test_size=0.5,
        stratify=eurosat_labels[tmp_e], random_state=SEED)

    # S1-AAD independent random split
    idx_s = np.arange(N_s)
    tr_s, tmp_s = train_test_split(
        idx_s, test_size=VAL_FRAC + TEST_FRAC, random_state=SEED)
    val_s, te_s = train_test_split(
        tmp_s, test_size=0.5, random_state=SEED)

    splits = {
        'train': (tr_e,  tr_s),
        'val':   (val_e, val_s),
        'test':  (te_e,  te_s)
    }
    counts = {}
    for split, (ie, is_) in splits.items():
        np.save(os.path.join(output_path,
                             f'{split}_eurosat_images.npy'),
                eurosat_images[ie])
        np.save(os.path.join(output_path,
                             f'{split}_eurosat_labels.npy'),
                eurosat_labels[ie])
        np.save(os.path.join(output_path,
                             f'{split}_s1_images.npy'),
                s1_images[is_])
        np.save(os.path.join(output_path,
                             f'{split}_s1_bboxes.npy'),
                s1_bboxes[is_])
        np.save(os.path.join(output_path,
                             f'{split}_s1_masks.npy'),
                s1_masks[is_])
        counts[split] = {
            'eurosat': int(len(ie)),
            's1aad':   int(len(is_))
        }

    with open(os.path.join(output_path, 'split_info.json'), 'w') as f:
        json.dump(counts, f, indent=2)
    print("Splits saved:")
    for k, v in counts.items():
        print(f"  {k:5s}: EuroSAT={v['eurosat']:5d}  S1-AAD={v['s1aad']:3d}")


def main(args):
    print("Loading EuroSAT...")
    eurosat_images, eurosat_labels = load_eurosat(args.eurosat_path)

    print("Loading S1-AAD...")
    s1_images, s1_bboxes, s1_masks = load_s1aad(args.s1aad_path)

    print("Splitting and saving...")
    split_and_save(eurosat_images, eurosat_labels,
                   s1_images, s1_bboxes, s1_masks,
                   args.output_path)
    print("Preprocessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess EuroSAT and S1-AAD for SentinelFuseNet')
    parser.add_argument('--eurosat_path', required=True,
                        help='Path to EuroSAT/2750 directory')
    parser.add_argument('--s1aad_path',   required=True,
                        help='Path to S1-AAD root directory')
    parser.add_argument('--output_path',  required=True,
                        help='Output directory for .npy files')
    args = parser.parse_args()
    main(args)
