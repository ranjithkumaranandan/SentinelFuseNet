# Dataset Setup

## 1. EuroSAT
- **Source:** https://github.com/phelber/EuroSAT
- **Samples:** 27,000 · **Classes:** 10 · **Size:** 64×64 RGB
- **Sensor:** Sentinel-2 multispectral

## 2. S1-AAD
- **Source:** https://doi.org/10.1016/j.dib.2026.112472
- **Samples:** 1,040 · **Classes:** 1 · **Size:** 200×200 PNG
- **Sensor:** Sentinel-1 SAR (C-band)

## 3. PASTIS
- **Source:** https://github.com/VSainteuf/pastis-benchmark
- **Samples:** 2,433 · **Classes:** 18 · **Size:** 128×128
- **Sensor:** Sentinel-2 time-series

## Preprocessing
```bash
python datasets/preprocess.py \
    --eurosat_path /path/to/EuroSAT \
    --s1aad_path /path/to/S1-AAD \
    --output_path /path/to/output
```

## Output Files
```
preprocessed/
├── eurosat_images.npy  → (27000, 128, 128, 3)
├── eurosat_labels.npy  → (27000,)
├── s1_images.npy       → (1040, 128, 128, 3)
├── s1_bboxes.npy       → (1040, 5)
└── s1_masks.npy        → (1040, 128, 128)
```
