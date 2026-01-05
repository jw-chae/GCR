# GCR: Geometry-Consistent Routing for Task-Agnostic
Continual Anomaly Detection
## Features

* **Multi-head routing**: select a plausible head without category labels at test time
* **Prototype bank** per category (greedy (k)-center coreset)
* **Energy-based scoring** (LogSumExp / soft-min over prototype distances)
* Metrics: **Image AUROC (macro)**, **Pixel AP (micro)**
* **Continual** evaluation: add categories task-by-task and evaluate over seen categories
* Optional: **EMA2** (`--stage2_ema`) to calibrate per-prototype anisotropy (diagonal precision)

---

## Installation

Conda is recommended for reproducibility.

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate gs_an_env
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate gs_an_env
```

Notes:

* `environment.lock.yml` is a snapshot from the development machine and may not be portable across GPUs/drivers.
* OpenCLIP may download pretrained weights on first run.

### Pip (alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install torch/torchvision separately if needed (CUDA/CPU depends on your setup)
# Example for CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Dataset

### MVTec AD

Expected layout:

```
PATH_TO_YOUR_MVTEC_AD/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*.png
  cable/...
  capsule/...
```

### VisA

This repo assumes the root points to a directory containing category folders.
If you use a common VisA layout like `visa_pytorch/`, set `--data_root` accordingly:

```
PATH_TO_YOUR_VISA/
  visa_pytorch/
    candle/
    capsules/
    ...
```

---

## Quick Start

### Single run (routed over all selected categories)

**MVTec:**

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_YOUR_MVTEC_AD \
  --save_root ./results/gs_an_mvtec_single
```

**VisA:**

```bash
python cli/run.py \
  --dataset visa \
  --data_root PATH_TO_YOUR_VISA \
  --save_root ./results/gs_an_visa_single
```

Subset example:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_YOUR_MVTEC_AD \
  --save_root ./results/gs_an_capsule_single \
  --categories capsule
```

Outputs:

* `results.csv` (one row)

  * `i_auroc`: macro average over categories
  * `p_ap`: pixel AP over all pixels (micro)
* `results_per_category.csv` (one row per category)
* `normal_scores.png` (overlayed histogram of normal image scores per category)

---

## Continual Routing (Task-wise)

Continual mode evaluates categories in sequence using routed inference over **seen** categories.

**MVTec:**

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_YOUR_MVTEC_AD \
  --save_root ./results/gs_an_mvtec_continual \
  --continual
```

**VisA:**

```bash
python cli/run.py \
  --dataset visa \
  --data_root PATH_TO_YOUR_VISA \
  --save_root ./results/gs_an_visa_continual \
  --continual
```

Explicit task order:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_YOUR_MVTEC_AD \
  --save_root ./results/gs_an_mvtec_continual \
  --continual \
  --cat_order bottle cable capsule carpet
```

Example used in experiments:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_YOUR_MVTEC_AD \
  --save_root ./results/gs_an_continual_score_k196 \
  --continual \
  --memory_bank_size 196 \
  --routing_rule score
```

Outputs:

* `results.csv` (one row per task)
* `results_per_category.csv` (one row per task × category)
* `normal_scores_taskXX.png` (normal score histograms per task)

---

## EMA2 (Stage2)

Enable EMA-based anisotropy update:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_YOUR_MVTEC_AD \
  --save_root ./results/gs_an_mvtec_continual_ema2 \
  --continual \
  --stage2_ema
```

Stage2 runs per category after prototype bank construction.

---

## Fixed Paper Configuration

For reproducibility, the following are hard-coded in `cli/run.py`:

* `BACKBONE = ViT-B-16`
* `PRETRAINED = laion400m_e32`
* `LAYER_INDICES = [6]`
* `MEMORY_BANK_SIZE = 64`
* `INFERENCE_MODE = lse`
* `SCORE_MODE = topq`, `TOPQ = 0.01`
* `ROUTING_PATCH_SAMPLES = 32`
* `STAGE2_BATCHES = 50`, `EMA_ALPHA = 0.05`
* Patch L2 normalization is off by default (not exposed as a CLI flag)

CLI overrides:

* `--memory_bank_size` overrides the default bank size.

---

## Metrics

* `i_auroc`: macro AUROC across categories (routed inference)
* `p_ap`: pixel Average Precision over all pixels (micro)

---

## Troubleshooting

* `FileNotFoundError: No categories under ...`

  * Verify `--data_root` points to the dataset root containing category folders.

* CUDA OOM

  * Reduce `--batch_size`.

* Slow routing

  * Reduce `ROUTING_PATCH_SAMPLES` in `cli/run.py` (may affect accuracy).

---
