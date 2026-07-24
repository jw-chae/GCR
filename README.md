# GCR: Geometry-Consistent Routing for Task-Agnostic Continual Anomaly Detection

This repository contains the core implementation used in the paper. It provides the paper-default GCR pipeline, task-agnostic continual evaluation, and the controlled score-based versus geometry-based routing comparison. The released code focuses on the main method and principal reported results rather than every internal experiment launcher.

## Main features

- Frozen LAION-400M-pretrained OpenCLIP ViT-B/16 features
- Category-specific prototype banks built with greedy $k$-center selection
- Geometry-consistent routing without test-time category labels
- Dense anomaly scoring only inside the selected category head
- LogSumExp energy over the nearest prototypes
- Category-wise macro I-AUROC and macro P-AP
- Continual I-FM and P-FM summaries
- Optional diagonal-geometry update for ablation only

## Installation

Conda is recommended.

```bash
conda env create -f environment.yml
conda activate gcr_env
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate gcr_env
```

Pip can also be used:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

OpenCLIP may download pretrained weights on the first run.

## Dataset layout

### MVTec AD

```text
PATH_TO_MVTEC_AD/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*.png
  cable/
  ...
```

### VisA

The data root must contain category directories. For a common converted layout:

```text
PATH_TO_VISA/
  visa_pytorch/
    candle/
    capsules/
    ...
```

When `PATH_TO_VISA/visa_pytorch` exists, the runner resolves it automatically.

## Paper-default configuration

The default configuration in `cli/run.py` matches the main paper:

| Item | Default |
|---|---|
| Backbone | OpenCLIP ViT-B/16 |
| Pretraining | LAION-400M (`laion400m_e32`) |
| Feature block | 6 |
| Input resolution | $224\times224$ |
| Prototype bank size | $K=196$ per category |
| Nearest prototypes for scoring | $K'=16$ |
| Routing tokens | $M_r=32$ |
| Routing distance | Raw isotropic squared Euclidean distance |
| Patch-feature L2 normalization | Off |
| Patch aggregation | LogSumExp energy |
| Image score | Mean of the largest 1% pixel scores |
| Global random seed | 0 |
| Diagonal adaptation | Off by default; ablation only |

The default run uses a single seed-0 evaluation. Category folders are sorted lexicographically unless `--cat_order` is supplied.

## Quick start

### Single routed evaluation

MVTec AD:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_MVTEC_AD \
  --save_root ./results/gcr_mvtec_single
```

VisA:

```bash
python cli/run.py \
  --dataset visa \
  --data_root PATH_TO_VISA \
  --save_root ./results/gcr_visa_single
```

A category subset can be selected explicitly:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_MVTEC_AD \
  --save_root ./results/gcr_capsule_single \
  --categories capsule
```

### Task-agnostic continual evaluation

MVTec AD:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_MVTEC_AD \
  --save_root ./results/gcr_mvtec_continual \
  --continual
```

VisA:

```bash
python cli/run.py \
  --dataset visa \
  --data_root PATH_TO_VISA \
  --save_root ./results/gcr_visa_continual \
  --continual
```

By default, categories are introduced in lexicographic directory order. An explicit order can be supplied:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_MVTEC_AD \
  --save_root ./results/gcr_custom_order \
  --continual \
  --cat_order bottle cable capsule carpet
```

## Controlled routing comparison

The main routing comparison uses the same frozen features, prototypes, and within-head scoring while changing only the routing rule:

```bash
bash scripts/run_routing_ablation.sh PATH_TO_MVTEC_AD
```

This runs:

- score-based routing
- geometry-consistent routing

Both use $K=196$ and seed 0.

## Optional diagonal-geometry ablation

The paper uses isotropic geometry by default. The following option enables the optional EMA-based diagonal update used only in ablation:

```bash
python cli/run.py \
  --dataset mvtec \
  --data_root PATH_TO_MVTEC_AD \
  --save_root ./results/gcr_mvtec_diagonal_ablation \
  --continual \
  --stage2_ema
```

## Output files and metrics

### `results.csv`

For single evaluation, this file contains one routed-evaluation row.

For continual evaluation, it contains one row per continual step followed by a final forgetting-summary row with:

- `i_fm`: image-level forgetting measure
- `p_fm`: pixel-level forgetting measure
- `forgetting`: backward-compatible alias of `i_fm`

### `results_per_category.csv`

Contains category-wise I-AUROC and P-AP at each evaluated step. Final per-category I-FM and P-FM rows are also appended in continual mode.

### Metric definitions

- `i_auroc`: macro-average of category-wise image AUROC after task-agnostic routing
- `p_ap`: macro-average of category-wise pixel average precision after task-agnostic routing
- `routing_acc`: fraction of images routed to their ground-truth category
- `i_fm`: average drop from each pre-final category's best post-introduction I-AUROC to its final-step I-AUROC
- `p_fm`: analogous forgetting measure for category-wise P-AP

Ground-truth category labels are used only for evaluation and never for routing or scoring.

The conditional values `p_ap_correct_routed` and `p_ap_misrouted` used in the routing diagnostic pool all pixels inside the corresponding correctly routed or misrouted subset.

## Relevant command-line options

```text
--dataset {mvtec,visa}
--data_root PATH
--save_root PATH
--continual
--categories CAT [CAT ...]
--cat_order CAT [CAT ...]
--memory_bank_size INT
--routing_rule {geometry,score}
--seed INT
--batch_size INT
--device DEVICE
--stage2_ema
```

## Reproducibility notes

- The default global seed is 0.
- PyTorch, NumPy, and Python random generators are seeded.
- cuDNN benchmarking is disabled and deterministic behavior is requested where supported.
- Routing samples $M_r=32$ patch positions from the PyTorch random generator initialized by the global seed.
- Prototype initialization is seeded from the global seed.
- The main paper reports the seed-0 configuration.

## Troubleshooting

`FileNotFoundError: No categories under ...`

- Check that `--data_root` points to the directory containing category folders.

CUDA out-of-memory

- Reduce `--batch_size`.

Slow routing

- Routing compares the sampled tokens with every observed category bank. The paper-default routing sample count is fixed at $M_r=32$.
