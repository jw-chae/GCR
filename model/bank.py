from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def get_coreset(features: np.ndarray, n_samples: int, seed: int) -> torch.Tensor:
    feats = np.asarray(features, dtype=np.float32)
    n = int(feats.shape[0])
    if n == 0:
        return torch.empty((0, feats.shape[1]), dtype=torch.float32)

    n_samples = min(int(n_samples), n)
    rng = np.random.RandomState(int(seed))
    idx = [int(rng.randint(n))]

    norms = (feats * feats).sum(axis=1)
    y0 = idx[0]
    md = norms + float(norms[y0]) - 2.0 * (feats @ feats[y0])

    for _ in tqdm(range(1, n_samples), desc="Coreset", leave=False):
        f = int(np.argmax(md))
        idx.append(f)
        d = norms + float(norms[f]) - 2.0 * (feats @ feats[f])
        md = np.minimum(md, d)

    return torch.tensor(feats[idx], dtype=torch.float32)


@torch.no_grad()
def collect_train_features(backbone, dl_train, cfg) -> Tuple[np.ndarray, np.ndarray]:
    feats_np = []
    cats_np = []
    for imgs, cat_id in tqdm(dl_train, desc="CollectTrainFeats", leave=False):
        imgs = imgs.to(cfg["device"], non_blocking=True).contiguous(memory_format=torch.channels_last)
        f = backbone(imgs)  # (B,C,h,w)
        bsz, ctot, h, w = f.shape
        f2 = f.float().permute(0, 2, 3, 1).reshape(bsz, -1, ctot)
        feats_np.append(f2.reshape(-1, ctot).cpu().numpy())
        cats_np.append(np.repeat(cat_id.numpy().astype(np.int64), f2.size(1)))
    feats_np = np.concatenate(feats_np, axis=0)
    cats_np = np.concatenate(cats_np, axis=0)
    return feats_np, cats_np


@torch.no_grad()
def build_banks(backbone, dl_train, categories: List[str], cfg) -> Dict[int, torch.Tensor]:
    feats_flat, cats_flat = collect_train_features(backbone, dl_train, cfg)
    k = int(cfg["memory_bank_size"])

    per_cat: Dict[int, torch.Tensor] = {}
    for ci, _cat in enumerate(categories):
        feats_c = feats_flat[cats_flat == ci]
        bank = get_coreset(feats_c, k, seed=int(cfg["seed"]) + 777 + int(ci)).to(cfg["device"])
        per_cat[int(ci)] = bank
    return per_cat
