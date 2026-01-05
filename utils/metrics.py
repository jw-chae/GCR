import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_ap(y_true, y_score) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def image_score_from_map(mp_up: torch.Tensor, score_mode: str, topq: float) -> torch.Tensor:
    mode = str(score_mode).lower()
    if mode != "topq":
        raise ValueError(f"Only score_mode='topq' is supported (got {score_mode})")

    v = mp_up.view(mp_up.size(0), -1)
    q = min(max(float(topq), 1e-6), 1.0)
    k = max(int(round(v.size(1) * q)), 1)
    topk_vals, _ = torch.topk(v, k=k, dim=1, largest=True)
    return topk_vals.mean(dim=1)
