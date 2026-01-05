import os
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for score distribution plots") from exc


def _normalize_items(per_cat_scores: Dict[str, Iterable[float]]) -> List[Tuple[str, np.ndarray]]:
    items: List[Tuple[str, np.ndarray]] = []
    for name, scores in per_cat_scores.items():
        arr = np.asarray(list(scores), dtype=np.float32)
        if arr.size == 0:
            continue
        items.append((str(name), arr))
    return items


def save_normal_score_distributions(
    per_cat_scores: Dict[str, Iterable[float]],
    out_path: str,
    *,
    title: str,
    bins: int = 50,
) -> bool:
    items = _normalize_items(per_cat_scores)
    if not items:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=160)
    for name, scores in items:
        plt.hist(scores, bins=bins, density=True, histtype="step", alpha=0.9, label=name)

    plt.xlabel("Normal image score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True
