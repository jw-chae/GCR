import os
import time
from typing import Any, Dict, List, Optional

import torch

from engine.evaluator import eval_routed
from model.bank import build_banks
from model.module import MultiHeadModel
from utils.datasets import MultiMVTecTest, MultiMVTecTrain, make_loader
from utils.io import append_csv
from utils.plot import save_normal_score_distributions


@torch.no_grad()
def run_continual_routed(
    *,
    backbone,
    data_root: str,
    cat_order: List[str],
    categories_all: List[str],
    cfg: Dict[str, Any],
    save_root: str,
) -> None:
    if str(cfg.get("geometry", "multihead")).lower() != "multihead":
        raise ValueError("run_continual_routed requires geometry=multihead")
    if not cat_order:
        raise ValueError("cat_order must be non-empty")

    device = str(cfg["device"])
    backbone.eval()

    os.makedirs(save_root, exist_ok=True)
    results_csv = os.path.join(save_root, "results.csv")
    per_cat_csv = os.path.join(save_root, "results_per_category.csv")
    if os.path.exists(results_csv):
        os.remove(results_csv)
    if os.path.exists(per_cat_csv):
        os.remove(per_cat_csv)

    cat_to_id = {str(c): int(i) for i, c in enumerate(categories_all)}
    seen_banks: Dict[int, torch.Tensor] = {}
    stage2_logprec_by_ci: Dict[int, torch.Tensor] = {}
    per_task_cat_auc: List[Dict[int, float]] = []

    stage2_enabled = bool(cfg.get("stage2_ema", False))
    max_batches: Optional[int] = None
    if stage2_enabled:
        mb = int(cfg.get("stage2_batches", 50) or 0)
        max_batches = None if mb <= 0 else mb

    seen: List[str] = []
    for t, cat in enumerate(cat_order):
        cat = str(cat)
        if cat not in cat_to_id:
            raise ValueError(f"Unknown category in cat_order: {cat}")
        global_ci = int(cat_to_id[cat])

        ds_tr = MultiMVTecTrain(data_root, [cat], input_size=int(cfg["input_size"]))
        dl_tr = make_loader(ds_tr, cfg, shuffle=False)
        banks_local = build_banks(backbone, dl_tr, [cat], cfg)
        seen_banks[int(global_ci)] = banks_local[0]

        model = MultiHeadModel(dict(seen_banks), cfg=cfg).to(device)
        model.eval()

        for ci, lp in stage2_logprec_by_ci.items():
            k = str(int(ci))
            if k in model.gmm:
                model.gmm[k].log_prec.data.copy_(lp.to(device=device, dtype=model.gmm[k].log_prec.dtype))

        if stage2_enabled:
            _ = model.stage2_update_ema(
                backbone=backbone,
                dl_train=dl_tr,
                cat_id=int(global_ci),
                device=device,
                ema_alpha=float(cfg.get("ema_alpha", 0.05)),
                max_batches=max_batches,
            )
            stage2_logprec_by_ci[int(global_ci)] = model.gmm[str(int(global_ci))].log_prec.detach().clone().cpu()

        if cat not in seen:
            seen.append(cat)
        candidate_cat_ids = [int(cat_to_id[c]) for c in seen]

        ds_seen = MultiMVTecTest(data_root, list(categories_all), input_size=int(cfg["input_size"]))
        dl_seen = make_loader(ds_seen, cfg, shuffle=False)

        cfg_eval = dict(cfg)
        cfg_eval["routing_diagnostics"] = bool(t == (len(cat_order) - 1))
        if cfg_eval["routing_diagnostics"]:
            cfg_eval["overlay_dir"] = os.path.join(save_root, "overlays")
        metrics = eval_routed(
            backbone=backbone,
            model=model,
            dl_test=dl_seen,
            candidate_cat_ids=candidate_cat_ids,
            cfg=cfg_eval,
        )

        ts = int(time.time())
        append_csv(
            [
                {
                    "timestamp_unix": int(ts),
                    "run_type": "continual_routed",
                    "task": int(t + 1),
                    "category_added": str(cat),
                    "n_seen": int(len(seen)),
                    "stage2_ema": bool(stage2_enabled),
                    "routing_rule": str(cfg.get("routing_rule", "geometry")),
                    "i_auroc": float(metrics.get("i_auroc", float("nan"))),
                    "p_ap": float(metrics.get("p_ap", float("nan"))),
                    "routing_acc": float(metrics.get("routing_acc", float("nan"))),
                    "routing_acc_all": float(metrics.get("routing_acc_all", float("nan"))),
                    "routing_acc_normal": float(metrics.get("routing_acc_normal", float("nan"))),
                    "routing_acc_anom": float(metrics.get("routing_acc_anom", float("nan"))),
                    "p_ap_correct_routed": float(metrics.get("p_ap_correct_routed", float("nan"))),
                    "p_ap_misrouted": float(metrics.get("p_ap_misrouted", float("nan"))),
                    "n_samples": int(metrics.get("n_samples", 0)),
                }
            ],
            results_csv,
        )

        per_cat_auc = dict(metrics.get("per_category_i_auroc", {}) or {})
        per_cat_p_ap = dict(metrics.get("per_category_p_ap", {}) or {})
        per_task_cat_auc.append({int(ci): float(per_cat_auc.get(int(ci), float("nan"))) for ci in candidate_cat_ids})
        per_cat_normals = dict(metrics.get("per_category_normal_scores", {}) or {})
        rows = []
        for ci in candidate_cat_ids:
            name = categories_all[int(ci)] if 0 <= int(ci) < len(categories_all) else str(ci)
            rows.append(
                {
                    "timestamp_unix": int(ts),
                    "run_type": "continual_routed",
                    "task": int(t + 1),
                    "category": str(name),
                    "category_id": int(ci),
                    "n_seen": int(len(seen)),
                    "stage2_ema": bool(stage2_enabled),
                    "i_auroc": float(per_cat_auc.get(int(ci), float("nan"))),
                    "p_ap": float(per_cat_p_ap.get(int(ci), float("nan"))),
                }
            )
        append_csv(rows, per_cat_csv)

        plot_scores = {}
        for ci in candidate_cat_ids:
            name = categories_all[int(ci)] if 0 <= int(ci) < len(categories_all) else str(ci)
            plot_scores[str(name)] = per_cat_normals.get(int(ci), [])
        save_normal_score_distributions(
            plot_scores,
            os.path.join(save_root, f"normal_scores_task{t + 1:02d}.png"),
            title=f"Task {t + 1} Normal Score Distributions",
        )

    t_total = len(cat_order)
    if t_total > 1:
        fm_per_cat: Dict[int, float] = {}
        final_task_idx = t_total - 1
        final_task = per_task_cat_auc[final_task_idx] if final_task_idx < len(per_task_cat_auc) else {}
        for idx in range(0, t_total - 1):
            cat = str(cat_order[idx])
            if cat not in cat_to_id:
                continue
            ci = int(cat_to_id[cat])
            prev_vals = []
            for t in range(idx, t_total - 1):
                if t < len(per_task_cat_auc):
                    val = per_task_cat_auc[t].get(ci, float("nan"))
                    if val == val:
                        prev_vals.append(float(val))
            final_val = float(final_task.get(ci, float("nan")))
            if not prev_vals or final_val != final_val:
                fm_per_cat[ci] = float("nan")
            else:
                fm_per_cat[ci] = float(max(prev_vals) - final_val)
        fm_vals = [v for v in fm_per_cat.values() if v == v]
        fm_overall = float(sum(fm_vals) / len(fm_vals)) if fm_vals else float("nan")

        append_csv(
            [
                {
                    "timestamp_unix": int(time.time()),
                    "run_type": "continual_routed_forgetting",
                    "n_tasks": int(t_total),
                    "stage2_ema": bool(stage2_enabled),
                    "routing_rule": str(cfg.get("routing_rule", "geometry")),
                    "forgetting": float(fm_overall),
                }
            ],
            results_csv,
        )
