import argparse
import os
import sys
import time
from typing import Any, Dict, List

import torch


def _stabilize_sys_path() -> None:
    impl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    workspace_root = os.path.abspath(os.path.join(impl_root, ".."))
    for p in (workspace_root, impl_root):
        if p and p not in sys.path:
            sys.path.insert(0, p)


_stabilize_sys_path()

from engine.evaluator import eval_routed
from engine.trainer import run_continual_routed
from model.bank import build_banks
from model.module import FeatureExtractorMulti, MultiHeadModel
from utils.datasets import MultiMVTecTest, MultiMVTecTrain, list_categories, make_loader
from utils.io import append_csv
from utils.seed import seed_everything as set_seed
from utils.setup import set_tf32
from utils.plot import save_normal_score_distributions


BACKBONE = "ViT-B-16"
PRETRAINED = "laion400m_e32"
LAYER_INDICES = [6]
INPUT_SIZE = 224
MEMORY_BANK_SIZE = 96
CHUNK_SIZE = 64
LOGPREC_MIN = -6.0
LOGPREC_MAX = 6.0
LOGPI_TEMP = 1.0
TOPK_COMPONENTS = 16
SCORE_MODE = "topq"
TOPQ = 0.01
INFERENCE_MODE = "lse"
ENERGY_SCORE = "energy"
GEOMETRY = "multihead"
DECISION = "energy"
ROUTING_PATCH_SAMPLES = 32
STAGE2_BATCHES = 50
EMA_ALPHA = 0.05
PATCH_NORM = False


def _resolve_data_root_inplace(cfg: Dict[str, Any]) -> None:
    dataset = str(cfg.get("dataset", "mvtec")).lower()
    data_root = str(cfg.get("data_root", ""))
    if dataset == "visa":
        cand = os.path.join(data_root, "visa_pytorch")
        if os.path.isdir(cand):
            cfg["data_root"] = cand


def _split_list(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        out.extend(parts)
    return out


def _prepare_categories(cfg: Dict[str, Any]) -> List[str]:
    cats_all = list_categories(cfg["data_root"])
    if not cats_all:
        raise FileNotFoundError(f"No categories under {cfg['data_root']}")
    want = cfg.get("categories")
    if want:
        keep = [c for c in want if c in set(cats_all)]
        if not keep:
            raise FileNotFoundError(f"None requested categories exist: {sorted(want)}")
        return keep
    return cats_all


def run_single(cfg: Dict[str, Any]) -> None:
    t_start = time.time()
    set_seed(int(cfg["seed"]))
    set_tf32()
    torch.backends.cudnn.benchmark = True

    os.makedirs(cfg["save_root"], exist_ok=True)
    _resolve_data_root_inplace(cfg)

    cats = _prepare_categories(cfg)

    ds_tr = MultiMVTecTrain(cfg["data_root"], cats, input_size=int(cfg["input_size"]))
    ds_te = MultiMVTecTest(cfg["data_root"], cats, input_size=int(cfg["input_size"]))
    dl_tr = make_loader(ds_tr, cfg, shuffle=True)
    dl_te = make_loader(ds_te, cfg, shuffle=False)

    backbone = FeatureExtractorMulti(
        cfg["backbone"],
        cfg["pretrained"],
        cfg["layer_indices"],
        backend=cfg["vision_backend"],
    ).to(cfg["device"])
    backbone.eval()

    banks = build_banks(backbone, dl_tr, cats, cfg)
    model = MultiHeadModel(banks, cfg=cfg).to(cfg["device"])
    model = model.to(memory_format=torch.channels_last)

    if bool(cfg.get("stage2_ema", False)):
        max_batches = int(cfg.get("stage2_batches", STAGE2_BATCHES))
        if max_batches <= 0:
            max_batches = None
        for ci, cat in enumerate(cats):
            ds_ci = MultiMVTecTrain(cfg["data_root"], [cat], input_size=int(cfg["input_size"]))
            dl_ci = make_loader(ds_ci, cfg, shuffle=False)
            _ = model.stage2_update_ema(
                backbone=backbone,
                dl_train=dl_ci,
                cat_id=int(ci),
                device=cfg["device"],
                ema_alpha=float(cfg.get("ema_alpha", EMA_ALPHA)),
                max_batches=max_batches,
            )

    metrics = eval_routed(
        backbone=backbone,
        model=model,
        dl_test=dl_te,
        candidate_cat_ids=list(range(len(cats))),
        cfg=cfg,
    )

    results_csv = os.path.join(cfg["save_root"], "results.csv")
    per_cat_csv = os.path.join(cfg["save_root"], "results_per_category.csv")
    if os.path.exists(results_csv):
        os.remove(results_csv)
    if os.path.exists(per_cat_csv):
        os.remove(per_cat_csv)
    append_csv(
        [
            {
                "timestamp_unix": int(t_start),
                "run_type": "single_routed",
                "data_root": str(cfg.get("data_root", "")),
                "categories": "|".join([str(c) for c in cats]),
                "n_categories": int(len(cats)),
                "stage2_ema": bool(cfg.get("stage2_ema", False)),
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
    rows = []
    for ci, name in enumerate(cats):
        rows.append(
            {
                "timestamp_unix": int(t_start),
                "run_type": "single_routed",
                "category": str(name),
                "category_id": int(ci),
                "n_categories": int(len(cats)),
                "stage2_ema": bool(cfg.get("stage2_ema", False)),
                "i_auroc": float(per_cat_auc.get(int(ci), float("nan"))),
                "p_ap": float(per_cat_p_ap.get(int(ci), float("nan"))),
            }
        )
    append_csv(rows, per_cat_csv)

    per_cat_normals = dict(metrics.get("per_category_normal_scores", {}) or {})
    plot_scores = {str(cats[int(ci)]): per_cat_normals.get(int(ci), []) for ci in range(len(cats))}
    save_normal_score_distributions(
        plot_scores,
        os.path.join(cfg["save_root"], "normal_scores.png"),
        title="Normal Score Distributions",
    )

    print("\n=== DONE (single routed) ===")
    print(f"Saved -> {cfg['save_root']}")


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    p.add_argument("--save_root", type=str, default="./results/gsdfm_submit")
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--categories", type=str, nargs="*", default=None)
    p.add_argument("--cat_order", type=str, nargs="*", default=None)
    p.add_argument("--continual", action="store_true")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--memory_bank_size", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage2_ema", action="store_true")
    p.add_argument(
        "--routing_rule",
        type=str,
        default="geometry",
        choices=["geometry", "score"],
    )
    p.add_argument(
        "--vision_backend",
        type=str,
        default="open_clip",
        choices=["open_clip", "torchvision_vitb16"],
    )

    args = p.parse_args()

    cfg: Dict[str, Any] = vars(args)
    cfg["categories"] = _split_list(cfg["categories"] or [])
    cfg["cat_order"] = _split_list(cfg["cat_order"] or [])

    cfg.update(
        {
            "geometry": GEOMETRY,
            "decision": DECISION,
            "energy_score": ENERGY_SCORE,
            "inference_mode": INFERENCE_MODE,
            "score_mode": SCORE_MODE,
            "topq": TOPQ,
            "backbone": BACKBONE,
            "pretrained": PRETRAINED,
            "layer_indices": list(LAYER_INDICES),
            "input_size": INPUT_SIZE,
            "memory_bank_size": int(cfg["memory_bank_size"]) if cfg.get("memory_bank_size") else MEMORY_BANK_SIZE,
            "chunk_size": CHUNK_SIZE,
            "logprec_min": LOGPREC_MIN,
            "logprec_max": LOGPREC_MAX,
            "logpi_temp": LOGPI_TEMP,
            "topk": TOPK_COMPONENTS,
            "routing_eval_patch_samples": ROUTING_PATCH_SAMPLES,
            "stage2_batches": STAGE2_BATCHES,
            "ema_alpha": EMA_ALPHA,
            "patch_norm": bool(cfg.get("patch_norm", PATCH_NORM)),
            "vision_backend": cfg["vision_backend"],
            "routing_rule": str(cfg.get("routing_rule", "geometry")).lower(),
        }
    )

    return cfg


def main() -> None:
    cfg = parse_args()
    _resolve_data_root_inplace(cfg)

    if bool(cfg.get("continual", False)) or cfg.get("cat_order"):
        set_seed(int(cfg["seed"]))
        set_tf32()
        torch.backends.cudnn.benchmark = True
        cats = _prepare_categories(cfg)
        cat_order = cfg.get("cat_order") or cats
        run_continual_routed(
            backbone=FeatureExtractorMulti(
                cfg["backbone"],
                cfg["pretrained"],
                cfg["layer_indices"],
                backend=cfg["vision_backend"],
            ).to(cfg["device"]),
            data_root=cfg["data_root"],
            cat_order=list(cat_order),
            categories_all=list(cats),
            cfg=cfg,
            save_root=str(cfg["save_root"]),
        )
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
