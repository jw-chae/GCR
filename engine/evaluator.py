import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from matplotlib import cm as _mpl_cm

from model.module import MultiHeadModel
from utils.metrics import image_score_from_map, safe_ap, safe_auc


@torch.no_grad()
def eval_routed(
    backbone,
    model: MultiHeadModel,
    dl_test,
    candidate_cat_ids: List[int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(model, MultiHeadModel):
        raise RuntimeError("Routing eval requires MultiHeadModel.")
    if not candidate_cat_ids:
        raise RuntimeError("candidate_cat_ids must be non-empty.")

    device = cfg["device"]
    input_size = int(cfg["input_size"])
    patch_samples = int(cfg.get("routing_eval_patch_samples", 32))
    score_mode = str(cfg.get("score_mode", "topq"))
    topq = float(cfg.get("topq", 0.01))
    routing_rule = str(cfg.get("routing_rule", "geometry")).lower()
    routing_diagnostics = bool(cfg.get("routing_diagnostics", False))
    overlay_dir = str(cfg.get("overlay_dir", "") or "")
    if routing_diagnostics and overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
    # CLIP mean/std used in dataset normalization.
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    backbone.eval()
    model.eval()

    cand_set = {int(ci) for ci in candidate_cat_ids}
    img_scores: List[np.ndarray] = []
    img_labels: List[np.ndarray] = []
    per_cat_scores: Dict[int, List[float]] = {}
    per_cat_labels: Dict[int, List[int]] = {}
    per_cat_normals: Dict[int, List[float]] = {}
    px_scores: List[np.ndarray] = []
    px_masks: List[np.ndarray] = []
    per_cat_px_scores: Dict[int, List[np.ndarray]] = {}
    per_cat_px_masks: Dict[int, List[np.ndarray]] = {}
    routing_correct = 0
    routing_total = 0
    routing_correct_normal = 0
    routing_total_normal = 0
    routing_correct_anom = 0
    routing_total_anom = 0
    px_scores_correct: List[np.ndarray] = []
    px_masks_correct: List[np.ndarray] = []
    px_scores_misrouted: List[np.ndarray] = []
    px_masks_misrouted: List[np.ndarray] = []

    img_idx = 0
    for batch in tqdm(dl_test, desc="EvalRouted", leave=False):
        imgs, cat_id, y, mask = batch
        y_np = y.numpy().astype(np.int64)
        true_ci_np = cat_id.numpy().astype(np.int64)

        keep = np.asarray([int(ci) in cand_set for ci in true_ci_np], dtype=bool)
        if not bool(keep.any()):
            continue

        imgs = imgs[keep].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y_keep = y_np[keep]
        true_ci_keep = true_ci_np[keep]
        mask_keep = mask[keep]

        feats = backbone(imgs)
        if routing_rule == "score":
            cand = [int(ci) for ci in candidate_cat_ids]
            cand_t = torch.tensor(cand, device=feats.device, dtype=torch.long)
            scores = []
            for ci in cand:
                ci_tensor = torch.full((feats.size(0),), int(ci), device=feats.device, dtype=torch.long)
                token_map = model.forward_tokens_infer(feats, ci_tensor)
                mp_up = F.interpolate(
                    token_map.unsqueeze(1),
                    size=(input_size, input_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                sc_img = image_score_from_map(mp_up, score_mode=score_mode, topq=topq)
                scores.append(sc_img)
            score_mat = torch.stack(scores, dim=1)
            min_idx = torch.argmin(score_mat, dim=1)
            routed_top1 = cand_t[min_idx]
        else:
            routed_cids, _ = model.route_topk(
                feats,
                candidate_cat_ids=candidate_cat_ids,
                topk=1,
                patch_samples=patch_samples,
            )
            routed_top1 = routed_cids[:, 0]

        token_map = model.forward_tokens_infer(feats, routed_top1)
        mp_up = F.interpolate(
            token_map.unsqueeze(1),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        sc_img = image_score_from_map(mp_up, score_mode=score_mode, topq=topq)

        sc_img_np = sc_img.detach().float().cpu().numpy().astype(np.float32)
        img_scores.append(sc_img_np)
        img_labels.append(y_keep.astype(np.int64))
        routed_top1_np = routed_top1.detach().cpu().numpy().astype(np.int64)
        is_correct = routed_top1_np == true_ci_keep
        routing_correct += int(is_correct.sum())
        routing_total += int(routed_top1_np.size)
        if routing_diagnostics:
            y_keep_np = y_keep.astype(np.int64)
            routing_total_normal += int((y_keep_np == 0).sum())
            routing_total_anom += int((y_keep_np == 1).sum())
            routing_correct_normal += int(((y_keep_np == 0) & is_correct).sum())
            routing_correct_anom += int(((y_keep_np == 1) & is_correct).sum())
        mp_up_np = mp_up.detach().float().cpu().numpy().astype(np.float32)
        mask_keep_np = mask_keep.squeeze(1).detach().cpu().numpy().astype(np.uint8)
        for i in range(len(y_keep)):
            ci = int(true_ci_keep[i])
            score = float(sc_img_np[i])
            label = int(y_keep[i])
            per_cat_scores.setdefault(ci, []).append(score)
            per_cat_labels.setdefault(ci, []).append(label)
            if label == 0:
                per_cat_normals.setdefault(ci, []).append(score)
            per_cat_px_scores.setdefault(ci, []).append(mp_up_np[i])
            per_cat_px_masks.setdefault(ci, []).append(mask_keep_np[i])
        if routing_diagnostics:
            if is_correct.any():
                px_scores_correct.append(mp_up_np[is_correct])
                px_masks_correct.append(mask_keep_np[is_correct])
            if (~is_correct).any():
                px_scores_misrouted.append(mp_up_np[~is_correct])
                px_masks_misrouted.append(mask_keep_np[~is_correct])
            if overlay_dir:
                imgs_denorm = (imgs * clip_std + clip_mean).clamp(0.0, 1.0)
                imgs_np = imgs_denorm.detach().cpu().numpy()
                for i in range(imgs_np.shape[0]):
                    img = (imgs_np[i].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                    heat = mp_up_np[i]
                    hmin = float(np.min(heat))
                    hmax = float(np.max(heat))
                    heat_n = (heat - hmin) / (hmax - hmin + 1e-6)
                    cmap = _mpl_cm.get_cmap("viridis")
                    heat_rgb = (cmap(heat_n)[..., :3] * 255.0).astype(np.float32)
                    overlay = (0.5 * img.astype(np.float32) + 0.5 * heat_rgb).clip(0, 255).astype(np.uint8)
                    ci = int(true_ci_keep[i])
                    label = int(y_keep[i])
                    routed = int(routed_top1_np[i])
                    base = f"{img_idx:06d}_c{ci}_y{label}_r{routed}"
                    Image.fromarray(img).save(os.path.join(overlay_dir, f"{base}.png"))
                    Image.fromarray(overlay).save(os.path.join(overlay_dir, f"{base}_overlay.png"))
                    img_idx += 1
        px_scores.append(mp_up_np)
        px_masks.append(mask_keep_np)

    img_scores_np = np.concatenate(img_scores, axis=0) if img_scores else np.zeros((0,), dtype=np.float32)
    img_labels_np = np.concatenate(img_labels, axis=0) if img_labels else np.zeros((0,), dtype=np.int64)
    px_scores_np = np.concatenate(px_scores, axis=0) if px_scores else np.zeros((0, input_size, input_size), dtype=np.float32)
    px_masks_np = np.concatenate(px_masks, axis=0) if px_masks else np.zeros((0, input_size, input_size), dtype=np.uint8)

    per_vals: List[float] = []
    per_cat_auroc: Dict[int, float] = {}
    for ci in candidate_cat_ids:
        scores = per_cat_scores.get(int(ci), [])
        labels = per_cat_labels.get(int(ci), [])
        auc = safe_auc(labels, scores) if scores else float("nan")
        per_cat_auroc[int(ci)] = float(auc)
        if np.isfinite(float(auc)):
            per_vals.append(float(auc))

    i_auroc_macro = float(np.mean(per_vals)) if per_vals else float("nan")
    routing_acc = float(routing_correct / routing_total) if routing_total > 0 else float("nan")
    per_cat_p_ap: Dict[int, float] = {}
    per_p_ap_vals: List[float] = []
    for ci in candidate_cat_ids:
        scores = per_cat_px_scores.get(int(ci), [])
        masks = per_cat_px_masks.get(int(ci), [])
        if not scores or not masks:
            per_cat_p_ap[int(ci)] = float("nan")
            continue
        scores_np = np.asarray(scores, dtype=np.float32)
        masks_np = np.asarray(masks, dtype=np.uint8)
        ap = safe_ap(masks_np.reshape(-1), scores_np.reshape(-1))
        per_cat_p_ap[int(ci)] = float(ap)
        if np.isfinite(float(ap)):
            per_p_ap_vals.append(float(ap))
    p_ap_macro = float(np.mean(per_p_ap_vals)) if per_p_ap_vals else float("nan")

    out = {
        "i_auroc": float(i_auroc_macro),
        "p_ap": float(p_ap_macro),
        "n_samples": int(img_labels_np.size),
        "routing_acc": float(routing_acc),
        "per_category_i_auroc": per_cat_auroc,
        "per_category_p_ap": per_cat_p_ap,
        "per_category_normal_scores": per_cat_normals,
    }
    if routing_diagnostics:
        routing_acc_normal = float(routing_correct_normal / routing_total_normal) if routing_total_normal > 0 else float("nan")
        routing_acc_anom = float(routing_correct_anom / routing_total_anom) if routing_total_anom > 0 else float("nan")
        if px_scores_correct and px_masks_correct:
            pc_scores = np.concatenate(px_scores_correct, axis=0)
            pc_masks = np.concatenate(px_masks_correct, axis=0)
            p_ap_correct = float(safe_ap(pc_masks.reshape(-1), pc_scores.reshape(-1)))
        else:
            p_ap_correct = float("nan")
        if px_scores_misrouted and px_masks_misrouted:
            pm_scores = np.concatenate(px_scores_misrouted, axis=0)
            pm_masks = np.concatenate(px_masks_misrouted, axis=0)
            p_ap_mis = float(safe_ap(pm_masks.reshape(-1), pm_scores.reshape(-1)))
        else:
            p_ap_mis = float("nan")
        out.update(
            {
                "routing_acc_all": float(routing_acc),
                "routing_acc_normal": float(routing_acc_normal),
                "routing_acc_anom": float(routing_acc_anom),
                "p_ap_correct_routed": float(p_ap_correct),
                "p_ap_misrouted": float(p_ap_mis),
            }
        )
    return out
