import logging
import math
from typing import Dict, List, Optional, Tuple

import open_clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _normalize_pretrained(pretrained: Optional[str]) -> Optional[str]:
    if pretrained is None:
        return None
    s = str(pretrained).strip()
    if s.lower() in {"", "none", "null"}:
        return None
    return s


def _default_pretrained_for(backbone: str) -> str:
    if str(backbone) == "ViT-B-16":
        return "laion400m_e32"
    try:
        for model_name, pretrained_name in open_clip.list_pretrained():
            if model_name == backbone:
                return pretrained_name
    except Exception:
        pass
    return "openai"


def _validate_pretrained(backbone: str, pretrained: str) -> None:
    try:
        avail = [(m, p) for (m, p) in open_clip.list_pretrained() if m == backbone]
    except Exception:
        return
    if not avail:
        return
    if (backbone, pretrained) not in avail:
        examples = ", ".join(sorted({p for (_, p) in avail})[:8])
        raise ValueError(
            f"Unknown OpenCLIP pretrained='{pretrained}' for backbone='{backbone}'. "
            f"Examples for this backbone: {examples}"
        )


class FeatureExtractorMulti(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: str,
        layer_indices: List[int],
        backend: str = "open_clip",
    ):
        super().__init__()
        self.layer_indices = list(map(int, layer_indices))
        self.backend = str(backend).strip().lower()
        if self.backend == "torchvision_vitb16":
            try:
                from torchvision.models.vision_transformer import ViT_B_16_Weights, vit_b_16
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import torchvision for ViT-B-16. "
                    "Check your torch/torchvision install."
                ) from exc
            print(f"Loading torchvision ViT-B-16 (IMAGENET1K_V1) | layers={self.layer_indices}")
            self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.model.eval()
        else:
            pretrained_arg = _normalize_pretrained(pretrained)
            if pretrained_arg is None:
                pretrained_arg = _default_pretrained_for(backbone)
                logging.warning(
                    "OpenCLIP pretrained was not set (None/empty). Falling back to '%s' for backbone '%s'.",
                    pretrained_arg,
                    backbone,
                )
            _validate_pretrained(backbone, pretrained_arg)
            print(f"Loading OpenCLIP: {backbone} ({pretrained_arg}) | layers={self.layer_indices}")
            self.model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_arg)
            self.model.eval()
        self._feats: Dict[int, torch.Tensor] = {}
        for li in self.layer_indices:
            if self.backend == "torchvision_vitb16":
                blk = self.model.encoder.layers[li]
            else:
                blk = self.model.visual.transformer.resblocks[li]
            blk.register_forward_hook(self._make_hook(li))

    def _make_hook(self, li: int):
        def _hook(_module, _inp, out):
            self._feats[li] = out
        return _hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feats.clear()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=x.is_cuda):
                if self.backend == "torchvision_vitb16":
                    x = self.model._process_input(x)
                    n = x.shape[0]
                    batch_class_token = self.model.class_token.expand(n, -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    _ = self.model.encoder(x)
                else:
                    _ = self.model.encode_image(x)

        feats_hw = []
        for li in self.layer_indices:
            feat = self._feats.get(li, None)
            if feat is None:
                raise RuntimeError(f"Hook did not capture features for layer {li}.")
            if feat.dim() != 3:
                raise RuntimeError(f"Unexpected feature shape at layer {li}: {tuple(feat.shape)}")

            if feat.shape[0] == 197 and feat.shape[1] != 197:
                feat = feat.permute(1, 0, 2)

            feat = feat[:, 1:, :]
            bsz, lsz, csz = feat.shape
            h = int(math.sqrt(lsz))
            feat = feat[:, : h * h, :]
            feat = feat.permute(0, 2, 1).contiguous().view(bsz, csz, h, h)
            feats_hw.append(feat.float())

        return torch.cat(feats_hw, dim=1)


class GaussianMixtureEnergyDiag(nn.Module):
    def __init__(
        self,
        k: int,
        c: int,
        chunk_size: int,
        logprec_min: float,
        logprec_max: float,
        logpi_temp: float,
        topk: Optional[int],
    ):
        super().__init__()
        self.k = int(k)
        self.c = int(c)
        self.chunk_size = int(chunk_size)

        self.log_prec = nn.Parameter(torch.zeros(self.k, self.c), requires_grad=False)
        self.log_pi = nn.Parameter(torch.zeros(self.k), requires_grad=False)

        self.logprec_min = float(logprec_min)
        self.logprec_max = float(logprec_max)
        self.logpi_temp = float(logpi_temp)
        self.topk = None if topk is None else int(topk)

    def _log_pi_norm(self) -> torch.Tensor:
        log_pi = self.log_pi / max(self.logpi_temp, 1e-12)
        return log_pi - torch.logsumexp(log_pi, dim=0)

    def score_components(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, csz = q.shape
        assert csz == self.c
        ksz = mu.shape[0]
        assert ksz == self.k

        log_prec = self.log_prec.clamp(self.logprec_min, self.logprec_max)
        prec = torch.exp(log_prec)

        outs = []
        for i in range(0, ksz, self.chunk_size):
            end = min(i + self.chunk_size, ksz)
            mu_c = mu[i:end].unsqueeze(0).unsqueeze(0)
            pr_c = prec[i:end].unsqueeze(0).unsqueeze(0)
            diff = q.unsqueeze(2) - mu_c
            d = (diff * diff * pr_c).sum(dim=-1)
            outs.append(d)
        return torch.cat(outs, dim=-1)

    def soft_nll(self, d_all: torch.Tensor) -> torch.Tensor:
        ksz = d_all.size(-1)
        log_pi = self._log_pi_norm().view(1, 1, ksz)
        log_prec = self.log_prec.clamp(self.logprec_min, self.logprec_max)
        log_det = log_prec.sum(dim=1).view(1, 1, ksz)
        # Likelihood-consistent energy includes the log-det term.
        s = log_pi + 0.5 * log_det - 0.5 * d_all
        if self.topk is not None and self.topk < ksz:
            s_top, _ = torch.topk(s, k=self.topk, dim=-1, largest=True)
            return -torch.logsumexp(s_top, dim=-1)
        return -torch.logsumexp(s, dim=-1)


class MultiHeadModel(nn.Module):
    def __init__(self, banks: Dict[int, torch.Tensor], cfg: Dict[str, object]):
        super().__init__()
        self.cfg = cfg
        self.num_cats = len(banks)

        any_bank = next(iter(banks.values()))
        k, cin = any_bank.shape

        self.mu = nn.ParameterDict()
        self.gmm = nn.ModuleDict()
        for ci, bank in banks.items():
            mu0 = bank.clone()
            self.mu[str(int(ci))] = nn.Parameter(mu0, requires_grad=False)
            self.gmm[str(int(ci))] = GaussianMixtureEnergyDiag(
                k=k,
                c=cin,
                chunk_size=int(cfg["chunk_size"]),
                logprec_min=float(cfg["logprec_min"]),
                logprec_max=float(cfg["logprec_max"]),
                logpi_temp=float(cfg["logpi_temp"]),
                topk=cfg.get("topk", None),
            )

    @torch.no_grad()
    def forward_tokens_infer(self, feats: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        bsz, csz, h, w = feats.shape
        q = feats.permute(0, 2, 3, 1).contiguous().view(bsz, h * w, csz)

        out_token_map = torch.zeros(bsz, h, w, device=feats.device)
        uniq = cat_ids.unique()
        for ci in uniq:
            mask = cat_ids == ci
            if not mask.any():
                continue
            q_sub = q[mask]
            mu_c = self.mu[str(int(ci))]
            gmm_c = self.gmm[str(int(ci))]
            d_all = gmm_c.score_components(q_sub, mu_c)
            tok = gmm_c.soft_nll(d_all)
            out_token_map[mask] = tok.view(-1, h, w)
        return out_token_map

    @torch.no_grad()
    def route_topk(
        self,
        feats: torch.Tensor,
        candidate_cat_ids: List[int],
        topk: int = 1,
        patch_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not candidate_cat_ids:
            raise RuntimeError("No candidate categories for routing.")

        bsz, csz, h, w = feats.shape
        q = feats.permute(0, 2, 3, 1).contiguous().view(bsz, h * w, csz).float()
        n_tokens = int(q.size(1))

        ps = patch_samples
        if ps is None:
            ps = int(self.cfg.get("routing_eval_patch_samples", 0))
        if ps and 0 < int(ps) < n_tokens:
            idx = torch.randperm(n_tokens, device=q.device)[: int(ps)]
            q = q.index_select(1, idx)
            n_tokens = int(q.size(1))

        cand = [int(ci) for ci in candidate_cat_ids]
        m = len(cand)
        if m == 0:
            raise RuntimeError("No candidate categories for routing.")

        qf = q.reshape(bsz * n_tokens, csz)
        qn = (qf * qf).sum(dim=1, keepdim=True)

        mus = [self.mu[str(int(ci))].float() for ci in cand]
        k0 = int(mus[0].size(0))
        if any(int(mu.size(0)) != k0 for mu in mus):
            raise RuntimeError("All candidate categories must have the same number of prototypes.")
        mu_all = torch.cat(mus, dim=0)
        mun = (mu_all * mu_all).sum(dim=1, keepdim=True).t()
        dot = qf @ mu_all.t()
        dist = qn + mun - 2.0 * dot
        dist = dist.view(bsz * n_tokens, m, k0)
        d_patch = dist.min(dim=2)[0]
        d_img = d_patch.view(bsz, n_tokens, m).sum(dim=1)

        k = min(max(int(topk), 1), m)
        topd, topi = torch.topk(d_img, k=k, dim=1, largest=False)
        cand_t = torch.tensor(cand, device=d_img.device, dtype=torch.long)
        routed = cand_t[topi]
        return routed, topd

    @torch.no_grad()
    def stage2_update_ema(
        self,
        backbone: nn.Module,
        dl_train: DataLoader,
        cat_id: int,
        device: str,
        ema_alpha: float = 0.05,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        backbone.eval()
        self.eval()
        ci = int(cat_id)
        mu = self.mu[str(ci)]
        gmm = self.gmm[str(ci)]

        logprec_min = float(self.cfg.get("logprec_min", -6.0))
        logprec_max = float(self.cfg.get("logprec_max", 6.0))
        a = float(ema_alpha)
        eps = 1e-12

        used = 0
        for bidx, batch in enumerate(tqdm(dl_train, desc=f"Stage2EMA@{ci}", leave=False)):
            if max_batches is not None and bidx >= int(max_batches):
                break
            imgs = batch[0]
            imgs = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

            feats = backbone(imgs)
            bsz, csz, h, w = feats.shape
            q = feats.permute(0, 2, 3, 1).contiguous().view(bsz, h * w, csz)
            qf = q.reshape(-1, csz).float()

            mu_f = mu.float()
            diff = qf.unsqueeze(1) - mu_f.unsqueeze(0)
            d = (diff * diff).sum(dim=-1)
            af = d.argmin(dim=-1)

            mu_af = mu_f[af]
            diff2 = (qf - mu_af) ** 2

            ksz = mu_f.size(0)
            sumsq = torch.zeros((ksz, csz), device=imgs.device, dtype=torch.float32)
            cnt = torch.zeros((ksz,), device=imgs.device, dtype=torch.float32)
            sumsq.index_add_(0, af, diff2)
            cnt.index_add_(0, af, torch.ones((af.numel(),), device=imgs.device, dtype=torch.float32))
            batch_var = sumsq / cnt.clamp_min(1.0).unsqueeze(1)

            cur_log_prec = gmm.log_prec.detach().float().clamp(logprec_min, logprec_max)
            cur_var = torch.exp(-cur_log_prec)
            new_var = (1.0 - a) * cur_var + a * batch_var
            new_log_prec = (-torch.log(new_var.clamp_min(eps))).clamp(logprec_min, logprec_max)
            gmm.log_prec.data.copy_(new_log_prec)

            used += 1

        return {"enabled": True, "batches_used": float(used), "ema_alpha": float(a), "cat_id": float(ci)}
