import glob
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")


def list_mvtec_categories(data_root: str) -> List[str]:
    if not os.path.isdir(data_root):
        return []
    out = []
    for name in os.listdir(data_root):
        p = os.path.join(data_root, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return sorted(out)


def list_categories(data_root: str) -> List[str]:
    return list_mvtec_categories(data_root)


def _glob_images(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    out: List[str] = []
    for ext in IMG_EXTENSIONS:
        out.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    kept: List[str] = []
    for p in sorted(set(out)):
        try:
            if os.path.getsize(p) <= 0:
                continue
        except Exception:
            continue
        kept.append(p)
    return kept


def _safe_open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def _safe_open_mask(path: str) -> Image.Image:
    m = Image.open(path)
    m.load()
    return m.convert("L")


def _mask_path(img_path: str, category_root: str, mask_index: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    parts = img_path.replace("\\", "/").split("/")
    if "test" not in parts:
        return None
    idx = parts.index("test")
    cls = parts[idx + 1]
    if cls == "good":
        return None

    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    gt_dir = os.path.join(category_root, "ground_truth", cls)

    for cand in (f"{stem}.png", f"{stem}_mask.png"):
        p = os.path.join(gt_dir, cand)
        if os.path.exists(p):
            return p

    if mask_index is not None:
        files = mask_index.get(cls, [])
        for mp in files:
            try:
                mstem = os.path.splitext(os.path.basename(mp))[0]
            except Exception:
                continue
            if stem in mstem:
                return mp
    else:
        if os.path.isdir(gt_dir):
            for mp in sorted(glob.glob(os.path.join(gt_dir, "*"))):
                mstem = os.path.splitext(os.path.basename(mp))[0]
                if stem in mstem:
                    return mp
    return None


class MultiMVTecTrain(Dataset):
    """Train-good images for bank construction."""

    def __init__(self, data_root: str, categories: List[str], input_size: int):
        self.data_root = str(data_root)
        self.categories = list(categories)
        self.input_size = int(input_size)

        self.img_tf = T.Compose(
            [
                T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        self.items: List[Tuple[str, int]] = []
        for ci, cat in enumerate(self.categories):
            cat_root = os.path.join(self.data_root, cat)
            img_dir = os.path.join(cat_root, "train", "good")
            for p in _glob_images(img_dir):
                self.items.append((p, int(ci)))

    def __len__(self) -> int:
        return len(self.items)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def __getitem__(self, idx: int):
        last_err: Optional[Exception] = None
        for off in range(5):
            p, ci = self.items[(idx + off) % len(self.items)]
            try:
                img = _safe_open_rgb(p)
                x = self._norm(self.img_tf(img))
                return x, ci
            except Exception as e:
                last_err = e
                continue
        raise OSError(f"Failed to load any image after retries (idx={idx}). Last error: {last_err}")


class MultiMVTecTest(Dataset):
    """Test images with masks. Returns (img, cat_id, label, mask)."""

    def __init__(self, data_root: str, categories: List[str], input_size: int):
        self.data_root = str(data_root)
        self.categories = list(categories)
        self.input_size = int(input_size)

        self.img_tf = T.Compose(
            [
                T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        self.mask_tf = T.Compose(
            [
                T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        self.items: List[Tuple[str, int, int, Optional[str]]] = []
        for ci, cat in enumerate(self.categories):
            cat_root = os.path.join(self.data_root, cat)
            test_dir = os.path.join(cat_root, "test")

            mask_index: Dict[str, List[str]] = {}
            gt_root = os.path.join(cat_root, "ground_truth")
            if os.path.isdir(gt_root):
                for cls in sorted(os.listdir(gt_root)):
                    cls_dir = os.path.join(gt_root, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    files: List[str] = []
                    for ext in IMG_EXTENSIONS:
                        files.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
                    mask_index[cls] = sorted(set(files))

            for cls in sorted(os.listdir(test_dir)):
                cls_dir = os.path.join(test_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                y = 0 if cls == "good" else 1
                for p in _glob_images(cls_dir):
                    mpath = _mask_path(p, cat_root, mask_index=mask_index)
                    self.items.append((p, int(ci), int(y), mpath))

    def __len__(self) -> int:
        return len(self.items)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def __getitem__(self, idx: int):
        last_err: Optional[Exception] = None
        for off in range(5):
            p, ci, y, mpath = self.items[(idx + off) % len(self.items)]
            try:
                img = _safe_open_rgb(p)
                x = self._norm(self.img_tf(img))
                if mpath is None:
                    mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
                else:
                    try:
                        m = _safe_open_mask(mpath)
                        mask = self.mask_tf(m)
                        mask = (mask > 0.5).float()
                    except Exception:
                        mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
                return x, ci, y, mask
            except Exception as e:
                last_err = e
                continue
        raise OSError(f"Failed to load any test image after retries (idx={idx}). Last error: {last_err}")


def make_loader(ds: Dataset, cfg: Dict[str, object], shuffle: bool) -> DataLoader:
    def _env_int(name: str, default: int) -> int:
        v = os.getenv(name)
        return default if v is None else int(v)

    def _env_bool(name: str, default: bool) -> bool:
        v = os.getenv(name)
        return default if v is None else bool(int(v))

    device = str(cfg.get("device", ""))
    nw = max(_env_int("GSDFM_NUM_WORKERS", int(cfg.get("num_workers", 0))), 0)
    prefetch = _env_int("GSDFM_PREFETCH_FACTOR", int(cfg.get("prefetch_factor", 4)))
    pin = _env_bool("GSDFM_PIN_MEMORY", bool(cfg.get("pin_memory", ("cuda" in device))))
    persist = _env_bool("GSDFM_PERSISTENT_WORKERS", bool(cfg.get("persistent_workers", True)))

    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=bool(shuffle),
        num_workers=nw,
        pin_memory=bool(pin),
        persistent_workers=bool(persist) if nw > 0 else False,
        prefetch_factor=int(prefetch) if nw > 0 else None,
        drop_last=False,
    )
