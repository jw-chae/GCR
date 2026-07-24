import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all RNGs and request deterministic execution where supported."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some CUDA operations do not provide deterministic kernels on every
    # platform. warn_only preserves portability while surfacing such cases.
    torch.use_deterministic_algorithms(True, warn_only=True)
