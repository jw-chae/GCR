import os

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_versioned_run_dir(base_save_root: str, prefix: str = "v", width: int = 3) -> str:
    ensure_dir(base_save_root)
    i = 1
    while True:
        name = f"{prefix}{i:0{width}d}"
        cand = os.path.join(base_save_root, name)
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=False)
            return cand
        i += 1
