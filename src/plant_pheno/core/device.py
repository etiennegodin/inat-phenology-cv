from __future__ import annotations

import torch


def get_device() -> torch.device:
    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {d}")
    return torch.device(d)


def set_device(d: str = "cuda") -> torch.device:
    print(f"Running on {d}")
    return torch.device(d)
