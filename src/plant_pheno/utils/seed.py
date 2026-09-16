from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, set_cuda_deterministic: bool = False) -> None:
    """Sets seed across all RNGs: Python random, NumPy, PyTorch CPU, and PyTorch CUDA.

    Args:
        seed: The integer seed to apply.
        set_cuda_deterministic: If True, sets CuDNN to deterministic mode.
            Defaults to False to preserve GPU performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if set_cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function to seed
    Python random and NumPy in worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
