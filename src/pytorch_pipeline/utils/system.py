import os
from dataclasses import asdict, dataclass

import torch
from psutil import virtual_memory


@dataclass
class HardwareProfile:
    gpu: str | None
    vram: float
    ram: float
    cpu_count: int | None

    def to_dict(self):
        return asdict(self)


def resolve_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
        round(virtual_memory().total / (1024**3), 3),
        os.cpu_count(),
    )
