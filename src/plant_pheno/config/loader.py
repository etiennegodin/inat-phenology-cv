from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .params import (
    DataLoadersParams,
    DatasetParams,
    PathsParams,
)
from .system import HardwareProfile

logger = logging.getLogger(__name__)


@dataclass
class Config:
    config_path: Path
    paths_params: PathsParams
    dataloaders_params: DataLoadersParams
    hardware_profile: HardwareProfile
    git_branch: str
    git_hash: str
    dataset_params: DatasetParams = field(default_factory=DatasetParams)
    cuda: bool = False
    test: bool = False
    max_img_resolution: int = 500

    def to_dict(self):
        return asdict(self)
