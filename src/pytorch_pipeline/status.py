from .config import Config
from .dataset import get_samples


def status(configs: Config):
    df = get_samples(configs.paths, configs.samples_params)
    print(f"Dataset size={df.shape[0]}")
