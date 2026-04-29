from .train.dataset import get_samples
from .utils.config import Config


def status(configs: Config):
    df = get_samples(configs.paths, configs.samples_params)
    print(f"Dataset size={df.shape[0]}")
