from .train.dataset import get_samples
from .utils.configs import Config


def status(configs: Config):
    df = get_samples(configs.paths_params, configs.dataset_params)
    print(f"Dataset size={df.shape[0]}")
