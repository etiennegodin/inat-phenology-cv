OPTUNA_SAMPLERS = {
    "tpe": ("optuna.samplers", "TPESampler", {}),
    "random": ("optuna.samplers", "RandomSampler", {}),
}
OPTUNA_PRUNERS = {
    "median": ("optuna.pruners", "MedianPruner", {}),
    "hyperband": ("optuna.pruners", "HyperbandPruner", {}),
    "none": ("optuna.pruners", "NopPruner", {}),
}
