# ── HYPERPARAMETER SEARCH SPACES (used by Optuna) ──────────────────────────────

COARSE_SEARCH_SPACE = {
    "base_lr": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    },
    "head_dropout_prob": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.05},
    "backbone_decay": {
        "type": "float",
        "low": 0.8,
        "high": 0.98,
    },
    "attention_dropout_prob": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
    "attention_neurons": {
        "type": "categorical",
        "choices": [64, 128, 256],
    },
    "gated": {
        "type": "categorical",
        "choices": [True, False],
    },
}

SEARCH_SPACES = {"coarse": COARSE_SEARCH_SPACE}
