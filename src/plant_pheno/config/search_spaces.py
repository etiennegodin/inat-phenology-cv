# ── HYPERPARAMETER SEARCH SPACES (used by Optuna) ──────────────────────────────

BASE_SEARCH_SPACE = {
    "base_lr": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-4,
        "log": True,
    },
    "head_dropout_prob": {"type": "float", "low": 0.5, "high": 0.5},
    "backbone_decay": {
        "type": "float",
        "low": 0.98,
        "high": 0.98,
    },
    "attention_dropout_prob": {"type": "float", "low": 0.0, "high": 0.0},
    "attention_neurons": {
        "type": "categorical",
        "choices": [128],
    },
    "gated": {
        "type": "categorical",
        "choices": [True],
    },
}

BAD_SEARCH_SPACE = {
    "base_lr": {
        "type": "float",
        "low": 1e-2,
        "high": 1e-2,
        "log": True,
    },
    "head_dropout_prob": {"type": "float", "low": 0.7, "high": 0.7},
    "backbone_decay": {
        "type": "float",
        "low": 0.98,
        "high": 0.98,
    },
    "attention_dropout_prob": {"type": "float", "low": 0.7, "high": 0.7},
    "attention_neurons": {
        "type": "categorical",
        "choices": [128],
    },
    "gated": {
        "type": "categorical",
        "choices": [True],
    },
}

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

SEARCH_SPACES = {
    "bad": BAD_SEARCH_SPACE,
    "base": BASE_SEARCH_SPACE,
    "coarse": COARSE_SEARCH_SPACE,
}
