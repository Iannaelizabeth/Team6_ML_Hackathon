"""
model_3.py - Model 3: HistGradientBoosting  (family: boosting)

Builds trees one after another, each correcting the previous trees' errors.
Handles missing feature values natively (no imputer) and is fast on ~1M rows.
Early stopping holds out 10% of training rows and records train/validation loss
per iteration - a real training history, plotted to model_3_training_curve.png.
Limitation: that 10% hold-out is row-level (not trip-grouped), so the validation
curve is slightly optimistic; the reported CV metrics are trip-grouped.

Run from the repo root:  python -m src.model_3
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_NUMBER = 3
MODEL_NAME = "hist_gradient_boosting"
MODEL_FAMILY = "boosting"

DEFAULT_PARAMS: dict = {
    "learning_rate": 0.1,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0,
}

# For bayesian_optimization.py
SEARCH_SPACE: dict = {
    "learning_rate": ("log-uniform", 0.01, 0.3),
    "max_leaf_nodes": ("int", 15, 127),
    "min_samples_leaf": ("int", 20, 500),
    "l2_regularization": ("log-uniform", 1e-6, 10.0),
}


def params_from_search(sample: dict) -> dict:
    p = dict(DEFAULT_PARAMS)
    for k in ("learning_rate", "l2_regularization"):
        if k in sample:
            p[k] = float(sample[k])
    for k in ("max_leaf_nodes", "min_samples_leaf", "max_iter"):
        if k in sample:
            p[k] = int(sample[k])
    return p


def build_model(params: dict | None = None, random_state: int = 42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return HistGradientBoostingClassifier(
        **p, loss="log_loss", early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, scoring="loss", random_state=random_state)


def training_history(model) -> dict | None:
    """Per-iteration log loss actually recorded by the final fit (sklearn stores negative loss)."""
    if len(getattr(model, "train_score_", [])) == 0:
        return None
    return {"train": [-float(v) for v in model.train_score_],
            "validation": [-float(v) for v in model.validation_score_],
            "xlabel": "boosting iteration", "ylabel": "log loss"}


def extra_results(model, features) -> dict:
    return {"n_iterations_used": int(model.n_iter_)}


if __name__ == "__main__":
    from types import SimpleNamespace
    from src.model_runner import run
    run(SimpleNamespace(**globals()))
