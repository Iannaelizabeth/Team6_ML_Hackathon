"""Reusable Random Forest and MLP model factories for candidate ranking."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_random_forest(config=None, random_state=42):
    """Return an imputation + Random Forest pipeline (no unnecessary scaling)."""
    params = {"n_estimators": 120, "max_depth": None, "min_samples_split": 2,
              "min_samples_leaf": 1, "max_features": "sqrt",
              "class_weight": "balanced_subsample", "n_jobs": -1,
              "random_state": random_state}
    params.update(config or {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", RandomForestClassifier(**params)),
    ])


def make_mlp(config=None, random_state=42):
    """Return a scaled MLP pipeline; scaling/imputation fit only on train folds."""
    params = {"hidden_layer_sizes": (64, 32), "activation": "relu", "solver": "adam",
              "alpha": 1e-4, "batch_size": 512, "learning_rate_init": 1e-3,
              "max_iter": 30, "early_stopping": False, "tol": 1e-4,
              "random_state": random_state, "verbose": False}
    params.update(config or {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(**params)),
    ])


def balanced_sample_weights(y):
    """Class-balanced per-row weights, preserving every training row."""
    import numpy as np
    labels, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    weights = {label: len(y) / (len(labels) * count) for label, count in zip(labels, counts)}
    return np.asarray([weights[int(label)] for label in y], dtype=float)
