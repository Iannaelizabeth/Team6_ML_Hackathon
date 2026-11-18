"""
model_4.py - Model 4: k-Nearest Neighbours  (family: instance-based)

Scores a candidate by looking at the k most similar training candidates (in scaled
feature space) and taking their distance-weighted share of positives.
Missing values are median-imputed and features standardised - essential for KNN,
since distances would otherwise be dominated by features measured in metres.
Limitations: no training phase to speak of, but slow inference (every prediction
searches ~1M stored rows) and coarse probabilities (a fraction of k neighbours).
Expect ~20 min for the full run; use --skip-continuity to save a few minutes.

Run from the repo root:  python -m src.model_4
"""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NUMBER = 4
MODEL_NAME = "k_nearest_neighbours"
MODEL_FAMILY = "instance-based"

DEFAULT_PARAMS: dict = {"n_neighbors": 25, "weights": "distance", "p": 2}

# For bayesian_optimization.py
SEARCH_SPACE: dict = {
    "n_neighbors": ("int", 5, 100),
    "weights": ("categorical", ["uniform", "distance"]),
    "p": ("categorical", [1, 2]),
}


def params_from_search(sample: dict) -> dict:
    p = dict(DEFAULT_PARAMS)
    if "n_neighbors" in sample:
        p["n_neighbors"] = int(sample["n_neighbors"])
    if "weights" in sample:
        p["weights"] = str(sample["weights"])
    if "p" in sample:
        p["p"] = int(sample["p"])
    return p


def build_model(params: dict | None = None, random_state: int = 42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(**p, n_jobs=-1)),
    ])


if __name__ == "__main__":
    from types import SimpleNamespace
    from src.model_runner import run
    run(SimpleNamespace(**globals()))
