"""Bayesian hyperparameter search for Model 3 (HistGradientBoosting)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from src.evaluation import DEFAULT_FEATURES, classification_metrics, grouped_cv_splits
from src.model_3 import build_model
from src.model_runner import DEFAULT_MAX_OBSERVATIONS_PER_TRIP, load_labeled, sample_observations_per_trip
from src.preprocessing import PROJECT_ROOT

RANDOM_STATE = 42
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def optimize(radius: int = 50, folds: int = 3, trials: int = 6,
             max_observations_per_trip: int = DEFAULT_MAX_OBSERVATIONS_PER_TRIP,
             seed: int = RANDOM_STATE) -> dict:
    """Optimize observation-level Top-1 using trip-grouped cross-validation."""
    if trials < 4:
        raise ValueError("trials must be at least 4 for a useful TPE search")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if max_observations_per_trip < 1:
        raise ValueError("max_observations_per_trip must be positive")

    features = list(DEFAULT_FEATURES)
    _, train = load_labeled(radius, "train", features)
    train = sample_observations_per_trip(train, max_observations_per_trip)
    if train["label"].nunique() < 2:
        raise ValueError("Training sample must contain both label classes")
    splits = grouped_cv_splits(train, folds)
    X, y = train[features], train["label"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 40, 140, step=20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        }
        top1_scores, macro_f1_scores, log_losses = [], [], []
        for fold, (tr, va) in enumerate(splits, 1):
            model = build_model(params, seed + fold)
            model.fit(X.iloc[tr], y[tr])
            probabilities = model.predict_proba(X.iloc[va])[:, 1]
            metrics = classification_metrics(train.iloc[va], probabilities)
            top1_scores.append(metrics["observation_top1_accuracy"])
            macro_f1_scores.append(metrics["macro_f1"])
            log_losses.append(metrics["log_loss"])
        trial.set_user_attr("cv_top1", float(pd.Series(top1_scores).mean()))
        trial.set_user_attr("cv_macro_f1", float(pd.Series(macro_f1_scores).mean()))
        trial.set_user_attr("cv_log_loss", float(pd.Series(log_losses).mean()))
        return trial.user_attrs["cv_top1"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3),
        study_name="model_3_hist_gradient_boosting",
    )
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"model_3_bayes_r{radius}"
    best = {
        "best_value_top1": float(study.best_value),
        "best_params": study.best_params,
        "radius_m": radius,
        "folds": len(splits),
        "trials": trials,
        "max_observations_per_trip": max_observations_per_trip,
        "training_rows": len(train),
        "training_trips": int(train["trip_id"].nunique()),
        "seed": seed,
    }
    (MODEL_DIR / f"{stem}_best_params.json").write_text(json.dumps(study.best_params, indent=2))
    (MODEL_DIR / f"{stem}_results.json").write_text(json.dumps(best, indent=2))
    study.trials_dataframe().to_csv(MODEL_DIR / f"{stem}_trials.csv", index=False)
    print(f"Best CV Top-1: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Saved {stem}_best_params.json, {stem}_results.json, and {stem}_trials.csv to {MODEL_DIR}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=50, choices=[30, 50, 75])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--max-observations-per-trip", type=int, default=DEFAULT_MAX_OBSERVATIONS_PER_TRIP)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    optimize(args.radius, args.folds, args.trials, args.max_observations_per_trip, args.seed)


if __name__ == "__main__":
    main()
