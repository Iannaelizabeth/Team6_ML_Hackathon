"""Compare tuned and untuned Model 3 with identical grouped folds and data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.evaluation import DEFAULT_FEATURES, classification_metrics, grouped_cv_splits
from src.model_3 import DEFAULT_PARAMS, build_model
from src.model_runner import load_labeled, sample_observations_per_trip
from src.preprocessing import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def _cross_validate(
    train: pd.DataFrame,
    features: list[str],
    params: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    X, y = train[features], train["label"].to_numpy()
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        model = build_model(params, seed + fold)
        fit_started = time.perf_counter()
        model.fit(X.iloc[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
        inference_seconds = time.perf_counter() - inference_started
        metrics = classification_metrics(train.iloc[valid_idx], probabilities)
        rows.append({
            "fold": fold,
            **metrics,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "n_iterations_used": int(model.n_iter_),
        })
    fold_results = pd.DataFrame(rows)
    summary = {
        "cv_macro_f1": float(fold_results["macro_f1"].mean()),
        "cv_observation_top1_accuracy": float(
            fold_results["observation_top1_accuracy"].mean()
        ),
        "cv_log_loss": float(fold_results["log_loss"].mean()),
        "cv_fit_seconds": float(fold_results["fit_seconds"].mean()),
        "cv_inference_seconds": float(fold_results["inference_seconds"].mean()),
        "cv_mean_iterations_used": float(fold_results["n_iterations_used"].mean()),
    }
    return fold_results, summary


def _fit_and_score_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    params: dict,
    seed: int,
) -> dict:
    model = build_model(params, seed)
    model.fit(train[features], train["label"].to_numpy())
    started = time.perf_counter()
    probabilities = model.predict_proba(test[features])[:, 1]
    inference_seconds = time.perf_counter() - started
    metrics = classification_metrics(test, probabilities)
    return {
        "student_test_observation_top1_accuracy": metrics["observation_top1_accuracy"],
        "student_test_macro_f1": metrics["macro_f1"],
        "student_test_log_loss": metrics["log_loss"],
        "inference_seconds": inference_seconds,
        "refit_iterations_used": int(model.n_iter_),
    }


def _save_comparison_chart(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Cross-validation", "Student-test"]
    models = ["untuned", "tuned"]
    metric_pairs = [
        ("cv_observation_top1_accuracy", "student_test_observation_top1_accuracy",
         "Observation Top-1"),
        ("cv_macro_f1", "student_test_macro_f1", "Macro F1"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(labels))
    width = 0.34
    colors = {"untuned": "#50718f", "tuned": "#d2543d"}
    for ax, (cv_metric, test_metric, title) in zip(axes, metric_pairs):
        for offset, model_name in enumerate(models):
            row = summary.loc[summary["model"].eq(model_name)].iloc[0]
            values = [row[cv_metric], row[test_metric]]
            ax.bar(x + (offset - 0.5) * width, values, width,
                   label=model_name.title(), color=colors[model_name])
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compare(
    radius: int = 50,
    folds: int = 3,
    max_observations_per_trip: int = 300,
    seed: int = 42,
    params_path: Path | None = None,
) -> pd.DataFrame:
    if max_observations_per_trip < 1:
        raise ValueError("max_observations_per_trip must be positive")
    params_path = params_path or MODEL_DIR / "model_3_bayes_r50_best_params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"Tuned parameters not found: {params_path}")
    tuned_params = json.loads(params_path.read_text(encoding="utf-8"))
    if "max_iter" in tuned_params:
        raise ValueError(f"{params_path} must not specify max_iter; use early stopping")

    features = list(DEFAULT_FEATURES)
    _, train = load_labeled(radius, "train", features)
    train = sample_observations_per_trip(train, max_observations_per_trip)
    _, test = load_labeled(radius, "student_test", features)
    splits = grouped_cv_splits(train, folds)

    model_params = {"untuned": dict(DEFAULT_PARAMS), "tuned": tuned_params}
    cv_by_model: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for model_name, params in model_params.items():
        folds_df, cv_summary = _cross_validate(train, features, params, splits, seed)
        cv_by_model[model_name] = folds_df
        test_summary = _fit_and_score_test(train, test, features, params, seed)
        summary_rows.append({
            "record_type": "comparison",
            "model": model_name,
            "seed": seed,
            **cv_summary,
            **test_summary,
        })

    untuned_seed_scores = []
    sensitivity_rows = []
    for sensitivity_seed in range(seed, seed + 5):
        if sensitivity_seed == seed:
            folds_df = cv_by_model["untuned"]
        else:
            folds_df, _ = _cross_validate(
                train, features, model_params["untuned"], splits, sensitivity_seed
            )
        score = float(folds_df["observation_top1_accuracy"].mean())
        untuned_seed_scores.append(score)
        sensitivity_rows.append({
            "record_type": "untuned_seed_sensitivity",
            "model": "untuned",
            "seed": sensitivity_seed,
            "cv_macro_f1": float(folds_df["macro_f1"].mean()),
            "cv_observation_top1_accuracy": score,
            "cv_log_loss": float(folds_df["log_loss"].mean()),
            "cv_mean_iterations_used": float(folds_df["n_iterations_used"].mean()),
        })

    seed_spread = max(untuned_seed_scores) - min(untuned_seed_scores)
    summary = pd.DataFrame(summary_rows)
    summary["untuned_seed_cv_top1_spread"] = np.nan
    summary.loc[summary["model"].eq("untuned"), "untuned_seed_cv_top1_spread"] = seed_spread
    seed_sensitivity = pd.DataFrame(sensitivity_rows)
    seed_sensitivity["untuned_seed_cv_top1_spread"] = seed_spread
    output = pd.concat([summary, seed_sensitivity], ignore_index=True, sort=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MODEL_DIR / "model_3_tuning_comparison.csv"
    chart_path = MODEL_DIR / "model_3_tuning_comparison.png"
    output.to_csv(csv_path, index=False)
    _save_comparison_chart(summary, chart_path)

    table_columns = [
        "model",
        "cv_macro_f1",
        "cv_observation_top1_accuracy",
        "cv_log_loss",
        "student_test_observation_top1_accuracy",
        "student_test_macro_f1",
        "student_test_log_loss",
        "inference_seconds",
        "cv_mean_iterations_used",
        "refit_iterations_used",
        "untuned_seed_cv_top1_spread",
    ]
    print(summary[table_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "Untuned CV Top-1 by seed: "
        + ", ".join(f"{score:.4f}" for score in untuned_seed_scores)
        + f" (spread={seed_spread:.4f})"
    )
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison chart: {chart_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=50, choices=[30, 50, 75])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-observations-per-trip", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params-json", type=Path, default=None)
    args = parser.parse_args()
    compare(args.radius, args.folds, args.max_observations_per_trip, args.seed, args.params_json)


if __name__ == "__main__":
    main()
