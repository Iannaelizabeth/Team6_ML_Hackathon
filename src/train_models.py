"""Run the Member 2 trip-grouped RF/MLP training and experiment workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation import (DEFAULT_FEATURES, REPORT_DIR, analyze_features,
                            cross_validate_models, feature_ablation, save_final_models)
from src.feature_engineering import ORIGINAL_FEATURES, ENGINEERED_FEATURES
from src.preprocessing import PROJECT_ROOT


def run_training(radius=50, n_splits=3, random_state=42, run_ablation=True,
                 rf_config=None, mlp_config=None, run_parameter_sweep=False):
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    feature_dir = PROJECT_ROOT / "outputs" / "processed"
    model_dir = PROJECT_ROOT / "outputs" / "models"
    train_path = feature_dir / f"train_labeled_r{radius}.csv"
    if not train_path.exists(): raise FileNotFoundError(f"Run feature engineering and labeling first: {train_path}")
    train = pd.read_csv(train_path, dtype={"trip_id": str})
    train = train.loc[pd.to_numeric(train.label, errors="coerce").notna()].copy()
    if train.empty or train.label.nunique() < 2: raise ValueError("No usable binary training labels")
    all_features_path = feature_dir / f"train_features_r{radius}.csv"
    feature_only = pd.read_csv(all_features_path, usecols=DEFAULT_FEATURES)
    analyze_features(feature_only, DEFAULT_FEATURES, radius=radius, random_state=random_state)
    rf_config = rf_config or {"n_estimators": 80, "max_depth": 24, "min_samples_leaf": 2,
                              "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}
    mlp_config = mlp_config or {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "batch_size": 512,
                                "max_iter": 20, "early_stopping": False}
    fold_results, importances, _ = cross_validate_models(train, DEFAULT_FEATURES, n_splits,
                                                          rf_config, mlp_config, random_state)
    report_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    full_fold_path = report_dir / f"model_cv_full_feature_folds_r{radius}.csv"
    fold_path = report_dir / f"model_cv_folds_r{radius}.csv"
    importance_path = report_dir / f"feature_importance_r{radius}.csv"
    fold_results.to_csv(full_fold_path, index=False)
    ablation_path = None
    ablation_best_set = None
    selection_path = report_dir / f"feature_selection_r{radius}.json"
    if run_ablation:
        subsets = {
            "original_only": ORIGINAL_FEATURES,
            "original_plus_motion": ORIGINAL_FEATURES + ["gps_heading", "road_heading", "heading_difference", "gps_speed", "time_difference"],
            "original_plus_context": ORIGINAL_FEATURES + ["distance_to_previous", "distance_to_next", "connectivity_score"],
            "all_features": DEFAULT_FEATURES,
        }
        # Reuse full-feature CV RF folds above; evaluate the three reduced sets.
        reduced = {k: v for k, v in subsets.items() if k != "all_features"}
        ablation = feature_ablation(train, reduced, n_splits=n_splits, rf_config=rf_config, random_state=random_state)
        full = fold_results.loc[fold_results.model.eq("random_forest")].copy(); full["feature_set"] = "all_features"
        ablation = pd.concat([ablation, full], ignore_index=True)
        ablation_path = report_dir / f"feature_ablation_r{radius}.csv"; ablation.to_csv(ablation_path, index=False)
        selection_summary = ablation.groupby("feature_set", as_index=False).agg(
            macro_f1_mean=("macro_f1", "mean"), top1_accuracy_mean=("observation_top1_accuracy", "mean"))
        ablation_best_set = selection_summary.sort_values(["macro_f1_mean", "top1_accuracy_mean"], ascending=False).iloc[0].feature_set

    # Fit the feature ranker on each outer training fold, select top-k there,
    # then score both member models on that fold's untouched trip groups.
    selected_fold_results, selector_importances, _ = cross_validate_models(
        train, DEFAULT_FEATURES, n_splits, rf_config, mlp_config, random_state,
        permutation_sample=0, fold_feature_selection=True, top_k=8)
    fs = selector_importances.loc[selector_importances.method.eq("fold_selector_train_importance")].copy()
    feature_rank = fs.groupby("feature", as_index=False).agg(
        selected_folds=("selected", "sum"), mean_train_importance=("importance", "mean"))
    feature_rank = feature_rank.sort_values(["selected_folds", "mean_train_importance"], ascending=False)
    selected_features = feature_rank.head(8).feature.tolist()
    selector_importances.to_csv(report_dir / f"fold_feature_selection_r{radius}.csv", index=False)
    importances = pd.concat([importances, selector_importances], ignore_index=True, sort=False)
    importances.to_csv(importance_path, index=False)
    selection_reason = "Top eight RF importance-ranked features were selected independently inside each training fold; retained features for the final full-training fit are the most frequently selected, then ranked by mean training-fold importance. The outer-fold scores use fold-specific selections."
    selection = {"radius_m": radius, "selected_feature_set": "fold_ranked_top_8",
                 "ablation_best_feature_set": ablation_best_set,
                 "retained_features": selected_features, "reason": selection_reason}
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    config = {"radius_m": radius, "n_splits": n_splits, "random_state": random_state,
              "features": list(selected_features), "selected_feature_set": "fold_ranked_top_8",
              "random_forest": rf_config, "mlp": mlp_config,
              "training_rows": len(train), "training_trips": int(train.trip_id.nunique())}
    (report_dir / f"model_config_r{radius}.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    selected_fold_results.to_csv(fold_path, index=False)
    summary = selected_fold_results.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        log_loss_mean=("log_loss", "mean"), top1_accuracy_mean=("observation_top1_accuracy", "mean"),
        train_seconds_sum=("train_seconds", "sum"), inference_seconds_sum=("inference_seconds", "sum"))
    summary_path = report_dir / f"model_cv_summary_r{radius}.csv"; summary.to_csv(summary_path, index=False)

    sweep_path = None
    if run_parameter_sweep:
        alternatives = [
            ("random_forest", {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 4,
                                "min_samples_split": 4, "max_features": 0.7,
                                "class_weight": "balanced", "n_jobs": -1}),
            ("mlp", {"hidden_layer_sizes": (96, 48), "activation": "tanh", "alpha": 1e-3,
                     "learning_rate_init": 5e-4, "batch_size": 512, "max_iter": 20,
                     "early_stopping": False}),
        ]
        sweep_rows = []
        for name, config_alt in alternatives:
            opts = {"rf_config": config_alt if name == "random_forest" else rf_config,
                    "mlp_config": config_alt if name == "mlp" else mlp_config,
                    "models": (name,), "permutation_sample": 0}
            swept, _, _ = cross_validate_models(train, DEFAULT_FEATURES, n_splits,
                random_state=random_state, fold_feature_selection=True, top_k=8, **opts)
            swept["experiment"] = f"{name}_alternate_config"
            sweep_rows.append(swept)
        sweep = pd.concat(sweep_rows, ignore_index=True)
        sweep_path = report_dir / f"model_parameter_sweep_r{radius}.csv"; sweep.to_csv(sweep_path, index=False)

    fitted_paths = save_final_models(train, selected_features, rf_config, mlp_config, random_state, radius)
    fig_dir = PROJECT_ROOT / "outputs" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    ax = summary.set_index("model")[["macro_f1_mean", "top1_accuracy_mean"]].plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    ax.set(title=f"Trip-grouped model comparison (r={radius}m)", ylabel="Mean fold score", xlabel="Model")
    ax.grid(axis="y", alpha=.25); plt.tight_layout(); plt.savefig(fig_dir / f"model_comparison_r{radius}.png", dpi=160); plt.close()
    if not importances.empty:
        imp = importances.loc[importances.method.eq("permutation_validation_sample")].groupby("feature", as_index=False).importance.mean().sort_values("importance")
        ax = imp.plot.barh(x="feature", y="importance", legend=False, figsize=(8, 6))
        ax.set(title=f"RF permutation importance (r={radius}m)", xlabel="Mean decrease in held-out macro F1", ylabel="Feature")
        plt.tight_layout(); plt.savefig(fig_dir / f"feature_importance_r{radius}.png", dpi=160); plt.close()
    return {"folds": fold_path, "full_feature_folds": full_fold_path,
            "summary": summary_path, "importance": importance_path,
            "ablation": ablation_path, "feature_selection": selection_path, "parameter_sweep": sweep_path,
            "models": {k: str(v[0]) for k,v in fitted_paths.items()}, "metrics": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--sweep", action="store_true", help="Also run alternate RF and MLP hyperparameter configurations")
    args = p.parse_args()
    result = run_training(args.radius, args.folds, args.seed, not args.skip_ablation,
                          run_parameter_sweep=args.sweep)
    print(result["metrics"].to_string(index=False)); print("Artifacts:", {k:v for k,v in result.items() if k != "metrics"})


if __name__ == "__main__": main()
