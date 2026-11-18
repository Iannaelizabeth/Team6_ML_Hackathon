"""Trip-grouped training, evaluation, feature analysis and experiment logging."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning

from src.feature_engineering import ENGINEERED_FEATURES, ORIGINAL_FEATURES
from src.models import balanced_sample_weights, make_mlp, make_random_forest
from src.preprocessing import PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
DEFAULT_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES
IDENTIFIERS = ["trip_id", "partition", "timestamp", "observation_id", "candidate_arc_id", "label"]
MODEL_ROSTER = {
    "random_forest": "Member 2",
    "mlp": "Member 2",
    "xgboost": "Member 3",
    "rbf_svm": "Member 3",
    "logistic_regression": "Member 4",
}


def observation_top1_accuracy(frame: pd.DataFrame, probabilities, include_misses=False):
    """Observation-level ranking accuracy; misses/ambiguous labels excluded by default."""
    cols = ["trip_id", "observation_id", "label"] + (["candidate_arc_id"] if "candidate_arc_id" in frame else [])
    f = frame[cols].copy()
    f["probability"] = np.asarray(probabilities, dtype=float)
    f["label"] = pd.to_numeric(f.label, errors="coerce")
    f = f.loc[f.label.notna()]
    if f.empty: return np.nan, 0
    key = ["trip_id", "observation_id"]
    has_positive = f.groupby(key).label.transform("max").eq(1)
    if not include_misses: f = f.loc[has_positive]
    if f.empty: return np.nan, 0
    f = f.sort_values(["probability", "candidate_arc_id"] if "candidate_arc_id" in frame else ["probability"], ascending=[False, True] if "candidate_arc_id" in frame else [False])
    top = f.drop_duplicates(key, keep="first")
    return float(top.label.eq(1).mean()), int(len(top))


def classification_metrics(frame, probabilities):
    """Candidate-row and observation-rank metrics from binary probabilities."""
    y = pd.to_numeric(frame.label, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    y = y[valid].astype(int)
    p = np.asarray(probabilities, dtype=float)[valid]
    pred = (p >= 0.5).astype(int)
    top1, top1_n = observation_top1_accuracy(frame.loc[valid].copy(), p)
    return {
        "candidate_rows_evaluated": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else np.nan,
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])) if len(np.unique(y)) > 1 else np.nan,
        "observation_top1_accuracy": top1,
        "top1_observations": top1_n,
    }


def grouped_cv_splits(frame, n_splits=3):
    """Materialize deterministic GroupKFold splits with trip isolation."""
    groups = frame.trip_id.astype(str).to_numpy()
    count = min(int(n_splits), len(np.unique(groups)))
    if count < 2: raise ValueError("At least two distinct trips are required for grouped CV")
    splitter = GroupKFold(n_splits=count)
    splits = list(splitter.split(np.zeros(len(frame)), frame.label, groups))
    for train_idx, valid_idx in splits:
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise AssertionError("GroupKFold split leakage: a trip appears in both folds")
    return splits


def _fit_model(name, model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if name.lower() == "mlp":
            model.fit(x, y, classifier__sample_weight=balanced_sample_weights(y))
        else:
            model.fit(x, y)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return convergence


def cross_validate_models(frame: pd.DataFrame, features=None, n_splits=3,
                          rf_config=None, mlp_config=None, random_state=42,
                          permutation_sample=3000, models=("random_forest", "mlp"),
                          extra_models=None, fold_feature_selection=False, top_k=8):
    """Evaluate registered probability classifiers on identical trip-grouped folds."""
    features = features or DEFAULT_FEATURES
    missing = [c for c in features + ["trip_id", "label"] if c not in frame]
    if missing: raise ValueError(f"Training data missing columns: {missing}")
    data = frame.loc[pd.to_numeric(frame.label, errors="coerce").notna()].reset_index(drop=True)
    data["label"] = data.label.astype(int)
    X = data[features]
    y = data.label.to_numpy()
    groups = data.trip_id.astype(str).to_numpy()
    splits = grouped_cv_splits(data, n_splits)
    results, importances = [], []
    factories = {
        "random_forest": lambda seed: make_random_forest(rf_config, seed),
        "mlp": lambda seed: make_mlp(mlp_config, seed),
    }
    factories.update(extra_models or {})
    unknown = set(models) - set(factories)
    if unknown: raise ValueError(f"No model factories registered for: {sorted(unknown)}")
    factories = {name: factory for name, factory in factories.items() if name in models}
    fitted = {name: [] for name in factories}
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_features = list(features)
        if fold_feature_selection:
            selector = make_random_forest(rf_config, random_state + fold)
            _fit_model("random_forest", selector, X.iloc[train_idx][fold_features], y[train_idx])
            selector_imp = selector.named_steps["classifier"].feature_importances_
            rank = np.argsort(selector_imp)[::-1][:min(top_k, len(fold_features))]
            fold_features = [fold_features[i] for i in rank]
            for feature, value in zip(features, selector_imp):
                importances.append({"model": "random_forest", "fold": fold,
                                    "method": "fold_selector_train_importance", "feature": feature,
                                    "importance": float(value), "selected": feature in fold_features})
        x_train, x_valid = X.iloc[train_idx][fold_features], X.iloc[valid_idx][fold_features]
        for model_name, factory in factories.items():
            print(f"CV fold {fold}/{len(splits)}: fitting {model_name} on {len(train_idx):,} rows", flush=True)
            model = factory(random_state + fold)
            start = time.perf_counter()
            convergence = _fit_model(model_name, model, x_train, y[train_idx])
            train_seconds = time.perf_counter() - start
            print(f"CV fold {fold}/{len(splits)}: {model_name} fit in {train_seconds:.1f}s", flush=True)
            start = time.perf_counter()
            probability = model.predict_proba(x_valid)[:, 1]
            inference_seconds = time.perf_counter() - start
            metric_frame = data.iloc[valid_idx].copy()
            metrics = classification_metrics(metric_frame, probability)
            results.append({"model": model_name, "fold": fold,
                            "config": json.dumps(rf_config if model_name == "random_forest" else mlp_config, sort_keys=True, default=str),
                            "features": ";".join(fold_features), "train_trips": len(np.unique(groups[train_idx])),
                            "validation_trips": len(np.unique(groups[valid_idx])),
                            "train_seconds": train_seconds, "inference_seconds": inference_seconds,
                            "inference_seconds_per_row": inference_seconds / max(1, len(valid_idx)),
                            "convergence_warnings": " | ".join(convergence), **metrics})
            fitted[model_name].append(model)
            # RF impurity importance is computed only from the fold's training fit.
            if model_name == "random_forest":
                imps = model.named_steps["classifier"].feature_importances_
                for feature, value in zip(fold_features, imps):
                    importances.append({"model": model_name, "fold": fold, "method": "impurity_train_fold",
                                        "feature": feature, "importance": float(value)})
                # A bounded held-out permutation diagnostic. It is reported, not
                # used to fit model preprocessing or to claim unbiased selection.
                if permutation_sample > 0:
                    rng = np.random.default_rng(random_state + fold)
                    sample_idx = valid_idx if len(valid_idx) <= permutation_sample else rng.choice(valid_idx, permutation_sample, replace=False)
                    perm = permutation_importance(model, X.iloc[sample_idx][fold_features], y[sample_idx], scoring="f1_macro",
                                                  n_repeats=2, random_state=random_state + fold, n_jobs=-1)
                    for feature, mean, std in zip(features, perm.importances_mean, perm.importances_std):
                        importances.append({"model": model_name, "fold": fold, "method": "permutation_validation_sample",
                                            "feature": feature, "importance": float(mean), "std": float(std)})
    results_df, importances_df = pd.DataFrame(results), pd.DataFrame(importances)
    return results_df, importances_df, fitted


def feature_ablation(frame, feature_sets, n_splits=3, rf_config=None, random_state=42):
    """Compare predefined feature subsets on identical trip-grouped RF folds."""
    rows = []
    for set_name, features in feature_sets.items():
        result, _, _ = cross_validate_models(frame, features=features, n_splits=n_splits,
                                             rf_config=rf_config, mlp_config={}, random_state=random_state,
                                             permutation_sample=0, models=("random_forest",))
        result["feature_set"] = set_name
        rows.append(result.loc[result.model.eq("random_forest")])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def analyze_features(frame, features=None, radius=50, plot_sample=100_000, random_state=42):
    """Write training-only missingness, distributions and feature correlation diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = features or DEFAULT_FEATURES
    sample = frame[features].sample(min(plot_sample, len(frame)), random_state=random_state) if len(frame) else frame[features]
    missing = pd.DataFrame({"feature": features,
                            "missing_rows": [int(frame[f].isna().sum()) for f in features],
                            "missing_pct": [float(100*frame[f].isna().mean()) for f in features]})
    description = frame[features].describe(percentiles=[.05, .25, .5, .75, .95]).T.reset_index(names="feature")
    corr = sample.corr(method="spearman")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    missing.to_csv(REPORT_DIR / f"feature_missingness_r{radius}.csv", index=False)
    description.to_csv(REPORT_DIR / f"feature_distributions_r{radius}.csv", index=False)
    corr.to_csv(REPORT_DIR / f"feature_correlations_r{radius}.csv")
    figure_dir = PROJECT_ROOT / "outputs" / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(f"Training feature Spearman correlations (r={radius}m)")
    fig.tight_layout(); fig.savefig(figure_dir / f"feature_correlations_r{radius}.png", dpi=160); plt.close(fig)
    axes = sample.hist(figsize=(14, 10), bins=40, grid=False)
    plt.suptitle(f"Training feature distributions (sample n={len(sample):,}, r={radius}m)")
    plt.tight_layout(); plt.savefig(figure_dir / f"feature_distributions_r{radius}.png", dpi=160); plt.close("all")
    return missing, description, corr


def save_final_models(frame, features=None, rf_config=None, mlp_config=None, random_state=42, radius=50):
    """Fit and persist full-training pipelines after grouped CV completes."""
    features = features or DEFAULT_FEATURES
    data = frame.loc[pd.to_numeric(frame.label, errors="coerce").notna()].copy()
    y = data.label.astype(int).to_numpy(); x = data[features]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = {"random_forest": make_random_forest(rf_config, random_state),
              "mlp": make_mlp(mlp_config, random_state)}
    paths = {}
    import joblib
    for name, model in models.items():
        warnings_list = _fit_model(name, model, x, y)
        path = MODEL_DIR / f"{name}_r{radius}.joblib"
        joblib.dump(model, path)
        paths[name] = (path, warnings_list)
    return paths


def evaluate_student_test(model_path, test_path, features=DEFAULT_FEATURES):
    """Development-stage student-test evaluation; labels are never used in fit."""
    import joblib
    model = joblib.load(model_path)
    test = pd.read_csv(test_path, dtype={"trip_id": str})
    labeled = test.loc[pd.to_numeric(test.label, errors="coerce").notna()].copy()
    start = time.perf_counter(); p = model.predict_proba(labeled[features])[:, 1]; infer = time.perf_counter()-start
    return {"inference_seconds": infer, "inference_seconds_per_row": infer/max(1,len(labeled)), **classification_metrics(labeled, p)}
