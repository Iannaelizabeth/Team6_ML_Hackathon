"""
model_runner.py
===============
One shared train / evaluate routine for models 3, 4 and 5, so every model is
measured in exactly the same way:

  data    : outputs/processed/{train,student_test}_labeled_r<radius>.csv  (team pipeline)
  features: evaluation.DEFAULT_FEATURES
  labels  : team nullable labels; unlabelled (ambiguous / gap / unalignable) rows excluded
  CV      : evaluation.grouped_cv_splits   (GroupKFold by trip - same folds for every model)
  metrics : evaluation.classification_metrics  + Top-1 incl. misses, candidate miss rate,
            confusion matrix, and timings
  runtime : deterministic per-trip observation cap (default 30); student-test stays complete

A model module must define:
    MODEL_NUMBER, MODEL_NAME, MODEL_FAMILY, DEFAULT_PARAMS, SEARCH_SPACE,
    build_model(params=None, random_state=42) -> sklearn estimator with predict_proba,
    params_from_search(sample) -> params
and may define:
    extra_results(model, features) -> dict          (e.g. coefficients)
    training_history(model) -> {"train": [...], "validation": [...], "ylabel": "..."} or None
        Only for models that really record per-iteration losses - the guidelines forbid
        artificial training curves.

Outputs (outputs/models/): model_<n><tag>.joblib, _results.json, _cv.csv,
_confusion_matrix.png and, if available, _training_curve.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.candidate_generation import MAX_CANDIDATES
from src.evaluation import (DEFAULT_FEATURES, classification_metrics, grouped_cv_splits,
                           observation_top1_accuracy)
from src.preprocessing import PROJECT_ROOT

RANDOM_STATE = 42
DEFAULT_RADIUS_M = 50
DEFAULT_MAX_OBSERVATIONS_PER_TRIP = 30
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
ID_COLUMNS = ["trip_id", "observation_id", "candidate_arc_id", "alignment_status", "candidate_covered", "label"]


def load_labeled(radius: int, partition: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all rows, labelled rows) of the team's labeled candidate file."""
    path = PROCESSED_DIR / f"{partition}_labeled_r{radius}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the team pipeline first "
                                f"(candidate_generation -> feature_engineering -> labeling) at r{radius}")
    all_rows = pd.read_csv(path, usecols=ID_COLUMNS + features, dtype={"trip_id": str})
    lab = all_rows.loc[pd.to_numeric(all_rows["label"], errors="coerce").notna()].reset_index(drop=True)
    lab["label"] = lab["label"].astype(int)
    return all_rows, lab


def candidate_miss_rate(df: pd.DataFrame) -> float:
    """Share of trainable observations whose reference arc is not among the candidates
    (team labelling status 'candidate_miss')."""
    obs = df.drop_duplicates(["trip_id", "observation_id"])
    trainable = obs[obs["alignment_status"].isin(["confident", "candidate_miss"])]
    return float(trainable["alignment_status"].eq("candidate_miss").mean()) if len(trainable) else float("nan")


def sample_observations_per_trip(df: pd.DataFrame, max_observations_per_trip: int | None) -> pd.DataFrame:
    """Keep all candidate rows for a deterministic sample of observations in each trip."""
    if max_observations_per_trip is None:
        return df.reset_index(drop=True)
    if max_observations_per_trip < 1:
        raise ValueError("max_observations_per_trip must be positive or None")

    sampled = []
    for seed, (_, trip) in enumerate(df.groupby("trip_id", sort=True)):
        observations = trip["observation_id"].drop_duplicates()
        count = min(max_observations_per_trip, len(observations))
        selected = observations if count == len(observations) else observations.sample(
            n=count, random_state=RANDOM_STATE + seed)
        sampled.append(trip.loc[trip["observation_id"].isin(selected)])
    return pd.concat(sampled).sort_index().reset_index(drop=True) if sampled else df.iloc[0:0].copy()


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib not installed - plots skipped)")
        return None


def save_confusion_png(cm: list, path: Path, title: str):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(np.array(cm)):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["not reference", "reference"])
    ax.set_yticks([0, 1], ["not reference", "reference"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_history_png(history: dict, path: Path, title: str):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for key in ("train", "validation"):
        if history.get(key):
            ax.plot(range(1, len(history[key]) + 1), history[key], label=key)
    ax.set_xlabel(history.get("xlabel", "iteration"))
    ax.set_ylabel(history.get("ylabel", "loss"))
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(spec) -> dict:
    """Train + evaluate the model defined by module `spec` (see module docstring)."""
    n = spec.MODEL_NUMBER
    ap = argparse.ArgumentParser(description=f"Train and evaluate model {n} ({spec.MODEL_NAME}) on the team pipeline.")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M)
    ap.add_argument("--folds", type=int, default=3, help="use the same number of folds as every other model")
    ap.add_argument("--params-json", default=None, help="tuned params from bayesian_optimization.py")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _tuned")
    ap.add_argument("--max-observations-per-trip", type=int, default=DEFAULT_MAX_OBSERVATIONS_PER_TRIP,
                    help="deterministic training observation cap per trip; use 0 for all observations")
    a = ap.parse_args()

    if a.max_observations_per_trip < 0:
        ap.error("--max-observations-per-trip must be non-negative")
    max_observations_per_trip = a.max_observations_per_trip or None
    features = list(DEFAULT_FEATURES)
    params = json.loads(Path(a.params_json).read_text()) if a.params_json else None
    full_params = {**spec.DEFAULT_PARAMS, **(params or {})}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"model_{n}{a.tag}"

    print(f"Model {n}: {spec.MODEL_NAME} ({spec.MODEL_FAMILY})  params={full_params}")
    _, train = load_labeled(a.radius, "train", features)
    full_train_rows = len(train)
    full_train_observations = len(train[["trip_id", "observation_id"]].drop_duplicates())
    train = sample_observations_per_trip(train, max_observations_per_trip)
    test_all, test = load_labeled(a.radius, "student_test", features)
    sample_description = (f", capped at {max_observations_per_trip:,} observations/trip"
                          if max_observations_per_trip else ", full dataset")
    print(f"  train: {len(train):,}/{full_train_rows:,} labelled rows "
          f"({len(train[['trip_id', 'observation_id']].drop_duplicates()):,}/"
          f"{full_train_observations:,} observations{sample_description}), "
          f"{train.trip_id.nunique()} trips, positive rate {train.label.mean():.3f}")
    print(f"  student_test: {len(test):,} labelled rows, {test.trip_id.nunique()} trips")

    # ---- grouped CV on the 80 training trips ----
    print(f"\n{a.folds}-fold trip-grouped CV:")
    X, y = train[features], train["label"].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(grouped_cv_splits(train, a.folds), 1):
        model = spec.build_model(params, RANDOM_STATE + fold)
        t0 = time.perf_counter(); model.fit(X.iloc[tr], y[tr]); train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = model.predict_proba(X.iloc[va])[:, 1]; infer_s = time.perf_counter() - t0
        m = classification_metrics(train.iloc[va], p)
        m.update(model=spec.MODEL_NAME, fold=fold, config=json.dumps(full_params, sort_keys=True, default=str),
                 features=";".join(features), train_seconds=train_s, inference_seconds=infer_s,
                 inference_seconds_per_row=infer_s / max(1, len(va)))
        rows.append(m)
        print(f"  fold {fold}: macro_f1={m['macro_f1']:.4f} top1={m['observation_top1_accuracy']:.4f} "
              f"log_loss={m['log_loss']:.4f} train={train_s:.1f}s infer={infer_s:.1f}s")
    cv = pd.DataFrame(rows)
    cv.to_csv(MODEL_DIR / f"{stem}_cv.csv", index=False)
    num = cv.select_dtypes("number").drop(columns=["fold"])
    cv_mean, cv_std = num.mean().to_dict(), num.std().to_dict()

    # ---- refit on all training trips, evaluate on student_test ----
    print("\nRefitting on all training trips, scoring student-testing trips ...")
    model = spec.build_model(params, RANDOM_STATE)
    t0 = time.perf_counter(); model.fit(X, y); train_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = model.predict_proba(test[features])[:, 1]; infer_s = time.perf_counter() - t0
    st = classification_metrics(test, p)
    st["observation_top1_including_misses"], _ = observation_top1_accuracy(test, p, include_misses=True)
    st["candidate_miss_rate"] = candidate_miss_rate(test_all)
    st["confusion_matrix"] = confusion_matrix(test["label"], (p >= 0.5).astype(int), labels=[0, 1]).tolist()
    st.update(train_seconds=train_s, inference_seconds=infer_s, inference_seconds_per_row=infer_s / max(1, len(test)))
    st = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in st.items()}

    extra = spec.extra_results(model, features) if hasattr(spec, "extra_results") else {}
    history = spec.training_history(model) if hasattr(spec, "training_history") else None
    results = {"model_number": n, "model_name": spec.MODEL_NAME, "family": spec.MODEL_FAMILY,
               "params": full_params, "random_state": RANDOM_STATE, "radius_m": a.radius,
               "max_candidates": MAX_CANDIDATES, "max_observations_per_trip": max_observations_per_trip,
               "training_rows_before_sampling": full_train_rows, "training_rows": len(train),
               "training_observations_before_sampling": full_train_observations,
               "training_observations": len(train[["trip_id", "observation_id"]].drop_duplicates()),
               "features": features, "folds": a.folds,
               "train_trips": int(train.trip_id.nunique()), "student_test_trips": int(test.trip_id.nunique()),
               "cv_mean": cv_mean, "cv_std": cv_std, "student_test": st, **extra,
               "training_history": history}
    (MODEL_DIR / f"{stem}_results.json").write_text(json.dumps(results, indent=2, default=str))
    save_confusion_png(st["confusion_matrix"], MODEL_DIR / f"{stem}_confusion_matrix.png",
                       f"Model {n} - {spec.MODEL_NAME}\nstudent-testing candidates")
    if history:
        save_history_png(history, MODEL_DIR / f"{stem}_training_curve.png",
                         f"Model {n} - {spec.MODEL_NAME} (final fit)")

    import joblib
    joblib.dump({"model": model, "model_name": spec.MODEL_NAME, "family": spec.MODEL_FAMILY,
                 "params": full_params, "feature_columns": features, "radius_m": a.radius,
                 "max_candidates": MAX_CANDIDATES, "max_observations_per_trip": max_observations_per_trip,
                 "metrics": {"cv_mean": cv_mean, "student_test": st}},
                MODEL_DIR / f"{stem}.joblib")

    print(f"\nCV mean:      macro_f1={cv_mean['macro_f1']:.4f} (+/- {cv_std['macro_f1']:.4f})  "
          f"top1={cv_mean['observation_top1_accuracy']:.4f}  log_loss={cv_mean['log_loss']:.4f}")
    print(f"Student test: macro_f1={st['macro_f1']:.4f}  top1={st['observation_top1_accuracy']:.4f} "
          f"(incl. misses {st['observation_top1_including_misses']:.4f})  log_loss={st['log_loss']:.4f}")
    print(f"              precision={st['precision']:.4f}  recall={st['recall']:.4f}  accuracy={st['accuracy']:.4f}  "
          f"candidate_miss={st['candidate_miss_rate']:.4f}")
    print(f"              train={train_s:.1f}s  inference={infer_s:.1f}s for {len(test):,} rows")
    if extra:
        print("Extra:", extra)
    print(f"Saved {stem}.joblib, {stem}_results.json, {stem}_cv.csv, and available plots to {MODEL_DIR}")
    return results
