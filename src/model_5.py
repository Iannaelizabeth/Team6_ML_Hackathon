"""
model_5.py
==========
Model 5: Logistic Regression  (family: linear / probabilistic)

Logistic regression computes a weighted sum of the features and passes it through
the sigmoid function, giving a probability between 0 and 1. It is the linear
baseline of the five models; its probabilities tend to be well calibrated, which
matters because hmm.py uses the selected model's probabilities as HMM emissions.
Limitation: it cannot learn feature interactions (e.g. "heading matters only when
the vehicle is moving") unless they are engineered explicitly.

TEAM PIPELINE - same data and metrics as every other model
-----------------------------------------------------------
  input   : outputs/processed/{train,student_test}_labeled_r50.csv
            (preprocessing -> candidate_generation -> feature_engineering -> labeling)
  features: evaluation.DEFAULT_FEATURES (3 original + 8 engineered)
  labels  : the team's nullable labels - ambiguous / gap / unalignable observations are excluded
  CV      : evaluation.grouped_cv_splits  (GroupKFold by trip, identical folds to models 1-4)
  metrics : evaluation.classification_metrics  (accuracy, precision, recall, macro F1,
            log loss, observation-level Top-1)
Added here for the report table: Top-1 including candidate misses, candidate miss rate,
confusion matrix, and train/inference time.

Partitions: 80 training trips -> fitting + grouped CV; 10 student-testing trips ->
development comparison. Evaluator-testing data is never touched.

Run from the repo root (after the team pipeline has produced the labeled CSVs):
    python -m src.model_5
Outputs in outputs/models/: model_5.joblib, model_5_results.json, model_5_cv.csv,
and, when matplotlib is installed, model_5_confusion_matrix.png. Logistic regression
has no epoch-wise training history, so no loss curve is produced (the guidelines
forbid artificial training curves).

SHARED INTERFACE for bayesian_optimization.py
---------------------------------------------
    MODEL_NAME, DEFAULT_PARAMS, SEARCH_SPACE, build_model(params), params_from_search(sample)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.candidate_generation import MAX_CANDIDATES
from src.evaluation import (DEFAULT_FEATURES, classification_metrics, grouped_cv_splits,
                            observation_top1_accuracy)
from src.preprocessing import PROJECT_ROOT

RANDOM_STATE = 42
MODEL_NAME = "logistic_regression"
MODEL_FAMILY = "linear / probabilistic"
DEFAULT_RADIUS_M = 50
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

DEFAULT_PARAMS: dict = {
    "C": 1.0,               # inverse regularisation strength: smaller C = simpler model
    "class_weight": None,   # None keeps probabilities calibrated for the HMM (see note below)
    "max_iter": 2000,
}

# Consumed by Member 3's bayesian_optimization.py (library-neutral format).
# class_weight="balanced" raises recall on the rare positive class but pushes all
# probabilities upward, so they stop being true probabilities - that hurts the HMM
# emission. Only choose it if it wins on Top-1 / route metrics, not just macro F1.
SEARCH_SPACE: dict = {
    "C": ("log-uniform", 1e-4, 1e2),
    "class_weight": ("categorical", [None, "balanced"]),
}


def params_from_search(sample: dict) -> dict:
    p = dict(DEFAULT_PARAMS)
    if "C" in sample:
        p["C"] = float(sample["C"])
    if "class_weight" in sample:
        cw = sample["class_weight"]
        p["class_weight"] = None if cw in (None, "None", "none") else cw
    return p


def build_model(params: dict | None = None, random_state: int = RANDOM_STATE) -> Pipeline:
    """Median imputation + scaling + L2 logistic regression (lbfgs).
    Imputation and scaling are fitted inside the pipeline, i.e. on training folds only.
    Scaling matters because regularisation penalises all coefficients equally.
    `penalty` is not passed: it is deprecated in scikit-learn >= 1.8 and L2 is the default."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"],
                                          solver="lbfgs", random_state=random_state)),
    ])


def coefficients(model: Pipeline, features: list[str]) -> dict:
    """Standardised coefficients, largest effect first. Positive = more likely the reference arc."""
    coef = model.named_steps["classifier"].coef_[0]
    order = np.argsort(-np.abs(coef))
    return {features[i]: round(float(coef[i]), 4) for i in order}


def _load_labeled(path: Path, features: list[str]) -> pd.DataFrame:
    cols = ["trip_id", "observation_id", "candidate_arc_id", "alignment_status", "candidate_covered", "label", *features]
    df = pd.read_csv(path, usecols=cols, dtype={"trip_id": str})
    return df


def _labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[pd.to_numeric(df["label"], errors="coerce").notna()].reset_index(drop=True)
    d["label"] = d["label"].astype(int)
    return d


def candidate_miss_rate(df: pd.DataFrame) -> float:
    """Share of trainable observations (confident or candidate_miss) whose reference arc
    is not in the candidate set - the team labelling marks these 'candidate_miss'."""
    obs = df.drop_duplicates(["trip_id", "observation_id"])
    trainable = obs[obs["alignment_status"].isin(["confident", "candidate_miss"])]
    return float(trainable["alignment_status"].eq("candidate_miss").mean()) if len(trainable) else float("nan")


def _save_confusion_png(cm: list, path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - confusion matrix saved in JSON only)")
        return False
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
    return True


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate model 5 (Logistic Regression) on the team pipeline.")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M)
    ap.add_argument("--folds", type=int, default=3, help="use the same number of folds as models 1-4")
    ap.add_argument("--params-json", default=None, help="tuned params from bayesian_optimization.py")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _tuned")
    a = ap.parse_args()

    features = list(DEFAULT_FEATURES)
    params = json.loads(Path(a.params_json).read_text()) if a.params_json else None
    full_params = {**DEFAULT_PARAMS, **(params or {})}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / f"train_labeled_r{a.radius}.csv"
    test_path = PROCESSED_DIR / f"student_test_labeled_r{a.radius}.csv"
    for pth in (train_path, test_path):
        if not pth.exists():
            raise FileNotFoundError(f"{pth} not found - run the team pipeline first "
                                    f"(candidate_generation -> feature_engineering -> labeling) at r{a.radius}")

    print(f"Loading {train_path.name} and {test_path.name} ...")
    train_all, test_all = _load_labeled(train_path, features), _load_labeled(test_path, features)
    train, test = _labelled_rows(train_all), _labelled_rows(test_all)
    print(f"  train: {len(train):,} labelled candidate rows, {train.trip_id.nunique()} trips, "
          f"positive rate {train.label.mean():.3f}")
    print(f"  student_test: {len(test):,} labelled candidate rows, {test.trip_id.nunique()} trips")

    # ---- grouped CV on the 80 training trips (same splitter as the team) ----
    print(f"\n{a.folds}-fold trip-grouped CV:")
    X, y = train[features], train["label"].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(grouped_cv_splits(train, a.folds), 1):
        model = build_model(params, RANDOM_STATE + fold)
        t0 = time.perf_counter(); model.fit(X.iloc[tr], y[tr]); train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = model.predict_proba(X.iloc[va])[:, 1]; infer_s = time.perf_counter() - t0
        m = classification_metrics(train.iloc[va], p)
        m.update(model=MODEL_NAME, fold=fold, config=json.dumps(full_params, sort_keys=True, default=str),
                 features=";".join(features), train_seconds=train_s, inference_seconds=infer_s,
                 inference_seconds_per_row=infer_s / max(1, len(va)))
        rows.append(m)
        print(f"  fold {fold}: macro_f1={m['macro_f1']:.4f} top1={m['observation_top1_accuracy']:.4f} "
              f"log_loss={m['log_loss']:.4f} train={train_s:.1f}s")
    cv = pd.DataFrame(rows)
    cv.to_csv(MODEL_DIR / f"model_5{a.tag}_cv.csv", index=False)
    num = cv.select_dtypes("number").drop(columns=["fold"])
    cv_mean, cv_std = num.mean().to_dict(), num.std().to_dict()

    # ---- refit on all training trips, evaluate on the student-testing trips ----
    print("\nRefitting on all training trips, scoring student-testing trips ...")
    model = build_model(params, RANDOM_STATE)
    t0 = time.perf_counter(); model.fit(X, y); train_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = model.predict_proba(test[features])[:, 1]; infer_s = time.perf_counter() - t0
    st = classification_metrics(test, p)
    st["observation_top1_including_misses"], _ = observation_top1_accuracy(test, p, include_misses=True)
    st["candidate_miss_rate"] = candidate_miss_rate(test_all)
    st["confusion_matrix"] = confusion_matrix(test["label"], (p >= 0.5).astype(int), labels=[0, 1]).tolist()
    st.update(train_seconds=train_s, inference_seconds=infer_s, inference_seconds_per_row=infer_s / max(1, len(test)))
    st = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in st.items()}

    coefs = coefficients(model, features)
    results = {"model_name": MODEL_NAME, "family": MODEL_FAMILY, "params": full_params,
               "random_state": RANDOM_STATE, "radius_m": a.radius, "max_candidates": MAX_CANDIDATES,
               "features": features, "folds": a.folds,
               "train_trips": int(train.trip_id.nunique()), "student_test_trips": int(test.trip_id.nunique()),
               "cv_mean": cv_mean, "cv_std": cv_std, "student_test": st, "coefficients": coefs}
    (MODEL_DIR / f"model_5{a.tag}_results.json").write_text(json.dumps(results, indent=2, default=str))
    confusion_png_saved = _save_confusion_png(
        st["confusion_matrix"], MODEL_DIR / f"model_5{a.tag}_confusion_matrix.png",
        "Model 5 - Logistic Regression\nstudent-testing candidates")

    import joblib
    joblib.dump({"model": model, "model_name": MODEL_NAME, "family": MODEL_FAMILY, "params": full_params,
                 "feature_columns": features, "radius_m": a.radius, "max_candidates": MAX_CANDIDATES,
                 "metrics": {"cv_mean": cv_mean, "student_test": st}},
                MODEL_DIR / f"model_5{a.tag}.joblib")

    print(f"\nCV mean:      macro_f1={cv_mean['macro_f1']:.4f} (+/- {cv_std['macro_f1']:.4f})  "
          f"top1={cv_mean['observation_top1_accuracy']:.4f}  log_loss={cv_mean['log_loss']:.4f}")
    print(f"Student test: macro_f1={st['macro_f1']:.4f}  top1={st['observation_top1_accuracy']:.4f} "
          f"(incl. misses {st['observation_top1_including_misses']:.4f})  log_loss={st['log_loss']:.4f}")
    print(f"              precision={st['precision']:.4f}  recall={st['recall']:.4f}  accuracy={st['accuracy']:.4f}  "
          f"candidate_miss={st['candidate_miss_rate']:.4f}")
    print(f"              train={train_s:.1f}s  inference={infer_s * 1e3:.0f} ms for {len(test):,} rows")
    print("Coefficients (standardised, largest effect first):", coefs)
    saved_files = [f"model_5{a.tag}.joblib", f"model_5{a.tag}_results.json", f"model_5{a.tag}_cv.csv"]
    if confusion_png_saved:
        saved_files.append(f"model_5{a.tag}_confusion_matrix.png")
    print(f"Saved {', '.join(saved_files)} to {MODEL_DIR}")


if __name__ == "__main__":
    main()