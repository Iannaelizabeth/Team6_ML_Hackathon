"""Run development-only student-test evaluation after training is finished."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation import DEFAULT_FEATURES, evaluate_student_test
from src.labeling import label_candidate_file
from src.preprocessing import PROJECT_ROOT


def evaluate_models(radius=50):
    """Label public student-test trips, score saved models, and persist metrics.

    This function must be run only after model configurations are fixed. It
    never fits, selects, or tunes models on student-test rows.
    """
    label_path, diagnostics = label_candidate_file("student_test", radius)
    model_dir = PROJECT_ROOT / "outputs" / "models"
    selection_path = PROJECT_ROOT / "outputs" / "reports" / f"feature_selection_r{radius}.json"
    features = DEFAULT_FEATURES
    if selection_path.exists(): features = json.loads(selection_path.read_text(encoding="utf-8")).get("retained_features", features)
    result_rows = []
    for name in ["random_forest", "mlp"]:
        model_path = model_dir / f"{name}_r{radius}.joblib"
        if not model_path.exists(): raise FileNotFoundError(f"Train and save the model first: {model_path}")
        result_rows.append({"model": name, "radius_m": radius,
                            **evaluate_student_test(model_path, label_path, features)})
    output = PROJECT_ROOT / "outputs" / "reports" / f"student_test_model_metrics_r{radius}.csv"
    pd.DataFrame(result_rows).to_csv(output, index=False)
    return label_path, diagnostics, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    args = p.parse_args()
    label_path, diagnostics, output = evaluate_models(args.radius)
    print(json.dumps({"labeled_student_test": str(label_path), "diagnostics": str(diagnostics),
                      "metrics": str(output)}, indent=2))


if __name__ == "__main__": main()
