"""Compare all route decoders on the configured evaluator trips."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import PROJECT_ROOT, discover_trips
from src.route_evaluation import evaluate_prediction
from src.route_matching import decode_prepared, prepare_match


def _selected_model_path() -> Path:
    configured = os.environ.get("PATHSNAP_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PATHSNAP_MODEL_PATH does not exist: {path}")
        return path
    tuned = PROJECT_ROOT / "outputs" / "models" / "model_3_tuned.joblib"
    baseline = PROJECT_ROOT / "outputs" / "models" / "model_3.joblib"
    path = tuned if tuned.is_file() else baseline
    if not path.is_file():
        raise FileNotFoundError(f"No Model 3 artifact found at {tuned} or {baseline}")
    return path


def compare_decoders(partition: str = "student_test") -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = _selected_model_path()
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(f"Model artifact has an unsupported format: {model_path}")

    decoders = ("classifier_only", "hmm_gaussian", "hmm_classifier")
    rows = []
    trip_ids = discover_trips(partition)
    for trip_id in trip_ids:
        prepared = prepare_match(partition, trip_id, artifact)
        for decoder in decoders:
            result = decode_prepared(prepared, decoder)
            metrics = evaluate_prediction(
                partition,
                trip_id,
                result,
                prepared.decoder_candidates,
                prepared.trip,
            )
            rows.append({
                "trip_id": trip_id,
                "decoder": decoder,
                **metrics,
                "runtime_ms": result.runtime_ms,
                "boosting_iterations": getattr(artifact["model"], "n_iter_", None),
            })

    per_trip = pd.DataFrame(rows)
    metrics_to_average = [
        "pointwise_accuracy",
        "candidate_miss_rate",
        "route_precision",
        "route_recall",
        "route_f1",
        "route_mismatch_fraction",
        "route_continuity",
        "n_discontinuities",
        "evaluable_observations",
        "runtime_ms",
        "boosting_iterations",
    ]
    summary = per_trip.groupby("decoder", sort=False)[metrics_to_average].mean().reset_index()
    output_dir = PROJECT_ROOT / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_trip.to_csv(output_dir / "decoder_comparison_per_trip.csv", index=False)
    summary.to_csv(output_dir / "decoder_comparison_summary.csv", index=False)

    chart_metrics = ["pointwise_accuracy", "route_f1", "route_continuity"]
    axes = summary.set_index("decoder")[chart_metrics].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1),
        ylabel="Mean score", title=f"Decoder comparison ({partition})",
        rot=0,
    )
    axes.legend(["Pointwise accuracy", "Route F1", "Route continuity"], loc="lower right")
    axes.figure.tight_layout()
    axes.figure.savefig(output_dir / "decoder_comparison.png", dpi=160)
    plt.close(axes.figure)
    print(f"Model: {model_path}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return per_trip, summary


if __name__ == "__main__":
    partition = os.environ.get("PATHSNAP_EVAL_PARTITION", "student_test")
    compare_decoders(partition)
