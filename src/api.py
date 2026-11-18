"""Local FastAPI service and evaluator UI for Model 3 route matching."""
from __future__ import annotations

import os
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.preprocessing import PROJECT_ROOT, discover_trips
from src.route_matching import load_hmm_config, match_trip as match_partition_trip

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
UI_PATH = PROJECT_ROOT / "static" / "evaluator.html"
app = FastAPI(title="PathSnap Route Evaluator", version="1.0.0")
logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not any(getattr(handler, "_pathsnap_stdout", False) for handler in logger.handlers):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler._pathsnap_stdout = True
    stdout_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    logger.addHandler(stdout_handler)


def _eval_partition() -> str:
    return os.environ.get("PATHSNAP_EVAL_PARTITION", "student_test")


def _model_path() -> Path:
    configured = os.environ.get("PATHSNAP_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PATHSNAP_MODEL_PATH does not exist: {path}")
        return path
    tuned = MODEL_DIR / "model_3_tuned.joblib"
    baseline = MODEL_DIR / "model_3.joblib"
    path = tuned if tuned.is_file() else baseline
    if not path.is_file():
        raise FileNotFoundError(f"No Model 3 artifact found at {tuned} or {baseline}")
    return path


@lru_cache(maxsize=1)
def _load_artifact() -> tuple[dict, Path]:
    path = _model_path()
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(f"Model artifact has an unsupported format: {path}")
    return artifact, path


@app.get("/", include_in_schema=False)
def evaluator_ui() -> FileResponse:
    if not UI_PATH.is_file():
        raise HTTPException(status_code=500, detail="Evaluator UI file is missing")
    return FileResponse(UI_PATH, media_type="text/html")


@app.get("/api/health")
def health() -> dict:
    try:
        artifact, path = _load_artifact()
        partition = _eval_partition()
        trip_count = len(discover_trips(partition))
        config = load_hmm_config()
    except (OSError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model": artifact.get("model_name", "HistGradientBoosting"),
        "model_artifact": path.name,
        "radius_m": artifact.get("radius_m", 50),
        "decoder": "hmm_classifier",
        "partition": partition,
        "trip_count": trip_count,
        "sigma_z": config.sigma_z,
        "beta": config.beta,
    }


@app.get("/api/trips")
def trips() -> dict:
    partition = _eval_partition()
    try:
        trip_ids = discover_trips(partition)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"partition": partition, "trips": trip_ids}


@app.post("/api/trips/{trip_id}/match")
def match_trip(trip_id: str) -> dict:
    partition = _eval_partition()
    started = time.perf_counter()
    try:
        artifact, _ = _load_artifact()
        result = match_partition_trip(partition, trip_id, artifact)
        evaluation = result["evaluation"]
        logger.info(
            "match trip_id=%s partition=%s decoder=%s observations=%d "
            "pointwise_accuracy=%s route_f1=%s route_continuity=%s runtime_ms=%.1f",
            trip_id,
            partition,
            result["decoder"],
            len(result["gps_points"]),
            evaluation["pointwise_accuracy"],
            evaluation["route_f1"],
            evaluation["route_continuity"],
            (time.perf_counter() - started) * 1000,
        )
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.exception(
            "match failed trip_id=%s partition=%s runtime_ms=%.1f",
            trip_id, partition, (time.perf_counter() - started) * 1000,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
