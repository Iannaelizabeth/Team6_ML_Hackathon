"""Local FastAPI service and evaluator UI for Model 3 route matching."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.preprocessing import PROJECT_ROOT, discover_trips
from src.route_matching import match_student_trip

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
UI_PATH = PROJECT_ROOT / "static" / "evaluator.html"
app = FastAPI(title="PathSnap Route Evaluator", version="1.0.0")


def _model_path() -> Path:
    configured = os.environ.get("PATHSNAP_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PATHSNAP_MODEL_PATH does not exist: {path}")
        return path
    tuned = MODEL_DIR / "model_3_bayes_r50.joblib"
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
    except (OSError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model": artifact.get("model_name", "HistGradientBoosting"),
        "model_artifact": path.name,
        "radius_m": artifact.get("radius_m", 50),
        "decoder": "hmm_classifier",
    }


@app.get("/api/trips")
def trips() -> dict:
    try:
        trip_ids = discover_trips("student_test")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"partition": "student_test", "trips": trip_ids}


@app.post("/api/trips/{trip_id}/match")
def match_trip(trip_id: str) -> dict:
    try:
        artifact, _ = _load_artifact()
        return match_student_trip(trip_id, artifact)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
