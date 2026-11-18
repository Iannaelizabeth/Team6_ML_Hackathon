import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src import api, route_evaluation
from src.data_contract import Trip
from src.hmm import HMMConfig, MatchResult, RoadGraph, run_decoder
from src.route_evaluation import _route_continuity
from src.route_matching import PreparedMatch


EVALUATION_KEYS = {
    "pointwise_accuracy",
    "candidate_miss_rate",
    "route_precision",
    "route_recall",
    "route_f1",
    "route_mismatch_fraction",
    "route_continuity",
    "n_discontinuities",
    "evaluable_observations",
}


def _synthetic_trip() -> Trip:
    return Trip(
        trip_id="synthetic",
        track=pd.DataFrame({
            "lat": [0.0, 0.0], "lon": [0.0, 0.001], "t": [0.0, 1.0],
            "x": [0.0, 10.0], "y": [0.0, 0.0],
        }),
        nodes_ll=np.array([[0.0, 0.0], [0.0, 0.001], [0.0, 0.002], [0.0, 0.003]]),
        nodes_xy=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
        arcs=np.array([[0, 1], [1, 2], [2, 3]]),
        route=None,
        origin=(0.0, 0.0),
        arc_len=np.array([10.0, 10.0, 10.0]),
        arc_bearing=np.array([90.0, 90.0, 90.0]),
    )


def test_evaluate_prediction_scores_only_confident_points_and_returns_metrics(monkeypatch):
    trip = _synthetic_trip()
    raw = {
        "track": pd.DataFrame({
            "longitude": [0.0, 0.001], "latitude": [0.0, 0.0],
            "timestamp": [0.0, 1.0],
        }),
        "nodes": pd.DataFrame({"longitude": [0.0] * 4, "latitude": [0.0] * 4}),
        "arcs": pd.DataFrame({
            "from_node_id": [0, 1, 2], "to_node_id": [1, 2, 3],
        }),
        "route": pd.DataFrame({"arc_id": [0, 1, 2]}),
    }
    monkeypatch.setattr(route_evaluation, "load_trip", lambda *_: raw)
    monkeypatch.setattr(
        route_evaluation,
        "align_track_to_route",
        lambda *_: pd.DataFrame({
            "observation_id": [0, 1],
            "alignment_status": ["confident", "confident"],
            "aligned_route_arc_id": [0, 1],
            "route_distance_m": [1.0, 2.0],
        }),
    )
    result = MatchResult(
        "hmm_classifier",
        pd.DataFrame({"point_idx": [0, 1], "arc": [0, 1]}),
        [0, 1, 2],
        2, 0, 0, 0, 1.0,
    )
    candidates = pd.DataFrame({"point_idx": [0, 1], "arc": [0, 1]})

    metrics = route_evaluation.evaluate_prediction(
        "student_test", "synthetic", result, candidates, trip
    )

    assert set(metrics) == EVALUATION_KEYS
    assert metrics["pointwise_accuracy"] == 1.0
    assert metrics["candidate_miss_rate"] == 0.0
    assert metrics["route_f1"] == 1.0
    assert metrics["route_continuity"] == 1.0
    assert metrics["n_discontinuities"] == 0
    assert metrics["evaluable_observations"] == 2


def test_match_response_has_metrics_without_reference_fields(monkeypatch):
    evaluation = {
        "pointwise_accuracy": 1.0,
        "candidate_miss_rate": 0.0,
        "route_precision": 1.0,
        "route_recall": 1.0,
        "route_f1": 1.0,
        "route_mismatch_fraction": 0.0,
        "route_continuity": 1.0,
        "n_discontinuities": 0,
        "evaluable_observations": 2,
    }
    response = {
        "trip_id": "synthetic",
        "decoder": "hmm_classifier",
        "gps_points": [],
        "evaluation": evaluation,
    }
    calls = {}

    def fake_match(partition, trip_id, artifact, decoder="hmm_classifier"):
        calls.update(partition=partition, trip_id=trip_id, artifact=artifact)
        return response

    monkeypatch.setenv("PATHSNAP_EVAL_PARTITION", "evaluator_test")
    monkeypatch.setattr(api, "_load_artifact", lambda: ({}, api.MODEL_DIR / "fake.joblib"))
    monkeypatch.setattr(api, "match_partition_trip", fake_match)

    result = TestClient(api.app).post("/api/trips/synthetic/match")

    assert result.status_code == 200
    payload = result.json()
    assert set(payload["evaluation"]) == EVALUATION_KEYS
    forbidden = {"route", "reference_route", "aligned_route_arc_id", "reference_geometry"}
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(payload["evaluation"])
    assert calls["partition"] == "evaluator_test"
    assert calls["trip_id"] == "synthetic"


def test_eval_partition_changes_trip_listing(monkeypatch):
    monkeypatch.setattr(api, "discover_trips", lambda partition: [partition + "_trip"])
    client = TestClient(api.app)
    for partition in ("student_test", "evaluator_test"):
        monkeypatch.setenv("PATHSNAP_EVAL_PARTITION", partition)
        response = client.get("/api/trips")
        assert response.status_code == 200
        assert response.json() == {
            "partition": partition,
            "trips": [partition + "_trip"],
        }


def test_decoder_accepts_trip_without_route():
    trip = _synthetic_trip()
    candidates = pd.DataFrame({
        "point_idx": [0], "arc": [0], "frac_along": [0.0],
        "dist_m": [0.0], "proj_x": [0.0], "proj_y": [0.0],
    })

    result = run_decoder("hmm_gaussian", trip, candidates, HMMConfig())

    assert trip.route is None
    assert result.matched["arc"].tolist() == [0]


def test_route_continuity_uses_observation_predictions_not_gap_filled_route():
    trip = Trip(
        trip_id="discontinuous",
        track=pd.DataFrame({
            "lat": [0.0, 0.0], "lon": [0.0, 0.001], "t": [0.0, 1.0],
            "x": [0.0, 80.0], "y": [0.0, 0.0],
        }),
        nodes_ll=np.array([[0.0, 0.0], [0.0, 0.0001], [0.0, 0.0007], [0.0, 0.0008]]),
        nodes_xy=np.array([[0.0, 0.0], [10.0, 0.0], [70.0, 0.0], [80.0, 0.0]]),
        arcs=np.array([[0, 1], [1, 2], [2, 3]]),
        route=None,
        origin=(0.0, 0.0),
        arc_len=np.array([10.0, 60.0, 10.0]),
        arc_bearing=np.array([90.0, 90.0, 90.0]),
    )
    matched = pd.DataFrame({"point_idx": [1, 0], "arc": [2, 0]})

    continuity, discontinuities = _route_continuity(matched, RoadGraph(trip))

    assert continuity == 0.0
    assert discontinuities == 1


def test_match_trip_serializes_prediction_without_reference_fields(monkeypatch):
    trip = _synthetic_trip()
    prepared = PreparedMatch(
        partition="student_test",
        trip_id="synthetic",
        trip=trip,
        features=pd.DataFrame(),
        decoder_candidates=pd.DataFrame(),
        probabilities=np.array([]),
        artifact={},
        config=HMMConfig(),
        graph=RoadGraph(trip),
    )
    result = MatchResult(
        "hmm_classifier",
        pd.DataFrame({
            "point_idx": [0],
            "arc": [0],
            "snap_lon": [0.0],
            "snap_lat": [0.0],
        }),
        [0],
        2, 1, 0, 0, 1.0,
    )
    evaluation = {key: None for key in EVALUATION_KEYS}
    evaluation["evaluable_observations"] = 0
    monkeypatch.setattr("src.route_matching.prepare_match", lambda *_: prepared)
    monkeypatch.setattr("src.route_matching.decode_prepared", lambda *_: result)
    monkeypatch.setattr(route_evaluation, "evaluate_prediction", lambda *_: evaluation)

    from src.route_matching import match_trip

    payload = match_trip("student_test", "synthetic", {})

    assert set(payload["evaluation"]) == EVALUATION_KEYS
    assert {
        "route", "reference_route", "aligned_route_arc_id", "reference_geometry",
    }.isdisjoint(payload)
