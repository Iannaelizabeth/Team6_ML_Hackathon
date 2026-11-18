"""Inference adapter joining team candidate features to HMM/Viterbi decoding."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.candidate_generation import (
    MAX_CANDIDATES,
    build_segment_index,
    create_local_transformer,
    find_candidates,
    project_coordinates,
)
from src.data_contract import Trip, load_trip as load_decoder_trip, trip_from_points
from src.feature_engineering import (
    ENGINEERED_FEATURES,
    ORIGINAL_FEATURES,
    engineer_trip_candidates,
)
from src.hmm import HMMConfig, MatchResult, RoadGraph, predict_candidate_proba, run_decoder
from src.preprocessing import DATA_ROOT, PROJECT_ROOT, discover_trips, load_trip, preprocess_track

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
HMM_CONFIG_PATH = MODEL_DIR / "hmm_config.json"


@dataclass
class PreparedMatch:
    partition: str
    trip_id: str
    trip: Trip
    features: pd.DataFrame
    decoder_candidates: pd.DataFrame
    probabilities: np.ndarray
    artifact: dict
    config: HMMConfig
    graph: RoadGraph


def load_hmm_config() -> HMMConfig:
    """Load calibrated HMM values if available, otherwise use decoder defaults."""
    if not HMM_CONFIG_PATH.is_file():
        return HMMConfig()
    with HMM_CONFIG_PATH.open(encoding="utf-8") as file:
        return HMMConfig.from_dict(json.load(file))


def _generate_candidates(partition: str, trip_id: str, raw: dict,
                          clean_track: pd.DataFrame, radius: int) -> pd.DataFrame:
    """Use the team spatial candidate functions without passing reference routes."""
    if "route" in raw:
        raise ValueError("Reference route must be removed before candidate generation")

    transformer = create_local_transformer(raw["nodes"])
    segment_index = build_segment_index(raw["nodes"], raw["arcs"], transformer)
    gps_x, gps_y = project_coordinates(
        clean_track.longitude.to_numpy(dtype=float),
        clean_track.latitude.to_numpy(dtype=float),
        transformer,
    )
    gps_x, gps_y = np.asarray(gps_x), np.asarray(gps_y)
    valid_gps = np.isfinite(gps_x) & np.isfinite(gps_y)
    rows = []
    for position, (_, observation) in enumerate(clean_track.iterrows()):
        candidates = []
        if valid_gps[position]:
            candidates = find_candidates(
                gps_x[position],
                gps_y[position],
                observation["observation_id"],
                segment_index,
                radius,
                MAX_CANDIDATES,
            )
        rows.extend({
            "trip_id": str(trip_id),
            "partition": partition,
            "timestamp": float(observation["timestamp"]),
            **candidate,
        } for candidate in candidates)

    columns = [
        "trip_id",
        "partition",
        "timestamp",
        "observation_id",
        "candidate_arc_id",
        "perpendicular_distance",
        "projection_ratio",
        "road_length",
    ]
    return pd.DataFrame(rows, columns=columns)


def prepare_match(partition: str, trip_id: str, artifact: dict) -> PreparedMatch:
    """Build prediction inputs with reference routes removed from the pipeline."""
    if trip_id not in set(discover_trips(partition)):
        raise KeyError(f"Unknown trip {trip_id!r} in partition {partition!r}")

    radius = int(artifact.get("radius_m", 50))
    loaded_data = load_trip(partition, trip_id)
    prediction_data = {key: value for key, value in loaded_data.items() if key != "route"}
    del loaded_data
    if "route" in prediction_data:
        raise AssertionError("Reference route leaked into prediction inputs")

    clean_track, _ = preprocess_track(prediction_data["track"])
    candidates = _generate_candidates(
        partition, trip_id, prediction_data, clean_track, radius
    )
    if candidates.empty:
        features = pd.DataFrame(columns=[
            *candidates.columns,
            *ORIGINAL_FEATURES,
            *ENGINEERED_FEATURES,
        ])
    else:
        features = engineer_trip_candidates(
            candidates,
            clean_track,
            prediction_data["nodes"],
            prediction_data["arcs"],
        )

    base_trip = load_decoder_trip(DATA_ROOT / partition / trip_id, trip_id)
    base_trip.route = None
    trip = trip_from_points(
        base_trip,
        [
            {"lat": row.latitude, "lon": row.longitude, "t": row.timestamp}
            for row in clean_track.itertuples(index=False)
        ],
    )
    trip.route = None

    if features.empty:
        decoder_candidates = pd.DataFrame(columns=[
            "point_idx", "arc", "frac_along", "dist_m", "proj_x", "proj_y",
        ])
    else:
        observation_positions = {
            int(observation_id): position
            for position, observation_id in enumerate(clean_track.observation_id.to_numpy())
        }
        point_idx = features.observation_id.map(observation_positions)
        if point_idx.isna().any():
            raise ValueError(f"Candidate observations do not align with trip {trip_id}")

        arc = pd.to_numeric(features.candidate_arc_id, errors="raise").to_numpy(dtype=int)
        if ((arc < 0) | (arc >= len(trip.arcs))).any():
            raise ValueError(f"Candidate arc IDs are outside the network for trip {trip_id}")
        fraction = pd.to_numeric(features.projection_ratio, errors="raise").to_numpy(dtype=float)
        endpoints = trip.arcs[arc]
        start = trip.nodes_xy[endpoints[:, 0]]
        end = trip.nodes_xy[endpoints[:, 1]]
        projected = start + fraction[:, None] * (end - start)
        decoder_candidates = pd.DataFrame({
            "point_idx": point_idx.to_numpy(dtype=int),
            "arc": arc,
            "frac_along": fraction,
            "dist_m": pd.to_numeric(
                features.perpendicular_distance, errors="raise"
            ).to_numpy(dtype=float),
            "proj_x": projected[:, 0],
            "proj_y": projected[:, 1],
        })

    probabilities = predict_candidate_proba(artifact, features)
    config = load_hmm_config()
    graph = RoadGraph(trip, config.sp_limit_m)
    return PreparedMatch(
        partition=partition,
        trip_id=trip_id,
        trip=trip,
        features=features,
        decoder_candidates=decoder_candidates,
        probabilities=probabilities,
        artifact=artifact,
        config=config,
        graph=graph,
    )


def decode_prepared(prepared: PreparedMatch, decoder: str = "hmm_classifier") -> MatchResult:
    """Decode prepared inputs; `prepared.trip.route` is always None."""
    if prepared.trip.route is not None:
        raise ValueError("Decoder trip unexpectedly contains a reference route")
    return run_decoder(
        decoder,
        prepared.trip,
        prepared.decoder_candidates,
        prepared.config,
        prepared.artifact,
        graph=prepared.graph,
        proba=prepared.probabilities,
    )


def _road_network_segments(trip, margin_deg: float = 0.0015,
                           max_segments: int = 50000) -> list[list[list[float]]]:
    """Road-network arcs near the GPS track as [[lon, lat], [lon, lat]] pairs.

    Map geometry only (no reference route), used as the UI background layer.
    """
    lat = trip.track.lat.to_numpy(dtype=float)
    lon = trip.track.lon.to_numpy(dtype=float)
    if len(lat) == 0 or len(trip.arcs) == 0:
        return []
    lat_min, lat_max = np.nanmin(lat) - margin_deg, np.nanmax(lat) + margin_deg
    lon_min, lon_max = np.nanmin(lon) - margin_deg, np.nanmax(lon) + margin_deg
    nodes = np.asarray(trip.nodes_ll, dtype=float)  # (N, 2) lat, lon
    arcs = np.asarray(trip.arcs, dtype=int)
    start, end = nodes[arcs[:, 0]], nodes[arcs[:, 1]]

    def inside(points):
        return ((points[:, 0] >= lat_min) & (points[:, 0] <= lat_max)
                & (points[:, 1] >= lon_min) & (points[:, 1] <= lon_max))

    keep = np.flatnonzero(inside(start) | inside(end))[:max_segments]
    return [
        [[round(float(start[i, 1]), 7), round(float(start[i, 0]), 7)],
         [round(float(end[i, 1]), 7), round(float(end[i, 0]), 7)]]
        for i in keep
    ]


def match_trip(partition: str, trip_id: str, artifact: dict,
               decoder: str = "hmm_classifier") -> dict:
    """Match and evaluate one trip, returning predictions and metrics only."""
    prepared = prepare_match(partition, trip_id, artifact)
    result = decode_prepared(prepared, decoder)

    from src.route_evaluation import evaluate_prediction

    evaluation = evaluate_prediction(
        partition,
        trip_id,
        result,
        prepared.decoder_candidates,
        prepared.trip,
    )
    matched = result.matched
    gps_points = [
        [float(lon), float(lat)]
        for lon, lat in zip(prepared.trip.track.lon, prepared.trip.track.lat)
    ]
    snapped_points = [
        [float(row.snap_lon), float(row.snap_lat)]
        for row in matched.itertuples(index=False)
    ]
    matched_arcs = [
        {"point_index": int(row.point_idx), "arc_id": int(row.arc)}
        for row in matched.itertuples(index=False)
    ]
    route_coordinates: list[list[float]] = []
    for arc_id in result.route_arcs:
        latlon = prepared.trip.arc_latlon(int(arc_id))
        route_coordinates.extend([[float(lon), float(lat)] for lat, lon in latlon])

    road_network = _road_network_segments(prepared.trip)

    return {
        "trip_id": trip_id,
        "partition": partition,
        "decoder": result.decoder,
        "road_network": road_network,
        "gps_points": gps_points,
        "snapped_points": snapped_points,
        "matched_arcs": matched_arcs,
        "route_coordinates": route_coordinates,
        "route_arc_count": len(result.route_arcs),
        "matched_points": len(matched),
        "unmatched_points": result.n_unmatched,
        "sequence_breaks": result.n_breaks,
        "unfilled_route_gaps": result.gaps_unfilled,
        "runtime_ms": result.runtime_ms,
        "evaluation": evaluation,
    }


def match_student_trip(trip_id: str, artifact: dict) -> dict:
    """Backward-compatible wrapper for the student-test partition."""
    return match_trip("student_test", trip_id, artifact)
