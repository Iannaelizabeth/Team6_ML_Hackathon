"""Inference adapter joining the team feature pipeline to HMM/Viterbi decoding."""
from __future__ import annotations

import pandas as pd

from src.candidate_generation import generate_trip_candidates
from src.data_contract import load_trip as load_decoder_trip
from src.data_contract import trip_from_points
from src.feature_engineering import engineer_trip_candidates
from src.hmm import HMMConfig, predict_candidate_proba, run_decoder
from src.preprocessing import DATA_ROOT, discover_trips, load_trip, preprocess_track


def match_student_trip(trip_id: str, artifact: dict) -> dict:
    """Match one known student-test trip without returning or using its reference route."""
    available = set(discover_trips("student_test"))
    if trip_id not in available:
        raise KeyError(f"Unknown student-test trip: {trip_id}")

    radius = int(artifact.get("radius_m", 50))
    raw = load_trip("student_test", trip_id)
    clean_track, _ = preprocess_track(raw["track"])
    candidates, _ = generate_trip_candidates("student_test", trip_id, radius)
    if candidates.empty:
        raise ValueError(f"No road candidates were generated for trip {trip_id}")
    features = engineer_trip_candidates(candidates, clean_track, raw["nodes"], raw["arcs"])

    base = load_decoder_trip(DATA_ROOT / "student_test" / trip_id, trip_id)
    trip = trip_from_points(
        base,
        [
            {"lat": row.latitude, "lon": row.longitude, "t": row.timestamp}
            for row in clean_track.itertuples(index=False)
        ],
    )
    trip.route = None

    observation_positions = {
        int(observation_id): position
        for position, observation_id in enumerate(clean_track.observation_id.to_numpy())
    }
    point_idx = features.observation_id.map(observation_positions)
    if point_idx.isna().any():
        raise ValueError("Candidate features contain observations absent from the cleaned track")

    arc = pd.to_numeric(features.candidate_arc_id, errors="raise").to_numpy(dtype=int)
    fraction = pd.to_numeric(features.projection_ratio, errors="raise").to_numpy(dtype=float)
    endpoints = trip.arcs[arc]
    start = trip.nodes_xy[endpoints[:, 0]]
    end = trip.nodes_xy[endpoints[:, 1]]
    projected = start + fraction[:, None] * (end - start)
    decoder_candidates = pd.DataFrame({
        "point_idx": point_idx.to_numpy(dtype=int),
        "arc": arc,
        "frac_along": fraction,
        "dist_m": pd.to_numeric(features.perpendicular_distance, errors="raise").to_numpy(dtype=float),
        "proj_x": projected[:, 0],
        "proj_y": projected[:, 1],
    })

    probabilities = predict_candidate_proba(artifact, features)
    result = run_decoder(
        "hmm_classifier",
        trip,
        decoder_candidates,
        HMMConfig(),
        artifact,
        proba=probabilities,
    )
    matched = result.matched
    gps_points = [
        [float(lon), float(lat)]
        for lon, lat in zip(trip.track.lon, trip.track.lat)
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
        latlon = trip.arc_latlon(int(arc_id))
        route_coordinates.extend([[float(lon), float(lat)] for lat, lon in latlon])

    return {
        "trip_id": trip_id,
        "decoder": result.decoder,
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
    }
