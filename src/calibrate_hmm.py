"""Calibrate HMM emission and transition scales from confident train alignments."""
from __future__ import annotations

import json
import math

import numpy as np

from src.data_contract import load_trip as load_decoder_trip
from src.hmm import RoadGraph
from src.labeling import align_track_to_route
from src.preprocessing import DATA_ROOT, PROJECT_ROOT, discover_trips, load_trip, preprocess_track


def calibrate() -> dict:
    sigma_samples: list[float] = []
    transition_samples: list[float] = []
    trips_used: list[str] = []
    for trip_id in discover_trips("train"):
        raw = load_trip("train", trip_id)
        clean_track, _ = preprocess_track(raw["track"])
        alignment = align_track_to_route(
            clean_track, raw["nodes"], raw["arcs"], raw["route"]
        )
        confident = alignment.loc[
            alignment.alignment_status.eq("confident")
        ].sort_values(["timestamp", "observation_id"], kind="stable")
        if confident.empty:
            continue

        trip = load_decoder_trip(DATA_ROOT / "train" / trip_id, trip_id)
        trip.route = None
        graph = RoadGraph(trip)
        position_by_observation = {
            int(observation_id): position
            for position, observation_id in enumerate(clean_track.observation_id.to_numpy())
        }
        point_ids = []
        arcs = []
        offsets = []
        for row in confident.itertuples(index=False):
            position = position_by_observation.get(int(row.observation_id))
            arc = int(row.aligned_route_arc_id)
            distance = float(row.route_distance_m)
            if position is None or not np.isfinite(distance):
                continue
            point = trip.track.iloc[position]
            start_node, end_node = trip.arcs[arc]
            start = trip.nodes_xy[start_node]
            end = trip.nodes_xy[end_node]
            vector = end - start
            length_squared = float(np.dot(vector, vector))
            fraction = float(np.clip(
                np.dot(np.array([point.x, point.y]) - start, vector)
                / max(length_squared, 1e-9),
                0.0,
                1.0,
            ))
            point_ids.append(position)
            arcs.append(arc)
            offsets.append(fraction * trip.arc_len[arc])
            sigma_samples.append(distance)

        for idx in range(len(point_ids) - 1):
            first, second = point_ids[idx], point_ids[idx + 1]
            route_distance = graph.route_distance(
                arcs[idx], offsets[idx], arcs[idx + 1], offsets[idx + 1]
            )
            if not np.isfinite(route_distance):
                continue
            point_a = trip.track.iloc[first]
            point_b = trip.track.iloc[second]
            gps_distance = math.hypot(point_b.x - point_a.x, point_b.y - point_a.y)
            transition_samples.append(abs(route_distance - gps_distance))
        if point_ids:
            trips_used.append(trip_id)

    if not sigma_samples:
        raise ValueError("No confident training alignments are available for calibration")
    if not transition_samples:
        raise ValueError("No reachable consecutive confident training pairs for beta calibration")

    sigma_z = 1.4826 * float(np.median(sigma_samples))
    beta = float(np.median(transition_samples)) / math.log(2)
    if not np.isfinite(sigma_z) or sigma_z <= 0:
        raise ValueError(f"Calibrated sigma_z must be positive; got {sigma_z}")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"Calibrated beta must be positive; got {beta}")

    output = {
        "sigma_z": sigma_z,
        "beta": beta,
        "trip_ids_used": trips_used,
        "confident_observations": len(sigma_samples),
        "transition_pairs": len(transition_samples),
    }
    output_path = PROJECT_ROOT / "outputs" / "models" / "hmm_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    result = calibrate()
    print(json.dumps(result, indent=2))
    print(f"Saved {PROJECT_ROOT / 'outputs' / 'models' / 'hmm_config.json'}")


if __name__ == "__main__":
    main()
