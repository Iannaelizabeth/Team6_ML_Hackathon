"""Sequence-aware reference-route alignment and candidate binary labels.

    Route arc IDs come from `.route` in chronological order. GPS observations are
aligned by a forward, monotone sequence matcher over nearby route arcs, with
spatial distances and temporal/geodesic progression costs. Ambiguous or unalignable observations
are retained in diagnostics and receive nullable labels in candidate exports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer

from src.feature_engineering import _trip_groups
from src.preprocessing import DATA_ROOT, PROJECT_ROOT, load_trip, preprocess_track

LABEL_DIR = PROJECT_ROOT / "outputs" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
GEOD = Geod(ellps="WGS84")


def _route_geometry(track, nodes, arcs, route):
    node_lon = nodes.longitude.to_numpy(dtype=float)
    node_lat = nodes.latitude.to_numpy(dtype=float)
    valid_coords = np.isfinite(node_lon) & np.isfinite(node_lat) & (np.abs(node_lon) <= 180) & (np.abs(node_lat) <= 90)
    all_lon = np.r_[node_lon[valid_coords], track.longitude.to_numpy(dtype=float)]
    all_lat = np.r_[node_lat[valid_coords], track.latitude.to_numpy(dtype=float)]
    lon0, lat0 = np.nanmean(all_lon), np.nanmean(all_lat)
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    nx, ny = transformer.transform(node_lon, node_lat)
    nx, ny = np.asarray(nx), np.asarray(ny)
    route_ids = pd.to_numeric(route.arc_id, errors="coerce").to_numpy(dtype=float)
    valid_route = np.isfinite(route_ids) & (route_ids % 1 == 0) & (route_ids >= 0) & (route_ids < len(arcs))
    positions = np.flatnonzero(valid_route)
    aids = route_ids[positions].astype(int)
    endpoints = arcs.iloc[aids][["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    valid_end = np.isfinite(endpoints).all(axis=1) & (endpoints % 1 == 0).all(axis=1)
    valid_end &= (endpoints[:, 0] >= 0) & (endpoints[:, 0] < len(nodes)) & (endpoints[:, 1] >= 0) & (endpoints[:, 1] < len(nodes))
    positions, aids, endpoints = positions[valid_end], aids[valid_end], endpoints[valid_end].astype(int)
    fn, tn = endpoints.T
    x1, y1, x2, y2 = nx[fn], ny[fn], nx[tn], ny[tn]
    good = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
    positions, aids, fn, tn = positions[good], aids[good], fn[good], tn[good]
    x1, y1, x2, y2 = x1[good], y1[good], x2[good], y2[good]
    lengths = np.hypot(x2-x1, y2-y1)
    cumulative = np.r_[0.0, np.cumsum(lengths[:-1])]
    broken = np.r_[False, tn[:-1] != fn[1:]] if len(aids) else np.array([], dtype=bool)
    return transformer, positions, aids, fn, tn, x1, y1, x2, y2, lengths, cumulative, broken


def align_track_to_route(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                         route: pd.DataFrame, max_distance_m: float = 75.0,
                         ambiguity_margin_m: float = 5.0, gps_gap_seconds: float = 60.0,
                         states_per_observation: int = 12) -> pd.DataFrame:
    """Align each valid GPS fix to route order, never by row-index equality.

    At each fix, choose among the nearest route segments that do not move
    backward from the previous selected route position. The score combines
    point-to-directed-segment distance and the difference between expected
    route-chainage advance and GPS geodesic displacement. This is a forward
    greedy sequence matcher, not an HMM or Viterbi decoder. Disconnected route
    arcs and competing choices are exposed in per-observation status fields.
    """
    if track.empty or route.empty:
        return pd.DataFrame(columns=["observation_id", "aligned_route_arc_id", "aligned_route_position",
                                     "route_distance_m", "alignment_status", "alignment_margin_m"])
    tr = track.copy()
    if "observation_id" not in tr: tr["observation_id"] = tr.index
    tr = tr.sort_values(["timestamp", "observation_id"], kind="stable").reset_index(drop=True)
    geom = _route_geometry(tr, nodes, arcs, route)
    transformer, route_pos, aids, from_ids, to_ids, x1, y1, x2, y2, lengths, cumulative, broken = geom
    if not len(aids):
        raise ValueError("Reference route has no valid directed arc geometry")
    lon, lat = tr.longitude.to_numpy(dtype=float), tr.latitude.to_numpy(dtype=float)
    gx, gy = transformer.transform(lon, lat); gx, gy = np.asarray(gx), np.asarray(gy)
    dx, dy = x2-x1, y2-y1
    l2 = np.maximum(dx*dx + dy*dy, 1e-9)
    ts = tr.timestamp.to_numpy(dtype=float)
    n = len(tr)
    chosen, selected_dist, selected_ratio = np.zeros(n, dtype=int), np.full(n, np.nan), np.full(n, np.nan)
    candidate_states, candidate_distances, candidate_ratios, candidate_scores = [], [], [], []
    route_order_violations = np.zeros(n, dtype=bool)
    previous_state, previous_chain = None, None
    for t in range(n):
        ratio = np.clip(((gx[t]-x1)*dx + (gy[t]-y1)*dy) / l2, 0.0, 1.0)
        px, py = x1 + ratio*dx, y1 + ratio*dy
        dist = np.hypot(gx[t]-px, gy[t]-py)
        k = min(states_per_observation, len(dist))
        ids = np.argpartition(dist, k-1)[:k] if k < len(dist) else np.arange(len(dist))
        ids = ids[np.argsort(dist[ids])]
        if previous_state is None:
            scores = dist[ids] / 20.0
            allowed = np.arange(len(ids))
        else:
            allowed = np.flatnonzero(ids >= previous_state)
            if not len(allowed):
                allowed = np.arange(len(ids))
                route_order_violations[t] = True
            _, _, displacement = GEOD.inv(lon[t-1], lat[t-1], lon[t], lat[t])
            dt = ts[t] - ts[t-1]
            scale = max(20.0, 5.0 * max(dt, 1.0))
            chainage = cumulative[ids] + ratio[ids]*lengths[ids]
            delta = chainage - previous_chain
            progress_cost = np.abs(delta - displacement) / scale
            broken_cost = np.zeros(len(ids))
            for j, cur in enumerate(ids):
                if cur > previous_state and broken[previous_state+1:cur+1].any(): broken_cost[j] = 2.0
            scores = dist[ids]/20.0 + progress_cost + broken_cost
        local = allowed[int(np.argmin(scores[allowed]))]
        chosen[t] = ids[local]; selected_dist[t] = dist[ids[local]]; selected_ratio[t] = ratio[ids[local]]
        candidate_states.append(ids); candidate_distances.append(dist[ids]); candidate_ratios.append(ratio[ids]); candidate_scores.append(scores)
        previous_state = chosen[t]
        previous_chain = cumulative[chosen[t]] + selected_ratio[t]*lengths[chosen[t]]
    selected_aid = aids[chosen]
    selected_route_position = route_pos[chosen]
    chain = cumulative[chosen] + selected_ratio * lengths[chosen]

    output = []
    for t in range(n):
        statuses = []
        if not np.isfinite([lon[t], lat[t], ts[t]]).all(): statuses.append("invalid_timestamp_or_coordinate")
        if selected_dist[t] > max_distance_m: statuses.append("unalignable_spatial")
        if t and ts[t] - ts[t-1] > gps_gap_seconds: statuses.append("gps_gap")
        if route_order_violations[t]: statuses.append("route_order_violation")
        if t and chosen[t] > chosen[t-1]:
            lo, hi = chosen[t-1], chosen[t]
            if hi < len(broken) and broken[lo+1:hi+1].any(): statuses.append("route_discontinuity")
        alternatives = []
        allowed_floor = chosen[t-1] if t else 0
        for j, state_id in enumerate(candidate_states[t]):
            if (state_id >= allowed_floor and aids[state_id] != selected_aid[t]
                    and candidate_distances[t][j] <= selected_dist[t] + ambiguity_margin_m
                    and candidate_scores[t][j] <= candidate_scores[t][int(np.where(candidate_states[t] == chosen[t])[0][0])] + 1.0):
                alternatives.append(float(candidate_distances[t][j]))
        margin = min(alternatives) - selected_dist[t] if alternatives else np.inf
        if alternatives: statuses.append("ambiguous_reference")
        if not statuses: statuses = ["confident"]
        # Gap, broken topology and competing arcs all prevent a positive label.
        status = "|".join(dict.fromkeys(statuses))
        output.append({
            "observation_id": tr.iloc[t].observation_id,
            "alignment_method": "forward_monotone_v1",
            "timestamp": ts[t], "aligned_route_arc_id": int(selected_aid[t]),
            "aligned_route_position": int(selected_route_position[t]),
            "route_distance_m": float(selected_dist[t]),
            "aligned_route_chainage_m": float(chain[t]),
            "alignment_margin_m": float(margin) if np.isfinite(margin) else np.nan,
            "alignment_status": status,
        })
    return pd.DataFrame(output)


def label_candidate_trip(candidates: pd.DataFrame, alignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach binary labels while retaining ambiguous/missed status explicitly."""
    if candidates.duplicated(["trip_id", "observation_id", "candidate_arc_id"]).any():
        raise ValueError("Duplicate candidate key trip_id+observation_id+candidate_arc_id")
    joined = candidates.merge(alignment, on=["observation_id", "timestamp"], how="left", validate="many_to_one", suffixes=("", "_alignment"))
    matched = joined.candidate_arc_id.eq(joined.aligned_route_arc_id)
    joined["candidate_covered"] = matched.groupby(joined.observation_id).transform("any")
    status = joined.alignment_status.fillna("unalignable_spatial")
    confident = status.eq("confident")
    candidate_miss = confident & ~joined.candidate_covered
    joined.loc[candidate_miss, "alignment_status"] = "candidate_miss"
    # Only confident associations and explicit route candidate misses are
    # trainable. Ambiguous/gap/discontinuous observations receive nullable labels.
    eligible = confident | candidate_miss
    labels = np.where(joined.candidate_arc_id.eq(joined.aligned_route_arc_id), 1, 0)
    joined["label"] = pd.array(np.where(eligible, labels, None), dtype="Int8")
    obs_diag = alignment.copy()
    present = joined.groupby("observation_id").candidate_covered.first()
    obs_diag["candidate_covered"] = obs_diag.observation_id.map(present).fillna(False).astype(bool)
    obs_diag.loc[obs_diag.alignment_status.eq("confident") & ~obs_diag.candidate_covered, "alignment_status"] = "candidate_miss"
    return joined, obs_diag


def label_candidate_file(partition="train", radius=50, input_path: Path | None = None,
                         output_path: Path | None = None) -> tuple[Path, Path]:
    """Sequence-align and label candidate rows tripwise; writes reports and CSV."""
    if input_path is None:
        input_path = LABEL_DIR / f"{partition}_features_r{radius}.csv"
        if not input_path.exists(): input_path = LABEL_DIR / f"{partition}_candidates_r{radius}.csv"
        if not input_path.exists(): input_path = PROJECT_ROOT / "data" / "candidates" / f"{partition}_candidates_r{radius}.csv"
    output_path = output_path or LABEL_DIR / f"{partition}_labeled_r{radius}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists(): output_path.unlink()
    report_rows, obs_frames = [], []
    for trip_id, cand in _trip_groups(input_path):
        cache_path = LABEL_DIR / f"{partition}_{trip_id}_route_alignment_forward.csv"
        if cache_path.exists():
            alignment = pd.read_csv(cache_path)
        else:
            raw = load_trip(partition, trip_id)
            clean, _ = preprocess_track(raw["track"])
            alignment = align_track_to_route(clean, raw["nodes"], raw["arcs"], raw["route"])
            alignment.to_csv(cache_path, index=False)
        labelled, obs_diag = label_candidate_trip(cand, alignment)
        obs_diag.insert(0, "trip_id", trip_id); obs_diag.insert(1, "partition", partition); obs_diag["radius_m"] = radius
        obs_frames.append(obs_diag)
        labelled.to_csv(output_path, mode="a", index=False, header=not output_path.exists())
        counts = obs_diag.alignment_status.value_counts()
        coverage_eligible = obs_diag.alignment_status.isin(["confident", "candidate_miss"])
        covered_count = int((coverage_eligible & obs_diag.candidate_covered).sum())
        coverage_denominator = int(coverage_eligible.sum())
        positive_observations = int(labelled.loc[labelled.label.eq(1), "observation_id"].nunique())
        report_rows.append({"trip_id": trip_id, "partition": partition, "radius_m": radius,
                            "gps_observations": len(alignment), "candidate_rows": len(cand),
                            "confident_observations": int(counts.get("confident", 0)),
                            "candidate_misses": int(counts.get("candidate_miss", 0)),
                            "positive_observations": positive_observations,
                            "candidate_covered_observations": covered_count,
                            "coverage_eligible_observations": coverage_denominator,
                            "labeled_candidate_rows": int(labelled.label.notna().sum()),
                            "ambiguous_candidate_rows": int(labelled.label.isna().sum()),
                            "ambiguous_observations": int(obs_diag.alignment_status.str.contains("ambiguous|gps_gap|discontinuity|order_violation", regex=True).sum()),
                            "unalignable_observations": int(obs_diag.alignment_status.str.contains("unalignable|invalid", regex=True).sum()),
                            "candidate_coverage_pct": float(100 * covered_count / coverage_denominator) if coverage_denominator else np.nan})
    report = pd.DataFrame(report_rows)
    diag_path = REPORT_DIR / f"{partition}_alignment_diagnostics_r{radius}.csv"
    summary_path = REPORT_DIR / f"{partition}_label_summary_r{radius}.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(diag_path, index=False)
    summary = report.groupby(["partition", "radius_m"], as_index=False).agg(
        trips=("trip_id", "nunique"), gps_observations=("gps_observations", "sum"),
        confident_observations=("confident_observations", "sum"), candidate_misses=("candidate_misses", "sum"),
        positive_observations=("positive_observations", "sum"),
        candidate_covered_observations=("candidate_covered_observations", "sum"),
        coverage_eligible_observations=("coverage_eligible_observations", "sum"),
        labeled_candidate_rows=("labeled_candidate_rows", "sum"),
        ambiguous_candidate_rows=("ambiguous_candidate_rows", "sum"),
        ambiguous_observations=("ambiguous_observations", "sum"), unalignable_observations=("unalignable_observations", "sum")
    )
    summary["candidate_coverage_pct"] = np.where(summary.coverage_eligible_observations > 0,
        100 * summary.candidate_covered_observations / summary.coverage_eligible_observations, np.nan)
    summary.to_csv(summary_path, index=False)
    return output_path, diag_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", choices=["train", "student_test"], default="train")
    p.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    args = p.parse_args()
    print(label_candidate_file(args.partition, args.radius))


if __name__ == "__main__": main()
