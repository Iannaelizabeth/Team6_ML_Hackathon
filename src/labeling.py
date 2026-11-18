"""Sequence-aware reference-route alignment and candidate binary labels.

Route arc IDs come from `.route` in chronological order. GPS observations are
aligned by a monotone Viterbi pass over nearby route arcs, with spatial emissions
and temporal/geodesic progression costs. Ambiguous or unalignable observations
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

    The dynamic program permits staying on or advancing along route positions,
    never moving backward. Its emission is projected point-to-directed-segment
    distance; transition cost compares route chainage advance to GPS geodesic
    displacement, scaled by elapsed time. Route arcs that are disconnected are
    recorded as a status warning when the selected path crosses them.
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
    state_positions, state_distances, state_ratios = [], [], []
    for i in range(len(tr)):
        ratio = np.clip(((gx[i]-x1)*dx + (gy[i]-y1)*dy) / l2, 0.0, 1.0)
        px, py = x1 + ratio*dx, y1 + ratio*dy
        dist = np.hypot(gx[i]-px, gy[i]-py)
        k = min(states_per_observation, len(dist))
        ids = np.argpartition(dist, k-1)[:k] if k < len(dist) else np.arange(len(dist))
        ids = ids[np.argsort(dist[ids])]
        state_positions.append(ids.astype(int)); state_distances.append(dist[ids]); state_ratios.append(ratio[ids])

    ts = tr.timestamp.to_numpy(dtype=float)
    lonlat = np.column_stack([lon, lat])
    n = len(tr)
    costs = [state_distances[0] / 20.0]
    backpointers = [np.full(len(costs[0]), -1, dtype=int)]
    for t in range(1, n):
        _, _, displacement = GEOD.inv(lonlat[t-1, 0], lonlat[t-1, 1], lonlat[t, 0], lonlat[t, 1])
        dt = ts[t] - ts[t-1]
        scale = max(20.0, 5.0 * max(dt, 1.0))
        prev_states, curr_states = state_positions[t-1], state_positions[t]
        prev_cost = costs[-1]
        current_cost = np.full(len(curr_states), np.inf)
        back = np.full(len(curr_states), -1, dtype=int)
        for ci, cur in enumerate(curr_states):
            eligible = np.flatnonzero(prev_states <= cur)
            if not len(eligible):
                # Keep the path defined for GPS jitter/backtracking, but mark
                # the resulting observation unconfident below.
                eligible = np.arange(len(prev_states))
            delta_m = cumulative[cur] - cumulative[prev_states[eligible]]
            progress_cost = np.abs(delta_m - displacement) / scale
            backward_penalty = np.where(prev_states[eligible] > cur, 25.0, 0.0)
            broken_penalty = np.zeros(len(eligible))
            for ix, prev in enumerate(prev_states[eligible]):
                lo, hi = sorted((int(prev), int(cur)))
                if hi > lo and broken[lo+1:hi+1].any(): broken_penalty[ix] = 2.0
            totals = prev_cost[eligible] + progress_cost + backward_penalty + broken_penalty
            best_local = int(np.argmin(totals))
            current_cost[ci] = state_distances[t][ci] / 20.0 + totals[best_local]
            back[ci] = eligible[best_local]
        costs.append(current_cost); backpointers.append(back)
    chosen_state = np.zeros(n, dtype=int)
    chosen_state[-1] = int(np.argmin(costs[-1]))
    for t in range(n-1, 0, -1): chosen_state[t-1] = backpointers[t][chosen_state[t]]
    chosen = np.array([state_positions[t][chosen_state[t]] for t in range(n)], dtype=int)
    selected_dist = np.array([state_distances[t][chosen_state[t]] for t in range(n)], dtype=float)
    selected_ratio = np.array([state_ratios[t][chosen_state[t]] for t in range(n)], dtype=float)
    selected_aid = aids[chosen]
    selected_route_position = route_pos[chosen]
    chain = cumulative[chosen] + selected_ratio * lengths[chosen]

    output = []
    for t in range(n):
        statuses = []
        if not np.isfinite([lon[t], lat[t], ts[t]]).all(): statuses.append("invalid_timestamp_or_coordinate")
        if selected_dist[t] > max_distance_m: statuses.append("unalignable_spatial")
        if t and ts[t] - ts[t-1] > gps_gap_seconds: statuses.append("gps_gap")
        if t and chosen[t] < chosen[t-1]: statuses.append("route_order_violation")
        if t and chosen[t] > chosen[t-1]:
            lo, hi = chosen[t-1], chosen[t]
            if hi < len(broken) and broken[lo+1:hi+1].any(): statuses.append("route_discontinuity")
        lower = selected_route_position[t-1] if t else 0
        upper = selected_route_position[t+1] if t+1 < n else len(route)-1
        alternatives = []
        for j, (state_ids, state_ds) in enumerate(zip(state_positions[t], state_distances[t])):
            route_index = route_pos[state_ids]
            if lower <= route_index <= upper and aids[state_ids] != selected_aid[t] and state_ds <= selected_dist[t] + ambiguity_margin_m:
                alternatives.append(float(state_ds))
        margin = min(alternatives) - selected_dist[t] if alternatives else np.inf
        if alternatives: statuses.append("ambiguous_reference")
        if not statuses: statuses = ["confident"]
        # Gap, broken topology and competing arcs all prevent a positive label.
        status = "|".join(dict.fromkeys(statuses))
        output.append({
            "observation_id": tr.iloc[t].observation_id,
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
    obs_diag["candidate_covered"] = obs_diag.observation_id.map(present).fillna(False)
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
        raw = load_trip(partition, trip_id)
        clean, _ = preprocess_track(raw["track"])
        alignment = align_track_to_route(clean, raw["nodes"], raw["arcs"], raw["route"])
        labelled, obs_diag = label_candidate_trip(cand, alignment)
        obs_diag.insert(0, "trip_id", trip_id); obs_diag.insert(1, "partition", partition); obs_diag["radius_m"] = radius
        obs_frames.append(obs_diag)
        labelled.to_csv(output_path, mode="a", index=False, header=not output_path.exists())
        counts = obs_diag.alignment_status.value_counts()
        report_rows.append({"trip_id": trip_id, "partition": partition, "radius_m": radius,
                            "gps_observations": len(alignment), "candidate_rows": len(cand),
                            "confident_observations": int(counts.get("confident", 0)),
                            "candidate_misses": int(counts.get("candidate_miss", 0)),
                            "ambiguous_observations": int(obs_diag.alignment_status.str.contains("ambiguous|gps_gap|discontinuity|order_violation", regex=True).sum()),
                            "unalignable_observations": int(obs_diag.alignment_status.str.contains("unalignable|invalid", regex=True).sum()),
                            "candidate_coverage_pct": float(100 * obs_diag.candidate_covered.mean()) if len(obs_diag) else np.nan})
    report = pd.DataFrame(report_rows)
    diag_path = REPORT_DIR / f"{partition}_alignment_diagnostics_r{radius}.csv"
    summary_path = REPORT_DIR / f"{partition}_label_summary_r{radius}.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(diag_path, index=False)
    report.groupby(["partition", "radius_m"], as_index=False).agg(
        trips=("trip_id", "nunique"), gps_observations=("gps_observations", "sum"),
        confident_observations=("confident_observations", "sum"), candidate_misses=("candidate_misses", "sum"),
        ambiguous_observations=("ambiguous_observations", "sum"), unalignable_observations=("unalignable_observations", "sum")
    ).to_csv(summary_path, index=False)
    return output_path, diag_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", choices=["train", "student_test"], default="train")
    p.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    args = p.parse_args()
    print(label_candidate_file(args.partition, args.radius))


if __name__ == "__main__": main()
