"""Leakage-safe spatial and temporal features for PathSnap candidates."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer

from src.preprocessing import PROJECT_ROOT, load_trip, preprocess_track

FEATURE_DIR = PROJECT_ROOT / "outputs" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
ORIGINAL_FEATURES = ["perpendicular_distance", "projection_ratio", "road_length"]
ENGINEERED_FEATURES = ["gps_heading", "road_heading", "heading_difference", "gps_speed",
                       "time_difference", "distance_to_previous", "distance_to_next", "connectivity_score"]
GEOD = Geod(ellps="WGS84")


def geodesic_distance_m(lon1, lat1, lon2, lat2):
    """WGS84 geodesic distance in metres; inputs may be scalars or arrays."""
    _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    return np.asarray(distance, dtype=float)


def _heading(lon1, lat1, lon2, lat2):
    bearing, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    bearing = np.asarray(bearing, dtype=float) % 360.0
    distance = np.asarray(distance, dtype=float)
    return np.where(np.isfinite(distance) & (distance > 0.01), bearing, np.nan)


def angular_difference_degrees(a, b):
    """Smallest absolute circular difference between headings, in [0, 180]."""
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0)


def track_context(track: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-observation GPS motion features, isolated to one trip."""
    required = {"longitude", "latitude", "timestamp"}
    if not required.issubset(track.columns):
        raise ValueError(f"track missing columns: {sorted(required - set(track.columns))}")
    t = track.copy()
    if "observation_id" not in t: t["observation_id"] = t.index
    t = t.sort_values(["timestamp", "observation_id"], kind="stable").reset_index(drop=True)
    n = len(t)
    lon, lat, ts = (t[c].to_numpy(dtype=float) for c in ["longitude", "latitude", "timestamp"])
    prev_lon = np.r_[np.nan, lon[:-1]] if n else np.array([], dtype=float)
    prev_lat = np.r_[np.nan, lat[:-1]] if n else np.array([], dtype=float)
    next_lon = np.r_[lon[1:], np.nan] if n else np.array([], dtype=float)
    next_lat = np.r_[lat[1:], np.nan] if n else np.array([], dtype=float)
    prev_ts = np.r_[np.nan, ts[:-1]] if n else np.array([], dtype=float)
    next_ts = np.r_[ts[1:], np.nan] if n else np.array([], dtype=float)
    dt_prev, dt_next = ts - prev_ts, next_ts - ts
    d_prev = geodesic_distance_m(prev_lon, prev_lat, lon, lat) if n else np.array([])
    d_next = geodesic_distance_m(lon, lat, next_lon, next_lat) if n else np.array([])
    heading, speed = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if i == 0 and n > 1:
            heading[i] = _heading(lon[i], lat[i], next_lon[i], next_lat[i])
            if dt_next[i] > 0: speed[i] = d_next[i] / dt_next[i]
        elif i == n - 1 and n > 1:
            heading[i] = _heading(prev_lon[i], prev_lat[i], lon[i], lat[i])
            if dt_prev[i] > 0: speed[i] = d_prev[i] / dt_prev[i]
        elif 0 < i < n - 1:
            heading[i] = _heading(prev_lon[i], prev_lat[i], next_lon[i], next_lat[i])
            span = dt_prev[i] + dt_next[i]
            if span > 0: speed[i] = (d_prev[i] + d_next[i]) / span
    t["gps_heading"], t["gps_speed"] = heading, speed
    t["time_difference"] = np.where(dt_prev > 0, dt_prev, np.nan)
    t["_prev_lon"], t["_prev_lat"] = prev_lon, prev_lat
    t["_next_lon"], t["_next_lat"] = next_lon, next_lat
    ids = t.observation_id.to_numpy()
    t["_prev_observation_id"] = np.r_[np.nan, ids[:-1]] if n else []
    t["_next_observation_id"] = np.r_[ids[1:], np.nan] if n else []
    return t


def _trip_transformer(nodes, track):
    coords = pd.concat([nodes[["longitude", "latitude"]], track[["longitude", "latitude"]]], ignore_index=True)
    valid = coords.longitude.between(-180, 180) & coords.latitude.between(-90, 90)
    if not valid.any(): raise ValueError("No valid coordinates for local projection")
    lon0, lat0 = coords.loc[valid, ["longitude", "latitude"]].mean().to_list()
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def engineer_trip_candidates(candidates: pd.DataFrame, track: pd.DataFrame,
                             nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    """Add eight label-independent features to candidates for one trip.

    Connectivity is the fraction of available adjacent observation sides for
    which directed candidate arcs join at a shared node. The previous/next
    distance features measure each candidate's projected road point to adjacent
    GPS fixes. Reference routes and labels are never read.
    """
    required = set(ORIGINAL_FEATURES + ["observation_id", "candidate_arc_id"])
    missing = required - set(candidates.columns)
    if missing: raise ValueError(f"candidate rows missing columns: {sorted(missing)}")
    ctx = track_context(track)
    transformer = _trip_transformer(nodes, ctx)
    node_xy = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    nx, ny = transformer.transform(node_xy[:, 0], node_xy[:, 1])
    nx, ny = np.asarray(nx), np.asarray(ny)
    arc_end = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    arc_valid = np.isfinite(arc_end).all(axis=1) & (arc_end % 1 == 0).all(axis=1)
    arc_valid &= (arc_end[:, 0] >= 0) & (arc_end[:, 0] < len(nodes)) & (arc_end[:, 1] >= 0) & (arc_end[:, 1] < len(nodes))
    from_node, to_node = np.full(len(arcs), -1, dtype=np.int64), np.full(len(arcs), -1, dtype=np.int64)
    x1, y1, x2, y2 = (np.full(len(arcs), np.nan) for _ in range(4))
    ids = np.flatnonzero(arc_valid)
    from_node[ids], to_node[ids] = arc_end[ids].astype(np.int64).T
    x1[ids], y1[ids], x2[ids], y2[ids] = nx[from_node[ids]], ny[from_node[ids]], nx[to_node[ids]], ny[to_node[ids]]
    arc_heading = (np.degrees(np.arctan2(x2-x1, y2-y1)) + 360.0) % 360.0
    arc_heading[~arc_valid] = np.nan
    ctx_cols = ["observation_id", "gps_heading", "gps_speed", "time_difference", "_prev_lon", "_prev_lat",
                "_next_lon", "_next_lat", "_prev_observation_id", "_next_observation_id", "longitude", "latitude"]
    out = candidates.merge(ctx[ctx_cols], on="observation_id", how="left", validate="many_to_one", sort=False)
    raw_ids = pd.to_numeric(out.candidate_arc_id, errors="coerce").to_numpy(dtype=float)
    arc_ok = np.isfinite(raw_ids) & (raw_ids % 1 == 0) & (raw_ids >= 0) & (raw_ids < len(arcs))
    safe_arc = np.where(arc_ok, raw_ids, 0).astype(int)
    out["road_heading"] = np.where(arc_ok, arc_heading[safe_arc], np.nan)
    out["heading_difference"] = angular_difference_degrees(out.gps_heading, out.road_heading)
    ratio = np.clip(pd.to_numeric(out.projection_ratio, errors="coerce").to_numpy(dtype=float), 0, 1)
    foot_x = x1[safe_arc] + ratio*(x2[safe_arc]-x1[safe_arc])
    foot_y = y1[safe_arc] + ratio*(y2[safe_arc]-y1[safe_arc])
    foot_x[~arc_ok] = np.nan; foot_y[~arc_ok] = np.nan
    prev_lon, prev_lat = out["_prev_lon"].to_numpy(dtype=float), out["_prev_lat"].to_numpy(dtype=float)
    next_lon, next_lat = out["_next_lon"].to_numpy(dtype=float), out["_next_lat"].to_numpy(dtype=float)
    px, py = transformer.transform(prev_lon, prev_lat); qx, qy = transformer.transform(next_lon, next_lat)
    out["distance_to_previous"] = np.hypot(foot_x-np.asarray(px), foot_y-np.asarray(py))
    out["distance_to_next"] = np.hypot(foot_x-np.asarray(qx), foot_y-np.asarray(qy))

    obs_candidates = {}
    for oid, aid, ok in zip(out.observation_id.to_numpy(), safe_arc, arc_ok):
        if ok: obs_candidates.setdefault(oid, []).append((from_node[aid], to_node[aid]))
    obs_candidates = {k: (set(a for a, _ in v), set(b for _, b in v)) for k, v in obs_candidates.items()}
    scores = np.full(len(out), np.nan)
    this_from, this_to = from_node[safe_arc], to_node[safe_arc]
    for i, (prev_id, next_id) in enumerate(zip(out["_prev_observation_id"].to_numpy(), out["_next_observation_id"].to_numpy())):
        connected = available = 0
        if pd.notna(prev_id) and prev_id in obs_candidates:
            available += 1; connected += int(this_from[i] >= 0 and this_from[i] in obs_candidates[prev_id][1])
        if pd.notna(next_id) and next_id in obs_candidates:
            available += 1; connected += int(this_to[i] >= 0 and this_to[i] in obs_candidates[next_id][0])
        if available: scores[i] = connected / available
    out["connectivity_score"] = scores
    helpers = ["_prev_lon", "_prev_lat", "_next_lon", "_next_lat", "_prev_observation_id", "_next_observation_id", "longitude", "latitude"]
    return out.drop(columns=helpers)


def _trip_groups(path: Path, chunksize=250_000) -> Iterator[tuple[str, pd.DataFrame]]:
    """Stream trip-contiguous candidate CSVs while preserving all rows."""
    last_id, buffer, emitted = None, None, set()
    for chunk in pd.read_csv(path, dtype={"trip_id": "string"}, chunksize=chunksize):
        if "trip_id" not in chunk: raise ValueError(f"{path} lacks trip_id")
        for trip_id, group in chunk.groupby("trip_id", sort=False):
            trip_id = str(trip_id)
            if last_id is None: last_id, buffer = trip_id, group.copy()
            elif trip_id == last_id: buffer = pd.concat([buffer, group], ignore_index=True)
            else:
                if trip_id in emitted: raise ValueError("Candidate CSV is not grouped by trip_id")
                emitted.add(last_id); yield last_id, buffer
                last_id, buffer = trip_id, group.copy()
    if last_id is not None: yield last_id, buffer


def process_candidate_file(partition: str, radius: int, input_path: Path | None = None,
                           output_path: Path | None = None, chunksize=250_000) -> Path:
    """Generate a full enriched CSV, processing one trip at a time."""
    if input_path is None:
        source_name = f"{partition}_candidates_r{radius}.csv"
        input_path = PROJECT_ROOT / "data" / "candidates" / source_name
        if not input_path.exists(): input_path = FEATURE_DIR / source_name
    output_path = output_path or FEATURE_DIR / f"{partition}_features_r{radius}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists(): raise FileNotFoundError(input_path)
    if output_path.exists(): output_path.unlink()
    for trip_id, candidates in _trip_groups(input_path, chunksize=chunksize):
        trip = load_trip(partition, trip_id)
        clean_track, _ = preprocess_track(trip["track"])
        enriched = engineer_trip_candidates(candidates, clean_track, trip["nodes"], trip["arcs"])
        enriched.to_csv(output_path, mode="a", index=False, header=not output_path.exists())
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["train", "student_test"], default="train")
    parser.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    args = parser.parse_args()
    print(process_candidate_file(args.partition, args.radius))


if __name__ == "__main__": main()
