"""
data_contract.py
================
The data interface that Member 4's files (model_5.py, hmm.py, main.py) depend on.

WHY THIS FILE EXISTS
--------------------
Member 1 owns preprocessing.py / feature_engineering.py and candidate generation.
Until those land, this file is a *reference implementation* of the same interface
so that the HMM, model 5 and the API can be built and tested today.

Once Member 1's modules are ready, either:
  (a) make Member 1's functions return the same shapes documented below, and
      replace the bodies here with imports from preprocessing / feature_engineering, or
  (b) keep this file as the single shared module for the whole team.
Agree on ONE of these in your first team meeting - the column names in
FEATURE_COLUMNS and CANDIDATE_COLUMNS are the contract.

DATASET FORMAT (assumed - verify in Member 1's EDA)
---------------------------------------------------
For each trip <id> there are four whitespace-separated text files:
  <id>.nodes : one node per line   ->  x y            (x = longitude, y = latitude)
  <id>.arcs  : one arc per line    ->  from_node to_node   (node indices, 0-based)
  <id>.track : one GPS fix per line ->  x y t          (lon, lat, time in seconds)
  <id>.route : one arc index per line - the ground-truth arc sequence
The arc index is the 0-based line number in .arcs. Arc IDs are LOCAL TO EACH TRIP:
arc 17 in trip A is a different road from arc 17 in trip B.

Coordinate order is auto-detected when a column exceeds +/-90 degrees; otherwise
lon/lat (x y) order is assumed. Pass coord_order="latlon" if EDA shows otherwise.
1-based indices in .arcs/.route are auto-detected.

KEY TERMS
---------
candidate   : a road segment (arc) close enough to a GPS fix that the fix might
              lie on it. Each GPS fix gets up to `max_candidates` of them.
projection  : the closest point on an arc to a GPS fix; `frac` is how far along the
              arc that point is (0 = start node, 1 = end node).
label       : for training, the arc each GPS fix truly lies on. The .route file only
              gives the ordered arc sequence, not per-fix labels, so labels are derived
              by a monotone alignment (see label_points_with_route).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8
TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")

# Feature columns every model (models 1-5) is trained on. Changing this list
# means re-training every model - agree on it as a team.
FEATURE_COLUMNS: list[str] = [
    "dist_m",            # perpendicular distance GPS fix -> arc (metres)
    "dist_rel_m",        # dist_m minus the smallest candidate distance for that fix
    "dist_rank",         # 0 = nearest candidate for that fix
    "n_candidates",      # how many candidates the fix has (ambiguity)
    "frac_along",        # projection position along the arc, 0..1
    "at_endpoint",       # 1 if the projection is clamped to an arc end
    "arc_len_m",         # arc length (metres)
    "heading_diff_deg",  # angle between GPS travel direction and arc direction, 0..180
    "heading_valid",     # 0 if the vehicle barely moved, so heading is unreliable
    "speed_mps",         # GPS speed estimate (metres / second)
    "dt_prev_s",         # seconds since the previous GPS fix
]

CANDIDATE_COLUMNS: list[str] = [
    "trip_id", "point_idx", "arc", "frac_along", "proj_x", "proj_y",
    *[c for c in FEATURE_COLUMNS if c != "frac_along"],
    "label",             # 1 = this candidate is the true arc, 0 = not, -1 = unknown
]


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # lat, lon, t, x, y  (x/y = local metres)
    nodes_ll: np.ndarray         # (N, 2) lat, lon
    nodes_xy: np.ndarray         # (N, 2) local metres
    arcs: np.ndarray             # (A, 2) from_node, to_node
    route: np.ndarray | None     # (R,) ground-truth arc indices, or None
    origin: tuple[float, float]  # (lat0, lon0) of the local projection
    arc_len: np.ndarray          # (A,) metres
    arc_bearing: np.ndarray      # (A,) compass degrees, 0 = north

    # -- coordinate helpers ------------------------------------------------
    def xy_to_latlon(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat0, lon0 = self.origin
        lat = lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
        lon = lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
        return lat, lon

    def arc_latlon(self, arc: int) -> list[list[float]]:
        u, v = self.arcs[arc]
        return [self.nodes_ll[u].tolist(), self.nodes_ll[v].tolist()]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _read_numeric(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"[\s,;]+", header=None, comment="#", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return df.to_numpy(dtype=float)


def _latlon_from_xy(a: np.ndarray, b: np.ndarray, coord_order: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) given two coordinate columns a, b."""
    if coord_order == "latlon":
        return a, b
    if coord_order == "lonlat":
        return b, a
    # auto
    if np.nanmax(np.abs(a)) > 90:
        return b, a          # a is longitude
    if np.nanmax(np.abs(b)) > 90:
        return a, b          # b is longitude
    return b, a              # ambiguous -> dataset convention "x y" = lon lat


def _maybe_one_based(idx: np.ndarray, n: int) -> np.ndarray:
    idx = idx.astype(np.int64)
    if idx.size and idx.min() >= 1 and idx.max() == n:
        return idx - 1
    return idx


def list_trips(data_dir: str | Path) -> dict[str, Path]:
    """Map trip_id -> directory that contains <trip_id>.track/.nodes/.arcs."""
    data_dir = Path(data_dir)
    trips: dict[str, Path] = {}
    for track in sorted(data_dir.rglob("*.track")):
        tid = track.stem
        folder = track.parent
        if (folder / f"{tid}.nodes").exists() and (folder / f"{tid}.arcs").exists():
            trips[tid] = folder
    return trips


def load_trip(folder: str | Path, trip_id: str, coord_order: str = "auto") -> Trip:
    if not TRIP_ID_PATTERN.match(trip_id):
        raise ValueError(f"Invalid trip id: {trip_id!r}")
    folder = Path(folder)

    nodes_raw = _read_numeric(folder / f"{trip_id}.nodes")
    n_lat, n_lon = _latlon_from_xy(nodes_raw[:, 0], nodes_raw[:, 1], coord_order)
    nodes_ll = np.column_stack([n_lat, n_lon])

    arcs = _read_numeric(folder / f"{trip_id}.arcs")[:, :2]
    arcs = _maybe_one_based(arcs, len(nodes_ll)).reshape(-1, 2)

    tr = _read_numeric(folder / f"{trip_id}.track")
    t_lat, t_lon = _latlon_from_xy(tr[:, 0], tr[:, 1], coord_order)
    t = tr[:, 2] if tr.shape[1] >= 3 else np.arange(len(tr), dtype=float)

    route = None
    route_path = folder / f"{trip_id}.route"
    if route_path.exists() and route_path.stat().st_size > 0:
        r = _read_numeric(route_path)
        if r.size:
            route = _maybe_one_based(r[:, 0], len(arcs))

    # Local equirectangular projection (accurate to <0.1% over a city-sized area)
    lat0, lon0 = float(np.mean(n_lat)), float(np.mean(n_lon))
    coslat = math.cos(math.radians(lat0))

    def to_xy(lat, lon):
        return (np.radians(lon - lon0) * EARTH_RADIUS_M * coslat,
                np.radians(lat - lat0) * EARTH_RADIUS_M)

    nx, ny = to_xy(n_lat, n_lon)
    nodes_xy = np.column_stack([nx, ny])
    tx, ty = to_xy(t_lat, t_lon)
    track = pd.DataFrame({"lat": t_lat, "lon": t_lon, "t": t, "x": tx, "y": ty})

    a, b = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
    d = b - a
    arc_len = np.hypot(d[:, 0], d[:, 1])
    arc_bearing = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360) % 360

    return Trip(trip_id, track, nodes_ll, nodes_xy, arcs, route,
                (lat0, lon0), arc_len, arc_bearing)


def trip_from_points(base: Trip, points: Iterable[dict]) -> Trip:
    """Re-use a trip's road network with a different (e.g. user-uploaded) GPS track."""
    pts = list(points)
    lat = np.array([p["lat"] for p in pts], float)
    lon = np.array([p["lon"] for p in pts], float)
    t = np.array([p.get("t", i) if p.get("t") is not None else i for i, p in enumerate(pts)], float)
    lat0, lon0 = base.origin
    coslat = math.cos(math.radians(lat0))
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * coslat
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    track = pd.DataFrame({"lat": lat, "lon": lon, "t": t, "x": x, "y": y})
    return Trip(base.trip_id, track, base.nodes_ll, base.nodes_xy, base.arcs, None,
                base.origin, base.arc_len, base.arc_bearing)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def project_points_to_arcs(px, py, ax, ay, bx, by):
    """Distances (P, A) and projection fractions (P, A) of points onto segments."""
    dx, dy = bx - ax, by - ay
    l2 = np.where(dx * dx + dy * dy == 0, 1e-9, dx * dx + dy * dy)
    frac = ((px[:, None] - ax[None]) * dx[None] + (py[:, None] - ay[None]) * dy[None]) / l2[None]
    frac = np.clip(frac, 0.0, 1.0)
    qx = ax[None] + frac * dx[None]
    qy = ay[None] + frac * dy[None]
    dist = np.hypot(px[:, None] - qx, py[:, None] - qy)
    return dist, frac


def _arc_endpoints(trip: Trip, arc_idx: np.ndarray | None = None):
    arcs = trip.arcs if arc_idx is None else trip.arcs[arc_idx]
    a, b = trip.nodes_xy[arcs[:, 0]], trip.nodes_xy[arcs[:, 1]]
    return a[:, 0], a[:, 1], b[:, 0], b[:, 1]


def _chunks(n_rows: int, n_cols: int, budget: int = 2_000_000):
    step = max(1, budget // max(1, n_cols))
    for s in range(0, n_rows, step):
        yield s, min(n_rows, s + step)


# ----------------------------------------------------------------------------
# Ground-truth labels: monotone alignment of GPS fixes to the route
# ----------------------------------------------------------------------------
def label_points_with_route(trip: Trip, max_label_dist_m: float = 75.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign every GPS fix to one arc of the ground-truth route such that the
    assignment never moves backwards along the route (dynamic programming,
    O(T x R)). Simply taking the nearest route arc fails at loops and where the
    route passes the same junction twice.

    Returns (label_arc, label_dist): label_arc = -1 where the fix is further
    than max_label_dist_m from its aligned arc (treated as unlabelled).
    """
    T = len(trip.track)
    if trip.route is None or len(trip.route) == 0 or T == 0:
        return np.full(T, -1), np.full(T, np.nan)
    route = trip.route
    R = len(route)
    ax, ay, bx, by = _arc_endpoints(trip, route)
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    cost = np.empty((T, R))
    for s, e in _chunks(T, R):
        cost[s:e], _ = project_points_to_arcs(px[s:e], py[s:e], ax, ay, bx, by)

    back = np.zeros((T, R), dtype=np.int32)
    ar = np.arange(R)
    D = cost[0].copy()
    for t in range(1, T):
        pm = np.minimum.accumulate(D)
        back[t] = np.maximum.accumulate(np.where(D == pm, ar, 0))
        D = cost[t] + pm
    r = int(np.argmin(D))
    pos = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        pos[t] = r
        r = back[t, r]
    label_arc = route[pos].astype(np.int64)
    label_dist = cost[np.arange(T), pos]
    label_arc[label_dist > max_label_dist_m] = -1
    return label_arc, label_dist


# ----------------------------------------------------------------------------
# Candidates + features
# ----------------------------------------------------------------------------
def point_kinematics(trip: Trip) -> pd.DataFrame:
    x, y, t = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    n = len(x)
    dt_prev = np.r_[np.nan, np.diff(t)]
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_prev > 0, step / dt_prev, np.nan)
    # central-difference heading (more stable than one-sided)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    hx, hy = x[i1] - x[i0], y[i1] - y[i0]
    moved = np.hypot(hx, hy)
    heading = (np.degrees(np.arctan2(hx, hy)) + 360) % 360
    heading_valid = (moved >= 3.0).astype(int)
    out = pd.DataFrame({
        "point_idx": np.arange(n),
        "dt_prev_s": np.nan_to_num(dt_prev, nan=0.0),
        "step_m": np.nan_to_num(step, nan=0.0),
        "speed_mps": np.nan_to_num(np.clip(speed, 0, 70), nan=0.0),
        "heading_deg": heading,
        "heading_valid": heading_valid,
    })
    return out


def generate_candidates(trip: Trip, radius_m: float = 75.0, max_candidates: int = 8,
                        fallback_radius_m: float = 200.0) -> pd.DataFrame:
    """Up to `max_candidates` nearest arcs within `radius_m` of every GPS fix.
    Fixes with nothing inside radius_m get their nearest arcs within fallback_radius_m;
    fixes with nothing even there get no candidates (the HMM treats them as gaps)."""
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    ax, ay, bx, by = _arc_endpoints(trip)
    A = len(trip.arcs)
    rows = []
    for s, e in _chunks(len(px), A):
        dist, frac = project_points_to_arcs(px[s:e], py[s:e], ax, ay, bx, by)
        k = min(max_candidates, A)
        nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
        for i in range(e - s):
            idx = nearest[i]
            d = dist[i, idx]
            keep = d <= radius_m
            if not keep.any():
                keep = d <= fallback_radius_m
            idx, d = idx[keep], d[keep]
            order = np.argsort(d, kind="stable")
            idx, d = idx[order], d[order]
            f = frac[i, idx]
            for rank, (arc, dd, ff) in enumerate(zip(idx, d, f)):
                rows.append((s + i, int(arc), float(dd), float(ff), rank))
    cand = pd.DataFrame(rows, columns=["point_idx", "arc", "dist_m", "frac_along", "dist_rank"])
    if cand.empty:
        return cand
    a = trip.arcs[cand["arc"].to_numpy()]
    pa, pb = trip.nodes_xy[a[:, 0]], trip.nodes_xy[a[:, 1]]
    f = cand["frac_along"].to_numpy()[:, None]
    proj = pa + f * (pb - pa)
    cand["proj_x"], cand["proj_y"] = proj[:, 0], proj[:, 1]
    return cand


def build_trip_frame(trip: Trip, radius_m: float = 75.0, max_candidates: int = 8,
                     with_labels: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
      points : one row per GPS fix (kinematics + label_arc + has_true_candidate)
      cand   : one row per (fix, candidate arc) with FEATURE_COLUMNS and `label`
    """
    points = point_kinematics(trip)
    if with_labels and trip.route is not None:
        label_arc, label_dist = label_points_with_route(trip)
    else:
        label_arc, label_dist = np.full(len(points), -1), np.full(len(points), np.nan)
    points["label_arc"] = label_arc
    points["label_dist_m"] = label_dist

    cand = generate_candidates(trip, radius_m, max_candidates)
    if cand.empty:
        points["has_true_candidate"] = False
        return points, pd.DataFrame(columns=CANDIDATE_COLUMNS)

    cand = cand.merge(points[["point_idx", "dt_prev_s", "speed_mps", "heading_deg",
                              "heading_valid", "label_arc"]], on="point_idx", how="left")
    g = cand.groupby("point_idx")
    cand["dist_rel_m"] = cand["dist_m"] - g["dist_m"].transform("min")
    cand["n_candidates"] = g["arc"].transform("size")
    cand["at_endpoint"] = ((cand["frac_along"] <= 1e-3) | (cand["frac_along"] >= 1 - 1e-3)).astype(int)
    cand["arc_len_m"] = trip.arc_len[cand["arc"].to_numpy()]
    diff = np.abs(cand["heading_deg"].to_numpy() - trip.arc_bearing[cand["arc"].to_numpy()]) % 360
    diff = np.where(diff > 180, 360 - diff, diff)
    cand["heading_diff_deg"] = np.where(cand["heading_valid"] == 1, diff, 90.0)
    cand["label"] = np.where(cand["label_arc"] < 0, -1,
                             (cand["arc"] == cand["label_arc"]).astype(int))
    cand["trip_id"] = trip.trip_id

    has_true = cand.loc[cand["label"] == 1, "point_idx"].unique()
    points["has_true_candidate"] = points["point_idx"].isin(has_true)
    return points, cand[CANDIDATE_COLUMNS].reset_index(drop=True)


def build_dataset(data_dir: str | Path, trip_ids: list[str] | None = None,
                  max_trips: int | None = None, cache_path: str | Path | None = None,
                  radius_m: float = 75.0, max_candidates: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Candidate-level training table for many trips (cached to disk if cache_path given)."""
    if cache_path and Path(cache_path).exists():
        return pd.read_pickle(cache_path)
    index = list_trips(data_dir)
    ids = trip_ids or list(index)
    if max_trips:
        ids = ids[:max_trips]
    frames = []
    for i, tid in enumerate(ids, 1):
        try:
            trip = load_trip(index[tid], tid)
            _, cand = build_trip_frame(trip, radius_m, max_candidates)
            frames.append(cand[cand["label"] >= 0])
        except Exception as exc:  # one bad trip should not kill the whole run
            print(f"  [skip] {tid}: {exc}")
        if verbose and (i % 25 == 0 or i == len(ids)):
            print(f"  features built for {i}/{len(ids)} trips")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df
