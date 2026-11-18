"""
hmm.py
======
HMM-based route reconstruction (map matching), following Newson & Krumm (2009),
"Hidden Markov Map Matching Through Noise and Sparseness", with two extensions:
  * the emission probability can come from a trained classifier (models 1-5), and
  * explicit handling of GPS gaps and disconnected road segments.

Pipeline for one trip
---------------------
1. Candidates per GPS fix (data_contract.build_trip_frame).
2. Emission log-probability per candidate:
     "gaussian"   : log N(distance; 0, sigma_z)       <- classic Newson-Krumm
     "classifier" : log P(candidate is true | features)  <- model 1-5 probability
3. Transition log-probability between candidates of consecutive fixes:
     route distance d_route = driving distance along the road graph between the two
     projections; straight distance d_gc = distance between the two GPS fixes.
     log p = -log(beta) - |d_route - d_gc| / beta        (exponential model)
   Real driving rarely takes a long detour between two close fixes, so a large
   |d_route - d_gc| is unlikely.
4. viterbi.viterbi() picks the most likely candidate sequence.
5. Consecutive matched arcs are joined with shortest paths to give the full arc route.

Gaps and disconnections
-----------------------
* A fix with no candidate within the search radius is skipped (counted as unmatched).
* A time gap > max_gap_s forces a chain break (transitions are meaningless over long gaps).
* beta grows with the sampling interval (longer gaps -> more route uncertainty).
* If no candidate pair is reachable on the road graph (disconnected network pieces),
  every transition is -inf and Viterbi breaks the chain instead of failing.

Run as a script to compare classifier-only vs HMM decoders on held-out trips:
    python hmm.py --data-dir data/ --model artifacts/model_5.joblib --max-trips 100
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from data_contract import FEATURE_COLUMNS, Trip, build_trip_frame, list_trips, load_trip
from viterbi import viterbi

LOG_EPS = 1e-6
DECODERS = ("hmm_classifier", "hmm_gaussian", "classifier_only")


# ----------------------------------------------------------------------------
# Road graph with cached shortest paths
# ----------------------------------------------------------------------------
class RoadGraph:
    """Directed road graph for one trip. Shortest paths are cached per source node."""

    def __init__(self, trip: Trip, limit_m: float = 5000.0):
        self.trip = trip
        self.limit_m = limit_m
        n = len(trip.nodes_xy)
        w = np.maximum(trip.arc_len, 1e-3)   # scipy treats explicit zeros as "no edge"
        df = pd.DataFrame({"u": trip.arcs[:, 0], "v": trip.arcs[:, 1], "w": w,
                           "arc": np.arange(len(w))})
        best = df.sort_values("w").drop_duplicates(["u", "v"])   # parallel arcs -> keep shortest
        self.csr = csr_matrix((best["w"], (best["u"], best["v"])), shape=(n, n))
        self.arc_of_edge = {(int(u), int(v)): int(a) for u, v, a in best[["u", "v", "arc"]].to_numpy()}
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _sp(self, src: int) -> tuple[np.ndarray, np.ndarray]:
        hit = self._cache.get(src)
        if hit is None:
            d, p = dijkstra(self.csr, directed=True, indices=src,
                            return_predecessors=True, limit=self.limit_m)
            hit = self._cache[src] = (d, p)
        return hit

    def node_distance(self, u: int, v: int) -> float:
        return 0.0 if u == v else float(self._sp(u)[0][v])

    def route_distance(self, arc_a: int, off_a: float, arc_b: int, off_b: float,
                       backtrack_tol_m: float = 15.0) -> float:
        """Driving distance from offset off_a on arc_a to offset off_b on arc_b (metres)."""
        if arc_a == arc_b and off_b >= off_a - backtrack_tol_m:
            return max(off_b - off_a, 0.0)   # small backwards jitter on the same arc = GPS noise
        u_end = int(self.trip.arcs[arc_a, 1])
        v_start = int(self.trip.arcs[arc_b, 0])
        return (self.trip.arc_len[arc_a] - off_a) + self.node_distance(u_end, v_start) + off_b

    def connecting_arcs(self, arc_a: int, arc_b: int) -> list[int] | None:
        """Arcs strictly between arc_a and arc_b on the shortest path; None if unreachable."""
        u = int(self.trip.arcs[arc_a, 1])
        v = int(self.trip.arcs[arc_b, 0])
        if u == v:
            return []
        dist, pred = self._sp(u)
        if not np.isfinite(dist[v]):
            return None
        nodes = [v]
        while nodes[-1] != u:
            p = int(pred[nodes[-1]])
            if p < 0:
                return None
            nodes.append(p)
        nodes.reverse()
        return [self.arc_of_edge[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]


# ----------------------------------------------------------------------------
# Configuration and result containers
# ----------------------------------------------------------------------------
@dataclass
class HMMConfig:
    sigma_z: float = 10.0          # GPS noise std-dev (m) for the Gaussian emission
    beta: float = 10.0             # transition scale (m); larger = more tolerant of detours
    max_gap_s: float = 180.0       # time gap that forces a chain break
    beta_gap_cap: float = 10.0     # beta is multiplied by min(dt / median_dt, cap)
    backtrack_tol_m: float = 15.0
    emission_weight: float = 1.0   # multiplies the emission log-prob (balance vs transition)
    sp_limit_m: float = 5000.0     # Dijkstra search radius; beyond this = unreachable

    @classmethod
    def from_dict(cls, d: dict | None) -> "HMMConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MatchResult:
    decoder: str
    matched: pd.DataFrame                  # one row per matched fix
    route_arcs: list[int]                  # reconstructed arc sequence
    n_points: int
    n_unmatched: int
    n_breaks: int
    gaps_unfilled: int                     # consecutive matched arcs with no road path
    runtime_ms: float
    metrics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Emission / transition models
# ----------------------------------------------------------------------------
def gaussian_log_emission(dist_m: np.ndarray, sigma_z: float) -> np.ndarray:
    return -0.5 * (dist_m / sigma_z) ** 2 - math.log(math.sqrt(2 * math.pi) * sigma_z)


def classifier_log_emission(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, LOG_EPS, 1.0))


def predict_candidate_proba(model_artifact: dict, cand: pd.DataFrame) -> np.ndarray:
    """P(candidate is the true arc). Works with any sklearn-style classifier saved
    in the shared artifact format: {"model": estimator, "feature_columns": [...]}."""
    cols = model_artifact.get("feature_columns", FEATURE_COLUMNS)
    if cand.empty:
        return np.zeros(0)
    return model_artifact["model"].predict_proba(cand[cols].to_numpy(dtype=float))[:, 1]


# ----------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------
def _group_candidates(cand: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    return [(int(pid), g) for pid, g in cand.groupby("point_idx", sort=True)]


def _matched_frame(trip: Trip, rows: list[dict]) -> pd.DataFrame:
    m = pd.DataFrame(rows)
    if m.empty:
        return pd.DataFrame(columns=["point_idx", "arc", "frac_along", "dist_m", "segment",
                                     "gps_lat", "gps_lon", "snap_lat", "snap_lon"])
    lat, lon = trip.xy_to_latlon(m["proj_x"].to_numpy(), m["proj_y"].to_numpy())
    m["snap_lat"], m["snap_lon"] = lat, lon
    m["gps_lat"] = trip.track["lat"].to_numpy()[m["point_idx"]]
    m["gps_lon"] = trip.track["lon"].to_numpy()[m["point_idx"]]
    return m.drop(columns=["proj_x", "proj_y"])


def reconstruct_route(graph: RoadGraph, matched: pd.DataFrame) -> tuple[list[int], int]:
    """Join consecutive matched arcs with shortest paths. Returns (arcs, unfilled_gaps)."""
    route: list[int] = []
    unfilled = 0
    prev = None
    for arc in matched["arc"].astype(int).tolist():
        if prev is None:
            route.append(arc)
        elif arc != prev:
            link = graph.connecting_arcs(prev, arc)
            if link is None:
                unfilled += 1
            else:
                route.extend(link)
            route.append(arc)
        prev = arc
    # collapse immediate repeats
    out = [a for i, a in enumerate(route) if i == 0 or a != route[i - 1]]
    return out, unfilled


def hmm_decode(trip: Trip, cand: pd.DataFrame, config: HMMConfig,
               proba: np.ndarray | None = None, graph: RoadGraph | None = None,
               decoder_name: str | None = None) -> MatchResult:
    """Viterbi decoding. If `proba` is given the emission is the classifier probability,
    otherwise the Gaussian distance model."""
    t0 = time.perf_counter()
    graph = graph or RoadGraph(trip, config.sp_limit_m)
    use_clf = proba is not None
    name = decoder_name or ("hmm_classifier" if use_clf else "hmm_gaussian")
    n_points = len(trip.track)
    if cand.empty:
        return MatchResult(name, _matched_frame(trip, []), [], n_points, n_points, 0, 0,
                           (time.perf_counter() - t0) * 1e3)

    cand = cand.copy()
    if use_clf:
        cand["_em"] = config.emission_weight * classifier_log_emission(proba)
    else:
        cand["_em"] = config.emission_weight * gaussian_log_emission(cand["dist_m"].to_numpy(), config.sigma_z)
    cand["_off"] = cand["frac_along"].to_numpy() * trip.arc_len[cand["arc"].to_numpy()]

    groups = _group_candidates(cand)
    px, py, pt = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    dts = np.diff(pt[[pid for pid, _ in groups]]) if len(groups) > 1 else np.array([1.0])
    median_dt = float(np.median(dts[dts > 0])) if (dts > 0).any() else 1.0

    log_em = [g["_em"].to_numpy() for _, g in groups]
    log_tr: list[np.ndarray | None] = []
    for (pa, ga), (pb, gb) in zip(groups[:-1], groups[1:]):
        dt = pt[pb] - pt[pa]
        if dt > config.max_gap_s:
            log_tr.append(None)
            continue
        beta = config.beta * float(np.clip(dt / median_dt if median_dt > 0 else 1.0, 1.0, config.beta_gap_cap))
        d_gc = math.hypot(px[pb] - px[pa], py[pb] - py[pa])
        arcs_a, offs_a = ga["arc"].to_numpy(), ga["_off"].to_numpy()
        arcs_b, offs_b = gb["arc"].to_numpy(), gb["_off"].to_numpy()
        m = np.empty((len(arcs_a), len(arcs_b)))
        for i, (aa, oa) in enumerate(zip(arcs_a, offs_a)):
            for j, (ab, ob) in enumerate(zip(arcs_b, offs_b)):
                d_route = graph.route_distance(int(aa), oa, int(ab), ob, config.backtrack_tol_m)
                m[i, j] = (-math.log(beta) - abs(d_route - d_gc) / beta) if np.isfinite(d_route) else -np.inf
        log_tr.append(m)

    res = viterbi(log_em, log_tr)
    rows = []
    for seg_id, (s, e) in enumerate(res.segments):
        for step in range(s, e + 1):
            pid, g = groups[step]
            r = g.iloc[res.path[step]]
            rows.append({"point_idx": pid, "arc": int(r["arc"]), "frac_along": float(r["frac_along"]),
                         "dist_m": float(r["dist_m"]), "segment": seg_id,
                         "proj_x": float(r["proj_x"]), "proj_y": float(r["proj_y"])})
    matched = _matched_frame(trip, rows)
    route, unfilled = reconstruct_route(graph, matched)
    return MatchResult(name, matched, route, n_points, n_points - len(groups), res.n_breaks,
                       unfilled, (time.perf_counter() - t0) * 1e3)


def classifier_only_decode(trip: Trip, cand: pd.DataFrame, proba: np.ndarray,
                           graph: RoadGraph | None = None, sp_limit_m: float = 5000.0) -> MatchResult:
    """Baseline: each fix independently takes its highest-probability candidate."""
    t0 = time.perf_counter()
    graph = graph or RoadGraph(trip, sp_limit_m)
    n_points = len(trip.track)
    if cand.empty:
        return MatchResult("classifier_only", _matched_frame(trip, []), [], n_points, n_points, 0, 0, 0.0)
    c = cand.assign(_p=proba)
    best = c.loc[c.groupby("point_idx")["_p"].idxmax()].sort_values("point_idx")
    rows = [{"point_idx": int(r.point_idx), "arc": int(r.arc), "frac_along": float(r.frac_along),
             "dist_m": float(r.dist_m), "segment": 0, "proj_x": float(r.proj_x), "proj_y": float(r.proj_y)}
            for r in best.itertuples()]
    matched = _matched_frame(trip, rows)
    route, unfilled = reconstruct_route(graph, matched)
    return MatchResult("classifier_only", matched, route, n_points, n_points - len(best), 0,
                       unfilled, (time.perf_counter() - t0) * 1e3)


def run_decoder(decoder: str, trip: Trip, cand: pd.DataFrame, config: HMMConfig,
                model_artifact: dict | None = None, graph: RoadGraph | None = None,
                proba: np.ndarray | None = None) -> MatchResult:
    graph = graph or RoadGraph(trip, config.sp_limit_m)
    if decoder == "hmm_gaussian":
        return hmm_decode(trip, cand, config, None, graph)
    if model_artifact is None and proba is None:
        raise ValueError(f"decoder {decoder!r} needs a trained model")
    if proba is None:
        proba = predict_candidate_proba(model_artifact, cand)
    if decoder == "hmm_classifier":
        return hmm_decode(trip, cand, config, proba, graph)
    if decoder == "classifier_only":
        return classifier_only_decode(trip, cand, proba, graph)
    raise ValueError(f"unknown decoder {decoder!r}; choose from {DECODERS}")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_match(result: MatchResult, trip: Trip, points: pd.DataFrame) -> dict:
    """
    point_accuracy        : share of labelled fixes matched to their true arc
    point_accuracy_reachable : same, only over fixes whose true arc was a candidate
                            (separates decoder errors from candidate-generation misses)
    route_precision/recall/f1 : length-weighted overlap of predicted vs true arc sets
    route_mismatch_fraction : (length wrongly added + length wrongly missed) / true length
                            (Newson & Krumm's metric; 0 is perfect, can exceed 1)
    """
    out: dict = {}
    lab = points[points["label_arc"] >= 0][["point_idx", "label_arc", "has_true_candidate"]]
    pred = result.matched[["point_idx", "arc"]] if not result.matched.empty else pd.DataFrame(columns=["point_idx", "arc"])
    j = lab.merge(pred, on="point_idx", how="left")
    correct = (j["arc"] == j["label_arc"])
    out["point_accuracy"] = float(correct.mean()) if len(j) else float("nan")
    reach = j["has_true_candidate"].astype(bool)
    out["point_accuracy_reachable"] = float(correct[reach].mean()) if reach.any() else float("nan")
    out["candidate_recall"] = float(reach.mean()) if len(j) else float("nan")

    if trip.route is not None and len(trip.route):
        L = trip.arc_len
        true_set, pred_set = set(map(int, trip.route)), set(result.route_arcs)
        tp = sum(L[a] for a in true_set & pred_set)
        fp = sum(L[a] for a in pred_set - true_set)
        fn = sum(L[a] for a in true_set - pred_set)
        true_len = sum(L[a] for a in true_set)
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        out.update(route_precision=p, route_recall=r,
                   route_f1=(2 * p * r / (p + r)) if p + r > 0 else 0.0,
                   route_mismatch_fraction=(fp + fn) / true_len if true_len > 0 else float("nan"))
    out.update(n_breaks=result.n_breaks, gaps_unfilled=result.gaps_unfilled,
               n_unmatched=result.n_unmatched, runtime_ms=result.runtime_ms)
    return out


def calibrate(trips: list[Trip], points_list: list[pd.DataFrame]) -> HMMConfig:
    """
    Estimate sigma_z and beta from labelled trips (Newson & Krumm, section 4):
      sigma_z = 1.4826 * median(distance GPS fix -> true arc)   (robust std-dev)
      beta    = median(|d_route - d_gc|) / ln 2  on true consecutive pairs
    Use TRAINING trips only - calibrating on test trips leaks information.
    """
    dists, diffs = [], []
    for trip, pts in zip(trips, points_list):
        lab = pts[pts["label_arc"] >= 0]
        dists.extend(lab["label_dist_m"].dropna().tolist())
        if len(lab) < 2:
            continue
        g = RoadGraph(trip)
        ax, ay, bx, by = (trip.nodes_xy[trip.arcs[:, 0], 0], trip.nodes_xy[trip.arcs[:, 0], 1],
                          trip.nodes_xy[trip.arcs[:, 1], 0], trip.nodes_xy[trip.arcs[:, 1], 1])
        x, y = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
        idx = lab["point_idx"].to_numpy()
        arcs = lab["label_arc"].to_numpy()
        dx, dy = bx[arcs] - ax[arcs], by[arcs] - ay[arcs]
        l2 = np.maximum(dx * dx + dy * dy, 1e-9)
        frac = np.clip(((x[idx] - ax[arcs]) * dx + (y[idx] - ay[arcs]) * dy) / l2, 0, 1)
        off = frac * trip.arc_len[arcs]
        for k in range(len(idx) - 1):
            dr = g.route_distance(int(arcs[k]), off[k], int(arcs[k + 1]), off[k + 1])
            if np.isfinite(dr):
                diffs.append(abs(dr - math.hypot(x[idx[k + 1]] - x[idx[k]], y[idx[k + 1]] - y[idx[k]])))
    cfg = HMMConfig()
    if dists:
        cfg.sigma_z = max(1.0, 1.4826 * float(np.median(dists)))
    if diffs:
        cfg.beta = max(1.0, float(np.median(diffs)) / math.log(2))
    return cfg


# ----------------------------------------------------------------------------
# CLI: classifier-only vs HMM comparison
# ----------------------------------------------------------------------------
def compare_decoders(data_dir: str, model_path: str | None, max_trips: int | None,
                     out_dir: str = "artifacts", calib_trips: int = 20) -> pd.DataFrame:
    import joblib

    index = list_trips(data_dir)
    artifact = joblib.load(model_path) if model_path and Path(model_path).exists() else None
    if artifact and artifact.get("test_trip_ids"):
        eval_ids = [t for t in artifact["test_trip_ids"] if t in index]
        train_ids = [t for t in artifact.get("train_trip_ids", []) if t in index]
        print(f"Evaluating on {len(eval_ids)} held-out trips saved in the model artifact.")
    else:
        ids = list(index)
        train_ids, eval_ids = ids[:calib_trips], ids[calib_trips:]
        print("No held-out split in the artifact - using a simple first-N calibration split.")
    if max_trips:
        eval_ids = eval_ids[:max_trips]

    # calibrate sigma_z / beta on training trips only
    calib = []
    for tid in train_ids[:calib_trips]:
        tr = load_trip(index[tid], tid)
        calib.append((tr, build_trip_frame(tr)[0]))
    config = calibrate([c[0] for c in calib], [c[1] for c in calib]) if calib else HMMConfig()
    print(f"Calibrated: sigma_z={config.sigma_z:.2f} m, beta={config.beta:.2f} m")

    decoders = DECODERS if artifact else ("hmm_gaussian",)
    rows = []
    for i, tid in enumerate(eval_ids, 1):
        trip = load_trip(index[tid], tid)
        points, cand = build_trip_frame(trip)
        graph = RoadGraph(trip, config.sp_limit_m)
        proba = predict_candidate_proba(artifact, cand) if artifact else None
        for dec in decoders:
            res = run_decoder(dec, trip, cand, config, artifact, graph, proba)
            rows.append({"trip_id": tid, "decoder": dec, **evaluate_match(res, trip, points)})
        if i % 10 == 0 or i == len(eval_ids):
            print(f"  decoded {i}/{len(eval_ids)} trips")

    df = pd.DataFrame(rows)
    summary = df.groupby("decoder").agg(
        point_accuracy=("point_accuracy", "mean"),
        point_acc_reachable=("point_accuracy_reachable", "mean"),
        route_f1=("route_f1", "mean"),
        route_mismatch=("route_mismatch_fraction", "mean"),
        breaks_per_trip=("n_breaks", "mean"),
        runtime_ms=("runtime_ms", "mean"),
    ).sort_values("route_f1", ascending=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "decoder_comparison_per_trip.csv", index=False)
    summary.to_csv(Path(out_dir) / "decoder_comparison_summary.csv")
    with open(Path(out_dir) / "hmm_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print("\n", summary.round(4).to_string())
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare classifier-only vs HMM map matching.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="artifacts/model_5.joblib",
                    help="any model artifact in the shared format (e.g. the tuned best model)")
    ap.add_argument("--max-trips", type=int, default=None)
    ap.add_argument("--out-dir", default="artifacts")
    a = ap.parse_args()
    compare_decoders(a.data_dir, a.model, a.max_trips, a.out_dir)
