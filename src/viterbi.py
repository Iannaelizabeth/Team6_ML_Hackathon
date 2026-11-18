"""
viterbi.py
==========
Log-space Viterbi decoding for an HMM whose number of hidden states changes at
every time step (in map matching: a different set of candidate road segments
for each GPS fix).

Terms
-----
hidden state   : the thing we cannot observe directly - here, which road segment the
                 vehicle is really on at time t.
emission       : P(observation | state) - how likely the GPS fix is if the vehicle
                 were on that segment.
transition     : P(state_t | state_{t-1}) - how plausible it is to drive from one
                 candidate segment to the next between two fixes.
Viterbi        : dynamic programming that finds the single most likely sequence of
                 hidden states in O(T x K^2) instead of trying all K^T sequences.
log space      : we add log-probabilities instead of multiplying probabilities, so long
                 trips do not underflow to 0.0.
break          : if NO state at step t can be reached from ANY state at t-1 (all
                 transitions are -inf, e.g. disconnected road segments or a long GPS
                 gap), the chain is split and decoding restarts at t. Without this,
                 one impossible step would make the whole trip's probability -inf.

This module is pure algorithm - it knows nothing about roads. hmm.py builds the
emission and transition arrays and calls viterbi().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEG_INF = -np.inf


@dataclass
class ViterbiResult:
    path: np.ndarray                         # (T,) chosen state index at every step
    segments: list[tuple[int, int]] = field(default_factory=list)  # inclusive (start, end) per unbroken chain
    log_score: float = 0.0                   # sum of best log-scores over segments

    @property
    def n_breaks(self) -> int:
        return max(0, len(self.segments) - 1)


def _backtrack(path: np.ndarray, backs: list, start: int, end: int, delta_end: np.ndarray) -> float:
    k = int(np.argmax(delta_end))
    best = float(delta_end[k])
    path[end] = k
    for s in range(end, start, -1):
        k = int(backs[s][k])
        path[s - 1] = k
    return best


def viterbi(log_emissions: list[np.ndarray],
            log_transitions: list[np.ndarray | None],
            log_initial: np.ndarray | None = None) -> ViterbiResult:
    """
    Parameters
    ----------
    log_emissions   : list of length T; element t is a 1-D array (K_t,) of log P(obs_t | state).
    log_transitions : list of length T-1; element t is a (K_t, K_{t+1}) array of
                      log P(state_{t+1}=j | state_t=i), or None to force a break
                      between step t and t+1.
    log_initial     : optional (K_0,) log prior for the first step (default: uniform).

    Returns
    -------
    ViterbiResult with the best state index at every step and the unbroken segments.
    """
    T = len(log_emissions)
    if T == 0:
        return ViterbiResult(np.zeros(0, dtype=int), [], 0.0)
    if len(log_transitions) != T - 1:
        raise ValueError(f"need {T - 1} transition matrices, got {len(log_transitions)}")

    path = np.zeros(T, dtype=int)
    backs: list[np.ndarray | None] = [None] * T
    segments: list[tuple[int, int]] = []
    total = 0.0

    delta = np.asarray(log_emissions[0], float).copy()
    if log_initial is not None:
        delta = delta + log_initial
    seg_start = 0

    for t in range(1, T):
        em = np.asarray(log_emissions[t], float)
        tr = log_transitions[t - 1]
        broken = tr is None
        if not broken:
            tr = np.asarray(tr, float)
            if tr.shape != (delta.shape[0], em.shape[0]):
                raise ValueError(f"transition {t - 1} has shape {tr.shape}, "
                                 f"expected {(delta.shape[0], em.shape[0])}")
            scores = delta[:, None] + tr                    # (K_prev, K_t)
            best_prev = np.argmax(scores, axis=0)           # (K_t,)
            best = scores[best_prev, np.arange(em.shape[0])]
            broken = not np.isfinite(best).any()

        if broken:
            total += _backtrack(path, backs, seg_start, t - 1, delta)
            segments.append((seg_start, t - 1))
            seg_start = t
            delta = em.copy()
            continue

        delta = best + em
        backs[t] = best_prev

    total += _backtrack(path, backs, seg_start, T - 1, delta)
    segments.append((seg_start, T - 1))
    return ViterbiResult(path, segments, total)


# ----------------------------------------------------------------------------
# Self-test: textbook example (Wikipedia "Viterbi algorithm" - healthy/fever)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    states = ["Healthy", "Fever"]
    start = np.log([0.6, 0.4])
    trans = np.log([[0.7, 0.3], [0.4, 0.6]])
    emit = {"normal": [0.5, 0.1], "cold": [0.4, 0.3], "dizzy": [0.1, 0.6]}
    obs = ["normal", "cold", "dizzy"]

    res = viterbi([np.log(emit[o]) for o in obs], [trans, trans], start)
    decoded = [states[i] for i in res.path]
    print("decoded:", decoded, " probability:", round(float(np.exp(res.log_score)), 5))
    assert decoded == ["Healthy", "Healthy", "Fever"]
    assert abs(np.exp(res.log_score) - 0.01512) < 1e-6

    # break handling: step 1 -> 2 impossible, so the chain must split into two segments
    impossible = np.full((2, 2), NEG_INF)
    res2 = viterbi([np.log(emit[o]) for o in obs], [trans, impossible], start)
    assert res2.segments == [(0, 1), (2, 2)] and res2.n_breaks == 1
    print("break handling OK:", res2.segments)
    print("viterbi.py self-test passed")
