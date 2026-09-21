"""Numba-JIT active-set beam Viterbi — same semantics as decoder.viterbi
(and native gmm-align-compiled), compiled to machine code. Emissions are
computed on the GPU (or CPU torch) up front; the DP loop then runs at a few
microseconds per frame instead of ~150 us of interpreter/kernel dispatch.

Optional: importing this module requires numba; callers fall back to the
pure-numpy decoder when it is unavailable.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_INF = np.float32(np.inf)


@njit(cache=True, fastmath=False)
def _decode(E, pdfc, self_cost, arc_start, a_dst, a_cost,
            starts, final_cost_dense, beam):
    T = E.shape[0]
    S = pdfc.shape[0]
    empty = np.empty(0, dtype=np.int32)

    last_seen = np.full(S, -1, dtype=np.int64)
    scratch = np.empty(S, dtype=np.float32)
    slot = np.empty(S, dtype=np.int64)
    active = np.empty(S, dtype=np.int64)
    cost_act = np.empty(S, dtype=np.float32)
    frame_dst = np.empty(S, dtype=np.int64)
    frame_src = np.empty(S, dtype=np.int64)
    costbuf = np.empty(S, dtype=np.float32)

    n_act = 0
    for i in range(starts.shape[0]):
        s = starts[i]
        active[n_act] = s
        cost_act[n_act] = E[0, pdfc[s]]
        n_act += 1

    cap = 1024 + T * 8
    wd = np.empty(cap, dtype=np.int32)
    ws = np.empty(cap, dtype=np.int32)
    off = np.empty(T + 1, dtype=np.int64)
    off[0] = 0
    off[1] = 0
    wpos = 0

    for t in range(1, T):
        n_new = 0
        for i in range(n_act):
            src = active[i]
            c0 = cost_act[i]
            # stay (self-loop)
            sc = c0 + self_cost[src]
            d = src
            if last_seen[d] != t:
                last_seen[d] = t
                slot[d] = n_new
                frame_dst[n_new] = d
                frame_src[n_new] = src
                scratch[d] = sc
                n_new += 1
            elif sc < scratch[d]:
                scratch[d] = sc
                frame_src[slot[d]] = src
            # forward arcs
            for k in range(arc_start[src], arc_start[src + 1]):
                d = a_dst[k]
                sc = c0 + a_cost[k]
                if last_seen[d] != t:
                    last_seen[d] = t
                    slot[d] = n_new
                    frame_dst[n_new] = d
                    frame_src[n_new] = src
                    scratch[d] = sc
                    n_new += 1
                elif sc < scratch[d]:
                    scratch[d] = sc
                    frame_src[slot[d]] = src

        best = _INF
        for i in range(n_new):
            d = frame_dst[i]
            c = scratch[d] + E[t, pdfc[d]]
            costbuf[i] = c
            if c < best:
                best = c
        if best == _INF:
            return 1, empty

        if wpos + n_new > cap:
            newcap = cap * 2
            while wpos + n_new > newcap:
                newcap *= 2
            nwd = np.empty(newcap, dtype=np.int32)
            nws = np.empty(newcap, dtype=np.int32)
            nwd[:wpos] = wd[:wpos]
            nws[:wpos] = ws[:wpos]
            wd, ws, cap = nwd, nws, newcap

        thr = best + beam
        n2 = 0
        for i in range(n_new):
            d = frame_dst[i]
            wd[wpos] = d
            ws[wpos] = frame_src[i]
            wpos += 1
            if costbuf[i] <= thr:
                active[n2] = d
                cost_act[n2] = costbuf[i]
                n2 += 1
        off[t + 1] = wpos
        n_act = n2
        if n_act == 0:
            return 1, empty

    best_tot = _INF
    best_s = np.int64(-1)
    for i in range(n_act):
        s = active[i]
        tot = cost_act[i] + final_cost_dense[s]
        if tot < best_tot:
            best_tot = tot
            best_s = s
    if best_s < 0:
        return 2, empty

    path = np.empty(T, dtype=np.int32)
    cur = best_s
    path[T - 1] = np.int32(cur)
    for t in range(T - 1, 0, -1):
        found = np.int64(-1)
        for k in range(off[t], off[t + 1]):
            if wd[k] == cur:
                found = np.int64(ws[k])
                break
        if found < 0:
            return 3, empty
        cur = found
        path[t - 1] = np.int32(cur)
    return 0, path


class JitDecoder:
    """Prepares graph arrays once, decodes with the JIT kernel per beam."""

    def __init__(self, graph):
        S = len(graph.pdf)
        order = np.argsort(graph.arc_src, kind="stable")
        self.a_dst = graph.arc_dst[order].astype(np.int64)
        self.a_cost = graph.arc_cost[order].astype(np.float32)
        a_src = graph.arc_src[order]
        self.arc_start = np.searchsorted(
            a_src, np.arange(S + 1)).astype(np.int64)
        self.pdfc = graph.pdf_compact.astype(np.int64)
        self.self_cost = graph.self_cost.astype(np.float32)
        self.starts = graph.start_states.astype(np.int64)
        self.final_cost = np.full(S, np.inf, dtype=np.float32)
        self.final_cost[graph.final_states] = graph.final_costs

    def decode(self, E: np.ndarray, beam: float) -> np.ndarray | None:
        status, path = _decode(
            E, self.pdfc, self.self_cost, self.arc_start, self.a_dst,
            self.a_cost, self.starts, self.final_cost, np.float64(beam))
        return path if status == 0 else None
