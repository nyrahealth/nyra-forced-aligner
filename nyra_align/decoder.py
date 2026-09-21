"""GMM emissions + beam Viterbi over the decode graph.

Score semantics replicate `gmm-align-compiled` with its default scales:
total cost = sum over frames of
    acoustic_scale * (-loglike(pdf_of_occupied_state, frame))
  + 1.0 * (-log transition prob)   for every self-loop / forward transition,
plus the trained exit transition after the last frame. Beam pruning keeps
states within `beam` of the frame's best cost; the caller retries with wider
beams on failure (same ladder as the native pipeline).
"""

from __future__ import annotations

import numpy as np
import torch

from .graph import DecodeGraph


class EmissionComputer:
    """Batched diag-GMM log-likelihoods for the pdfs used in one graph."""

    def __init__(self, model, unique_pdfs: np.ndarray, device: str = "cpu"):
        self.device = device
        rows = []
        seg = []
        for j, pdf in enumerate(unique_pdfs):
            lo, hi = int(model.pdf_offsets[pdf]), int(model.pdf_offsets[pdf + 1])
            rows.extend(range(lo, hi))
            seg.extend([j] * (hi - lo))
        rows = np.asarray(rows, dtype=np.int64)
        self.n_pdfs = len(unique_pdfs)
        self.gconsts = torch.from_numpy(model.gconsts[rows]).to(device)
        self.miv = torch.from_numpy(model.means_invvars[rows]).to(device)
        self.neg_half_iv = torch.from_numpy(
            -0.5 * model.inv_vars[rows]).to(device)
        self.seg = torch.from_numpy(np.asarray(seg, dtype=np.int64)).to(device)

    def loglikes(self, feats: np.ndarray) -> np.ndarray:
        """(B, D) features -> (B, n_pdfs) log-likelihoods (float32)."""
        with torch.no_grad():
            x = torch.from_numpy(feats).to(self.device)
            s = (x @ self.miv.T + (x * x) @ self.neg_half_iv.T
                 + self.gconsts)                       # (B, G)
            B = s.shape[0]
            seg = self.seg.unsqueeze(0).expand(B, -1)
            m = torch.full((B, self.n_pdfs), float("-inf"),
                           device=self.device, dtype=s.dtype)
            m.scatter_reduce_(1, seg, s, reduce="amax")
            e = torch.exp(s - m.gather(1, seg))
            tot = torch.zeros((B, self.n_pdfs), device=self.device,
                              dtype=s.dtype)
            tot.scatter_add_(1, seg, e)
            ll = m + torch.log(tot)
            return ll.cpu().numpy().astype(np.float32)


class _BlockedEmissions:
    """Lazy per-block emission COSTS (= -acoustic_scale * loglike)."""

    def __init__(self, feats: np.ndarray, computer: EmissionComputer,
                 acoustic_scale: float, block: int = 4096):
        self.feats = feats
        self.computer = computer
        self.scale = acoustic_scale
        self.block = block
        self._cache_start = -1
        self._cache: np.ndarray | None = None

    def row(self, t: int) -> np.ndarray:
        b0 = (t // self.block) * self.block
        if b0 != self._cache_start:
            b1 = min(b0 + self.block, len(self.feats))
            ll = self.computer.loglikes(self.feats[b0:b1])
            self._cache = (-self.scale * ll).astype(np.float32)
            self._cache_start = b0
        return self._cache[t - self._cache_start]


def viterbi(graph: DecodeGraph, emissions: _BlockedEmissions,
            n_frames: int, beam: float):
    """Beam Viterbi. Returns per-frame state path (np.int32 (T,)) or None."""
    S = len(graph.pdf)
    order = np.argsort(graph.arc_src, kind="stable")
    a_src = graph.arc_src[order]
    a_dst = graph.arc_dst[order]
    a_cost = graph.arc_cost[order]
    arc_start = np.searchsorted(a_src, np.arange(S))
    arc_end = np.searchsorted(a_src, np.arange(S) + 1)
    arc_cnt = (arc_end - arc_start).astype(np.int64)

    scratch = np.full(S, np.inf, dtype=np.float32)
    last_seen = np.full(S, -1, dtype=np.int64)

    active = graph.start_states.astype(np.int64)
    e0 = emissions.row(0)
    cost_active = e0[graph.pdf_compact[active]].astype(np.float32)

    bp_dst: list[np.ndarray] = [np.empty(0, dtype=np.int32)]
    bp_src: list[np.ndarray] = [np.empty(0, dtype=np.int32)]

    for t in range(1, n_frames):
        # ---- candidate transitions from the active set --------------------
        stay_score = cost_active + graph.self_cost[active]

        cnt = arc_cnt[active]
        total = int(cnt.sum())
        if total:
            starts = arc_start[active]
            offs = np.arange(total, dtype=np.int64) - np.repeat(
                np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
            flat = np.repeat(starts, cnt) + offs
            move_dst = a_dst[flat].astype(np.int64)
            move_score = np.repeat(cost_active, cnt) + a_cost[flat]
            move_src = np.repeat(active, cnt)
            dsts = np.concatenate((active, move_dst))
            scores = np.concatenate((stay_score, move_score))
            srcs = np.concatenate((active, move_src))
        else:
            dsts, scores, srcs = active, stay_score, active

        # ---- min-reduce per destination ------------------------------------
        fresh = last_seen[dsts] != t
        scratch[dsts[fresh]] = np.inf
        last_seen[dsts] = t
        np.minimum.at(scratch, dsts, scores)

        win = scores <= scratch[dsts]
        bp_dst.append(dsts[win].astype(np.int32))
        bp_src.append(srcs[win].astype(np.int32))

        # NOT dsts[fresh]: the vectorised mask flags every occurrence of a newly
        # seen state, so duplicates would enter (and double) the active set
        new_active = np.unique(dsts)
        e = emissions.row(t)
        new_cost = (scratch[new_active]
                    + e[graph.pdf_compact[new_active]]).astype(np.float32)

        # ---- beam prune -----------------------------------------------------
        m = float(new_cost.min())
        if not np.isfinite(m):
            return None
        keep = new_cost <= m + beam
        active = new_active[keep]
        cost_active = new_cost[keep]

    # ---- terminate -----------------------------------------------------------
    final_cost = np.full(S, np.inf, dtype=np.float32)
    final_cost[graph.final_states] = graph.final_costs
    totals = cost_active + final_cost[active]
    if not np.isfinite(totals).any():
        return None
    best = int(active[int(np.argmin(totals))])

    # ---- traceback ------------------------------------------------------------
    path = np.empty(n_frames, dtype=np.int32)
    s = best
    path[n_frames - 1] = s
    for t in range(n_frames - 1, 0, -1):
        hits = np.flatnonzero(bp_dst[t] == s)
        if len(hits) == 0:
            raise RuntimeError(f"traceback broken at frame {t}")
        s = int(bp_src[t][hits[0]])
        path[t - 1] = s
    return path


def align_path(graph: DecodeGraph, path: np.ndarray,
               frame_shift: float):
    """State path -> phone segments [(token_idx, phone_id, start_s, end_s)]."""
    inst = graph.state_instance[path]
    changes = np.flatnonzero(np.diff(inst)) + 1
    bounds = np.concatenate(([0], changes, [len(path)]))
    segs = []
    for i in range(len(bounds) - 1):
        f0, f1 = int(bounds[i]), int(bounds[i + 1])
        st = path[f0]
        segs.append((int(graph.state_token[st]), int(graph.state_phone[st]),
                     f0 * frame_shift, f1 * frame_shift))
    return segs


def decode_with_ladder(graph: DecodeGraph, feats: np.ndarray,
                       computer: EmissionComputer, acoustic_scale: float,
                       beam_ladder) -> np.ndarray | None:
    emissions = _BlockedEmissions(feats, computer, acoustic_scale)
    n = len(feats)
    for beam, retry in beam_ladder:
        for b in (beam, retry):
            path = viterbi(graph, emissions, n, float(b))
            if path is not None:
                return path
    return None
