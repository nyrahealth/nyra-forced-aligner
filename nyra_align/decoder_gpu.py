"""Dense GPU Viterbi: exact (no pruning), single pass, entirely on device.

Because every state is kept alive each frame, there is no beam and no
failure/retry ladder — the returned path is the true Viterbi optimum of the
graph (what native Kaldi converges to at wide beams). Backpointers are stored
as per-state ordinals of the winning incoming arc (int16), so memory is
T x S x 2 bytes; the traceback walks entirely on the GPU and transfers only
the final path.

The per-frame update (about a dozen small kernels) is captured into a CUDA
graph when possible, collapsing kernel-launch overhead; otherwise it runs as
an eager loop with identical semantics.
"""

from __future__ import annotations

import numpy as np
import torch

from .graph import DecodeGraph

_INF = float("inf")
_MAX_BP_BYTES = 8 << 30       # fall back to CPU decoder beyond this


def _emission_costs(feats_t: torch.Tensor, computer, acoustic_scale: float,
                    block: int = 8192) -> torch.Tensor:
    """(T, D) features on device -> (T, P') emission COSTS on device."""
    outs = []
    for b0 in range(0, feats_t.shape[0], block):
        x = feats_t[b0:b0 + block]
        with torch.no_grad():
            s = (x @ computer.miv.T + (x * x) @ computer.neg_half_iv.T
                 + computer.gconsts)
            B = s.shape[0]
            seg = computer.seg.unsqueeze(0).expand(B, -1)
            m = torch.full((B, computer.n_pdfs), -_INF,
                           device=x.device, dtype=s.dtype)
            m.scatter_reduce_(1, seg, s, reduce="amax")
            e = torch.exp(s - m.gather(1, seg))
            tot = torch.zeros((B, computer.n_pdfs), device=x.device,
                              dtype=s.dtype)
            tot.scatter_add_(1, seg, e)
            outs.append(-acoustic_scale * (m + torch.log(tot)))
    return torch.cat(outs, dim=0).float()


class _Step:
    """One Viterbi frame update over static buffers (CUDA-graph friendly)."""

    def __init__(self, dev, E, pdfc, src, dst, w, ordinal, bp, S):
        self.E, self.pdfc = E, pdfc
        self.src, self.dst, self.w = src, dst, w
        self.ordinal, self.bp = ordinal, bp
        self.S = S
        self.cost = torch.full((S,), _INF, device=dev)
        self.tmp = torch.empty(S, device=dev)
        self.bp_t = torch.empty(S, dtype=torch.int16, device=dev)
        self.t_idx = torch.zeros(1, dtype=torch.long, device=dev)
        self.big_ord = torch.tensor([32767], dtype=torch.int16, device=dev)

    def run(self):
        scores = self.cost.index_select(0, self.src) + self.w
        self.tmp.fill_(_INF)
        self.tmp.scatter_reduce_(0, self.dst, scores, reduce="amin")
        win = scores <= self.tmp.index_select(0, self.dst)
        cand = torch.where(win, self.ordinal, self.big_ord)
        self.bp_t.fill_(32767)
        self.bp_t.scatter_reduce_(0, self.dst, cand, reduce="amin")
        self.bp.index_copy_(0, self.t_idx, self.bp_t.unsqueeze(0))
        e_t = self.E.index_select(0, self.t_idx).squeeze(0)
        self.cost.copy_(self.tmp + e_t.index_select(0, self.pdfc))
        self.t_idx.add_(1)


def viterbi_gpu(graph: DecodeGraph, feats_t: torch.Tensor, computer,
                acoustic_scale: float) -> np.ndarray | None:
    dev = feats_t.device
    T = feats_t.shape[0]
    S = len(graph.pdf)
    if T * S * 2 > _MAX_BP_BYTES:
        return None  # caller falls back to the CPU active-set decoder

    # arcs incl. self-loops, sorted by destination; per-dst winner ordinals
    src = np.concatenate((graph.arc_src, np.arange(S, dtype=np.int32)))
    dst = np.concatenate((graph.arc_dst, np.arange(S, dtype=np.int32)))
    w = np.concatenate((graph.arc_cost, graph.self_cost))
    order = np.argsort(dst, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    dst_start = np.searchsorted(dst, np.arange(S + 1)).astype(np.int64)
    ordinal = (np.arange(len(dst)) - dst_start[dst]).astype(np.int16)
    if int((dst_start[1:] - dst_start[:-1]).max()) > 32000:
        return None

    E = _emission_costs(feats_t, computer, acoustic_scale)
    t_src = torch.from_numpy(src.astype(np.int64)).to(dev)
    t_dst = torch.from_numpy(dst.astype(np.int64)).to(dev)
    t_w = torch.from_numpy(w.astype(np.float32)).to(dev)
    t_ord = torch.from_numpy(ordinal).to(dev)
    t_pdfc = torch.from_numpy(graph.pdf_compact.astype(np.int64)).to(dev)
    bp = torch.empty((T, S), dtype=torch.int16, device=dev)

    step = _Step(dev, E, t_pdfc, t_src, t_dst, t_w, t_ord, bp, S)

    # frame 0: start states pay their first emission, nothing else
    starts = torch.from_numpy(graph.start_states.astype(np.int64)).to(dev)
    e0 = E[0].index_select(0, t_pdfc.index_select(0, starts))
    step.cost.fill_(_INF)
    step.cost.scatter_(0, starts, e0)
    step.t_idx.fill_(1)

    n_steps = T - 1
    if n_steps > 0:
        # graph capture costs ~0.2 s once; worth it only for long decodes
        use_graph = (dev.type == "cuda" and n_steps > 3000)
        captured = None
        saved_cost = saved_t = None
        if use_graph:
            try:
                s_stream = torch.cuda.Stream()
                s_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_stream):
                    saved_cost = step.cost.clone()
                    saved_t = step.t_idx.clone()
                    for _ in range(3):        # warmup
                        step.run()
                    step.cost.copy_(saved_cost)
                    step.t_idx.copy_(saved_t)
                torch.cuda.current_stream().wait_stream(s_stream)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    step.run()
                captured = g
            except Exception:
                if saved_cost is not None:
                    torch.cuda.synchronize()
                    step.cost.copy_(saved_cost)
                    step.t_idx.copy_(saved_t)
                captured = None
        if captured is not None:
            for _ in range(n_steps):
                captured.replay()
        else:
            for _ in range(n_steps):
                step.run()

    # terminate
    fin = torch.from_numpy(graph.final_states.astype(np.int64)).to(dev)
    fin_cost = torch.from_numpy(graph.final_costs).to(dev)
    totals = step.cost.index_select(0, fin) + fin_cost
    best_i = int(torch.argmin(totals).item())
    if not np.isfinite(float(totals[best_i].item())):
        return None

    # traceback on device: pos_{t-1} = src[dst_start[pos_t] + bp[t, pos_t]]
    t_dst_start = torch.from_numpy(dst_start).to(dev)
    path = torch.empty(T, dtype=torch.long, device=dev)
    pos = fin[best_i].reshape(1)
    path[T - 1] = pos
    for t in range(T - 1, 0, -1):
        o = bp[t].index_select(0, pos).long()
        arc = t_dst_start.index_select(0, pos) + o
        pos = t_src.index_select(0, arc)
        path[t - 1] = pos
    return path.cpu().numpy().astype(np.int32)
