"""Build the utterance decode graph (pure-Python replacement for
compile-train-graphs).

Semantics replicated from the training pipeline's hand-built L.fst composed
with a linear word acceptor and the triphone context/H transducers:

  * words in the given order, each a linear chain of its pronunciation phones
    (alternative pronunciations become parallel branches);
  * optional silence: at every word junction (including start and end) any
    number of SIL passes is allowed (SIL self-loop on L's state 0);
  * full triphone context across junctions: the last phone of a word sees
    either SIL or the first phone of the next word as right context (and
    symmetrically for left contexts), exactly as Kaldi's C composition does;
    utterance edges pad the context window with 0;
  * each context-dependent phone expands to its HMM states with pdfs from the
    decision-tree table and trained transition costs.

The result is a state-level graph for the occupancy-form Viterbi in
decoder.py: emitting states with a self-loop cost, forward arcs with costs,
entry states, and final states with exit costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import NyraModel


@dataclass
class Token:
    """One transcript token to align (word or event tag)."""
    word: str                      # lexicon form actually aligned
    original: str                  # raw token as given by the user
    prons: list[list[str]]         # alternative pronunciations (phone names)
    index: int                     # token position in the transcript


@dataclass
class DecodeGraph:
    # per emitting state
    pdf: np.ndarray            # int32 (S,)
    self_cost: np.ndarray      # float32 (S,)
    # forward arcs between emitting states (cross-instance arcs collapsed)
    arc_src: np.ndarray        # int32 (A,)
    arc_dst: np.ndarray        # int32 (A,)
    arc_cost: np.ndarray       # float32 (A,)
    start_states: np.ndarray   # int32 — may begin consuming at frame 0
    final_states: np.ndarray   # int32
    final_costs: np.ndarray    # float32 — exit cost to reach the final node
    # traceback metadata per state
    state_token: np.ndarray    # int32 token index (-1 = silence)
    state_phone: np.ndarray    # int32 phone id
    state_instance: np.ndarray  # int32 unique phone-instance id
    unique_pdfs: np.ndarray    # sorted unique pdf ids used in the graph
    pdf_compact: np.ndarray    # (S,) index into unique_pdfs per state


class _Lattice:
    """Phone-level lattice: nodes + arcs (phone, token_index)."""

    def __init__(self):
        self.arcs_from: dict[int, list[tuple[int, int, int, int]]] = {}
        self.n_nodes = 0

    def node(self) -> int:
        n = self.n_nodes
        self.n_nodes += 1
        self.arcs_from[n] = []
        return n

    def arc(self, src: int, dst: int, phone: int, token: int) -> int:
        aid = len(self._all)
        self._all.append((src, dst, phone, token))
        self.arcs_from[src].append((aid, dst, phone, token))
        return aid

    _all: list


def build_graph(tokens: list[Token], model: NyraModel,
                junction_phones: list[str] | None = None,
                junction_costs: dict[str, float] | None = None) -> DecodeGraph:
    """junction_phones: phones allowed in the optional loop between words
    (default: only the silence phone). junction_costs: extra entry cost
    (in log domain, added on the arc into that phone instance) per phone
    name — e.g. an insertion penalty for non-speech events."""
    sil = model.silence_id
    p2i = model.phone_to_id
    junction_ids = [p2i[p] for p in (junction_phones or [model.meta["silence_phone"]])]
    jcost = {p2i[p]: float(c) for p, c in (junction_costs or {}).items()}

    # ---- phone lattice ----------------------------------------------------
    lat = _Lattice()
    lat._all = []
    junctions = [lat.node()]
    for tok in tokens:
        nxt = lat.node()
        for pron in tok.prons:
            ids = [p2i[p] for p in pron]
            cur = junctions[-1]
            for k, pid in enumerate(ids):
                dst = nxt if k == len(ids) - 1 else lat.node()
                lat.arc(cur, dst, pid, tok.index)
                cur = dst
        junctions.append(nxt)
    for j in junctions:
        for jp in junction_ids:
            lat.arc(j, j, jp, -1)       # optional non-speech, repeatable
    start_node, end_node = junctions[0], junctions[-1]

    # ---- delayed triphone expansion ----------------------------------------
    # Expanded node: (lattice_node_after_pending, left_phone, pending_arc_id).
    # Taking lattice arc `b` from a node with pending arc `a` emits the HMM
    # instance for phone(a) with left=left_phone, right=phone(b).
    instances: list[tuple[int, int, int, int, int]] = []  # l, c, r, token, id
    inst_arcs: list[tuple[int, int]] = []   # (from_instance, to_instance)
    inst_start: list[int] = []              # instances that may start the path
    inst_final: list[int] = []              # instances that may end the path

    # node key -> dict of outgoing (arc_id -> instance id of the HMM emitted
    # when that arc is taken from this node)
    expanded: dict[tuple[int, int, int], dict[int, int]] = {}
    # instance id emitted when flushing at the end node
    flush_of: dict[tuple[int, int, int], int] = {}

    def new_instance(l: int, c: int, r: int, token: int) -> int:
        iid = len(instances)
        instances.append((l, c, r, token, iid))
        return iid

    stack: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    # Initial arcs: no pending phone yet; consume one lattice arc for free.
    init_keys: list[tuple[tuple[int, int, int], int]] = []
    for aid, dst, phone, token in lat.arcs_from[start_node]:
        key = (dst, 0, aid)
        init_keys.append((key, aid))
        if key not in seen:
            seen.add(key)
            stack.append(key)

    while stack:
        key = stack.pop()
        node, left, aid = key
        _src, _dst, c_phone, c_token = lat._all[aid]
        out: dict[int, int] = {}
        for bid, bdst, b_phone, _b_token in lat.arcs_from[node]:
            iid = new_instance(left, c_phone, b_phone, c_token)
            out[bid] = iid
            nkey = (bdst, c_phone, bid)
            if nkey not in seen:
                seen.add(nkey)
                stack.append(nkey)
        expanded[key] = out
        if node == end_node:
            iid = new_instance(left, c_phone, 0, c_token)
            flush_of[key] = iid

    # Wire instance arcs: taking arc b from key emits expanded[key][b]; the
    # successor key's emissions (including its flush) follow that instance.
    for key, out in expanded.items():
        node, left, aid = key
        _s, _d, c_phone, _t = lat._all[aid]
        for bid, iid in out.items():
            _bs, bdst, _bp, _bt = lat._all[bid]
            nkey = (bdst, c_phone, bid)
            for _cid, jid in expanded.get(nkey, {}).items():
                inst_arcs.append((iid, jid))
            if nkey in flush_of:
                inst_arcs.append((iid, flush_of[nkey]))
        if key in flush_of:
            inst_final.append(flush_of[key])

    # start instances: first emission after any initial arc
    for key, aid in init_keys:
        for _bid, iid in expanded.get(key, {}).items():
            inst_start.append(iid)
        if key in flush_of:
            inst_start.append(flush_of[key])

    if not instances:
        raise ValueError("empty decode graph (no alignable tokens)")

    # ---- expand instances to HMM states ------------------------------------
    pdf_l, self_l, tokmeta, phmeta, instmeta = [], [], [], [], []
    first_state: list[int] = []
    last_state: list[int] = []
    exit_cost: list[float] = []
    arc_src: list[int] = []
    arc_dst: list[int] = []
    arc_cost: list[float] = []

    table = model.pdf_table
    for (l, c, r, token, iid) in instances:
        states = model.topo[c]
        n_emit = len(states)
        base = len(pdf_l)
        first_state.append(base)
        last_state.append(base + n_emit - 1)
        prev_fwd = None
        for s in range(n_emit):
            pdf = int(table[l, c, r, states[s][0]])
            if pdf < 0:
                raise ValueError(
                    f"no pdf for context ({l},{c},{r}) class {states[s][0]}")
            _ts, self_cost, fwd_cost = model.transition_ids(c, s, pdf)
            pdf_l.append(pdf)
            self_l.append(self_cost)
            tokmeta.append(token)
            phmeta.append(c)
            instmeta.append(iid)
            if s > 0:
                arc_src.append(base + s - 1)
                arc_dst.append(base + s)
                arc_cost.append(prev_fwd)
            prev_fwd = fwd_cost
        exit_cost.append(prev_fwd)

    for a, b in inst_arcs:
        arc_src.append(last_state[a])
        arc_dst.append(first_state[b])
        arc_cost.append(exit_cost[a] + jcost.get(instances[b][1], 0.0))

    start_states = np.asarray(sorted({first_state[i] for i in inst_start}),
                              dtype=np.int32)
    finals = sorted({i for i in inst_final})
    final_states = np.asarray([last_state[i] for i in finals], dtype=np.int32)
    final_costs = np.asarray([exit_cost[i] for i in finals], dtype=np.float32)

    pdf_arr = np.asarray(pdf_l, dtype=np.int32)
    unique_pdfs, pdf_compact = np.unique(pdf_arr, return_inverse=True)

    return DecodeGraph(
        pdf=pdf_arr,
        self_cost=np.asarray(self_l, dtype=np.float32),
        arc_src=np.asarray(arc_src, dtype=np.int32),
        arc_dst=np.asarray(arc_dst, dtype=np.int32),
        arc_cost=np.asarray(arc_cost, dtype=np.float32),
        start_states=start_states,
        final_states=final_states,
        final_costs=final_costs,
        state_token=np.asarray(tokmeta, dtype=np.int32),
        state_phone=np.asarray(phmeta, dtype=np.int32),
        state_instance=np.asarray(instmeta, dtype=np.int32),
        unique_pdfs=unique_pdfs.astype(np.int64),
        pdf_compact=pdf_compact.astype(np.int32),
    )
