"""Parsers for Kaldi text-format artifacts (final.mdl, tree).

Used only at export time (converting a trained Kaldi GMM-HMM model into the
portable nyra-align artifact). Inference never touches these.

Formats handled:
  * TransitionModel + AmDiagGmm as printed by `gmm-copy --binary=false`
    (both the old <Triples> and the newer <Tuples> transition entries).
  * ContextDependency (decision tree) as printed by `copy-tree --binary=false`
    (EventMap nodes CE / SE / TE / NULL).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class _Tokens:
    """Whitespace token stream with lookahead over a Kaldi text dump."""

    def __init__(self, text: str):
        self.toks = text.split()
        self.pos = 0

    def peek(self) -> str:
        return self.toks[self.pos]

    def next(self) -> str:
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def expect(self, want: str) -> None:
        got = self.next()
        if got != want:
            raise ValueError(f"expected {want!r}, got {got!r} at pos {self.pos}")

    def next_int(self) -> int:
        return int(self.next())

    def next_float(self) -> float:
        return float(self.next())

    def read_vector(self) -> np.ndarray:
        """Read `[ v1 v2 ... ]`."""
        self.expect("[")
        vals = []
        while self.peek() != "]":
            vals.append(float(self.next()))
        self.next()  # ]
        return np.asarray(vals, dtype=np.float64)

    def read_matrix(self) -> np.ndarray:
        """Read `[ r00 r01 ... \n r10 ... ]` (rows have no separators in the
        token stream, so the caller must reshape; Kaldi prints matrices with
        newlines which we cannot see here). We instead read all floats until
        `]` and let the caller reshape."""
        self.expect("[")
        vals = []
        while self.peek() != "]":
            vals.append(float(self.next()))
        self.next()
        return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# HMM topology
# ---------------------------------------------------------------------------

@dataclass
class TopologyEntry:
    phones: list[int]
    # per emitting state: (pdf_class, [(dest_state, prob), ...])
    states: list[tuple[int, list[tuple[int, float]]]]


def parse_topology(t: _Tokens) -> list[TopologyEntry]:
    t.expect("<Topology>")
    entries: list[TopologyEntry] = []
    while t.peek() == "<TopologyEntry>":
        t.next()
        t.expect("<ForPhones>")
        phones = []
        while t.peek() != "</ForPhones>":
            phones.append(t.next_int())
        t.next()
        states: list[tuple[int, list[tuple[int, float]]]] = []
        while t.peek() == "<State>":
            t.next()
            t.next_int()  # state index (sequential)
            if t.peek() == "<PdfClass>":
                t.next()
                pdf_class = t.next_int()
                trans = []
                while t.peek() == "<Transition>":
                    t.next()
                    dest = t.next_int()
                    prob = t.next_float()
                    trans.append((dest, prob))
                states.append((pdf_class, trans))
            # final (non-emitting) state has no PdfClass / transitions
            t.expect("</State>")
        t.expect("</TopologyEntry>")
        entries.append(TopologyEntry(phones=phones, states=states))
    t.expect("</Topology>")
    return entries


# ---------------------------------------------------------------------------
# TransitionModel + AmDiagGmm
# ---------------------------------------------------------------------------

@dataclass
class TransitionModel:
    topology: list[TopologyEntry]
    # per transition-state (1-indexed in Kaldi; list is 0-indexed):
    # (phone, hmm_state, forward_pdf, self_loop_pdf)
    tuples: list[tuple[int, int, int, int]]
    log_probs: np.ndarray        # indexed by transition-id (1-based; [0] unused)
    state2id: np.ndarray         # first transition-id of each transition-state

    def topo_entry(self, phone: int) -> TopologyEntry:
        for e in self.topology:
            if phone in e.phones:
                return e
        raise KeyError(f"phone {phone} not in topology")


@dataclass
class DiagGmm:
    gconsts: np.ndarray       # (M,)  includes log-weights + normalizers
    weights: np.ndarray       # (M,)
    means_invvars: np.ndarray  # (M, D)
    inv_vars: np.ndarray      # (M, D)


def parse_mdl(text: str) -> tuple[TransitionModel, list[DiagGmm]]:
    t = _Tokens(text)
    t.expect("<TransitionModel>")
    topo = parse_topology(t)

    tuples: list[tuple[int, int, int, int]] = []
    tag = t.next()
    if tag == "<Triples>":
        n = t.next_int()
        for _ in range(n):
            phone, hmm_state, pdf = t.next_int(), t.next_int(), t.next_int()
            tuples.append((phone, hmm_state, pdf, pdf))
        t.expect("</Triples>")
    elif tag == "<Tuples>":
        n = t.next_int()
        for _ in range(n):
            phone, hmm_state = t.next_int(), t.next_int()
            fwd_pdf, sl_pdf = t.next_int(), t.next_int()
            tuples.append((phone, hmm_state, fwd_pdf, sl_pdf))
        t.expect("</Tuples>")
    else:
        raise ValueError(f"expected <Triples> or <Tuples>, got {tag!r}")

    t.expect("<LogProbs>")
    log_probs = t.read_vector()
    t.expect("</LogProbs>")
    t.expect("</TransitionModel>")

    # state2id: cumulative transition counts (Kaldi ComputeDerived)
    by_phone = {}
    for e in topo:
        for p in e.phones:
            by_phone[p] = e
    state2id = np.zeros(len(tuples) + 2, dtype=np.int64)
    cur = 1
    for i, (phone, hmm_state, _f, _s) in enumerate(tuples):
        state2id[i + 1] = cur
        cur += len(by_phone[phone].states[hmm_state][1])
    state2id[len(tuples) + 1] = cur
    # Kaldi prints log_probs_ as a 1-based vector: element 0 is a dummy.
    if cur != len(log_probs):
        raise ValueError(
            f"transition count mismatch: derived {cur - 1} ids, "
            f"log-probs has {len(log_probs)} (incl. dummy)")

    tm = TransitionModel(
        topology=topo, tuples=tuples,
        log_probs=log_probs,
        state2id=state2id,
    )

    # Acoustic model
    t.expect("<DIMENSION>")
    dim = t.next_int()
    t.expect("<NUMPDFS>")
    num_pdfs = t.next_int()
    gmms: list[DiagGmm] = []
    for _ in range(num_pdfs):
        t.expect("<DiagGMM>")
        t.expect("<GCONSTS>")
        gconsts = t.read_vector()
        t.expect("<WEIGHTS>")
        weights = t.read_vector()
        t.expect("<MEANS_INVVARS>")
        miv = t.read_matrix().reshape(len(gconsts), dim)
        t.expect("<INV_VARS>")
        iv = t.read_matrix().reshape(len(gconsts), dim)
        t.expect("</DiagGMM>")
        gmms.append(DiagGmm(gconsts, weights, miv, iv))
    return tm, gmms


# ---------------------------------------------------------------------------
# Decision tree (ContextDependency / EventMap)
# ---------------------------------------------------------------------------

@dataclass
class _CE:
    pdf: int


@dataclass
class _SE:
    key: int
    values: set
    yes: object
    no: object


@dataclass
class _TE:
    key: int
    table: list


_PDF_CLASS_KEY = -1


def _parse_event_map(t: _Tokens):
    tag = t.next()
    if tag == "CE":
        return _CE(t.next_int())
    if tag == "SE":
        key = t.next_int()
        t.expect("[")
        vals = set()
        while t.peek() != "]":
            vals.add(t.next_int())
        t.next()
        t.expect("{")
        yes = _parse_event_map(t)
        no = _parse_event_map(t)
        t.expect("}")
        return _SE(key, vals, yes, no)
    if tag == "TE":
        key = t.next_int()
        size = t.next_int()
        t.expect("(")
        table = [_parse_event_map(t) for _ in range(size)]
        t.expect(")")
        return _TE(key, table)
    if tag == "NULL":
        return None
    raise ValueError(f"unknown EventMap node {tag!r}")


@dataclass
class ContextDependency:
    context_width: int   # N (3 for triphone)
    central_position: int  # P (1)
    root: object

    def map_pdf(self, phone_window: list[int], pdf_class: int) -> int | None:
        """Kaldi ContextDependency::Compute for one (window, pdf-class)."""
        event = {_PDF_CLASS_KEY: pdf_class}
        for i, p in enumerate(phone_window):
            event[i] = p
        node = self.root
        while node is not None:
            if isinstance(node, _CE):
                return node.pdf
            if isinstance(node, _SE):
                node = node.yes if event.get(node.key) in node.values else node.no
            elif isinstance(node, _TE):
                idx = event.get(node.key)
                if idx is None or not (0 <= idx < len(node.table)):
                    return None
                node = node.table[idx]
            else:
                return None
        return None


def parse_tree(text: str) -> ContextDependency:
    t = _Tokens(text)
    t.expect("ContextDependency")
    n = t.next_int()
    p = t.next_int()
    t.expect("ToPdf")
    root = _parse_event_map(t)
    if t.peek() == "EndContextDependency":
        t.next()
    return ContextDependency(context_width=n, central_position=p, root=root)
