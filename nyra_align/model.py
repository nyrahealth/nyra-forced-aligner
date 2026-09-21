"""Portable model artifact: loads the exported npz + meta + lexicon."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def resolve_model_dir(model: str | Path) -> Path:
    """Local directory, or a Hugging Face Hub repo id ('org/name') which is
    downloaded (and cached) with huggingface_hub."""
    p = Path(model)
    if p.is_dir():
        return p
    s = str(model)
    if "/" in s and not s.startswith(("/", ".", "~")) and s.count("/") == 1:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (GatedRepoError,
                                            RepositoryNotFoundError)
        try:
            return Path(snapshot_download(repo_id=s))
        except (RepositoryNotFoundError, GatedRepoError) as exc:
            hint = (" The Pro model is not publicly released; it is available "
                    "on request (licensing@nyra-labs.com)." if s.endswith("_pro") else "")
            raise PermissionError(
                f"cannot access model {s!r} on the Hugging Face Hub: it does not "
                f"exist or your account has no access (set HF_TOKEN or run "
                f"`huggingface-cli login`).{hint}") from exc
    raise FileNotFoundError(f"model not found: {model}")


class NyraModel:
    def __init__(self, model_dir: str | Path):
        model_dir = resolve_model_dir(model_dir)
        self.dir = model_dir
        self.meta = json.loads((model_dir / "meta.json").read_text())
        npz = np.load(model_dir / "model.npz")
        self.arrays = {k: npz[k] for k in npz.files}

        # GMM parameters
        self.gconsts = self.arrays["gconsts"]
        self.means_invvars = self.arrays["means_invvars"]
        self.inv_vars = self.arrays["inv_vars"]
        self.pdf_offsets = self.arrays["pdf_offsets"]
        self.num_pdfs = len(self.pdf_offsets) - 1
        self.dim = self.means_invvars.shape[1]

        # Transition model
        self.tuples = self.arrays["tuples"]          # (n_ts, 4)
        self.state2id = self.arrays["state2id"]
        self.log_probs = self.arrays["log_probs"]
        self.pdf_table = self.arrays["pdf_table"]

        if not np.array_equal(self.tuples[:, 2], self.tuples[:, 3]):
            raise ValueError("forward/self-loop pdfs differ; not supported")
        self.tstate_index: dict[tuple[int, int, int], int] = {
            (int(p), int(s), int(fp)): i
            for i, (p, s, fp, _sp) in enumerate(self.tuples)
        }

        # Phones
        self.phone_to_id: dict[str, int] = {
            p: int(i) for p, i in self.meta["phones"].items()}
        self.id_to_phone = {i: p for p, i in self.phone_to_id.items()}
        self.silence_id = self.phone_to_id[self.meta["silence_phone"]]

        # Topology: phone id -> list of (pdf_class, [(dest, prob), ...])
        self.topo: dict[int, list] = {}
        for entry in self.meta["topology"]:
            states = [(s["pdf_class"], s["transitions"])
                      for s in entry["states"]]
            for p in entry["phones"]:
                self.topo[p] = states

        # Lexicon: word -> list of pronunciations (each a list of phone names)
        self.lexicon: dict[str, list[list[str]]] = {}
        for line in (model_dir / "lexicon.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                self.lexicon.setdefault(parts[0], []).append(parts[1:])

        self.phone_map = json.loads((model_dir / "phone_map.json").read_text())
        self.sound_tags: dict[str, str] = self.meta["sound_tags"]

        self.frame_shift = float(self.meta["frame_shift"])
        self.acoustic_scale = float(self.meta["acoustic_scale"])
        self.beam_ladder = [tuple(b) for b in self.meta["beam_ladder"]]
        self.feature_chain = self.meta["feature_chain"]
        self._torch_arrays: dict[str, dict] = {}

    @property
    def wavlm_path(self) -> str:
        """WavLM weights: a subdirectory bundled with the model, else a Hub id."""
        name = self.meta.get("wavlm_model", "microsoft/wavlm-large")
        local = self.dir / name
        return str(local) if local.is_dir() else name

    def torch_arrays(self, device: str) -> dict:
        """Feature-chain matrices as torch tensors on `device` (cached)."""
        if device not in self._torch_arrays:
            import torch
            self._torch_arrays[device] = {
                k: torch.from_numpy(v).to(device)
                for k, v in self.arrays.items()
                if k.endswith("_W") or k.endswith("_b")
            }
        return self._torch_arrays[device]

    def transition_ids(self, phone: int, hmm_state: int,
                       pdf: int) -> tuple[int, float, float]:
        """(tstate, self_loop_cost, forward_cost) for one HMM state.

        Costs are -log transition probability (trained values from the model,
        applied with scale 1.0 exactly like gmm-align-compiled defaults).
        """
        ts = self.tstate_index[(phone, hmm_state, pdf)]
        first_tid = int(self.state2id[ts + 1])
        transitions = self.topo[phone][hmm_state][1]
        self_cost = fwd_cost = None
        for idx, (dest, _prob) in enumerate(transitions):
            cost = -float(self.log_probs[first_tid + idx])
            if dest == hmm_state:
                self_cost = cost
            else:
                fwd_cost = cost
        if self_cost is None or fwd_cost is None:
            raise ValueError(
                f"unsupported topology for phone {phone} state {hmm_state}")
        return ts, self_cost, fwd_cost
