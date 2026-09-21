"""Export a trained Kaldi GMM-HMM aligner into the portable nyra-align format.

This is the ONLY place Kaldi binaries are used (gmm-copy / copy-tree, to dump
the binary model/tree as text). The output directory is self-contained:

    model.npz        gmm params, transition model, triphone->pdf table,
                     feature-projection matrices
    meta.json        topology, phone table, feature-chain + decoding config
    lexicon.txt      word -> canonical phones (training + eval-extended)
    phone_map.json   espeak IPA -> canonical phone mapping (G2P at inference)

Usage:
    nyra-align-export --model-dir .../models/wavlm_extra_evalext \
        --out /path/to/nyra-model-en [--kaldi-bin /path/to/kaldi/bin]
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from .kaldi_text import parse_mdl, parse_tree

DEFAULT_KALDI_BIN = "/home/ubuntu/miniconda3/envs/aligner/bin"

# espeak misreads these; hand-crafted pronunciations in training conventions
# (mirrors evaluation/extend_eval_lexicon.py).
MANUAL_PRONS = {
    "yknow": "Y AX N OW",
    "um-hum": "AH M HH AH M",
    "uh-huh": "AH HH AH",
    "th=": "TH", "ch=": "CH", "sh=": "SH", "wh=": "W",
}


def _run_text_dump(binary: str, args: list[str], kaldi_bin: str) -> str:
    import os
    env = {**os.environ,
           "PATH": f"{kaldi_bin}:{os.environ.get('PATH', '')}",
           "LD_LIBRARY_PATH": f"{Path(kaldi_bin).parent}/lib:"
                              f"{os.environ.get('LD_LIBRARY_PATH', '')}"}
    res = subprocess.run([f"{kaldi_bin}/{binary}", *args],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"{binary} failed:\n{res.stderr[:2000]}")
    return res.stdout


def _load_phones(lang_dir: Path) -> tuple[dict[str, int], list[str]]:
    phone_to_id: dict[str, int] = {}
    for line in (lang_dir / "phones.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            phone_to_id[parts[0]] = int(parts[1])
    real = [p for p in phone_to_id
            if not p.startswith("#") and p != "<eps>"]
    return phone_to_id, real


def export_model(model_dir: Path, out_dir: Path, kaldi_bin: str,
                 phone_map_path: Path | None = None) -> Path:
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_dir = model_dir / "lang"
    tri_dir = model_dir / "tri"
    metadata = json.loads((model_dir / "metadata.json").read_text())

    # ---- transition model + GMMs -----------------------------------------
    mdl_text = _run_text_dump(
        "gmm-copy", ["--binary=false", str(tri_dir / "final.mdl"), "-"],
        kaldi_bin)
    tm, gmms = parse_mdl(mdl_text)

    pdf_offsets = np.zeros(len(gmms) + 1, dtype=np.int64)
    for i, g in enumerate(gmms):
        pdf_offsets[i + 1] = pdf_offsets[i] + len(g.gconsts)
    gconsts = np.concatenate([g.gconsts for g in gmms]).astype(np.float32)
    miv = np.vstack([g.means_invvars for g in gmms]).astype(np.float32)
    inv_vars = np.vstack([g.inv_vars for g in gmms]).astype(np.float32)

    # ---- decision tree -> dense (l, c, r, pdf_class) table ----------------
    tree_text = _run_text_dump(
        "copy-tree", ["--binary=false", str(tri_dir / "tree"), "-"], kaldi_bin)
    tree = parse_tree(tree_text)
    if tree.context_width != 3 or tree.central_position != 1:
        raise ValueError("only triphone trees (N=3, P=1) supported")

    phone_to_id, real_phones = _load_phones(lang_dir)
    max_pid = max(phone_to_id[p] for p in real_phones)
    n_classes = max(len(e.states) for e in tm.topology)
    pdf_table = np.full((max_pid + 1, max_pid + 1, max_pid + 1, n_classes),
                        -1, dtype=np.int32)
    ctx_ids = [0] + [phone_to_id[p] for p in real_phones]
    for c_name in real_phones:
        c = phone_to_id[c_name]
        topo = tm.topo_entry(c)
        for l in ctx_ids:
            for r in ctx_ids:
                for s, (pdf_class, _trans) in enumerate(topo.states):
                    pdf = tree.map_pdf([l, c, r], pdf_class)
                    if pdf is not None:
                        pdf_table[l, c, r, s] = pdf

    # ---- transition-state lookup ------------------------------------------
    tuples = np.asarray(tm.tuples, dtype=np.int32)          # (n_tstates, 4)
    state2id = tm.state2id.astype(np.int64)
    log_probs = tm.log_probs.astype(np.float32)

    # ---- feature transforms -----------------------------------------------
    arrays: dict[str, np.ndarray] = dict(
        gconsts=gconsts, means_invvars=miv, inv_vars=inv_vars,
        pdf_offsets=pdf_offsets, pdf_table=pdf_table,
        tuples=tuples, state2id=state2id, log_probs=log_probs,
    )
    chain: list[dict] = []

    pca_path = model_dir / "pca" / "pca.pkl"
    if pca_path.exists():
        with pca_path.open("rb") as f:
            pca = pickle.load(f)
        W = pca.components_.astype(np.float32)               # (k, 1024)
        arrays["pca_W"] = W
        arrays["pca_b"] = (-pca.mean_ @ W.T).astype(np.float32)
        chain.append({"op": "linear", "W": "pca_W", "b": "pca_b"})

    chain.append({"op": "cmvn"})

    with (model_dir / "lda" / "lda.pkl").open("rb") as f:
        lda = pickle.load(f)
    in_dim = lda.scalings_.shape[0]
    out_dim = lda.transform(np.zeros((1, in_dim))).shape[1]
    base_dim = int(arrays["pca_W"].shape[0]) if "pca_W" in arrays else in_dim
    if in_dim % base_dim:
        raise ValueError(f"LDA input dim {in_dim} not a multiple of {base_dim}")
    n_ctx = in_dim // base_dim
    if n_ctx % 2 == 0:
        raise ValueError(f"even splice width {n_ctx}")
    context = (n_ctx - 1) // 2
    if context:
        chain.append({"op": "splice", "context": context})
    lda_W = lda.scalings_[:, :out_dim].astype(np.float32).T   # (out, in)
    arrays["lda_W"] = lda_W
    # sklearn: svd solver centers on xbar_; eigen/lsqr solvers do not center.
    xbar = getattr(lda, "xbar_", None)
    if xbar is not None and getattr(lda, "solver", "svd") == "svd":
        arrays["lda_b"] = (-xbar @ lda_W.T).astype(np.float32)
    else:
        arrays["lda_b"] = np.zeros(out_dim, dtype=np.float32)
    chain.append({"op": "linear", "W": "lda_W", "b": "lda_b"})

    mllt_path = tri_dir / "mllt_total.npy"
    if mllt_path.exists():
        mllt = np.load(mllt_path).astype(np.float32)
        arrays["mllt_W"] = mllt
        arrays["mllt_b"] = np.zeros(mllt.shape[0], dtype=np.float32)
        chain.append({"op": "linear", "W": "mllt_W", "b": "mllt_b"})

    dim = miv.shape[1]
    if chain[-1]["op"] == "linear":
        last_out = arrays[chain[-1]["W"]].shape[0]
        if last_out != dim:
            raise ValueError(f"chain output {last_out} != GMM dim {dim}")

    np.savez_compressed(out_dir / "model.npz", **arrays)

    # ---- lexicon + phone map ----------------------------------------------
    lex_lines = (lang_dir / "lexicon.txt").read_text().splitlines()
    (out_dir / "lexicon.txt").write_text("\n".join(lex_lines) + "\n")

    if phone_map_path is None:
        phone_map_path = Path(
            "/home/ubuntu/code/forced_alignment_training/phone_maps/english.json")
    phone_map = json.loads(Path(phone_map_path).read_text())
    phone_map["manual_prons"] = MANUAL_PRONS
    (out_dir / "phone_map.json").write_text(
        json.dumps(phone_map, indent=1, ensure_ascii=False))

    # ---- topology + meta ----------------------------------------------------
    topo_json = [
        {"phones": e.phones,
         "states": [{"pdf_class": pc, "transitions": tr} for pc, tr in e.states]}
        for e in tm.topology
    ]
    silence_phone = "SIL"
    meta = {
        "format_version": 1,
        "source_model": str(model_dir),
        "variant": metadata.get("variant"),
        "frame_shift": metadata.get("frame_shift_ms", 10) / 1000.0,
        "acoustic_scale": 0.1,
        "beam_ladder": [[40, 160], [100, 400], [2000, 40000]],
        "wavlm_model": "microsoft/wavlm-large",
        "wavlm_layer": 23,
        "sample_rate": 16000,
        "feature_chain": chain,
        "phones": {p: i for p, i in phone_to_id.items()},
        "silence_phone": silence_phone,
        "topology": topo_json,
        "num_pdfs": len(gmms),
        "espeak": {"language": "en-us", "with_stress": False},
        "sound_tags": {
            "[UH]": "FUH", "[UM]": "FUM", "[laughter]": "LAU",
            "[sniff]": "SNF", "[throatclearing]": "TCL", "[cough]": "COF",
            "[sigh]": "SGH", "[breath]": "BRE", "[lipsmack]": "LPS",
            "[yawn]": "YWN", "[noise]": "NSN", "[crying]": "CRY",
            "[fart]": "FRT", "[scream]": "SCR", "[sneeze]": "SNZ",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"exported {len(gmms)} pdfs, {len(tuples)} transition-states, "
          f"{len(lex_lines)} lexicon entries -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kaldi-bin", default=DEFAULT_KALDI_BIN)
    ap.add_argument("--phone-map", default=None)
    args = ap.parse_args()
    export_model(Path(args.model_dir), Path(args.out), args.kaldi_bin,
                 Path(args.phone_map) if args.phone_map else None)


if __name__ == "__main__":
    main()
