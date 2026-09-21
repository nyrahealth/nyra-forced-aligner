#!/usr/bin/env python3
"""Parity: nyra-align pure-Python decoder vs native Kaldi on identical features.

For N Buckeye test chunks:
  features extracted ONCE (nyra-align front-end);
  native: feats ark -> compile-train-graphs -> gmm-align-compiled (same ladder)
          -> ali-to-phones CTM;
  ours:   build_graph -> decode_with_ladder -> align_path.
Compares phone label sequences and boundary frames.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, "/home/ubuntu/code")
from forced_alignment_training.train import _write_feats_binary_ark  # noqa: E402

from nyra_align.aligner import Aligner  # noqa: E402
from nyra_align.decoder import EmissionComputer, align_path, decode_with_ladder  # noqa: E402
from nyra_align.decoder_gpu import _emission_costs  # noqa: E402
from nyra_align.decoder_jit import JitDecoder  # noqa: E402
from nyra_align.graph import build_graph  # noqa: E402

MODEL_DIR = Path("/ephemeral/mfcc_wavlm_phone_ablation/models/wavlm_extra_evalext")
NYRA_MODEL = "/ephemeral/nyra_model_en"
KALDI = "/home/ubuntu/miniconda3/envs/aligner/bin"
WORK = Path("/ephemeral/nyra_parity")
N_UTTS = int(os.environ.get("N_UTTS", "30"))

ENV = {**os.environ, "PATH": f"{KALDI}:{os.environ.get('PATH','')}",
       "LD_LIBRARY_PATH": f"/home/ubuntu/miniconda3/envs/aligner/lib"}


def run(cmd, **kw):
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]}: {r.stderr[:1500]}")
    return r


def native_align(utt_id, feats, words, lang, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    scp = _write_feats_binary_ark({utt_id: feats}, tmp / f"{utt_id}.ark")
    w2i = {}
    for line in (lang / "words.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 2:
            w2i[p[0]] = p[1]
    ids = [w2i[w] for w in words]
    (tmp / "text_int").write_text(f"{utt_id} {' '.join(ids)}\n")
    mdl, tree = MODEL_DIR / "tri/final.mdl", MODEL_DIR / "tri/tree"
    run([f"{KALDI}/compile-train-graphs",
         f"--read-disambig-syms={lang}/disambig.int",
         str(tree), str(mdl), str(lang / "L.fst"),
         f"ark:{tmp}/text_int", f"ark:{tmp}/g.fsts"])
    ali = tmp / "ali.gz"
    ok = False
    for beam, retry in ((40, 160), (100, 400), (2000, 40000)):
        r = subprocess.run(
            [f"{KALDI}/gmm-align-compiled", f"--beam={beam}",
             f"--retry-beam={retry}", "--careful=false",
             "--acoustic-scale=0.1", str(mdl), f"ark:{tmp}/g.fsts",
             f"scp:{scp}", f"ark:|gzip -c > {ali}"],
            env=ENV, capture_output=True, text=True)
        chk = subprocess.run(["bash", "-c", f"gunzip -c {ali} | head -c 20"],
                             env=ENV, capture_output=True)
        if chk.stdout.strip():
            ok = True
            break
    if not ok:
        return None
    r = run([f"{KALDI}/ali-to-phones", "--ctm-output", "--frame-shift=0.01",
             str(mdl), f"ark:gunzip -c {ali}|", "-"])
    segs = []
    i2p = {}
    for line in (lang / "phones.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 2:
            i2p[p[1]] = p[0]
    for line in r.stdout.strip().splitlines():
        p = line.split()
        segs.append((i2p[p[4]], round(float(p[2]) * 100),
                     round((float(p[2]) + float(p[3])) * 100)))
    return segs


def main():
    chunks = [c for c in json.load(open("/home/ubuntu/data/buckeye/eval_chunks.json"))
              if c["split"] == "test"]
    rng = np.random.RandomState(0)
    picks = [chunks[i] for i in rng.choice(len(chunks), N_UTTS, replace=False)]

    aligner = Aligner(NYRA_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    lang = MODEL_DIR / "lang"

    n_ok = n_seq = n_frames_total = n_frames_exact = 0
    max_diff = 0
    diffs = []
    for c in picks:
        wav = f"/home/ubuntu/data/buckeye/{c['audio']}"
        w, sr = torchaudio.load(wav)
        s0 = int(round(float(c["chunk_start_s"]) * sr))
        s1 = int(round(float(c["chunk_end_s"]) * sr))
        seg = w[:, s0:s1]
        if sr != 16000:
            seg = torchaudio.functional.resample(seg, sr, 16000)

        feats = aligner.features(seg.numpy())
        tokens, skipped = aligner.tokenize(c["transcript"])
        if not tokens:
            continue
        words = [t.word for t in tokens]

        native = native_align(c["id"], feats, words, lang,
                              WORK / c["id"])
        if native is None:
            print(f"[native-fail] {c['id']}")
            continue

        graph = build_graph(tokens, aligner.model)
        comp = EmissionComputer(aligner.model, graph.unique_pdfs, aligner.device)
        ft = torch.from_numpy(feats).to(aligner.device)
        E = _emission_costs(ft, comp, 0.1).cpu().numpy()
        dec = JitDecoder(graph)
        path = None
        for beam, retry in aligner.model.beam_ladder:
            path = dec.decode(E, beam) 
            if path is None:
                path = dec.decode(E, retry)
            if path is not None:
                break
        if path is None:
            print(f"[nyra-fail] {c['id']}")
            continue
        segs = align_path(graph, path, 0.01)
        ours = [(aligner.model.id_to_phone[ph], round(s * 100), round(e * 100))
                for _tok, ph, s, e in segs]

        n_ok += 1
        same_seq = [x[0] for x in ours] == [x[0] for x in native]
        n_seq += same_seq
        if same_seq:
            for (pn, ns, ne), (_o, os_, oe) in zip(native, ours):
                n_frames_total += 2
                n_frames_exact += (ns == os_) + (ne == oe)
                d = max(abs(ns - os_), abs(ne - oe))
                diffs.append(d)
                max_diff = max(max_diff, d)
        else:
            print(f"[seq-mismatch] {c['id']}: native {len(native)} phones, "
                  f"ours {len(ours)}")

    print(f"\n=== PARITY ({n_ok} utts) ===")
    print(f"identical phone sequence: {n_seq}/{n_ok}")
    if n_frames_total:
        print(f"boundary frames exact:    {n_frames_exact}/{n_frames_total} "
              f"({100 * n_frames_exact / n_frames_total:.2f}%)")
        print(f"max boundary diff:        {max_diff} frames "
              f"({max_diff * 10} ms)")
        d = np.array(diffs)
        print(f"boundary diff mean {d.mean():.3f} frames, "
              f">1 frame: {(d > 1).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
