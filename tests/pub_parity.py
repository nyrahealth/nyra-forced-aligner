import json, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, torchaudio
sys.path.insert(0, "/home/ubuntu/code")
from forced_alignment_training.train import _write_feats_binary_ark
from nyra_align.aligner import Aligner
from nyra_align.decoder import EmissionComputer, align_path
from nyra_align.decoder_gpu import _emission_costs
from nyra_align.decoder_jit import JitDecoder
from nyra_align.graph import build_graph
KALDI = "/home/ubuntu/miniconda3/envs/aligner/bin"
ENV = {**os.environ, "PATH": f"{KALDI}:{os.environ.get('PATH','')}",
       "LD_LIBRARY_PATH": "/home/ubuntu/miniconda3/envs/aligner/lib"}
N = int(os.environ.get("N_UTTS", "30"))

def run(cmd, **kw):
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, **kw)
    if r.returncode != 0: raise RuntimeError(f"{cmd[0]}: {r.stderr[:800]}")
    return r

def native(uid, feats, words, mdir, tmp):
    lang = mdir/"lang"; tmp.mkdir(parents=True, exist_ok=True)
    scp = _write_feats_binary_ark({uid: feats}, tmp/f"{uid}.ark")
    w2i = dict(l.split() for l in (lang/"words.txt").read_text().splitlines() if len(l.split())==2)
    (tmp/"t").write_text(f"{uid} {' '.join(w2i[w] for w in words)}\n")
    run([f"{KALDI}/compile-train-graphs", f"--read-disambig-syms={lang}/disambig.int",
         str(mdir/"tri/tree"), str(mdir/"tri/final.mdl"), str(lang/"L.fst"),
         f"ark:{tmp}/t", f"ark:{tmp}/g.fsts"])
    for beam, retry in ((40,160),(100,400),(2000,40000)):
        subprocess.run([f"{KALDI}/gmm-align-compiled", f"--beam={beam}", f"--retry-beam={retry}",
             "--careful=false", "--acoustic-scale=0.1", str(mdir/"tri/final.mdl"),
             f"ark:{tmp}/g.fsts", f"scp:{scp}", f"ark:|gzip -c > {tmp}/ali.gz"], env=ENV, capture_output=True)
        if subprocess.run(["bash","-c",f"gunzip -c {tmp}/ali.gz | head -c 20"], env=ENV, capture_output=True).stdout.strip(): break
    else: return None
    r = run([f"{KALDI}/ali-to-phones","--ctm-output","--frame-shift=0.01", str(mdir/"tri/final.mdl"),
             f"ark:gunzip -c {tmp}/ali.gz|","-"])
    i2p = dict((l.split()[1], l.split()[0]) for l in (lang/"phones.txt").read_text().splitlines() if len(l.split())==2)
    return [(i2p[p[4]], round(float(p[2])*100), round((float(p[2])+float(p[3]))*100))
            for p in (l.split() for l in r.stdout.strip().splitlines())]

chunks = [c for c in json.load(open("/home/ubuntu/data/buckeye/eval_chunks.json")) if c["split"]=="test"]
picks = [chunks[i] for i in np.random.RandomState(0).choice(len(chunks), N, replace=False)]
BK = Path("/home/ubuntu/data/models/forced_alignment")
M = Path("/ephemeral/mfcc_wavlm_phone_ablation/models")
for label, port, kal in (("noise LDA", BK/"wavlm_ldadirect_noiselda_v1/nyra_portable", M/"pub_wavlm_ldadirect_noise_evalext"),
                         ("clean LDA", BK/"wavlm_ldadirect_cleanlda_v1/nyra_portable", M/"pub_wavlm_ldadirect_clean_evalext")):
    al = Aligner(str(port), device="cuda", precision="fp32")
    nseq = nok = fr_tot = fr_ok = 0; maxd = 0
    for c in picks:
        w, sr = torchaudio.load(f"/home/ubuntu/data/buckeye/{c['audio']}")
        seg = w[:, int(round(float(c["chunk_start_s"])*sr)):int(round(float(c["chunk_end_s"])*sr))]
        if sr != 16000: seg = torchaudio.functional.resample(seg, sr, 16000)
        feats = al.features(seg.numpy())
        toks, _ = al.tokenize(c["transcript"])
        if not toks: continue
        nat = native(c["id"], feats, [t.word for t in toks], kal, Path(f"/ephemeral/pub_parity/{label.split()[0]}/{c['id']}"))
        if nat is None: print("  native failed", c["id"]); continue
        g = build_graph(toks, al.model); comp = EmissionComputer(al.model, g.unique_pdfs, "cuda")
        E = _emission_costs(torch.from_numpy(feats).cuda(), comp, 0.1).cpu().numpy()
        dec = JitDecoder(g); path = None
        for beam, retry in al.model.beam_ladder:
            path = dec.decode(E, beam)
            if path is None: path = dec.decode(E, retry)
            if path is not None: break
        ours = [(al.model.id_to_phone[ph], round(s*100), round(e*100)) for _t, ph, s, e in align_path(g, path, 0.01)]
        nok += 1
        if [x[0] for x in ours] == [x[0] for x in nat]:
            nseq += 1
            for (pn,ns,ne),(_o,os_,oe) in zip(nat, ours):
                fr_tot += 2; fr_ok += (ns==os_)+(ne==oe); maxd = max(maxd, abs(ns-os_), abs(ne-oe))
        else: print(f"  seq mismatch {c['id']}: native {len(nat)} vs ours {len(ours)} phones")
    print(f"{label}: {nseq}/{nok} identical phone sequences | boundary frames exact {fr_ok}/{fr_tot} "
          f"({100*fr_ok/max(fr_tot,1):.2f}%) | max diff {maxd} frames ({maxd*10} ms)")
