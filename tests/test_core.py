"""Fast tests: no model download, no GPU, no Kaldi, no espeak.

A tiny synthetic model (8 phones, 1 Gaussian per state, context-independent
tree) exercises the real graph builder, emission code and both Viterbi
backends end to end.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

PHONES = {"<eps>": 0, "SIL": 1, "AA": 2, "B": 3, "S": 4, "TH": 5, "FUM": 6, "LAU": 7, "W": 8}
DIM = 2


def _mean(phone_id: int, state: int) -> np.ndarray:
    return np.array([3.0 * phone_id, 1.5 * state], dtype=np.float32)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_model")
    real = [p for p in PHONES.values() if p > 0]
    n_pdf = len(real) * 3
    means = np.stack([_mean(p, s) for p in real for s in range(3)])
    gconsts = (-0.5 * (DIM * math.log(2 * math.pi) + (means ** 2).sum(1))).astype(np.float32)
    max_pid = max(real)
    table = np.full((max_pid + 1, max_pid + 1, max_pid + 1, 3), -1, dtype=np.int32)
    for p in real:
        for s in range(3):
            table[:, p, :, s] = (p - 1) * 3 + s
    tuples = np.array([(p, s, (p - 1) * 3 + s, (p - 1) * 3 + s) for p in real for s in range(3)], dtype=np.int32)
    n_ts = len(tuples)
    state2id = np.zeros(n_ts + 2, dtype=np.int64)
    state2id[1:] = 1 + 2 * np.arange(n_ts + 1)
    log_probs = np.full(2 * n_ts + 1, math.log(0.5), dtype=np.float32)
    np.savez(d / "model.npz", gconsts=gconsts, means_invvars=means,
             inv_vars=np.ones_like(means), pdf_offsets=np.arange(n_pdf + 1, dtype=np.int64),
             pdf_table=table, tuples=tuples, state2id=state2id, log_probs=log_probs)
    meta = {
        "frame_shift": 0.01, "acoustic_scale": 0.1,
        "beam_ladder": [[40, 160], [100, 400], [2000, 40000]],
        "wavlm_model": "microsoft/wavlm-large", "wavlm_layer": 23,
        "feature_chain": [], "phones": PHONES, "silence_phone": "SIL",
        "topology": [{"phones": real, "states": [
            {"pdf_class": s, "transitions": [[s, 0.5], [s + 1, 0.5]]} for s in range(3)]}],
        "espeak": {"language": "en-us", "with_stress": False},
        "sound_tags": {"[UM]": "FUM", "[laughter]": "LAU"},
    }
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "lexicon.txt").write_text(
        "ab AA B\nba B AA\n[UM] FUM\n[laughter] LAU\n[um] S\nth* AA AA AA AA\n")
    (d / "phone_map.json").write_text(json.dumps({"ipa_to_canonical": {}, "manual_prons": {}}))
    return d


def _features(segments, frames_per_state=2, seed=0):
    """segments: list of phone names -> (feats, [(name, start_frame, end_frame)])."""
    rng = np.random.RandomState(seed)
    rows, spans, t = [], [], 0
    for name in segments:
        n = 3 * frames_per_state
        for s in range(3):
            for _ in range(frames_per_state):
                rows.append(_mean(PHONES[name], s) + 0.05 * rng.randn(DIM).astype(np.float32))
        spans.append((name, t, t + n)); t += n
    return np.stack(rows).astype(np.float32), spans


def test_language_gate():
    from nyra_align import Aligner, resolve_model
    assert resolve_model(None) == "nyralabs/nyra_forced_aligner_en"
    assert resolve_model(None, pro=True) == "nyralabs/nyra_forced_aligner_en_pro"
    assert resolve_model(None, language="en-US").endswith("_en")
    for lang in ("de", "fr-FR"):
        with pytest.raises(NotImplementedError):
            Aligner(language=lang)


def test_cutoff_and_fragment_rules():
    from nyra_align.g2p import consonant_fragment_pron, cutoff_stem, word_candidates
    assert cutoff_stem("th-") == "th" and cutoff_stem("Resched-,") == "resched"
    assert cutoff_stem("well-known") is None and cutoff_stem("the") is None
    assert consonant_fragment_pron("th") == ["TH"]
    assert consonant_fragment_pron("str") == ["S", "T", "R"]
    assert consonant_fragment_pron("so") is None          # has a vowel -> G2P instead
    assert "okay" in word_candidates("Okay.")


def test_tokenizer_conventions(model_dir):
    from nyra_align import Aligner
    al = Aligner(model_dir, device="cpu")
    toks, skipped = al.tokenize("Ab, [um] th- BA. [Laughter] [music]")
    got = {t.original: (t.word, t.prons[0]) for t in toks}
    assert got["Ab,"] == ("ab", ["AA", "B"]) and got["BA."] == ("ba", ["B", "AA"])
    assert got["[um]"] == ("[UM]", ["FUM"])          # not the junk "[um]" lexicon entry
    assert got["[Laughter]"] == ("[laughter]", ["LAU"])
    assert got["th-"] == ("th*", ["TH"])             # sounds, overriding the lexicon's letter names
    assert skipped == ["[music]"]


def test_alignment_recovers_boundaries_and_backends_agree(model_dir):
    import torch
    from nyra_align import Aligner
    from nyra_align.decoder import EmissionComputer, align_path, decode_with_ladder
    from nyra_align.decoder_gpu import _emission_costs
    from nyra_align.decoder_jit import JitDecoder
    from nyra_align.graph import build_graph

    feats, spans = _features(["SIL", "AA", "B", "SIL", "FUM", "B", "AA", "SIL"])
    al = Aligner(model_dir, device="cpu")
    res = al.align_features(feats, "ab [UM] ba")
    assert [w.word for w in res.words] == ["ab", "[UM]", "ba"]
    assert [w.type for w in res.words] == ["w", "f", "w"]
    expect = {"ab": (spans[1][1], spans[2][2]), "[UM]": (spans[4][1], spans[4][2]), "ba": (spans[5][1], spans[6][2])}
    for w in res.words:
        s, e = expect[w.word]
        assert abs(w.start - s * 0.01) <= 0.0101 and abs(w.end - e * 0.01) <= 0.0101, (w, s, e)
    assert res.events and res.events[0].word == "[UM]"
    assert "IntervalTier" in res.to_textgrid() and json.loads(res.to_json())["words"]

    tokens, _ = al.tokenize("ab [UM] ba")
    graph = build_graph(tokens, al.model)
    comp = EmissionComputer(al.model, graph.unique_pdfs, "cpu")
    E = _emission_costs(torch.from_numpy(feats), comp, 0.1).numpy()
    jit_path = JitDecoder(graph).decode(E, 40.0)
    np_path = decode_with_ladder(graph, feats, comp, 0.1, al.model.beam_ladder)
    assert jit_path is not None and np.array_equal(jit_path, np_path)
    assert [seg[1] for seg in align_path(graph, np_path, 0.01)][0] == PHONES["SIL"]


def test_too_short_audio_fails_cleanly(model_dir):
    from nyra_align import Aligner
    feats, _ = _features(["AA"], frames_per_state=1)      # 3 frames, transcript needs >= 12
    with pytest.raises(RuntimeError):
        Aligner(model_dir, device="cpu").align_features(feats, "ab ba")


def test_tree_parser():
    from nyra_align.kaldi_text import parse_tree
    tree = parse_tree("ContextDependency 3 1 ToPdf TE 1 3 ( NULL CE 7 SE -1 [ 0 ] { CE 1 CE 2 } ) EndContextDependency")
    assert tree.map_pdf([0, 1, 0], 0) == 7
    assert tree.map_pdf([0, 2, 0], 0) == 1 and tree.map_pdf([0, 2, 0], 2) == 2


def test_audio_loading_without_torchaudio(tmp_path):
    """File I/O must work from a clean install: soundfile + soxr only."""
    import soundfile as sf
    import torch
    from nyra_align.features import load_audio
    sr = 8000
    t = np.arange(sr) / sr                                   # 1 s, stereo, 8 kHz
    stereo = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1).astype(np.float32)
    path = tmp_path / "tone.wav"; sf.write(path, stereo, sr)
    wav = load_audio(str(path))
    assert isinstance(wav, torch.Tensor) and wav.dtype == torch.float32
    assert wav.shape[0] == 1 and abs(wav.shape[1] - 16000) <= 2      # mono, resampled to 16 kHz
    same = load_audio(np.zeros(16000, dtype=np.float32))              # arrays pass through
    assert same.shape == (1, 16000)
    with pytest.raises(RuntimeError):
        (tmp_path / "bad.m4a").write_bytes(b"not audio"); load_audio(str(tmp_path / "bad.m4a"))
