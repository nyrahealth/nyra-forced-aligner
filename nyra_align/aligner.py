"""The user-facing Aligner: audio + transcript -> word/phone timings.

Pure-Python inference path (no Kaldi):
  WavLM features -> model projection chain -> decode graph for the transcript
  -> GMM beam Viterbi (beam ladder) -> word/phone segments.

OOV words are phonemized with espeak through the same mapping used to build
the training lexicon, then added to a runtime lexicon cache.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .decoder import EmissionComputer, align_path, decode_with_ladder
from .decoder_gpu import _emission_costs, viterbi_gpu
from .features import (apply_chain, apply_chain_torch, extract_wavlm,
                       load_audio, load_wavlm)

try:
    from .decoder_jit import JitDecoder
except ImportError:          # numba not installed
    JitDecoder = None
from .g2p import (G2P, consonant_fragment_pron, cutoff_stem,
                  preferred_oov_form, word_candidates)
from .graph import Token, build_graph
from .model import NyraModel
from .outputs import AlignmentResult, Phone, Word

log = logging.getLogger(__name__)

# Published models per language. `pro` = noise-robust variant.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "en": {"standard": "nyralabs/nyra_forced_aligner_en",
           "pro": "nyralabs/nyra_forced_aligner_en_pro"},
}
SUPPORTED_LANGUAGES = tuple(DEFAULT_MODELS)


def _norm_lang(language: str) -> str:
    return language.strip().lower().replace("_", "-").split("-")[0]


def resolve_model(model: str | Path | None, language: str = "en",
                  pro: bool = False) -> str | Path:
    """Pick the published model for `language` unless an explicit model
    (local dir or Hub id) is given. Only English is available so far."""
    lang = _norm_lang(language)
    if lang not in DEFAULT_MODELS:
        raise NotImplementedError(
            f"language {language!r} is not supported yet; available: "
            f"{', '.join(SUPPORTED_LANGUAGES)}")
    if model is not None:
        return model
    return DEFAULT_MODELS[lang]["pro" if pro else "standard"]


class Aligner:
    def __init__(self, model: str | Path | None = None, *,
                 language: str = "en", pro: bool = False,
                 device: str | None = None, precision: str | None = None,
                 wavlm_batch: int = 8):
        """Forced aligner.

        model:     local model directory or Hugging Face repo id. Default:
                   the published model for `language` (`pro=True` selects the
                   noise-robust variant). Private repos need `HF_TOKEN` or a
                   `huggingface-cli login`.
        language:  alignment language; only 'en' is implemented so far
                   (NotImplementedError otherwise).
        precision: 'fp16' | 'bf16' | 'fp32'. Defaults to fp16 on CUDA
                   (verified boundary-safe, ~2x faster WavLM), fp32 on CPU.
        """
        self.language = _norm_lang(language)
        self.model = NyraModel(resolve_model(model, language, pro))
        model_lang = _norm_lang(self.model.meta.get("espeak", {}).get("language", "en"))
        if model_lang != self.language:
            raise NotImplementedError(
                f"model is for {model_lang!r}, requested language {self.language!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if precision is None:
            precision = "fp16" if device.startswith("cuda") else "fp32"
        self.precision = precision
        self.wavlm_batch = wavlm_batch
        self._wavlm = None
        self.g2p = G2P(self.model.phone_map,
                       language=self.model.meta["espeak"]["language"],
                       with_stress=self.model.meta["espeak"]["with_stress"])
        # runtime lexicon = model lexicon + espeak-extended entries
        self.lexicon = {w: list(prons)
                        for w, prons in self.model.lexicon.items()}
        # event tags are matched case-insensitively ("[um]", "[Laughter]" ->
        # the model's "[UM]", "[laughter]") so ASR output in any casing reaches
        # the dedicated filler / vocal-event units
        self._tag_canon = {t.lower(): t for t in self.model.sound_tags
                           if t in self.lexicon}

    # -- components ---------------------------------------------------------

    @property
    def wavlm(self):
        if self._wavlm is None:
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                     "fp32": torch.float32}[self.precision]
            self._wavlm = load_wavlm(self.model.wavlm_path,
                                     self.device, dtype=dtype)
        return self._wavlm

    def features(self, audio) -> np.ndarray:
        return self.features_torch(audio).cpu().numpy()

    def features_torch(self, audio) -> "torch.Tensor":
        """Extract + project features fully on self.device (float32)."""
        waveform = load_audio(audio)
        raw = extract_wavlm(self.wavlm, waveform,
                            layer=self.model.meta["wavlm_layer"],
                            device=self.device, batch_size=self.wavlm_batch,
                            return_torch=True)
        return apply_chain_torch(raw, self.model.feature_chain,
                                 self.model.torch_arrays(self.device))

    # -- transcript -----------------------------------------------------------

    def tokenize(self, text: str) -> tuple[list[Token], list[str]]:
        """Map raw transcript tokens to lexicon forms; espeak the OOVs."""
        tokens: list[Token] = []
        skipped: list[str] = []
        known_phones = set(self.model.phone_to_id)
        for raw in text.split():
            core = raw.strip(".,!?;:\u2026\"'()")
            tag = self._tag_canon.get(core.lower())
            if tag is not None:
                tokens.append(Token(word=tag, original=raw,
                                    prons=self.lexicon[tag], index=len(tokens)))
                continue
            stem = cutoff_stem(raw)
            if stem is not None:            # "th-" -> cut-off form "th*"
                form = stem + "*"
                # vowel-less fragments ("th-", "s-"): use the sounds, overriding
                # espeak's letter-name spelling ("T IY EY CH"), whose minimum
                # duration cannot fit a short fragment and displaces neighbours
                frag = consonant_fragment_pron(stem)
                if frag and all(p in known_phones for p in frag):
                    self.lexicon[form] = [frag]
                elif form not in self.lexicon:
                    pron = self.g2p.pronounce(form)
                    if pron and all(p in known_phones for p in pron):
                        self.lexicon[form] = [pron]
                if form in self.lexicon:
                    tokens.append(Token(word=form, original=raw,
                                        prons=self.lexicon[form], index=len(tokens)))
                else:
                    skipped.append(raw)
                continue
            cands = word_candidates(raw)
            if not cands:
                continue
            form = next((c for c in cands if c in self.lexicon), None)
            if form is None:
                oov = preferred_oov_form(cands)
                pron = self.g2p.pronounce(oov) if oov else None
                if pron and all(p in known_phones for p in pron):
                    self.lexicon[oov] = [pron]
                    form = oov
                    log.info("OOV %r -> espeak %s", raw, " ".join(pron))
                else:
                    skipped.append(raw)
                    continue
            tokens.append(Token(word=form, original=raw,
                                prons=self.lexicon[form], index=len(tokens)))
        return tokens, skipped

    # -- main entry -------------------------------------------------------------

    def align(self, audio, text: str) -> AlignmentResult:
        feats = self.features_torch(audio)
        return self.align_features(feats, text)

    def align_features(self, feats, text: str,
                       junction_phones: list[str] | None = None,
                       junction_costs: dict[str, float] | None = None,
                       acoustic_scale: float | None = None) -> AlignmentResult:
        model = self.model
        tokens, skipped = self.tokenize(text)
        duration = len(feats) * model.frame_shift
        if not tokens:
            return AlignmentResult([], [], skipped, duration)
        ac_scale = model.acoustic_scale if acoustic_scale is None else acoustic_scale

        graph = build_graph(tokens, model, junction_phones, junction_costs)
        computer = EmissionComputer(model, graph.unique_pdfs, self.device)

        feats_t = (feats if isinstance(feats, torch.Tensor)
                   else torch.from_numpy(np.ascontiguousarray(feats)))
        feats_t = feats_t.to(self.device, dtype=torch.float32)

        path = None
        if JitDecoder is not None:
            # emissions on GPU (if available), DP loop in compiled code
            E = _emission_costs(feats_t, computer, ac_scale).cpu().numpy()
            dec = JitDecoder(graph)
            for beam, retry in model.beam_ladder:
                path = dec.decode(E, beam)
                if path is None:
                    path = dec.decode(E, retry)
                if path is not None:
                    break
        elif self.device.startswith("cuda"):
            path = viterbi_gpu(graph, feats_t, computer, ac_scale)
        if path is None:
            feats_np = (feats.cpu().numpy()
                        if isinstance(feats, torch.Tensor) else feats)
            path = decode_with_ladder(graph, feats_np, computer,
                                      ac_scale, model.beam_ladder)
        if path is None:
            raise RuntimeError(
                f"alignment failed at all beams "
                f"({len(tokens)} tokens, {len(feats)} frames)")

        segs = align_path(graph, path, model.frame_shift)
        sound_tags = set(model.sound_tags.values())
        id2p = model.id_to_phone

        words: list[Word] = []
        silences: list[Phone] = []
        for tok_idx, phone_id, start, end in segs:
            name = id2p[phone_id]
            if tok_idx < 0:
                silences.append(Phone(name, start, end))
                continue
            tok = tokens[tok_idx]
            if not words or words[-1]._tok_idx != tok_idx:  # type: ignore
                w = Word(word=tok.original, start=start, end=end,
                         type=_word_type(tok.word, name, sound_tags))
                w._tok_idx = tok_idx  # type: ignore
                words.append(w)
            words[-1].phones.append(Phone(name, start, end))
            words[-1].end = end
        for w in words:
            del w._tok_idx  # type: ignore
        return AlignmentResult(words, silences, skipped, duration)


def _word_type(word: str, phone: str, sound_tags: set[str]) -> str:
    if word in ("[UH]", "[UM]"):
        return "f"
    if word.startswith("[") and word.endswith("]") and phone in sound_tags:
        return "s"
    if word.endswith("*"):
        return "c"
    return "w"
