"""Transcript normalization + espeak G2P, matching training conventions.

The IPA tokenizer and IPA->canonical mapping are vendored verbatim from the
training code (forced_alignment_training/build_lexicon.py) so that OOV words
phonemized at inference get exactly the pronunciations the model was trained
with. espeak is invoked through phonemizer with the same backend settings as
training: language 'en-us', no stress marks, phone separator ' '.
"""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

_BRACKET_RE = re.compile(r"^\[[\w\s*]+\]$")
_FRAGMENT_RE = re.compile(r"^\S+\*$")
_EDGE_PUNCT_RE = re.compile(r"^[^\w\[\]<>*=]+|[^\w\[\]<>*=]+$")
_STRESS_MARKS = {"ˈ", "ˌ", "ˑ", "ː"}
_TIE_BARS = {"͡", "͜"}


def _strip_stress(ipa: str) -> str:
    return "".join(c for c in ipa if c not in _STRESS_MARKS)


def tokenize_ipa(ipa: str) -> list[str]:
    """Vendored from training build_lexicon._tokenize_ipa (unchanged)."""
    ipa = _strip_stress(ipa).strip()
    if not ipa:
        return []
    if " " in ipa:
        tokens = []
        for t in ipa.split():
            cleaned = t.replace("͡", "").replace("͜", "").strip()
            if cleaned:
                tokens.append(cleaned)
        return tokens
    tokens = []
    i = 0
    chars = list(ipa)
    n = len(chars)
    while i < n:
        c = chars[i]
        if c in ("\t", "\n"):
            i += 1
            continue
        best = None
        for length in (4, 3, 2, 1):
            if i + length > n:
                continue
            candidate = "".join(chars[i:i + length])
            base_chars = [ch for ch in candidate if ch not in _TIE_BARS
                          and unicodedata.category(ch) != "Mn"]
            if not base_chars:
                continue
            best = candidate
            break
        if best:
            tokens.append(best)
            i += len(best)
        else:
            i += 1
    return tokens


def map_ipa_to_canonical(ipa_tokens: list[str],
                         phone_map: dict[str, str]) -> list[str]:
    """Vendored from training build_lexicon._map_ipa_to_canonical."""
    result = []
    i = 0
    n = len(ipa_tokens)
    while i < n:
        if i + 1 < n:
            bigram = ipa_tokens[i] + ipa_tokens[i + 1]
            if bigram in phone_map:
                mapped = phone_map[bigram]
                if mapped:
                    result.append(mapped)
                i += 2
                continue
        token = ipa_tokens[i]
        if token in phone_map:
            mapped = phone_map[token]
            if mapped:
                result.append(mapped)
        else:
            stripped = "".join(c for c in token
                               if unicodedata.category(c) != "Mn"
                               and c not in _TIE_BARS)
            if stripped and stripped in phone_map:
                mapped = phone_map[stripped]
                if mapped:
                    result.append(mapped)
            elif stripped:
                result.append(f"UNK_{stripped}")
        i += 1
    return result


class G2P:
    """espeak-backed G2P producing canonical phones (lazy backend init)."""

    def __init__(self, phone_map: dict, language: str = "en-us",
                 with_stress: bool = False):
        self.ipa_to_canonical: dict[str, str] = phone_map["ipa_to_canonical"]
        self.manual_prons: dict[str, str] = phone_map.get("manual_prons", {})
        self.language = language
        self.with_stress = with_stress
        self._backend = None
        self._separator = None
        self._cache: dict[str, list[str] | None] = {}

    def _init_backend(self):
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        self._backend = EspeakBackend(
            self.language, preserve_punctuation=False,
            with_stress=self.with_stress)
        self._separator = Separator(phone=" ", word=None, syllable="")

    @staticmethod
    def g2p_input(form: str) -> str:
        """Partial-word markers 'th=' -> 'th' before phonemizing."""
        return form[:-1] if form.endswith("=") and len(form) > 1 else form

    def pronounce(self, word: str) -> list[str] | None:
        """Canonical phones for one (normalized) word, or None."""
        if word in self._cache:
            return self._cache[word]
        if word in self.manual_prons:
            pron = self.manual_prons[word].split()
            self._cache[word] = pron
            return pron
        stem = word[:-1] if word.endswith("*") and len(word) > 1 else word
        stem = self.g2p_input(stem)
        if not stem or not any(ch.isalpha() for ch in stem):
            self._cache[word] = None
            return None
        if self._backend is None:
            self._init_backend()
        ipa = self._backend.phonemize([stem], separator=self._separator,
                                      strip=True)[0]
        phones = map_ipa_to_canonical(tokenize_ipa(ipa), self.ipa_to_canonical)
        phones = [p for p in phones if not p.startswith("UNK_")]
        pron = phones or None
        self._cache[word] = pron
        return pron


# CrisperWhisper-style cut-off: "th-", "resched-," (trailing hyphen, optional punctuation)
_CUTOFF_RE = re.compile(r"^([A-Za-z][A-Za-z']*)-+[.,!?;:\u2026]*$")
_VOWELS = set("aeiouy")
# consonant-only fragments: spell the SOUNDS, not the letter names espeak would give
_CONSONANT_PHONES = {
    "b": ["B"], "c": ["K"], "d": ["D"], "f": ["F"], "g": ["G"], "h": ["HH"],
    "j": ["JH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"], "p": ["P"],
    "q": ["K"], "r": ["R"], "s": ["S"], "t": ["T"], "v": ["V"], "w": ["W"],
    "x": ["K", "S"], "z": ["Z"],
}
_DIGRAPHS = {"th": ["TH"], "ch": ["CH"], "sh": ["SH"], "wh": ["W"], "ph": ["F"]}


def cutoff_stem(raw: str) -> str | None:
    """'th-' / 'resched-,' -> 'th' / 'resched'; None if not a cut-off token."""
    m = _CUTOFF_RE.match(raw)
    return m.group(1).lower() if m else None


def consonant_fragment_pron(stem: str) -> list[str] | None:
    """Pronunciation for a vowel-less cut-off fragment ('th', 's', 'str')."""
    if any(ch in _VOWELS for ch in stem):
        return None
    out: list[str] = []
    i = 0
    while i < len(stem):
        if stem[i:i + 2] in _DIGRAPHS:
            out += _DIGRAPHS[stem[i:i + 2]]; i += 2
        elif stem[i] in _CONSONANT_PHONES:
            out += _CONSONANT_PHONES[stem[i]]; i += 1
        else:
            return None
    return out or None


def word_candidates(word: str) -> list[str]:
    """Plausible lexicon forms for a raw transcript token ('extra' inventory).

    Mirrors run_mfcc_wavlm_phone_ablation.word_candidates_for_lexicon.
    """
    if not word:
        return []
    candidates = [word]
    stripped = _EDGE_PUNCT_RE.sub("", word)
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    lower = stripped.lower() if stripped else word.lower()
    if lower and lower not in candidates:
        candidates.append(lower)
    if lower.endswith("*") and lower[:-1] and lower[:-1] not in candidates:
        candidates.append(lower[:-1])
    if lower.startswith("<ext-") and lower.endswith(">"):
        ext = lower[5:-1]
        if ext and ext not in candidates:
            candidates.append(ext)
    return candidates


def preferred_oov_form(raw_cands: list[str]) -> str | None:
    """The single form to add to the lexicon for an OOV token: the last
    candidate (lowercased, stripped) — mirrors extend_eval_lexicon."""
    usable = [c for c in raw_cands
              if c and not (c.startswith("[") and c.endswith("]"))]
    return usable[-1] if usable else None
