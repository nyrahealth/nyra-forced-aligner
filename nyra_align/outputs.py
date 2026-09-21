"""Alignment result containers and writers (JSON / TextGrid)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Phone:
    phone: str
    start: float
    end: float


@dataclass
class Word:
    word: str
    start: float
    end: float
    type: str = "w"          # w=word, f=filler, s=sound event, c=cutoff
    phones: list[Phone] = field(default_factory=list)


@dataclass
class AlignmentResult:
    words: list[Word]
    silences: list[Phone]
    skipped: list[str]                    # tokens that could not be aligned
    duration: float

    @property
    def events(self) -> list[Word]:
        return [w for w in self.words if w.type in ("f", "s", "c")]

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "skipped_tokens": self.skipped,
            "words": [
                {
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "type": w.type,
                    "phones": [
                        {"phone": p.phone,
                         "start": round(p.start, 3),
                         "end": round(p.end, 3)} for p in w.phones
                    ],
                }
                for w in self.words
            ],
        }

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(s)
        return s

    def to_textgrid(self, path: str | Path | None = None) -> str:
        def esc(s: str) -> str:
            return s.replace('"', '""')

        word_iv = _fill_gaps(
            [(w.start, w.end, w.word) for w in self.words], self.duration)
        phone_iv = _fill_gaps(
            [(p.start, p.end, p.phone)
             for w in self.words for p in w.phones]
            + [(s.start, s.end, "SIL") for s in self.silences],
            self.duration)

        lines = [
            'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
            "xmin = 0", f"xmax = {self.duration:.3f}",
            "tiers? <exists>", "size = 2", "item []:",
        ]
        for tier_no, (name, ivs) in enumerate(
                [("words", word_iv), ("phones", phone_iv)], start=1):
            lines += [
                f"    item [{tier_no}]:",
                '        class = "IntervalTier"',
                f'        name = "{name}"',
                "        xmin = 0", f"        xmax = {self.duration:.3f}",
                f"        intervals: size = {len(ivs)}",
            ]
            for i, (s, e, label) in enumerate(ivs, start=1):
                lines += [
                    f"        intervals [{i}]:",
                    f"            xmin = {s:.3f}",
                    f"            xmax = {e:.3f}",
                    f'            text = "{esc(label)}"',
                ]
        out = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(out)
        return out


def _fill_gaps(intervals: list[tuple[float, float, str]],
               duration: float) -> list[tuple[float, float, str]]:
    """Sort intervals and fill gaps with empty labels (Praat requirement)."""
    ivs = sorted((s, e, t) for s, e, t in intervals if e > s)
    out: list[tuple[float, float, str]] = []
    cur = 0.0
    for s, e, t in ivs:
        s, e = max(s, cur), max(e, cur)
        if s > cur:
            out.append((cur, s, ""))
        if e > s:
            out.append((s, e, t))
            cur = e
    if duration > cur:
        out.append((cur, duration, ""))
    return out
