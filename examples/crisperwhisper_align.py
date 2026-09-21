#!/usr/bin/env python3
"""CrisperWhisper 2.0 transcript -> nyra-forced-aligner timings.

CrisperWhisper 2.0 writes down exactly what was said: fillers ([UM], [UH]),
repetitions, cut-offs ("th-") and vocal events ([laughter]). This example
transcribes a recording verbatim and then force-aligns that transcript, so
every word AND every filler / vocal sound gets a precise start and end time.

    pip install "crisperwhisper[transformers]"        # or [ct2] on NVIDIA/Linux
    pip install nyra-forced-aligner

    python examples/crisperwhisper_align.py interview.wav --out interview.TextGrid
    python examples/crisperwhisper_align.py interview.wav --pro      # noisy audio
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--pro", action="store_true", help="noise-robust aligner model")
    ap.add_argument("--language", default="en")
    ap.add_argument("--cw-model", default="large",
                    help="CrisperWhisper 2.0 size or Hub id (large / turbo / medium / small)")
    ap.add_argument("--cw-backend", default="auto", help="auto / ct2 / transformers")
    ap.add_argument("--out", default=None, help="write .TextGrid or .json")
    args = ap.parse_args()

    from crisperwhisper import CrisperWhisperModel
    from nyra_align import Aligner

    # 1) verbatim transcript (CrisperWhisper's own word times kept for comparison)
    asr = CrisperWhisperModel(args.cw_model, backend=args.cw_backend)
    tr = asr.transcribe(args.audio, language=args.language, word_timestamps=True)
    print(f"\ntranscript: {tr.text}\n")

    # 2) force-align that transcript; casing, punctuation, [um]/[UM] and "th-"
    #    cut-offs are handled by the aligner's tokenizer
    aligner = Aligner(language=args.language, pro=args.pro)
    result = aligner.align(args.audio, tr.text)

    # 3) side by side. Both lists follow transcript order; tokens the aligner
    #    could not pronounce (result.skipped) are absent from result.words.
    cw_words = [w for w in tr.words if w.word.strip() not in set(result.skipped)]
    paired = len(cw_words) == len(result.words)
    print(f"{'word':<16s}{'aligner':>17s}   {'CrisperWhisper':>17s}  type")
    for i, w in enumerate(result.words):
        cw = f"{cw_words[i].start:7.2f} -{cw_words[i].end:7.2f}" if paired else " " * 16
        print(f"{w.word:<16s}{w.start:8.3f} -{w.end:7.3f}   {cw}   {w.type}")
    if result.skipped:
        print(f"\nnot aligned (no pronunciation): {result.skipped}")
    print(f"\n{len(result.words)} words, {len(result.events)} fillers / vocal events / cut-offs")

    if args.out:
        (result.to_json if args.out.lower().endswith(".json") else result.to_textgrid)(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
