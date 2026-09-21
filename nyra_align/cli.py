"""Command-line interface.

Single file (published English model by default; --pro = noise-robust):
    nyra-align audio.wav --text "so i uhm [laughter] went there" --out out.TextGrid

Corpus mode (paired audio + .txt/.lab transcripts, MFA-style layout):
    nyra-align corpus_dir/ --pro --out-dir aligned/ --format json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _write(result, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        result.to_json(out_path)
    else:
        result.to_textgrid(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="audio file or corpus directory")
    ap.add_argument("--model", default=None,
                    help="model dir or Hub id (default: published model for --language)")
    ap.add_argument("--language", default="en", help="alignment language (only 'en' so far)")
    ap.add_argument("--pro", action="store_true", help="use the noise-robust model")
    ap.add_argument("--text", default=None, help="transcript string")
    ap.add_argument("--text-file", default=None, help="transcript file")
    ap.add_argument("--out", default=None, help="output file (single mode)")
    ap.add_argument("--out-dir", default=None, help="output dir (corpus mode)")
    ap.add_argument("--format", choices=["json", "textgrid"], default=None)
    ap.add_argument("--device", default=None, help="cpu / cuda (auto)")
    args = ap.parse_args()

    from .aligner import Aligner

    aligner = Aligner(args.model, language=args.language, pro=args.pro,
                      device=args.device)
    inp = Path(args.input)

    if inp.is_dir():
        out_dir = Path(args.out_dir or (inp.parent / (inp.name + "_aligned")))
        fmt = args.format or "textgrid"
        ext = ".json" if fmt == "json" else ".TextGrid"
        pairs = []
        for audio in sorted(inp.rglob("*")):
            if audio.suffix.lower() not in AUDIO_EXTS:
                continue
            for text_ext in (".txt", ".lab"):
                tf = audio.with_suffix(text_ext)
                if tf.exists():
                    pairs.append((audio, tf))
                    break
        if not pairs:
            sys.exit(f"no audio+transcript pairs found under {inp}")
        n_fail = 0
        t0 = time.time()
        for audio, tf in pairs:
            rel = audio.relative_to(inp).with_suffix(ext)
            try:
                result = aligner.align(str(audio), tf.read_text().strip())
                _write(result, out_dir / rel, fmt)
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"[fail] {audio.name}: {exc}", file=sys.stderr)
        dt = time.time() - t0
        print(f"aligned {len(pairs) - n_fail}/{len(pairs)} files "
              f"in {dt:.1f}s -> {out_dir}")
        return

    if args.text is None and args.text_file is None:
        sys.exit("--text or --text-file required for single-file mode")
    text = args.text if args.text is not None else \
        Path(args.text_file).read_text().strip()

    result = aligner.align(str(inp), text)
    if args.out:
        fmt = args.format or (
            "json" if args.out.lower().endswith(".json") else "textgrid")
        _write(result, Path(args.out), fmt)
        print(f"wrote {args.out} ({len(result.words)} words, "
              f"{len(result.skipped)} skipped)")
    else:
        print(result.to_json())


if __name__ == "__main__":
    main()
