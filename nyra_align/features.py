"""Front-end: WavLM feature extraction + the model's projection chain.

Mirrors the training/eval feature path exactly:
  waveform (16 kHz mono)
    -> WavLM-large hidden layer 23 (20 ms frames), long audio in 30 s windows
       with 1 s overlap, overlap frames dropped from the later window
    -> linear upsample 20 ms -> 10 ms
    -> feature chain from meta.json (e.g. PCA -> per-utt CMVN -> splice -> LDA)
"""

from __future__ import annotations

import numpy as np
import torch

MAX_AUDIO_SECONDS = 30
BATCH_OVERLAP_SECONDS = 1


def load_wavlm(model_name: str = "microsoft/wavlm-large",
               device: str = "cpu",
               dtype: "torch.dtype | None" = None) -> "torch.nn.Module":
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    if dtype is not None and dtype != torch.float32:
        model.to(dtype)
    return model


def _read_file(path: str) -> tuple[np.ndarray, int]:
    """Decode an audio file -> (channels, samples) float32, sample rate.

    soundfile (bundled libsndfile: wav, flac, ogg, mp3) is the primary reader;
    torchaudio is tried only if it happens to be installed and soundfile
    cannot decode the container (e.g. m4a)."""
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.T, int(sr)
    except Exception as sf_err:                      # noqa: BLE001
        try:
            import torchaudio
            wav, sr = torchaudio.load(path)
            return wav.numpy().astype(np.float32), int(sr)
        except Exception:                            # noqa: BLE001
            raise RuntimeError(
                f"cannot decode {path!r} ({sf_err}). Convert it to wav/flac first, "
                f"e.g. `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav`.") from sf_err


def load_audio(audio, sample_rate: int = 16_000) -> torch.Tensor:
    """Accept a path, numpy array, or tensor; return mono 16 kHz (1, T).

    Arrays/tensors are assumed to be sampled at `sample_rate`."""
    if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
        wav, sr = _read_file(str(audio))
    else:
        wav = np.asarray(audio.detach().cpu() if isinstance(audio, torch.Tensor) else audio,
                         dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        sr = sample_rate
    if wav.shape[0] > 1:
        wav = wav.mean(axis=0, keepdims=True)
    if sr != 16_000:
        import soxr
        wav = soxr.resample(wav[0], sr, 16_000).astype(np.float32)[None, :]
    return torch.from_numpy(np.ascontiguousarray(wav))


def extract_wavlm(model, waveform: torch.Tensor, layer: int = 23,
                  device: str = "cpu", batch_size: int = 8,
                  return_torch: bool = False):
    """(1, N) waveform -> (T_10ms, 1024) float32, windowed for long audio.

    Long recordings are cut into 30 s windows with 1 s overlap; all
    equal-length windows run through WavLM in batches (one forward per
    `batch_size` windows), the shorter tail window runs on its own. The
    stitch, 20->10 ms upsample, and (optionally) everything downstream stay
    on `device`.
    """
    sr = 16_000
    num_samples = waveform.shape[1]
    max_samples = MAX_AUDIO_SECONDS * sr
    overlap_samples = BATCH_OVERLAP_SECONDS * sr

    if num_samples <= max_samples:
        chunks = [waveform]
    else:
        chunks, start = [], 0
        while start < num_samples:
            end = min(start + max_samples, num_samples)
            chunks.append(waveform[:, start:end])
            if end == num_samples:
                break
            start = end - overlap_samples

    model_dtype = next(model.parameters()).dtype
    full = [c for c in chunks if c.shape[1] == max_samples]
    all_feats: list[torch.Tensor] = [None] * len(chunks)  # type: ignore
    with torch.no_grad():
        for b0 in range(0, len(full), batch_size):
            batch = torch.cat(full[b0:b0 + batch_size], dim=0)
            batch = batch.to(device=device, dtype=model_dtype)
            out = model(batch, output_hidden_states=True)
            hs = out.hidden_states[layer].float()
            for j in range(hs.shape[0]):
                all_feats[b0 + j] = hs[j]
        for i, c in enumerate(chunks):
            if all_feats[i] is None:
                out = model(c.to(device=device, dtype=model_dtype),
                            output_hidden_states=True)
                all_feats[i] = out.hidden_states[layer][0].float()

    if len(all_feats) == 1:
        feats_20ms = all_feats[0]
    else:
        overlap_frames = overlap_samples // 320
        merged = [all_feats[0]]
        for i in range(1, len(all_feats)):
            merged.append(all_feats[i][overlap_frames:]
                          if overlap_frames > 0 else all_feats[i])
        feats_20ms = torch.cat(merged, dim=0)

    feats_10ms = torch.nn.functional.interpolate(
        feats_20ms.T.unsqueeze(0), scale_factor=2,
        mode="linear", align_corners=False,
    ).squeeze(0).T
    if return_torch:
        return feats_10ms.float()
    return feats_10ms.cpu().numpy().astype(np.float32)


def splice(feats: np.ndarray, context: int) -> np.ndarray:
    padded = np.pad(feats, ((context, context), (0, 0)), mode="edge")
    width = 2 * context + 1
    return np.lib.stride_tricks.sliding_window_view(
        padded, (width, feats.shape[1])
    ).reshape(-1, feats.shape[1] * width)[: feats.shape[0]].astype(np.float32)


def apply_chain(raw: np.ndarray, chain: list[dict],
                arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the exported feature chain to raw WavLM features."""
    x = raw
    for op in chain:
        kind = op["op"]
        if kind == "linear":
            x = x @ arrays[op["W"]].T + arrays[op["b"]]
        elif kind == "cmvn":
            mean = x.mean(axis=0)
            std = np.maximum(x.std(axis=0), 1e-10)
            x = (x - mean) / std
        elif kind == "splice":
            x = splice(x, op["context"])
        else:
            raise ValueError(f"unknown chain op {kind!r}")
        x = x.astype(np.float32)
    return x


def splice_torch(x: torch.Tensor, context: int) -> torch.Tensor:
    pad_l = x[:1].expand(context, -1)
    pad_r = x[-1:].expand(context, -1)
    padded = torch.cat([pad_l, x, pad_r], dim=0)
    width = 2 * context + 1
    # (T, D, width) -> (T, width, D) -> (T, width*D), matching numpy layout
    win = padded.unfold(0, width, 1)
    return win.permute(0, 2, 1).reshape(x.shape[0], -1)


def apply_chain_torch(x: torch.Tensor, chain: list[dict],
                      tarrays: dict[str, torch.Tensor]) -> torch.Tensor:
    """GPU/torch version of apply_chain (matches the numpy path)."""
    for op in chain:
        kind = op["op"]
        if kind == "linear":
            x = x @ tarrays[op["W"]].T + tarrays[op["b"]]
        elif kind == "cmvn":
            mean = x.mean(dim=0)
            std = torch.clamp(x.std(dim=0, unbiased=False), min=1e-10)
            x = (x - mean) / std
        elif kind == "splice":
            x = splice_torch(x, op["context"])
        else:
            raise ValueError(f"unknown chain op {kind!r}")
        x = x.float()
    return x
