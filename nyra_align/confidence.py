"""Per-word alignment confidence.

For every frame, compare how well the state the alignment put there explains
the audio with how well the best state of the whole model would: the
difference of their GMM log-likelihoods (nats, >= 0). A word whose audio really
is that word stays close to the best (median ~1.4 nats per frame on Buckeye);
a word the transcript got wrong, or one pushed onto the wrong audio by
untranscribed speech next to it, is far from it (median 25-33).

    confidence = exp(-mean log-likelihood gap per frame / CONFIDENCE_SCALE)

so 1 is a perfect fit. Calibrated on an hour of Buckeye with perturbed
transcripts: correct words have a median of 0.87; below 0.5 catches 99% of
substituted words, 93% of words misplaced by an untranscribed stretch and 96%
of words of an unrelated transcript, and also flags ~10% of correct words
(fast, reduced or noisy speech). Treat it as a ranking for review, not a
probability.
"""

from __future__ import annotations

import numpy as np
import torch

CONFIDENCE_SCALE = 10.0


def _pdf_loglikes(x: torch.Tensor, computer) -> torch.Tensor:
    """(B, D) features -> (B, P) log-likelihoods of the computer's pdfs."""
    s = x @ computer.miv.T + (x * x) @ computer.neg_half_iv.T + computer.gconsts
    seg = computer.seg.unsqueeze(0).expand(x.shape[0], -1)
    m = torch.full((x.shape[0], computer.n_pdfs), -float("inf"),
                   device=x.device, dtype=s.dtype)
    m.scatter_reduce_(1, seg, s, reduce="amax")
    tot = torch.zeros_like(m)
    tot.scatter_add_(1, seg, torch.exp(s - m.gather(1, seg)))
    return m + torch.log(tot)


def best_log_likelihoods(feats_t: torch.Tensor, all_pdfs, block: int = 8192) -> np.ndarray:
    """(T, D) features -> (T,) best log-likelihood over every pdf of the model.

    `all_pdfs` is an EmissionComputer over `np.arange(model.num_pdfs)`.
    """
    out = []
    with torch.no_grad():
        for b0 in range(0, feats_t.shape[0], block):
            out.append(_pdf_loglikes(feats_t[b0:b0 + block], all_pdfs).max(dim=1).values)
    return torch.cat(out).float().cpu().numpy()


def frame_log_likelihoods(feats_t: torch.Tensor, all_pdfs, frame_pdfs: np.ndarray,
                          block: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Log-likelihood of each frame's aligned pdf, and of the best pdf, in one pass."""
    pdfs = torch.from_numpy(np.ascontiguousarray(frame_pdfs, dtype=np.int64)).to(feats_t.device)
    forced, best = [], []
    with torch.no_grad():
        for b0 in range(0, feats_t.shape[0], block):
            ll = _pdf_loglikes(feats_t[b0:b0 + block], all_pdfs)
            forced.append(ll.gather(1, pdfs[b0:b0 + block, None])[:, 0])
            best.append(ll.max(dim=1).values)
    return (torch.cat(forced).float().cpu().numpy(),
            torch.cat(best).float().cpu().numpy())


def word_confidence(forced: np.ndarray, best: np.ndarray) -> float:
    """Confidence of one word from its frames' aligned and best log-likelihoods."""
    if len(forced) == 0:
        return 0.0
    gap = float(np.mean(np.maximum(best - forced, 0.0)))
    return float(np.exp(-gap / CONFIDENCE_SCALE))
