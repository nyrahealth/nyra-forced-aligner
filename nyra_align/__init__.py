"""nyra-align: verbatim-aware, noise-robust forced alignment without Kaldi.

    from nyra_align import Aligner
    aligner = Aligner()                    # English, published model, device auto
    aligner = Aligner(pro=True)            # noise-robust variant
    result = aligner.align("audio.wav", "so i uhm [laughter] went there")
    result.words                # Word(word, start, end, type, phones)
    result.to_textgrid("out.TextGrid")
"""

from .aligner import (DEFAULT_MODELS, SUPPORTED_LANGUAGES, Aligner,
                      resolve_model)
from .model import NyraModel
from .outputs import AlignmentResult, Phone, Word

from .version import __version__
__all__ = ["Aligner", "NyraModel", "AlignmentResult", "Word", "Phone",
           "DEFAULT_MODELS", "SUPPORTED_LANGUAGES", "resolve_model"]
