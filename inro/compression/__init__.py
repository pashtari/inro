"""Image codecs, rate-distortion evaluation and integer arithmetic coding.

PIL and FUTON codecs map image tensors to bytes and back, and
:func:`evaluate_compression` benchmarks either. These APIs live under
``inro.compression``, not in the top-level namespace.
"""

from . import arithmetic_coding, codecs, evaluation
from .arithmetic_coding import *
from .codecs import *
from .evaluation import *

__all__ = [
    *codecs.__all__,
    *evaluation.__all__,
    *arithmetic_coding.__all__,
]
