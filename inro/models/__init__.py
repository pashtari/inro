"""Implicit neural representations (INRs): architectures, encodings, solvers and bounds.

Every public name is also available at the top level, as in ``inro.FUTON``.
"""

from .dip import DIPSkip, DIPUNet
from .finer import FINER, FinerLayer
from .finr import FINR
from .fresh import fresh_distance, fresh_select, fresh_spectrum
from .futon import (
    FUTON,
    ChebyshevBasis,
    CosineBasis,
    CPCombiner,
    FUTONEncoding,
    HadamardCombiner,
    LanczosBasis,
    LegendreBasis,
    SharedRows,
    SincBasis,
    Taps,
    TRCombiner,
    TuckerCombiner,
    TriangleBasis,
)
from .futon_bounds import FUTONBound, futon_bound
from .futon_solvers import BCD, LeastSquares, MultiplicativeUpdate
from .gaplanes import GAPlanes
from .gauss import Gauss, GaussLayer
from .grid_inr import GridINR
from .instant_ngp import HashEncoding, InstantNGP
from .mfn import MFN, GaborFilter
from .mlp import MLP, ReLULayer
from .multivector import FeatureGrid, MultiVector
from .pemlp import PEMLP, PositionalEncoding
from .rff import RFF, RFFEncoding
from .siren import SIREN, SineLayer
from .tensorf import TensoRF
from .wire import WIRE, ComplexGaborLayer, RealGaborLayer, RealWIRE

__all__ = [
    "MLP",
    "ReLULayer",
    "RFFEncoding",
    "RFF",
    "PositionalEncoding",
    "PEMLP",
    "SineLayer",
    "SIREN",
    "GaussLayer",
    "Gauss",
    "RealGaborLayer",
    "ComplexGaborLayer",
    "RealWIRE",
    "WIRE",
    "FinerLayer",
    "FINER",
    "FINR",
    "fresh_spectrum",
    "fresh_distance",
    "fresh_select",
    "GaborFilter",
    "MFN",
    "FeatureGrid",
    "MultiVector",
    "SharedRows",
    "Taps",
    "CosineBasis",
    "SincBasis",
    "LegendreBasis",
    "ChebyshevBasis",
    "TriangleBasis",
    "LanczosBasis",
    "HadamardCombiner",
    "CPCombiner",
    "TRCombiner",
    "TuckerCombiner",
    "FUTONEncoding",
    "FUTON",
    "BCD",
    "LeastSquares",
    "MultiplicativeUpdate",
    "FUTONBound",
    "futon_bound",
    "GAPlanes",
    "TensoRF",
    "HashEncoding",
    "InstantNGP",
    "GridINR",
    "DIPUNet",
    "DIPSkip",
]
