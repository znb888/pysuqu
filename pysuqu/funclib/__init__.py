"""
Function library for superconducting qubit simulation.
"""

from importlib import import_module

from .mathlib import *
from .noisemodel import *

# ``predistortion`` predates the transmission-chain stages and still declares
# names such as ``FIRFilterStage`` and ``DerivativePrecorrectionStage``.  It is
# imported first on purpose so the transmission-chain definitions installed
# afterwards keep winning, while the predistortion-only helpers become
# reachable through ``pysuqu.funclib``.
_OPTIONAL_SUBMODULES = (
    '.qutiplib',
    '.awgenerator',
    '.predistortion',
    '.transmission',
    '.interfile',
)


for _submodule in _OPTIONAL_SUBMODULES:
    try:
        _module = import_module(_submodule, __name__)
    except ModuleNotFoundError:
        continue

    _public_names = getattr(_module, '__all__', None)
    if _public_names is None:
        _public_names = [name for name in vars(_module) if not name.startswith('_')]

    globals().update({name: getattr(_module, name) for name in _public_names})
