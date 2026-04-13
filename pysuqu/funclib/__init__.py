"""
Function library for superconducting qubit simulation.
"""

from importlib import import_module

from .mathlib import *
from .noisemodel import *

_OPTIONAL_SUBMODULES = (
    '.qutiplib',
    '.awgenerator',
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
