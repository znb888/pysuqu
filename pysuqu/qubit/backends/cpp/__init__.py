"""Implementation modules for the optional C++ propagation backends.

The public adapter remains in :mod:`pysuqu.qubit.backends.cpp_backend` for
backwards compatibility.  These focused modules provide stable import points
for integrations that need to inspect native loading, preparation, execution,
or result conversion without importing private source files.
"""

from .execution import NativeExecution
from .native_loader import (
    _native_propagate,
    _native_propagate_banded,
    _native_propagate_csr,
    _native_propagate_fused_csr,
    _native_propagate_interaction_csr,
    _native_propagate_lindblad_csr,
)
from .results import _decode_complex_payload

__all__ = [
    "NativeExecution",
    "_decode_complex_payload",
    "_native_propagate",
    "_native_propagate_banded",
    "_native_propagate_csr",
    "_native_propagate_fused_csr",
    "_native_propagate_interaction_csr",
    "_native_propagate_lindblad_csr",
]
