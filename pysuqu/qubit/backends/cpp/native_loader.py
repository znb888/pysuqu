"""Optional native extension loading for C++ propagation adapters.

Each entry point is resolved independently so a partially installed or older
extension can still expose the kernels it supports.  Importing this module is
safe when no compiler-built extension is installed.
"""

from __future__ import annotations


def _load_native(name: str):
    try:
        from .... import _native
    except (ImportError, ModuleNotFoundError, OSError):
        return None
    try:
        return getattr(_native, name)
    except AttributeError:
        return None


_native_propagate = _load_native("propagate")
_native_propagate_csr = _load_native("propagate_csr")
_native_propagate_banded = _load_native("propagate_banded")
_native_propagate_fused_csr = _load_native("propagate_fused_csr")
_native_propagate_interaction_csr = _load_native("propagate_interaction_csr")
_native_propagate_lindblad_csr = _load_native("propagate_lindblad_csr")


__all__ = [
    "_load_native",
    "_native_propagate",
    "_native_propagate_banded",
    "_native_propagate_csr",
    "_native_propagate_fused_csr",
    "_native_propagate_interaction_csr",
    "_native_propagate_lindblad_csr",
]
