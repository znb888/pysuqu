"""Optional compiled kernels for pysuqu.

The package is importable even when the extension has not been built.  The
public Python backends inspect this module lazily and provide a useful error or
fallback rather than failing package import.
"""

try:
    from ._dynamics import propagate
except (ImportError, ModuleNotFoundError, OSError):
    propagate = None

try:
    from ._dynamics import propagate_csr
except (ImportError, ModuleNotFoundError, OSError):
    propagate_csr = None

try:
    from ._dynamics import propagate_banded
except (ImportError, ModuleNotFoundError, OSError):
    propagate_banded = None

try:
    from ._dynamics import propagate_fused_csr
except (ImportError, ModuleNotFoundError, OSError):
    propagate_fused_csr = None

try:
    from ._dynamics import propagate_interaction_csr
except (ImportError, ModuleNotFoundError, OSError):
    propagate_interaction_csr = None

try:
    from ._dynamics import propagate_lindblad_csr
except (ImportError, ModuleNotFoundError, OSError):
    propagate_lindblad_csr = None

__all__ = [
    "propagate",
    "propagate_csr",
    "propagate_banded",
    "propagate_fused_csr",
    "propagate_interaction_csr",
    "propagate_lindblad_csr",
]
