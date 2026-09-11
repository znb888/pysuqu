"""Focused import points for native operator and trace preparation helpers.

The implementation remains in the established adapter module so existing
monkeypatches and serialized plans continue to work across the 2.1 series.
"""

import math

import numpy as np

from ...propagation import BackendUnavailable, UnsupportedBackendError
from ..cpp_backend import (
    _banded_bundle_from_arrays,
    _banded_bundle_from_components,
    _connected_components,
    _conjugate_trace,
    _csr_bundle_from_arrays,
    _csr_bundle_from_components,
    _csr_to_dense,
    _diagonal_from_csr,
    _freeze_array,
    _fused_csr_bundle_from_arrays,
    _fused_csr_bundle_from_components,
    _interaction_bundle_from_components,
    _lindblad_effective_operators,
    _native_lindblad_csr_components,
    _native_lindblad_rate_components,
    _native_polynomial_coefficients,
    _prepare_native_trace_payloads,
    _qobj_csr,
    _qobj_from_sparse,
    _qobj_matrix,
    _slice_operator,
    _split_csr_bundle,
    _static_collapse_operator,
    _transform_block_diagonal_basis,
)


class _NormalizedNativeSpline:
    """Represent an interpolation spline in a dimensionless time basis.

    This small utility mirrors the native preparation contract and is useful
    to callers that need coefficients on normalized intervals.  The legacy
    adapter keeps its own coefficient builder, so adding this import point is
    backward compatible.
    """

    def __init__(self, t_axis: np.ndarray, values: np.ndarray, order: int):
        try:
            from scipy.interpolate import make_interp_spline
        except ImportError as exc:  # pragma: no cover - scipy is a dependency
            raise BackendUnavailable("scipy is required for native spline coefficients") from exc
        self.origin = float(t_axis[0])
        self.scale = float(t_axis[-1] - t_axis[0])
        self.t_axis = np.asarray(t_axis, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise UnsupportedBackendError("native spline requires a finite positive time span")
        self.normalized_t = (self.t_axis - self.origin) / self.scale
        if not np.all(np.diff(self.normalized_t) > 0):
            raise UnsupportedBackendError("native spline time normalization requires distinct knots")
        self.spline = make_interp_spline(self.normalized_t, values, k=int(order), bc_type=None)
        self.normalized_knots = np.unique(np.asarray(self.spline.t, dtype=np.float64))
        self.knots = self.origin + self.scale * self.normalized_knots
        indices = np.searchsorted(self.normalized_t, self.normalized_knots)
        safe = np.minimum(indices, len(self.normalized_t) - 1)
        original = (indices < len(self.normalized_t)) & (
            self.normalized_t[safe] == self.normalized_knots
        )
        self.knots[original] = self.t_axis[safe[original]]
        if not np.all(np.diff(self.knots) > 0):
            raise UnsupportedBackendError("native spline breakpoints must remain distinct")

    def _coordinates(self, times: np.ndarray) -> np.ndarray:
        coordinates = (np.asarray(times, dtype=np.float64) - self.origin) / self.scale
        for physical, normalized in ((self.t_axis, self.normalized_t), (self.knots, self.normalized_knots)):
            indices = np.searchsorted(physical, times)
            safe = np.minimum(indices, len(physical) - 1)
            matches = (indices < len(physical)) & (physical[safe] == times)
            coordinates[matches] = normalized[safe[matches]]
        return coordinates

    def values(self, times: np.ndarray) -> np.ndarray:
        return np.asarray(self.spline(self._coordinates(times)), dtype=np.complex128)

    def coefficients(self, times: np.ndarray, order: int) -> np.ndarray:
        starts = self._coordinates(np.asarray(times, dtype=np.float64)[:-1])
        widths = np.diff(times) / self.scale
        coefficients = np.empty((len(starts), int(order) + 1), dtype=np.complex128)
        for power in range(int(order) + 1):
            derivative = np.asarray(self.spline(starts, nu=power), dtype=np.complex128)
            coefficient = derivative / float(math.factorial(power))
            for _ in range(power):
                coefficient *= widths
            coefficients[:, power] = coefficient
        return coefficients


__all__ = [
    "_NormalizedNativeSpline",
    "_banded_bundle_from_arrays",
    "_banded_bundle_from_components",
    "_connected_components",
    "_conjugate_trace",
    "_csr_bundle_from_arrays",
    "_csr_bundle_from_components",
    "_csr_to_dense",
    "_diagonal_from_csr",
    "_freeze_array",
    "_fused_csr_bundle_from_arrays",
    "_fused_csr_bundle_from_components",
    "_interaction_bundle_from_components",
    "_lindblad_effective_operators",
    "_native_lindblad_csr_components",
    "_native_lindblad_rate_components",
    "_native_polynomial_coefficients",
    "_prepare_native_trace_payloads",
    "_qobj_csr",
    "_qobj_from_sparse",
    "_qobj_matrix",
    "_slice_operator",
    "_split_csr_bundle",
    "_static_collapse_operator",
    "_transform_block_diagonal_basis",
]
