"""Python adapter for the optional C++ propagation kernel.

The extension is deliberately optional.  Installing the normal ``pysuqu``
package continues to work without a compiler; users who request ``backend='cpp'``
receive a precise error when the extension has not been built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import qutip as qt

from ..propagation import (
    BackendUnavailable,
    BatchPropagationResult,
    UnsupportedBackendError,
    _validate_trace,
)


try:
    from ..._native import propagate as _native_propagate
except (ImportError, ModuleNotFoundError, OSError):
    _native_propagate = None

try:
    from ..._native import propagate_csr as _native_propagate_csr
except (ImportError, ModuleNotFoundError, OSError):
    _native_propagate_csr = None

try:
    from ..._native import propagate_banded as _native_propagate_banded
except (ImportError, ModuleNotFoundError, OSError):
    _native_propagate_banded = None

try:
    from ..._native import propagate_fused_csr as _native_propagate_fused_csr
except (ImportError, ModuleNotFoundError, OSError):
    _native_propagate_fused_csr = None

try:
    from ..._native import propagate_interaction_csr as _native_propagate_interaction_csr
except (ImportError, ModuleNotFoundError, OSError):
    _native_propagate_interaction_csr = None


def cpp_backend_available() -> bool:
    return callable(_native_propagate)


def cpp_csr_backend_available() -> bool:
    """Return whether the optional CSR kernel is available."""
    return callable(_native_propagate_csr)


def cpp_banded_backend_available() -> bool:
    """Return whether the optional exact banded kernel is available."""
    return callable(_native_propagate_banded)


def cpp_fused_csr_backend_available() -> bool:
    """Return whether the optional fused-union CSR kernel is available."""
    return callable(_native_propagate_fused_csr)


def cpp_interaction_backend_available() -> bool:
    """Return whether the exact diagonal interaction-picture kernel exists."""
    return callable(_native_propagate_interaction_csr)


def _qobj_matrix(value: qt.Qobj) -> np.ndarray:
    data = np.asarray(value.full(), dtype=np.complex128)
    if not np.all(np.isfinite(data)):
        raise ValueError("Native Hamiltonian and drive operators must contain finite values.")
    return np.ascontiguousarray(data)


def _qobj_csr(value: qt.Qobj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exact CSR view of a Qobj without dropping small entries."""
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - scipy is a package dependency
        raise BackendUnavailable("scipy is required for the native CSR backend") from exc

    data = getattr(value, "data", None)
    as_scipy = getattr(data, "as_scipy", None)
    as_ndarray = getattr(data, "as_ndarray", None)
    if callable(as_scipy):
        source = as_scipy()
    elif callable(as_ndarray):
        source = as_ndarray()
    else:
        source = data
    matrix = sparse.csr_matrix(source, dtype=np.complex128)
    matrix.sum_duplicates()
    # Removing explicit mathematical zeros is exact and reduces the hot-loop
    # work for operators assembled by tensor products.
    matrix.eliminate_zeros()
    matrix.sort_indices()
    values = np.ascontiguousarray(matrix.data, dtype=np.complex128)
    indices = np.ascontiguousarray(matrix.indices, dtype=np.int64)
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Native Hamiltonian and drive operators must contain finite values.")
    return values, indices, indptr


def _csr_to_dense(
    matrix: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int,
) -> np.ndarray:
    """Materialize a dense matrix from an already prepared CSR tuple."""
    data, indices, indptr = matrix
    result = np.zeros((n, n), dtype=np.complex128)
    for row in range(n):
        begin = int(indptr[row])
        end = int(indptr[row + 1])
        if end > begin:
            result[row, indices[begin:end]] += data[begin:end]
    return np.ascontiguousarray(result, dtype=np.complex128)


def _banded_bundle_from_components(
    matrices: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert exact CSR operators into a shared diagonal-offset layout.

    The returned arrays are ``offsets[B]``, ``h0[B, n]`` and
    ``controls[K, B, n]``.  Missing structural entries remain mathematical
    zeros; no magnitude-based pruning is performed.
    """
    offset_set = set()
    for _, indices, indptr in matrices:
        for row in range(n):
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            offset_set.update(int(column) - row for column in indices[begin:end])
    if not offset_set:
        offset_set.add(0)
    offsets = np.ascontiguousarray(np.array(sorted(offset_set), dtype=np.int64))
    offset_index = {int(offset): index for index, offset in enumerate(offsets)}
    values = np.zeros(
        (len(matrices), len(offsets), n),
        dtype=np.complex128,
    )
    for operator_index, (data, indices, indptr) in enumerate(matrices):
        for row in range(n):
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            for entry in range(begin, end):
                band = offset_index[int(indices[entry]) - row]
                values[operator_index, band, row] += data[entry]
    return (
        offsets,
        np.ascontiguousarray(values[0], dtype=np.complex128),
        np.ascontiguousarray(values[1:], dtype=np.complex128),
    )


def _banded_bundle_from_arrays(
    values: Sequence[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a banded bundle from dense arrays during one-time preparation."""
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - scipy is a package dependency
        raise BackendUnavailable("scipy is required for the native banded backend") from exc
    matrices = []
    for value in values:
        matrix = sparse.csr_matrix(np.asarray(value, dtype=np.complex128))
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        matrices.append(
            (
                np.ascontiguousarray(matrix.data, dtype=np.complex128),
                np.ascontiguousarray(matrix.indices, dtype=np.int64),
                np.ascontiguousarray(matrix.indptr, dtype=np.int64),
            )
        )
    return _banded_bundle_from_components(matrices, n)


def _fused_csr_bundle_from_components(
    matrices: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one CSR pattern containing the static and all control operators.

    The first two returned arrays are the static values and control values at
    the union pattern; the remaining arrays are column indices and row
    pointers.  Structural zeros are retained in the value planes so the
    operation remains mathematically identical to applying each input matrix.
    """
    coordinates = set()
    for _, indices, indptr in matrices:
        for row in range(n):
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            coordinates.update((row, int(column)) for column in indices[begin:end])
    ordered = sorted(coordinates)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    for row, _ in ordered:
        row_ptr[row + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)
    indices = np.ascontiguousarray(
        np.array([column for _, column in ordered], dtype=np.int64)
    )
    position = {coordinate: index for index, coordinate in enumerate(ordered)}
    static_values = np.zeros(len(ordered), dtype=np.complex128)
    control_values = np.zeros(
        (max(0, len(matrices) - 1), len(ordered)),
        dtype=np.complex128,
    )
    for operator_index, (data, matrix_indices, indptr) in enumerate(matrices):
        target = static_values if operator_index == 0 else control_values[operator_index - 1]
        for row in range(n):
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            for entry in range(begin, end):
                target[position[(row, int(matrix_indices[entry]))]] += data[entry]
    return (
        np.ascontiguousarray(static_values, dtype=np.complex128),
        np.ascontiguousarray(control_values, dtype=np.complex128),
        indices,
        np.ascontiguousarray(row_ptr, dtype=np.int64),
    )


def _fused_csr_bundle_from_arrays(
    values: Sequence[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a fused CSR bundle from dense arrays during one-time preparation."""
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - scipy is a package dependency
        raise BackendUnavailable("scipy is required for the native CSR backend") from exc
    matrices = []
    for value in values:
        matrix = sparse.csr_matrix(np.asarray(value, dtype=np.complex128))
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        matrices.append(
            (
                np.ascontiguousarray(matrix.data, dtype=np.complex128),
                np.ascontiguousarray(matrix.indices, dtype=np.int64),
                np.ascontiguousarray(matrix.indptr, dtype=np.int64),
            )
        )
    return _fused_csr_bundle_from_components(matrices, n)


def _diagonal_from_csr(
    matrix: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int,
) -> Tuple[bool, np.ndarray]:
    """Return an exact diagonal test and the diagonal values of a CSR matrix."""
    data, indices, indptr = matrix
    diagonal = np.zeros(n, dtype=np.complex128)
    for row in range(n):
        begin = int(indptr[row])
        end = int(indptr[row + 1])
        for entry in range(begin, end):
            column = int(indices[entry])
            if column != row and data[entry] != 0.0:
                return False, diagonal
            if column == row:
                diagonal[row] += data[entry]
    return True, diagonal


def _interaction_bundle_from_components(
    controls: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a fused CSR union for controls in an interaction picture."""
    empty = (
        np.empty(0, dtype=np.complex128),
        np.empty(0, dtype=np.int64),
        np.zeros(n + 1, dtype=np.int64),
    )
    return _fused_csr_bundle_from_components([empty, *controls], n)


@dataclass(frozen=True)
class _NativePlan:
    """Immutable numerical payload shared by repeated native propagations."""

    matrix_format: str
    static_matrix: Optional[np.ndarray]
    controls: Optional[np.ndarray]
    iq: np.ndarray
    t_axis: np.ndarray
    lo_freqs: np.ndarray
    mode: int
    sparse_kernel: str = "standard"
    frame: str = "lab"
    metadata: Optional[Dict[str, Any]] = None
    h0_csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    controls_csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    band_offsets: Optional[np.ndarray] = None
    h0_banded: Optional[np.ndarray] = None
    controls_banded: Optional[np.ndarray] = None
    fused_csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    interaction_energies: Optional[np.ndarray] = None
    interaction_basis: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _NativePayload:
    plan: _NativePlan
    initial_matrix: np.ndarray
    dims: List[Any]
    normalize_columns: np.ndarray


def _freeze_array(value: np.ndarray) -> np.ndarray:
    """Mark prepared numerical storage read-only so accidental mutation is visible."""
    array = np.ascontiguousarray(value)
    array.flags.writeable = False
    return array


def _csr_bundle_from_components(matrices, n: Optional[int] = None):
    """Pack already-converted CSR tuples without repeating data conversion."""
    if not matrices:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.int64),
            np.empty((0, int(n or 0) + 1), dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            0,
        )
    data = np.ascontiguousarray(
        np.concatenate([item[0] for item in matrices]),
        dtype=np.complex128,
    )
    indices = np.ascontiguousarray(
        np.concatenate([item[1] for item in matrices]),
        dtype=np.int64,
    )
    indptr = np.ascontiguousarray(
        np.stack([item[2] for item in matrices], axis=0),
        dtype=np.int64,
    )
    offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
    for index, item in enumerate(matrices):
        offsets[index + 1] = offsets[index] + int(item[0].size)
    return data, indices, indptr, offsets, int(data.size)


def _csr_bundle_from_arrays(values: Sequence[np.ndarray], n: Optional[int] = None):
    """Build concatenated CSR storage for already materialized dense arrays."""
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - scipy is a package dependency
        raise BackendUnavailable("scipy is required for the native CSR backend") from exc
    matrices = []
    for value in values:
        matrix = sparse.csr_matrix(np.asarray(value, dtype=np.complex128))
        matrix.eliminate_zeros()
        matrix.sort_indices()
        matrices.append(
            (
                np.ascontiguousarray(matrix.data, dtype=np.complex128),
                np.ascontiguousarray(matrix.indices, dtype=np.int64),
                np.ascontiguousarray(matrix.indptr, dtype=np.int64),
            )
        )
    return _csr_bundle_from_components(matrices, n=n)


def _split_csr_bundle(bundle, n_controls: int):
    """Split a [static, control...] CSR bundle into native ABI components."""
    data, indices, indptr, offsets, _ = bundle
    static_end = int(offsets[1])
    h0 = (
        np.ascontiguousarray(data[:static_end], dtype=np.complex128),
        np.ascontiguousarray(indices[:static_end], dtype=np.int64),
        np.ascontiguousarray(indptr[0], dtype=np.int64),
    )
    control_data = np.ascontiguousarray(data[static_end:], dtype=np.complex128)
    control_indices = np.ascontiguousarray(indices[static_end:], dtype=np.int64)
    control_indptr = np.ascontiguousarray(indptr[1:1 + n_controls], dtype=np.int64)
    control_offsets = np.ascontiguousarray(
        offsets[1:2 + n_controls] - static_end,
        dtype=np.int64,
    )
    return h0, (control_data, control_indices, control_indptr, control_offsets)


def _decode_complex_payload(payload, shape):
    """Decode the compact bytes returned by the CPython C extension."""
    if payload is None:
        return None
    if isinstance(payload, (bytes, bytearray, memoryview)):
        array = np.frombuffer(payload, dtype=np.complex128)
    else:
        array = np.asarray(payload, dtype=np.complex128)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise RuntimeError(
            "native backend returned {} complex values, expected {}".format(
                array.size,
                expected,
            )
        )
    return np.array(array, dtype=np.complex128, copy=True).reshape(shape)


@dataclass
class NativePropagationResult:
    """Small Result-compatible object returned by the native backend."""

    final_state: qt.Qobj
    states: List[qt.Qobj]
    times: np.ndarray
    stats: Dict[str, Any]
    # Native propagation currently rejects e_ops, but exposing an empty
    # expectation collection keeps the common QuTiP Result attribute contract.
    expect: List[Any] = field(default_factory=list)


class CppPropagationBackend:
    """Prepared native backend with dense and exact CSR propagation paths.

    Operator conversion and validation are performed once per prepared
    context.  State vectors remain the only per-call payload, so repeated
    fidelity/channel evaluations do not rebuild Hamiltonian storage.
    """

    def __init__(self, prepared) -> None:
        if not cpp_backend_available():
            raise BackendUnavailable(
                "The optional C++ backend is not built. "
                "Install the native build dependencies and set PYSUQU_BUILD_NATIVE=1."
            )
        self.prepared = prepared
        if prepared.c_ops:
            raise UnsupportedBackendError(
                "The native backend supports coherent ket propagation only; "
                "use backend='qutip_compiled' for c_ops."
            )
        self._plan: Optional[_NativePlan] = None

    @classmethod
    def from_prepared(cls, prepared) -> "CppPropagationBackend":
        return cls(prepared)

    def _requested_sparse_kernel(self) -> str:
        """Return the validated sparse-kernel preference.

        ``extra['sparse_kernel']`` was accepted by the first structured-sparse
        release.  Keep reading it as a compatibility alias, while the typed
        ``PropagationOptions.sparse_kernel`` field is now authoritative.
        """
        requested = str(
            getattr(self.prepared.options, "sparse_kernel", "auto")
        ).lower()
        legacy = getattr(self.prepared.options, "extra", {}).get("sparse_kernel")
        if requested == "auto" and legacy is not None:
            requested = str(legacy).lower()
        if requested not in {"auto", "standard", "fused"}:
            raise ValueError(
                "sparse_kernel must be 'auto', 'standard', or 'fused'"
            )
        return requested

    def _resolve_matrix_format(
        self,
        n: int,
        total_nnz: int,
        matrix_count: int,
        band_count: Optional[int] = None,
    ) -> str:
        requested = str(getattr(self.prepared.options, "matrix_format", "auto")).lower()
        sparse_kernel = self._requested_sparse_kernel()
        if sparse_kernel == "fused" and requested in {"dense", "banded"}:
            raise UnsupportedBackendError(
                "sparse_kernel='fused' requires matrix_format='csr' or 'auto'."
            )
        if sparse_kernel == "fused" and requested == "auto":
            if not cpp_fused_csr_backend_available():
                raise BackendUnavailable(
                    "The native fused CSR kernel is unavailable; rebuild the optional extension."
                )
            # An explicit kernel choice is authoritative even for small
            # dimensions where the automatic density heuristic would prefer
            # a dense array.
            return "csr"
        if requested == "banded":
            if not cpp_banded_backend_available():
                raise BackendUnavailable(
                    "The native banded kernel is unavailable; rebuild the optional extension."
                )
            if band_count is None:
                raise UnsupportedBackendError(
                    "banded matrix storage requires a diagonal-offset analysis."
                )
            if band_count > max(8, n // 2):
                raise UnsupportedBackendError(
                    "banded storage is too wide for this operator; use matrix_format='csr'."
                )
            return "banded"
        if requested == "csr":
            if not cpp_csr_backend_available():
                raise BackendUnavailable(
                    "The native CSR kernel is unavailable; rebuild the optional extension."
                )
            return "csr"
        if requested == "dense" or (requested == "auto" and n < 8):
            return "dense"
        if (
            requested == "auto"
            and band_count is not None
            and cpp_banded_backend_available()
            and n >= 8
            and band_count <= min(8, max(4, n // 4))
            and total_nnz >= 1.5 * max(1, band_count * n)
        ):
            return "banded"
        slots = max(1, int(matrix_count) * int(n) * int(n))
        density = float(total_nnz) / float(slots)
        if (
            cpp_csr_backend_available()
            and n >= 8
            and density <= float(getattr(self.prepared.options, "sparse_threshold", 0.6))
        ):
            return "csr"
        return "dense"

    @staticmethod
    def _require_regular_grid(t_axis: np.ndarray) -> None:
        """Validate the strictly increasing grid accepted by the native kernel.

        The native interpolator resolves the actual source interval and width
        for every stage, so equal spacing is not a physical or numerical
        requirement.  Keep this helper name for compatibility with callers
        that used the former private method.
        """
        if len(t_axis) < 2:
            return
        deltas = np.diff(t_axis)
        if np.any(deltas <= 0.0) or not np.all(np.isfinite(deltas)):
            raise UnsupportedBackendError(
                "The C++ backend requires a finite, strictly increasing tlist grid."
            )

    @staticmethod
    def _freeze_plan_arrays(plan: _NativePlan) -> _NativePlan:
        metadata = plan.metadata
        if metadata is not None:
            metadata = dict(metadata)
            for key in ("basis_transform", "full_basis"):
                value = metadata.get(key)
                if isinstance(value, np.ndarray):
                    metadata[key] = _freeze_array(value)
        h0_csr = plan.h0_csr
        if h0_csr is not None:
            h0_csr = tuple(_freeze_array(item) for item in h0_csr)
        controls_csr = plan.controls_csr
        if controls_csr is not None:
            controls_csr = tuple(_freeze_array(item) for item in controls_csr)
        band_offsets = plan.band_offsets
        if band_offsets is not None:
            band_offsets = _freeze_array(band_offsets)
        h0_banded = plan.h0_banded
        if h0_banded is not None:
            h0_banded = _freeze_array(h0_banded)
        controls_banded = plan.controls_banded
        if controls_banded is not None:
            controls_banded = _freeze_array(controls_banded)
        fused_csr = plan.fused_csr
        if fused_csr is not None:
            fused_csr = tuple(_freeze_array(item) for item in fused_csr)
        interaction_energies = plan.interaction_energies
        if interaction_energies is not None:
            interaction_energies = _freeze_array(interaction_energies)
        interaction_basis = plan.interaction_basis
        if interaction_basis is not None:
            interaction_basis = _freeze_array(interaction_basis)
        return _NativePlan(
            matrix_format=plan.matrix_format,
            static_matrix=(None if plan.static_matrix is None else _freeze_array(plan.static_matrix)),
            controls=(None if plan.controls is None else _freeze_array(plan.controls)),
            iq=_freeze_array(plan.iq),
            t_axis=_freeze_array(plan.t_axis),
            lo_freqs=_freeze_array(plan.lo_freqs),
            mode=plan.mode,
            sparse_kernel=plan.sparse_kernel,
            frame=plan.frame,
            metadata=metadata,
            h0_csr=h0_csr,
            controls_csr=controls_csr,
            band_offsets=band_offsets,
            h0_banded=h0_banded,
            controls_banded=controls_banded,
            fused_csr=fused_csr,
            interaction_energies=interaction_energies,
            interaction_basis=interaction_basis,
        )

    def _plan_from_qobjs(
        self,
        static_operator: qt.Qobj,
        drive_operators: Sequence[qt.Qobj],
        iq: np.ndarray,
        t_axis: np.ndarray,
        lo_freqs: np.ndarray,
        mode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _NativePlan:
        n = int(static_operator.shape[0])
        requested = str(getattr(self.prepared.options, "matrix_format", "auto")).lower()
        sparse_kernel = self._requested_sparse_kernel()
        if sparse_kernel == "fused" and requested in {"dense", "banded"}:
            raise UnsupportedBackendError(
                "sparse_kernel='fused' requires matrix_format='csr' or 'auto'."
            )
        if requested == "dense":
            matrix_format = "dense"
            control_csrs = []
            static_csr = None
            banded_bundle = None
            fused_bundle = None
            use_fused = False
        else:
            static_csr = _qobj_csr(static_operator)
            control_csrs = [_qobj_csr(operator) for operator in drive_operators]
            matrices = [static_csr, *control_csrs]
            total_nnz = sum(int(item[0].size) for item in matrices)
            slots = max(1, (1 + len(control_csrs)) * n * n)
            density = float(total_nnz) / float(slots)
            dense_fast_path = (
                requested == "auto"
                and sparse_kernel != "fused"
                and density > float(
                    getattr(self.prepared.options, "sparse_threshold", 0.6)
                )
            )
            # Build only layouts that can still be selected.  In particular,
            # an explicit fused kernel never needs a band analysis, and a
            # standard CSR request should not pay for a second union pattern.
            need_banded = (
                requested == "banded"
                or (requested == "auto" and sparse_kernel != "fused")
            ) and not dense_fast_path
            need_fused = (
                requested == "fused_csr"
                or sparse_kernel == "fused"
                or (
                    requested == "auto"
                    and sparse_kernel == "auto"
                    and len(control_csrs) >= 2
                )
            ) and not dense_fast_path
            banded_bundle = (
                _banded_bundle_from_components(matrices, n)
                if need_banded
                else None
            )
            fused_bundle = (
                _fused_csr_bundle_from_components(matrices, n)
                if need_fused and cpp_fused_csr_backend_available()
                else None
            )
            band_count = None if banded_bundle is None else int(banded_bundle[0].size)
            matrix_format = self._resolve_matrix_format(
                n,
                total_nnz,
                1 + len(control_csrs),
                band_count=band_count,
            )
            union_nnz = None if fused_bundle is None else int(fused_bundle[0].size)
            use_fused = (
                matrix_format == "csr"
                and fused_bundle is not None
                and (
                    sparse_kernel == "fused"
                    or (
                        requested == "auto"
                        and sparse_kernel == "auto"
                        and union_nnz is not None
                        and len(control_csrs) >= 2
                        and total_nnz >= 1.5 * max(1, union_nnz)
                    )
                )
            )
            if (
                requested == "auto"
                and matrix_format == "banded"
                and fused_bundle is not None
                and len(control_csrs) >= 2
                and sparse_kernel == "auto"
                and union_nnz is not None
                and total_nnz >= 1.5 * max(1, union_nnz)
            ):
                matrix_format = "csr"
                use_fused = True
        if requested == "fused_csr":
            if not cpp_fused_csr_backend_available():
                raise BackendUnavailable(
                    "The native fused CSR kernel is unavailable; rebuild the optional extension."
                )
            if fused_bundle is None:
                raise UnsupportedBackendError(
                    "fused CSR storage requires the CSR structure during preparation."
                )
            matrix_format = "csr"
            use_fused = True
        if use_fused:
            static_values, control_values, indices, indptr = fused_bundle
            return self._freeze_plan_arrays(_NativePlan(
                matrix_format="csr",
                static_matrix=None,
                controls=None,
                iq=np.asarray(iq, dtype=np.complex128),
                t_axis=np.asarray(t_axis, dtype=np.float64),
                lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
                mode=mode,
                sparse_kernel="fused",
                metadata=metadata,
                fused_csr=(static_values, control_values, indices, indptr),
            ))
        if matrix_format == "banded":
            if banded_bundle is None:
                raise UnsupportedBackendError(
                    "banded matrix storage requires the CSR structure during preparation."
                )
            offsets, h0_banded, controls_banded = banded_bundle
            return self._freeze_plan_arrays(_NativePlan(
                matrix_format="banded",
                static_matrix=None,
                controls=None,
                iq=np.asarray(iq, dtype=np.complex128),
                t_axis=np.asarray(t_axis, dtype=np.float64),
                lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
                mode=mode,
                metadata=metadata,
                band_offsets=offsets,
                h0_banded=h0_banded,
                controls_banded=controls_banded,
            ))
        if matrix_format == "csr":
            assert static_csr is not None
            bundle = _csr_bundle_from_components([static_csr, *control_csrs], n=n)
            h0_csr, controls_csr = _split_csr_bundle(bundle, len(drive_operators))
            return self._freeze_plan_arrays(_NativePlan(
                matrix_format="csr",
                static_matrix=None,
                controls=None,
                iq=np.asarray(iq, dtype=np.complex128),
                t_axis=np.asarray(t_axis, dtype=np.float64),
                lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
                mode=mode,
                metadata=metadata,
                h0_csr=h0_csr,
                controls_csr=controls_csr,
            ))

        if static_csr is not None:
            static_matrix = _csr_to_dense(static_csr, n)
        else:
            static_matrix = _qobj_matrix(static_operator)
        if drive_operators:
            if control_csrs:
                control_matrices = [_csr_to_dense(matrix, n) for matrix in control_csrs]
            else:
                control_matrices = [_qobj_matrix(operator) for operator in drive_operators]
            controls = np.ascontiguousarray(
                np.stack(control_matrices, axis=0),
                dtype=np.complex128,
            )
        else:
            controls = np.empty((0, n, n), dtype=np.complex128)
        return self._freeze_plan_arrays(_NativePlan(
            matrix_format="dense",
            static_matrix=static_matrix,
            controls=controls,
            iq=np.asarray(iq, dtype=np.complex128),
            t_axis=np.asarray(t_axis, dtype=np.float64),
            lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
            mode=mode,
            metadata=metadata,
        ))

    def _plan_from_interaction_qobjs(
        self,
        static_operator: qt.Qobj,
        drive_operators: Sequence[qt.Qobj],
        iq: np.ndarray,
        t_axis: np.ndarray,
        lo_freqs: np.ndarray,
        mode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _NativePlan:
        """Prepare an exact interaction-picture plan.

        A structurally diagonal H0 is used directly.  For a finite Hermitian
        H0, an exact unitary eigendecomposition supplies the same diagonal
        representation (without truncating levels); transformed drives share
        one union CSR pattern.  No operator entries are thresholded or
        projected here.
        """
        if not cpp_interaction_backend_available():
            raise BackendUnavailable(
                "The native interaction-picture kernel is unavailable; rebuild the optional extension."
            )
        requested_format = str(
            getattr(self.prepared.options, "matrix_format", "auto")
        ).lower()
        if requested_format in {"dense", "banded"}:
            raise UnsupportedBackendError(
                "frame='interaction_exact' requires matrix_format='auto', 'csr', or 'fused_csr'."
            )
        static_csr = _qobj_csr(static_operator)
        static_diagonal, energies = _diagonal_from_csr(
            static_csr,
            int(static_operator.shape[0]),
        )
        if static_diagonal and np.any(np.imag(energies) != 0.0):
            # The interaction kernel's diagonal integrating factor is a
            # unitary phase.  A complex diagonal term belongs to a
            # non-Hermitian/open-system model and must stay on the lab path
            # rather than being silently projected to its real part.
            raise UnsupportedBackendError(
                "frame='interaction_exact' requires real diagonal energies."
            )
        interaction_basis = None
        if static_diagonal:
            control_csrs = [_qobj_csr(operator) for operator in drive_operators]
        else:
            # A Hermitian static Hamiltonian can be diagonalized exactly by a
            # unitary basis change.  Keep this opt-in frame explicit and limit
            # the dense eigendecomposition to practical gate dimensions.
            dimension = int(static_operator.shape[0])
            if dimension > 256:
                raise UnsupportedBackendError(
                    "frame='interaction_exact' diagonalization is limited to dimension <= 256."
                )
            static_matrix = _qobj_matrix(static_operator)
            # ``eigh`` assumes a Hermitian input.  Permit only the tiny
            # antisymmetry introduced by floating-point construction of a
            # mathematically Hermitian Qobj; anything larger is rejected
            # instead of being silently projected into a new Hamiltonian.
            adjoint = static_matrix.conj().T
            scale = max(1.0, float(np.linalg.norm(static_matrix)))
            if not np.isfinite(scale):
                raise UnsupportedBackendError(
                    "frame='interaction_exact' requires a finite static Hamiltonian."
                )
            hermitian_residual = float(np.linalg.norm(static_matrix - adjoint))
            hermitian_tolerance = 64.0 * np.finfo(np.float64).eps * scale
            if hermitian_residual > hermitian_tolerance:
                raise UnsupportedBackendError(
                    "frame='interaction_exact' requires a Hermitian static Hamiltonian."
                )
            # Canonicalize only this bounded round-off residue so the LAPACK
            # eigensolver sees a Hermitian matrix.  The residual is many orders
            # below the native integration tolerance for ordinary inputs.
            hermitian = (
                static_matrix
                if hermitian_residual == 0.0
                else 0.5 * (static_matrix + adjoint)
            )
            energies_real, interaction_basis = np.linalg.eigh(hermitian)
            basis_residual = float(
                np.linalg.norm(
                    interaction_basis.conj().T @ interaction_basis
                    - np.eye(dimension, dtype=np.complex128)
                )
            )
            eigen_residual = float(
                np.linalg.norm(
                    static_matrix @ interaction_basis
                    - interaction_basis * energies_real[np.newaxis, :]
                )
                / scale
            )
            if basis_residual > 1e-10 or eigen_residual > 1e-10:
                raise UnsupportedBackendError(
                    "frame='interaction_exact' eigendecomposition failed its numerical residual check."
                )
            energies = np.asarray(energies_real, dtype=np.complex128)
            transformed_controls = []
            for operator in drive_operators:
                matrix = _qobj_matrix(operator)
                transformed_controls.append(
                    np.asarray(
                        interaction_basis.conj().T @ matrix @ interaction_basis,
                        dtype=np.complex128,
                    )
                )
            control_csrs = []
            for matrix in transformed_controls:
                try:
                    from scipy import sparse
                except ImportError as exc:  # pragma: no cover
                    raise BackendUnavailable(
                        "scipy is required for the native interaction backend"
                    ) from exc
                csr = sparse.csr_matrix(matrix, dtype=np.complex128)
                csr.sum_duplicates()
                csr.eliminate_zeros()
                csr.sort_indices()
                control_csrs.append(
                    (
                        np.ascontiguousarray(csr.data, dtype=np.complex128),
                        np.ascontiguousarray(csr.indices, dtype=np.int64),
                        np.ascontiguousarray(csr.indptr, dtype=np.int64),
                    )
                )
        # For an already diagonal H0, a fully diagonal drive stays cheapest in
        # the lab frame.  Once H0 was diagonalized above, however, retaining
        # the basis transform is required even when the transformed drives are
        # diagonal; otherwise the native result would be returned in the
        # eigenbasis instead of the user's original basis.
        controls_are_diagonal = all(
            _diagonal_from_csr(item, int(static_operator.shape[0]))[0]
            for item in control_csrs
        )
        # ``frame='interaction_exact'`` is an explicit representation choice.
        # Only the automatic selector may choose the cheaper lab-frame
        # diagonal shortcut; an explicit request must be reflected in the
        # resolved plan and its audit statistics.
        requested_frame = str(
            getattr(self.prepared.options, "frame", "lab")
        ).lower()
        if (
            controls_are_diagonal
            and interaction_basis is None
            and requested_frame == "auto"
        ):
            return self._plan_from_qobjs(
                static_operator,
                drive_operators,
                iq,
                t_axis,
                lo_freqs,
                mode,
                metadata=metadata,
            )
        bundle = _interaction_bundle_from_components(
            control_csrs,
            int(static_operator.shape[0]),
        )
        return self._freeze_plan_arrays(_NativePlan(
            matrix_format="csr",
            static_matrix=None,
            controls=None,
            iq=np.asarray(iq, dtype=np.complex128),
            t_axis=np.asarray(t_axis, dtype=np.float64),
            lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
            mode=mode,
            sparse_kernel="interaction",
            frame="interaction_exact",
            metadata=metadata,
            fused_csr=bundle,
            interaction_energies=np.asarray(energies, dtype=np.complex128),
            interaction_basis=interaction_basis,
        ))

    def _plan_from_arrays(
        self,
        static_matrix: np.ndarray,
        controls: np.ndarray,
        iq: np.ndarray,
        t_axis: np.ndarray,
        lo_freqs: np.ndarray,
        mode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _NativePlan:
        static_matrix = np.ascontiguousarray(static_matrix, dtype=np.complex128)
        controls = np.ascontiguousarray(controls, dtype=np.complex128)
        n = int(static_matrix.shape[0])
        requested = str(getattr(self.prepared.options, "matrix_format", "auto")).lower()
        sparse_kernel = self._requested_sparse_kernel()
        if sparse_kernel == "fused" and requested in {"dense", "banded"}:
            raise UnsupportedBackendError(
                "sparse_kernel='fused' requires matrix_format='csr' or 'auto'."
            )
        need_csr = requested != "dense" and (
            requested in {"csr", "banded", "fused_csr"}
            or sparse_kernel == "fused"
            or (cpp_csr_backend_available() and n >= 8)
        )
        bundle = (
            _csr_bundle_from_arrays(
                [static_matrix] + [controls[index] for index in range(controls.shape[0])],
                n=n,
            )
            if need_csr
            else None
        )
        total_nnz = bundle[4] if bundle is not None else (1 + controls.shape[0]) * n * n
        density = float(total_nnz) / float(max(1, (1 + controls.shape[0]) * n * n))
        dense_fast_path = (
            requested == "auto"
            and sparse_kernel != "fused"
            and density > float(
                getattr(self.prepared.options, "sparse_threshold", 0.6)
            )
        )
        need_banded = (
            requested == "banded"
            or (requested == "auto" and sparse_kernel != "fused")
        ) and not dense_fast_path
        banded_bundle = None
        if bundle is not None and need_banded:
            matrices = []
            data, indices, indptr, offsets, _ = bundle
            for index in range(1 + controls.shape[0]):
                begin = int(offsets[index])
                end = int(offsets[index + 1])
                matrices.append(
                    (
                        np.ascontiguousarray(data[begin:end], dtype=np.complex128),
                        np.ascontiguousarray(indices[begin:end], dtype=np.int64),
                        np.ascontiguousarray(indptr[index], dtype=np.int64),
                    )
                )
            banded_bundle = _banded_bundle_from_components(matrices, n)
        need_fused = (
            requested == "fused_csr"
            or sparse_kernel == "fused"
            or (
                requested == "auto"
                and sparse_kernel == "auto"
                and controls.shape[0] >= 2
            )
        ) and not dense_fast_path
        fused_bundle = None
        if bundle is not None and need_fused:
            matrices = []
            data, indices, indptr, offsets, _ = bundle
            for index in range(1 + controls.shape[0]):
                begin = int(offsets[index])
                end = int(offsets[index + 1])
                matrices.append(
                    (
                        np.ascontiguousarray(data[begin:end], dtype=np.complex128),
                        np.ascontiguousarray(indices[begin:end], dtype=np.int64),
                        np.ascontiguousarray(indptr[index], dtype=np.int64),
                    )
                )
            fused_bundle = _fused_csr_bundle_from_components(matrices, n)
        band_count = None if banded_bundle is None else int(banded_bundle[0].size)
        matrix_format = self._resolve_matrix_format(
            n,
            total_nnz,
            1 + controls.shape[0],
            band_count=band_count,
        )
        union_nnz = None if fused_bundle is None else int(fused_bundle[0].size)
        use_fused = (
            matrix_format == "csr"
            and fused_bundle is not None
            and (
                sparse_kernel == "fused"
                or (
                    requested == "auto"
                    and sparse_kernel == "auto"
                    and union_nnz is not None
                    and controls.shape[0] >= 2
                    and total_nnz >= 1.5 * max(1, union_nnz)
                )
            )
        )
        if (
            requested == "auto"
            and matrix_format == "banded"
            and fused_bundle is not None
            and controls.shape[0] >= 2
            and sparse_kernel == "auto"
            and union_nnz is not None
            and total_nnz >= 1.5 * max(1, union_nnz)
        ):
            matrix_format = "csr"
            use_fused = True
        if requested == "fused_csr":
            if not cpp_fused_csr_backend_available():
                raise BackendUnavailable(
                    "The native fused CSR kernel is unavailable; rebuild the optional extension."
                )
            if fused_bundle is None:
                raise UnsupportedBackendError(
                    "fused CSR storage requires the CSR structure during preparation."
                )
            matrix_format = "csr"
            use_fused = True
        if use_fused:
            static_values, control_values, indices, indptr = fused_bundle
            return self._freeze_plan_arrays(_NativePlan(
                matrix_format="csr",
                static_matrix=None,
                controls=None,
                iq=np.asarray(iq, dtype=np.complex128),
                t_axis=np.asarray(t_axis, dtype=np.float64),
                lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
                mode=mode,
                sparse_kernel="fused",
                metadata=metadata,
                fused_csr=(static_values, control_values, indices, indptr),
            ))
        if matrix_format == "banded":
            if banded_bundle is None:
                raise UnsupportedBackendError(
                    "banded matrix storage requires the CSR structure during preparation."
                )
            offsets, h0_banded, controls_banded = banded_bundle
            return self._freeze_plan_arrays(_NativePlan(
                matrix_format="banded",
                static_matrix=None,
                controls=None,
                iq=np.asarray(iq, dtype=np.complex128),
                t_axis=np.asarray(t_axis, dtype=np.float64),
                lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
                mode=mode,
                metadata=metadata,
                band_offsets=offsets,
                h0_banded=h0_banded,
                controls_banded=controls_banded,
            ))
        if matrix_format == "csr":
            if bundle is None:
                raise BackendUnavailable("The native CSR kernel is unavailable.")
            h0_csr, controls_csr = _split_csr_bundle(bundle, controls.shape[0])
            return self._freeze_plan_arrays(_NativePlan(
                matrix_format="csr",
                static_matrix=None,
                controls=None,
                iq=np.asarray(iq, dtype=np.complex128),
                t_axis=np.asarray(t_axis, dtype=np.float64),
                lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
                mode=mode,
                metadata=metadata,
                h0_csr=h0_csr,
                controls_csr=controls_csr,
            ))
        return self._freeze_plan_arrays(_NativePlan(
            matrix_format="dense",
            static_matrix=static_matrix,
            controls=controls,
            iq=np.asarray(iq, dtype=np.complex128),
            t_axis=np.asarray(t_axis, dtype=np.float64),
            lo_freqs=np.asarray(lo_freqs, dtype=np.float64),
            mode=mode,
            metadata=metadata,
        ))

    def _build_plan(self) -> _NativePlan:
        prepared = self.prepared
        if prepared.drive_terms and int(prepared.options.coefficient_order) != 1:
            raise UnsupportedBackendError(
                "The native backend uses exact piecewise-linear trace interpolation; "
                "set coefficient_order=1 or use a QuTiP backend."
            )
        if prepared.backend == "cpp_rwa":
            return self._build_rwa_plan()

        static_operator = prepared.static_hamiltonian
        if not isinstance(static_operator, qt.Qobj):
            raise TypeError("static_hamiltonian must be a qutip.Qobj")
        if static_operator.shape[0] != static_operator.shape[1] or static_operator.issuper:
            raise UnsupportedBackendError(
                "The native backend requires a square operator Hamiltonian, not a superoperator."
            )
        operators: List[qt.Qobj] = []
        traces = []
        lo_freqs = []
        modes = []
        for term in prepared.drive_terms:
            if not isinstance(term.operator, qt.Qobj):
                raise TypeError("Every drive operator must be a qutip.Qobj.")
            if term.operator.shape != static_operator.shape:
                raise ValueError("Every drive operator must match the static Hamiltonian dimension.")
            if term.operator.issuper:
                raise UnsupportedBackendError("The native backend does not support superoperator drive terms.")
            raw_t_axis, raw_values, domain, lo_freq = _validate_trace(term.trace)
            t_axis = np.ascontiguousarray(raw_t_axis, dtype=np.float64)
            values = np.ascontiguousarray(raw_values, dtype=np.complex128)
            if not np.all(np.isfinite(t_axis)) or not np.all(np.isfinite(values)):
                raise ValueError("Native traces must contain only finite samples and times.")
            if not np.isfinite(lo_freq):
                raise ValueError("Native trace lo_freq must be finite.")
            if term.mode not in {"rf", "complex_envelope"}:
                raise ValueError("Each native drive term mode must be 'rf' or 'complex_envelope'.")
            operators.append(term.operator)
            traces.append(values)
            lo_freqs.append(lo_freq if domain == "iq_complex" else 0.0)
            modes.append(1 if term.mode == "complex_envelope" else 0)

        t_axis = np.ascontiguousarray(prepared.tlist, dtype=np.float64)
        self._require_regular_grid(t_axis)
        for term in prepared.drive_terms:
            source_t = np.asarray(term.trace.t_axis, dtype=np.float64)
            if (
                len(source_t) != len(t_axis)
                or not np.all(np.isfinite(source_t))
                or not np.array_equal(source_t, t_axis)
            ):
                raise UnsupportedBackendError(
                    "The C++ backend currently requires each trace grid to match tlist exactly."
                )
        if len(set(modes)) > 1:
            raise UnsupportedBackendError("Mixed RF and complex-envelope native terms are not supported yet.")
        iq = (
            np.ascontiguousarray(np.stack(traces, axis=0), dtype=np.complex128)
            if traces
            else np.empty((0, len(t_axis)), dtype=np.complex128)
        )
        frame = str(getattr(prepared.options, "frame", "lab")).lower()
        if frame in {"interaction_exact", "auto"}:
            try:
                return self._plan_from_interaction_qobjs(
                    static_operator,
                    operators,
                    iq,
                    t_axis,
                    np.asarray(lo_freqs, dtype=np.float64),
                    modes[0] if modes else 1,
                )
            except (BackendUnavailable, UnsupportedBackendError):
                if frame == "interaction_exact":
                    raise
            except np.linalg.LinAlgError as exc:
                if frame == "interaction_exact":
                    raise UnsupportedBackendError(
                        "frame='interaction_exact' eigendecomposition failed."
                    ) from exc
        return self._plan_from_qobjs(
            static_operator,
            operators,
            iq,
            t_axis,
            np.asarray(lo_freqs, dtype=np.float64),
            modes[0] if modes else 1,
        )

    def _build_rwa_plan(self) -> _NativePlan:
        prepared = self.prepared
        if str(getattr(prepared.options, "frame", "lab")).lower() != "lab":
            raise UnsupportedBackendError(
                "frame='interaction_exact' is a separate exact representation; "
                "it cannot be combined with the approximate cpp_rwa backend."
            )
        if len(prepared.drive_terms) != 1:
            raise UnsupportedBackendError("cpp_fast currently supports one drive term.")
        term = prepared.drive_terms[0]
        raw_t_axis, raw_values, domain, lo_freq = _validate_trace(term.trace)
        if domain != "iq_complex":
            raise UnsupportedBackendError(
                "cpp_fast requires an iq_complex trace so the carrier can be removed analytically."
            )
        if term.mode != "rf":
            raise UnsupportedBackendError("cpp_fast expects mode='rf' and performs its own RWA transform.")
        if (
            prepared.static_hamiltonian.shape[0] != prepared.static_hamiltonian.shape[1]
            or prepared.static_hamiltonian.issuper
        ):
            raise UnsupportedBackendError(
                "cpp_fast requires a square operator Hamiltonian, not a superoperator."
            )
        if term.operator.shape != prepared.static_hamiltonian.shape:
            raise ValueError("Every drive operator must match the static Hamiltonian dimension.")
        static_matrix = _qobj_matrix(prepared.static_hamiltonian)
        n = static_matrix.shape[0]
        hermitian_static = 0.5 * (static_matrix + static_matrix.conj().T)
        static_scale = max(1.0, float(np.linalg.norm(static_matrix)))
        if np.linalg.norm(static_matrix - hermitian_static) > 1e-10 * static_scale:
            raise UnsupportedBackendError("cpp_fast requires a Hermitian static Hamiltonian.")
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian_static)
        requested_levels = prepared.options.extra.get("active_levels")
        if requested_levels is None:
            active_levels = n
        else:
            active_levels = int(requested_levels)
            if active_levels < 2 or active_levels > n:
                raise ValueError("active_levels must be between 2 and the Hamiltonian dimension.")
        active_vectors = eigenvectors[:, :active_levels]
        active_values = eigenvalues[:active_levels]
        drive_matrix = _qobj_matrix(term.operator)
        if term.operator.issuper:
            raise UnsupportedBackendError("cpp_fast does not support superoperator drive terms.")
        drive_matrix = active_vectors.conj().T @ drive_matrix @ active_vectors
        drive_scale = max(1e-30, float(np.linalg.norm(drive_matrix)))
        diagonal_drive = np.diag(np.diag(drive_matrix))
        lower = np.tril(drive_matrix, -1)
        upper = np.triu(drive_matrix, 1)
        nearest = np.zeros_like(drive_matrix)
        for row in range(active_levels):
            for column in range(active_levels):
                if abs(row - column) == 1:
                    nearest[row, column] = drive_matrix[row, column]
        discarded_ratio = float(np.linalg.norm(drive_matrix - nearest) / drive_scale)
        max_discarded = float(prepared.options.extra.get("rwa_max_discarded_ratio", 0.35))
        if not np.isfinite(max_discarded) or max_discarded < 0.0:
            raise ValueError("rwa_max_discarded_ratio must be a finite non-negative number.")
        if discarded_ratio > max_discarded:
            raise UnsupportedBackendError(
                "cpp_fast discarded drive terms exceed rwa_max_discarded_ratio "
                "({:.3g} > {:.3g}).".format(discarded_ratio, max_discarded)
            )
        if np.linalg.norm(diagonal_drive) > max_discarded * drive_scale:
            raise UnsupportedBackendError("cpp_fast does not support a large diagonal drive component.")
        if not np.isfinite(lo_freq) or abs(lo_freq) <= 0.0:
            raise UnsupportedBackendError("cpp_fast requires a finite, non-zero LO frequency.")
        t_axis = np.ascontiguousarray(raw_t_axis, dtype=np.float64)
        values = np.ascontiguousarray(raw_values, dtype=np.complex128)
        prepared_t = np.ascontiguousarray(prepared.tlist, dtype=np.float64)
        self._require_regular_grid(prepared_t)
        if len(t_axis) != len(prepared_t) or not np.array_equal(t_axis, prepared_t):
            raise UnsupportedBackendError(
                "The C++ backend currently requires the RWA trace grid to match tlist exactly."
            )
        controls = np.ascontiguousarray(
            np.stack([0.5 * upper, 0.5 * lower], axis=0),
            dtype=np.complex128,
        )
        iq = np.ascontiguousarray(np.stack([values, np.conj(values)], axis=0), dtype=np.complex128)
        h_rot = np.diag(np.asarray(active_values, dtype=np.complex128))
        frame_omega = 2.0 * np.pi * lo_freq
        for level in range(active_levels):
            h_rot[level, level] -= level * frame_omega
        metadata = {
            "frame_frequency": lo_freq,
            "frame_omega": frame_omega,
            "t0": float(t_axis[0]),
            "discarded_ratio": discarded_ratio,
            "basis_transform": active_vectors,
            "full_basis": eigenvectors,
            "active_levels": active_levels,
            "full_dimension": n,
        }
        return self._plan_from_arrays(
            h_rot,
            controls,
            iq,
            t_axis,
            np.zeros(2, dtype=np.float64),
            1,
            metadata,
        )

    def _payload(self, states: Sequence[qt.Qobj]) -> _NativePayload:
        if self._plan is None:
            self._plan = self._build_plan()
        plan = self._plan
        if plan.matrix_format == "csr":
            if plan.sparse_kernel in {"fused", "interaction"}:
                assert plan.fused_csr is not None
                native_dimension = int(plan.fused_csr[3].size - 1)
            else:
                assert plan.h0_csr is not None
                native_dimension = int(plan.h0_csr[2].size - 1)
        elif plan.matrix_format == "banded":
            assert plan.h0_banded is not None
            native_dimension = int(plan.h0_banded.shape[1])
        else:
            assert plan.static_matrix is not None
            native_dimension = int(plan.static_matrix.shape[0])
        metadata = plan.metadata
        full_dimension = int(metadata.get("full_dimension", native_dimension)) if metadata else native_dimension
        initial = []
        dims = []
        normalize_columns = []
        normalize_output = self.prepared.options.extra.get("normalize_output", True) is not False
        try:
            normalization_atol = float(qt.settings.core["atol"])
        except (AttributeError, KeyError, TypeError, ValueError):
            normalization_atol = 1e-12
        for state in states:
            if not isinstance(state, qt.Qobj) or not state.isket:
                raise UnsupportedBackendError("The native backend currently supports ket initial states only.")
            vector = _qobj_matrix(state)
            if vector.shape != (full_dimension, 1):
                raise ValueError("Initial state dimension does not match the Hamiltonian.")
            if metadata is not None:
                active_vectors = metadata["basis_transform"]
                full_basis = metadata["full_basis"]
                active_levels = int(metadata["active_levels"])
                projected = np.asarray(active_vectors.conj().T @ vector[:, 0], dtype=np.complex128)
                omitted = np.linalg.norm(full_basis[:, active_levels:].conj().T @ vector[:, 0])
                active_levels_tol = float(self.prepared.options.extra.get("active_levels_tol", 1e-8))
                if omitted > active_levels_tol:
                    raise UnsupportedBackendError(
                        "initial state has support outside active_levels; increase active_levels."
                    )
                vector_data = np.array(projected, dtype=np.complex128, copy=True)
                frame_omega = float(metadata["frame_omega"])
                t0 = float(metadata["t0"])
                for level in range(active_levels):
                    vector_data[level] *= np.exp(1j * level * frame_omega * t0)
            elif plan.interaction_basis is not None:
                # The native interaction kernel evolves coordinates in the
                # exact eigenbasis of H0.  This is a unitary coordinate change,
                # not a level truncation or other physical approximation.
                vector_data = np.asarray(
                    plan.interaction_basis.conj().T @ vector[:, 0],
                    dtype=np.complex128,
                )
            else:
                vector_data = np.array(vector[:, 0], dtype=np.complex128, copy=True)
            initial.append(vector_data)
            dims.append(state.dims)
            normalize_columns.append(
                normalize_output
                and abs(float(np.linalg.norm(vector_data)) - 1.0) <= normalization_atol
            )
        return _NativePayload(
            plan=plan,
            initial_matrix=np.ascontiguousarray(np.column_stack(initial), dtype=np.complex128),
            dims=dims,
            normalize_columns=np.asarray(normalize_columns, dtype=bool),
        )

    @staticmethod
    def _apply_frame(vector: np.ndarray, time: float, metadata):
        if metadata is None:
            return vector
        result = np.array(vector, dtype=np.complex128, copy=True)
        omega = float(metadata["frame_omega"])
        for level in range(result.shape[0]):
            result[level, ...] *= np.exp(-1j * level * omega * float(time))
        basis_transform = metadata.get("basis_transform")
        if basis_transform is not None:
            result = np.asarray(basis_transform @ result, dtype=np.complex128)
        return result

    @staticmethod
    def _apply_interaction_basis(vector: np.ndarray, basis: Optional[np.ndarray]):
        """Transform native interaction-picture coordinates to user basis."""
        if basis is None:
            return vector
        return np.asarray(basis @ vector, dtype=np.complex128)

    def _normalize_output(self, values: np.ndarray, normalize_columns=None) -> np.ndarray:
        if self.prepared.options.extra.get("normalize_output", True) is False:
            return values
        result = np.array(values, dtype=np.complex128, copy=True)
        if result.ndim == 2:
            norms = np.sqrt(np.sum(np.abs(result) ** 2, axis=0))
            for column, norm in enumerate(norms):
                should_normalize = (
                    True
                    if normalize_columns is None
                    else bool(np.asarray(normalize_columns, dtype=bool)[column])
                )
                if should_normalize and norm > 0:
                    result[:, column] /= norm
        return result

    def _call_native(self, payload: _NativePayload):
        plan = payload.plan
        common = {
            "mode": int(plan.mode),
            "atol": float(self.prepared.options.atol),
            "rtol": float(self.prepared.options.rtol),
            "max_steps": int(self.prepared.options.extra.get("native_max_steps", 2_000_000)),
            "store_trajectory": bool(self.prepared.options.store_states),
        }
        if plan.frame == "interaction_exact":
            if _native_propagate_interaction_csr is None or plan.fused_csr is None:
                raise BackendUnavailable(
                    "The native interaction-picture kernel is unavailable."
                )
            if plan.interaction_energies is None:
                raise UnsupportedBackendError(
                    "interaction-picture plan is missing diagonal energies."
                )
            _, control_values, indices, indptr = plan.fused_csr
            return _native_propagate_interaction_csr(
                plan.interaction_energies,
                control_values,
                indices,
                indptr,
                plan.iq,
                plan.t_axis,
                payload.initial_matrix,
                plan.lo_freqs,
                **common,
            )
        if plan.sparse_kernel == "fused":
            if _native_propagate_fused_csr is None or plan.fused_csr is None:
                raise BackendUnavailable("The native fused CSR kernel is unavailable.")
            static_values, control_values, indices, indptr = plan.fused_csr
            return _native_propagate_fused_csr(
                static_values,
                indices,
                indptr,
                control_values,
                plan.iq,
                plan.t_axis,
                payload.initial_matrix,
                plan.lo_freqs,
                **common,
            )
        if plan.matrix_format == "banded":
            if (
                _native_propagate_banded is None
                or plan.band_offsets is None
                or plan.h0_banded is None
                or plan.controls_banded is None
            ):
                raise BackendUnavailable("The native banded kernel is unavailable.")
            return _native_propagate_banded(
                plan.h0_banded,
                plan.controls_banded,
                plan.band_offsets,
                plan.iq,
                plan.t_axis,
                payload.initial_matrix,
                plan.lo_freqs,
                **common,
            )
        if plan.matrix_format == "csr":
            if _native_propagate_csr is None or plan.h0_csr is None or plan.controls_csr is None:
                raise BackendUnavailable("The native CSR kernel is unavailable.")
            h0_data, h0_indices, h0_indptr = plan.h0_csr
            controls_data, controls_indices, controls_indptr, controls_offsets = plan.controls_csr
            return _native_propagate_csr(
                h0_data,
                h0_indices,
                h0_indptr,
                controls_data,
                controls_indices,
                controls_indptr,
                controls_offsets,
                plan.iq,
                plan.t_axis,
                payload.initial_matrix,
                plan.lo_freqs,
                **common,
            )
        assert plan.static_matrix is not None and plan.controls is not None
        return _native_propagate(
            plan.static_matrix,
            plan.controls,
            plan.iq,
            plan.t_axis,
            payload.initial_matrix,
            plan.lo_freqs,
            **common,
        )

    def _resolved_stats(self, stats, plan: _NativePlan) -> Dict[str, Any]:
        resolved = dict(stats or {})
        resolved["matrix_format"] = plan.matrix_format
        resolved["sparse_kernel"] = plan.sparse_kernel
        resolved["frame"] = plan.frame
        if plan.metadata is not None:
            resolved.update(
                {
                    "backend": "cpp_rwa",
                    "approximation": "rwa",
                    "rwa_discarded_ratio": plan.metadata["discarded_ratio"],
                    "frame_frequency": plan.metadata["frame_frequency"],
                    "active_levels": plan.metadata["active_levels"],
                }
            )
        else:
            resolved["backend"] = "cpp"
            resolved["approximation"] = "none"
        return resolved

    def propagate(self, initial_state: qt.Qobj) -> NativePropagationResult:
        payload = self._payload([initial_state])
        final, trajectory, stats = self._call_native(payload)
        n = payload.initial_matrix.shape[0]
        final = _decode_complex_payload(final, (n, 1))
        final = self._apply_interaction_basis(final, payload.plan.interaction_basis)
        final = self._apply_frame(final, payload.plan.t_axis[-1], payload.plan.metadata)
        final = self._normalize_output(final, payload.normalize_columns)
        state = qt.Qobj(final[:, 0], dims=payload.dims[0])
        states = []
        trajectory_array = _decode_complex_payload(
            trajectory,
            (len(payload.plan.t_axis), n, 1),
        )
        if trajectory_array is not None:
            if payload.plan.interaction_basis is not None:
                trajectory_array = np.einsum(
                    "ij,tjb->tib",
                    payload.plan.interaction_basis,
                    trajectory_array,
                    optimize=True,
                )
            if payload.plan.metadata is not None:
                trajectory_array = np.stack(
                    [self._apply_frame(item, time, payload.plan.metadata)
                     for item, time in zip(trajectory_array, payload.plan.t_axis)],
                    axis=0,
                )
            trajectory_array = np.stack(
                [self._normalize_output(item, payload.normalize_columns) for item in trajectory_array],
                axis=0,
            )
            states = [qt.Qobj(item[:, 0], dims=payload.dims[0]) for item in trajectory_array]
        resolved_stats = self._resolved_stats(stats, payload.plan)
        return NativePropagationResult(
            final_state=state,
            states=states,
            times=np.array(payload.plan.t_axis, copy=True),
            stats=resolved_stats,
        )

    def propagate_batch(self, initial_states: Sequence[qt.Qobj]) -> BatchPropagationResult:
        states = list(initial_states)
        if not states:
            return BatchPropagationResult([], [], np.array(self.prepared.tlist, copy=True), {})
        payload = self._payload(states)
        final, trajectory, stats = self._call_native(payload)
        n, batch_count = payload.initial_matrix.shape
        final = _decode_complex_payload(final, (n, batch_count))
        final = self._apply_interaction_basis(final, payload.plan.interaction_basis)
        final = self._apply_frame(final, payload.plan.t_axis[-1], payload.plan.metadata)
        final = self._normalize_output(final, payload.normalize_columns)
        final_states = [
            qt.Qobj(final[:, index], dims=payload.dims[index])
            for index in range(batch_count)
        ]
        trajectory_array = _decode_complex_payload(
            trajectory,
            (len(payload.plan.t_axis), n, batch_count),
        )
        if trajectory_array is not None:
            if payload.plan.interaction_basis is not None:
                trajectory_array = np.einsum(
                    "ij,tjb->tib",
                    payload.plan.interaction_basis,
                    trajectory_array,
                    optimize=True,
                )
            if payload.plan.metadata is not None:
                trajectory_array = np.stack(
                    [self._apply_frame(item, time, payload.plan.metadata)
                     for item, time in zip(trajectory_array, payload.plan.t_axis)],
                    axis=0,
                )
            trajectory_array = np.stack(
                [self._normalize_output(item, payload.normalize_columns) for item in trajectory_array],
                axis=0,
            )
        resolved_stats = self._resolved_stats(stats, payload.plan)
        per_state_results = []
        for index, state in enumerate(final_states):
            per_state_results.append(
                NativePropagationResult(
                    final_state=state,
                    states=(
                        [
                            qt.Qobj(item[:, index], dims=payload.dims[index])
                            for item in trajectory_array
                        ]
                        if trajectory_array is not None
                        else []
                    ),
                    times=np.array(payload.plan.t_axis, copy=True),
                    stats=dict(resolved_stats),
                )
            )
        return BatchPropagationResult(
            final_states=final_states,
            results=per_state_results,
            times=np.array(payload.plan.t_axis, copy=True),
            stats=resolved_stats,
        )


__all__ = [
    "CppPropagationBackend",
    "NativePropagationResult",
    "cpp_backend_available",
    "cpp_banded_backend_available",
    "cpp_csr_backend_available",
    "cpp_fused_csr_backend_available",
    "cpp_interaction_backend_available",
]
