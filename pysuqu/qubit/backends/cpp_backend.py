"""Python adapter for the optional C++ propagation kernel.

The extension is deliberately optional.  Installing the normal ``pysuqu``
package continues to work without a compiler; users who request ``backend='cpp'``
receive a precise error when the extension has not been built.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import OrderedDict
import hashlib
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import qutip as qt

from ..propagation import (
    BackendUnavailable,
    BatchPropagationResult,
    DynamicCollapseRate,
    DriveTerm,
    PreparedPropagation,
    PropagationOptions,
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

try:
    from ..._native import propagate_lindblad_csr as _native_propagate_lindblad_csr
except (ImportError, ModuleNotFoundError, OSError):
    _native_propagate_lindblad_csr = None


# Prepared plans are immutable after construction.  A small process-local LRU
# therefore lets repeated gate/fidelity calls share the expensive Qobj -> CSR,
# band, union-pattern, and interaction-basis preparation work.  The cache is
# deliberately kept in this adapter (rather than the extension) so importing
# pysuqu remains safe when the optional native module is absent.
_PLAN_CACHE: "OrderedDict[bytes, _NativePlan]" = OrderedDict()
_PLAN_CACHE_LOCK = threading.RLock()


def clear_native_plan_cache() -> None:
    """Drop all process-local immutable native propagation plans."""
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE.clear()


def native_plan_cache_info() -> Dict[str, int]:
    """Return a snapshot of the native plan cache size and entry count."""
    with _PLAN_CACHE_LOCK:
        return {"entries": len(_PLAN_CACHE)}


def _hash_array(hasher, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes(order="C"))


def _hash_qobj(hasher, value: qt.Qobj) -> None:
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    # Hash the canonical sparse representation.  This makes equivalent Qobj
    # storage formats share one plan while preserving every nonzero exactly.
    data, indices, indptr = _qobj_csr(value)
    _hash_array(hasher, data)
    _hash_array(hasher, indices)
    _hash_array(hasher, indptr)


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


def cpp_lindblad_backend_available() -> bool:
    """Return whether the native Lindblad wrapper can delegate to C++."""
    return callable(_native_propagate_lindblad_csr) or cpp_backend_available()


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
    if data is None:
        source = value.full()
    elif callable(as_scipy):
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


def _qobj_from_sparse(matrix) -> qt.Qobj:
    """Construct a Qobj from SciPy sparse data across QuTiP/stub versions."""
    try:
        return qt.Qobj(matrix)
    except (TypeError, ValueError):
        toarray = getattr(matrix, "toarray", None)
        if not callable(toarray):
            raise
        return qt.Qobj(np.asarray(toarray(), dtype=np.complex128))


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
    # Row-major integer keys let NumPy build and map the sorted union without
    # per-nonzero Python tuples.  Keep a pair-key fallback for int64 overflow.
    pair_keys = n > math.isqrt(np.iinfo(np.int64).max)
    key_dtype = [("row", np.int64), ("column", np.int64)] if pair_keys else np.int64
    keys = np.empty(sum(len(data) for data, _, _ in matrices), dtype=key_dtype)
    row_numbers = np.arange(n, dtype=np.int64)
    offset = 0
    for data, matrix_indices, indptr in matrices:
        end = offset + len(data)
        rows = np.repeat(row_numbers, np.diff(indptr))
        if pair_keys:
            keys["row"][offset:end] = rows
            keys["column"][offset:end] = matrix_indices
        else:
            keys[offset:end] = rows * n + matrix_indices
        offset = end
    union, positions = np.unique(keys, return_inverse=True)
    if pair_keys:
        rows, indices = union["row"], union["column"]
    elif n:
        rows, indices = np.divmod(union, n)
    else:
        rows = indices = np.empty(0, dtype=np.int64)
    row_ptr = np.empty(n + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(np.bincount(rows, minlength=n), out=row_ptr[1:])
    static_values = np.zeros(len(union), dtype=np.complex128)
    control_values = np.zeros((max(0, len(matrices) - 1), len(union)), dtype=np.complex128)
    offset = 0
    for operator_index, (data, _, _) in enumerate(matrices):
        target = static_values if operator_index == 0 else control_values[operator_index - 1]
        end = offset + len(data)
        # add.at retains input-order accumulation, including duplicate entries
        # in noncanonical CSR input, rather than changing cancellation order.
        np.add.at(target, positions[offset:end], data)
        offset = end
    return (
        np.ascontiguousarray(static_values, dtype=np.complex128),
        np.ascontiguousarray(control_values, dtype=np.complex128),
        np.ascontiguousarray(indices, dtype=np.int64),
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


def _connected_components(
    static_operator: qt.Qobj,
    drive_operators: Sequence[qt.Qobj],
) -> List[np.ndarray]:
    """Find exact invariant coordinate blocks in the operator union.

    An edge is present whenever any operator contains a mathematically
    nonzero off-diagonal entry.  The graph is treated as undirected because a
    one-way matrix entry still couples the corresponding invariant subspaces.
    No numerical threshold is used: tiny entries remain physical couplings.
    """
    n = int(static_operator.shape[0])
    parent = list(range(n))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for operator in (static_operator, *drive_operators):
        _data, indices, indptr = _qobj_csr(operator)
        for row in range(n):
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            for entry in range(begin, end):
                column = int(indices[entry])
                if column != row:
                    union(row, column)
    groups: Dict[int, List[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _conjugate_trace(trace: Any) -> Any:
    """Return a trace with conjugated samples while preserving its metadata."""
    values = np.conj(np.asarray(getattr(trace, "values"), dtype=np.complex128))
    clone = getattr(trace, "clone", None)
    if callable(clone):
        return clone(values=values)
    try:
        from ...funclib.transmission import SignalTrace

        return SignalTrace(
            t_axis=np.asarray(trace.t_axis, dtype=np.float64),
            values=values,
            sample_rate=float(trace.sample_rate),
            domain=trace.domain,
            plane=trace.plane,
            lo_freq=float(getattr(trace, "lo_freq", 0.0) or 0.0),
            label=str(getattr(trace, "label", "signal")),
            metadata=dict(getattr(trace, "metadata", {}) or {}),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise UnsupportedBackendError(
            "native Lindblad propagation requires trace objects that can be conjugated"
        ) from exc


def _slice_operator(operator: qt.Qobj, indices: np.ndarray) -> qt.Qobj:
    try:
        from scipy import sparse

        data, columns, indptr = _qobj_csr(operator)
        matrix = sparse.csr_matrix(
            (data, columns, indptr),
            shape=operator.shape,
        )
        sliced = matrix[indices, :][:, indices].tocsr()
        sliced.sum_duplicates()
        sliced.eliminate_zeros()
        sliced.sort_indices()
        return _qobj_from_sparse(sliced)
    except (BackendUnavailable, TypeError, ValueError):
        matrix = _qobj_matrix(operator)
        sliced = matrix[np.ix_(indices, indices)]
        return qt.Qobj(np.ascontiguousarray(sliced, dtype=np.complex128))


def _transform_block_diagonal_basis(
    operator: qt.Qobj,
    basis: np.ndarray,
    components: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute B^H A B using block structure without dense n^3 products."""
    try:
        from scipy import sparse

        data, columns, indptr = _qobj_csr(operator)
        source = sparse.csr_matrix((data, columns, indptr), shape=operator.shape)
        result = np.zeros(operator.shape, dtype=np.complex128)
        for column_indices in components:
            block_basis = basis[np.ix_(column_indices, column_indices)]
            intermediate = source[:, column_indices].dot(block_basis)
            for row_indices in components:
                row_basis = basis[np.ix_(row_indices, row_indices)]
                result[np.ix_(row_indices, column_indices)] = (
                    row_basis.conj().T @ intermediate[row_indices, :]
                )
        return np.ascontiguousarray(result, dtype=np.complex128)
    except (BackendUnavailable, TypeError, ValueError):
        matrix = _qobj_matrix(operator)
        return np.ascontiguousarray(basis.conj().T @ matrix @ basis, dtype=np.complex128)


def _static_collapse_operator(value: Any, dimension: int) -> qt.Qobj:
    """Normalize one static collapse-operator specification.

    QuTiP also accepts ``[operator, coefficient]`` entries.  A numeric
    coefficient is folded into the operator exactly; callable/string
    coefficients are rejected because evaluating them inside the native
    GIL-free loop would change the callback semantics.
    """
    if isinstance(value, DynamicCollapseRate):
        value = value.operator
    coefficient = 1.0
    operator = value
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise UnsupportedBackendError(
                "native Lindblad propagation expects Qobj collapse operators "
                "or [Qobj, numeric] pairs"
            )
        operator, coefficient = value
        if not np.isscalar(coefficient) or isinstance(coefficient, (str, bytes)):
            raise UnsupportedBackendError(
                "time-dependent collapse-operator coefficients require a QuTiP backend"
            )
    if not isinstance(operator, qt.Qobj):
        raise TypeError("collapse operators must be qutip.Qobj instances")
    if getattr(operator, "issuper", False) or operator.shape != (dimension, dimension):
        raise UnsupportedBackendError(
            "native Lindblad propagation requires square operator collapse operators"
        )
    try:
        coefficient = complex(coefficient)
    except (TypeError, ValueError) as exc:
        raise UnsupportedBackendError("collapse-operator coefficient must be numeric") from exc
    if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
        raise ValueError("collapse-operator coefficients must be finite")
    if coefficient == 1.0:
        return operator
    return coefficient * operator


def _lindblad_effective_operators(
    static_hamiltonian: qt.Qobj,
    drive_terms: Sequence[DriveTerm],
    c_ops: Sequence[Any],
) -> Tuple[qt.Qobj, List[DriveTerm]]:
    """Construct the exact ordinary-operator representation of a Lindbladian.

    For column-major ``vec(rho)``, ``d vec(rho) / dt = L vec(rho)``.  The
    native ket kernel solves ``d y / dt = -i H_native y``, so we pass
    ``H_native = i L``.  This is a representation change only; no rotating
    wave, level, or magnitude approximation is introduced.
    """
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - scipy is a package dependency
        raise BackendUnavailable("scipy is required for native Lindblad propagation") from exc

    dimension = int(static_hamiltonian.shape[0])
    if (
        static_hamiltonian.shape != (dimension, dimension)
        or getattr(static_hamiltonian, "issuper", False)
    ):
        raise UnsupportedBackendError(
            "native Lindblad propagation requires a square operator Hamiltonian"
        )
    def scipy_csr(operator: qt.Qobj):
        data, columns, indptr = _qobj_csr(operator)
        return sparse.csr_matrix(
            (data, columns, indptr),
            shape=operator.shape,
            dtype=np.complex128,
        )

    static_matrix = scipy_csr(static_hamiltonian)
    identity = sparse.identity(dimension, format="csr", dtype=np.complex128)

    def hamiltonian_superoperator(operator: qt.Qobj):
        matrix = scipy_csr(operator)
        return sparse.kron(identity, matrix, format="csr") - sparse.kron(
            matrix.conjugate(), identity, format="csr"
        )

    effective_static = hamiltonian_superoperator(static_hamiltonian)
    for raw_operator in c_ops:
        operator = _static_collapse_operator(raw_operator, dimension)
        collapse = scipy_csr(operator)
        dagger_product = collapse.getH().dot(collapse).tocsr()
        dissipator = sparse.kron(collapse.conjugate(), collapse, format="csr")
        dissipator = dissipator - 0.5 * sparse.kron(
            identity, dagger_product, format="csr"
        )
        dissipator = dissipator - 0.5 * sparse.kron(
            dagger_product.transpose(), identity, format="csr"
        )
        effective_static = effective_static + 1j * dissipator
    effective_static = effective_static.tocsr()
    effective_static.sum_duplicates()
    effective_static.eliminate_zeros()
    effective_static.sort_indices()

    effective_drives: List[DriveTerm] = []
    for term in drive_terms:
        if term.mode not in {"rf", "complex_envelope"}:
            raise UnsupportedBackendError(
                "native Lindblad propagation supports rf and complex_envelope drive modes"
            )
        operator_matrix = scipy_csr(term.operator)
        if term.mode == "complex_envelope":
            # H_native = i L contains f(t) (I kron A) and
            # -conj(f(t)) (conj(A) kron I).  Keeping these as two controls
            # preserves the exact adjoint on the right side for complex
            # coefficients; collapsing them into one control would silently
            # change the equation whenever f has an imaginary part.
            left = sparse.kron(identity, operator_matrix, format="csr")
            right = -sparse.kron(operator_matrix.conjugate(), identity, format="csr")
            for matrix, trace in (
                (left, term.trace),
                (right, _conjugate_trace(term.trace)),
            ):
                matrix = matrix.tocsr()
                matrix.sum_duplicates()
                matrix.eliminate_zeros()
                matrix.sort_indices()
                effective_drives.append(
                    DriveTerm(_qobj_from_sparse(matrix), trace, mode="complex_envelope")
                )
        else:
            effective = hamiltonian_superoperator(term.operator).tocsr()
            effective.sum_duplicates()
            effective.eliminate_zeros()
            effective.sort_indices()
            effective_drives.append(
                DriveTerm(_qobj_from_sparse(effective), term.trace, mode=term.mode)
            )
    return _qobj_from_sparse(effective_static), effective_drives


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
    # Optional coefficients in ascending powers of the normalized interval
    # coordinate u=(t-t_i)/(t_{i+1}-t_i).  The native kernel falls back to its
    # historical linear interpolation when this payload is absent.
    iq_polynomial: Optional[np.ndarray] = None
    coefficient_order: int = 1
    output_t_axis: Optional[np.ndarray] = None
    output_indices: Optional[np.ndarray] = None
    control_modes: Optional[np.ndarray] = None


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


def _native_lindblad_csr_components(prepared):
    """Pack the physical Hamiltonian/collapse operators for the matrix-free ABI."""
    n = int(prepared.static_hamiltonian.shape[0])
    hamiltonian_components = [_qobj_csr(prepared.static_hamiltonian)]
    hamiltonian_components.extend(_qobj_csr(term.operator) for term in prepared.drive_terms)
    hamiltonian_bundle = _csr_bundle_from_components(hamiltonian_components, n=n)
    h0, controls = _split_csr_bundle(hamiltonian_bundle, len(prepared.drive_terms))
    collapse_components = []
    for raw_operator in prepared.c_ops:
        collapse_components.append(
            _qobj_csr(_static_collapse_operator(raw_operator, n))
        )
    if not collapse_components:
        raise UnsupportedBackendError(
            "matrix-free native Lindblad propagation requires at least one collapse operator"
        )
    collapse_bundle = _csr_bundle_from_components(collapse_components, n=n)
    return h0, controls, collapse_bundle


def _native_lindblad_rate_components(prepared):
    """Pack descriptor-backed scalar rates on the prepared solver grid.

    ``None`` denotes an all-static collapse list.  Descriptor traces must be
    real, finite, non-negative and sampled exactly on ``prepared.tlist``;
    rejecting mismatches avoids silently changing callback semantics.
    """
    if not any(isinstance(item, DynamicCollapseRate) for item in prepared.c_ops):
        return None
    tlist = np.asarray(prepared.tlist, dtype=np.float64)
    rates = np.empty((len(prepared.c_ops), len(tlist)), dtype=np.float64)
    for index, item in enumerate(prepared.c_ops):
        if not isinstance(item, DynamicCollapseRate):
            rates[index] = 1.0
            continue
        trace_t = np.asarray(getattr(item.rate_trace, "t_axis", None), dtype=np.float64)
        trace_values = np.asarray(getattr(item.rate_trace, "values", None))
        if trace_t.shape != tlist.shape or not np.array_equal(trace_t, tlist):
            raise UnsupportedBackendError(
                "native dynamic collapse rate traces must use the prepared solver grid"
            )
        if np.iscomplexobj(trace_values):
            if np.any(np.abs(np.imag(trace_values)) > 0.0):
                raise UnsupportedBackendError("native collapse rates must be real-valued")
            trace_values = np.real(trace_values)
        values = np.asarray(trace_values, dtype=np.float64)
        if values.shape != tlist.shape or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("native collapse rate traces must be finite and non-negative")
        rates[index] = values
    rates.setflags(write=False)
    return rates


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


def _native_polynomial_coefficients(
    trace: Any,
    order: int,
) -> Tuple[np.ndarray, int]:
    """Build QuTiP-compatible piecewise polynomial coefficients.

    Coefficients are stored in ascending powers of the normalized interval
    coordinate.  Orders 0 and 1 are constructed directly; higher orders use
    SciPy's public ``make_interp_spline`` exactly as QuTiP's array coefficient
    does, then convert derivatives to the native normalized basis.
    """
    t_axis, values, _domain, _lo_freq = _validate_trace(trace)
    requested = int(order)
    effective = min(requested, max(0, len(t_axis) - 1))
    intervals = max(0, len(t_axis) - 1)
    complex_values = np.asarray(values, dtype=np.complex128)
    if intervals == 0:
        return np.empty((0, effective + 1), dtype=np.complex128), effective
    if effective == 0:
        coefficients = complex_values[:-1, np.newaxis]
        return np.ascontiguousarray(coefficients, dtype=np.complex128), effective
    if effective == 1:
        coefficients = np.empty((intervals, 2), dtype=np.complex128)
        coefficients[:, 0] = complex_values[:-1]
        coefficients[:, 1] = np.diff(complex_values)
        return np.ascontiguousarray(coefficients), effective
    try:
        from scipy.interpolate import make_interp_spline
    except ImportError as exc:  # pragma: no cover - scipy is a package dependency
        raise BackendUnavailable("scipy is required for native spline coefficients") from exc
    spline = make_interp_spline(t_axis, complex_values, k=effective, bc_type=None)
    coefficients = np.empty((intervals, effective + 1), dtype=np.complex128)
    for interval in range(intervals):
        dt = float(t_axis[interval + 1] - t_axis[interval])
        start = float(t_axis[interval])
        for power in range(effective + 1):
            derivative = np.asarray(spline(start, nu=power), dtype=np.complex128)
            coefficients[interval, power] = (
                derivative / float(math.factorial(power)) * (dt ** power)
            )
    return np.ascontiguousarray(coefficients), effective


def _prepare_native_trace_payloads(
    traces: Sequence[Any],
    output_t_axis: np.ndarray,
    order: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, np.ndarray]:
    """Prepare a common exact source grid for native interpolation.

    QuTiP's spline builder may introduce interior knots that are not present in
    the user's output grid (notably for quadratic splines).  The native kernel
    uses one source grid, so we form the exact union of all spline knots and
    requested output times, evaluate each trace there, and return indices for
    restoring the original output trajectory.
    """
    output_t_axis = np.ascontiguousarray(np.asarray(output_t_axis, dtype=np.float64))
    requested = int(order)
    if not traces:
        return (
            output_t_axis,
            np.empty((0, len(output_t_axis)), dtype=np.complex128),
            None,
            requested,
            np.arange(len(output_t_axis), dtype=np.int64),
        )
    validated = []
    knots = [output_t_axis]
    spline_objects = []
    effective_orders = []
    for trace in traces:
        t_axis, values, _domain, _lo_freq = _validate_trace(trace)
        if output_t_axis[0] < t_axis[0] or output_t_axis[-1] > t_axis[-1]:
            raise UnsupportedBackendError(
                "native trace interpolation requires tlist to lie within each trace domain"
            )
        complex_values = np.asarray(values, dtype=np.complex128)
        effective = min(requested, max(0, len(t_axis) - 1))
        spline = None
        if effective >= 2:
            try:
                from scipy.interpolate import make_interp_spline
            except ImportError as exc:  # pragma: no cover
                raise BackendUnavailable("scipy is required for native spline coefficients") from exc
            spline = make_interp_spline(t_axis, complex_values, k=effective, bc_type=None)
            spline_knots = np.unique(np.asarray(spline.t, dtype=np.float64))
            knots.append(
                spline_knots[
                    (spline_knots >= output_t_axis[0])
                    & (spline_knots <= output_t_axis[-1])
                ]
            )
        trace_knots = np.asarray(t_axis, dtype=np.float64)
        knots.append(
            trace_knots[
                (trace_knots >= output_t_axis[0])
                & (trace_knots <= output_t_axis[-1])
            ]
        )
        validated.append((np.asarray(t_axis, dtype=np.float64), complex_values))
        spline_objects.append(spline)
        effective_orders.append(effective)
    source_t_axis = np.unique(np.concatenate(knots)).astype(np.float64, copy=False)
    source_t_axis = np.ascontiguousarray(source_t_axis, dtype=np.float64)
    if source_t_axis.size == 0 or np.any(np.diff(source_t_axis) <= 0):
        raise UnsupportedBackendError("native source interpolation grid must be strictly increasing")
    global_order = max(effective_orders) if effective_orders else requested
    source_values = np.empty(
        (len(traces), len(source_t_axis)),
        dtype=np.complex128,
    )
    polynomial = np.zeros(
        (len(traces), max(0, len(source_t_axis) - 1), global_order + 1),
        dtype=np.complex128,
    )
    for control, ((trace_t, trace_values), spline, effective) in enumerate(
        zip(validated, spline_objects, effective_orders)
    ):
        if effective == 0:
            positions = np.searchsorted(trace_t, source_t_axis, side="right") - 1
            positions = np.clip(positions, 0, len(trace_t) - 1)
            source_values[control] = trace_values[positions]
        elif effective == 1:
            source_values[control] = np.interp(
                source_t_axis,
                trace_t,
                trace_values.real,
            ) + 1j * np.interp(
                source_t_axis,
                trace_t,
                trace_values.imag,
            )
        else:
            source_values[control] = np.asarray(spline(source_t_axis), dtype=np.complex128)
        for interval in range(max(0, len(source_t_axis) - 1)):
            dt = float(source_t_axis[interval + 1] - source_t_axis[interval])
            start = float(source_t_axis[interval])
            if effective == 0:
                polynomial[control, interval, 0] = source_values[control, interval]
            elif effective == 1:
                polynomial[control, interval, 0] = source_values[control, interval]
                polynomial[control, interval, 1] = (
                    source_values[control, interval + 1]
                    - source_values[control, interval]
                )
            else:
                for power in range(effective + 1):
                    derivative = np.asarray(spline(start, nu=power), dtype=np.complex128)
                    polynomial[control, interval, power] = (
                        derivative / float(math.factorial(power)) * (dt ** power)
                    )
    output_indices = np.searchsorted(source_t_axis, output_t_axis)
    safe_indices = np.clip(output_indices, 0, max(0, len(source_t_axis) - 1))
    if np.any(output_indices >= len(source_t_axis)) or not np.array_equal(
        source_t_axis[safe_indices], output_t_axis
    ):
        raise RuntimeError("native source grid did not retain the requested output times")
    return (
        source_t_axis,
        np.ascontiguousarray(source_values, dtype=np.complex128),
        np.ascontiguousarray(polynomial, dtype=np.complex128),
        global_order,
        np.ascontiguousarray(output_indices, dtype=np.int64),
    )


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
        self._plan_cache_hit = False
        self._plan_cache_digest: Optional[bytes] = None
        self._block_backends = None

    @classmethod
    def from_prepared(cls, prepared) -> "CppPropagationBackend":
        return cls(prepared)

    def _plan_cache_key(self) -> bytes:
        """Build a content-addressed key for the immutable numerical plan.

        The key intentionally excludes initial states and solver tolerances
        that only affect integration.  It includes every input that changes
        matrix layout, frame transformation, or coefficient representation.
        """
        hasher = hashlib.blake2b(digest_size=24)
        prepared = self.prepared
        hasher.update(b"pysuqu-native-plan-v2\0")
        hasher.update(str(prepared.backend).lower().encode("utf-8"))
        options = prepared.options
        for name in (
            "matrix_format",
            "frame",
            "sparse_kernel",
            "coefficient_order",
            "block_decompose",
        ):
            hasher.update(str(getattr(options, name, None)).encode("utf-8"))
            hasher.update(b"\0")
        hasher.update(repr(float(getattr(options, "sparse_threshold", 0.6))).encode("ascii"))
        hasher.update(b"\0")
        extras = getattr(options, "extra", {}) or {}
        # These legacy options can alter the selected RWA plan.  Other extra
        # values are QuTiP-only and do not affect native matrix preparation.
        for name in ("active_levels", "rwa_max_discarded_ratio"):
            hasher.update(name.encode("ascii"))
            hasher.update(repr(extras.get(name, None)).encode("utf-8"))
            hasher.update(b"\0")
        _hash_qobj(hasher, prepared.static_hamiltonian)
        for term in prepared.drive_terms:
            hasher.update(str(term.mode).encode("utf-8"))
            hasher.update(b"\0")
            _hash_qobj(hasher, term.operator)
            t_axis, values, domain, lo_freq = _validate_trace(term.trace)
            hasher.update(str(domain).encode("ascii"))
            hasher.update(repr(float(lo_freq)).encode("ascii"))
            _hash_array(hasher, t_axis)
            _hash_array(hasher, values)
        _hash_array(hasher, np.asarray(prepared.tlist, dtype=np.float64))
        return hasher.digest()

    def _ensure_plan(self) -> _NativePlan:
        """Resolve this backend's plan, consulting the bounded process cache."""
        if self._plan is not None:
            return self._plan
        cache_size = int(getattr(self.prepared.options, "plan_cache_size", 16))
        key = self._plan_cache_key() if cache_size > 0 else None
        if key is not None:
            with _PLAN_CACHE_LOCK:
                cached = _PLAN_CACHE.get(key)
                if cached is not None:
                    _PLAN_CACHE.move_to_end(key)
                    self._plan = cached
                    self._plan_cache_hit = True
                    self._plan_cache_digest = key
                    return cached
        plan = self._build_plan()
        if key is not None:
            with _PLAN_CACHE_LOCK:
                _PLAN_CACHE[key] = plan
                _PLAN_CACHE.move_to_end(key)
                while len(_PLAN_CACHE) > cache_size:
                    _PLAN_CACHE.popitem(last=False)
        self._plan = plan
        self._plan_cache_hit = False
        self._plan_cache_digest = key
        return plan

    def _ensure_block_backends(self):
        """Prepare independent exact block propagators when structure allows it."""
        if self._block_backends is not None:
            return self._block_backends
        options = self.prepared.options
        requested = str(getattr(options, "block_decompose", "auto")).lower()
        if requested == "off":
            self._block_backends = ()
            return self._block_backends
        if self.prepared.c_ops or self.prepared.backend == "cpp_rwa":
            self._block_backends = ()
            return self._block_backends
        frame = str(getattr(options, "frame", "lab")).lower()
        if frame != "lab":
            # A unitary interaction-basis transform can destroy a sparse block
            # pattern.  Keep the explicitly selected frame authoritative.
            self._block_backends = ()
            return self._block_backends
        n = int(self.prepared.static_hamiltonian.shape[0])
        if (
            self.prepared.static_hamiltonian.shape[0]
            != self.prepared.static_hamiltonian.shape[1]
            or getattr(self.prepared.static_hamiltonian, "issuper", False)
        ):
            self._block_backends = ()
            return self._block_backends
        if requested == "auto" and n < 16:
            self._block_backends = ()
            return self._block_backends
        # A diagonal problem already has an O(n * samples) exact path.  Splitting
        # it into one-dimensional Python backends would add overhead without
        # reducing the native arithmetic.
        if _diagonal_from_csr(
            _qobj_csr(self.prepared.static_hamiltonian),
            n,
        )[0] and all(
            _diagonal_from_csr(_qobj_csr(term.operator), n)[0]
            for term in self.prepared.drive_terms
        ):
            self._block_backends = ()
            return self._block_backends
        components = _connected_components(
            self.prepared.static_hamiltonian,
            [term.operator for term in self.prepared.drive_terms],
        )
        if len(components) <= 1:
            self._block_backends = ()
            return self._block_backends
        # Explicit ``on`` is useful for benchmarking small blocks; automatic
        # selection avoids overhead when a decomposition would be nearly the
        # same size as the original problem.
        largest = max(len(item) for item in components)
        if requested == "auto" and largest >= 0.9 * n:
            self._block_backends = ()
            return self._block_backends
        block_options = replace(options, backend="cpp", block_decompose="off")
        backends = []
        for indices in components:
            static = _slice_operator(self.prepared.static_hamiltonian, indices)
            terms = [
                type(term)(
                    _slice_operator(term.operator, indices),
                    term.trace,
                    mode=term.mode,
                )
                for term in self.prepared.drive_terms
            ]
            block_prepared = self.prepared.__class__(
                static,
                terms,
                self.prepared.tlist,
                c_ops=[],
                options=block_options,
                args=self.prepared.args,
                backend="cpp",
            )
            backends.append((indices, CppPropagationBackend.from_prepared(block_prepared)))
        self._block_backends = tuple(backends)
        return self._block_backends

    def _attach_polynomial_coefficients(
        self,
        plan: _NativePlan,
        coefficient_order: int,
        polynomial: Optional[np.ndarray] = None,
        effective_order: Optional[int] = None,
        output_t_axis: Optional[np.ndarray] = None,
        output_indices: Optional[np.ndarray] = None,
        control_modes: Optional[np.ndarray] = None,
    ) -> _NativePlan:
        """Attach exact per-interval coefficients for non-linear interpolation."""
        requested = int(coefficient_order)
        if polynomial is None and requested != 1 and self.prepared.drive_terms:
            # Retain a defensive direct path for callers constructing plans in
            # tests or downstream extensions without the common-grid helper.
            arrays = []
            effective_orders = []
            for term in self.prepared.drive_terms:
                coefficients, effective = _native_polynomial_coefficients(term.trace, requested)
                arrays.append(coefficients)
                effective_orders.append(effective)
            if len(set(effective_orders)) > 1:
                raise UnsupportedBackendError(
                    "native polynomial traces must have a common effective interpolation order"
                )
            effective_order = effective_orders[0] if effective_orders else requested
            polynomial = (
                np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.complex128)
                if arrays
                else None
            )
        if effective_order is None:
            effective_order = requested
        return self._freeze_plan_arrays(
            replace(
                plan,
                iq_polynomial=polynomial,
                coefficient_order=int(effective_order),
                output_t_axis=output_t_axis,
                output_indices=output_indices,
                control_modes=control_modes,
            )
        )

    @staticmethod
    def _combine_block_results(
        original_states: Sequence[qt.Qobj],
        block_payloads,
        tlist: np.ndarray,
    ) -> BatchPropagationResult:
        """Reassemble block-wise native results in the caller's basis."""
        states = list(original_states)
        if not states:
            return BatchPropagationResult([], [], np.array(tlist, copy=True), {})
        n = int(_qobj_matrix(states[0]).shape[0])
        batch_count = len(states)
        final_matrix = np.zeros((n, batch_count), dtype=np.complex128)
        block_results = []
        for indices, batch_result in block_payloads:
            for column, state in enumerate(batch_result.final_states):
                final_matrix[indices, column] = _qobj_matrix(state)[:, 0]
            block_results.append((indices, batch_result))

        aggregate: Dict[str, Any] = {}
        numeric_sum = {
            "steps",
            "rhs_evaluations",
            "exponential_evaluations",
            "exponential_cache_hits",
            "diagonal_interval_evaluations",
            "zero_interval_evaluations",
            "krylov_evaluations",
            "krylov_iterations",
        }
        numeric_max = {"max_error", "max_trial_error"}
        for _indices, batch_result in block_results:
            for key, value in dict(batch_result.stats or {}).items():
                if key in numeric_sum:
                    aggregate[key] = aggregate.get(key, 0) + int(value)
                elif key in numeric_max:
                    aggregate[key] = max(float(aggregate.get(key, 0.0)), float(value))
                elif key not in aggregate:
                    aggregate[key] = value
        aggregate.update(
            {
                "block_decomposed": True,
                "block_count": len(block_results),
                "block_sizes": tuple(int(len(indices)) for indices, _ in block_results),
                "integrator": "cpp_block_decomposed",
            }
        )
        final_states = [
            qt.Qobj(final_matrix[:, column], dims=states[column].dims)
            for column in range(batch_count)
        ]
        per_state_results = []
        store_states = bool(
            block_results
            and block_results[0][1].results
            and block_results[0][1].results[0].states
        )
        for column, original in enumerate(states):
            trajectories = []
            if store_states:
                for time_index, _time in enumerate(tlist):
                    vector = np.zeros(n, dtype=np.complex128)
                    for indices, batch_result in block_results:
                        block_states = batch_result.results[column].states
                        if time_index < len(block_states):
                            vector[indices] = _qobj_matrix(block_states[time_index])[:, 0]
                    trajectories.append(qt.Qobj(vector, dims=original.dims))
            state_stats = dict(aggregate)
            per_state_results.append(
                NativePropagationResult(
                    final_state=final_states[column],
                    states=trajectories,
                    times=np.array(tlist, copy=True),
                    stats=state_stats,
                )
            )
        return BatchPropagationResult(
            final_states=final_states,
            results=per_state_results,
            times=np.array(tlist, copy=True),
            stats=aggregate,
        )

    def _propagate_blockwise(self, states: Sequence[qt.Qobj]):
        blocks = self._ensure_block_backends()
        if not blocks:
            return None
        payloads = []
        for indices, backend in blocks:
            sub_states = []
            for state in states:
                vector = _qobj_matrix(state)
                sub_states.append(
                    qt.Qobj(
                        np.ascontiguousarray(vector[indices, :], dtype=np.complex128),
                        dims=[[len(indices)], [1]],
                    )
                )
            payloads.append((indices, backend.propagate_batch(sub_states)))
        return self._combine_block_results(states, payloads, self.prepared.tlist)

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
        iq_polynomial = plan.iq_polynomial
        if iq_polynomial is not None:
            iq_polynomial = _freeze_array(iq_polynomial)
        output_t_axis = plan.output_t_axis
        if output_t_axis is not None:
            output_t_axis = _freeze_array(output_t_axis)
        output_indices = plan.output_indices
        if output_indices is not None:
            output_indices = _freeze_array(output_indices)
        control_modes = plan.control_modes
        if control_modes is not None:
            control_modes = _freeze_array(control_modes)
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
            iq_polynomial=iq_polynomial,
            coefficient_order=plan.coefficient_order,
            output_t_axis=output_t_axis,
            output_indices=output_indices,
            control_modes=control_modes,
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
        interaction_block_count = 1
        interaction_block_max = int(static_operator.shape[0])
        if static_diagonal:
            control_csrs = [_qobj_csr(operator) for operator in drive_operators]
        else:
            # A Hermitian static Hamiltonian can be diagonalized exactly by a
            # unitary basis change.  For a block-diagonal sparse H0, diagonalize
            # each connected block independently so the dense eigensolver never
            # sees the full Hilbert-space dimension.
            dimension = int(static_operator.shape[0])
            block_components = _connected_components(static_operator, [])
            interaction_block_count = len(block_components)
            interaction_block_max = max(len(item) for item in block_components)
            if dimension > 256 and (
                len(block_components) <= 1
                or max(len(item) for item in block_components) > 256
            ):
                raise UnsupportedBackendError(
                    "frame='interaction_exact' requires H0 blocks of dimension <= 256."
                )
            static_matrix = (
                _qobj_matrix(static_operator)
                if dimension <= 256
                else None
            )
            # ``eigh`` assumes a Hermitian input.  Permit only the tiny
            # antisymmetry introduced by floating-point construction of a
            # mathematically Hermitian Qobj; anything larger is rejected
            # instead of being silently projected into a new Hamiltonian.
            if static_matrix is not None:
                adjoint = static_matrix.conj().T
                scale = max(1.0, float(np.linalg.norm(static_matrix)))
            else:
                static_sparse = _qobj_csr(static_operator)
                try:
                    from scipy import sparse
                except ImportError as exc:  # pragma: no cover
                    raise BackendUnavailable(
                        "scipy is required for block interaction preparation"
                    ) from exc
                sparse_matrix = sparse.csr_matrix(
                    (static_sparse[0], static_sparse[1], static_sparse[2]),
                    shape=(dimension, dimension),
                )
                scale = max(1.0, float(np.max(np.asarray(np.abs(sparse_matrix).sum(axis=0)).ravel())))
                adjoint = None
            if not np.isfinite(scale):
                raise UnsupportedBackendError(
                    "frame='interaction_exact' requires a finite static Hamiltonian."
                )
            if static_matrix is not None:
                hermitian_residual = float(np.linalg.norm(static_matrix - adjoint))
            else:
                hermitian_residual = 0.0
                for indices in block_components:
                    block = sparse_matrix[indices, :][:, indices].toarray()
                    hermitian_residual = max(
                        hermitian_residual,
                        float(np.linalg.norm(block - block.conj().T)),
                    )
            hermitian_tolerance = 64.0 * np.finfo(np.float64).eps * scale
            if hermitian_residual > hermitian_tolerance:
                raise UnsupportedBackendError(
                    "frame='interaction_exact' requires a Hermitian static Hamiltonian."
                )
            # Canonicalize only this bounded round-off residue so the LAPACK
            # eigensolver sees a Hermitian matrix.  The residual is many orders
            # below the native integration tolerance for ordinary inputs.
            if static_matrix is not None:
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
            else:
                interaction_basis = np.zeros(
                    (dimension, dimension), dtype=np.complex128
                )
                energies_real = np.empty(dimension, dtype=np.float64)
                basis_residual = 0.0
                eigen_residual = 0.0
                for indices in block_components:
                    block = sparse_matrix[indices, :][:, indices].toarray()
                    block = 0.5 * (block + block.conj().T)
                    block_values, block_vectors = np.linalg.eigh(block)
                    interaction_basis[np.ix_(indices, indices)] = block_vectors
                    energies_real[indices] = block_values
                    block_basis_residual = float(
                        np.linalg.norm(
                            block_vectors.conj().T @ block_vectors
                            - np.eye(len(indices), dtype=np.complex128)
                        )
                    )
                    block_eigen_residual = float(
                        np.linalg.norm(
                            block @ block_vectors
                            - block_vectors * block_values[np.newaxis, :]
                        )
                        / scale
                    )
                    basis_residual = max(basis_residual, block_basis_residual)
                    eigen_residual = max(eigen_residual, block_eigen_residual)
            if basis_residual > 1e-10 or eigen_residual > 1e-10:
                raise UnsupportedBackendError(
                    "frame='interaction_exact' eigendecomposition failed its numerical residual check."
                )
            energies = np.asarray(energies_real, dtype=np.complex128)
            transformed_controls = []
            for operator in drive_operators:
                matrix = (
                    _transform_block_diagonal_basis(
                        operator,
                        interaction_basis,
                        block_components,
                    )
                    if static_matrix is None
                    else _qobj_matrix(operator)
                )
                transformed_controls.append(
                    np.asarray(
                        matrix
                        if static_matrix is None
                        else interaction_basis.conj().T @ matrix @ interaction_basis,
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
        interaction_metadata = dict(metadata or {})
        interaction_metadata.update(
            {
                "interaction_block_count": interaction_block_count,
                "interaction_block_max": interaction_block_max,
            }
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
            metadata=interaction_metadata,
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
        coefficient_order = int(getattr(prepared.options, "coefficient_order", 1))
        if prepared.backend == "cpp_rwa":
            if coefficient_order != 1:
                raise UnsupportedBackendError(
                    "cpp_rwa requires coefficient_order=1; use the exact native path "
                    "for polynomial coefficient representations."
                )
            return self._build_rwa_plan()

        static_operator = prepared.static_hamiltonian
        if not isinstance(static_operator, qt.Qobj):
            raise TypeError("static_hamiltonian must be a qutip.Qobj")
        if (
            static_operator.shape[0] != static_operator.shape[1]
            or getattr(static_operator, "issuper", False)
        ):
            raise UnsupportedBackendError(
                "The native backend requires a square operator Hamiltonian, not a superoperator."
            )
        operators: List[qt.Qobj] = []
        trace_axes = []
        traces = []
        lo_freqs = []
        modes = []
        for term in prepared.drive_terms:
            if not isinstance(term.operator, qt.Qobj):
                raise TypeError("Every drive operator must be a qutip.Qobj.")
            if term.operator.shape != static_operator.shape:
                raise ValueError("Every drive operator must match the static Hamiltonian dimension.")
            if getattr(term.operator, "issuper", False):
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
            trace_axes.append(t_axis)
            traces.append(values)
            lo_freqs.append(lo_freq if domain == "iq_complex" else 0.0)
            modes.append(1 if term.mode == "complex_envelope" else 0)

        output_t_axis = np.ascontiguousarray(prepared.tlist, dtype=np.float64)
        self._require_regular_grid(output_t_axis)
        coefficient_order = int(getattr(prepared.options, "coefficient_order", 1))
        grid_matches = all(
            np.array_equal(axis, output_t_axis)
            for axis in trace_axes
        )
        needs_polynomial_grid = coefficient_order != 1 or not grid_matches
        if needs_polynomial_grid:
            (
                t_axis,
                iq,
                polynomial,
                effective_order,
                output_indices,
            ) = _prepare_native_trace_payloads(
                [term.trace for term in prepared.drive_terms],
                output_t_axis,
                coefficient_order,
            )
        else:
            t_axis = output_t_axis
            iq = (
                np.ascontiguousarray(np.stack(traces, axis=0), dtype=np.complex128)
                if traces
                else np.empty((0, len(t_axis)), dtype=np.complex128)
            )
            polynomial = None
            effective_order = coefficient_order
            output_indices = np.arange(len(t_axis), dtype=np.int64)
        native_mode = (modes[0] if modes else 1) if len(set(modes)) <= 1 else 2
        control_modes = np.ascontiguousarray(np.asarray(modes, dtype=np.int64))
        frame = str(getattr(prepared.options, "frame", "lab")).lower()
        plan = None
        if frame in {"interaction_exact", "auto"}:
            try:
                plan = self._plan_from_interaction_qobjs(
                    static_operator,
                    operators,
                    iq,
                    t_axis,
                    np.asarray(lo_freqs, dtype=np.float64),
                    native_mode,
                )
            except (BackendUnavailable, UnsupportedBackendError):
                if frame == "interaction_exact":
                    raise
            except np.linalg.LinAlgError as exc:
                if frame == "interaction_exact":
                    raise UnsupportedBackendError(
                        "frame='interaction_exact' eigendecomposition failed."
                    ) from exc
        if plan is None:
            plan = self._plan_from_qobjs(
                static_operator,
                operators,
                iq,
                t_axis,
                np.asarray(lo_freqs, dtype=np.float64),
                native_mode,
            )
        return self._attach_polynomial_coefficients(
            plan,
            coefficient_order,
            polynomial=polynomial,
            effective_order=effective_order,
            output_t_axis=output_t_axis,
            output_indices=output_indices,
            control_modes=control_modes,
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
            or getattr(prepared.static_hamiltonian, "issuper", False)
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
        if getattr(term.operator, "issuper", False):
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
        plan = self._ensure_plan()
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
            if metadata is not None and "basis_transform" in metadata:
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
        if metadata is None or "frame_omega" not in metadata:
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

    @staticmethod
    def _invoke_native(function, *args, **kwargs):
        """Call a kernel, retaining compatibility with older extensions."""
        try:
            return function(*args, **kwargs)
        except TypeError as error:
            legacy = dict(kwargs)
            # These keywords were added after the original optional module.
            # They are safe to omit only for their exact default semantics.
            if legacy.get("sparse_expm") == 1:
                legacy.pop("sparse_expm", None)
            if legacy.get("iq_polynomial") is None:
                legacy.pop("iq_polynomial", None)
            if legacy.get("coefficient_order") == 1:
                legacy.pop("coefficient_order", None)
            if legacy.get("parallel") == 1:
                legacy.pop("parallel", None)
            if legacy.get("collapse_rates") is None:
                legacy.pop("collapse_rates", None)
            modes = legacy.get("control_modes")
            if modes is None or (
                isinstance(modes, np.ndarray)
                and (modes.size == 0 or np.all(modes == modes.flat[0]))
            ):
                legacy.pop("control_modes", None)
            if len(legacy) == len(kwargs):
                raise
            # Do not hide a TypeError from a genuinely active new feature.
            if kwargs.get("iq_polynomial") is not None:
                raise
            if kwargs.get("coefficient_order", 1) != 1:
                raise
            if kwargs.get("collapse_rates") is not None:
                raise
            modes = kwargs.get("control_modes")
            if isinstance(modes, np.ndarray) and modes.size > 0 and not np.all(modes == modes.flat[0]):
                raise
            return function(*args, **legacy)

    def _call_native(self, payload: _NativePayload):
        plan = payload.plan
        common = {
            "mode": int(plan.mode),
            "atol": float(self.prepared.options.atol),
            "rtol": float(self.prepared.options.rtol),
            "max_steps": int(self.prepared.options.extra.get("native_max_steps", 2_000_000)),
            "store_trajectory": bool(self.prepared.options.store_states),
            "sparse_expm": {
                "off": 0,
                "auto": 1,
                "on": 2,
            }.get(str(getattr(self.prepared.options, "sparse_expm", "auto")).lower(), 1),
            "parallel": {
                "off": 0,
                "auto": 1,
                "on": 2,
            }.get(str(getattr(self.prepared.options, "parallel", "auto")).lower(), 1),
            "iq_polynomial": plan.iq_polynomial,
            "coefficient_order": int(plan.coefficient_order),
            "control_modes": plan.control_modes,
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
            return self._invoke_native(_native_propagate_interaction_csr,
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
            return self._invoke_native(_native_propagate_fused_csr,
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
            return self._invoke_native(_native_propagate_banded,
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
            return self._invoke_native(_native_propagate_csr,
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
        return self._invoke_native(_native_propagate,
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
        resolved.setdefault("block_decomposed", False)
        resolved["sparse_expm"] = str(
            getattr(self.prepared.options, "sparse_expm", "auto")
        ).lower()
        resolved["coefficient_order"] = int(plan.coefficient_order)
        resolved["source_sample_count"] = int(plan.t_axis.size)
        resolved["output_sample_count"] = int(
            plan.output_t_axis.size
            if plan.output_t_axis is not None
            else plan.t_axis.size
        )
        if plan.metadata is not None:
            for key in ("interaction_block_count", "interaction_block_max"):
                if key in plan.metadata:
                    resolved[key] = plan.metadata[key]
        resolved["plan_cache_hit"] = bool(self._plan_cache_hit)
        resolved["plan_cache_entries"] = native_plan_cache_info()["entries"]
        if plan.metadata is not None and "basis_transform" in plan.metadata:
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
        block_result = self._propagate_blockwise([initial_state])
        if block_result is not None:
            return block_result.results[0]
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
            output_indices = payload.plan.output_indices
            if output_indices is not None:
                trajectory_array = trajectory_array[output_indices]
            states = [qt.Qobj(item[:, 0], dims=payload.dims[0]) for item in trajectory_array]
        resolved_stats = self._resolved_stats(stats, payload.plan)
        return NativePropagationResult(
            final_state=state,
            states=states,
            times=np.array(
                payload.plan.output_t_axis
                if payload.plan.output_t_axis is not None
                else payload.plan.t_axis,
                copy=True,
            ),
            stats=resolved_stats,
        )

    def propagate_batch(self, initial_states: Sequence[qt.Qobj]) -> BatchPropagationResult:
        states = list(initial_states)
        if not states:
            return BatchPropagationResult([], [], np.array(self.prepared.tlist, copy=True), {})
        block_result = self._propagate_blockwise(states)
        if block_result is not None:
            return block_result
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
            output_indices = payload.plan.output_indices
            if output_indices is not None:
                trajectory_array = trajectory_array[output_indices]
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
                    times=np.array(
                        payload.plan.output_t_axis
                        if payload.plan.output_t_axis is not None
                        else payload.plan.t_axis,
                        copy=True,
                    ),
                    stats=dict(resolved_stats),
                )
            )
        return BatchPropagationResult(
            final_states=final_states,
            results=per_state_results,
            times=np.array(
                payload.plan.output_t_axis
                if payload.plan.output_t_axis is not None
                else payload.plan.t_axis,
                copy=True,
            ),
            stats=resolved_stats,
        )


class LindbladCppPropagationBackend:
    """Exact native path for static-collapse Lindblad evolution.

    The wrapper vectorizes the density matrix once and delegates all numerical
    work to :class:`CppPropagationBackend`.  It intentionally accepts only
    static collapse operators; unsupported time-dependent forms remain on the
    QuTiP route instead of being sampled or silently altered.
    """

    def __init__(self, prepared) -> None:
        if not cpp_backend_available():
            raise BackendUnavailable(
                "The optional C++ backend is not built. "
                "Install the native build dependencies and set PYSUQU_BUILD_NATIVE=1."
            )
        if not prepared.c_ops:
            raise ValueError("LindbladCppPropagationBackend requires at least one collapse operator")
        if prepared.backend == "cpp_rwa":
            raise UnsupportedBackendError(
                "the approximate RWA path cannot be combined with native Lindblad propagation"
            )
        frame = str(getattr(prepared.options, "frame", "lab")).lower()
        if frame == "interaction_exact":
            raise UnsupportedBackendError(
                "interaction_exact is currently defined for coherent ket propagation only"
            )
        dimension = int(prepared.static_hamiltonian.shape[0])
        if prepared.static_hamiltonian.shape != (dimension, dimension):
            raise UnsupportedBackendError("native Lindblad propagation requires a square Hamiltonian")
        extra = dict(getattr(prepared.options, "extra", {}) or {})
        # A density-vector's Euclidean norm is not its trace.  Disable the ket
        # adapter's optional norm correction and preserve the Lindblad trace
        # exactly as produced by the numerical integrator.
        extra["normalize_output"] = False
        proxy_options = replace(
            prepared.options,
            backend="cpp",
            frame="lab",
            matrix_format="csr",
            sparse_kernel="standard",
            block_decompose="off",
            extra=extra,
        )
        self.prepared = prepared
        self.dimension = dimension
        self._direct = callable(_native_propagate_lindblad_csr)
        self._proxy = None
        self._backend = None
        self._direct_plan = None
        self._direct_h0 = None
        self._direct_controls = None
        self._direct_collapses = None
        self._direct_collapse_rates = None
        if self._direct:
            # A coherent proxy is used only for the shared trace/grid plan;
            # the native call below applies the physical Lindblad equation
            # directly and never uses its Hamiltonian as a Liouvillian.
            self._proxy = PreparedPropagation(
                prepared.static_hamiltonian,
                prepared.drive_terms,
                prepared.tlist,
                c_ops=[],
                options=proxy_options,
                args=prepared.args,
                backend="cpp",
            )
            self._backend = CppPropagationBackend.from_prepared(self._proxy)
            self._direct_plan = self._backend._ensure_plan()
            self._direct_h0, self._direct_controls, self._direct_collapses = (
                _native_lindblad_csr_components(prepared)
            )
            self._direct_collapse_rates = _native_lindblad_rate_components(prepared)
        else:
            effective_static, effective_drives = _lindblad_effective_operators(
                prepared.static_hamiltonian,
                prepared.drive_terms,
                prepared.c_ops,
            )
            self._proxy = PreparedPropagation(
                effective_static,
                effective_drives,
                prepared.tlist,
                c_ops=[],
                options=proxy_options,
                args=prepared.args,
                backend="cpp",
            )
            self._backend = CppPropagationBackend.from_prepared(self._proxy)

    def _density(self, state: qt.Qobj) -> Tuple[qt.Qobj, Any]:
        if not isinstance(state, qt.Qobj):
            raise TypeError("initial_state must be a qutip.Qobj")
        if state.shape != (self.dimension, self.dimension) and not state.isket:
            raise ValueError("initial state dimension does not match the Hamiltonian")
        if state.isket:
            if state.shape[0] != self.dimension:
                raise ValueError("initial state dimension does not match the Hamiltonian")
            density = state * state.dag()
        elif state.isoper and state.shape == (self.dimension, self.dimension):
            density = state
        else:
            raise UnsupportedBackendError(
                "native Lindblad propagation accepts ket or density-matrix initial states"
            )
        values = np.asarray(density.full(), dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("initial density matrix must contain finite values")
        return density, density.dims

    def _vector_state(self, density: qt.Qobj) -> qt.Qobj:
        vector = np.asarray(density.full(), dtype=np.complex128).reshape(-1, order="F")
        return qt.Qobj(
            np.ascontiguousarray(vector[:, np.newaxis], dtype=np.complex128),
            dims=[[self.dimension * self.dimension], [1]],
        )

    def _density_state(self, vector_state: qt.Qobj, dims) -> qt.Qobj:
        return self._density_array(vector_state.full(), dims)

    def _density_array(self, values: np.ndarray, dims) -> qt.Qobj:
        vector = np.asarray(values, dtype=np.complex128).reshape(-1)
        if vector.size != self.dimension * self.dimension:
            raise ValueError("native Lindblad result has an invalid vector dimension")
        matrix = np.asarray(
            vector.reshape((self.dimension, self.dimension), order="F"),
            dtype=np.complex128,
        )
        return qt.Qobj(matrix, dims=dims)

    def _direct_native_call(self, initial_matrix: np.ndarray):
        """Invoke the matrix-free Lindblad extension for one batch payload."""
        plan = self._direct_plan
        if plan is None or self._direct_h0 is None or self._direct_controls is None:
            raise BackendUnavailable("native Lindblad plan is not initialized")
        if _native_propagate_lindblad_csr is None:
            raise BackendUnavailable("the matrix-free Lindblad extension is unavailable")
        h0_data, h0_indices, h0_indptr = self._direct_h0
        controls_data, controls_indices, controls_indptr, controls_offsets = self._direct_controls
        collapse_data, collapse_indices, collapse_indptr, collapse_offsets, _ = self._direct_collapses
        common = {
            "mode": int(plan.mode),
            "atol": float(self.prepared.options.atol),
            "rtol": float(self.prepared.options.rtol),
            "max_steps": int(self.prepared.options.extra.get("native_max_steps", 2_000_000)),
            "store_trajectory": bool(self.prepared.options.store_states),
            "sparse_expm": {
                "off": 0,
                "auto": 1,
                "on": 2,
            }.get(str(getattr(self.prepared.options, "sparse_expm", "auto")).lower(), 1),
            "iq_polynomial": plan.iq_polynomial,
            "coefficient_order": int(plan.coefficient_order),
            "control_modes": plan.control_modes,
            "parallel": {
                "off": 0,
                "auto": 1,
                "on": 2,
            }.get(str(getattr(self.prepared.options, "parallel", "auto")).lower(), 1),
            "collapse_rates": self._direct_collapse_rates,
        }
        return self._backend._invoke_native(
            _native_propagate_lindblad_csr,
            h0_data,
            h0_indices,
            h0_indptr,
            controls_data,
            controls_indices,
            controls_indptr,
            controls_offsets,
            collapse_data,
            collapse_indices,
            collapse_indptr,
            collapse_offsets,
            plan.iq,
            plan.t_axis,
            np.ascontiguousarray(initial_matrix, dtype=np.complex128),
            plan.lo_freqs,
            **common,
        )

    def _direct_result(self, initial_states: Sequence[qt.Qobj]):
        states = list(initial_states)
        densities = []
        dims = []
        for state in states:
            density, state_dims = self._density(state)
            densities.append(
                np.asarray(density.full(), dtype=np.complex128).reshape(-1, order="F")
            )
            dims.append(state_dims)
        initial_matrix = np.ascontiguousarray(np.column_stack(densities), dtype=np.complex128)
        final_payload, trajectory_payload, native_stats = self._direct_native_call(initial_matrix)
        plan = self._direct_plan
        batch_count = len(states)
        final_vectors = _decode_complex_payload(
            final_payload,
            (self.dimension * self.dimension, batch_count),
        )
        final_states = []
        for index in range(batch_count):
            final_states.append(
                self._density_array(final_vectors[:, index], dims[index])
            )
        trajectory = _decode_complex_payload(
            trajectory_payload,
            (len(plan.t_axis), self.dimension * self.dimension, batch_count),
        )
        if trajectory is not None and plan.output_indices is not None:
            trajectory = trajectory[np.asarray(plan.output_indices, dtype=np.int64)]
        output_times = np.array(
            plan.output_t_axis if plan.output_t_axis is not None else plan.t_axis,
            copy=True,
        )
        per_state = []
        for index in range(batch_count):
            item_states = []
            if trajectory is not None:
                item_states = [
                    self._density_array(sample[:, index], dims[index])
                    for sample in trajectory
                ]
            stats = dict(native_stats or {})
            stats.update(
                {
                    "open_system": "lindblad",
                    "liouvillian_dimension": self.dimension * self.dimension,
                    "collapse_operators": len(self.prepared.c_ops),
                    "matrix_free": True,
                }
            )
            per_state.append(
                NativePropagationResult(
                    final_state=final_states[index],
                    states=item_states,
                    times=np.array(output_times, copy=True),
                    stats=stats,
                )
            )
        stats = dict(native_stats or {})
        stats.update(
            {
                "open_system": "lindblad",
                "liouvillian_dimension": self.dimension * self.dimension,
                "collapse_operators": len(self.prepared.c_ops),
                "matrix_free": True,
            }
        )
        return BatchPropagationResult(
            final_states=final_states,
            results=per_state,
            times=output_times,
            stats=stats,
        )

    def propagate(self, initial_state: qt.Qobj) -> NativePropagationResult:
        if self._direct:
            return self._direct_result([initial_state]).results[0]
        density, dims = self._density(initial_state)
        result = self._backend.propagate(self._vector_state(density))
        states = [self._density_state(item, dims) for item in result.states]
        stats = dict(result.stats or {})
        stats.update(
            {
                "open_system": "lindblad",
                "liouvillian_dimension": self.dimension * self.dimension,
                "collapse_operators": len(self.prepared.c_ops),
            }
        )
        return NativePropagationResult(
            final_state=self._density_state(result.final_state, dims),
            states=states,
            times=np.array(result.times, copy=True),
            stats=stats,
        )

    def propagate_batch(self, initial_states: Sequence[qt.Qobj]) -> BatchPropagationResult:
        states = list(initial_states)
        if not states:
            return BatchPropagationResult([], [], np.array(self.prepared.tlist, copy=True), {})
        if self._direct:
            return self._direct_result(states)
        densities = []
        dims = []
        for state in states:
            density, state_dims = self._density(state)
            densities.append(self._vector_state(density))
            dims.append(state_dims)
        result = self._backend.propagate_batch(densities)
        final_states = [
            self._density_state(state, dims[index])
            for index, state in enumerate(result.final_states)
        ]
        per_state = []
        for index, item in enumerate(result.results):
            stats = dict(item.stats or {})
            stats.update(
                {
                    "open_system": "lindblad",
                    "liouvillian_dimension": self.dimension * self.dimension,
                    "collapse_operators": len(self.prepared.c_ops),
                }
            )
            per_state.append(
                NativePropagationResult(
                    final_state=final_states[index],
                    states=[self._density_state(value, dims[index]) for value in item.states],
                    times=np.array(item.times, copy=True),
                    stats=stats,
                )
            )
        stats = dict(result.stats or {})
        stats.update(
            {
                "open_system": "lindblad",
                "liouvillian_dimension": self.dimension * self.dimension,
                "collapse_operators": len(self.prepared.c_ops),
            }
        )
        return BatchPropagationResult(
            final_states=final_states,
            results=per_state,
            times=np.array(result.times, copy=True),
            stats=stats,
        )


__all__ = [
    "CppPropagationBackend",
    "NativePropagationResult",
    "cpp_backend_available",
    "cpp_banded_backend_available",
    "cpp_csr_backend_available",
    "cpp_fused_csr_backend_available",
    "cpp_interaction_backend_available",
    "LindbladCppPropagationBackend",
    "cpp_lindblad_backend_available",
    "clear_native_plan_cache",
    "native_plan_cache_info",
]
