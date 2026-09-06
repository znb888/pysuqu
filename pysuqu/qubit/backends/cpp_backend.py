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


def cpp_backend_available() -> bool:
    return callable(_native_propagate)


def cpp_csr_backend_available() -> bool:
    """Return whether the optional CSR kernel is available."""
    return callable(_native_propagate_csr)


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
    metadata: Optional[Dict[str, Any]] = None
    h0_csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    controls_csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None


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

    def _resolve_matrix_format(self, n: int, total_nnz: int, matrix_count: int) -> str:
        requested = str(getattr(self.prepared.options, "matrix_format", "auto")).lower()
        if requested == "csr":
            if not cpp_csr_backend_available():
                raise BackendUnavailable(
                    "The native CSR kernel is unavailable; rebuild the optional extension."
                )
            return "csr"
        if requested == "dense" or (requested == "auto" and n < 8):
            return "dense"
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
        """Reject grids the native interpolation indexer cannot represent."""
        if len(t_axis) < 2:
            return
        deltas = np.diff(t_axis)
        reference = float(deltas[0])
        tolerance = 1e-9 * max(1.0, abs(reference))
        if np.any(deltas <= 0.0) or np.any(np.abs(deltas - reference) > tolerance):
            raise UnsupportedBackendError(
                "The C++ backend currently requires a regular tlist grid."
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
        return _NativePlan(
            matrix_format=plan.matrix_format,
            static_matrix=(None if plan.static_matrix is None else _freeze_array(plan.static_matrix)),
            controls=(None if plan.controls is None else _freeze_array(plan.controls)),
            iq=_freeze_array(plan.iq),
            t_axis=_freeze_array(plan.t_axis),
            lo_freqs=_freeze_array(plan.lo_freqs),
            mode=plan.mode,
            metadata=metadata,
            h0_csr=h0_csr,
            controls_csr=controls_csr,
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
        if requested == "dense":
            matrix_format = "dense"
            control_csrs = []
            static_csr = None
        else:
            static_csr = _qobj_csr(static_operator)
            control_csrs = [_qobj_csr(operator) for operator in drive_operators]
            total_nnz = int(static_csr[0].size) + sum(int(item[0].size) for item in control_csrs)
            matrix_format = self._resolve_matrix_format(n, total_nnz, 1 + len(control_csrs))
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

        static_matrix = _qobj_matrix(static_operator)
        controls = (
            np.ascontiguousarray(
                np.stack([_qobj_matrix(operator) for operator in drive_operators], axis=0),
                dtype=np.complex128,
            )
            if drive_operators
            else np.empty((0, n, n), dtype=np.complex128)
        )
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
        need_csr = requested != "dense" and (requested == "csr" or (cpp_csr_backend_available() and n >= 8))
        bundle = (
            _csr_bundle_from_arrays(
                [static_matrix] + [controls[index] for index in range(controls.shape[0])],
                n=n,
            )
            if need_csr
            else None
        )
        total_nnz = bundle[4] if bundle is not None else (1 + controls.shape[0]) * n * n
        matrix_format = self._resolve_matrix_format(n, total_nnz, 1 + controls.shape[0])
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
                or not np.allclose(source_t, t_axis, rtol=1e-12, atol=1e-12)
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
        if len(t_axis) != len(prepared_t) or not np.allclose(t_axis, prepared_t, rtol=1e-12, atol=1e-12):
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
            assert plan.h0_csr is not None
            native_dimension = int(plan.h0_csr[2].size - 1)
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
        final = self._apply_frame(final, payload.plan.t_axis[-1], payload.plan.metadata)
        final = self._normalize_output(final, payload.normalize_columns)
        state = qt.Qobj(final[:, 0], dims=payload.dims[0])
        states = []
        trajectory_array = _decode_complex_payload(
            trajectory,
            (len(payload.plan.t_axis), n, 1),
        )
        if trajectory_array is not None:
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
    "cpp_csr_backend_available",
]
