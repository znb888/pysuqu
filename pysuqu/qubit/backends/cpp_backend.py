"""Python adapter for the optional C++ propagation kernel.

The extension is deliberately optional.  Installing the normal ``pysuqu``
package continues to work without a compiler; users who request ``backend='cpp'``
receive a precise error when the extension has not been built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

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


def cpp_backend_available() -> bool:
    return callable(_native_propagate)


def _qobj_matrix(value: qt.Qobj) -> np.ndarray:
    data = np.asarray(value.full(), dtype=np.complex128)
    if not np.all(np.isfinite(data)):
        raise ValueError("Native Hamiltonian and drive operators must contain finite values.")
    return np.ascontiguousarray(data)


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


class CppPropagationBackend:
    """Adapt a :class:`PreparedPropagation` payload to the C++ extension."""

    def __init__(self, prepared) -> None:
        if not cpp_backend_available():
            raise BackendUnavailable(
                "The optional C++ backend is not built. "
                "Install the native build dependencies and set PYSUQU_BUILD_NATIVE=1."
            )
        self.prepared = prepared
        if prepared.c_ops:
            raise UnsupportedBackendError(
                "The first C++ backend supports coherent ket propagation only; "
                "use backend='qutip_compiled' for c_ops."
            )

    @classmethod
    def from_prepared(cls, prepared) -> "CppPropagationBackend":
        return cls(prepared)

    def _payload(self, states: Sequence[qt.Qobj]):
        prepared = self.prepared
        static_matrix = _qobj_matrix(prepared.static_hamiltonian)
        n = static_matrix.shape[0]

        if prepared.backend == "cpp_rwa":
            return self._rwa_payload(states, static_matrix)

        operators = []
        traces = []
        lo_freqs = []
        modes = []

        for term in prepared.drive_terms:
            operator = _qobj_matrix(term.operator)
            if operator.shape != (n, n):
                raise ValueError("Every drive operator must match the static Hamiltonian dimension.")
            trace = term.trace
            raw_t_axis, raw_values, domain, lo_freq = _validate_trace(trace)
            t_axis = np.ascontiguousarray(raw_t_axis, dtype=np.float64)
            values = np.ascontiguousarray(raw_values, dtype=np.complex128)
            if values.ndim != 1 or len(values) != len(t_axis):
                raise ValueError("Each native trace must be a one-dimensional sampled array.")
            if not np.all(np.isfinite(t_axis)) or not np.all(np.isfinite(values)):
                raise ValueError("Native traces must contain only finite samples and times.")
            if not np.isfinite(lo_freq):
                raise ValueError("Native trace lo_freq must be finite.")
            operators.append(operator)
            traces.append(values)
            # ``rf_real`` samples already include the carrier.  Passing their
            # metadata LO through to the native RF mixer would apply the
            # carrier a second time, so use zero to make mode=0 interpolate the
            # real samples directly.  IQ traces retain their analytic LO.
            lo_freqs.append(
                lo_freq
                if domain == "iq_complex"
                else 0.0
            )
            if term.mode not in {"rf", "complex_envelope"}:
                raise ValueError("Each native drive term mode must be 'rf' or 'complex_envelope'.")
            modes.append(1 if term.mode == "complex_envelope" else 0)

        # The native ABI uses one regular t axis and one IQ row per drive.  A
        # prepared context already has a common solver grid; reject accidental
        # mismatches rather than silently resampling a physical waveform.
        t_axis = np.ascontiguousarray(prepared.tlist, dtype=np.float64)
        if len(t_axis) < 1:
            raise ValueError("Native propagation requires a non-empty tlist.")
        for trace in prepared.drive_terms:
            source_t = np.asarray(trace.trace.t_axis, dtype=np.float64)
            if (
                len(source_t) != len(t_axis)
                or not np.all(np.isfinite(source_t))
                or not np.allclose(source_t, t_axis, rtol=1e-12, atol=1e-12)
            ):
                raise UnsupportedBackendError(
                    "The C++ backend currently requires each trace grid to match tlist exactly."
                )

        if operators:
            controls = np.ascontiguousarray(np.stack(operators, axis=0), dtype=np.complex128)
            iq = np.ascontiguousarray(np.stack(traces, axis=0), dtype=np.complex128)
        else:
            # Keep the ABI well-defined for a static Hamiltonian.  The C++
            # kernel treats a zero-sized control axis as an ordinary no-drive
            # propagation and can then take very large adaptive steps.
            controls = np.empty((0, n, n), dtype=np.complex128)
            iq = np.empty((0, len(t_axis)), dtype=np.complex128)
        initial = []
        dims = []
        for state in states:
            if not isinstance(state, qt.Qobj) or not state.isket:
                raise UnsupportedBackendError("The C++ backend currently supports ket initial states only.")
            vector = _qobj_matrix(state)
            if vector.shape != (n, 1):
                raise ValueError("Initial state dimension does not match the Hamiltonian.")
            initial.append(vector[:, 0])
            dims.append(state.dims)
        initial_matrix = np.ascontiguousarray(np.column_stack(initial), dtype=np.complex128)
        # A single native call currently accepts one mode for all terms.  Mixed
        # RF/envelope terms are uncommon; reject them until a per-term mode ABI
        # is added instead of applying the wrong carrier convention.
        if len(set(modes)) > 1:
            raise UnsupportedBackendError("Mixed RF and complex-envelope native terms are not supported yet.")
        mode = modes[0] if modes else 1
        return (
            static_matrix,
            controls,
            iq,
            t_axis,
            initial_matrix,
            np.asarray(lo_freqs, dtype=np.float64),
            mode,
            dims,
            None,
        )

    def _rwa_payload(self, states: Sequence[qt.Qobj], static_matrix: np.ndarray):
        """Build a first-order rotating-frame/RWA payload.

        The transmon basis used by pysuqu is an energy basis, so the frame
        generator is ``n * omega_lo``.  Only nearest-neighbour drive matrix
        elements are retained; the discarded ratio is reported and guarded by
        an explicit option rather than being silently presented as exact.
        """
        prepared = self.prepared
        if len(prepared.drive_terms) != 1:
            raise UnsupportedBackendError("cpp_fast currently supports one drive term.")
        term = prepared.drive_terms[0]
        trace = term.trace
        raw_t_axis, raw_values, domain, lo_freq = _validate_trace(trace)
        if domain != "iq_complex":
            raise UnsupportedBackendError(
                "cpp_fast requires an iq_complex trace so the carrier can be removed analytically."
            )
        if term.mode != "rf":
            raise UnsupportedBackendError("cpp_fast expects mode='rf' and performs its own RWA transform.")

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
        static_eigen = np.diag(np.asarray(active_values, dtype=np.complex128))

        drive_matrix = _qobj_matrix(term.operator)
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
        max_discarded = float(
            prepared.options.extra.get("rwa_max_discarded_ratio", 0.35)
        )
        if not np.isfinite(max_discarded) or max_discarded < 0.0:
            raise ValueError("rwa_max_discarded_ratio must be a finite non-negative number.")
        if discarded_ratio > max_discarded:
            raise UnsupportedBackendError(
                "cpp_fast discarded drive terms exceed rwa_max_discarded_ratio "
                "({:.3g} > {:.3g}).".format(discarded_ratio, max_discarded)
            )
        if np.linalg.norm(diagonal_drive) > max_discarded * drive_scale:
            raise UnsupportedBackendError("cpp_fast does not support a large diagonal drive component.")

        if not np.isfinite(lo_freq):
            raise ValueError("Native IQ trace lo_freq must be finite.")
        if abs(lo_freq) <= 0.0:
            raise UnsupportedBackendError("cpp_fast requires a non-zero LO frequency.")
        t_axis = np.ascontiguousarray(raw_t_axis, dtype=np.float64)
        values = np.ascontiguousarray(raw_values, dtype=np.complex128)
        if values.ndim != 1 or len(values) != len(t_axis):
            raise ValueError("The native IQ trace must be one-dimensional.")
        if not np.all(np.isfinite(t_axis)) or not np.all(np.isfinite(values)):
            raise ValueError("Native IQ traces must contain only finite samples and times.")
        active_levels_tol = float(prepared.options.extra.get("active_levels_tol", 1e-8))
        if not np.isfinite(active_levels_tol) or active_levels_tol < 0.0:
            raise ValueError("active_levels_tol must be a finite non-negative number.")
        prepared_t = np.ascontiguousarray(prepared.tlist, dtype=np.float64)
        if len(t_axis) != len(prepared_t) or not np.allclose(
            t_axis,
            prepared_t,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise UnsupportedBackendError(
                "The C++ backend currently requires the RWA trace grid to match tlist exactly."
            )

        # For R = exp(-i n omega t), the slowly varying terms are z on the
        # upper triangle and z* on the lower triangle, each with factor 1/2.
        controls = np.ascontiguousarray(
            np.stack([0.5 * upper, 0.5 * lower], axis=0),
            dtype=np.complex128,
        )
        iq = np.ascontiguousarray(
            np.stack([values, np.conj(values)], axis=0),
            dtype=np.complex128,
        )
        h_rot = np.array(static_eigen, dtype=np.complex128, copy=True)
        frame_omega = 2.0 * np.pi * lo_freq
        for level in range(active_levels):
            h_rot[level, level] -= level * frame_omega

        initial = []
        dims = []
        t0 = float(t_axis[0])
        for state in states:
            if not isinstance(state, qt.Qobj) or not state.isket:
                raise UnsupportedBackendError("The C++ backend currently supports ket initial states only.")
            full_vector = _qobj_matrix(state)
            if full_vector.shape != (n, 1):
                raise ValueError("Initial state dimension does not match the Hamiltonian.")
            vector = np.asarray(active_vectors.conj().T @ full_vector[:, 0], dtype=np.complex128)
            omitted = np.linalg.norm(eigenvectors[:, active_levels:].conj().T @ full_vector[:, 0])
            if omitted > active_levels_tol:
                raise UnsupportedBackendError(
                    "initial state has support outside active_levels; increase active_levels."
                )
            vector = np.array(vector, dtype=np.complex128, copy=True)
            # Convert the lab-frame initial state into the rotating frame at t0.
            for level in range(active_levels):
                vector[level] *= np.exp(1j * level * frame_omega * t0)
            initial.append(vector)
            dims.append(state.dims)
        initial_matrix = np.ascontiguousarray(np.column_stack(initial), dtype=np.complex128)
        metadata = {
            "frame_frequency": lo_freq,
            "frame_omega": frame_omega,
            "t0": t0,
            "discarded_ratio": discarded_ratio,
            "basis_transform": active_vectors,
            "active_levels": active_levels,
            "full_dimension": n,
        }
        return (
            h_rot,
            controls,
            iq,
            t_axis,
            initial_matrix,
            np.zeros(2, dtype=np.float64),
            1,
            dims,
            metadata,
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

    def _normalize_output(self, values: np.ndarray) -> np.ndarray:
        """Match QuTiP's default normalize_output behavior for non-Hermitian modes."""
        if self.prepared.options.extra.get("normalize_output", True) is False:
            return values
        result = np.array(values, dtype=np.complex128, copy=True)
        if result.ndim == 2:
            norms = np.sqrt(np.sum(np.abs(result) ** 2, axis=0))
            for column, norm in enumerate(norms):
                if norm > 0:
                    result[:, column] /= norm
        return result

    def propagate(self, initial_state: qt.Qobj) -> NativePropagationResult:
        payload = self._payload([initial_state])
        final, trajectory, stats = _native_propagate(
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            payload[4],
            payload[5],
            mode=payload[6],
            atol=float(self.prepared.options.atol),
            rtol=float(self.prepared.options.rtol),
            max_steps=int(self.prepared.options.extra.get("native_max_steps", 2_000_000)),
            store_trajectory=bool(self.prepared.options.store_states),
        )
        final = _decode_complex_payload(
            final,
            (payload[0].shape[0], payload[4].shape[1]),
        )
        final = self._apply_frame(final, self.prepared.tlist[-1], payload[8])
        final = self._normalize_output(final)
        state = qt.Qobj(final[:, 0], dims=payload[7][0])
        states = []
        if trajectory is not None:
            trajectory = _decode_complex_payload(
                trajectory,
                (len(self.prepared.tlist), payload[0].shape[0], payload[4].shape[1]),
            )
            if payload[8] is not None:
                trajectory = np.stack(
                    [self._apply_frame(item, time, payload[8]) for item, time in zip(trajectory, self.prepared.tlist)],
                    axis=0,
                )
            if trajectory is not None:
                trajectory = np.stack(
                    [self._normalize_output(item) for item in trajectory],
                    axis=0,
                )
            states = [qt.Qobj(item[:, 0], dims=payload[7][0]) for item in trajectory]
        resolved_stats = dict(stats or {})
        if payload[8] is not None:
            resolved_stats.update(
                {
                    "backend": "cpp_rwa",
                    "approximation": "rwa",
                    "rwa_discarded_ratio": payload[8]["discarded_ratio"],
                    "frame_frequency": payload[8]["frame_frequency"],
                    "active_levels": payload[8]["active_levels"],
                }
            )
        else:
            resolved_stats["backend"] = "cpp"
            resolved_stats["approximation"] = "none"
        return NativePropagationResult(
            final_state=state,
            states=states,
            times=np.array(self.prepared.tlist, copy=True),
            stats=resolved_stats,
        )

    def propagate_batch(self, initial_states: Sequence[qt.Qobj]) -> BatchPropagationResult:
        payload = self._payload(initial_states)
        final, trajectory, stats = _native_propagate(
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            payload[4],
            payload[5],
            mode=payload[6],
            atol=float(self.prepared.options.atol),
            rtol=float(self.prepared.options.rtol),
            max_steps=int(self.prepared.options.extra.get("native_max_steps", 2_000_000)),
            store_trajectory=bool(self.prepared.options.store_states),
        )
        final = _decode_complex_payload(
            final,
            (payload[0].shape[0], payload[4].shape[1]),
        )
        final = self._apply_frame(final, self.prepared.tlist[-1], payload[8])
        final = self._normalize_output(final)
        dims = payload[7]
        final_states = [qt.Qobj(final[:, index], dims=dims[index]) for index in range(final.shape[1])]
        per_state_results = []
        trajectory_array = _decode_complex_payload(
            trajectory,
            (len(self.prepared.tlist), payload[0].shape[0], payload[4].shape[1]),
        )
        if trajectory_array is not None and payload[8] is not None:
            trajectory_array = np.stack(
                [self._apply_frame(item, time, payload[8]) for item, time in zip(trajectory_array, self.prepared.tlist)],
                axis=0,
            )
        if trajectory_array is not None:
            trajectory_array = np.stack(
                [self._normalize_output(item) for item in trajectory_array],
                axis=0,
            )
        if trajectory_array is not None:
            for index, state in enumerate(final_states):
                per_state_results.append(
                    NativePropagationResult(
                        final_state=state,
                        states=[
                            qt.Qobj(item[:, index], dims=dims[index])
                            for item in trajectory_array
                        ],
                        times=np.array(self.prepared.tlist, copy=True),
                        stats=dict(stats or {}),
                    )
                )
        else:
            per_state_results = [
                NativePropagationResult(
                    final_state=state,
                    states=[],
                    times=np.array(self.prepared.tlist, copy=True),
                    stats=dict(stats or {}),
                )
                for state in final_states
            ]
        resolved_stats = dict(stats or {})
        if payload[8] is not None:
            resolved_stats.update(
                {
                    "backend": "cpp_rwa",
                    "approximation": "rwa",
                    "rwa_discarded_ratio": payload[8]["discarded_ratio"],
                    "frame_frequency": payload[8]["frame_frequency"],
                    "active_levels": payload[8]["active_levels"],
                }
            )
        else:
            resolved_stats["backend"] = "cpp"
            resolved_stats["approximation"] = "none"
        for result in per_state_results:
            result.stats = dict(resolved_stats)
        return BatchPropagationResult(
            final_states=final_states,
            results=per_state_results,
            times=np.array(self.prepared.tlist, copy=True),
            stats=resolved_stats,
        )


__all__ = [
    "CppPropagationBackend",
    "NativePropagationResult",
    "cpp_backend_available",
]
