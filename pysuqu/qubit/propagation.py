"""Numerical options and result types for sampled quantum propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import qutip as qt


_INTERNAL_OPTION_KEYS = {
    "active_levels",
    "active_levels_tol",
    "backend",
    "matrix_format",
    "frame",
    "profile",
    "sparse_threshold",
    "sparse_kernel",
    "plan_cache_size",
    "block_decompose",
    "sparse_expm",
    "parallel",
    "native_max_steps",
    "rwa_max_discarded_ratio",
}

_PROFILE_METHODS = {
    "reference": None,
    "fast": "vern7",
    "fast_exact": "vern7",
    "fallback": "dop853",
    "stiff": "bdf",
}


class BackendUnavailable(RuntimeError):
    """Raised when an explicitly requested optional backend is unavailable."""


class UnsupportedBackendError(ValueError):
    """Raised when a backend cannot represent a requested propagation."""


@dataclass(frozen=True)
class DynamicCollapseRate:
    """Describe a collapse operator with an exact scalar rate trace.

    The descriptor keeps the original QuTiP coefficient for the reference
    solver while allowing native adapters to consume a sampled non-negative
    rate ``r(t)`` for the equivalent dissipator ``r(t) D[C]``.
    """

    operator: qt.Qobj
    rate_trace: Any
    qutip_coefficient: Any
    label: str = "dynamic collapse rate"

    def qutip_spec(self):
        """Return the standard QuTiP ``[operator, coefficient]`` payload."""
        return [self.operator, self.qutip_coefficient]


@dataclass(frozen=True)
class PropagationOptions:
    """Numerical options shared by the prepared propagation backends.

    ``extra`` is intentionally retained so QuTiP-specific options can be
    forwarded without making this public container depend on one QuTiP
    release.  The defaults match the historical gate-level defaults.
    """

    backend: str = "qutip"
    method: Optional[str] = None
    atol: float = 1e-8
    rtol: float = 1e-6
    nsteps: int = 5000
    store_states: bool = True
    store_final_state: Optional[bool] = None
    coefficient_order: int = 1
    rf_oversample: Optional[int] = None
    use_solver_class: bool = False
    matrix_format: str = "auto"
    profile: str = "reference"
    sparse_threshold: float = 0.6
    extra: Mapping[str, Any] = field(default_factory=dict)
    # New representation controls follow the original positional fields so
    # existing callers that construct this dataclass positionally retain their
    # previous meaning.
    # ``interaction_exact`` is an exact integrating-factor/eigenbasis
    # transform; ``auto`` tries it and falls back to the lab representation.
    frame: str = "lab"
    # Select the exact sparse arithmetic kernel independently of the physical
    # equation/frame.  ``auto`` lets the native adapter choose from measured
    # storage costs; ``standard`` and ``fused`` are reproducible overrides.
    sparse_kernel: str = "auto"
    # Runtime-only controls.  They are appended to preserve positional
    # compatibility with older callers of this dataclass.
    plan_cache_size: int = 16
    block_decompose: str = "auto"
    sparse_expm: str = "auto"
    # Execution policy only; it never changes the represented equation.
    parallel: str = "auto"

    def __post_init__(self) -> None:
        if self.coefficient_order not in (0, 1, 2, 3):
            raise ValueError("coefficient_order must be one of 0, 1, 2, or 3")
        if self.rf_oversample is not None and int(self.rf_oversample) < 1:
            raise ValueError("rf_oversample must be positive when provided")
        if (
            not np.isfinite(self.atol)
            or not np.isfinite(self.rtol)
            or self.atol <= 0
            or self.rtol <= 0
        ):
            raise ValueError("atol and rtol must be finite and positive")
        if self.nsteps < 1:
            raise ValueError("nsteps must be positive")
        if self.matrix_format not in {"auto", "dense", "csr", "banded", "fused_csr"}:
            raise ValueError(
                "matrix_format must be 'auto', 'dense', 'csr', 'banded', or 'fused_csr'"
            )
        if self.frame not in {"lab", "interaction_exact", "auto"}:
            raise ValueError("frame must be 'lab', 'interaction_exact', or 'auto'")
        if self.sparse_kernel not in {"auto", "standard", "fused"}:
            raise ValueError("sparse_kernel must be 'auto', 'standard', or 'fused'")
        if int(self.plan_cache_size) != self.plan_cache_size or self.plan_cache_size < 0:
            raise ValueError("plan_cache_size must be a non-negative integer")
        if self.block_decompose not in {"auto", "on", "off"}:
            raise ValueError("block_decompose must be 'auto', 'on', or 'off'")
        if self.sparse_expm not in {"auto", "on", "off"}:
            raise ValueError("sparse_expm must be 'auto', 'on', or 'off'")
        if self.parallel not in {"auto", "on", "off"}:
            raise ValueError("parallel must be 'auto', 'on', or 'off'")
        if self.profile not in _PROFILE_METHODS:
            raise ValueError(
                "profile must be 'reference', 'fast', 'fast_exact', 'fallback', or 'stiff'"
            )
        if not np.isfinite(self.sparse_threshold) or not (0.0 < self.sparse_threshold <= 1.0):
            raise ValueError("sparse_threshold must be finite and in (0, 1]")

    @classmethod
    def from_mapping(
        cls,
        options: Optional[Mapping[str, Any]] = None,
        *,
        backend: Optional[str] = None,
    ) -> "PropagationOptions":
        """Build options from a legacy QuTiP options mapping."""
        if isinstance(options, cls):
            if backend is None or options.backend == backend:
                return options
            return cls(
                backend=backend,
                method=options.method,
                atol=options.atol,
                rtol=options.rtol,
                nsteps=options.nsteps,
                store_states=options.store_states,
                store_final_state=options.store_final_state,
                coefficient_order=options.coefficient_order,
                rf_oversample=options.rf_oversample,
                use_solver_class=options.use_solver_class,
                matrix_format=options.matrix_format,
                frame=options.frame,
                sparse_kernel=options.sparse_kernel,
                plan_cache_size=options.plan_cache_size,
                block_decompose=options.block_decompose,
                sparse_expm=options.sparse_expm,
                parallel=options.parallel,
                profile=options.profile,
                sparse_threshold=options.sparse_threshold,
                extra=dict(options.extra),
            )

        values = dict(options or {})
        option_backend = values.pop("backend", "qutip")
        resolved_backend = backend if backend is not None else option_backend
        known = {
            "method",
            "atol",
            "rtol",
            "nsteps",
            "store_states",
            "store_final_state",
            "coefficient_order",
            "rf_oversample",
            "use_solver_class",
            "matrix_format",
            "frame",
            "sparse_kernel",
            "plan_cache_size",
            "block_decompose",
            "sparse_expm",
            "parallel",
            "profile",
            "sparse_threshold",
        }
        payload = {key: values.pop(key) for key in tuple(values) if key in known}
        extra = dict(values.pop("extra", {}) or {})
        extra.update(values)
        payload["extra"] = extra
        payload["backend"] = resolved_backend
        return cls(**payload)

    def qutip_options(self) -> Dict[str, Any]:
        """Return a QuTiP-compatible options dictionary."""
        options: Dict[str, Any] = {
            "nsteps": int(self.nsteps),
            "atol": float(self.atol),
            "rtol": float(self.rtol),
            "store_states": bool(self.store_states),
        }
        if self.store_final_state is not None:
            options["store_final_state"] = bool(self.store_final_state)
        elif not self.store_states:
            # A final-state-only propagation must still expose its result to
            # callers such as ``calculate_fidelity``.
            options["store_final_state"] = True
        method = self.method
        if method is None:
            method = _PROFILE_METHODS[self.profile]
        if method is not None:
            options["method"] = method
        options.update(
            {
                key: value
                for key, value in dict(self.extra).items()
                if key not in _INTERNAL_OPTION_KEYS
            }
        )
        return options


@dataclass(frozen=True)
class DriveTerm:
    """One operator and its sampled solver-facing trace."""

    operator: qt.Qobj
    trace: Any
    mode: str = "rf"


@dataclass
class BatchPropagationResult:
    """Result of propagating several initial kets in one prepared context."""

    final_states: List[qt.Qobj]
    results: List[Any]
    times: np.ndarray
    stats: Dict[str, Any]


def _validate_trace(trace: Any) -> Tuple[np.ndarray, np.ndarray, str, float]:
    t_axis = np.asarray(getattr(trace, "t_axis", None), dtype=np.float64)
    values = np.asarray(getattr(trace, "values", None))
    if t_axis.ndim != 1 or values.ndim != 1:
        raise ValueError("A propagation trace must contain one-dimensional t_axis and values.")
    if len(t_axis) == 0 or len(values) != len(t_axis):
        raise ValueError("Trace t_axis and values must have the same non-zero length.")
    if (
        not np.all(np.isfinite(t_axis))
        or not np.all(np.isfinite(values))
        or np.any(np.diff(t_axis) <= 0)
    ):
        raise ValueError("Trace times and values must be finite; t_axis must be strictly increasing.")
    domain = getattr(trace, "domain", None)
    if domain not in {"rf_real", "iq_complex"}:
        raise ValueError("Propagation traces must use domain 'rf_real' or 'iq_complex'.")
    lo_freq = float(getattr(trace, "lo_freq", 0.0) or 0.0)
    return t_axis, values, domain, lo_freq


def _trace_to_python_callable(trace: Any, mode: str = "rf"):
    """Create the legacy callback semantics for the reference backend."""
    t_axis, values, domain, lo_freq = _validate_trace(trace)
    if mode not in {"rf", "complex_envelope"}:
        raise ValueError("mode must be 'rf' or 'complex_envelope'.")
    if mode == "complex_envelope":
        if np.iscomplexobj(values):
            real_values = np.asarray(np.real(values), dtype=np.float64)
            imag_values = np.asarray(np.imag(values), dtype=np.float64)

            def coefficient(t: float, *args, **kwargs):
                return np.interp(t, t_axis, real_values, left=0.0, right=0.0) + 1j * np.interp(
                    t, t_axis, imag_values, left=0.0, right=0.0
                )

            return coefficient

        real_values = np.asarray(values, dtype=np.float64)

        def coefficient(t: float, *args, **kwargs):
            return np.interp(t, t_axis, real_values, left=0.0, right=0.0)

        return coefficient

    if domain == "iq_complex":
        real_values = np.asarray(np.real(values), dtype=np.float64)
        imag_values = np.asarray(np.imag(values), dtype=np.float64)

        def coefficient(t: float, *args, **kwargs):
            i_val = np.interp(t, t_axis, real_values, left=0.0, right=0.0)
            q_val = np.interp(t, t_axis, imag_values, left=0.0, right=0.0)
            phase = 2.0 * np.pi * lo_freq * t
            return i_val * np.cos(phase) - q_val * np.sin(phase)

        return coefficient

    real_values = np.asarray(np.real(values), dtype=np.float64)

    def coefficient(t: float, *args, **kwargs):
        return np.interp(t, t_axis, real_values, left=0.0, right=0.0)

    return coefficient


def _oversampled_rf_values(
    t_axis: np.ndarray,
    values: np.ndarray,
    lo_freq: float,
    oversample: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an IQ trace densely enough for a compiled RF coefficient.

    The original AWG grid can be below the RF Nyquist rate.  The dense grid is
    only an internal coefficient representation; it is never exposed as the
    user-facing AWG trace.  Linear interpolation is retained to match the
    legacy callback at the envelope level.
    """
    if len(t_axis) == 1:
        dense_t = np.array(t_axis, copy=True)
        envelope = np.asarray(values, dtype=np.complex128)
        phase = 2.0 * np.pi * lo_freq * dense_t
        return dense_t, np.real(envelope) * np.cos(phase) - np.imag(envelope) * np.sin(phase)

    source_dt = float(np.min(np.diff(t_axis)))
    if oversample is not None:
        factor = max(1, int(oversample))
        target_dt = source_dt / factor
    elif abs(lo_freq) > 0.0:
        # 32 points per carrier cycle and eight points per AWG interval gives
        # a stable linear representation for the compiled coefficient path.
        target_dt = min(source_dt / 8.0, 1.0 / (32.0 * abs(lo_freq)))
    else:
        target_dt = source_dt / 8.0

    span = float(t_axis[-1] - t_axis[0])
    count = max(2, int(np.ceil(span / target_dt)) + 1)
    # Avoid an accidental multi-gigabyte allocation for pathological inputs.
    if count > 2_000_001:
        count = 2_000_001
    dense_t = np.linspace(t_axis[0], t_axis[-1], count, dtype=np.float64)
    real_values = np.interp(
        dense_t,
        t_axis,
        np.asarray(np.real(values), dtype=np.float64),
        left=0.0,
        right=0.0,
    )
    imag_values = np.interp(
        dense_t,
        t_axis,
        np.asarray(np.imag(values), dtype=np.float64),
        left=0.0,
        right=0.0,
    )
    phase = 2.0 * np.pi * lo_freq * dense_t
    rf_values = real_values * np.cos(phase) - imag_values * np.sin(phase)
    return dense_t, np.asarray(rf_values, dtype=np.float64)


def _compiled_trace_arrays(
    trace: Any,
    *,
    mode: str,
    coefficient_order: int,
    rf_oversample: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    t_axis, values, domain, lo_freq = _validate_trace(trace)
    if mode not in {"rf", "complex_envelope"}:
        raise ValueError("mode must be 'rf' or 'complex_envelope'.")

    if mode == "complex_envelope":
        return t_axis, np.asarray(values, dtype=np.complex128)
    if domain == "iq_complex":
        return _oversampled_rf_values(t_axis, values, lo_freq, rf_oversample)
    return t_axis, np.asarray(np.real(values), dtype=np.float64)


def _exact_rf_coefficient(trace):
    """Build an exact RF coefficient for an IQ trace.

    An array coefficient can only interpolate sampled real values; using it
    for a multi-GHz carrier would silently approximate the continuous mixer.
    This callback retains the same piecewise-linear IQ interpolation and
    analytic carrier convention as the native kernel.  Callers that explicitly
    provide ``rf_oversample`` may still opt into the faster sampled coefficient
    representation through ``_compiled_trace_arrays``.
    """
    t_axis, values, domain, lo_freq = _validate_trace(trace)
    if domain != "iq_complex":
        return _make_qutip_coefficient(
            np.asarray(np.real(values), dtype=np.float64),
            t_axis,
            order=1,
        )
    real_values = np.asarray(np.real(values), dtype=np.float64)
    imag_values = np.asarray(np.imag(values), dtype=np.float64)

    def coefficient(t: float, *args, **kwargs):
        i_value = np.interp(t, t_axis, real_values, left=0.0, right=0.0)
        q_value = np.interp(t, t_axis, imag_values, left=0.0, right=0.0)
        phase = 2.0 * np.pi * lo_freq * t
        return i_value * np.cos(phase) - q_value * np.sin(phase)

    return coefficient


def _make_qutip_coefficient(values: np.ndarray, t_axis: np.ndarray, order: int):
    """Build a compiled QuTiP coefficient, with a compatible fallback."""
    coefficient_factory = getattr(qt, "coefficient", None)
    if callable(coefficient_factory):
        try:
            return coefficient_factory(values, tlist=t_axis, order=order)
        except (TypeError, ValueError):
            # Older QuTiP releases may not accept the explicit coefficient API.
            pass

    # QuTiP 4.7 accepts a callable in the list Hamiltonian format.  Keep the
    # fallback exact, even though it forfeits the compiled-coefficient speedup.
    if np.iscomplexobj(values):
        real_values = np.asarray(np.real(values), dtype=np.float64)
        imag_values = np.asarray(np.imag(values), dtype=np.float64)

        def callback(t: float, *args, **kwargs):
            return np.interp(t, t_axis, real_values, left=0.0, right=0.0) + 1j * np.interp(
                t, t_axis, imag_values, left=0.0, right=0.0
            )

        return callback

    real_values = np.asarray(values, dtype=np.float64)

    def callback(t: float, *args, **kwargs):
        return np.interp(t, t_axis, real_values, left=0.0, right=0.0)

    return callback


def native_backend_available() -> bool:
    """Return whether the optional C++ extension can be imported."""
    try:
        from .._native import propagate
    except (ImportError, ModuleNotFoundError, OSError):
        return False
    return callable(propagate)


def clear_native_plan_cache() -> None:
    """Clear the optional native adapter's process-local plan cache."""
    from .backends.cpp_backend import clear_native_plan_cache as _clear

    _clear()


def native_plan_cache_info() -> Dict[str, int]:
    """Return a snapshot of the optional native plan cache."""
    from .backends.cpp_backend import native_plan_cache_info as _info

    return _info()


class PreparedPropagation:
    """Reusable Hamiltonian/trace context for repeated initial states."""

    def __init__(
        self,
        static_hamiltonian: qt.Qobj,
        drive_terms: Iterable[DriveTerm],
        tlist: Sequence[float],
        *,
        c_ops: Optional[Sequence[Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
        args: Optional[Mapping[str, Any]] = None,
        backend: str = "qutip_compiled",
    ) -> None:
        if not isinstance(static_hamiltonian, qt.Qobj):
            raise TypeError("static_hamiltonian must be a qutip.Qobj.")
        self.static_hamiltonian = static_hamiltonian
        self.drive_terms = tuple(drive_terms)
        self.tlist = np.ascontiguousarray(np.asarray(tlist, dtype=np.float64))
        if self.tlist.ndim != 1 or len(self.tlist) == 0:
            raise ValueError("tlist must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(self.tlist)):
            raise ValueError("tlist must contain only finite values.")
        if len(self.tlist) > 1 and np.any(np.diff(self.tlist) <= 0):
            raise ValueError("tlist must be strictly increasing.")
        self.c_ops = list(c_ops or [])
        self.args = dict(args or {})
        self.options = PropagationOptions.from_mapping(options, backend=backend)
        self._requested_backend = str(self.options.backend).lower()
        self.backend = self._resolve_backend(self.options.backend)
        # ``auto`` is capability-aware: static-collapse requests may use the
        # exact native Lindblad adapter, while unsupported time-dependent
        # collapse specifications still fall back with an audit reason.
        if self._requested_backend == "auto" and self.c_ops:
            self.backend = "cpp" if native_backend_available() else "qutip_compiled"
        self._compiled_hamiltonian = None
        self._reference_hamiltonian = None
        self._qutip_solver = None
        self._qutip_solver_backend = None
        self._native_backend = None

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        name = str(backend).lower()
        if name in {"qutip", "reference", "qutip_reference"}:
            return "qutip"
        if name in {"qutip_compiled", "compiled", "fast_qutip"}:
            return "qutip_compiled"
        if name in {"auto"}:
            return "cpp" if native_backend_available() else "qutip_compiled"
        if name in {"cpp", "cpp_exact", "cpp_reference", "native"}:
            return "cpp"
        if name in {"cpp_fast", "cpp_rwa", "fast_approx", "rwa"}:
            return "cpp_rwa"
        raise ValueError(
            "Unknown propagation backend {!r}; choose qutip, qutip_compiled, auto, cpp, or cpp_fast.".format(
                backend
            )
        )

    def _qutip_hamiltonian(self, compiled: bool):
        if compiled and self._compiled_hamiltonian is not None:
            return self._compiled_hamiltonian
        if not compiled and self._reference_hamiltonian is not None:
            return self._reference_hamiltonian

        h_total: List[Any] = [self.static_hamiltonian]
        for term in self.drive_terms:
            if not isinstance(term.operator, qt.Qobj):
                raise TypeError("Every drive operator must be a qutip.Qobj.")
            if compiled:
                _trace_t_axis, _trace_values, trace_domain, trace_lo_freq = _validate_trace(
                    term.trace
                )
                use_exact_rf = (
                    term.mode == "rf"
                    and trace_domain == "iq_complex"
                    and abs(trace_lo_freq) > 0.0
                    and self.options.rf_oversample is None
                    and self.options.coefficient_order == 1
                )
                if use_exact_rf:
                    coefficient = _exact_rf_coefficient(term.trace)
                else:
                    t_axis, values = _compiled_trace_arrays(
                        term.trace,
                        mode=term.mode,
                        coefficient_order=self.options.coefficient_order,
                        rf_oversample=self.options.rf_oversample,
                    )
                    coefficient = _make_qutip_coefficient(
                        values,
                        t_axis,
                        self.options.coefficient_order,
                    )
            else:
                coefficient = _trace_to_python_callable(term.trace, mode=term.mode)
            h_total.append([term.operator, coefficient])

        if compiled and callable(getattr(qt, "QobjEvo", None)):
            try:
                self._compiled_hamiltonian = qt.QobjEvo(h_total, tlist=self.tlist)
                return self._compiled_hamiltonian
            except (TypeError, ValueError, RuntimeError):
                # A QuTiP 4.x installation can still use the list payload.
                pass
        if compiled:
            # Cache the compatible list fallback as well; otherwise an older
            # QuTiP release would rebuild every coefficient on each state run.
            self._compiled_hamiltonian = h_total
            return self._compiled_hamiltonian
        if not compiled:
            self._reference_hamiltonian = h_total
        return h_total

    def _propagate_qutip(self, initial_state: qt.Qobj, e_ops=None, *, backend=None):
        active_backend = self.backend if backend is None else backend
        hamiltonian = self._qutip_hamiltonian(active_backend == "qutip_compiled")
        options = self.options.qutip_options()
        resolved_e_ops = [] if e_ops is None else e_ops
        resolved_c_ops = [
            value.qutip_spec() if isinstance(value, DynamicCollapseRate) else value
            for value in self.c_ops
        ]
        if self.options.use_solver_class and hasattr(qt, "MESolver"):
            # Solver classes compile QobjEvo once and can be reused.  Keep this
            # opt-in because QuTiP 4.x does not expose the class API.
            if resolved_c_ops:
                if self._qutip_solver is None or self._qutip_solver_backend != active_backend:
                    self._qutip_solver = qt.MESolver(
                        hamiltonian,
                        resolved_c_ops,
                        options=options,
                    )
                    self._qutip_solver_backend = active_backend
            elif hasattr(qt, "SESolver"):
                if self._qutip_solver is None or self._qutip_solver_backend != active_backend:
                    self._qutip_solver = qt.SESolver(hamiltonian, options=options)
                    self._qutip_solver_backend = active_backend
            if self._qutip_solver is not None:
                return self._qutip_solver.run(
                    initial_state,
                    self.tlist,
                    e_ops=resolved_e_ops,
                    args=self.args,
                )

        # Avoid the mesolve dispatch layer for the common closed-system ket
        # case.  The represented ODE is identical, but sesolve avoids building
        # an unnecessary master-equation wrapper on every prepared run.
        if (
            active_backend == "qutip_compiled"
            and not resolved_c_ops
            and getattr(self.static_hamiltonian, "issuper", False) is False
            and all(not getattr(term.operator, "issuper", False) for term in self.drive_terms)
            and getattr(initial_state, "isket", False)
            and hasattr(qt, "sesolve")
            and not isinstance(hamiltonian, list)
        ):
            return qt.sesolve(
                hamiltonian,
                initial_state,
                self.tlist,
                e_ops=resolved_e_ops,
                options=options,
                args=self.args,
            )

        return qt.mesolve(
            hamiltonian,
            initial_state,
            self.tlist,
            c_ops=resolved_c_ops,
            e_ops=resolved_e_ops,
            options=options,
            args=self.args,
        )

    def _get_native_backend(self):
        """Create the native adapter once for the lifetime of this context."""
        if self._native_backend is None:
            if not native_backend_available():
                raise BackendUnavailable("The optional C++ propagation extension is unavailable.")
            if self.c_ops:
                try:
                    from .backends.cpp_backend import LindbladCppPropagationBackend
                except ImportError as exc:
                    raise UnsupportedBackendError("This native backend does not support collapse operators.") from exc

                self._native_backend = LindbladCppPropagationBackend(self)
            else:
                from .backends.cpp_backend import CppPropagationBackend

                self._native_backend = CppPropagationBackend.from_prepared(self)
        return self._native_backend

    def _auto_fallback(
        self,
        initial_state: qt.Qobj,
        *,
        e_ops=None,
        reason: str = "unsupported capability",
    ):
        """Run the compatible QuTiP backend and retain an audit trail."""
        result = self._propagate_qutip(
            initial_state,
            e_ops=e_ops,
            backend="qutip_compiled",
        )
        stats = getattr(result, "stats", None)
        if isinstance(stats, dict):
            stats.update(
                {
                    "backend_requested": "auto",
                    "backend_fallback": "qutip_compiled",
                    "backend_fallback_reason": reason,
                }
            )
        return result

    @staticmethod
    def _has_expectation_operators(e_ops) -> bool:
        """Test whether an e_ops payload is non-empty without ndarray ambiguity."""
        if e_ops is None:
            return False
        try:
            return len(e_ops) > 0
        except TypeError:
            return True

    def propagate(self, initial_state: qt.Qobj, *, e_ops=None):
        """Propagate one initial state through the prepared context."""
        if not isinstance(initial_state, qt.Qobj):
            raise TypeError("initial_state must be a qutip.Qobj.")
        if self._has_expectation_operators(e_ops) and self.backend in {"cpp", "cpp_rwa"}:
            if self._requested_backend == "auto":
                return self._auto_fallback(
                    initial_state,
                    e_ops=e_ops,
                    reason="e_ops are not supported by the native backend",
                )
            raise UnsupportedBackendError(
                "The C++ backend currently returns states only; use a QuTiP backend for e_ops."
            )
        if self.backend in {"qutip", "qutip_compiled"}:
            return self._propagate_qutip(initial_state, e_ops=e_ops)

        try:
            return self._get_native_backend().propagate(initial_state)
        except (BackendUnavailable, UnsupportedBackendError) as exc:
            if self._requested_backend != "auto":
                raise
            return self._auto_fallback(initial_state, e_ops=e_ops, reason=str(exc))

    def propagate_batch(self, initial_states: Sequence[qt.Qobj]) -> BatchPropagationResult:
        """Propagate several initial kets, using one native call when possible."""
        states = list(initial_states)
        if not states:
            return BatchPropagationResult([], [], self.tlist.copy(), {})
        if self.backend in {"cpp", "cpp_rwa"}:
            try:
                return self._get_native_backend().propagate_batch(states)
            except (BackendUnavailable, UnsupportedBackendError) as exc:
                if self._requested_backend != "auto":
                    raise
                results = [
                    self._propagate_qutip(state, backend="qutip_compiled")
                    for state in states
                ]
                finals = []
                for result in results:
                    final = getattr(result, "final_state", None)
                    if final is None:
                        result_states = getattr(result, "states", None) or []
                        if not result_states:
                            raise ValueError("QuTiP result does not contain a final state.")
                        final = result_states[-1]
                    stats = getattr(result, "stats", None)
                    if isinstance(stats, dict):
                        stats.update(
                            {
                                "backend_requested": "auto",
                                "backend_fallback": "qutip_compiled",
                                "backend_fallback_reason": str(exc),
                            }
                        )
                    finals.append(final)
                fallback_stats = {
                    "backend_requested": "auto",
                    "backend_fallback": "qutip_compiled",
                    "backend_fallback_reason": str(exc),
                }
                return BatchPropagationResult(finals, results, self.tlist.copy(), fallback_stats)

        results = [self._propagate_qutip(state) for state in states]
        finals = []
        for result in results:
            final = getattr(result, "final_state", None)
            if final is None:
                result_states = getattr(result, "states", None) or []
                if not result_states:
                    raise ValueError("QuTiP result does not contain a final state.")
                final = result_states[-1]
            finals.append(final)
        return BatchPropagationResult(finals, results, self.tlist.copy(), {})


__all__ = [
    "BackendUnavailable",
    "BatchPropagationResult",
    "DynamicCollapseRate",
    "DriveTerm",
    "PreparedPropagation",
    "PropagationOptions",
    "UnsupportedBackendError",
    "native_backend_available",
    "clear_native_plan_cache",
    "native_plan_cache_info",
]
