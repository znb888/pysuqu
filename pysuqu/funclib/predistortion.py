"""Predistortion design and waveform stages.

This module owns waveform pre-correction algorithms. Transmission-chain
primitives remain in :mod:`pysuqu.funclib.transmission`; that module imports
these names at the end for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import expm
from scipy.signal import convolve, firwin, get_window, lfilter, ss2tf

from .transmission import (
    BaseBundleTransmissionStage,
    BaseTransmissionStage,
    FIRAlignment,
    OutOfBandPolicy,
    SignalDomain,
    SignalBundle,
    SignalPlane,
    SignalTrace,
    StageDomain,
    TouchstoneInterpolation,
    _ALL_PLANES,
    _as_array,
    _normalize_cutoff,
    _normalize_filter_kind,
    _validate_sample_rate_and_cutoff,
    evaluate_touchstone_response,
    load_touchstone_network,
)


DerivativeScheme = Literal["gradient", "backward", "tustin"]
FrequencyMode = Literal["absolute", "relative"]
StateMode = Literal["reset", "retain"]


def _validate_tail_samples(value: int) -> int:
    if not np.isfinite(value) or int(value) != value or value <= 0:
        raise ValueError("tail_samples must be a positive integer.")
    return int(value)


def waveform_tail_diagnostics(tail: np.ndarray, output: np.ndarray, sample_rate: float) -> Dict[str, Any]:
    """Measure zero-input output after the recording on an explicit horizon."""
    tail = np.asarray(tail, dtype=np.complex128)
    output = np.asarray(output, dtype=np.complex128)
    energy = float(np.sum(np.abs(tail) ** 2))
    total_energy = energy + float(np.sum(np.abs(output) ** 2))
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    edge_peak = float(np.max(np.abs(tail[-min(8, len(tail)):]))) if tail.size else 0.0
    return {
        "tail_rms": float(np.sqrt(np.mean(np.abs(tail) ** 2))) if tail.size else 0.0,
        "tail_peak": float(np.max(np.abs(tail))) if tail.size else 0.0,
        "tail_energy": energy,
        "tail_energy_fraction": energy / total_energy if total_energy else 0.0,
        "tail_observation_samples": len(tail),
        "tail_observation_ns": len(tail) / float(sample_rate),
        "tail_settled": edge_peak <= max(1e-15, 1e-6 * peak),
        "tail_definition": "zero_input_output_after_recording",
    }


def apply_stage_sequence(stage: Any, trace: SignalTrace, *, state_mode: StateMode = "retain", reset_indices: Sequence[int] = ()) -> SignalTrace:
    """Propagate idle samples; optionally clear digital history at gate starts.

    The physical forward channel is not reset by this digital-stage operation.
    The indices refer to the input timeline, including any negative pre-roll.
    """
    if state_mode not in ("reset", "retain"):
        raise ValueError("sequence state_mode must be 'reset' or 'retain'.")
    boundaries = [0]
    for index in reset_indices:
        if int(index) != index or index < 0 or index > len(trace.values):
            raise ValueError("reset_indices must be integer sample boundaries inside the trace.")
        boundaries.append(int(index))
    boundaries = sorted(set(boundaries + [len(trace.values)]))
    if state_mode == "retain" or len(boundaries) < 2:
        output = stage.apply(trace)
    else:
        pieces = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            reset = getattr(stage, "reset", None)
            if callable(reset):
                reset()
            segment = trace.clone(t_axis=trace.t_axis[start:stop], values=trace.values[start:stop])
            piece = stage.apply(segment)
            pieces.append(piece.values)
        output = trace.clone(values=np.concatenate(pieces), metadata=dict(piece.metadata))
    return output.clone(metadata={
        **output.metadata, "sequence_state_mode": state_mode,
        "sequence_reset_indices": boundaries[:-1] if state_mode == "reset" else [],
    })


def _validate_stage_trace_contract(
    trace: SignalTrace,
    *,
    expected_sample_rate: Optional[float] = None,
    expected_lo_freq_ghz: Optional[float] = None,
    stage_name: str = "predistortion",
    tolerance: float = 1e-8,
) -> None:
    """Validate the sampled-grid contract required by a digital stage."""
    rate = float(trace.sample_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{stage_name} requires a finite positive trace sample_rate.")
    if expected_sample_rate is not None:
        expected = float(expected_sample_rate)
        if not np.isfinite(expected) or expected <= 0.0:
            raise ValueError(f"{stage_name} has an invalid expected sample rate.")
        if not np.isclose(rate, expected, rtol=tolerance, atol=tolerance * max(1.0, expected)):
            raise ValueError(
                f"{stage_name} was designed for sample_rate={expected:g}, "
                f"received {rate:g}."
            )
    if len(trace.t_axis) > 1:
        spacing = np.diff(np.asarray(trace.t_axis, dtype=np.float64))
        expected_spacing = 1.0 / rate
        if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
            raise ValueError(f"{stage_name} requires a strictly increasing uniform t_axis.")
        if not np.allclose(
            spacing,
            expected_spacing,
            rtol=tolerance,
            atol=tolerance * max(1.0, expected_spacing),
        ):
            raise ValueError(
                f"{stage_name} requires t_axis spacing 1/sample_rate={expected_spacing:g} ns."
            )
    if expected_lo_freq_ghz is not None and trace.domain == "iq_complex":
        expected_lo = float(expected_lo_freq_ghz)
        if not np.isfinite(expected_lo) or not np.isclose(
            float(trace.lo_freq), expected_lo, rtol=tolerance, atol=tolerance
        ):
            raise ValueError(
                f"{stage_name} was designed around lo_freq={expected_lo:g} GHz, "
                f"received {float(trace.lo_freq):g} GHz."
            )


def _validate_derivative_scheme(scheme: str) -> DerivativeScheme:
    normalized = str(scheme).lower()
    if normalized not in ("gradient", "backward", "tustin"):
        raise ValueError(
            "Unsupported derivative scheme: "
            f"{scheme!r}. Expected 'gradient', 'backward', or 'tustin'."
        )
    return normalized  # type: ignore[return-value]


def _backward_difference(values: np.ndarray, sample_period: float, initial_value: complex) -> np.ndarray:
    result = np.empty_like(values, dtype=np.complex128)
    result[0] = (values[0] - initial_value) / sample_period
    if len(values) > 1:
        result[1:] = np.diff(values) / sample_period
    return result


def _tustin_difference(values: np.ndarray, sample_period: float, initial_value: complex) -> np.ndarray:
    result = np.empty_like(values, dtype=np.complex128)
    scale = 2.0 / sample_period
    result[0] = scale * (values[0] - initial_value)
    for index in range(1, len(values)):
        result[index] = scale * (values[index] - values[index - 1]) - result[index - 1]
    return result


def _resolve_derivative_orders(
    derivative_orders: Optional[Iterable[int]],
    coefficient_count: int,
) -> tuple[int, ...]:
    if coefficient_count < 0:
        raise ValueError("coefficient_count must be non-negative.")

    if derivative_orders is None:
        return tuple(range(1, coefficient_count + 1))

    orders = tuple(int(order) for order in derivative_orders)
    if not orders and coefficient_count == 0:
        return orders
    if len(orders) != coefficient_count:
        raise ValueError(
            "derivative_orders must contain exactly one order for each coefficient."
        )
    if any(order <= 0 for order in orders):
        raise ValueError("Derivative orders must be positive integers.")
    if len(set(orders)) != len(orders):
        raise ValueError("Derivative orders must be unique.")
    return orders



def design_inverse_fir_from_touchstone(
    *,
    lo_freq: float,
    sample_rate: float,
    num_taps: int,
    file_path: Union[str, Path, None] = None,
    network: Optional["TouchstoneNetwork"] = None,
    input_port: int = 1,
    output_port: int = 2,
    interpolation: TouchstoneInterpolation = "polar",
    out_of_band: OutOfBandPolicy = "zero",
    threshold_db: float = -20.0,
    window: Optional[Union[str, Tuple[Any, ...]]] = "kaiser",
    kaiser_beta: float = 6.0,
) -> np.ndarray:
    """
    Design one complex baseband FIR inverse filter from a measured Touchstone path.

    The kernel matches the centered truncation flow commonly used for digital
    pre-distortion: FFT-grid inversion -> IFFT -> fftshift -> centered crop.

    Args:
        lo_freq: LO frequency used to map baseband bins onto RF, in GHz.
        sample_rate: Digital sample rate of the baseband waveform, in samples/ns.
        num_taps: Number of FIR taps to keep after centered truncation.
        file_path: Touchstone file to load when ``network`` is not provided.
        network: Preloaded Touchstone network. When given, ``file_path`` is ignored.
        input_port: One-based source port index for the forward path.
        output_port: One-based destination port index for the forward path.
        interpolation: Complex interpolation mode used between measured samples.
        out_of_band: Policy used when the FFT grid extends beyond the measurement span.
        threshold_db: Stop-band guard threshold. Response bins below this magnitude are
            not inverted.
        window: Optional window passed to ``scipy.signal.get_window`` after truncation.
            Use ``None`` or ``"none"`` to disable windowing.
        kaiser_beta: Beta parameter used when ``window="kaiser"``.

    Returns:
        Complex FIR kernel with length ``num_taps``.
    """
    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    if network is None:
        if file_path is None:
            raise ValueError("design_inverse_fir_from_touchstone requires either file_path or network.")
        network = load_touchstone_network(file_path)

    n_fft = max(65536, int(num_taps) * 8)
    freqs_bb = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    freqs_rf = freqs_bb + float(lo_freq)
    response = evaluate_touchstone_response(
        freqs_rf,
        network=network,
        input_port=input_port,
        output_port=output_port,
        interpolation=interpolation,
        out_of_band=out_of_band,
    )

    threshold_linear = 10 ** (threshold_db / 20.0)
    inverse_response = np.zeros_like(response, dtype=np.complex128)
    mask_pass = np.abs(response) > threshold_linear
    inverse_response[mask_pass] = 1.0 / response[mask_pass]

    impulse = np.fft.ifft(inverse_response)
    impulse_centered = np.fft.fftshift(impulse)
    center_idx = len(impulse_centered) // 2
    start = center_idx - (num_taps // 2)
    end = start + num_taps
    kernel = np.asarray(impulse_centered[start:end], dtype=np.complex128)

    if window is None or window == "none":
        return kernel

    resolved_window = ("kaiser", kaiser_beta) if window == "kaiser" else window
    window_values = np.asarray(get_window(resolved_window, len(kernel)), dtype=np.float64)
    return kernel * window_values



def compute_derivative_basis(
    values: Union[np.ndarray, Iterable[complex]],
    sample_period: float,
    derivative_orders: Sequence[int],
    *,
    scheme: DerivativeScheme = "gradient",
    initial_value: complex = 0.0 + 0.0j,
    normalize_to_peak: bool = False,
    reference_peak: Optional[float] = None,
    edge_order: int = 2,
    normalization_epsilon: float = 1e-15,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Build one or more finite-difference derivative basis waveforms.

    Args:
        values: One-dimensional waveform samples.
        sample_period: Sample spacing in ns.
        derivative_orders: Positive derivative orders to generate.
        scheme: Derivative discretization: ``"gradient"`` preserves the legacy
            finite-record behavior, ``"backward"`` uses the causal backward
            difference, and ``"tustin"`` uses the bilinear IIR differentiator.
        initial_value: Value immediately before the first sample for causal
            ``backward``/``tustin`` schemes. Higher derivatives start with zero
            state at the left boundary.
        normalize_to_peak: When ``True``, each derivative basis is rescaled so
            that its peak magnitude matches ``reference_peak``.
        reference_peak: Target peak magnitude used when ``normalize_to_peak`` is
            enabled. Defaults to the peak magnitude of ``values``.
        edge_order: Forwarded to ``numpy.gradient``. The implementation falls
            back to first-order edges for very short traces.
        normalization_epsilon: Small floor used to avoid dividing by zero while
            normalizing basis functions.

    Returns:
        A pair ``(basis_list, basis_scales)`` where ``basis_list`` is ordered to
        match ``derivative_orders`` and ``basis_scales`` stores the multiplicative
        normalization applied to each basis.
    """
    base_values = np.asarray(values, dtype=np.complex128)
    if base_values.ndim != 1:
        raise ValueError("compute_derivative_basis expects a 1D waveform.")
    if len(base_values) == 0:
        raise ValueError("compute_derivative_basis expects at least one sample.")
    sample_period = float(sample_period)
    if not np.isfinite(sample_period) or sample_period <= 0:
        raise ValueError("sample_period must be finite and positive.")
    scheme = _validate_derivative_scheme(scheme)
    initial_value = complex(initial_value)
    if not np.isfinite(initial_value.real) or not np.isfinite(initial_value.imag):
        raise ValueError("initial_value must be finite.")

    orders = tuple(int(order) for order in derivative_orders)
    if any(order <= 0 for order in orders):
        raise ValueError("derivative_orders must contain only positive integers.")
    if not orders:
        return [], np.array([], dtype=np.float64)

    resolved_edge_order = 2 if edge_order >= 2 and len(base_values) >= 3 else 1
    current = np.asarray(base_values, dtype=np.complex128)
    raw_basis_by_order: dict[int, np.ndarray] = {}
    for order in range(1, max(orders) + 1):
        if scheme == "gradient":
            current = np.gradient(current, sample_period, edge_order=resolved_edge_order)
        elif scheme == "backward":
            current = _backward_difference(
                current,
                sample_period,
                initial_value=initial_value if order == 1 else 0.0 + 0.0j,
            )
        else:
            current = _tustin_difference(
                current,
                sample_period,
                initial_value=initial_value if order == 1 else 0.0 + 0.0j,
            )
        raw_basis_by_order[order] = np.asarray(current, dtype=np.complex128)

    target_peak = float(np.max(np.abs(base_values))) if reference_peak is None else float(reference_peak)
    if not np.isfinite(target_peak) or target_peak < 0.0:
        raise ValueError("reference_peak must be finite and non-negative.")
    normalization_epsilon = float(normalization_epsilon)
    if not np.isfinite(normalization_epsilon) or normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be finite and positive.")
    basis_list: list[np.ndarray] = []
    basis_scales: list[float] = []
    for order in orders:
        raw_basis = np.asarray(raw_basis_by_order[order], dtype=np.complex128)
        scale = 1.0
        if normalize_to_peak:
            basis_peak = float(np.max(np.abs(raw_basis)))
            if basis_peak <= normalization_epsilon or target_peak <= normalization_epsilon:
                scale = 0.0
            else:
                scale = target_peak / basis_peak
        basis_list.append(np.asarray(raw_basis * scale, dtype=np.complex128))
        basis_scales.append(float(scale))

    return basis_list, np.asarray(basis_scales, dtype=np.float64)


def apply_derivative_precorrection(
    values: Union[np.ndarray, Iterable[complex]],
    sample_period: float,
    coefficients: Union[np.ndarray, Iterable[complex]],
    *,
    derivative_orders: Optional[Sequence[int]] = None,
    base_coefficient: complex = 1.0 + 0.0j,
    scheme: DerivativeScheme = "gradient",
    initial_value: complex = 0.0 + 0.0j,
    normalize_to_peak: bool = True,
    phase_rad: float = 0.0,
    edge_order: int = 2,
    normalization_epsilon: float = 1e-15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply a derivative-polynomial pre-correction to one waveform.

    The corrected waveform is

    ``x_corr = exp(i phase_rad) * (c_0*x + sum_k c_k * d^(n_k)x/dt^(n_k))``

    where the derivatives can optionally be normalized to the input waveform's
    peak magnitude.

    Args:
        values: One-dimensional input waveform.
        sample_period: Sample spacing in ns.
        coefficients: Complex derivative weights.
        base_coefficient: Complex coefficient applied to the zero-order waveform.
        scheme: Derivative discretization passed to ``compute_derivative_basis``.
        initial_value: Value immediately before the first sample for causal
            ``backward``/``tustin`` schemes.
        derivative_orders: Positive derivative orders that correspond to
            ``coefficients``. When omitted, orders ``1..N`` are used.
        normalize_to_peak: Whether to normalize each derivative basis to the
            input waveform's peak magnitude.
        phase_rad: Optional global complex phase applied after summing all terms.
        edge_order: Forwarded to ``numpy.gradient``.
        normalization_epsilon: Small floor used while normalizing derivatives.

    Returns:
        ``(corrected_values, metadata)`` where ``metadata`` stores the resolved
        derivative orders, the per-basis normalization factors, and the phase.
    """
    base_values = np.asarray(values, dtype=np.complex128)
    if base_values.ndim != 1:
        raise ValueError("apply_derivative_precorrection expects a 1D waveform.")
    if len(base_values) == 0:
        raise ValueError("apply_derivative_precorrection expects at least one sample.")
    base_coefficient = complex(base_coefficient)
    if not np.isfinite(base_coefficient.real) or not np.isfinite(base_coefficient.imag):
        raise ValueError("base_coefficient must be finite.")
    scheme = _validate_derivative_scheme(scheme)
    initial_value = complex(initial_value)
    if not np.isfinite(initial_value.real) or not np.isfinite(initial_value.imag):
        raise ValueError("initial_value must be finite.")

    coeff_array = np.asarray(tuple(coefficients), dtype=np.complex128).reshape(-1)
    if np.any(~np.isfinite(coeff_array)):
        raise ValueError("coefficients must contain finite values.")
    if not np.isfinite(float(phase_rad)):
        raise ValueError("phase_rad must be finite.")
    if not np.isfinite(float(normalization_epsilon)) or normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be finite and positive.")
    orders = _resolve_derivative_orders(derivative_orders, len(coeff_array))
    if len(coeff_array) == 0 and abs(phase_rad) < normalization_epsilon:
        metadata = {
            "derivative_orders": [],
            "basis_scales": [],
            "phase_rad": float(phase_rad),
            "normalize_to_peak": bool(normalize_to_peak),
            "scheme": scheme,
            "initial_value_real": float(np.real(initial_value)),
            "initial_value_imag": float(np.imag(initial_value)),
            "base_coefficient_real": float(np.real(base_coefficient)),
            "base_coefficient_imag": float(np.imag(base_coefficient)),
        }
        return np.asarray(base_coefficient * base_values, dtype=np.complex128), metadata

    basis_list, basis_scales = compute_derivative_basis(
        base_values,
        sample_period,
        orders,
        scheme=scheme,
        initial_value=initial_value,
        normalize_to_peak=normalize_to_peak,
        edge_order=edge_order,
        normalization_epsilon=normalization_epsilon,
    )

    corrected = np.asarray(base_coefficient * base_values, dtype=np.complex128).copy()
    for coefficient, basis in zip(coeff_array, basis_list):
        corrected += coefficient * basis

    if abs(phase_rad) >= normalization_epsilon:
        corrected *= np.exp(1j * float(phase_rad))

    metadata = {
        "derivative_orders": [int(order) for order in orders],
        "basis_scales": [float(scale) for scale in basis_scales],
        "phase_rad": float(phase_rad),
        "normalize_to_peak": bool(normalize_to_peak),
        "scheme": scheme,
        "initial_value_real": float(np.real(initial_value)),
        "initial_value_imag": float(np.imag(initial_value)),
        "base_coefficient_real": float(np.real(base_coefficient)),
        "base_coefficient_imag": float(np.imag(base_coefficient)),
        "coefficients_real": [float(np.real(value)) for value in coeff_array],
        "coefficients_imag": [float(np.imag(value)) for value in coeff_array],
    }
    return np.asarray(corrected, dtype=np.complex128), metadata


def evaluate_derivative_polynomial_response(
    frequencies: Union[float, np.ndarray, Iterable[float]],
    sample_rate: float,
    coefficients: Union[np.ndarray, Iterable[complex]],
    *,
    derivative_orders: Optional[Sequence[int]] = None,
    base_coefficient: complex = 1.0 + 0.0j,
    scheme: DerivativeScheme = "gradient",
    nyquist_tolerance: float = 1e-12,
) -> np.ndarray:
    """Evaluate a sampled derivative-polynomial operator on a frequency grid.

    Frequencies use cycles/ns (numerically GHz), and ``sample_rate`` uses
    samples/ns. ``gradient`` returns the central-difference interior symbol;
    finite-record edge handling is intentionally excluded from this LTI view.
    The Tustin symbol is singular at Nyquist, where its idealized IIR
    differentiator has an unbounded gain.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    scheme = _validate_derivative_scheme(scheme)
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    base_coefficient = complex(base_coefficient)
    if not np.isfinite(base_coefficient.real) or not np.isfinite(base_coefficient.imag):
        raise ValueError("base_coefficient must be finite.")

    coefficient_array = np.asarray(tuple(coefficients), dtype=np.complex128).reshape(-1)
    orders = _resolve_derivative_orders(derivative_orders, len(coefficient_array))
    omega = 2.0 * np.pi * frequency_array / float(sample_rate)
    z_inverse = np.exp(-1j * omega)
    if scheme == "backward":
        operator = (1.0 - z_inverse) * float(sample_rate)
    elif scheme == "tustin":
        denominator = 1.0 + z_inverse
        if np.any(np.abs(denominator) <= float(nyquist_tolerance)):
            raise ValueError("Tustin derivative response is singular at Nyquist.")
        operator = 2.0 * float(sample_rate) * (1.0 - z_inverse) / denominator
    else:
        operator = 1j * np.sin(omega) * float(sample_rate)

    response = np.full(frequency_array.shape, base_coefficient, dtype=np.complex128)
    for order, coefficient in zip(orders, coefficient_array):
        response += coefficient * operator ** int(order)
    return np.asarray(response, dtype=np.complex128)


@dataclass(frozen=True)
class DerivativePrecorrectionDesign:
    """
    Frequency-domain fit result for a derivative-polynomial pre-correction stage.

    Args:
        coefficients: Complex weights applied to the derivative basis.
        derivative_orders: Positive derivative orders associated with each
            coefficient.
        phase_rad: Global phase applied after summing the derivative terms.
        normalize_to_peak: Whether the fitted basis used per-derivative peak
            normalization.
        edge_order: Edge handling order used when forming derivatives.
        basis_scales: Normalization factors observed on the template waveform
            used during fitting.
        spectral_weights: Frequency-domain weights used by the least-squares fit.
        residual_rms: Weighted RMS residual after fitting and phase alignment.
        base_coefficient: Complex coefficient applied to the zero-order waveform.
        scheme: Discretization used to form derivative basis waveforms.
        initial_value: Value immediately before the first sample for causal schemes.
        metadata: Additional design provenance, such as an all-pole model.
    """

    coefficients: np.ndarray
    derivative_orders: tuple[int, ...] = ()
    phase_rad: float = 0.0
    normalize_to_peak: bool = True
    edge_order: int = 2
    basis_scales: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spectral_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    residual_rms: float = 0.0
    base_coefficient: complex = 1.0 + 0.0j
    scheme: DerivativeScheme = "gradient"
    initial_value: complex = 0.0 + 0.0j
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        coeff_array = np.asarray(self.coefficients, dtype=np.complex128).reshape(-1)
        if np.any(~np.isfinite(coeff_array)):
            raise ValueError("coefficients must contain finite values.")
        raw_orders = tuple(int(order) for order in self.derivative_orders)
        object.__setattr__(self, "coefficients", coeff_array)
        object.__setattr__(
            self,
            "derivative_orders",
            _resolve_derivative_orders(raw_orders if raw_orders else None, len(coeff_array)),
        )
        object.__setattr__(
            self,
            "basis_scales",
            np.asarray(self.basis_scales, dtype=np.float64).reshape(-1),
        )
        object.__setattr__(
            self,
            "spectral_weights",
            np.asarray(self.spectral_weights, dtype=np.float64).reshape(-1),
        )
        if np.any(~np.isfinite(self.basis_scales)) or np.any(~np.isfinite(self.spectral_weights)):
            raise ValueError("basis_scales and spectral_weights must be finite.")
        phase = float(self.phase_rad)
        if not np.isfinite(phase):
            raise ValueError("phase_rad must be finite.")
        object.__setattr__(self, "phase_rad", phase)
        edge_order = int(self.edge_order)
        if edge_order not in (1, 2):
            raise ValueError("edge_order must be 1 or 2.")
        object.__setattr__(self, "edge_order", edge_order)
        residual = float(self.residual_rms)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("residual_rms must be finite and non-negative.")
        object.__setattr__(self, "residual_rms", residual)
        base_coefficient = complex(self.base_coefficient)
        if not np.isfinite(base_coefficient.real) or not np.isfinite(base_coefficient.imag):
            raise ValueError("base_coefficient must be finite.")
        object.__setattr__(self, "base_coefficient", base_coefficient)
        object.__setattr__(self, "scheme", _validate_derivative_scheme(self.scheme))
        initial_value = complex(self.initial_value)
        if not np.isfinite(initial_value.real) or not np.isfinite(initial_value.imag):
            raise ValueError("initial_value must be finite.")
        object.__setattr__(self, "initial_value", initial_value)
        if self.sample_rate is not None:
            sample_rate = float(self.sample_rate)
            if not np.isfinite(sample_rate) or sample_rate <= 0.0:
                raise ValueError("sample_rate must be finite and positive when provided.")
            object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "metadata", dict(self.metadata))


def design_derivative_precorrection(
    trace: SignalTrace,
    response_values: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    derivative_orders: Sequence[int],
    scheme: DerivativeScheme = "gradient",
    initial_value: complex = 0.0 + 0.0j,
    normalize_to_peak: bool = True,
    spectral_weight_power: float = 1.0,
    ridge: float = 1e-9,
    include_global_phase: bool = True,
    edge_order: int = 2,
    normalization_epsilon: float = 1e-15,
) -> DerivativePrecorrectionDesign:
    """
    Fit a low-order derivative polynomial that approximately inverts one response.

    ``response_values`` must already be evaluated on the FFT bins associated with
    ``trace``. For IQ traces this usually means evaluating the physical response
    on ``trace.lo_freq + fftfreq(n, d=1/sample_rate)``.

    Args:
        trace: Template AWG-side IQ waveform that defines the desired output.
        response_values: One response vector or a stack of response vectors,
            shaped ``(n_freq,)`` or ``(n_paths, n_freq)``.
        derivative_orders: Positive derivative orders included in the polynomial.
        scheme: Derivative discretization used by the fitted derivative basis.
        initial_value: Value immediately before the first sample for causal schemes.
        normalize_to_peak: Whether derivative basis functions are normalized to
            the template waveform's peak magnitude.
        spectral_weight_power: Frequency-domain weighting exponent derived from
            the template spectrum magnitude. ``0`` gives uniform weighting.
        ridge: Optional Tikhonov regularization strength.
        include_global_phase: Whether to align an additional shared phase after
            solving for the derivative coefficients.
        edge_order: Forwarded to ``numpy.gradient``.
        normalization_epsilon: Small floor used for normalization and phase logic.

    Returns:
        A ``DerivativePrecorrectionDesign`` that can be fed into
        ``DerivativePrecorrectionStage``.
    """
    scheme = _validate_derivative_scheme(scheme)
    if not isinstance(trace, SignalTrace):
        raise TypeError("trace must be a SignalTrace.")
    _validate_stage_trace_contract(trace, stage_name="derivative_design")
    initial_value = complex(initial_value)
    if not np.isfinite(initial_value.real) or not np.isfinite(initial_value.imag):
        raise ValueError("initial_value must be finite.")
    if not np.isfinite(float(spectral_weight_power)) or spectral_weight_power < 0:
        raise ValueError("spectral_weight_power must be non-negative.")
    if not np.isfinite(float(ridge)) or ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not np.isfinite(float(normalization_epsilon)) or normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be finite and positive.")

    response_array = np.asarray(response_values, dtype=np.complex128)
    if response_array.ndim == 1:
        response_array = response_array[np.newaxis, :]
    if response_array.ndim != 2 or response_array.shape[1] == 0:
        raise ValueError("response_values must have shape (n_freq,) or (n_paths, n_freq).")
    if np.any(~np.isfinite(response_array)):
        raise ValueError("response_values must contain finite values.")

    orders = tuple(int(order) for order in derivative_orders)
    if not orders:
        raise ValueError("derivative_orders must not be empty.")

    basis_list, basis_scales = compute_derivative_basis(
        trace.values,
        1.0 / trace.sample_rate,
        orders,
        scheme=scheme,
        initial_value=initial_value,
        normalize_to_peak=normalize_to_peak,
        edge_order=edge_order,
        normalization_epsilon=normalization_epsilon,
    )

    fft_length = int(response_array.shape[1])
    target_fft = np.fft.fft(trace.values, n=fft_length)
    basis_ffts = [np.fft.fft(basis, n=fft_length) for basis in basis_list]

    spectral_weights = np.abs(target_fft)
    max_weight = float(np.max(spectral_weights))
    if max_weight > normalization_epsilon:
        spectral_weights = spectral_weights / max_weight
    else:
        spectral_weights = np.ones_like(spectral_weights, dtype=np.float64)
    if spectral_weight_power == 0:
        spectral_weights = np.ones_like(spectral_weights, dtype=np.float64)
    elif spectral_weight_power != 1.0:
        spectral_weights = spectral_weights ** float(spectral_weight_power)

    weighted_rows: list[np.ndarray] = []
    weighted_targets: list[np.ndarray] = []
    for response in response_array:
        columns = [spectral_weights * response * basis_fft for basis_fft in basis_ffts]
        weighted_rows.append(np.column_stack(columns))
        weighted_targets.append(spectral_weights * (target_fft - response * target_fft))

    fit_matrix = np.vstack(weighted_rows)
    fit_target = np.concatenate(weighted_targets)
    if ridge > 0:
        fit_matrix = np.vstack(
            [
                fit_matrix,
                np.sqrt(float(ridge)) * np.eye(len(orders), dtype=np.complex128),
            ]
        )
        fit_target = np.concatenate(
            [
                fit_target,
                np.zeros(len(orders), dtype=np.complex128),
            ]
        )

    coefficients, *_ = np.linalg.lstsq(fit_matrix, fit_target, rcond=None)
    predistorted, _ = apply_derivative_precorrection(
        trace.values,
        1.0 / trace.sample_rate,
        coefficients,
        derivative_orders=orders,
        scheme=scheme,
        initial_value=initial_value,
        normalize_to_peak=normalize_to_peak,
        phase_rad=0.0,
        edge_order=edge_order,
        normalization_epsilon=normalization_epsilon,
    )
    predistorted_fft = np.fft.fft(predistorted, n=fft_length)

    phase_rad = 0.0
    if include_global_phase:
        overlap = np.sum(
            (spectral_weights[np.newaxis, :] ** 2)
            * response_array
            * predistorted_fft[np.newaxis, :]
            * np.conj(target_fft)[np.newaxis, :]
        )
        if abs(overlap) > normalization_epsilon:
            phase_rad = -float(np.angle(overlap))

    corrected_fft = np.exp(1j * phase_rad) * predistorted_fft[np.newaxis, :] * response_array
    residual = (spectral_weights[np.newaxis, :] * (corrected_fft - target_fft[np.newaxis, :])).reshape(-1)
    residual_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))

    return DerivativePrecorrectionDesign(
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        derivative_orders=orders,
        scheme=scheme,
        initial_value=initial_value,
        phase_rad=float(phase_rad),
        normalize_to_peak=normalize_to_peak,
        edge_order=edge_order,
        basis_scales=np.asarray(basis_scales, dtype=np.float64),
        spectral_weights=np.asarray(spectral_weights, dtype=np.float64),
        residual_rms=residual_rms,
        sample_rate=float(trace.sample_rate),
    )



@dataclass
class FIRFilterStage(BaseTransmissionStage):
    """
    Discrete FIR stage that matches the legacy awgenerator truncation rule.

    Args:
        kernel: FIR tap weights applied by linear convolution.
        alignment: ``"leading"`` keeps the first ``N`` samples, while
            ``"centered"`` removes the linear-phase group delay.
        name: Stage label used in diagnostics and metadata.
        domain: Accepted signal domain.
        allowed_planes: Reference planes that this stage can consume.
        is_lti: Whether the stage is linear time-invariant.
        output_plane: Optional plane override applied to output traces.
    """

    kernel: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alignment: FIRAlignment = "leading"
    name: str = "fir_filter"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = _ALL_PLANES
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None
    alignment_delay_samples: Optional[int] = None
    latency_samples: int = 0
    expected_sample_rate: Optional[float] = None
    expected_lo_freq_ghz: Optional[float] = None
    tail_samples: int = 128

    def __post_init__(self) -> None:
        self.tail_samples = _validate_tail_samples(self.tail_samples)
        self.kernel = _as_array(self.kernel)
        if self.kernel.ndim != 1:
            raise ValueError("FIRFilterStage.kernel must be one-dimensional.")
        if np.any(~np.isfinite(np.asarray(self.kernel, dtype=np.complex128))):
            raise ValueError("FIRFilterStage.kernel must contain finite values.")
        if self.alignment not in ("leading", "centered"):
            raise ValueError(f"Unsupported FIRFilterStage alignment: {self.alignment}")
        if self.alignment_delay_samples is not None:
            self.alignment_delay_samples = int(self.alignment_delay_samples)
            if self.alignment_delay_samples < 0:
                raise ValueError("alignment_delay_samples must be non-negative.")
        self.latency_samples = int(self.latency_samples)
        if self.latency_samples < 0:
            raise ValueError("latency_samples must be non-negative.")
        if self.expected_sample_rate is not None:
            expected = float(self.expected_sample_rate)
            if not np.isfinite(expected) or expected <= 0.0:
                raise ValueError("expected_sample_rate must be finite and positive.")
            self.expected_sample_rate = expected
        if self.expected_lo_freq_ghz is not None:
            expected_lo = float(self.expected_lo_freq_ghz)
            if not np.isfinite(expected_lo):
                raise ValueError("expected_lo_freq_ghz must be finite when provided.")
            self.expected_lo_freq_ghz = expected_lo

    @classmethod
    def from_windowed_sinc(
        cls,
        *,
        cutoff_freq: Union[float, Tuple[float, float]],
        sample_rate: float,
        num_taps: int = 33,
        filter_kind: str = "lowpass",
        window: str = "hamming",
        scale: bool = True,
        name: str = "fir_filter",
        **kwargs: Any,
    ) -> "FIRFilterStage":
        """
        Build an FIR stage from a windowed-sinc design.

        Args:
            cutoff_freq: Cutoff frequency or band edges, in GHz.
            sample_rate: Sample rate in samples/ns.
            num_taps: Number of FIR taps.
            filter_kind: One of ``lowpass``, ``highpass``, ``bandpass``, or ``bandstop``.
            window: Window name passed to ``scipy.signal.firwin``.
            scale: Whether to scale the passband gain to unity.
            name: Stage label used in diagnostics and metadata.
            **kwargs: Forwarded to the dataclass constructor.

        Returns:
            Configured ``FIRFilterStage`` instance.
        """
        kind = _normalize_filter_kind(filter_kind)
        normalized_cutoff = _normalize_cutoff(cutoff_freq, kind)
        _validate_sample_rate_and_cutoff(sample_rate, normalized_cutoff)
        kernel = firwin(
            num_taps,
            normalized_cutoff,
            fs=sample_rate,
            pass_zero=kind,
            window=window,
            scale=scale,
        )
        return cls(kernel=kernel, name=name, **kwargs)

    @classmethod
    def lowpass(
        cls,
        *,
        cutoff_freq: float,
        sample_rate: float,
        num_taps: int = 33,
        window: str = "hamming",
        name: str = "fir_lowpass",
        **kwargs: Any,
    ) -> "FIRFilterStage":
        """
        Convenience constructor for a low-pass FIR stage.

        Args:
            cutoff_freq: Low-pass cutoff in GHz.
            sample_rate: Sample rate in samples/ns.
            num_taps: Number of FIR taps.
            window: Window name passed to ``scipy.signal.firwin``.
            name: Stage label used in diagnostics and metadata.
            **kwargs: Forwarded to ``from_windowed_sinc``.

        Returns:
            Configured low-pass ``FIRFilterStage``.
        """
        return cls.from_windowed_sinc(
            cutoff_freq=cutoff_freq,
            sample_rate=sample_rate,
            num_taps=num_taps,
            filter_kind="lowpass",
            window=window,
            name=name,
            **kwargs,
        )

    @classmethod
    def highpass(
        cls,
        *,
        cutoff_freq: float,
        sample_rate: float,
        num_taps: int = 33,
        window: str = "hamming",
        name: str = "fir_highpass",
        **kwargs: Any,
    ) -> "FIRFilterStage":
        """
        Convenience constructor for a high-pass FIR stage.

        Args:
            cutoff_freq: High-pass cutoff in GHz.
            sample_rate: Sample rate in samples/ns.
            num_taps: Number of FIR taps.
            window: Window name passed to ``scipy.signal.firwin``.
            name: Stage label used in diagnostics and metadata.
            **kwargs: Forwarded to ``from_windowed_sinc``.

        Returns:
            Configured high-pass ``FIRFilterStage``.
        """
        return cls.from_windowed_sinc(
            cutoff_freq=cutoff_freq,
            sample_rate=sample_rate,
            num_taps=num_taps,
            filter_kind="highpass",
            window=window,
            name=name,
            **kwargs,
        )

    @classmethod
    def bandpass(
        cls,
        *,
        cutoff_freq: Tuple[float, float],
        sample_rate: float,
        num_taps: int = 65,
        window: str = "hamming",
        name: str = "fir_bandpass",
        **kwargs: Any,
    ) -> "FIRFilterStage":
        """
        Convenience constructor for a band-pass FIR stage.

        Args:
            cutoff_freq: Lower and upper band edges in GHz.
            sample_rate: Sample rate in samples/ns.
            num_taps: Number of FIR taps.
            window: Window name passed to ``scipy.signal.firwin``.
            name: Stage label used in diagnostics and metadata.
            **kwargs: Forwarded to ``from_windowed_sinc``.

        Returns:
            Configured band-pass ``FIRFilterStage``.
        """
        return cls.from_windowed_sinc(
            cutoff_freq=cutoff_freq,
            sample_rate=sample_rate,
            num_taps=num_taps,
            filter_kind="bandpass",
            window=window,
            name=name,
            **kwargs,
        )

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """
        Convolve the trace with the FIR kernel.

        Args:
            trace: Input waveform to filter.

        Returns:
            Filtered trace with legacy ``leading`` or ``centered`` truncation.
        """
        self._validate_trace(trace)
        _validate_stage_trace_contract(
            trace,
            expected_sample_rate=self.expected_sample_rate,
            expected_lo_freq_ghz=self.expected_lo_freq_ghz,
            stage_name=self.name,
        )
        if len(self.kernel) == 0:
            return trace.clone(
                metadata={
                    **trace.metadata,
                    "last_stage": self.name,
                    "alignment_delay_samples": 0,
                    "latency_samples": int(self.latency_samples),
                    "tail_rms": 0.0,
                    "edge_energy_fraction": 0.0,
                }
            )

        filtered_full = convolve(trace.values, self.kernel, mode="full", method="auto")
        total_energy = float(np.sum(np.abs(filtered_full) ** 2))
        if self.alignment == "centered":
            delay = (
                (len(self.kernel) - 1) // 2
                if self.alignment_delay_samples is None
                else self.alignment_delay_samples
            )
            if delay >= len(filtered_full):
                raise ValueError("alignment_delay_samples exceeds the convolution output length.")
            if delay + len(trace.values) > len(filtered_full):
                raise ValueError(
                    "alignment_delay_samples does not leave enough samples for the trace."
                )
            filtered = filtered_full[delay : delay + len(trace.values)]
        else:
            delay = 0
            filtered = filtered_full[: len(trace.values)]
        discarded_tail = filtered_full[delay + len(trace.values):]
        tail = np.zeros(self.tail_samples, dtype=np.complex128)
        count = min(len(discarded_tail), self.tail_samples)
        tail[:count] = discarded_tail[:count]
        discarded_energy = float(np.sum(np.abs(discarded_tail) ** 2))
        head_energy = float(np.sum(np.abs(filtered_full[:delay]) ** 2))
        return self._finalize_trace(
            trace,
            filtered,
            metadata_updates={
                "last_stage": self.name,
                "kernel_length": len(self.kernel),
                "alignment": self.alignment,
                "alignment_delay_samples": int(delay),
                "latency_samples": int(self.latency_samples),
                **waveform_tail_diagnostics(tail, filtered, trace.sample_rate),
                "discarded_tail_energy": discarded_energy,
                "discarded_head_energy": head_energy,
                "edge_energy_fraction": (discarded_energy + head_energy) / total_energy if total_energy else 0.0,
            },
        )

    def describe(self) -> str:
        """Return a compact FIR summary including tap count and alignment."""
        return (
            f"{self.name}[taps={len(self.kernel)}, {self.alignment}, "
            f"latency={self.latency_samples}]"
        )


@dataclass
class DerivativePrecorrectionStage(BaseTransmissionStage):
    """
    Derivative-polynomial pre-correction applied directly in time.

    Args:
        coefficients: Complex derivative weights. Each coefficient multiplies one
            derivative basis waveform.
        base_coefficient: Complex coefficient applied to the unmodified waveform.
        derivative_orders: Positive derivative orders associated with
            ``coefficients``. When omitted, orders ``1..N`` are assumed.
        scheme: Derivative discretization used by the stage.
        initial_value: Value immediately before the first sample for causal schemes.
        normalize_to_peak: Whether to normalize each derivative basis so its peak
            magnitude matches the input waveform's peak magnitude.
        phase_rad: Optional global phase applied after summing all derivative
            terms.
        edge_order: Forwarded to ``numpy.gradient``.
        normalization_epsilon: Small floor used while normalizing basis terms.
        design_metadata: Design provenance copied from a design payload.
        name: Stage label used in diagnostics and metadata.
        domain: Accepted signal domain.
        allowed_planes: Reference planes that this stage can consume.
        is_lti: Whether the stage is linear time-invariant. Normalized derivative
            bases are convenient but not strictly shift-invariant for finite
            records, so the default is ``False``.
        output_plane: Optional plane override applied to output traces.
    """

    coefficients: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    derivative_orders: tuple[int, ...] = ()
    base_coefficient: complex = 1.0 + 0.0j
    scheme: DerivativeScheme = "gradient"
    initial_value: complex = 0.0 + 0.0j
    normalize_to_peak: bool = True
    phase_rad: float = 0.0
    edge_order: int = 2
    normalization_epsilon: float = 1e-15
    design_metadata: Dict[str, Any] = field(default_factory=dict)
    name: str = "derivative_precorrection"
    domain: StageDomain = "iq_complex"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "qubit_iq")
    is_lti: bool = False
    output_plane: Optional[SignalPlane] = None
    expected_sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.complex128).reshape(-1)
        raw_orders = tuple(int(order) for order in self.derivative_orders)
        self.derivative_orders = _resolve_derivative_orders(
            raw_orders if raw_orders else None,
            len(self.coefficients),
        )
        self.base_coefficient = complex(self.base_coefficient)
        if not np.isfinite(self.base_coefficient.real) or not np.isfinite(self.base_coefficient.imag):
            raise ValueError("DerivativePrecorrectionStage.base_coefficient must be finite.")
        self.scheme = _validate_derivative_scheme(self.scheme)
        self.initial_value = complex(self.initial_value)
        if not np.isfinite(self.initial_value.real) or not np.isfinite(self.initial_value.imag):
            raise ValueError("DerivativePrecorrectionStage.initial_value must be finite.")
        self.design_metadata = dict(self.design_metadata)
        if self.expected_sample_rate is not None:
            expected_sample_rate = float(self.expected_sample_rate)
            if not np.isfinite(expected_sample_rate) or expected_sample_rate <= 0.0:
                raise ValueError("expected_sample_rate must be finite and positive.")
            self.expected_sample_rate = expected_sample_rate
        if self.edge_order not in (1, 2):
            raise ValueError("DerivativePrecorrectionStage.edge_order must be 1 or 2.")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive.")

    @classmethod
    def from_design(
        cls,
        design: DerivativePrecorrectionDesign,
        *,
        name: str = "derivative_precorrection",
        **kwargs: Any,
    ) -> "DerivativePrecorrectionStage":
        """
        Build a stage directly from ``design_derivative_precorrection`` output.

        Args:
            design: Fitted design payload.
            name: Stage label used in diagnostics and metadata.
            **kwargs: Forwarded to the dataclass constructor.

        Returns:
            Configured ``DerivativePrecorrectionStage``.
        """
        resolved_kwargs = dict(kwargs)
        resolved_kwargs.setdefault("expected_sample_rate", design.sample_rate)
        resolved_kwargs.setdefault("name", name)
        return cls(
            coefficients=np.asarray(design.coefficients, dtype=np.complex128),
            derivative_orders=tuple(design.derivative_orders),
            base_coefficient=complex(design.base_coefficient),
            scheme=design.scheme,
            initial_value=complex(design.initial_value),
            normalize_to_peak=bool(design.normalize_to_peak),
            phase_rad=float(design.phase_rad),
            edge_order=int(design.edge_order),
            design_metadata=dict(design.metadata),
            **resolved_kwargs,
        )

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """
        Add the configured derivative basis terms to the input IQ waveform.

        Args:
            trace: Input IQ trace to pre-correct.

        Returns:
            Corrected trace with derivative metadata attached.
        """
        self._validate_trace(trace)
        _validate_stage_trace_contract(
            trace,
            expected_sample_rate=self.expected_sample_rate,
            stage_name=self.name,
        )
        corrected, metadata = apply_derivative_precorrection(
            trace.values,
            1.0 / trace.sample_rate,
            self.coefficients,
            derivative_orders=self.derivative_orders,
            base_coefficient=self.base_coefficient,
            scheme=self.scheme,
            initial_value=self.initial_value,
            normalize_to_peak=self.normalize_to_peak,
            phase_rad=self.phase_rad,
            edge_order=self.edge_order,
            normalization_epsilon=self.normalization_epsilon,
        )
        metadata["last_stage"] = self.name
        if self.design_metadata:
            metadata["predistortion_design"] = dict(self.design_metadata)
        metadata["latency_samples"] = int(self.latency_samples)
        return self._finalize_trace(trace, corrected, metadata_updates=metadata)

    @property
    def latency_samples(self) -> int:
        """Return the schedule latency encoded by the derivative design."""
        delay_ns = float(self.design_metadata.get("delay_ns", 0.0) or 0.0)
        sample_rate = self.expected_sample_rate
        if sample_rate is None and self.design_metadata.get("sample_rate") is not None:
            sample_rate = float(self.design_metadata["sample_rate"])
        if sample_rate is None or not np.isfinite(sample_rate) or sample_rate <= 0.0:
            return 0
        return max(0, int(round(delay_ns * sample_rate)))

    def describe(self) -> str:
        """Return a compact summary including derivative orders and phase."""
        orders = ",".join(str(order) for order in self.derivative_orders) or "-"
        return (
            f"{self.name}[orders={orders}, scheme={self.scheme}, "
            f"normalize={self.normalize_to_peak}, phase={self.phase_rad:.4f} rad, "
            f"base={self.base_coefficient:.4g}]"
        )



@dataclass(frozen=True)
class AllPoleDerivativeModel:
    """Continuous-time all-pole model used for exact polynomial inversion.

    The convention is

    H(s) = gain * exp(-s*delay_ns) * prod_k (1 - s/p_k)**(-1).

    Ignoring the pure delay, the inverse is the finite polynomial

    H_inv(s) = (1/gain) * prod_k (1 - s/p_k) = sum_m c_m s**m.

    Therefore a desired waveform y(t) is pre-corrected as
    u(t) = sum_m c_m d^m y(t)/dt^m. The pure delay is non-causal and is
    represented as metadata; use a timing/guard adjustment to realize it.
    """

    poles_rad_per_ns: np.ndarray
    gain: complex = 1.0 + 0.0j
    delay_ns: float = 0.0
    sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        poles = np.asarray(self.poles_rad_per_ns, dtype=np.complex128).reshape(-1)
        if poles.size == 0:
            raise ValueError("AllPoleDerivativeModel requires at least one pole.")
        if np.any(~np.isfinite(poles)):
            raise ValueError("All poles must be finite.")
        if np.any(np.abs(poles) == 0):
            raise ValueError("All poles must be non-zero.")
        if np.any(np.real(poles) >= 0):
            raise ValueError("All poles must be stable (negative real part).")
        gain = complex(self.gain)
        if not np.isfinite(gain.real) or not np.isfinite(gain.imag) or abs(gain) == 0:
            raise ValueError("gain must be finite and non-zero.")
        delay = float(self.delay_ns)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("delay_ns must be finite and non-negative.")
        object.__setattr__(self, "poles_rad_per_ns", poles)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "delay_ns", delay)
        if self.sample_rate is not None:
            sample_rate = float(self.sample_rate)
            if not np.isfinite(sample_rate) or sample_rate <= 0.0:
                raise ValueError("sample_rate must be finite and positive when provided.")
            object.__setattr__(self, "sample_rate", sample_rate)

    def inverse_polynomial(self) -> np.ndarray:
        """Return ascending coefficients c_m of the delay-free inverse."""
        polynomial = np.array([1.0 + 0.0j], dtype=np.complex128)
        for pole in self.poles_rad_per_ns:
            polynomial = np.polynomial.polynomial.polymul(
                polynomial,
                np.array([1.0 + 0.0j, -1.0 / pole], dtype=np.complex128),
            )
        return polynomial / self.gain

    @property
    def order(self) -> int:
        """Return the number of poles / highest derivative order."""
        return int(self.poles_rad_per_ns.size)

    def describe(self) -> str:
        """Return a compact model summary for notebook diagnostics."""
        return f"all_pole(order={self.order}, delay={self.delay_ns:.3f} ns)"

def design_all_pole_derivative_precorrection(
    model: AllPoleDerivativeModel,
    *,
    derivative_orders: Optional[Sequence[int]] = None,
    scheme: DerivativeScheme = "gradient",
    initial_value: complex = 0.0 + 0.0j,
    delay_mode: Literal["metadata", "reject"] = "metadata",
    delay_tolerance_ns: float = 1e-9,
) -> DerivativePrecorrectionDesign:
    """Create an exact multi-order derivative design from stable all-pole roots.

    The polynomial is generated directly from the poles, so no numerical fit is
    needed. delay_mode="metadata" keeps the pure delay in the returned metadata
    and leaves its compensation to waveform timing; delay_mode="reject" raises
    when a non-zero delay is present because its inverse is non-causal.

    ``scheme`` selects how the continuous derivative polynomial is sampled:
    ``"gradient"`` preserves the legacy finite-record implementation,
    ``"backward"`` is a causal FIR difference operator, and ``"tustin"`` is a
    causal bilinear IIR difference operator.
    """
    if not isinstance(model, AllPoleDerivativeModel):
        raise TypeError("model must be an AllPoleDerivativeModel.")
    scheme = _validate_derivative_scheme(scheme)
    initial_value = complex(initial_value)
    if not np.isfinite(initial_value.real) or not np.isfinite(initial_value.imag):
        raise ValueError("initial_value must be finite.")
    if delay_mode not in ("metadata", "reject"):
        raise ValueError("delay_mode must be 'metadata' or 'reject'.")
    if delay_tolerance_ns < 0:
        raise ValueError("delay_tolerance_ns must be non-negative.")
    if delay_mode == "reject" and model.delay_ns > delay_tolerance_ns:
        raise ValueError(
            "A pure delay has a non-causal exact inverse; compensate timing or use delay_mode='metadata'."
        )

    polynomial = model.inverse_polynomial()
    if derivative_orders is None:
        orders = tuple(range(1, model.order + 1))
    else:
        orders = tuple(int(order) for order in derivative_orders)
        if any(order <= 0 for order in orders) or len(set(orders)) != len(orders):
            raise ValueError("derivative_orders must be unique positive integers.")
        if any(order > model.order for order in orders):
            raise ValueError("derivative_orders cannot exceed the all-pole model order.")

    coefficients = np.asarray([polynomial[order] for order in orders], dtype=np.complex128)
    return DerivativePrecorrectionDesign(
        coefficients=coefficients,
        derivative_orders=orders,
        base_coefficient=complex(polynomial[0]),
        scheme=scheme,
        initial_value=initial_value,
        normalize_to_peak=False,
        phase_rad=0.0,
        edge_order=2,
        basis_scales=np.ones(len(orders), dtype=np.float64),
        spectral_weights=np.array([], dtype=np.float64),
        residual_rms=0.0,
        metadata={
            "design_method": "all_pole_polynomial",
            "poles_rad_per_ns": model.poles_rad_per_ns.copy(),
            "gain": model.gain,
            "delay_ns": model.delay_ns,
            "delay_mode": delay_mode,
            "scheme": scheme,
            "initial_value": initial_value,
            "inverse_polynomial": polynomial.copy(),
            "sample_rate": model.sample_rate,
        },
        sample_rate=model.sample_rate,
    )


@dataclass(frozen=True)
class ZOHDiscreteModel:
    """Sampled state-space model obtained with zero-order-hold discretization.

    The state update is ``x[n+1] = A @ x[n] + B @ u[n]`` and the output is
    ``y[n] = C @ x[n] + D @ u[n]``.  ``sample_period`` is in ns, matching the
    waveform conventions used by :mod:`pysuqu`.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    sample_period: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrices = [np.asarray(value, dtype=np.complex128) for value in (self.A, self.B, self.C, self.D)]
        A, B, C, D = matrices
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("ZOHDiscreteModel.A must be square.")
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError("ZOHDiscreteModel.B must have the same state row count as A.")
        if C.ndim != 2 or C.shape[1] != A.shape[1]:
            raise ValueError("ZOHDiscreteModel.C must have the same state column count as A.")
        if D.ndim != 2 or D.shape != (C.shape[0], B.shape[1]):
            raise ValueError("ZOHDiscreteModel.D must have shape (n_outputs, n_inputs).")
        if any(np.any(~np.isfinite(matrix)) for matrix in (A, B, C, D)):
            raise ValueError("ZOHDiscreteModel matrices must contain finite values.")
        dt = float(self.sample_period)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("sample_period must be finite and positive.")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "sample_period", dt)
        metadata = dict(self.metadata)
        if "latency_samples" in metadata:
            latency = int(metadata["latency_samples"])
            if latency < 0:
                raise ValueError("latency_samples must be non-negative.")
            metadata["latency_samples"] = latency
        object.__setattr__(self, "metadata", metadata)

    @property
    def n_states(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_inputs(self) -> int:
        return int(self.B.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.C.shape[0])

    def response(self, frequencies: Union[float, np.ndarray, Iterable[float]], sample_rate: Optional[float] = None) -> np.ndarray:
        """Evaluate the discrete transfer matrix on frequencies in cycles/ns."""
        resolved_rate = 1.0 / self.sample_period if sample_rate is None else float(sample_rate)
        return evaluate_discrete_state_space_response(self, frequencies, resolved_rate)

    @property
    def sample_rate(self) -> float:
        """Return the design sample rate in samples/ns."""
        return 1.0 / self.sample_period

    @property
    def frequency_mode(self) -> FrequencyMode:
        """Return the frequency coordinate recorded by the design metadata."""
        mode = str(self.metadata.get("frequency_mode", "relative")).lower()
        return mode if mode in ("absolute", "relative") else "relative"  # type: ignore[return-value]


def discretize_zoh_state_space(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    sample_period: float,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ZOHDiscreteModel:
    """Discretize a continuous state-space model with an exact ZOH.

    The realification is intentional: some supported SciPy releases produce
    incorrect results for multi-dimensional matrices that are merely complex
    views of real values.  Realifying the complex system before the augmented
    matrix exponential is numerically stable and preserves complex I/Q inputs.
    """
    matrices = [np.asarray(value, dtype=np.complex128) for value in (A, B, C, D)]
    A_c, B_c, C_c, D_c = matrices
    if A_c.ndim != 2 or B_c.ndim != 2 or C_c.ndim != 2 or D_c.ndim != 2:
        raise ValueError("A, B, C, and D must be two-dimensional matrices.")
    n_states = A_c.shape[0]
    n_inputs = B_c.shape[1]
    if A_c.shape != (n_states, n_states) or B_c.shape[0] != n_states:
        raise ValueError("A and B have incompatible state dimensions.")
    if C_c.shape[1] != n_states or D_c.shape != (C_c.shape[0], n_inputs):
        raise ValueError("C and D have incompatible dimensions.")
    dt = float(sample_period)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("sample_period must be finite and positive.")
    if any(np.any(~np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("State-space matrices must be finite.")

    # x=[xr,xi], u=[ur,ui] representation of x'=A x+B u.
    ar = np.block([[A_c.real, -A_c.imag], [A_c.imag, A_c.real]])
    br = np.block([[B_c.real, -B_c.imag], [B_c.imag, B_c.real]])
    augmented = np.zeros((2 * n_states + 2 * n_inputs, 2 * n_states + 2 * n_inputs), dtype=np.float64)
    augmented[: 2 * n_states, : 2 * n_states] = ar
    augmented[: 2 * n_states, 2 * n_states :] = br
    exponential = expm(augmented * dt)
    ad_r = exponential[: 2 * n_states, : 2 * n_states]
    bd_r = exponential[: 2 * n_states, 2 * n_states :]
    ad = ad_r[:n_states, :n_states] + 1j * ad_r[n_states:, :n_states]
    bd = bd_r[:n_states, :n_inputs] + 1j * bd_r[n_states:, :n_inputs]
    return ZOHDiscreteModel(
        ad,
        bd,
        C_c,
        D_c,
        dt,
        metadata=metadata or {},
    )


def discretize_zoh_transfer_function(
    numerator: Sequence[complex],
    denominator: Sequence[complex],
    sample_period: float,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ZOHDiscreteModel:
    """Convert a continuous SISO transfer function to a ZOH discrete model."""
    from scipy.signal import tf2ss

    A, B, C, D = tf2ss(np.asarray(numerator), np.asarray(denominator))
    return discretize_zoh_state_space(A, B, C, D, sample_period, metadata=metadata)


def propagate_zoh_discrete(
    model: ZOHDiscreteModel,
    values: Union[np.ndarray, Sequence[complex]],
    *,
    initial_state: Optional[np.ndarray] = None,
    return_state: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Propagate a sampled input through a discrete state-space model."""
    if not isinstance(model, ZOHDiscreteModel):
        raise TypeError("model must be a ZOHDiscreteModel.")
    inputs = np.asarray(values, dtype=np.complex128)
    was_vector = inputs.ndim == 1
    if was_vector:
        inputs = inputs[:, np.newaxis]
    if inputs.ndim != 2 or inputs.shape[1] != model.n_inputs:
        raise ValueError(f"values must have shape (n_samples, {model.n_inputs}).")
    if np.any(~np.isfinite(inputs)):
        raise ValueError("values must contain finite samples.")
    if initial_state is None:
        state = np.zeros(model.n_states, dtype=np.complex128)
    else:
        state = np.asarray(initial_state, dtype=np.complex128).reshape(-1)
        if state.size != model.n_states:
            raise ValueError("initial_state has the wrong size.")
        if np.any(~np.isfinite(state)):
            raise ValueError("initial_state must contain finite values.")
    outputs = np.empty((inputs.shape[0], model.n_outputs), dtype=np.complex128)
    for index, input_row in enumerate(inputs):
        outputs[index] = model.C @ state + model.D @ input_row
        state = model.A @ state + model.B @ input_row
    result = outputs[:, 0] if was_vector and model.n_outputs == 1 else outputs
    return (result, state) if return_state else result


def evaluate_discrete_state_space_response(
    model: ZOHDiscreteModel,
    frequencies: Union[float, np.ndarray, Iterable[float]],
    sample_rate: float,
) -> np.ndarray:
    """Evaluate ``C (zI-A)^-1 B + D`` for a frequency grid."""
    if not isinstance(model, ZOHDiscreteModel):
        raise TypeError("model must be a ZOHDiscreteModel.")
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive.")
    frequencies_array = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(frequencies_array)):
        raise ValueError("frequencies must contain finite values.")
    result = np.empty((frequencies_array.size, model.n_outputs, model.n_inputs), dtype=np.complex128)
    identity = np.eye(model.n_states, dtype=np.complex128)
    for index, frequency in enumerate(frequencies_array):
        z = np.exp(2j * np.pi * float(frequency) / rate)
        if model.n_states:
            resolvent = np.linalg.solve(z * identity - model.A, model.B)
            result[index] = model.C @ resolvent + model.D
        else:
            result[index] = model.D
    if np.asarray(frequencies).ndim == 0:
        return result[0]
    return result


@dataclass(frozen=True)
class StrictDiscreteInverseDesign:
    """SISO causal/stable inverse represented as a discrete IIR transfer function."""

    numerator: np.ndarray
    denominator: np.ndarray
    delay_samples: int = 0
    exact_inverse_available: bool = True
    minimum_phase: bool = True
    stable_inverse: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    sample_rate: Optional[float] = None
    frequency_reference_ghz: Optional[float] = None

    def __post_init__(self) -> None:
        numerator = np.asarray(self.numerator, dtype=np.complex128).reshape(-1)
        denominator = np.asarray(self.denominator, dtype=np.complex128).reshape(-1)
        if numerator.size == 0 or denominator.size == 0:
            raise ValueError("An inverse transfer function needs non-empty numerator and denominator.")
        scale = denominator[0]
        if abs(scale) == 0.0:
            raise ValueError("Inverse denominator must have a non-zero leading coefficient.")
        if not np.isfinite(scale.real) or not np.isfinite(scale.imag):
            raise ValueError("Inverse denominator must have a finite leading coefficient.")
        normalized_numerator = numerator / scale
        normalized_denominator = denominator / scale
        object.__setattr__(self, "numerator", normalized_numerator)
        object.__setattr__(self, "denominator", normalized_denominator)
        object.__setattr__(self, "delay_samples", int(self.delay_samples))
        if self.delay_samples < 0:
            raise ValueError("delay_samples must be non-negative.")
        if np.any(~np.isfinite(normalized_numerator)) or np.any(~np.isfinite(normalized_denominator)):
            raise ValueError("Inverse coefficients must be finite.")
        if self.sample_rate is not None:
            sample_rate = float(self.sample_rate)
            if not np.isfinite(sample_rate) or sample_rate <= 0.0:
                raise ValueError("sample_rate must be finite and positive when provided.")
            object.__setattr__(self, "sample_rate", sample_rate)
        if self.frequency_reference_ghz is not None:
            reference = float(self.frequency_reference_ghz)
            if not np.isfinite(reference):
                raise ValueError("frequency_reference_ghz must be finite when provided.")
            object.__setattr__(self, "frequency_reference_ghz", reference)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def apply(self, values: Union[np.ndarray, Sequence[complex]], *, zi: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the inverse IIR using causal ``scipy.signal.lfilter``."""
        array = np.asarray(values, dtype=np.complex128)
        if array.ndim != 1:
            raise ValueError("StrictDiscreteInverseDesign.apply expects a 1D sample vector.")
        if np.any(~np.isfinite(array)):
            raise ValueError("StrictDiscreteInverseDesign.apply requires finite samples.")
        if zi is None:
            output = lfilter(self.numerator, self.denominator, array)
        else:
            state = np.asarray(zi, dtype=np.complex128).reshape(-1)
            if state.size != self.state_size:
                raise ValueError("zi has the wrong size for the inverse filter.")
            if np.any(~np.isfinite(state)):
                raise ValueError("zi must contain finite values.")
            output, _ = lfilter(self.numerator, self.denominator, array, zi=state)
        return np.asarray(output, dtype=np.complex128)

    @property
    def state_size(self) -> int:
        """Return the number of delay-state entries required by ``lfilter``."""
        return max(self.numerator.size, self.denominator.size) - 1

    def apply_with_state(
        self,
        values: Union[np.ndarray, Sequence[complex]],
        *,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the inverse and return its final ``lfilter`` state."""
        array = np.asarray(values, dtype=np.complex128)
        if array.ndim != 1:
            raise ValueError("StrictDiscreteInverseDesign.apply_with_state expects a 1D sample vector.")
        if np.any(~np.isfinite(array)):
            raise ValueError("StrictDiscreteInverseDesign.apply_with_state requires finite samples.")
        if self.state_size == 0:
            if state is not None:
                supplied_state = np.asarray(state, dtype=np.complex128).reshape(-1)
                if supplied_state.size != 0:
                    raise ValueError("state has the wrong size for the inverse filter.")
                if np.any(~np.isfinite(supplied_state)):
                    raise ValueError("state must contain finite values.")
            return self.apply(array), np.array([], dtype=np.complex128)
        if state is None:
            initial = np.zeros(self.state_size, dtype=np.complex128)
        else:
            initial = np.asarray(state, dtype=np.complex128).reshape(-1)
            if initial.size != self.state_size:
                raise ValueError("state has the wrong size for the inverse filter.")
            if np.any(~np.isfinite(initial)):
                raise ValueError("state must contain finite values.")
        output, final_state = lfilter(
            self.numerator,
            self.denominator,
            array,
            zi=initial,
        )
        return np.asarray(output, dtype=np.complex128), np.asarray(final_state, dtype=np.complex128)


@dataclass(kw_only=True)
class DiscreteStateSpaceStage(BaseTransmissionStage):
    """Transmission stage for a sampled SISO state-space model."""

    model: ZOHDiscreteModel
    name: str = "zoh_discrete"
    retain_state: bool = False
    state: Optional[np.ndarray] = None
    tail_samples: int = 128
    expected_sample_rate: Optional[float] = None
    expected_lo_freq_ghz: Optional[float] = None
    domain: StageDomain = "iq_complex"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "qubit_iq")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ZOHDiscreteModel):
            raise TypeError("model must be a ZOHDiscreteModel.")
        self.tail_samples = _validate_tail_samples(self.tail_samples)
        if self.model.n_inputs != 1 or self.model.n_outputs != 1:
            raise ValueError("DiscreteStateSpaceStage currently supports SISO models only.")
        if self.expected_lo_freq_ghz is None and self.model.metadata.get("lo_freq_ghz") is not None:
            self.expected_lo_freq_ghz = float(self.model.metadata["lo_freq_ghz"])
        if self.expected_lo_freq_ghz is not None and not np.isfinite(float(self.expected_lo_freq_ghz)):
            raise ValueError("expected_lo_freq_ghz must be finite when provided.")
        if self.expected_sample_rate is not None:
            expected_rate = float(self.expected_sample_rate)
            if not np.isfinite(expected_rate) or expected_rate <= 0.0:
                raise ValueError("expected_sample_rate must be finite and positive when provided.")
            self.expected_sample_rate = expected_rate
        # A retained state makes the stage history-dependent and therefore not LTI.
        self.is_lti = not bool(self.retain_state)
        if self.state is not None:
            self.state = np.asarray(self.state, dtype=np.complex128).reshape(-1)
            if self.state.size != self.model.n_states:
                raise ValueError("state has the wrong size for model.")

    def apply(self, trace: SignalTrace) -> SignalTrace:
        self._validate_trace(trace)
        _validate_stage_trace_contract(
            trace,
            expected_sample_rate=(
                self.model.sample_rate
                if self.expected_sample_rate is None
                else self.expected_sample_rate
            ),
            expected_lo_freq_ghz=self.expected_lo_freq_ghz,
            stage_name=self.name,
        )
        output, final_state = propagate_zoh_discrete(
            self.model,
            trace.values,
            initial_state=self.state if self.retain_state else None,
            return_state=True,
        )
        if self.retain_state:
            self.state = np.asarray(final_state, dtype=np.complex128)
        tail = propagate_zoh_discrete(self.model, np.zeros(self.tail_samples), initial_state=final_state)
        return self._finalize_trace(
            trace,
            output,
            metadata_updates={
                "last_stage": self.name,
                "sample_period": self.model.sample_period,
                "zoh_state_count": self.model.n_states,
                "state_mode": "retain" if self.retain_state else "reset",
                "latency_samples": int(self.model.metadata.get("latency_samples", 0)),
                "final_state_norm": float(np.linalg.norm(final_state)),
                **waveform_tail_diagnostics(tail, output, trace.sample_rate),
            },
        )

    @property
    def latency_samples(self) -> int:
        """Return the modeled scheduling latency in samples."""
        return int(self.model.metadata.get("latency_samples", 0))

    def reset(self) -> None:
        """Clear the retained state before a new trace or sequence."""
        self.state = None

    def snapshot_state(self) -> Optional[np.ndarray]:
        """Return a copy of the current state for deterministic replay."""
        return None if self.state is None else np.asarray(self.state, dtype=np.complex128).copy()

    def restore_state(self, state: Optional[np.ndarray]) -> None:
        """Restore a previously captured state."""
        if state is None:
            self.state = None
            return
        resolved = np.asarray(state, dtype=np.complex128).reshape(-1)
        if resolved.size != self.model.n_states:
            raise ValueError("state has the wrong size for model.")
        self.state = resolved.copy()

    def fresh(self) -> "DiscreteStateSpaceStage":
        """Return an equivalent stage with a reset state."""
        return DiscreteStateSpaceStage(
            model=self.model,
            tail_samples=self.tail_samples,
            name=self.name,
            retain_state=self.retain_state,
            state=None,
            expected_sample_rate=self.expected_sample_rate,
            expected_lo_freq_ghz=self.expected_lo_freq_ghz,
            domain=self.domain,
            allowed_planes=self.allowed_planes,
            output_plane=self.output_plane,
        )

    def describe(self) -> str:
        return f"{self.name}[states={self.model.n_states}, {'retain' if self.retain_state else 'reset'}]"


@dataclass(kw_only=True)
class StrictDiscreteInverseStage(BaseTransmissionStage):
    """Transmission stage applying a designed causal discrete inverse."""

    design: StrictDiscreteInverseDesign
    retain_state: bool = False
    state: Optional[np.ndarray] = None
    tail_samples: int = 128
    name: str = "strict_discrete_inverse"
    domain: StageDomain = "iq_complex"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "qubit_iq")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        if not isinstance(self.design, StrictDiscreteInverseDesign):
            raise TypeError("design must be a StrictDiscreteInverseDesign.")
        self.tail_samples = _validate_tail_samples(self.tail_samples)
        self.is_lti = not bool(self.retain_state)
        if self.state is not None:
            resolved = np.asarray(self.state, dtype=np.complex128).reshape(-1)
            if resolved.size != self.design.state_size:
                raise ValueError("state has the wrong size for the inverse filter.")
            self.state = resolved

    def apply(self, trace: SignalTrace) -> SignalTrace:
        self._validate_trace(trace)
        _validate_stage_trace_contract(
            trace,
            expected_sample_rate=self.design.sample_rate,
            expected_lo_freq_ghz=self.design.frequency_reference_ghz,
            stage_name=self.name,
        )
        state_before = self.snapshot_state()
        if self.retain_state:
            corrected, final_state = self.design.apply_with_state(
                trace.values,
                state=self.state,
            )
            self.state = final_state
        else:
            # Keep reset semantics while still exposing the IIR terminal state
            # for tail diagnostics. The state is deliberately not retained.
            corrected, final_state = self.design.apply_with_state(
                trace.values,
                state=None,
            )
        tail, _ = self.design.apply_with_state(np.zeros(self.tail_samples), state=final_state)
        state_transition_norm = None
        if state_before is not None or final_state.size:
            before = (
                np.zeros_like(final_state)
                if state_before is None
                else np.asarray(state_before, dtype=np.complex128)
            )
            state_transition_norm = float(np.linalg.norm(final_state - before))
        return self._finalize_trace(
            trace,
            corrected,
            metadata_updates={
                "last_stage": self.name,
                "delay_samples": self.design.delay_samples,
                "exact_inverse_available": self.design.exact_inverse_available,
                "inverse_diagnostics": dict(self.design.diagnostics),
                "state_mode": "retain" if self.retain_state else "reset",
                "latency_samples": int(self.design.delay_samples),
                "state_transition_norm": state_transition_norm,
                "final_state_norm": float(np.linalg.norm(final_state)),
                **waveform_tail_diagnostics(tail, corrected, trace.sample_rate),
            },
        )

    @property
    def latency_samples(self) -> int:
        return int(self.design.delay_samples)

    def reset(self) -> None:
        """Clear the retained IIR state."""
        self.state = None

    def snapshot_state(self) -> Optional[np.ndarray]:
        return None if self.state is None else np.asarray(self.state, dtype=np.complex128).copy()

    def restore_state(self, state: Optional[np.ndarray]) -> None:
        if state is None:
            self.state = None
            return
        resolved = np.asarray(state, dtype=np.complex128).reshape(-1)
        if resolved.size != self.design.state_size:
            raise ValueError("state has the wrong size for the inverse filter.")
        self.state = resolved.copy()

    def fresh(self) -> "StrictDiscreteInverseStage":
        """Return an equivalent stage with a reset state."""
        return StrictDiscreteInverseStage(
            design=self.design,
            tail_samples=self.tail_samples,
            retain_state=self.retain_state,
            state=None,
            name=self.name,
            domain=self.domain,
            allowed_planes=self.allowed_planes,
            output_plane=self.output_plane,
        )

    def describe(self) -> str:
        return f"{self.name}[delay={self.design.delay_samples}]"


@dataclass(frozen=True)
class MIMODiscreteInverseDesign:
    """Regularized inverse realization for a discrete MIMO state-space model."""

    model: ZOHDiscreteModel
    regularization: float = 0.0
    exact_inverse_available: bool = False
    stable_inverse: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, ZOHDiscreteModel):
            raise TypeError("model must be a ZOHDiscreteModel.")
        regularization = float(self.regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        object.__setattr__(self, "regularization", regularization)
        object.__setattr__(self, "exact_inverse_available", bool(self.exact_inverse_available))
        object.__setattr__(self, "stable_inverse", bool(self.stable_inverse))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def sample_rate(self) -> float:
        return self.model.sample_rate

    @property
    def frequency_reference_ghz(self) -> Optional[float]:
        value = self.model.metadata.get("frequency_reference_ghz", self.model.metadata.get("lo_freq_ghz"))
        return None if value is None else float(value)

    @property
    def latency_samples(self) -> int:
        return int(self.model.metadata.get("latency_samples", 0))


def _regularized_right_inverse(matrix: np.ndarray, regularization: float) -> np.ndarray:
    """Return a stable Tikhonov right/left pseudoinverse for one matrix."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    rows, columns = matrix.shape
    if rows >= columns:
        gram = matrix.conj().T @ matrix
        return np.linalg.solve(gram + regularization * np.eye(columns), matrix.conj().T)
    gram = matrix @ matrix.conj().T
    return matrix.conj().T @ np.linalg.solve(gram + regularization * np.eye(rows), np.eye(rows))


def check_mimo_discrete_inverse_realizability(
    model: ZOHDiscreteModel,
    *,
    regularization: float = 0.0,
    condition_limit: float = 1e8,
    stability_tolerance: float = 1e-8,
    frequencies: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Check direct-feedthrough invertibility and closed-loop inverse stability."""
    if not isinstance(model, ZOHDiscreteModel):
        raise TypeError("model must be a ZOHDiscreteModel.")
    regularization = float(regularization)
    condition_limit = float(condition_limit)
    stability_tolerance = float(stability_tolerance)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative.")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one.")
    if (
        not np.isfinite(stability_tolerance)
        or stability_tolerance < 0.0
        or stability_tolerance >= 1.0
    ):
        raise ValueError("stability_tolerance must be finite and in [0, 1).")
    direct = np.asarray(model.D, dtype=np.complex128)
    singular_values = np.linalg.svd(direct, compute_uv=False)
    min_singular = float(np.min(singular_values)) if singular_values.size else 0.0
    max_singular = float(np.max(singular_values)) if singular_values.size else 0.0
    condition = float(np.linalg.cond(direct)) if min_singular > 0.0 else float("inf")
    if frequencies is None:
        frequencies_array = np.asarray(
            model.metadata.get("fit_frequencies_ghz", [0.0]), dtype=np.float64
        ).reshape(-1)
        reference = model.metadata.get("frequency_reference_ghz")
        if reference is not None and frequencies_array.size:
            # Fit frequencies are stored in the model's relative coordinate.
            frequencies_array = frequencies_array
    else:
        frequencies_array = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(frequencies_array)):
        raise ValueError("frequencies must contain finite values.")
    frequency_conditions: list[float] = []
    frequency_regularization_candidates: list[float] = []
    if frequencies_array.size:
        responses = model.response(frequencies_array, sample_rate=model.sample_rate)
        for response in np.asarray(responses):
            try:
                singular = np.linalg.svd(response, compute_uv=False)
                frequency_conditions.append(float(np.linalg.cond(response)))
                if singular.size:
                    frequency_regularization_candidates.append(
                        max(float(singular[0]) ** 2, 1e-30) / condition_limit
                    )
            except np.linalg.LinAlgError:
                frequency_conditions.append(float("inf"))
    effective_regularization = max(0.0, regularization)
    max_frequency_condition = (
        float(np.max(frequency_conditions)) if frequency_conditions else condition
    )
    if effective_regularization == 0.0 and (
        not np.isfinite(condition)
        or condition > condition_limit
        or not np.isfinite(max_frequency_condition)
        or max_frequency_condition > condition_limit
    ):
        effective_regularization = max(
            max(max_singular**2, 1e-30) / condition_limit,
            max(frequency_regularization_candidates, default=0.0),
        )
    direct_inverse = _regularized_right_inverse(direct, effective_regularization)
    inverse_a = model.A - model.B @ direct_inverse @ model.C
    inverse_poles = np.linalg.eigvals(inverse_a)
    stable = bool(
        inverse_poles.size == 0
        or np.max(np.abs(inverse_poles)) < 1.0 - stability_tolerance
    )
    square_full_rank = bool(
        direct.shape[0] == direct.shape[1]
        and min_singular > max(stability_tolerance, 1e-14)
        and np.isfinite(condition)
        and condition <= condition_limit
    )
    forward_poles = np.linalg.eigvals(model.A)
    forward_stable = bool(not forward_poles.size or np.max(np.abs(forward_poles)) < 1.0 - stability_tolerance)
    exact = bool(square_full_rank and effective_regularization == 0.0 and stable and forward_stable)
    reasons = []
    if direct.shape[0] != direct.shape[1]:
        reasons.append("nonsquare_transfer")
    if min_singular <= max(stability_tolerance, 1e-14):
        reasons.append("singular_feedthrough")
    if condition > condition_limit or max_frequency_condition > condition_limit:
        reasons.append("condition_limit_exceeded")
    if not stable:
        reasons.append("unstable_inverse")
    if not forward_stable:
        reasons.append("unstable_forward")
    if effective_regularization:
        reasons.append("regularized_approximation")
    reason = "ok" if exact else ";".join(reasons)
    return {
        "direct_condition_number": condition,
        "forward_stable": forward_stable,
        "frequency_condition_number_max": max_frequency_condition,
        "frequency_condition_numbers": frequency_conditions,
        "direct_min_singular_value": min_singular,
        "direct_max_singular_value": max_singular,
        "regularization": float(effective_regularization),
        "inverse_poles": inverse_poles,
        "inverse_spectral_radius": (
            float(np.max(np.abs(inverse_poles))) if inverse_poles.size else 0.0
        ),
        "stable": stable,
        "square_full_rank": square_full_rank,
        "exact_inverse_available": exact,
        "reason": reason,
        "reason_codes": reasons,
        "input_count": model.n_inputs,
        "output_count": model.n_outputs,
    }


def design_mimo_discrete_inverse(
    model: ZOHDiscreteModel,
    *,
    regularization: float = 0.0,
    condition_limit: float = 1e8,
    stability_tolerance: float = 1e-8,
    frequencies: Optional[Sequence[float]] = None,
    allow_approximate: bool = False,
) -> MIMODiscreteInverseDesign:
    """Construct a causal regularized MIMO inverse realization.

    For ``y=Cx+Du`` the inverse realization uses ``D^+`` and
    ``A_i=A-BD^+C``.  A nonzero regularizer is reported explicitly and is
    treated as approximate; callers can set ``allow_approximate=False`` to
    require a square, well-conditioned, stable exact inverse.
    """
    diagnostics = check_mimo_discrete_inverse_realizability(
        model,
        regularization=regularization,
        condition_limit=condition_limit,
        stability_tolerance=stability_tolerance,
        frequencies=frequencies,
    )
    if not diagnostics["exact_inverse_available"] and not allow_approximate:
        raise ValueError(
            "No exact causal stable MIMO inverse is available: "
            + str(diagnostics["reason"])
        )
    if not diagnostics["stable"] or not diagnostics["forward_stable"]:
        raise ValueError("A regularized MIMO inverse must still be stable; use sampled FIR fallback.")
    effective_regularization = float(diagnostics["regularization"])
    direct_inverse = _regularized_right_inverse(model.D, effective_regularization)
    inverse_model = ZOHDiscreteModel(
        model.A - model.B @ direct_inverse @ model.C,
        model.B @ direct_inverse,
        -direct_inverse @ model.C,
        direct_inverse,
        model.sample_period,
        metadata={
            **dict(model.metadata),
            "design_method": "mimo_discrete_inverse",
            "inverse_of_model": True,
            "inverse_regularization": effective_regularization,
            "latency_samples": int(model.metadata.get("latency_samples", 0)),
        },
    )
    diagnostics = dict(diagnostics)
    if frequencies is not None:
        frequency_array = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    else:
        frequency_array = np.asarray(
            model.metadata.get("fit_frequencies_ghz", ()), dtype=np.float64
        ).reshape(-1)
    if frequency_array.size:
        forward_response = np.asarray(model.response(frequency_array), dtype=np.complex128)
        inverse_response = np.asarray(inverse_model.response(frequency_array), dtype=np.complex128)
        product = np.matmul(forward_response, inverse_response)
        identity = np.eye(product.shape[-1], dtype=np.complex128)
        residual = product - identity
        diagnostics["inverse_residual_rms"] = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
        diagnostics["inverse_residual_peak"] = float(np.max(np.abs(residual)))
    diagnostics["inverse_model"] = inverse_model
    return MIMODiscreteInverseDesign(
        model=inverse_model,
        regularization=effective_regularization,
        exact_inverse_available=bool(diagnostics["exact_inverse_available"]),
        stable_inverse=bool(diagnostics["stable"]),
        diagnostics=diagnostics,
    )


@dataclass(kw_only=True)
class MIMODiscreteInverseStage(BaseBundleTransmissionStage):
    """Stateful bundle stage for a MIMO discrete inverse realization."""

    design: MIMODiscreteInverseDesign
    tail_samples: int = 128
    input_channels: Tuple[str, ...] = ()
    output_channels: Tuple[str, ...] = ()
    retain_state: bool = False
    state: Optional[np.ndarray] = None
    expected_sample_rate: Optional[float] = None
    expected_lo_freq_ghz: Optional[float] = None
    name: str = "mimo_discrete_inverse"
    domain: StageDomain = "iq_complex"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "qubit_iq")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        if not isinstance(self.design, MIMODiscreteInverseDesign):
            raise TypeError("design must be a MIMODiscreteInverseDesign.")
        self.tail_samples = _validate_tail_samples(self.tail_samples)
        model = self.design.model
        if self.input_channels and len(self.input_channels) != model.n_inputs:
            raise ValueError("input_channels must match the inverse input dimension.")
        if self.output_channels and len(self.output_channels) != model.n_outputs:
            raise ValueError("output_channels must match the inverse output dimension.")
        self.input_channels = tuple(self.input_channels)
        self.output_channels = tuple(self.output_channels)
        self.is_lti = not bool(self.retain_state)
        if self.expected_sample_rate is None:
            self.expected_sample_rate = model.sample_rate
        else:
            self.expected_sample_rate = float(self.expected_sample_rate)
            if not np.isfinite(self.expected_sample_rate) or self.expected_sample_rate <= 0.0:
                raise ValueError("expected_sample_rate must be finite and positive.")
        if self.expected_lo_freq_ghz is None:
            self.expected_lo_freq_ghz = self.design.frequency_reference_ghz
        elif not np.isfinite(float(self.expected_lo_freq_ghz)):
            raise ValueError("expected_lo_freq_ghz must be finite when provided.")
        if self.state is not None:
            self.state = np.asarray(self.state, dtype=np.complex128).reshape(-1)
            if self.state.size != model.n_states:
                raise ValueError("state has the wrong size for the inverse model.")
            if np.any(~np.isfinite(self.state)):
                raise ValueError("state must contain finite values.")

    @property
    def latency_samples(self) -> int:
        return int(self.design.model.metadata.get("latency_samples", 0))

    def apply(self, bundle: SignalBundle) -> SignalBundle:
        self._validate_bundle(bundle)
        model = self.design.model
        source_names = self.input_channels or tuple(bundle.order)
        if len(source_names) != model.n_inputs:
            raise ValueError(f"{self.name} expected {model.n_inputs} input channels.")
        if any(name not in bundle.traces for name in source_names):
            missing = [name for name in source_names if name not in bundle.traces]
            raise ValueError(f"{self.name} missing input channel(s): {missing}")
        reference = bundle.traces[source_names[0]]
        for name in source_names:
            _validate_stage_trace_contract(
                bundle.traces[name],
                expected_sample_rate=self.expected_sample_rate,
                expected_lo_freq_ghz=self.expected_lo_freq_ghz,
                stage_name=self.name,
            )
        values = np.column_stack([bundle.traces[name].values for name in source_names])
        output, final_state = propagate_zoh_discrete(
            model,
            values,
            initial_state=self.state if self.retain_state else None,
            return_state=True,
        )
        if self.retain_state:
            self.state = final_state
        tail = propagate_zoh_discrete(model, np.zeros((self.tail_samples, model.n_inputs)), initial_state=final_state)
        state_norm = float(np.linalg.norm(final_state))
        names = self.output_channels or tuple(
            f"drive_{index}" for index in range(model.n_outputs)
        )
        if len(names) != model.n_outputs or len(set(names)) != model.n_outputs:
            raise ValueError("output_channels must provide unique inverse output names.")
        traces = {
            name: reference.clone(
                values=np.asarray(output[:, index], dtype=np.complex128),
                plane=self.output_plane or reference.plane,
                label=f"{reference.label}_{name}_predistorted",
                metadata={
                    **reference.metadata,
                    "last_stage": self.name,
                    "state_mode": "retain" if self.retain_state else "reset",
                    "latency_samples": self.latency_samples,
                    "mimo_inverse_regularization": self.design.regularization,
                    "state_transition_norm": state_norm,
                    **waveform_tail_diagnostics(tail[:, index], output[:, index], reference.sample_rate),
                },
            )
            for index, name in enumerate(names)
        }
        return self._finalize_bundle(
            bundle,
            traces,
            label=f"{bundle.label}_{self.name}",
            metadata_updates={
                "last_stage": self.name,
                "state_mode": "retain" if self.retain_state else "reset",
                "latency_samples": self.latency_samples,
                "state_transition_norm": state_norm,
            },
        )

    def reset(self) -> None:
        self.state = None

    def snapshot_state(self) -> Optional[np.ndarray]:
        return None if self.state is None else np.asarray(self.state, dtype=np.complex128).copy()

    def restore_state(self, state: Optional[np.ndarray]) -> None:
        if state is None:
            self.state = None
            return
        resolved = np.asarray(state, dtype=np.complex128).reshape(-1)
        if resolved.size != self.design.model.n_states:
            raise ValueError("state has the wrong size for the inverse model.")
        if np.any(~np.isfinite(resolved)):
            raise ValueError("state must contain finite values.")
        self.state = resolved.copy()

    def fresh(self) -> "MIMODiscreteInverseStage":
        """Return an equivalent MIMO stage with its state reset."""
        return MIMODiscreteInverseStage(
            design=self.design,
            tail_samples=self.tail_samples,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            retain_state=self.retain_state,
            state=None,
            expected_sample_rate=self.expected_sample_rate,
            expected_lo_freq_ghz=self.expected_lo_freq_ghz,
            name=self.name,
            domain=self.domain,
            allowed_planes=self.allowed_planes,
            output_plane=self.output_plane,
        )

    def describe(self) -> str:
        model = self.design.model
        return (
            f"{self.name}[{model.n_outputs}x{model.n_inputs}, "
            f"states={model.n_states}, {'retain' if self.retain_state else 'reset'}]"
        )


def check_discrete_inverse_realizability(
    model: ZOHDiscreteModel,
    *,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Inspect SISO poles/zeros before constructing a causal exact inverse."""
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    if model.n_inputs != 1 or model.n_outputs != 1:
        raise ValueError("check_discrete_inverse_realizability currently supports SISO models only.")
    numerator, denominator = ss2tf(model.A, model.B, model.C, model.D)
    numerator = np.asarray(numerator[0], dtype=np.complex128)
    denominator = np.asarray(denominator, dtype=np.complex128)
    while numerator.size > 1 and abs(numerator[0]) <= tolerance:
        numerator = numerator[1:]
    poles = np.roots(denominator) if denominator.size > 1 else np.array([], dtype=np.complex128)
    zeros = np.roots(numerator) if numerator.size > 1 else np.array([], dtype=np.complex128)
    delay_samples = int(max(0, len(np.asarray(ss2tf(model.A, model.B, model.C, model.D)[0][0])) - len(numerator)))
    minimum_phase = bool(np.all(np.abs(zeros) < 1.0 - tolerance))
    stable = bool(np.all(np.abs(poles) < 1.0 - tolerance))
    # Leading numerator zeros are an explicit sample delay in H(z). They do
    # not make the delay-free inverse unstable; the caller must account for
    # the reported latency in the waveform schedule.
    causal = bool(numerator.size > 0 and abs(numerator[0]) > tolerance)
    exact = bool(minimum_phase and stable and causal)
    reasons = []
    if not minimum_phase:
        reasons.append("nonminimum_phase")
    if not stable:
        reasons.append("unstable_forward")
    if not causal:
        reasons.append("zero_or_noncausal_transfer")
    reason = "ok" if exact else ";".join(reasons)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "poles": poles,
        "zeros": zeros,
        "delay_samples": delay_samples,
        "minimum_phase": minimum_phase,
        "inverse_stable": minimum_phase,
        "inverse_spectral_radius": float(np.max(np.abs(zeros))) if zeros.size else 0.0,
        "requires_pre_roll": bool(delay_samples),
        "stable": stable,
        "causal": causal,
        "exact_inverse_available": exact,
        "reason": reason,
        "reason_codes": reasons,
    }


def design_strict_discrete_inverse(
    model: ZOHDiscreteModel,
    *,
    tolerance: float = 1e-8,
    allow_approximate: bool = False,
) -> StrictDiscreteInverseDesign:
    """Design the exact causal SISO inverse, rejecting unsafe models by default."""
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    diagnostics = check_discrete_inverse_realizability(model, tolerance=tolerance)
    if allow_approximate and (not diagnostics["inverse_stable"] or not diagnostics["stable"]):
        raise ValueError("Approximation does not permit unstable inverse or forward poles; use explicit fallback.")
    if not diagnostics["exact_inverse_available"] and not allow_approximate:
        raise ValueError(
            "No causal stable exact inverse is available: " + str(diagnostics["reason"])
        )
    numerator = np.asarray(diagnostics["denominator"], dtype=np.complex128)
    denominator = np.asarray(diagnostics["numerator"], dtype=np.complex128)
    if abs(denominator[0]) <= tolerance:
        raise ValueError("The inverse denominator has a leading delay and is not causal.")
    diagnostics = dict(diagnostics)
    diagnostics["design_method"] = "strict_discrete_inverse"
    return StrictDiscreteInverseDesign(
        numerator=numerator,
        denominator=denominator,
        delay_samples=int(diagnostics["delay_samples"]),
        exact_inverse_available=bool(diagnostics["exact_inverse_available"]),
        minimum_phase=bool(diagnostics["minimum_phase"]),
        stable_inverse=bool(diagnostics["inverse_stable"]),
        diagnostics=diagnostics,
        sample_rate=model.sample_rate,
        frequency_reference_ghz=(
            None
            if model.metadata.get("lo_freq_ghz") is None
            else float(model.metadata["lo_freq_ghz"])
        ),
    )


@dataclass(frozen=True)
class FIRInverseDesign:
    """Sampled FIR inverse and diagnostics for truncation/regularization."""

    kernel: np.ndarray
    delay_samples: int
    regularization: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    sample_rate: Optional[float] = None
    frequency_reference_ghz: Optional[float] = None

    def __post_init__(self) -> None:
        kernel = np.asarray(self.kernel, dtype=np.complex128).reshape(-1)
        if kernel.size == 0 or np.any(~np.isfinite(kernel)):
            raise ValueError("FIRInverseDesign.kernel must be non-empty and finite.")
        object.__setattr__(self, "kernel", kernel)
        delay = int(self.delay_samples)
        if delay < 0:
            raise ValueError("delay_samples must be non-negative.")
        if delay >= kernel.size:
            raise ValueError("delay_samples must be smaller than the number of FIR taps.")
        object.__setattr__(self, "delay_samples", delay)
        regularization = float(self.regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        object.__setattr__(self, "regularization", regularization)
        if self.sample_rate is not None:
            sample_rate = float(self.sample_rate)
            if not np.isfinite(sample_rate) or sample_rate <= 0.0:
                raise ValueError("sample_rate must be finite and positive when provided.")
            object.__setattr__(self, "sample_rate", sample_rate)
        if self.frequency_reference_ghz is not None:
            reference = float(self.frequency_reference_ghz)
            if not np.isfinite(reference):
                raise ValueError("frequency_reference_ghz must be finite when provided.")
            object.__setattr__(self, "frequency_reference_ghz", reference)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _window_values(window: Optional[Union[str, Tuple[Any, ...]]], size: int) -> np.ndarray:
    if window is None or window == "none":
        return np.ones(size, dtype=np.float64)
    return np.asarray(get_window(window, size), dtype=np.float64)


def evaluate_fir_response(kernel: np.ndarray, frequencies: np.ndarray, sample_rate: float) -> np.ndarray:
    """Evaluate an FIR kernel using the ``z^-1`` convention."""
    taps = np.asarray(kernel, dtype=np.complex128).reshape(-1)
    if taps.size == 0 or np.any(~np.isfinite(taps)):
        raise ValueError("kernel must be non-empty and finite.")
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive.")
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    if np.any(~np.isfinite(frequency_array)):
        raise ValueError("frequencies must contain finite values.")
    omega = 2.0 * np.pi * frequency_array / rate
    powers = np.arange(taps.size, dtype=np.float64)
    if omega.ndim:
        return np.exp(-1j * omega[..., np.newaxis] * powers) @ taps
    return np.sum(taps * np.exp(-1j * omega * powers))


def evaluate_iir_response(
    numerator: Union[np.ndarray, Sequence[complex]],
    denominator: Union[np.ndarray, Sequence[complex]],
    frequencies: Union[np.ndarray, Sequence[float]],
    sample_rate: float,
) -> np.ndarray:
    """Evaluate a causal discrete IIR transfer function in ``z^-1`` form."""
    numerator_array = np.asarray(numerator, dtype=np.complex128).reshape(-1)
    denominator_array = np.asarray(denominator, dtype=np.complex128).reshape(-1)
    if numerator_array.size == 0 or denominator_array.size == 0:
        raise ValueError("IIR numerator and denominator must not be empty.")
    if not np.isfinite(float(sample_rate)) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive.")
    if np.any(~np.isfinite(numerator_array)) or np.any(~np.isfinite(denominator_array)):
        raise ValueError("IIR numerator and denominator must be finite.")
    if abs(denominator_array[0]) <= 0.0:
        raise ValueError("IIR denominator must have a non-zero leading coefficient.")
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    if np.any(~np.isfinite(frequency_array)):
        raise ValueError("frequencies must contain finite values.")
    scalar = frequency_array.ndim == 0
    omega = 2.0 * np.pi * frequency_array.reshape(-1) / float(sample_rate)
    numerator_response = np.exp(-1j * omega[:, np.newaxis] * np.arange(numerator_array.size)) @ numerator_array
    denominator_response = np.exp(-1j * omega[:, np.newaxis] * np.arange(denominator_array.size)) @ denominator_array
    if np.any(np.abs(denominator_response) <= 1e-15):
        raise ValueError("IIR response denominator is singular on the requested grid.")
    result = numerator_response / denominator_response
    return result[0] if scalar else result.reshape(frequency_array.shape)


def calculate_inverse_response_diagnostics(
    forward_response: Union[np.ndarray, Sequence[complex]],
    correction_response: Union[np.ndarray, Sequence[complex]],
    *,
    passband_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return residual, gain, and dynamic-range metrics for a correction path."""
    forward = np.asarray(forward_response, dtype=np.complex128)
    correction = np.asarray(correction_response, dtype=np.complex128)
    if forward.shape != correction.shape:
        raise ValueError("forward_response and correction_response must have the same shape.")
    if np.any(~np.isfinite(forward)) or np.any(~np.isfinite(correction)):
        raise ValueError("forward_response and correction_response must be finite.")
    if passband_mask is None:
        mask = np.ones(forward.shape, dtype=bool)
    else:
        mask = np.asarray(passband_mask, dtype=bool)
        if mask.shape != forward.shape:
            raise ValueError("passband_mask must match the response shape.")
    product = forward * correction
    error = product - 1.0
    selected = error[mask]
    return {
        "residual_rms": float(np.sqrt(np.mean(np.abs(selected) ** 2))) if selected.size else float("nan"),
        "residual_peak": float(np.max(np.abs(selected))) if selected.size else float("nan"),
        "forward_min_abs": float(np.min(np.abs(forward[mask]))) if np.any(mask) else float("nan"),
        "forward_max_abs": float(np.max(np.abs(forward[mask]))) if np.any(mask) else float("nan"),
        "correction_peak": float(np.max(np.abs(correction[mask]))) if np.any(mask) else float("nan"),
        "passband_fraction": float(np.mean(mask)),
    }


def design_sampled_fir_inverse(
    response_values: Union[np.ndarray, Sequence[complex]],
    *,
    sample_rate: float,
    num_taps: int,
    delay_samples: Optional[int] = None,
    regularization: float = 1e-8,
    passband_mask: Optional[np.ndarray] = None,
    window: Optional[Union[str, Tuple[Any, ...]]] = "kaiser",
    kaiser_beta: float = 6.0,
    frequency_reference_ghz: Optional[float] = None,
) -> FIRInverseDesign:
    """Design a causal sampled FIR inverse with Tikhonov regularization."""
    response = np.asarray(response_values, dtype=np.complex128).reshape(-1)
    if response.size == 0:
        raise ValueError("response_values must not be empty.")
    if np.any(~np.isfinite(response)):
        raise ValueError("response_values must be finite.")
    if num_taps <= 0 or num_taps > response.size:
        raise ValueError("num_taps must be positive and no larger than the response grid.")
    if (
        not np.isfinite(float(sample_rate))
        or sample_rate <= 0
        or not np.isfinite(float(regularization))
        or regularization < 0
    ):
        raise ValueError("sample_rate must be finite and positive and regularization non-negative.")
    delay = (num_taps - 1) // 2 if delay_samples is None else int(delay_samples)
    if delay < 0 or delay >= num_taps:
        raise ValueError("delay_samples must be non-negative and fit the FIR taps.")
    if passband_mask is None:
        mask = np.ones(response.size, dtype=bool)
    else:
        mask = np.asarray(passband_mask, dtype=bool).reshape(-1)
        if mask.size != response.size:
            raise ValueError("passband_mask must match response_values.")
    if not np.any(mask):
        raise ValueError("passband_mask must select at least one frequency bin.")
    inverse = np.zeros_like(response)
    magnitude_sq = np.abs(response) ** 2
    inverse[mask] = np.conj(response[mask]) / (magnitude_sq[mask] + float(regularization))
    omega = 2.0 * np.pi * np.fft.fftfreq(response.size, d=1.0 / float(sample_rate)) / float(sample_rate)
    desired = inverse * np.exp(-1j * omega * delay)
    impulse = np.fft.ifft(desired)
    kernel = np.asarray(impulse[:num_taps], dtype=np.complex128)
    resolved_window = ("kaiser", kaiser_beta) if window == "kaiser" else window
    kernel *= _window_values(resolved_window, num_taps)
    realized = np.fft.fft(kernel, n=response.size) * np.exp(1j * omega * delay)
    weighted_error = realized[mask] * response[mask] - 1.0
    diagnostics = {
        "design_method": "sampled_fir_tikhonov",
        "num_taps": int(num_taps),
        "delay_samples": int(delay),
        "regularization": float(regularization),
        "passband_fraction": float(np.mean(mask)),
        "residual_rms": float(np.sqrt(np.mean(np.abs(weighted_error) ** 2))) if np.any(mask) else float("nan"),
        "residual_peak": float(np.max(np.abs(weighted_error))),
        "inverse_gain_peak": float(np.max(np.abs(realized[mask]))),
        "forward_gain_dynamic_range": float(np.max(np.abs(response[mask])) / max(1e-30, np.min(np.abs(response[mask])))),
        "kernel_peak": float(np.max(np.abs(kernel))),
        "kernel_rms": float(np.sqrt(np.mean(np.abs(kernel) ** 2))),
        "kernel_tail_rms": float(np.sqrt(np.mean(np.abs(kernel[-max(1, num_taps // 10):]) ** 2))),
    }
    return FIRInverseDesign(
        kernel,
        delay,
        regularization,
        diagnostics,
        sample_rate=float(sample_rate),
        frequency_reference_ghz=frequency_reference_ghz,
    )


@dataclass(frozen=True)
class MIMOFIRInverseDesign:
    """Frequency-by-frequency regularized MIMO inverse represented by FIR kernels."""

    kernels: np.ndarray
    delay_samples: int
    regularization: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    sample_rate: Optional[float] = None
    frequency_reference_ghz: Optional[float] = None

    def __post_init__(self) -> None:
        kernels = np.asarray(self.kernels, dtype=np.complex128)
        if kernels.ndim != 3:
            raise ValueError("kernels must have shape (n_inputs, n_outputs, n_taps).")
        if kernels.size == 0 or np.any(~np.isfinite(kernels)):
            raise ValueError("kernels must be non-empty and finite.")
        object.__setattr__(self, "kernels", kernels)
        delay = int(self.delay_samples)
        if delay < 0:
            raise ValueError("delay_samples must be non-negative.")
        if delay >= kernels.shape[-1]:
            raise ValueError("delay_samples must be smaller than the number of MIMO FIR taps.")
        object.__setattr__(self, "delay_samples", delay)
        regularization = float(self.regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        object.__setattr__(self, "regularization", regularization)
        if self.sample_rate is not None:
            sample_rate = float(self.sample_rate)
            if not np.isfinite(sample_rate) or sample_rate <= 0.0:
                raise ValueError("sample_rate must be finite and positive when provided.")
            object.__setattr__(self, "sample_rate", sample_rate)
        if self.frequency_reference_ghz is not None:
            reference = float(self.frequency_reference_ghz)
            if not np.isfinite(reference):
                raise ValueError("frequency_reference_ghz must be finite when provided.")
            object.__setattr__(self, "frequency_reference_ghz", reference)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass
class MIMOFIRFilterStage(BaseBundleTransmissionStage):
    """Bundle stage that applies a designed MIMO FIR inverse.

    ``MIMOFIRInverseDesign.kernels`` uses ``(n_inputs, n_outputs, taps)``.
    The bundle therefore contains the desired output channels and the stage
    emits the corresponding physical input/drive channels.
    """

    design: MIMOFIRInverseDesign = field(default=None)  # type: ignore[assignment]
    input_channels: Tuple[str, ...] = ()
    output_channels: Tuple[str, ...] = ()
    preserve_delay: bool = True
    tail_samples: int = 128
    name: str = "mimo_fir_inverse"
    domain: StageDomain = "iq_complex"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "qubit_iq")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        if not isinstance(self.design, MIMOFIRInverseDesign):
            raise TypeError("design must be a MIMOFIRInverseDesign.")
        self.tail_samples = _validate_tail_samples(self.tail_samples)
        n_inputs, n_outputs, _ = self.design.kernels.shape
        if self.input_channels and len(self.input_channels) != n_outputs:
            raise ValueError("input_channels must match the design output dimension.")
        if self.output_channels and len(self.output_channels) != n_inputs:
            raise ValueError("output_channels must match the design input dimension.")
        self.input_channels = tuple(self.input_channels)
        self.output_channels = tuple(self.output_channels)

    @property
    def latency_samples(self) -> int:
        return int(self.design.delay_samples) if self.preserve_delay else 0

    def apply(self, bundle: SignalBundle) -> SignalBundle:
        self._validate_bundle(bundle)
        n_inputs, n_outputs, n_taps = self.design.kernels.shape
        source_names = self.input_channels or tuple(bundle.order)
        if len(source_names) != n_outputs:
            raise ValueError(
                f"{self.name} expected {n_outputs} desired channels, received {len(source_names)}."
            )
        missing = [name for name in source_names if name not in bundle.traces]
        if missing:
            raise ValueError(f"{self.name} could not find input channel(s): {missing}")
        reference = bundle.traces[source_names[0]]
        for name in source_names:
            trace = bundle.traces[name]
            _validate_stage_trace_contract(
                trace,
                expected_sample_rate=self.design.sample_rate,
                expected_lo_freq_ghz=self.design.frequency_reference_ghz,
                stage_name=self.name,
            )
            if trace.domain != reference.domain or trace.plane != reference.plane:
                raise ValueError(f"{self.name} requires aligned traces with one domain and plane.")
        sample_count = len(reference.values)
        full_outputs = np.zeros((n_inputs, sample_count + n_taps - 1), dtype=np.complex128)
        for input_index in range(n_inputs):
            for output_index, name in enumerate(source_names):
                kernel = self.design.kernels[input_index, output_index]
                if kernel.size != n_taps:
                    raise ValueError("MIMO kernel tap dimensions are inconsistent.")
                full = convolve(
                    reference.values * 0.0 + bundle.traces[name].values,
                    kernel,
                    mode="full",
                    method="auto",
                )
                full_outputs[input_index] += full
        start = 0 if self.preserve_delay else int(self.design.delay_samples)
        corrected = full_outputs[:, start:start + sample_count]
        discarded = full_outputs[:, start + sample_count:]
        tails = np.zeros((n_inputs, self.tail_samples), dtype=np.complex128)
        tail_count = min(self.tail_samples, discarded.shape[1])
        tails[:, :tail_count] = discarded[:, :tail_count]
        total_energy = float(np.sum(np.abs(full_outputs) ** 2))
        discarded_energy = float(np.sum(np.abs(discarded) ** 2) + np.sum(np.abs(full_outputs[:, :start]) ** 2))
        edge_energy_fraction = discarded_energy / total_energy if total_energy else 0.0
        names = self.output_channels or (
            source_names if n_inputs == n_outputs else tuple(f"drive_{index}" for index in range(n_inputs))
        )
        if len(names) != n_inputs or len(set(names)) != n_inputs:
            raise ValueError("output_channels must provide one unique name per MIMO input.")
        traces = {
            name: reference.clone(
                values=values,
                plane=self.output_plane or reference.plane,
                label=f"{reference.label}_{name}_predistorted",
                metadata={
                    **reference.metadata,
                    "last_stage": self.name,
                    "mimo_input_index": index,
                    "mimo_output_count": n_outputs,
                    "mimo_latency_samples": self.latency_samples,
                    **waveform_tail_diagnostics(tails[index], values, reference.sample_rate),
                    "edge_energy_fraction": edge_energy_fraction,
                },
            )
            for index, (name, values) in enumerate(zip(names, corrected))
        }
        return self._finalize_bundle(
            bundle,
            traces,
            label=f"{bundle.label}_{self.name}",
            metadata_updates={
                "last_stage": self.name,
                "mimo_latency_samples": self.latency_samples,
                "mimo_state_mode": "stateless",
                **waveform_tail_diagnostics(tails.T, corrected.T, reference.sample_rate),
                "edge_energy_fraction": edge_energy_fraction,
            },
        )

    def describe(self) -> str:
        n_inputs, n_outputs, n_taps = self.design.kernels.shape
        return f"{self.name}[{n_inputs}x{n_outputs}, taps={n_taps}, latency={self.latency_samples}]"


def design_mimo_sampled_fir_inverse(
    response_matrix: np.ndarray,
    *,
    sample_rate: float,
    num_taps: int,
    delay_samples: Optional[int] = None,
    regularization: float = 1e-8,
    passband_mask: Optional[np.ndarray] = None,
    window: Optional[Union[str, Tuple[Any, ...]]] = "kaiser",
    kaiser_beta: float = 6.0,
    frequency_reference_ghz: Optional[float] = None,
) -> MIMOFIRInverseDesign:
    """Design a regularized MIMO inverse for ``H[f]`` with shape ``(f, out, in)``."""
    response = np.asarray(response_matrix, dtype=np.complex128)
    if response.ndim != 3:
        raise ValueError("response_matrix must have shape (n_freq, n_outputs, n_inputs).")
    n_freq, n_outputs, n_inputs = response.shape
    if n_freq == 0 or n_outputs == 0 or n_inputs == 0:
        raise ValueError("response_matrix dimensions must all be positive.")
    if np.any(~np.isfinite(response)):
        raise ValueError("response_matrix must be finite.")
    if num_taps <= 0 or num_taps > n_freq:
        raise ValueError("num_taps must be positive and no larger than n_freq.")
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive.")
    regularization = float(regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative.")
    delay = (num_taps - 1) // 2 if delay_samples is None else int(delay_samples)
    if delay < 0 or delay >= num_taps:
        raise ValueError("delay_samples must be non-negative and fit the FIR taps.")
    if passband_mask is None:
        mask = np.ones(n_freq, dtype=bool)
    else:
        mask = np.asarray(passband_mask, dtype=bool).reshape(-1)
        if mask.size != n_freq:
            raise ValueError("passband_mask must match response_matrix frequency dimension.")
    if not np.any(mask):
        raise ValueError("passband_mask must select at least one frequency bin.")
    inverse = np.empty((n_freq, n_inputs, n_outputs), dtype=np.complex128)
    condition_numbers = np.empty(n_freq, dtype=np.float64)
    identity = np.eye(n_outputs, dtype=np.complex128)
    for index, matrix in enumerate(response):
        if not mask[index]:
            inverse[index] = 0.0
            condition_numbers[index] = np.inf
            continue
        gram = matrix @ matrix.conj().T
        inverse[index] = matrix.conj().T @ np.linalg.solve(gram + regularization * identity, identity)
        condition_numbers[index] = np.linalg.cond(matrix) if np.any(np.abs(matrix) > 0) else np.inf
    omega = 2.0 * np.pi * np.fft.fftfreq(n_freq, d=1.0 / sample_rate) / sample_rate
    kernels = np.fft.ifft(inverse * np.exp(-1j * omega[:, None, None] * delay), axis=0)[:num_taps]
    kernels = np.transpose(kernels, (1, 2, 0))
    resolved_window = ("kaiser", kaiser_beta) if window == "kaiser" else window
    kernels *= _window_values(resolved_window, num_taps)[None, None, :]
    realized = np.fft.fft(np.transpose(kernels, (2, 0, 1)), n=n_freq, axis=0)
    realized *= np.exp(1j * omega[:, None, None] * delay)
    residual = np.matmul(response, realized)
    residual -= np.eye(n_outputs, dtype=np.complex128)[None, :, :]
    selected_residual = residual[mask]
    diagnostics = {
        "design_method": "mimo_sampled_fir_tikhonov",
        "num_taps": int(num_taps),
        "delay_samples": int(delay),
        "regularization": regularization,
        "passband_fraction": float(np.mean(mask)),
        "residual_rms": float(np.sqrt(np.mean(np.abs(selected_residual) ** 2))),
        "condition_number_max": float(np.nanmax(condition_numbers[mask])),
        "condition_number_median": float(np.nanmedian(condition_numbers[mask])),
        "kernel_peak": float(np.max(np.abs(kernels))),
    }
    return MIMOFIRInverseDesign(
        kernels,
        delay,
        regularization,
        diagnostics,
        sample_rate=sample_rate,
        frequency_reference_ghz=frequency_reference_ghz,
    )


@dataclass(frozen=True)
class RationalFitResult:
    """Stable pole-residue fit of one measured complex transfer path."""

    poles_rad_per_ns: np.ndarray
    residues: np.ndarray
    direct: complex
    frequency_center_ghz: float
    residual_rms: float
    frequencies_ghz: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    fitted_response: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        poles = np.asarray(self.poles_rad_per_ns, dtype=np.complex128).reshape(-1)
        residues = np.asarray(self.residues, dtype=np.complex128).reshape(-1)
        if poles.size == 0 or poles.size != residues.size:
            raise ValueError("RationalFitResult requires one residue per pole.")
        if np.any(~np.isfinite(poles)) or np.any(~np.isfinite(residues)):
            raise ValueError("RationalFitResult poles and residues must be finite.")
        if not np.isfinite(complex(self.direct).real) or not np.isfinite(complex(self.direct).imag):
            raise ValueError("RationalFitResult.direct must be finite.")
        object.__setattr__(self, "poles_rad_per_ns", poles)
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "direct", complex(self.direct))
        center = float(self.frequency_center_ghz)
        residual_rms = float(self.residual_rms)
        frequencies_array = np.asarray(self.frequencies_ghz, dtype=np.float64).reshape(-1)
        fitted_response = np.asarray(self.fitted_response, dtype=np.complex128).reshape(-1)
        if not np.isfinite(center) or not np.isfinite(residual_rms) or residual_rms < 0.0:
            raise ValueError("RationalFitResult frequency center and residual must be finite.")
        if np.any(~np.isfinite(frequencies_array)) or np.any(~np.isfinite(fitted_response)):
            raise ValueError("RationalFitResult frequency samples must be finite.")
        if fitted_response.size not in (0, frequencies_array.size):
            raise ValueError("RationalFitResult fitted_response must align with frequencies_ghz.")
        object.__setattr__(self, "frequency_center_ghz", center)
        object.__setattr__(self, "residual_rms", residual_rms)
        object.__setattr__(self, "frequencies_ghz", frequencies_array)
        object.__setattr__(self, "fitted_response", fitted_response)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def evaluate(self, frequencies_ghz: Union[float, np.ndarray, Iterable[float]]) -> np.ndarray:
        """Evaluate the fitted continuous-time model at frequencies in GHz."""
        frequencies = np.asarray(frequencies_ghz, dtype=np.float64)
        s = 2j * np.pi * frequencies.reshape(-1)
        response = np.full(s.shape, self.direct, dtype=np.complex128)
        for pole, residue in zip(self.poles_rad_per_ns, self.residues):
            response += residue / (s - pole)
        return response[0] if frequencies.ndim == 0 else response.reshape(frequencies.shape)

    @property
    def order(self) -> int:
        return int(self.poles_rad_per_ns.size)

    @property
    def train_residual_rms(self) -> float:
        return float(self.metadata.get("train_residual_rms", self.residual_rms))

    @property
    def holdout_residual_rms(self) -> float:
        return float(self.metadata.get("holdout_residual_rms", np.nan))

    @property
    def passivity_violation(self) -> float:
        return float(self.metadata.get("passivity_violation", np.nan))


def fit_touchstone_rational(
    frequencies_ghz: Union[np.ndarray, Sequence[float]],
    response_values: Union[np.ndarray, Sequence[complex]],
    *,
    order: int = 4,
    frequency_center_ghz: Optional[float] = None,
    bandwidth_ghz: Optional[float] = None,
    stable_only: bool = True,
    candidate_orders: Optional[Sequence[int]] = None,
    auto_order: bool = False,
    max_order: Optional[int] = None,
    holdout_fraction: float = 0.0,
    pole_relocation_iterations: int = 0,
    enforce_passivity: bool = False,
    passivity_tolerance: float = 1e-6,
) -> RationalFitResult:
    """Fit a measured path with a deterministic stable pole-residue model.

    The linear residue solve is paired with optional order selection and a
    bounded pole-scale relocation search.  This is deliberately deterministic
    (there is no random initialization), which makes cached XY reports
    reproducible while still providing train/holdout and passivity diagnostics.
    """
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64).reshape(-1)
    response = np.asarray(response_values, dtype=np.complex128).reshape(-1)
    if frequencies.size != response.size or frequencies.size < 3:
        raise ValueError("frequencies_ghz and response_values must have at least three aligned samples.")
    if np.any(~np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_ghz must be finite and strictly increasing.")
    if np.any(~np.isfinite(response)):
        raise ValueError("response_values must be finite.")
    if int(order) <= 0:
        raise ValueError("order must be positive.")
    holdout_fraction = float(holdout_fraction)
    if not np.isfinite(holdout_fraction) or not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be finite and in [0, 1).")
    pole_relocation_iterations = int(pole_relocation_iterations)
    if pole_relocation_iterations < 0:
        raise ValueError("pole_relocation_iterations must be non-negative.")
    passivity_tolerance = float(passivity_tolerance)
    if not np.isfinite(passivity_tolerance) or passivity_tolerance < 0.0:
        raise ValueError("passivity_tolerance must be finite and non-negative.")
    center = float(np.median(frequencies) if frequency_center_ghz is None else frequency_center_ghz)
    span = float(np.ptp(frequencies) if bandwidth_ghz is None else bandwidth_ghz)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("bandwidth_ghz must be finite and positive.")

    if candidate_orders is None:
        if auto_order:
            upper = int(order if max_order is None else max_order)
            if upper <= 0:
                raise ValueError("max_order must be positive when auto_order is enabled.")
            candidate_orders = tuple(range(1, upper + 1))
        else:
            candidate_orders = (int(order),)
    else:
        candidate_orders = tuple(sorted({int(value) for value in candidate_orders}))
    if not candidate_orders or any(value <= 0 for value in candidate_orders):
        raise ValueError("candidate_orders must contain positive integers.")

    n_samples = frequencies.size
    holdout_mask = np.zeros(n_samples, dtype=bool)
    if holdout_fraction > 0.0 and n_samples >= 6:
        stride = max(2, int(round(1.0 / holdout_fraction)))
        holdout_mask[::stride] = True
        # Keep enough training rows for the largest requested model.
        if np.count_nonzero(~holdout_mask) < max(3, max(candidate_orders) + 1):
            holdout_mask[:] = False
            holdout_mask[-1] = True
    train_mask = ~holdout_mask
    if np.count_nonzero(train_mask) < 3:
        raise ValueError("The rational fit needs at least three training samples.")

    def solve_candidate(candidate_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if candidate_order + 1 > np.count_nonzero(train_mask):
            raise ValueError(
                f"Rational order {candidate_order} needs more training samples than available."
            )
        rates = 2.0 * np.pi * span * np.logspace(-1.0, 1.0, candidate_order)
        poles = -rates.astype(np.complex128) + 2j * np.pi * center
        s_train = 2j * np.pi * frequencies[train_mask]

        def fit_for_poles(pole_values: np.ndarray) -> tuple[np.ndarray, float]:
            matrix = np.column_stack(
                [np.ones(s_train.size, dtype=np.complex128)]
                + [1.0 / (s_train - pole) for pole in pole_values]
            )
            coefficients, *_ = np.linalg.lstsq(matrix, response[train_mask], rcond=None)
            residual = matrix @ coefficients - response[train_mask]
            return coefficients, float(np.sqrt(np.mean(np.abs(residual) ** 2)))

        coefficients, best_error = fit_for_poles(poles)
        # Relocate poles by a bounded logarithmic scale search.  Each pass
        # chooses the best scale for every pole while preserving stability.
        for _ in range(pole_relocation_iterations):
            improved = False
            for pole_index in range(poles.size):
                current_rate = max(1e-15, -float(np.real(poles[pole_index])))
                candidates = []
                for scale in (0.65, 0.82, 1.0, 1.22, 1.55):
                    trial = poles.copy()
                    trial[pole_index] = -current_rate * scale + 2j * np.pi * center
                    trial_coefficients, trial_error = fit_for_poles(trial)
                    candidates.append((trial_error, trial, trial_coefficients))
                trial_error, trial_poles, trial_coefficients = min(candidates, key=lambda item: item[0])
                if trial_error + 1e-15 < best_error:
                    poles = trial_poles
                    coefficients = trial_coefficients
                    best_error = trial_error
                    improved = True
            if not improved:
                break

        s_all = 2j * np.pi * frequencies
        matrix_all = np.column_stack(
            [np.ones(s_all.size, dtype=np.complex128)]
            + [1.0 / (s_all - pole) for pole in poles]
        )
        fitted_all = matrix_all @ coefficients
        return poles, coefficients, fitted_all

    candidate_results: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    failures: list[str] = []
    for candidate_order in candidate_orders:
        try:
            poles, coefficients, fitted = solve_candidate(candidate_order)
        except ValueError as error:
            failures.append(str(error))
            continue
        if stable_only and np.any(np.real(poles) >= 0.0):
            failures.append(f"Rational fit order {candidate_order} produced an unstable pole.")
            continue
        train_error = float(np.sqrt(np.mean(np.abs(fitted[train_mask] - response[train_mask]) ** 2)))
        holdout_error = (
            float(np.sqrt(np.mean(np.abs(fitted[holdout_mask] - response[holdout_mask]) ** 2)))
            if np.any(holdout_mask)
            else train_error
        )
        selection_score = holdout_error if np.any(holdout_mask) else train_error
        # Prefer the lower order when validation errors are numerically tied.
        selection_score += 1e-12 * int(candidate_order)
        candidate_results.append(
            (selection_score, int(candidate_order), poles, coefficients, fitted, train_error, holdout_error)
        )
    if not candidate_results:
        detail = failures[0] if failures else "no candidate model"
        raise ValueError(f"Unable to construct a stable rational fit: {detail}")
    _, selected_order, poles, coefficients, fitted, train_error, holdout_error = min(
        candidate_results, key=lambda item: item[0]
    )
    residual_rms = float(np.sqrt(np.mean(np.abs(fitted - response) ** 2)))
    passivity_violation = float(max(0.0, float(np.max(np.abs(fitted))) - 1.0))
    if enforce_passivity and passivity_violation > passivity_tolerance:
        raise ValueError(
            f"Rational fit violates scalar passivity by {passivity_violation:g} "
            f"(tolerance {passivity_tolerance:g})."
        )
    return RationalFitResult(
        poles_rad_per_ns=poles,
        residues=coefficients[1:],
        direct=coefficients[0],
        frequency_center_ghz=center,
        residual_rms=residual_rms,
        frequencies_ghz=frequencies,
        fitted_response=fitted,
        metadata={
            "design_method": "fixed_pole_rational_fit",
            "order": int(selected_order),
            "bandwidth_ghz": span,
            "stable_only": bool(stable_only),
            "candidate_orders": [int(value) for value in candidate_orders],
            "auto_order": bool(auto_order),
            "train_sample_count": int(np.count_nonzero(train_mask)),
            "holdout_sample_count": int(np.count_nonzero(holdout_mask)),
            "train_residual_rms": train_error,
            "holdout_residual_rms": holdout_error,
            "pole_relocation_iterations": int(pole_relocation_iterations),
            "passivity_violation": passivity_violation,
            "passivity_tolerance": passivity_tolerance,
            "passivity_checked": bool(enforce_passivity),
        },
    )


def check_touchstone_network_physicality(
    network: "TouchstoneNetwork",
    *,
    passivity_tolerance: float = 1e-6,
    reciprocity_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Report finite, passive, and reciprocal properties of a Touchstone network.

    The check is diagnostic by default.  A measured network may intentionally
    include gain or calibrated non-reciprocal components; callers decide
    whether a violation should reject a fit.
    """
    if not hasattr(network, "s_parameters") or not hasattr(network, "frequencies"):
        raise TypeError("network must provide frequencies and s_parameters.")
    passivity_tolerance = float(passivity_tolerance)
    reciprocity_tolerance = float(reciprocity_tolerance)
    if (
        not np.isfinite(passivity_tolerance)
        or passivity_tolerance < 0.0
        or not np.isfinite(reciprocity_tolerance)
        or reciprocity_tolerance < 0.0
    ):
        raise ValueError("physicality tolerances must be finite and non-negative.")
    frequencies = np.asarray(network.frequencies, dtype=np.float64).reshape(-1)
    matrices = np.asarray(network.s_parameters, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[0] != frequencies.size:
        raise ValueError("network.s_parameters must have shape (n_freq, n_ports, n_ports).")
    singular_max: list[float] = []
    reciprocity_max: list[float] = []
    for matrix in matrices:
        if np.any(~np.isfinite(matrix)):
            singular_max.append(float("inf"))
            reciprocity_max.append(float("inf"))
            continue
        singular_max.append(float(np.max(np.linalg.svd(matrix, compute_uv=False))))
        reciprocity_max.append(float(np.max(np.abs(matrix - matrix.T))))
    max_singular = float(np.max(singular_max)) if singular_max else float("nan")
    max_reciprocity = float(np.max(reciprocity_max)) if reciprocity_max else float("nan")
    return {
        "finite": bool(np.all(np.isfinite(matrices))),
        "max_scattering_singular_value": max_singular,
        "passivity_violation": max(0.0, max_singular - 1.0),
        "passive": bool(max_singular <= 1.0 + passivity_tolerance),
        "max_reciprocity_error": max_reciprocity,
        "reciprocal": bool(max_reciprocity <= reciprocity_tolerance),
        "passivity_tolerance": passivity_tolerance,
        "reciprocity_tolerance": reciprocity_tolerance,
    }


def fit_response_to_zoh(
    frequencies_ghz: Union[np.ndarray, Sequence[float]],
    response_values: Union[np.ndarray, Sequence[complex]],
    *,
    sample_period: float,
    order: int = 4,
    frequency_reference_ghz: float = 0.0,
    frequency_mode: FrequencyMode = "absolute",
    metadata: Optional[Dict[str, Any]] = None,
    candidate_orders: Optional[Sequence[int]] = None,
    auto_order: bool = False,
    max_order: Optional[int] = None,
    holdout_fraction: float = 0.0,
    pole_relocation_iterations: int = 0,
    enforce_passivity: bool = False,
    passivity_tolerance: float = 1e-6,
    latency_samples: int = 0,
) -> Tuple[RationalFitResult, ZOHDiscreteModel]:
    """Fit a sampled response and convert it to a ZOH state-space model.

    Touchstone data are normally expressed on an absolute RF frequency grid,
    while an IQ stage runs on frequency offsets around its LO.  When
    ``frequency_mode="absolute"`` the fitted poles are shifted by
    ``frequency_reference_ghz`` before discretization, so the returned model
    can be evaluated directly on baseband offsets.
    """
    if frequency_mode not in ("absolute", "relative"):
        raise ValueError("frequency_mode must be 'absolute' or 'relative'.")
    reference = float(frequency_reference_ghz)
    if not np.isfinite(reference):
        raise ValueError("frequency_reference_ghz must be finite.")
    if not np.isfinite(float(sample_period)) or float(sample_period) <= 0.0:
        raise ValueError("sample_period must be finite and positive.")
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64).reshape(-1)
    response = np.asarray(response_values, dtype=np.complex128).reshape(-1)
    if frequencies.size != response.size:
        raise ValueError("frequencies_ghz and response_values must have the same length.")
    if frequencies.size < 3:
        raise ValueError("At least three frequency samples are required for ZOH fitting.")
    if np.any(~np.isfinite(frequencies)) or np.any(~np.isfinite(response)):
        raise ValueError("frequencies_ghz and response_values must be finite.")
    latency_samples = int(latency_samples)
    if latency_samples < 0:
        raise ValueError("latency_samples must be non-negative.")
    fit_frequencies = frequencies - reference if frequency_mode == "absolute" else frequencies
    # A sampled model cannot represent content above Nyquist.  Discard bins
    # outside a conservative guard band instead of fitting aliased poles.
    nyquist_guard = 0.45 / float(sample_period)
    safe_mask = np.abs(fit_frequencies) <= nyquist_guard
    if np.count_nonzero(safe_mask) < 3:
        raise ValueError("Fewer than three samples remain inside the ZOH Nyquist fitting band.")
    fit_frequencies = fit_frequencies[safe_mask]
    fit_response = response[safe_mask]
    fit_bandwidth = float(np.ptp(fit_frequencies))
    fit_bandwidth = min(fit_bandwidth, 0.9 / float(sample_period))
    if fit_bandwidth <= 0.0:
        raise ValueError("The response frequency span must be positive for ZOH fitting.")
    fit = fit_touchstone_rational(
        fit_frequencies,
        fit_response,
        order=order,
        frequency_center_ghz=float(np.median(fit_frequencies)),
        bandwidth_ghz=fit_bandwidth,
        candidate_orders=candidate_orders,
        auto_order=auto_order,
        max_order=max_order,
        holdout_fraction=holdout_fraction,
        pole_relocation_iterations=pole_relocation_iterations,
        enforce_passivity=enforce_passivity,
        passivity_tolerance=passivity_tolerance,
    )
    poles = np.asarray(fit.poles_rad_per_ns, dtype=np.complex128).copy()
    # Fit the residues once more on the sampled ZOH basis.  A continuous-time
    # fit can contain large cancelling high-frequency terms; re-solving on the
    # discrete basis prevents the hold/sampling transform from changing the
    # intended response at the actual DAC rate.
    sample_rate = 1.0 / float(sample_period)
    z = np.exp(2j * np.pi * fit_frequencies / sample_rate)
    discrete_poles = np.exp(poles * float(sample_period))
    discrete_inputs = np.divide(
        np.expm1(poles * float(sample_period)),
        poles,
        out=np.full(poles.shape, float(sample_period), dtype=np.complex128),
        where=np.abs(poles) > 1e-15,
    )
    basis = np.column_stack(
        [np.ones(fit_frequencies.size, dtype=np.complex128)]
        + [input_gain / (z - pole) for input_gain, pole in zip(discrete_inputs, discrete_poles)]
    )
    discrete_coefficients, *_ = np.linalg.lstsq(basis, fit_response, rcond=None)
    discrete_fitted = basis @ discrete_coefficients
    discrete_residual_rms = float(np.sqrt(np.mean(np.abs(discrete_fitted - fit_response) ** 2)))
    discrete_condition_number = float(np.linalg.cond(basis))
    A = np.diag(poles)
    B = np.ones((fit.poles_rad_per_ns.size, 1), dtype=np.complex128)
    C = np.asarray(discrete_coefficients[1:], dtype=np.complex128).reshape(1, -1)
    D = np.asarray([[discrete_coefficients[0]]], dtype=np.complex128)
    model = discretize_zoh_state_space(
        A,
        B,
        C,
        D,
        sample_period,
        metadata={
            "design_method": "touchstone_rational_fit_zoh",
            "fit_residual_rms": fit.residual_rms,
            "fit": fit,
            "frequency_mode": "relative",
            "frequency_reference_ghz": reference if frequency_mode == "absolute" else None,
            "lo_freq_ghz": reference if frequency_mode == "absolute" else None,
            "shifted_poles_rad_per_ns": poles.copy(),
            "input_frequencies_ghz": frequencies.copy(),
            "fit_frequencies_ghz": fit_frequencies.copy(),
            "frequency_guard_ghz": float(nyquist_guard),
            "discarded_frequency_count": int(np.count_nonzero(~safe_mask)),
            "discrete_fit_residual_rms": discrete_residual_rms,
            "discrete_fit_condition_number": discrete_condition_number,
            "discrete_residues": np.asarray(discrete_coefficients[1:], dtype=np.complex128),
            "discrete_direct": complex(discrete_coefficients[0]),
            "latency_samples": int(latency_samples),
            **dict(metadata or {}),
        },
    )
    return fit, model


def fit_mimo_response_to_zoh(
    frequencies_ghz: Union[np.ndarray, Sequence[float]],
    response_matrix: np.ndarray,
    *,
    sample_period: float,
    order: int = 4,
    frequency_reference_ghz: float = 0.0,
    frequency_mode: FrequencyMode = "absolute",
    candidate_orders: Optional[Sequence[int]] = None,
    auto_order: bool = False,
    max_order: Optional[int] = None,
    holdout_fraction: float = 0.0,
    pole_relocation_iterations: int = 0,
    enforce_passivity: bool = False,
    passivity_tolerance: float = 1e-6,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[Tuple[int, int], RationalFitResult], ZOHDiscreteModel]:
    """Fit every path of an ``(frequency, output, input)`` MIMO response.

    The returned realization uses independent stable state blocks per path;
    this is exact for the fitted transfer matrix and provides a common ZOH
    state-space representation suitable for the MIMO inverse constructor.
    """
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64).reshape(-1)
    response = np.asarray(response_matrix, dtype=np.complex128)
    if frequency_mode not in ("absolute", "relative"):
        raise ValueError("frequency_mode must be 'absolute' or 'relative'.")
    if frequencies.size < 3 or np.any(~np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_ghz must contain at least three finite, strictly increasing samples.")
    if not np.isfinite(float(sample_period)) or float(sample_period) <= 0.0:
        raise ValueError("sample_period must be finite and positive.")
    if not np.isfinite(float(frequency_reference_ghz)):
        raise ValueError("frequency_reference_ghz must be finite.")
    if response.ndim != 3 or response.shape[0] != frequencies.size:
        raise ValueError("response_matrix must have shape (n_freq, n_outputs, n_inputs).")
    if response.shape[1] == 0 or response.shape[2] == 0:
        raise ValueError("response_matrix must have non-empty output and input dimensions.")
    if np.any(~np.isfinite(response)):
        raise ValueError("response_matrix must be finite.")
    fits: Dict[Tuple[int, int], RationalFitResult] = {}
    models: Dict[Tuple[int, int], ZOHDiscreteModel] = {}
    for output_index in range(response.shape[1]):
        for input_index in range(response.shape[2]):
            fit, model = fit_response_to_zoh(
                frequencies,
                response[:, output_index, input_index],
                sample_period=sample_period,
                order=order,
                frequency_reference_ghz=frequency_reference_ghz,
                frequency_mode=frequency_mode,
                candidate_orders=candidate_orders,
                auto_order=auto_order,
                max_order=max_order,
                holdout_fraction=holdout_fraction,
                pole_relocation_iterations=pole_relocation_iterations,
                enforce_passivity=enforce_passivity,
                passivity_tolerance=passivity_tolerance,
            )
            fits[(output_index, input_index)] = fit
            models[(output_index, input_index)] = model
    offsets: Dict[Tuple[int, int], int] = {}
    state_count = 0
    for key, model in models.items():
        offsets[key] = state_count
        state_count += model.n_states
    n_outputs, n_inputs = response.shape[1:]
    A = np.zeros((state_count, state_count), dtype=np.complex128)
    B = np.zeros((state_count, n_inputs), dtype=np.complex128)
    C = np.zeros((n_outputs, state_count), dtype=np.complex128)
    D = np.zeros((n_outputs, n_inputs), dtype=np.complex128)
    for (output_index, input_index), model in models.items():
        start = offsets[(output_index, input_index)]
        stop = start + model.n_states
        A[start:stop, start:stop] = model.A
        B[start:stop, input_index] = model.B[:, 0]
        C[output_index, start:stop] = model.C[0, :]
        D[output_index, input_index] = model.D[0, 0]
    model_frequency_values = (
        np.asarray(frequencies, dtype=np.float64) - float(frequency_reference_ghz)
        if frequency_mode == "absolute"
        else np.asarray(frequencies, dtype=np.float64)
    )
    model_metadata = {
        **dict(metadata or {}),
        "design_method": "mimo_pathwise_rational_fit_zoh",
        "frequency_mode": "relative",
        "frequency_reference_ghz": (
            float(frequency_reference_ghz) if frequency_mode == "absolute" else None
        ),
        "path_fit_residual_rms": {
            f"{output_index},{input_index}": float(fit.residual_rms)
            for (output_index, input_index), fit in fits.items()
        },
        "fit_frequencies_ghz": model_frequency_values.copy(),
        "input_frequencies_ghz": np.asarray(frequencies, dtype=np.float64).copy(),
        "latency_samples": 0,
    }
    return fits, ZOHDiscreteModel(A, B, C, D, float(sample_period), metadata=model_metadata)


def fit_touchstone_path_to_zoh(
    network: "TouchstoneNetwork",
    *,
    output_port: int,
    input_port: int,
    sample_period: float,
    order: int = 4,
    frequency_mask: Optional[np.ndarray] = None,
    frequency_reference_ghz: float = 0.0,
    frequency_mode: FrequencyMode = "absolute",
    candidate_orders: Optional[Sequence[int]] = None,
    auto_order: bool = False,
    max_order: Optional[int] = None,
    holdout_fraction: float = 0.0,
    pole_relocation_iterations: int = 0,
    enforce_passivity: bool = False,
    passivity_tolerance: float = 1e-6,
    latency_samples: int = 0,
) -> Tuple[RationalFitResult, ZOHDiscreteModel]:
    """Fit one Touchstone path and convert the result to a ZOH model."""
    frequencies = np.asarray(network.frequencies, dtype=np.float64)
    response = np.asarray(network.get_response(output_port, input_port), dtype=np.complex128)
    safe_mask = np.ones(frequencies.size, dtype=bool)
    if frequency_mode == "absolute":
        reference = float(frequency_reference_ghz)
        nyquist_guard = 0.45 / float(sample_period)
        safe_mask &= np.abs(frequencies - reference) <= nyquist_guard
    if frequency_mask is not None:
        mask = np.asarray(frequency_mask, dtype=bool).reshape(-1)
        if mask.size != frequencies.size:
            raise ValueError("frequency_mask must match the Touchstone frequency grid.")
        safe_mask &= mask
    if not np.any(safe_mask):
        raise ValueError(
            "No Touchstone samples remain in the requested baseband/Nyquist fitting band."
        )
    if np.count_nonzero(safe_mask) < 3:
        raise ValueError("At least three Touchstone samples are required for ZOH fitting.")
    frequencies = frequencies[safe_mask]
    response = response[safe_mask]
    return fit_response_to_zoh(
        frequencies,
        response,
        sample_period=sample_period,
        order=order,
        frequency_reference_ghz=frequency_reference_ghz,
        frequency_mode=frequency_mode,
        candidate_orders=candidate_orders,
        auto_order=auto_order,
        max_order=max_order,
        holdout_fraction=holdout_fraction,
        pole_relocation_iterations=pole_relocation_iterations,
        enforce_passivity=enforce_passivity,
        passivity_tolerance=passivity_tolerance,
        latency_samples=latency_samples,
        metadata={
            "output_port": int(output_port),
            "input_port": int(input_port),
        },
    )


def select_discrete_inverse(
    model: ZOHDiscreteModel,
    *,
    method: str = "auto",
    num_taps: int = 101,
    n_fft: int = 2048,
    regularization: float = 1e-8,
    condition_limit: float = 1e8,
    max_residual_rms: Optional[float] = None,
    max_inverse_gain: Optional[float] = None,
) -> Dict[str, Any]:
    """Select a stable strict inverse or an explicitly allowed sampled FIR.

    A strict request never changes method. An unstable forward realization is
    unavailable: an open-loop inverse is not a feedback stabilization design.
    Residuals are evaluated after removing the declared scheduling delay.
    """
    if method not in ("auto", "strict_discrete_inverse", "sampled_fir"):
        raise ValueError("method must be auto, strict_discrete_inverse, or sampled_fir.")
    if n_fft < num_taps or num_taps <= 0:
        raise ValueError("Require n_fft >= num_taps > 0.")
    for name, value in (("max_residual_rms", max_residual_rms), ("max_inverse_gain", max_inverse_gain)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative.")
    result: Dict[str, Any] = {"requested_method": method, "selected_method": None, "status": "unavailable", "reason_codes": []}
    forward_poles = np.linalg.eigvals(model.A)
    if forward_poles.size and np.max(np.abs(forward_poles)) >= 1.0 - 1e-8:
        result["reason_codes"].append("unstable_forward")
        return result
    frequencies = np.fft.fftfreq(n_fft, d=model.sample_period)
    forward = model.response(frequencies)
    siso = model.n_inputs == model.n_outputs == 1
    methods = ("strict_discrete_inverse", "sampled_fir") if method == "auto" else (method,)
    for candidate in methods:
        try:
            if candidate == "strict_discrete_inverse":
                if siso:
                    design = design_strict_discrete_inverse(model)
                    correction = evaluate_iir_response(design.numerator, design.denominator, frequencies, model.sample_rate)[:, None, None]
                else:
                    design = design_mimo_discrete_inverse(model, frequencies=frequencies, condition_limit=condition_limit)
                    correction = design.model.response(frequencies)
                delay = design.delay_samples if siso else design.latency_samples
            else:
                design = (
                    design_sampled_fir_inverse(forward[:, 0, 0], sample_rate=model.sample_rate, num_taps=num_taps, regularization=regularization)
                    if siso else design_mimo_sampled_fir_inverse(forward, sample_rate=model.sample_rate, num_taps=num_taps, regularization=regularization)
                )
                correction = (
                    np.fft.fft(design.kernel, n_fft)[:, None, None]
                    if siso else np.fft.fft(design.kernels, n_fft, axis=-1).transpose(2, 0, 1)
                )
                delay = design.delay_samples
            correction = correction * np.exp(2j * np.pi * frequencies * delay / model.sample_rate)[:, None, None]
            residual = np.matmul(forward, correction) - np.eye(model.n_outputs)[None, :, :]
            residual_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
            inverse_gain = float(np.max(np.linalg.svd(correction, compute_uv=False)))
            if not np.isfinite(residual_rms) or not np.isfinite(inverse_gain):
                raise ValueError("nonfinite_inverse_response")
            if max_residual_rms is not None and residual_rms > max_residual_rms:
                raise ValueError("inverse_residual_limit_exceeded")
            if max_inverse_gain is not None and inverse_gain > max_inverse_gain:
                raise ValueError("inverse_gain_limit_exceeded")
        except (ValueError, np.linalg.LinAlgError) as error:
            result["reason_codes"].append(f"{candidate}: {error}")
            continue
        result.update(status="available", selected_method=candidate, design=design,
                      residual_rms=residual_rms, inverse_gain_peak=inverse_gain,
                      delay_samples=int(delay), fallback_used=candidate != methods[0])
        return result
    return result


def compare_stage_state_modes(
    stage_factory: Any,
    traces: Sequence[SignalTrace],
    *,
    inter_segment_gap_samples: int = 0,
) -> Dict[str, Any]:
    """Run a short trace sequence with reset and retained state semantics.

    ``stage_factory`` may be a callable returning a fresh stage or an already
    configured stage exposing ``fresh``.  The result contains per-segment RMS,
    boundary jumps, and final-state norms, making state carry-over an explicit
    experiment variable rather than an implicit optimizer side effect.
    """
    if not traces:
        raise ValueError("traces must contain at least one SignalTrace.")
    gap_samples = int(inter_segment_gap_samples)
    if gap_samples < 0:
        raise ValueError("inter_segment_gap_samples must be non-negative.")
    reference_trace = traces[0]
    _validate_stage_trace_contract(reference_trace, stage_name="state_mode_probe")
    for trace in traces[1:]:
        _validate_stage_trace_contract(
            trace,
            expected_sample_rate=reference_trace.sample_rate,
            expected_lo_freq_ghz=(
                reference_trace.lo_freq
                if reference_trace.domain == "iq_complex"
                else None
            ),
            stage_name="state_mode_probe",
        )
        if trace.domain != reference_trace.domain or trace.plane != reference_trace.plane:
            raise ValueError("state-mode probe traces must share one domain and plane.")
        if trace.t_axis.shape != reference_trace.t_axis.shape or not np.allclose(
            trace.t_axis, reference_trace.t_axis
        ):
            raise ValueError("state-mode probe traces must share one t_axis.")

    gap_trace = None
    if gap_samples:
        dt = 1.0 / float(reference_trace.sample_rate)
        gap_axis = reference_trace.t_axis[-1] + dt * np.arange(
            1, gap_samples + 1, dtype=np.float64
        )
        gap_trace = reference_trace.clone(
            t_axis=gap_axis,
            values=np.zeros(gap_samples, dtype=np.complex128),
            label=f"{reference_trace.label}_inter_segment_gap",
        )

    def make_stage() -> Any:
        if callable(stage_factory):
            return stage_factory()
        fresh = getattr(stage_factory, "fresh", None)
        if callable(fresh):
            return fresh()
        import copy

        return copy.deepcopy(stage_factory)

    result: Dict[str, Any] = {}
    for mode, retain in (("reset", False), ("retain", True)):
        stage = make_stage()
        if hasattr(stage, "retain_state"):
            stage.retain_state = retain
            if hasattr(stage, "is_lti"):
                stage.is_lti = not retain
        reset = getattr(stage, "reset", None)
        if callable(reset):
            reset()
        segments: list[dict[str, float | int]] = []
        previous_last: complex | None = None
        for index, trace in enumerate(traces):
            gap_state_norm = 0.0
            if index and gap_trace is not None:
                if not retain and callable(reset):
                    reset()
                gap_output = stage.apply(gap_trace)
                if len(gap_output.values):
                    previous_last = complex(gap_output.values[-1])
                gap_state = getattr(stage, "snapshot_state", lambda: None)()
                gap_state_norm = (
                    0.0
                    if gap_state is None
                    else float(np.linalg.norm(np.asarray(gap_state)))
                )
            if not retain and callable(reset):
                reset()
            state_before = getattr(stage, "snapshot_state", lambda: None)()
            state_before_norm = (
                0.0
                if state_before is None
                else float(np.linalg.norm(np.asarray(state_before)))
            )
            output = stage.apply(trace)
            values = np.asarray(output.values, dtype=np.complex128)
            boundary_jump = 0.0 if previous_last is None else float(abs(values[0] - previous_last))
            state = getattr(stage, "snapshot_state", lambda: None)()
            state_norm = 0.0 if state is None else float(np.linalg.norm(np.asarray(state)))
            segments.append(
                {
                    "segment": int(index),
                    "output_rms": float(np.sqrt(np.mean(np.abs(values) ** 2))),
                    "output_peak": float(np.max(np.abs(values))) if values.size else 0.0,
                    "tail_rms": output.metadata.get("tail_rms"),
                    "tail_observation_samples": output.metadata.get("tail_observation_samples"),
                    "tail_settled": output.metadata.get("tail_settled"),
                    "boundary_jump": boundary_jump,
                    "start_state_norm": state_before_norm,
                    "gap_state_norm": gap_state_norm,
                    "final_state_norm": state_norm,
                }
            )
            previous_last = complex(values[-1]) if values.size else previous_last
        result[mode] = {
            "segments": segments,
            "final_state_norm": segments[-1]["final_state_norm"] if segments else 0.0,
        }
    return result

__all__ = [
    "AllPoleDerivativeModel",
    "DerivativeScheme",
    "DerivativePrecorrectionDesign",
    "DerivativePrecorrectionStage",
    "DiscreteStateSpaceStage",
    "FIRInverseDesign",
    "FIRFilterStage",
    "RationalFitResult",
    "MIMOFIRInverseDesign",
    "MIMOFIRFilterStage",
    "MIMODiscreteInverseDesign",
    "MIMODiscreteInverseStage",
    "FrequencyMode",
    "StateMode",
    "StrictDiscreteInverseDesign",
    "StrictDiscreteInverseStage",
    "ZOHDiscreteModel",
    "apply_derivative_precorrection",
    "apply_stage_sequence",
    "waveform_tail_diagnostics",
    "calculate_inverse_response_diagnostics",
    "check_discrete_inverse_realizability",
    "check_mimo_discrete_inverse_realizability",
    "check_touchstone_network_physicality",
    "compare_stage_state_modes",
    "select_discrete_inverse",
    "compute_derivative_basis",
    "design_all_pole_derivative_precorrection",
    "design_derivative_precorrection",
    "design_inverse_fir_from_touchstone",
    "design_mimo_sampled_fir_inverse",
    "design_mimo_discrete_inverse",
    "design_sampled_fir_inverse",
    "design_strict_discrete_inverse",
    "discretize_zoh_state_space",
    "discretize_zoh_transfer_function",
    "evaluate_discrete_state_space_response",
    "evaluate_derivative_polynomial_response",
    "evaluate_fir_response",
    "evaluate_iir_response",
    "fit_touchstone_path_to_zoh",
    "fit_mimo_response_to_zoh",
    "fit_response_to_zoh",
    "fit_touchstone_rational",
    "propagate_zoh_discrete",
]

