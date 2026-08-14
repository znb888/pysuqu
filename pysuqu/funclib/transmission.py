"""Transmission-chain primitives for control waveform propagation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
from scipy.signal import butter, convolve, firwin, get_window, lfilter, sosfilt


SignalDomain = Literal["iq_complex", "rf_real"]
SignalPlane = Literal["baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf"]
StageDomain = Literal["iq_complex", "rf_real", "any"]
FilterKind = Literal["lowpass", "highpass", "bandpass", "bandstop"]
FIRAlignment = Literal["leading", "centered"]
TouchstoneInterpolation = Literal["cartesian", "polar"]
OutOfBandPolicy = Literal["edge", "zero", "error"]

_VALID_DOMAINS = {"iq_complex", "rf_real"}
_VALID_PLANES = {"baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf"}
_ALL_PLANES = ("baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf")
_TOUCHSTONE_FREQ_SCALES = {
    "hz": 1e-9,
    "khz": 1e-6,
    "mhz": 1e-3,
    "ghz": 1.0,
}
_SUPPORTED_TWO_PORT_DATA_ORDERS = {"21_12", "12_21"}


def _as_array(values: Union[np.ndarray, Iterable[complex]]) -> np.ndarray:
    return np.asarray(values)


def _next_fft_length(signal_length: int, impulse_length: Optional[int] = None) -> int:
    effective_impulse_length = impulse_length if impulse_length is not None else signal_length
    target = max(1, signal_length + max(1, effective_impulse_length) - 1)
    fft_length = 1
    while fft_length < target:
        fft_length <<= 1
    return fft_length


def _normalize_filter_kind(kind: str) -> FilterKind:
    mapping = {
        "low": "lowpass",
        "lowpass": "lowpass",
        "high": "highpass",
        "highpass": "highpass",
        "band": "bandpass",
        "bandpass": "bandpass",
        "stop": "bandstop",
        "bandstop": "bandstop",
        "notch": "bandstop",
    }
    normalized = mapping.get(kind.lower())
    if normalized is None:
        raise ValueError(f"Unsupported filter kind: {kind}")
    return normalized


def _normalize_cutoff(
    cutoff_freq: Union[float, Tuple[float, float], np.ndarray],
    kind: FilterKind,
) -> Union[float, Tuple[float, float]]:
    values = np.asarray(cutoff_freq, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Cutoff frequencies must be positive and finite.")

    if kind in ("lowpass", "highpass"):
        if len(values) != 1:
            raise ValueError(f"{kind} expects a single cutoff frequency.")
        return float(values[0])

    if len(values) != 2:
        raise ValueError(f"{kind} expects a pair of cutoff frequencies.")
    if values[0] >= values[1]:
        raise ValueError("Band edges must be strictly increasing.")
    return float(values[0]), float(values[1])


def _validate_positive_sample_rate(sample_rate: float) -> None:
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite.")


def _validate_sample_rate_and_cutoff(
    sample_rate: float,
    cutoff_freq: Union[float, Tuple[float, float]],
) -> None:
    _validate_positive_sample_rate(sample_rate)
    nyquist = sample_rate / 2.0
    cutoff_values = np.asarray(cutoff_freq, dtype=np.float64).reshape(-1)
    if np.any(cutoff_values >= nyquist):
        raise ValueError(
            f"Cutoff frequencies must stay below the Nyquist frequency ({nyquist})."
        )


def _validate_filter_sample_rate(
    design_sample_rate: Optional[float],
    trace_sample_rate: float,
    stage_name: str,
) -> None:
    if design_sample_rate is None:
        return
    if not np.isclose(design_sample_rate, trace_sample_rate, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{stage_name} was designed for sample_rate={design_sample_rate}, "
            f"received trace sample_rate={trace_sample_rate}."
        )


def _resolve_derivative_orders(
    derivative_orders: Optional[Iterable[int]],
    coefficient_count: int,
) -> tuple[int, ...]:
    if coefficient_count < 0:
        raise ValueError("coefficient_count must be non-negative.")
    if derivative_orders is None:
        return tuple(range(1, coefficient_count + 1))

    raw_orders = tuple(derivative_orders)
    if len(raw_orders) != coefficient_count:
        raise ValueError(
            "derivative_orders must contain exactly one order for each coefficient."
        )

    orders = []
    for order in raw_orders:
        if isinstance(order, (bool, np.bool_)) or not isinstance(
            order,
            (int, np.integer),
        ):
            raise ValueError("Derivative orders must be positive integers.")
        orders.append(int(order))

    if any(order <= 0 for order in orders):
        raise ValueError("Derivative orders must be positive integers.")
    if len(set(orders)) != len(orders):
        raise ValueError("Derivative orders must be unique.")
    return tuple(orders)


def _infer_touchstone_port_count(path: Union[str, Path]) -> int:
    suffix = Path(path).suffix.lower()
    if not suffix.startswith(".s") or not suffix.endswith("p"):
        raise ValueError(
            f"Touchstone path must end with .sNp, received suffix {Path(path).suffix!r}."
        )
    digits = suffix[2:-1]
    if not digits.isdigit():
        raise ValueError(f"Unable to infer port count from Touchstone suffix {suffix!r}.")
    port_count = int(digits)
    if port_count <= 0:
        raise ValueError("Touchstone port count must be positive.")
    return port_count


def _read_text_with_fallbacks(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("touchstone", b"", 0, 1, f"Unable to decode {path}")


def _parse_touchstone_option_line(
    line: str,
) -> Tuple[str, str, str, Union[float, np.ndarray]]:
    tokens = line[1:].strip().split()
    if not tokens:
        return "ghz", "s", "ma", 50.0

    freq_unit = tokens[0].lower()
    parameter = tokens[1].lower() if len(tokens) > 1 else "s"
    data_format = tokens[2].lower() if len(tokens) > 2 else "ma"
    reference: Union[float, np.ndarray] = 50.0

    for idx, token in enumerate(tokens):
        if token.lower() == "r":
            if idx + 1 >= len(tokens):
                raise ValueError("Touchstone option line has R without a reference value.")
            reference = float(tokens[idx + 1])
            break

    if freq_unit not in _TOUCHSTONE_FREQ_SCALES:
        raise ValueError(f"Unsupported Touchstone frequency unit: {freq_unit}")
    if parameter != "s":
        raise ValueError(
            f"Only S-parameter Touchstone files are supported, received {parameter!r}."
        )
    if data_format not in ("ri", "ma", "db"):
        raise ValueError(f"Unsupported Touchstone data format: {data_format}")

    return freq_unit, parameter, data_format, reference


def _touchstone_pairs_to_complex(
    raw_pairs: np.ndarray,
    data_format: str,
) -> np.ndarray:
    if data_format == "ri":
        return raw_pairs[..., 0] + 1j * raw_pairs[..., 1]

    magnitude = raw_pairs[..., 0]
    phase_rad = np.deg2rad(raw_pairs[..., 1])
    if data_format == "db":
        magnitude = 10 ** (magnitude / 20.0)
    return magnitude * np.exp(1j * phase_rad)


def _touchstone_pair_order(
    port_count: int,
    *,
    two_port_data_order: str = "21_12",
) -> list[tuple[int, int]]:
    if port_count == 2:
        normalized = two_port_data_order.lower().replace("-", "_")
        if normalized not in _SUPPORTED_TWO_PORT_DATA_ORDERS:
            raise ValueError(
                f"Unsupported [Two-Port Data Order]: {two_port_data_order!r}. "
                f"Supported values: {sorted(_SUPPORTED_TWO_PORT_DATA_ORDERS)}"
            )
        if normalized == "12_21":
            return [(0, 0), (0, 1), (1, 0), (1, 1)]
        return [(0, 0), (1, 0), (0, 1), (1, 1)]

    return [
        (output_port, input_port)
        for input_port in range(port_count)
        for output_port in range(port_count)
    ]


def _interpolate_complex_response(
    query_freq: np.ndarray,
    reference_freq: np.ndarray,
    reference_response: np.ndarray,
    *,
    interpolation: TouchstoneInterpolation,
    out_of_band: OutOfBandPolicy,
    stage_name: str,
) -> np.ndarray:
    if interpolation not in ("cartesian", "polar"):
        raise ValueError(f"Unsupported Touchstone interpolation mode: {interpolation}")
    if out_of_band not in ("edge", "zero", "error"):
        raise ValueError(f"Unsupported Touchstone out_of_band policy: {out_of_band}")

    query = np.asarray(query_freq, dtype=np.float64)
    query_shape = query.shape
    query_values = query.reshape(-1)
    freq = np.asarray(reference_freq, dtype=np.float64)
    response = np.asarray(reference_response, dtype=np.complex128)
    if np.any(~np.isfinite(query_values)):
        raise ValueError(f"{stage_name} query frequencies must be finite.")
    if len(freq) == 0:
        raise ValueError(f"{stage_name} cannot interpolate an empty Touchstone response.")

    below = query_values < freq[0]
    above = query_values > freq[-1]
    if out_of_band == "error" and np.any(below | above):
        raise ValueError(
            f"{stage_name} queried frequencies outside the measured Touchstone span "
            f"[{freq[0]}, {freq[-1]}]."
        )

    if interpolation == "cartesian":
        if out_of_band == "zero":
            left_real = left_imag = right_real = right_imag = 0.0
        else:
            left_real = float(np.real(response[0]))
            left_imag = float(np.imag(response[0]))
            right_real = float(np.real(response[-1]))
            right_imag = float(np.imag(response[-1]))

        real_part = np.interp(
            query_values,
            freq,
            np.real(response),
            left=left_real,
            right=right_real,
        )
        imag_part = np.interp(
            query_values,
            freq,
            np.imag(response),
            left=left_imag,
            right=right_imag,
        )
        return np.asarray(real_part + 1j * imag_part).reshape(query_shape)

    magnitude = np.abs(response)
    phase = np.unwrap(np.angle(response))
    if out_of_band == "zero":
        left_mag = right_mag = 0.0
    else:
        left_mag = float(magnitude[0])
        right_mag = float(magnitude[-1])

    interp_mag = np.interp(
        query_values,
        freq,
        magnitude,
        left=left_mag,
        right=right_mag,
    )
    interp_phase = np.interp(
        query_values,
        freq,
        phase,
        left=float(phase[0]),
        right=float(phase[-1]),
    )
    return np.asarray(interp_mag * np.exp(1j * interp_phase)).reshape(query_shape)


def _normalize_output_values(
    values: np.ndarray,
    domain: StageDomain,
    stage_name: str,
) -> np.ndarray:
    if domain == "rf_real":
        normalized = np.real_if_close(values, tol=1000)
        if np.iscomplexobj(normalized):
            raise ValueError(
                f"{stage_name} produced complex values for an rf_real trace. "
                "Use iq_complex traces or make the stage response Hermitian."
            )
        return np.asarray(normalized, dtype=np.float64)

    return np.asarray(values, dtype=np.complex128)


def _enforce_real_self_conjugate_bins(
    response: np.ndarray,
    freq_axis: np.ndarray,
) -> np.ndarray:
    """Force FFT bins that must equal their own conjugate to be purely real."""
    normalized = np.asarray(response, dtype=np.complex128).copy()
    freq = np.asarray(freq_axis, dtype=np.float64)

    if len(freq) == 0:
        return normalized

    self_conjugate_mask = np.isclose(freq, 0.0)
    if len(freq) % 2 == 0 and len(freq) > 1:
        sample_rate = len(freq) * abs(freq[1] - freq[0])
        nyquist_freq = 0.5 * sample_rate
        self_conjugate_mask |= np.isclose(np.abs(freq), nyquist_freq)

    normalized[..., self_conjugate_mask] = np.real(
        normalized[..., self_conjugate_mask]
    )
    return normalized


def _describe_stage(stage: Any) -> str:
    describe = getattr(stage, "describe", None)
    if callable(describe):
        return str(describe())
    return str(getattr(stage, "name", stage.__class__.__name__))


@dataclass
class SignalTrace:
    """Sampled electrical waveform plus metadata about its reference plane."""

    t_axis: np.ndarray
    values: np.ndarray
    sample_rate: float
    domain: SignalDomain
    plane: SignalPlane
    lo_freq: float = 0.0
    label: str = "signal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_axis = np.asarray(self.t_axis, dtype=np.float64)
        self.values = _as_array(self.values)

        if self.t_axis.ndim != 1 or self.values.ndim != 1:
            raise ValueError("SignalTrace expects 1D t_axis and values arrays.")
        if len(self.t_axis) != len(self.values):
            raise ValueError(
                f"SignalTrace length mismatch: t_axis={len(self.t_axis)}, "
                f"values={len(self.values)}"
            )
        if self.sample_rate <= 0:
            raise ValueError("SignalTrace.sample_rate must be positive.")
        if self.domain not in _VALID_DOMAINS:
            raise ValueError(f"Unsupported signal domain: {self.domain}")
        if self.plane not in _VALID_PLANES:
            raise ValueError(f"Unsupported signal plane: {self.plane}")

        if self.domain == "rf_real":
            self.values = _normalize_output_values(self.values, self.domain, "SignalTrace")
        else:
            self.values = np.asarray(self.values, dtype=np.complex128)

    def clone(self, **changes: Any) -> "SignalTrace":
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)


def _validate_derivative_options(
    sample_period: float,
    edge_order: int,
    normalization_epsilon: float,
) -> None:
    if not np.isfinite(sample_period) or sample_period <= 0:
        raise ValueError("sample_period must be positive and finite.")
    if edge_order not in (1, 2):
        raise ValueError("edge_order must be 1 or 2.")
    if not np.isfinite(normalization_epsilon) or normalization_epsilon <= 0:
        raise ValueError("normalization_epsilon must be positive and finite.")


def compute_derivative_basis(
    values: Union[np.ndarray, Iterable[complex]],
    sample_period: float,
    derivative_orders: Sequence[int],
    *,
    normalize_to_peak: bool = False,
    reference_peak: Optional[float] = None,
    edge_order: int = 2,
    normalization_epsilon: float = 1e-15,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build finite-difference derivative basis waveforms in requested order."""
    base_values = np.asarray(values, dtype=np.complex128)
    if base_values.ndim != 1:
        raise ValueError("compute_derivative_basis expects a 1D waveform.")
    if np.any(~np.isfinite(base_values)):
        raise ValueError("Derivative basis input values must be finite.")
    _validate_derivative_options(sample_period, edge_order, normalization_epsilon)

    raw_orders = tuple(derivative_orders)
    orders = _resolve_derivative_orders(raw_orders, len(raw_orders))
    if not orders:
        return [], np.array([], dtype=np.float64)
    if len(base_values) < 2:
        raise ValueError("Derivative basis generation requires at least two samples.")

    resolved_edge_order = 2 if edge_order == 2 and len(base_values) >= 3 else 1
    current = base_values.copy()
    basis_by_order: dict[int, np.ndarray] = {}
    for order in range(1, max(orders) + 1):
        current = np.gradient(
            current,
            sample_period,
            edge_order=resolved_edge_order,
        )
        basis_by_order[order] = np.asarray(current, dtype=np.complex128)

    if reference_peak is None:
        target_peak = float(np.max(np.abs(base_values)))
    else:
        target_peak = float(reference_peak)
        if not np.isfinite(target_peak) or target_peak < 0:
            raise ValueError("reference_peak must be non-negative and finite.")

    basis_list = []
    basis_scales = []
    for order in orders:
        basis = basis_by_order[order]
        scale = 1.0
        if normalize_to_peak:
            basis_peak = float(np.max(np.abs(basis)))
            if (
                basis_peak <= normalization_epsilon
                or target_peak <= normalization_epsilon
            ):
                scale = 0.0
            else:
                scale = target_peak / basis_peak
        basis_list.append(np.asarray(basis * scale, dtype=np.complex128))
        basis_scales.append(float(scale))

    return basis_list, np.asarray(basis_scales, dtype=np.float64)


def apply_derivative_precorrection(
    values: Union[np.ndarray, Iterable[complex]],
    sample_period: float,
    coefficients: Union[np.ndarray, Iterable[complex]],
    *,
    derivative_orders: Optional[Sequence[int]] = None,
    normalize_to_peak: bool = True,
    phase_rad: float = 0.0,
    edge_order: int = 2,
    normalization_epsilon: float = 1e-15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add weighted derivative terms and an optional phase to one waveform."""
    base_values = np.asarray(values, dtype=np.complex128)
    if base_values.ndim != 1:
        raise ValueError("apply_derivative_precorrection expects a 1D waveform.")
    if np.any(~np.isfinite(base_values)):
        raise ValueError("Precorrection input values must be finite.")
    _validate_derivative_options(sample_period, edge_order, normalization_epsilon)
    if not np.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite.")

    coeff_array = np.asarray(tuple(coefficients), dtype=np.complex128).reshape(-1)
    if np.any(~np.isfinite(coeff_array)):
        raise ValueError("Derivative coefficients must be finite.")
    orders = _resolve_derivative_orders(derivative_orders, len(coeff_array))
    basis_list, basis_scales = compute_derivative_basis(
        base_values,
        sample_period,
        orders,
        normalize_to_peak=normalize_to_peak,
        edge_order=edge_order,
        normalization_epsilon=normalization_epsilon,
    )

    corrected = base_values.copy()
    for coefficient, basis in zip(coeff_array, basis_list):
        corrected += coefficient * basis
    if abs(phase_rad) >= normalization_epsilon:
        corrected *= np.exp(1j * float(phase_rad))

    metadata = {
        "derivative_orders": list(orders),
        "basis_scales": [float(scale) for scale in basis_scales],
        "phase_rad": float(phase_rad),
        "normalize_to_peak": bool(normalize_to_peak),
        "coefficients_real": [float(np.real(value)) for value in coeff_array],
        "coefficients_imag": [float(np.imag(value)) for value in coeff_array],
    }
    return np.asarray(corrected, dtype=np.complex128), metadata


@dataclass(frozen=True)
class DerivativePrecorrectionDesign:
    """Least-squares design result for derivative-polynomial precorrection."""

    coefficients: np.ndarray
    derivative_orders: tuple[int, ...] = ()
    phase_rad: float = 0.0
    normalize_to_peak: bool = True
    edge_order: int = 2
    basis_scales: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    spectral_weights: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    residual_rms: float = 0.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex128).reshape(-1)
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("Derivative coefficients must be finite.")
        raw_orders = tuple(self.derivative_orders)
        orders = _resolve_derivative_orders(
            raw_orders if raw_orders else None,
            len(coefficients),
        )
        basis_scales = np.asarray(self.basis_scales, dtype=np.float64).reshape(-1)
        spectral_weights = np.asarray(
            self.spectral_weights,
            dtype=np.float64,
        ).reshape(-1)
        if len(basis_scales) not in (0, len(coefficients)):
            raise ValueError("basis_scales must be empty or match coefficients length.")
        if np.any(~np.isfinite(basis_scales)):
            raise ValueError("basis_scales must be finite.")
        if np.any(~np.isfinite(spectral_weights)) or np.any(spectral_weights < 0):
            raise ValueError("spectral_weights must be non-negative and finite.")
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite.")
        if self.edge_order not in (1, 2):
            raise ValueError("edge_order must be 1 or 2.")
        if not np.isfinite(self.residual_rms) or self.residual_rms < 0:
            raise ValueError("residual_rms must be non-negative and finite.")

        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "derivative_orders", orders)
        object.__setattr__(self, "normalize_to_peak", bool(self.normalize_to_peak))
        object.__setattr__(self, "basis_scales", basis_scales)
        object.__setattr__(self, "spectral_weights", spectral_weights)


def design_derivative_precorrection(
    trace: SignalTrace,
    response_values: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    derivative_orders: Sequence[int],
    normalize_to_peak: bool = True,
    spectral_weight_power: float = 1.0,
    ridge: float = 1e-9,
    include_global_phase: bool = True,
    edge_order: int = 2,
    normalization_epsilon: float = 1e-15,
) -> DerivativePrecorrectionDesign:
    """Fit derivative weights that approximately invert frequency responses."""
    if not isinstance(trace, SignalTrace):
        raise TypeError("design_derivative_precorrection expects a SignalTrace.")
    if trace.domain != "iq_complex":
        raise ValueError("Derivative precorrection design requires an iq_complex trace.")
    if not np.isfinite(spectral_weight_power) or spectral_weight_power < 0:
        raise ValueError("spectral_weight_power must be non-negative and finite.")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be non-negative and finite.")
    _validate_derivative_options(
        1.0 / trace.sample_rate,
        edge_order,
        normalization_epsilon,
    )

    response_array = np.asarray(response_values, dtype=np.complex128)
    if response_array.ndim == 1:
        response_array = response_array[np.newaxis, :]
    if (
        response_array.ndim != 2
        or response_array.shape[0] == 0
        or response_array.shape[1] == 0
    ):
        raise ValueError(
            "response_values must have shape (n_freq,) or (n_paths, n_freq)."
        )
    if np.any(~np.isfinite(response_array)):
        raise ValueError("response_values must be finite.")

    raw_orders = tuple(derivative_orders)
    if not raw_orders:
        raise ValueError("derivative_orders must not be empty.")
    orders = _resolve_derivative_orders(raw_orders, len(raw_orders))
    basis_list, basis_scales = compute_derivative_basis(
        trace.values,
        1.0 / trace.sample_rate,
        orders,
        normalize_to_peak=normalize_to_peak,
        edge_order=edge_order,
        normalization_epsilon=normalization_epsilon,
    )

    fft_length = response_array.shape[1]
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

    fit_rows = []
    fit_targets = []
    for response in response_array:
        columns = [
            spectral_weights * response * basis_fft
            for basis_fft in basis_ffts
        ]
        fit_rows.append(np.column_stack(columns))
        fit_targets.append(spectral_weights * (target_fft - response * target_fft))
    fit_matrix = np.vstack(fit_rows)
    fit_target = np.concatenate(fit_targets)
    if ridge > 0:
        fit_matrix = np.vstack(
            [
                fit_matrix,
                np.sqrt(float(ridge))
                * np.eye(len(orders), dtype=np.complex128),
            ]
        )
        fit_target = np.concatenate(
            [fit_target, np.zeros(len(orders), dtype=np.complex128)]
        )

    coefficients, *_ = np.linalg.lstsq(fit_matrix, fit_target, rcond=None)
    predistorted, _ = apply_derivative_precorrection(
        trace.values,
        1.0 / trace.sample_rate,
        coefficients,
        derivative_orders=orders,
        normalize_to_peak=normalize_to_peak,
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

    corrected_fft = (
        np.exp(1j * phase_rad)
        * predistorted_fft[np.newaxis, :]
        * response_array
    )
    residual = (
        spectral_weights[np.newaxis, :]
        * (corrected_fft - target_fft[np.newaxis, :])
    ).reshape(-1)
    residual_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    return DerivativePrecorrectionDesign(
        coefficients=coefficients,
        derivative_orders=orders,
        phase_rad=phase_rad,
        normalize_to_peak=normalize_to_peak,
        edge_order=edge_order,
        basis_scales=basis_scales,
        spectral_weights=spectral_weights,
        residual_rms=residual_rms,
    )


@dataclass
class SignalBundle:
    """Aligned collection of traces used for multi-line and MIMO propagation."""

    traces: Dict[str, SignalTrace]
    order: Tuple[str, ...] = ()
    label: str = "bundle"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.traces = dict(self.traces)
        if not self.traces:
            raise ValueError("SignalBundle requires at least one trace.")

        for name, trace in self.traces.items():
            if not isinstance(name, str) or not name:
                raise ValueError("SignalBundle trace names must be non-empty strings.")
            if not isinstance(trace, SignalTrace):
                raise TypeError(
                    f"SignalBundle trace {name!r} must be a SignalTrace instance."
                )

        if self.order:
            order = tuple(self.order)
            if len(order) != len(self.traces) or set(order) != set(self.traces):
                raise ValueError(
                    "SignalBundle.order must contain each trace name exactly once."
                )
        else:
            order = tuple(self.traces.keys())
        self.order = order

        reference = self.traces[self.order[0]]
        for name in self.order:
            trace = self.traces[name]
            if trace.domain != reference.domain:
                raise ValueError("All SignalBundle traces must share the same domain.")
            if trace.plane != reference.plane:
                raise ValueError("All SignalBundle traces must share the same plane.")
            if not np.isclose(
                trace.sample_rate,
                reference.sample_rate,
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "All SignalBundle traces must share the same sample_rate."
                )
            if len(trace.t_axis) != len(reference.t_axis) or not np.allclose(
                trace.t_axis,
                reference.t_axis,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("All SignalBundle traces must share the same t_axis.")

    @property
    def names(self) -> Tuple[str, ...]:
        """Return trace names in propagation order."""
        return self.order

    @property
    def domain(self) -> SignalDomain:
        """Return the shared signal domain."""
        return self.traces[self.order[0]].domain

    @property
    def plane(self) -> SignalPlane:
        """Return the shared reference plane."""
        return self.traces[self.order[0]].plane

    @property
    def sample_rate(self) -> float:
        """Return the shared sample rate in samples/ns."""
        return float(self.traces[self.order[0]].sample_rate)

    @property
    def t_axis(self) -> np.ndarray:
        """Return the shared time axis in ns."""
        return self.traces[self.order[0]].t_axis

    @property
    def shared_lo_freq(self) -> Optional[float]:
        """Return the common LO in GHz, or ``None`` when traces differ."""
        lo_freqs = np.array(
            [self.traces[name].lo_freq for name in self.order],
            dtype=np.float64,
        )
        if np.all(np.isfinite(lo_freqs)) and np.allclose(
            lo_freqs,
            lo_freqs[0],
            rtol=1e-12,
            atol=1e-12,
        ):
            return float(lo_freqs[0])
        return None

    def __getitem__(self, name: str) -> SignalTrace:
        return self.traces[name]

    def clone(self, **changes: Any) -> "SignalBundle":
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)

    def describe(self) -> str:
        """Return a concise bundle summary."""
        return f"{self.label} ({len(self.order)} trace(s)): " + ", ".join(self.order)


@dataclass
class TouchstoneNetwork:
    """Parsed Touchstone S-parameter data in GHz simulation units."""

    frequencies: np.ndarray
    s_parameters: np.ndarray
    reference: Union[float, np.ndarray] = 50.0
    path: str = ""

    def __post_init__(self) -> None:
        self.frequencies = np.asarray(self.frequencies, dtype=np.float64)
        self.s_parameters = np.asarray(self.s_parameters, dtype=np.complex128)

        if self.frequencies.ndim != 1 or len(self.frequencies) == 0:
            raise ValueError(
                "TouchstoneNetwork.frequencies must be a non-empty 1D array."
            )
        if np.any(~np.isfinite(self.frequencies)):
            raise ValueError("TouchstoneNetwork frequencies must be finite.")
        if self.s_parameters.ndim != 3:
            raise ValueError(
                "TouchstoneNetwork.s_parameters must have shape "
                "(n_freq, n_ports, n_ports)."
            )
        if self.s_parameters.shape[0] != len(self.frequencies):
            raise ValueError(
                "TouchstoneNetwork frequency and S-parameter lengths do not match."
            )
        if self.s_parameters.shape[1] == 0:
            raise ValueError("TouchstoneNetwork must describe at least one port.")
        if self.s_parameters.shape[1] != self.s_parameters.shape[2]:
            raise ValueError(
                "TouchstoneNetwork expects a square S-parameter matrix at each frequency."
            )
        if np.any(~np.isfinite(self.s_parameters)):
            raise ValueError("TouchstoneNetwork S-parameters must be finite.")
        if np.any(np.diff(self.frequencies) <= 0):
            raise ValueError("TouchstoneNetwork frequencies must be strictly increasing.")

        reference = np.asarray(self.reference, dtype=np.float64)
        if reference.ndim == 0:
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError("TouchstoneNetwork reference impedance must be positive.")
            self.reference = float(reference)
        elif reference.ndim == 1 and len(reference) == self.n_ports:
            if np.any(~np.isfinite(reference)) or np.any(reference <= 0):
                raise ValueError(
                    "TouchstoneNetwork reference impedances must be positive and finite."
                )
            self.reference = reference
        else:
            raise ValueError(
                "TouchstoneNetwork reference must be scalar or contain one value per port."
            )

    @property
    def n_ports(self) -> int:
        """Return the number of ports described by the network."""
        return int(self.s_parameters.shape[1])

    def get_response(self, output_port: int, input_port: int) -> np.ndarray:
        """Return one selected one-based ``S[output, input]`` response."""
        if not (1 <= input_port <= self.n_ports):
            raise ValueError(
                f"input_port must be within [1, {self.n_ports}], received {input_port}."
            )
        if not (1 <= output_port <= self.n_ports):
            raise ValueError(
                f"output_port must be within [1, {self.n_ports}], received {output_port}."
            )
        return self.s_parameters[:, output_port - 1, input_port - 1]


def load_touchstone_network(file_path: Union[str, Path]) -> TouchstoneNetwork:
    """Load a full-matrix Touchstone ``.sNp`` file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix_port_count = _infer_touchstone_port_count(path)
    port_count = suffix_port_count
    freq_unit = "ghz"
    data_format = "ma"
    reference: Union[float, np.ndarray] = 50.0
    matrix_format = "full"
    two_port_data_order = "21_12"
    declared_frequency_count: Optional[int] = None
    numeric_tokens: list[float] = []
    saw_network_data_keyword = False

    for raw_line in _read_text_with_fallbacks(path).splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("#"):
            freq_unit, _, data_format, reference = _parse_touchstone_option_line(line)
            continue

        if lower.startswith("[") and "]" in lower:
            keyword_end = lower.index("]")
            keyword = lower[1:keyword_end].strip()
            payload = line[keyword_end + 1 :].strip()

            if keyword == "number of ports":
                port_count = int(payload)
                if port_count != suffix_port_count:
                    raise ValueError(
                        f"Touchstone [Number of Ports] declares {port_count}, but "
                        f"{path.suffix} declares {suffix_port_count}."
                    )
            elif keyword == "number of frequencies":
                declared_frequency_count = int(payload)
            elif keyword == "matrix format" and payload:
                matrix_format = payload.lower()
            elif keyword == "reference" and payload:
                reference_values = np.array(
                    [float(token) for token in payload.split()],
                    dtype=np.float64,
                )
                reference = (
                    float(reference_values[0])
                    if len(reference_values) == 1
                    else reference_values
                )
            elif keyword == "two-port data order" and payload:
                two_port_data_order = payload.lower().replace("-", "_")
            elif keyword == "network data":
                saw_network_data_keyword = True
            elif keyword in ("end", "noise data"):
                break
            continue

        if saw_network_data_keyword or not lower.startswith("["):
            normalized_line = line.replace("D", "E").replace("d", "e")
            numeric_tokens.extend(float(token) for token in normalized_line.split())

    if matrix_format != "full":
        raise ValueError(
            f"Touchstone matrix format {matrix_format!r} is not supported; "
            "please export full matrices."
        )

    numbers_per_point = 1 + 2 * port_count * port_count
    if len(numeric_tokens) == 0 or len(numeric_tokens) % numbers_per_point != 0:
        raise ValueError(
            f"Malformed Touchstone data in {path}: expected a multiple of "
            f"{numbers_per_point} numeric values per frequency point, received "
            f"{len(numeric_tokens)} values."
        )

    raw_data = np.asarray(numeric_tokens, dtype=np.float64).reshape(
        -1,
        numbers_per_point,
    )
    if declared_frequency_count is not None and len(raw_data) != declared_frequency_count:
        raise ValueError(
            f"Touchstone file declares {declared_frequency_count} frequencies, "
            f"but contains {len(raw_data)}."
        )

    frequencies = raw_data[:, 0] * _TOUCHSTONE_FREQ_SCALES[freq_unit]
    complex_pairs = _touchstone_pairs_to_complex(
        raw_data[:, 1:].reshape(-1, port_count * port_count, 2),
        data_format,
    )
    s_parameters = np.zeros(
        (len(frequencies), port_count, port_count),
        dtype=np.complex128,
    )
    for pair_index, (output_port, input_port) in enumerate(
        _touchstone_pair_order(
            port_count,
            two_port_data_order=two_port_data_order,
        )
    ):
        s_parameters[:, output_port, input_port] = complex_pairs[:, pair_index]

    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    s_parameters = s_parameters[order]
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"Touchstone frequencies in {path} must be strictly increasing.")

    return TouchstoneNetwork(
        frequencies=frequencies,
        s_parameters=s_parameters,
        reference=reference,
        path=str(path),
    )


def evaluate_touchstone_response(
    query_frequencies: Union[float, np.ndarray, Iterable[float]],
    *,
    file_path: Union[str, Path, None] = None,
    network: Optional[TouchstoneNetwork] = None,
    input_port: int = 1,
    output_port: int = 2,
    interpolation: TouchstoneInterpolation = "polar",
    out_of_band: OutOfBandPolicy = "edge",
) -> np.ndarray:
    """Evaluate one selected Touchstone path on a GHz frequency grid."""
    if network is None:
        if file_path is None:
            raise ValueError(
                "evaluate_touchstone_response requires either file_path or network."
            )
        network = load_touchstone_network(file_path)
    elif not isinstance(network, TouchstoneNetwork):
        raise TypeError("network must be a TouchstoneNetwork instance.")

    query = np.asarray(query_frequencies, dtype=np.float64)
    response = _interpolate_complex_response(
        query,
        network.frequencies,
        network.get_response(output_port, input_port),
        interpolation=interpolation,
        out_of_band=out_of_band,
        stage_name=f"S{output_port}{input_port}",
    )
    return np.asarray(response, dtype=np.complex128).reshape(query.shape)


def design_inverse_fir_from_touchstone(
    *,
    lo_freq: float,
    sample_rate: float,
    num_taps: int,
    file_path: Union[str, Path, None] = None,
    network: Optional[TouchstoneNetwork] = None,
    input_port: int = 1,
    output_port: int = 2,
    interpolation: TouchstoneInterpolation = "polar",
    out_of_band: OutOfBandPolicy = "zero",
    threshold_db: float = -20.0,
    window: Optional[Union[str, Tuple[Any, ...]]] = "kaiser",
    kaiser_beta: float = 6.0,
) -> np.ndarray:
    """Design a centered complex baseband FIR inverse for one Touchstone path."""
    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    _validate_positive_sample_rate(sample_rate)
    if not np.isfinite(lo_freq):
        raise ValueError("lo_freq must be finite.")
    if not np.isfinite(threshold_db):
        raise ValueError("threshold_db must be finite.")
    if window == "kaiser" and not np.isfinite(kaiser_beta):
        raise ValueError("kaiser_beta must be finite.")

    if network is None:
        if file_path is None:
            raise ValueError(
                "design_inverse_fir_from_touchstone requires either file_path "
                "or network."
            )
        network = load_touchstone_network(file_path)
    elif not isinstance(network, TouchstoneNetwork):
        raise TypeError("network must be a TouchstoneNetwork instance.")

    n_fft = max(65536, int(num_taps) * 8)
    freqs_bb = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    response = evaluate_touchstone_response(
        freqs_bb + float(lo_freq),
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

    impulse_centered = np.fft.fftshift(np.fft.ifft(inverse_response))
    center_idx = len(impulse_centered) // 2
    start = center_idx - (num_taps // 2)
    kernel = np.asarray(
        impulse_centered[start : start + num_taps],
        dtype=np.complex128,
    )
    if window is None or window == "none":
        return kernel

    resolved_window = ("kaiser", kaiser_beta) if window == "kaiser" else window
    window_values = np.asarray(
        get_window(resolved_window, len(kernel)),
        dtype=np.float64,
    )
    return kernel * window_values


@dataclass
class TransmissionResult:
    """Structured output that captures intermediate chain traces."""

    input_trace: SignalTrace
    output_trace: SignalTrace
    stage_outputs: list[SignalTrace] = field(default_factory=list)


@dataclass
class BundleTransmissionResult:
    """Structured output that captures intermediate bundle stages."""

    input_bundle: SignalBundle
    output_bundle: SignalBundle
    stage_outputs: list[SignalBundle] = field(default_factory=list)


class TransmissionStage(Protocol):
    """Structural interface for single-trace transmission stages."""

    name: str
    domain: StageDomain
    allowed_planes: Tuple[str, ...]
    is_lti: bool

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Transform one trace."""
        ...


class BundleTransmissionStage(Protocol):
    """Structural interface for stages that transform an aligned bundle."""

    name: str
    domain: StageDomain
    allowed_planes: Tuple[str, ...]
    is_lti: bool
    bundle_stage: bool

    def apply(self, bundle: SignalBundle) -> SignalBundle:
        """Transform one aligned bundle."""
        ...


@dataclass
class BaseTransmissionStage:
    """Shared validation and output helpers for transmission stages."""

    name: str = "stage"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = _ALL_PLANES
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def _validate_trace(self, trace: SignalTrace) -> None:
        if self.domain != "any" and trace.domain != self.domain:
            raise ValueError(
                f"{self.name} expects {self.domain} traces, received {trace.domain}."
            )
        if trace.plane not in self.allowed_planes:
            raise ValueError(
                f"{self.name} does not accept traces on plane {trace.plane}. "
                f"Allowed planes: {self.allowed_planes}"
            )

    def _finalize_trace(
        self,
        trace: SignalTrace,
        values: np.ndarray,
        *,
        plane: Optional[SignalPlane] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> SignalTrace:
        next_values = _normalize_output_values(values, trace.domain, self.name)
        next_metadata = dict(trace.metadata)
        if metadata_updates:
            next_metadata.update(metadata_updates)

        return trace.clone(
            values=next_values,
            plane=plane or self.output_plane or trace.plane,
            metadata=next_metadata,
        )

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Transform one trace."""
        raise NotImplementedError

    def describe(self) -> str:
        """Return a compact stage description for diagnostics."""
        return self.name


@dataclass
class BaseBundleTransmissionStage:
    """Shared validation and output helpers for bundle-level stages."""

    name: str = "bundle_stage"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = _ALL_PLANES
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None
    bundle_stage: bool = field(default=True, init=False, repr=False)

    def _validate_bundle(self, bundle: SignalBundle) -> None:
        if not isinstance(bundle, SignalBundle):
            raise TypeError(f"{self.name} expects a SignalBundle input.")
        if self.domain != "any" and bundle.domain != self.domain:
            raise ValueError(
                f"{self.name} expects {self.domain} bundles, received {bundle.domain}."
            )
        if bundle.plane not in self.allowed_planes:
            raise ValueError(
                f"{self.name} does not accept bundles on plane {bundle.plane}. "
                f"Allowed planes: {self.allowed_planes}"
            )

    def _finalize_bundle(
        self,
        bundle: SignalBundle,
        traces: Dict[str, SignalTrace],
        *,
        label: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> SignalBundle:
        next_traces = {
            name: trace.clone(plane=self.output_plane or trace.plane)
            for name, trace in traces.items()
        }
        next_metadata = dict(bundle.metadata)
        if metadata_updates:
            next_metadata.update(metadata_updates)
        return bundle.clone(
            traces=next_traces,
            order=tuple(next_traces.keys()),
            label=label or bundle.label,
            metadata=next_metadata,
        )

    def apply(self, bundle: SignalBundle) -> SignalBundle:
        """Transform one aligned bundle."""
        raise NotImplementedError

    def describe(self) -> str:
        """Return a compact stage description for diagnostics."""
        return self.name


@dataclass
class AttenuatorStage(BaseTransmissionStage):
    """Constant amplitude attenuation or gain."""

    loss_db: float = 0.0
    name: str = "attenuator"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "awg_rf", "qubit_iq", "qubit_rf")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Scale the waveform by the stage's voltage attenuation ratio."""
        self._validate_trace(trace)
        amplitude_ratio = 10 ** (-self.loss_db / 20.0)
        values = trace.values * amplitude_ratio
        return self._finalize_trace(
            trace,
            values,
            metadata_updates={"last_stage": self.name, "loss_db": self.loss_db},
        )

    def describe(self) -> str:
        """Return a compact attenuation summary."""
        return f"{self.name}[{self.loss_db:.3f} dB]"


@dataclass
class FIRFilterStage(BaseTransmissionStage):
    """Discrete FIR stage with leading or group-delay-compensated alignment."""

    kernel: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alignment: FIRAlignment = "leading"
    name: str = "fir_filter"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = _ALL_PLANES
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None
    sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        self.kernel = _as_array(self.kernel)
        if self.kernel.ndim != 1:
            raise ValueError("FIRFilterStage.kernel must be a 1D array.")
        if self.alignment not in ("leading", "centered"):
            raise ValueError(f"Unsupported FIRFilterStage alignment: {self.alignment}")
        if self.sample_rate is not None:
            _validate_positive_sample_rate(self.sample_rate)

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
        """Build an FIR stage from a windowed-sinc design."""
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
        return cls(kernel=kernel, name=name, sample_rate=sample_rate, **kwargs)

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
        """Build a low-pass FIR stage."""
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
        """Build a high-pass FIR stage."""
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
        """Build a band-pass FIR stage."""
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
        """Convolve the trace and preserve its original sample count."""
        self._validate_trace(trace)
        _validate_filter_sample_rate(self.sample_rate, trace.sample_rate, self.name)

        if len(trace.values) == 0 or len(self.kernel) == 0:
            filtered = trace.values
        else:
            filtered_full = convolve(
                trace.values,
                self.kernel,
                mode="full",
                method="auto",
            )
            if self.alignment == "centered":
                delay = (len(self.kernel) - 1) // 2
                filtered = filtered_full[delay : delay + len(trace.values)]
            else:
                filtered = filtered_full[: len(trace.values)]

        return self._finalize_trace(
            trace,
            filtered,
            metadata_updates={
                "last_stage": self.name,
                "kernel_length": len(self.kernel),
                "alignment": self.alignment,
                "design_sample_rate": self.sample_rate,
            },
        )

    def describe(self) -> str:
        """Return a compact FIR summary."""
        return f"{self.name}[taps={len(self.kernel)}, {self.alignment}]"


@dataclass
class DerivativePrecorrectionStage(BaseTransmissionStage):
    """Apply a DRAG-like derivative polynomial directly in the time domain."""

    coefficients: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.complex128)
    )
    derivative_orders: tuple[int, ...] = ()
    normalize_to_peak: bool = True
    phase_rad: float = 0.0
    edge_order: int = 2
    normalization_epsilon: float = 1e-15
    name: str = "derivative_precorrection"
    domain: StageDomain = "iq_complex"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "qubit_iq")
    is_lti: bool = False
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(
            self.coefficients,
            dtype=np.complex128,
        ).reshape(-1)
        if np.any(~np.isfinite(self.coefficients)):
            raise ValueError("Derivative coefficients must be finite.")
        raw_orders = tuple(self.derivative_orders)
        self.derivative_orders = _resolve_derivative_orders(
            raw_orders if raw_orders else None,
            len(self.coefficients),
        )
        _validate_derivative_options(
            1.0,
            self.edge_order,
            self.normalization_epsilon,
        )
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite.")

    @classmethod
    def from_design(
        cls,
        design: DerivativePrecorrectionDesign,
        *,
        name: str = "derivative_precorrection",
        **kwargs: Any,
    ) -> "DerivativePrecorrectionStage":
        """Build a stage from ``design_derivative_precorrection`` output."""
        if not isinstance(design, DerivativePrecorrectionDesign):
            raise TypeError("design must be a DerivativePrecorrectionDesign.")
        return cls(
            coefficients=design.coefficients.copy(),
            derivative_orders=design.derivative_orders,
            normalize_to_peak=design.normalize_to_peak,
            phase_rad=design.phase_rad,
            edge_order=design.edge_order,
            name=name,
            **kwargs,
        )

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Apply derivative precorrection and attach its resolved metadata."""
        self._validate_trace(trace)
        corrected, metadata = apply_derivative_precorrection(
            trace.values,
            1.0 / trace.sample_rate,
            self.coefficients,
            derivative_orders=self.derivative_orders,
            normalize_to_peak=self.normalize_to_peak,
            phase_rad=self.phase_rad,
            edge_order=self.edge_order,
            normalization_epsilon=self.normalization_epsilon,
        )
        metadata["last_stage"] = self.name
        return self._finalize_trace(
            trace,
            corrected,
            metadata_updates=metadata,
        )

    def describe(self) -> str:
        """Return derivative orders, normalization mode, and phase."""
        orders = ",".join(str(order) for order in self.derivative_orders) or "-"
        return (
            f"{self.name}[orders={orders}, normalize={self.normalize_to_peak}, "
            f"phase={self.phase_rad:.4f} rad]"
        )


@dataclass
class IIRFilterStage(BaseTransmissionStage):
    """Direct-form IIR stage implemented with ``scipy.signal.lfilter``."""

    b: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    a: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    name: str = "iir_filter"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = _ALL_PLANES
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None
    sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        self.b = _as_array(self.b)
        self.a = _as_array(self.a)
        if self.b.ndim != 1 or len(self.b) == 0:
            raise ValueError("IIRFilterStage.b must be a non-empty 1D array.")
        if self.a.ndim != 1 or len(self.a) == 0:
            raise ValueError("IIRFilterStage.a must be a non-empty 1D array.")
        if self.sample_rate is not None:
            _validate_positive_sample_rate(self.sample_rate)

    @classmethod
    def butterworth(
        cls,
        *,
        order: int,
        cutoff_freq: Union[float, Tuple[float, float]],
        sample_rate: float,
        filter_kind: str = "lowpass",
        name: str = "iir_butterworth",
        **kwargs: Any,
    ) -> "IIRFilterStage":
        """Build a Butterworth IIR stage in transfer-function form."""
        kind = _normalize_filter_kind(filter_kind)
        normalized_cutoff = _normalize_cutoff(cutoff_freq, kind)
        _validate_sample_rate_and_cutoff(sample_rate, normalized_cutoff)
        b, a = butter(
            order,
            normalized_cutoff,
            btype=kind,
            fs=sample_rate,
            output="ba",
        )
        return cls(b=b, a=a, name=name, sample_rate=sample_rate, **kwargs)

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Filter the waveform with ``scipy.signal.lfilter``."""
        self._validate_trace(trace)
        _validate_filter_sample_rate(self.sample_rate, trace.sample_rate, self.name)
        filtered = (
            trace.values
            if len(trace.values) == 0
            else lfilter(self.b, self.a, trace.values)
        )
        return self._finalize_trace(
            trace,
            filtered,
            metadata_updates={
                "last_stage": self.name,
                "numerator_length": len(self.b),
                "denominator_length": len(self.a),
                "design_sample_rate": self.sample_rate,
            },
        )

    def describe(self) -> str:
        """Return a compact IIR coefficient summary."""
        return f"{self.name}[len(b)={len(self.b)}, len(a)={len(self.a)}]"


@dataclass
class SOSFilterStage(BaseTransmissionStage):
    """Second-order-sections stage implemented with ``scipy.signal.sosfilt``."""

    sos: np.ndarray = field(
        default_factory=lambda: np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    )
    name: str = "sos_filter"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = _ALL_PLANES
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None
    sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        self.sos = np.asarray(self.sos, dtype=np.float64)
        if self.sos.ndim != 2 or self.sos.shape[1:] != (6,) or len(self.sos) == 0:
            raise ValueError(
                "SOSFilterStage.sos must have non-empty shape (n_sections, 6)."
            )
        if self.sample_rate is not None:
            _validate_positive_sample_rate(self.sample_rate)

    @classmethod
    def butterworth(
        cls,
        *,
        order: int,
        cutoff_freq: Union[float, Tuple[float, float]],
        sample_rate: float,
        filter_kind: str = "lowpass",
        name: str = "sos_butterworth",
        **kwargs: Any,
    ) -> "SOSFilterStage":
        """Build a Butterworth filter in SOS form."""
        kind = _normalize_filter_kind(filter_kind)
        normalized_cutoff = _normalize_cutoff(cutoff_freq, kind)
        _validate_sample_rate_and_cutoff(sample_rate, normalized_cutoff)
        sos = butter(
            order,
            normalized_cutoff,
            btype=kind,
            fs=sample_rate,
            output="sos",
        )
        return cls(sos=sos, name=name, sample_rate=sample_rate, **kwargs)

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Filter the waveform with ``scipy.signal.sosfilt``."""
        self._validate_trace(trace)
        _validate_filter_sample_rate(self.sample_rate, trace.sample_rate, self.name)
        filtered = (
            trace.values
            if len(trace.values) == 0
            else sosfilt(self.sos, trace.values)
        )
        return self._finalize_trace(
            trace,
            filtered,
            metadata_updates={
                "last_stage": self.name,
                "num_sections": int(self.sos.shape[0]),
                "design_sample_rate": self.sample_rate,
            },
        )

    def describe(self) -> str:
        """Return a compact SOS section summary."""
        return f"{self.name}[sections={int(self.sos.shape[0])}]"


@dataclass
class TransferFunctionStage(BaseTransmissionStage):
    """General linear stage defined by a complex transfer function ``H(f)``."""

    H: Union[Callable[[np.ndarray], Union[np.ndarray, complex]], np.ndarray] = field(
        default_factory=lambda: (lambda freq: np.ones_like(freq, dtype=np.complex128))
    )
    name: str = "transfer_function"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "awg_rf", "qubit_iq", "qubit_rf")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None
    min_impulse_length: Optional[int] = None

    @classmethod
    def from_impulse_response(
        cls,
        impulse_response: Union[np.ndarray, Iterable[complex]],
        *,
        name: str = "impulse_response",
        **kwargs: Any,
    ) -> "TransferFunctionStage":
        """Build a transfer-function stage from an impulse response."""
        impulse = _as_array(impulse_response)
        if impulse.ndim != 1 or len(impulse) == 0:
            raise ValueError("impulse_response must be a non-empty 1D array.")

        def response(freq_axis: np.ndarray) -> np.ndarray:
            return np.fft.fft(impulse, n=len(freq_axis))

        return cls(H=response, name=name, min_impulse_length=len(impulse), **kwargs)

    @classmethod
    def first_order_lowpass(
        cls,
        *,
        cutoff_freq: float,
        name: str = "first_order_lowpass",
        **kwargs: Any,
    ) -> "TransferFunctionStage":
        """Build a first-order low-pass stage with a pole frequency in GHz."""
        if not np.isfinite(cutoff_freq) or cutoff_freq <= 0:
            raise ValueError("cutoff_freq must be positive and finite.")
        return cls(
            H=lambda freq: 1.0 / (1.0 + 1j * freq / cutoff_freq),
            name=name,
            **kwargs,
        )

    @classmethod
    def first_order_highpass(
        cls,
        *,
        cutoff_freq: float,
        name: str = "first_order_highpass",
        **kwargs: Any,
    ) -> "TransferFunctionStage":
        """Build a first-order high-pass stage with a corner frequency in GHz."""
        if not np.isfinite(cutoff_freq) or cutoff_freq <= 0:
            raise ValueError("cutoff_freq must be positive and finite.")
        return cls(
            H=lambda freq: (1j * freq / cutoff_freq)
            / (1.0 + 1j * freq / cutoff_freq),
            name=name,
            **kwargs,
        )

    def _evaluate_response(self, freq_axis: np.ndarray) -> np.ndarray:
        response = self.H(freq_axis) if callable(self.H) else self.H
        response = np.asarray(response)
        if response.ndim == 0:
            response = np.full_like(freq_axis, response, dtype=np.complex128)
        if response.ndim != 1:
            raise ValueError(
                f"{self.name} expected H(f) to return a scalar or 1D response."
            )
        if len(response) != len(freq_axis):
            raise ValueError(
                f"{self.name} expected H(f) to return length {len(freq_axis)}, "
                f"received {len(response)}."
            )
        return np.asarray(response, dtype=np.complex128)

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Filter a trace in the frequency domain with ``H(f)``."""
        self._validate_trace(trace)
        if len(trace.values) == 0:
            return self._finalize_trace(
                trace,
                trace.values,
                metadata_updates={"last_stage": self.name, "fft_length": 0},
            )

        fft_length = _next_fft_length(
            len(trace.values),
            impulse_length=self.min_impulse_length or len(trace.values),
        )
        freq_axis = np.fft.fftfreq(fft_length, d=1.0 / trace.sample_rate)
        padded_values = np.pad(trace.values, (0, fft_length - len(trace.values)))
        response = self._evaluate_response(freq_axis)
        if trace.domain == "rf_real":
            response = _enforce_real_self_conjugate_bins(response, freq_axis)

        filtered = np.fft.ifft(
            np.fft.fft(padded_values, n=fft_length) * response,
            n=fft_length,
        )
        filtered = filtered[: len(trace.values)]

        return self._finalize_trace(
            trace,
            filtered,
            metadata_updates={
                "last_stage": self.name,
                "fft_length": fft_length,
            },
        )

    def describe(self) -> str:
        """Return a compact summary for a frequency-domain stage."""
        return f"{self.name}[fft]"


@dataclass
class TouchstoneStage(BaseTransmissionStage):
    """Linear single-trace stage backed by one Touchstone S-parameter path."""

    file_path: Union[str, Path] = ""
    input_port: int = 1
    output_port: int = 2
    interpolation: TouchstoneInterpolation = "polar"
    out_of_band: OutOfBandPolicy = "edge"
    frequency_mode: Literal["absolute", "relative"] = "absolute"
    network: Optional[TouchstoneNetwork] = None
    name: str = "touchstone"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "awg_rf", "qubit_iq", "qubit_rf")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        if self.interpolation not in ("cartesian", "polar"):
            raise ValueError(
                f"Unsupported Touchstone interpolation mode: {self.interpolation}"
            )
        if self.out_of_band not in ("edge", "zero", "error"):
            raise ValueError(
                f"Unsupported Touchstone out_of_band policy: {self.out_of_band}"
            )
        if self.frequency_mode not in ("absolute", "relative"):
            raise ValueError(
                f"Unsupported Touchstone frequency_mode: {self.frequency_mode}"
            )

        if self.network is None:
            if not self.file_path:
                raise ValueError(
                    "TouchstoneStage requires either file_path or a preloaded network."
                )
            self.network = load_touchstone_network(self.file_path)
        elif not isinstance(self.network, TouchstoneNetwork):
            raise TypeError(
                "TouchstoneStage.network must be a TouchstoneNetwork instance."
            )

        self.file_path = str(self.file_path or self.network.path)
        self.network.get_response(self.output_port, self.input_port)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        input_port: int = 1,
        output_port: int = 2,
        **kwargs: Any,
    ) -> "TouchstoneStage":
        """Build a stage directly from a Touchstone file."""
        return cls(
            file_path=file_path,
            input_port=input_port,
            output_port=output_port,
            **kwargs,
        )

    def _selected_response(self) -> np.ndarray:
        return self.network.get_response(self.output_port, self.input_port)

    def _interpolate_selected_response(self, query_freq: np.ndarray) -> np.ndarray:
        return _interpolate_complex_response(
            query_freq,
            self.network.frequencies,
            self._selected_response(),
            interpolation=self.interpolation,
            out_of_band=self.out_of_band,
            stage_name=self.name,
        )

    def _evaluate_response(
        self,
        trace: SignalTrace,
        freq_axis: np.ndarray,
    ) -> np.ndarray:
        if trace.domain == "iq_complex":
            base_freq = trace.lo_freq if self.frequency_mode == "absolute" else 0.0
            return self._interpolate_selected_response(base_freq + freq_axis)

        positive_response = self._interpolate_selected_response(np.abs(freq_axis))
        response = np.asarray(positive_response, dtype=np.complex128)
        negative_mask = freq_axis < 0
        response[negative_mask] = np.conj(response[negative_mask])
        return _enforce_real_self_conjugate_bins(response, freq_axis)

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Propagate one trace through the selected S-parameter response."""
        self._validate_trace(trace)
        if len(trace.values) == 0:
            return self._finalize_trace(
                trace,
                trace.values,
                metadata_updates={
                    "last_stage": self.name,
                    "touchstone_file": self.file_path,
                    "input_port": self.input_port,
                    "output_port": self.output_port,
                    "fft_length": 0,
                },
            )

        fft_length = _next_fft_length(len(trace.values))
        freq_axis = np.fft.fftfreq(fft_length, d=1.0 / trace.sample_rate)
        padded_values = np.pad(trace.values, (0, fft_length - len(trace.values)))
        response = self._evaluate_response(trace, freq_axis)
        filtered = np.fft.ifft(
            np.fft.fft(padded_values, n=fft_length) * response,
            n=fft_length,
        )[: len(trace.values)]

        return self._finalize_trace(
            trace,
            filtered,
            metadata_updates={
                "last_stage": self.name,
                "touchstone_file": self.file_path,
                "input_port": self.input_port,
                "output_port": self.output_port,
                "fft_length": fft_length,
            },
        )

    def describe(self) -> str:
        """Return a compact selected-path summary."""
        file_name = Path(self.file_path).name if self.file_path else "network"
        return f"{self.name}[S{self.output_port}{self.input_port}, {file_name}]"


@dataclass
class MIMOTouchstoneStage(BaseBundleTransmissionStage):
    """Bundle-level Touchstone stage for MIMO propagation and crosstalk."""

    file_path: Union[str, Path] = ""
    input_ports: Tuple[int, ...] = (1,)
    output_ports: Tuple[int, ...] = (1,)
    input_channels: Optional[Tuple[str, ...]] = None
    output_channels: Optional[Tuple[str, ...]] = None
    interpolation: TouchstoneInterpolation = "polar"
    out_of_band: OutOfBandPolicy = "edge"
    frequency_mode: Literal["absolute", "relative"] = "absolute"
    network: Optional[TouchstoneNetwork] = None
    name: str = "mimo_touchstone"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "awg_rf", "qubit_iq", "qubit_rf")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def __post_init__(self) -> None:
        for field_name, ports in (
            ("input_ports", self.input_ports),
            ("output_ports", self.output_ports),
        ):
            if not ports:
                raise ValueError(f"MIMOTouchstoneStage.{field_name} must not be empty.")
            if any(
                isinstance(port, bool) or not isinstance(port, (int, np.integer))
                for port in ports
            ):
                raise TypeError(
                    f"MIMOTouchstoneStage.{field_name} must contain integers."
                )
            normalized = tuple(int(port) for port in ports)
            if len(set(normalized)) != len(normalized):
                raise ValueError(
                    f"MIMOTouchstoneStage.{field_name} must not contain duplicates."
                )
            setattr(self, field_name, normalized)

        self.input_channels = self._normalize_channel_selection(
            self.input_channels,
            len(self.input_ports),
            "input_channels",
        )
        self.output_channels = self._normalize_channel_selection(
            self.output_channels,
            len(self.output_ports),
            "output_channels",
        )
        if self.interpolation not in ("cartesian", "polar"):
            raise ValueError(
                f"Unsupported Touchstone interpolation mode: {self.interpolation}"
            )
        if self.out_of_band not in ("edge", "zero", "error"):
            raise ValueError(
                f"Unsupported Touchstone out_of_band policy: {self.out_of_band}"
            )
        if self.frequency_mode not in ("absolute", "relative"):
            raise ValueError(
                f"Unsupported Touchstone frequency_mode: {self.frequency_mode}"
            )

        if self.network is None:
            if not self.file_path:
                raise ValueError(
                    "MIMOTouchstoneStage requires either file_path or a preloaded network."
                )
            self.network = load_touchstone_network(self.file_path)
        elif not isinstance(self.network, TouchstoneNetwork):
            raise TypeError(
                "MIMOTouchstoneStage.network must be a TouchstoneNetwork instance."
            )

        self.file_path = str(self.file_path or self.network.path)
        for output_port in self.output_ports:
            for input_port in self.input_ports:
                self.network.get_response(output_port, input_port)

    @staticmethod
    def _normalize_channel_selection(
        channels: Optional[Tuple[str, ...]],
        expected_length: int,
        field_name: str,
    ) -> Optional[Tuple[str, ...]]:
        if channels is None:
            return None
        normalized = tuple(channels)
        if len(normalized) != expected_length:
            raise ValueError(
                f"MIMOTouchstoneStage.{field_name} expected {expected_length} "
                f"name(s), received {len(normalized)}."
            )
        if any(not isinstance(name, str) or not name for name in normalized):
            raise ValueError(
                f"MIMOTouchstoneStage.{field_name} must contain non-empty strings."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"MIMOTouchstoneStage.{field_name} must not contain duplicates."
            )
        return normalized

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        input_ports: Tuple[int, ...],
        output_ports: Tuple[int, ...],
        **kwargs: Any,
    ) -> "MIMOTouchstoneStage":
        """Build a MIMO stage directly from a Touchstone file."""
        return cls(
            file_path=file_path,
            input_ports=input_ports,
            output_ports=output_ports,
            **kwargs,
        )

    def _resolve_input_channels(self, bundle: SignalBundle) -> Tuple[str, ...]:
        names = self.input_channels or bundle.order
        if len(names) != len(self.input_ports):
            raise ValueError(
                f"{self.name} expected {len(self.input_ports)} input channel(s), "
                f"received {len(names)}."
            )
        missing = [name for name in names if name not in bundle.traces]
        if missing:
            raise ValueError(f"{self.name} could not find input channel(s): {missing}")
        return tuple(names)

    def _resolve_output_channels(
        self,
        input_channels: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        if self.output_channels is not None:
            return self.output_channels
        if len(input_channels) == len(self.output_ports):
            return tuple(input_channels)
        return tuple(f"port_{port}" for port in self.output_ports)

    def _interpolate_pair_response(
        self,
        output_port: int,
        input_port: int,
        query_freq: np.ndarray,
    ) -> np.ndarray:
        return _interpolate_complex_response(
            query_freq,
            self.network.frequencies,
            self.network.get_response(output_port, input_port),
            interpolation=self.interpolation,
            out_of_band=self.out_of_band,
            stage_name=self.name,
        )

    def _evaluate_response_matrix(
        self,
        bundle: SignalBundle,
        freq_axis: np.ndarray,
    ) -> np.ndarray:
        response = np.zeros(
            (len(self.output_ports), len(self.input_ports), len(freq_axis)),
            dtype=np.complex128,
        )

        if bundle.domain == "iq_complex":
            if self.frequency_mode == "absolute":
                if bundle.shared_lo_freq is None:
                    raise ValueError(
                        f"{self.name} requires all iq_complex traces to share one "
                        "lo_freq in absolute frequency mode."
                    )
                base_freq = bundle.shared_lo_freq
            else:
                base_freq = 0.0
            query_freq = base_freq + freq_axis
            for out_idx, output_port in enumerate(self.output_ports):
                for in_idx, input_port in enumerate(self.input_ports):
                    response[out_idx, in_idx] = self._interpolate_pair_response(
                        output_port,
                        input_port,
                        query_freq,
                    )
            return response

        query_freq = np.abs(freq_axis)
        for out_idx, output_port in enumerate(self.output_ports):
            for in_idx, input_port in enumerate(self.input_ports):
                pair_response = self._interpolate_pair_response(
                    output_port,
                    input_port,
                    query_freq,
                )
                pair_response = np.asarray(pair_response, dtype=np.complex128)
                negative_mask = freq_axis < 0
                pair_response[negative_mask] = np.conj(pair_response[negative_mask])
                response[out_idx, in_idx] = _enforce_real_self_conjugate_bins(
                    pair_response,
                    freq_axis,
                )
        return response

    def _make_output_trace(
        self,
        bundle: SignalBundle,
        input_channels: Tuple[str, ...],
        output_channels: Tuple[str, ...],
        output_index: int,
        values: np.ndarray,
        fft_length: int,
    ) -> SignalTrace:
        channel_name = output_channels[output_index]
        reference_trace = bundle[
            input_channels[min(output_index, len(input_channels) - 1)]
        ]
        return reference_trace.clone(
            values=_normalize_output_values(values, bundle.domain, self.name),
            plane=self.output_plane or reference_trace.plane,
            label=f"{channel_name}_{self.name}",
            metadata={
                **reference_trace.metadata,
                "last_stage": self.name,
                "touchstone_file": self.file_path,
                "input_ports": self.input_ports,
                "output_ports": self.output_ports,
                "input_channels": input_channels,
                "output_channels": output_channels,
                "fft_length": fft_length,
            },
        )

    def apply(self, bundle: SignalBundle) -> SignalBundle:
        """Propagate an aligned bundle through the selected MIMO block."""
        self._validate_bundle(bundle)
        input_channels = self._resolve_input_channels(bundle)
        output_channels = self._resolve_output_channels(input_channels)
        num_samples = len(bundle.t_axis)

        if num_samples == 0:
            next_traces = {
                channel_name: self._make_output_trace(
                    bundle,
                    input_channels,
                    output_channels,
                    output_index,
                    np.array([], dtype=(
                        np.complex128 if bundle.domain == "iq_complex" else np.float64
                    )),
                    0,
                )
                for output_index, channel_name in enumerate(output_channels)
            }
        else:
            fft_length = _next_fft_length(num_samples)
            freq_axis = np.fft.fftfreq(fft_length, d=1.0 / bundle.sample_rate)
            response_matrix = self._evaluate_response_matrix(bundle, freq_axis)
            input_spectra = np.zeros(
                (len(input_channels), fft_length),
                dtype=np.complex128,
            )
            for index, channel_name in enumerate(input_channels):
                padded_values = np.pad(
                    bundle[channel_name].values,
                    (0, fft_length - num_samples),
                )
                input_spectra[index] = np.fft.fft(padded_values, n=fft_length)

            next_traces = {}
            for output_index, channel_name in enumerate(output_channels):
                output_spectrum = np.sum(
                    response_matrix[output_index] * input_spectra,
                    axis=0,
                )
                output_values = np.fft.ifft(
                    output_spectrum,
                    n=fft_length,
                )[:num_samples]
                next_traces[channel_name] = self._make_output_trace(
                    bundle,
                    input_channels,
                    output_channels,
                    output_index,
                    output_values,
                    fft_length,
                )

        return self._finalize_bundle(
            bundle,
            next_traces,
            label=f"{bundle.label}_{self.name}",
            metadata_updates={"last_stage": self.name},
        )

    def describe(self) -> str:
        """Return a compact MIMO port mapping summary."""
        file_name = Path(self.file_path).name if self.file_path else "network"
        mapping = f"out={self.output_ports}<-in={self.input_ports}"
        return f"{self.name}[{mapping}, {file_name}]"


@dataclass
class DelayStage(BaseTransmissionStage):
    """Time-domain delay with zero fill outside the original support."""

    delay_ns: float = 0.0
    name: str = "delay"
    domain: StageDomain = "any"
    allowed_planes: Tuple[str, ...] = ("awg_iq", "awg_rf", "qubit_iq", "qubit_rf")
    is_lti: bool = True
    output_plane: Optional[SignalPlane] = None

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Delay a waveform by interpolation on the existing sample grid."""
        self._validate_trace(trace)
        if self.delay_ns == 0.0 or len(trace.values) == 0:
            return self._finalize_trace(
                trace,
                trace.values,
                metadata_updates={
                    "last_stage": self.name,
                    "delay_ns": self.delay_ns,
                },
            )

        shifted_t = trace.t_axis - self.delay_ns
        if trace.domain == "rf_real":
            delayed = np.interp(
                shifted_t,
                trace.t_axis,
                trace.values,
                left=0.0,
                right=0.0,
            )
        else:
            delayed = np.interp(
                shifted_t,
                trace.t_axis,
                np.real(trace.values),
                left=0.0,
                right=0.0,
            ) + 1j * np.interp(
                shifted_t,
                trace.t_axis,
                np.imag(trace.values),
                left=0.0,
                right=0.0,
            )

        return self._finalize_trace(
            trace,
            delayed,
            metadata_updates={"last_stage": self.name, "delay_ns": self.delay_ns},
        )

    def describe(self) -> str:
        """Return a compact delay summary."""
        return f"{self.name}[{self.delay_ns:.3f} ns]"


@dataclass
class TransmissionChain:
    """Ordered collection of single-trace transmission stages."""

    name: str = "default_chain"
    stages: list[TransmissionStage] = field(default_factory=list)

    def append(self, stage: TransmissionStage) -> None:
        """Append one stage to the end of the chain."""
        self.stages.append(stage)

    def extend(self, stages: Iterable[TransmissionStage]) -> None:
        """Append several stages to the end of the chain."""
        self.stages.extend(stages)

    def describe(self) -> str:
        """Return a concise left-to-right description of the chain."""
        if not self.stages:
            return f"{self.name} (0 stage(s))"

        stage_descriptions = [_describe_stage(stage) for stage in self.stages]
        return (
            f"{self.name} ({len(self.stages)} stage(s)): "
            + " -> ".join(stage_descriptions)
        )

    @property
    def is_lti(self) -> bool:
        """Return ``True`` only when every stage in the chain is LTI."""
        return all(getattr(stage, "is_lti", False) for stage in self.stages)

    def impulse_response(
        self,
        *,
        num_samples: int,
        sample_rate: float,
        domain: SignalDomain = "rf_real",
        plane: SignalPlane = "awg_rf",
        lo_freq: float = 0.0,
        capture_history: bool = False,
    ) -> Union[SignalTrace, TransmissionResult]:
        """Estimate the chain response to a leading discrete impulse."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        dtype = np.complex128 if domain == "iq_complex" else np.float64
        values = np.zeros(num_samples, dtype=dtype)
        values[0] = 1.0
        t_axis = np.arange(num_samples, dtype=np.float64) / sample_rate
        trace = SignalTrace(
            t_axis=t_axis,
            values=values,
            sample_rate=sample_rate,
            domain=domain,
            plane=plane,
            lo_freq=lo_freq,
            label=f"{self.name}_impulse",
        )
        return self.apply(trace, capture_history=capture_history)

    def frequency_response(
        self,
        *,
        num_samples: int,
        sample_rate: float,
        domain: SignalDomain = "rf_real",
        plane: SignalPlane = "awg_rf",
        lo_freq: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the frequency response of an LTI chain."""
        if not self.is_lti:
            raise ValueError(
                "frequency_response is only defined for LTI transmission chains."
            )

        impulse_trace = self.impulse_response(
            num_samples=num_samples,
            sample_rate=sample_rate,
            domain=domain,
            plane=plane,
            lo_freq=lo_freq,
        )
        freq_axis = np.fft.fftfreq(num_samples, d=1.0 / sample_rate)
        response = np.fft.fft(impulse_trace.values, n=num_samples)
        return freq_axis, np.asarray(response, dtype=np.complex128)

    def apply(
        self,
        trace: SignalTrace,
        capture_history: bool = False,
    ) -> Union[SignalTrace, TransmissionResult]:
        """Apply every stage in sequence to one trace."""
        current = trace
        history: list[SignalTrace] = []

        for stage in self.stages:
            current = stage.apply(current)
            if capture_history:
                history.append(current)

        if capture_history:
            return TransmissionResult(
                input_trace=trace,
                output_trace=current,
                stage_outputs=history,
            )
        return current


@dataclass
class BundleTransmissionChain:
    """Ordered chain mixing single-trace and bundle-aware stages."""

    name: str = "default_bundle_chain"
    stages: list[Any] = field(default_factory=list)

    def append(self, stage: Any) -> None:
        """Append one stage."""
        self.stages.append(stage)

    def extend(self, stages: Iterable[Any]) -> None:
        """Append several stages in order."""
        self.stages.extend(stages)

    def describe(self) -> str:
        """Return a concise left-to-right description."""
        if not self.stages:
            return f"{self.name} (0 stage(s))"
        return (
            f"{self.name} ({len(self.stages)} stage(s)): "
            + " -> ".join(_describe_stage(stage) for stage in self.stages)
        )

    @property
    def is_lti(self) -> bool:
        """Return ``True`` only when every stage is LTI."""
        return all(getattr(stage, "is_lti", False) for stage in self.stages)

    def _apply_stage(self, stage: Any, bundle: SignalBundle) -> SignalBundle:
        apply = getattr(stage, "apply", None)
        if not callable(apply):
            raise TypeError(f"{_describe_stage(stage)} does not define apply().")

        if getattr(stage, "bundle_stage", False):
            output = apply(bundle)
            if not isinstance(output, SignalBundle):
                raise TypeError(
                    f"{_describe_stage(stage)} must return a SignalBundle when used "
                    "in BundleTransmissionChain."
                )
            return output

        output_traces: Dict[str, SignalTrace] = {}
        for name in bundle.order:
            trace_output = apply(bundle[name])
            if not isinstance(trace_output, SignalTrace):
                raise TypeError(
                    f"{_describe_stage(stage)} must return a SignalTrace when mapped "
                    "over a SignalBundle."
                )
            output_traces[name] = trace_output
        return bundle.clone(traces=output_traces, order=bundle.order)

    def apply(
        self,
        bundle: SignalBundle,
        capture_history: bool = False,
    ) -> Union[SignalBundle, BundleTransmissionResult]:
        """Apply every stage in sequence to one aligned bundle."""
        if not isinstance(bundle, SignalBundle):
            raise TypeError("BundleTransmissionChain expects a SignalBundle input.")

        current = bundle
        history: list[SignalBundle] = []
        for stage in self.stages:
            current = self._apply_stage(stage, current)
            if capture_history:
                history.append(current)

        if capture_history:
            return BundleTransmissionResult(
                input_bundle=bundle,
                output_bundle=current,
                stage_outputs=history,
            )
        return current


__all__ = [
    "AttenuatorStage",
    "BaseBundleTransmissionStage",
    "BaseTransmissionStage",
    "BundleTransmissionChain",
    "BundleTransmissionResult",
    "BundleTransmissionStage",
    "DelayStage",
    "DerivativePrecorrectionDesign",
    "DerivativePrecorrectionStage",
    "FIRFilterStage",
    "IIRFilterStage",
    "MIMOTouchstoneStage",
    "SignalBundle",
    "SignalTrace",
    "SOSFilterStage",
    "TouchstoneNetwork",
    "TouchstoneStage",
    "TransferFunctionStage",
    "TransmissionChain",
    "TransmissionResult",
    "TransmissionStage",
    "apply_derivative_precorrection",
    "compute_derivative_basis",
    "design_derivative_precorrection",
    "design_inverse_fir_from_touchstone",
    "evaluate_touchstone_response",
    "load_touchstone_network",
]
