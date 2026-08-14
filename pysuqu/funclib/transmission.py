"""Transmission-chain primitives for control waveform propagation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Protocol, Tuple, Union

import numpy as np
from scipy.signal import butter, convolve, firwin, lfilter, sosfilt


SignalDomain = Literal["iq_complex", "rf_real"]
SignalPlane = Literal["baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf"]
StageDomain = Literal["iq_complex", "rf_real", "any"]
FilterKind = Literal["lowpass", "highpass", "bandpass", "bandstop"]
FIRAlignment = Literal["leading", "centered"]

_VALID_DOMAINS = {"iq_complex", "rf_real"}
_VALID_PLANES = {"baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf"}
_ALL_PLANES = ("baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf")


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


@dataclass
class TransmissionResult:
    """Structured output that captures intermediate chain traces."""

    input_trace: SignalTrace
    output_trace: SignalTrace
    stage_outputs: list[SignalTrace] = field(default_factory=list)


class TransmissionStage(Protocol):
    """Structural interface for single-trace transmission stages."""

    name: str
    domain: StageDomain
    allowed_planes: Tuple[str, ...]
    is_lti: bool

    def apply(self, trace: SignalTrace) -> SignalTrace:
        """Transform one trace."""
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


__all__ = [
    "AttenuatorStage",
    "BaseTransmissionStage",
    "DelayStage",
    "FIRFilterStage",
    "IIRFilterStage",
    "SignalTrace",
    "SOSFilterStage",
    "TransferFunctionStage",
    "TransmissionChain",
    "TransmissionResult",
    "TransmissionStage",
]
