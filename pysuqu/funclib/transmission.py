"""Transmission-chain primitives for control waveform propagation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Literal, Optional, Protocol, Tuple, Union

import numpy as np


SignalDomain = Literal["iq_complex", "rf_real"]
SignalPlane = Literal["baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf"]
StageDomain = Literal["iq_complex", "rf_real", "any"]

_VALID_DOMAINS = {"iq_complex", "rf_real"}
_VALID_PLANES = {"baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf"}
_ALL_PLANES = ("baseband", "awg_iq", "awg_rf", "qubit_iq", "qubit_rf")


def _as_array(values: Union[np.ndarray, Iterable[complex]]) -> np.ndarray:
    return np.asarray(values)


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
    "SignalTrace",
    "TransmissionChain",
    "TransmissionResult",
    "TransmissionStage",
]
