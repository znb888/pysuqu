"""Circuit-topology helpers extracted from the legacy qubit base layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence, Union

import numpy as np
from scipy.constants import e, hbar, pi
from scipy.linalg import block_diag

from ..funclib.transmission import (
    OutOfBandPolicy,
    TouchstoneInterpolation,
    TouchstoneNetwork,
    evaluate_touchstone_response,
    load_touchstone_network,
)


def build_retain_nodes(struct: Sequence[int]) -> list[int]:
    """Rebuild retained-node indices from the circuit structure layout."""
    retain_nodes = []
    index = 0
    for item in struct:
        if item == 1:
            retain_nodes.append(index)
            index += 1
        elif item == 2:
            retain_nodes.append(index)
            index += 2
    return retain_nodes


def assemble_s_matrix_and_retain_nodes(struct: Sequence[int]) -> tuple[np.ndarray, list[int]]:
    """Assemble the circuit S-matrix and retained-node indices from the structure layout."""
    blocks = []
    for item in struct:
        if item == 1:
            blocks.append(np.array([[1.0]]))
        elif item == 2:
            blocks.append(np.array([[1.0, -1.0], [1.0, 1.0]]))

    return block_diag(*blocks), build_retain_nodes(struct)


def update_full_flux_from_reduced(
    reduced_flux: np.ndarray,
    current_full: np.ndarray,
    struct: Sequence[int],
    retain_nodes: Sequence[int],
) -> np.ndarray:
    """Update a full flux matrix from a reduced one using the retained-node layout."""
    full_flux = np.array(
        current_full,
        dtype=np.result_type(current_full, reduced_flux, float),
        copy=True,
    )

    for ii, retain_node in enumerate(retain_nodes):
        if struct[ii] == 2:
            full_flux[retain_node, retain_node + 1] = reduced_flux[ii, ii]
            full_flux[retain_node + 1, retain_node] = reduced_flux[ii, ii]
        else:
            full_flux[retain_node, retain_node] = reduced_flux[ii, ii]

    return full_flux


def extract_reduced_flux(
    full_flux: np.ndarray,
    struct: Sequence[int],
    retain_nodes: Sequence[int] | None = None,
) -> np.ndarray:
    """Extract the reduced flux matrix from the retained-node layout of a full matrix."""
    if retain_nodes is None:
        retain_nodes = build_retain_nodes(struct)

    return np.diag(
        [
            full_flux[retain_nodes[ii], retain_nodes[ii] + 1]
            if struct[ii] == 2
            else full_flux[retain_nodes[ii], retain_nodes[ii]]
            for ii in range(len(retain_nodes))
        ]
    )


def _project_retained_diagonal(
    values: np.ndarray,
    struct: Sequence[int],
    retain_nodes: Sequence[int],
) -> np.ndarray:
    """Project retained-node diagonal values from a full matrix layout."""
    return np.diag(
        [
            values[retain_nodes[ii], retain_nodes[ii] + 1]
            if struct[ii] == 2
            else values[retain_nodes[ii], retain_nodes[ii]]
            for ii in range(len(retain_nodes))
        ]
    )


def project_transformed_flux(
    flux: np.ndarray,
    struct: Sequence[int],
    retain_nodes: Sequence[int] | None = None,
) -> np.ndarray:
    """Project raw flux input into the transformed retained-node representation."""
    if retain_nodes is None:
        retain_nodes = build_retain_nodes(struct)

    flux_array = np.asarray(flux)
    if flux_array.ndim == 0:
        return np.full(len(retain_nodes), flux_array.item())
    if flux_array.ndim == 2:
        return _project_retained_diagonal(flux_array, struct, retain_nodes)
    return np.array(flux)


def project_transformed_junction_ratio(
    junc_ratio: np.ndarray,
    struct: Sequence[int],
    retain_nodes: Sequence[int] | None = None,
) -> np.ndarray:
    """Project raw junction-ratio input into the transformed retained-node representation."""
    if retain_nodes is None:
        retain_nodes = build_retain_nodes(struct)

    ratio_array = np.asarray(junc_ratio)
    if ratio_array.ndim == 2:
        return _project_retained_diagonal(ratio_array, struct, retain_nodes)
    return np.array(junc_ratio)


def convert_resistance_to_ej0(resis: float) -> float:
    """Convert a scalar junction resistance into its `Ej0` value."""
    critical_current = 280e-9
    reference_resistance = 1000.0
    current = critical_current * reference_resistance / resis
    return current * hbar / 2 / e


def convert_elements_to_energy_matrices(
    capac: np.ndarray,
    induc: np.ndarray,
    resis: np.ndarray,
    s_matrix: np.ndarray,
    retain_nodes: Sequence[int],
    struct: Sequence[int],
    resistance_to_ej0: Callable[[float], float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Convert circuit-element matrices into transformed `Ec`, `El`, and `Ej0` matrices."""
    capac_maxwell = -np.array(capac, copy=True)
    induc_maxwell = -1 / np.array(induc, copy=True)

    for ii in range(capac_maxwell.shape[0]):
        capac_maxwell[ii, ii] = -np.sum(capac_maxwell[ii])
        induc_maxwell[ii, ii] = -np.sum(induc_maxwell[ii])

    s_matrix_inv = np.linalg.inv(s_matrix)
    ec_matrix = e**2 * np.linalg.inv(capac_maxwell) / 2 / hbar / 1e9
    el_matrix = (hbar * pi / e / 2 / pi) ** 2 * induc_maxwell / hbar / 1e9
    ec_matrix_transform = np.dot(s_matrix, np.dot(ec_matrix, s_matrix.transpose()))[retain_nodes, :]
    el_matrix_transform = np.dot(s_matrix_inv.transpose(), np.dot(el_matrix, s_matrix_inv))[retain_nodes, :]
    ec_matrix_transform = ec_matrix_transform[:, retain_nodes]
    el_matrix_transform = el_matrix_transform[:, retain_nodes]

    resistance_to_ej0_vectorized = np.vectorize(resistance_to_ej0)
    ej_matrix = resistance_to_ej0_vectorized(resis) / hbar / 1e9
    ej0_matrix_transform = _project_retained_diagonal(ej_matrix, struct, retain_nodes)

    return (
        {'capac': capac_maxwell, 'induc': induc_maxwell},
        ec_matrix_transform,
        el_matrix_transform,
        ej0_matrix_transform,
    )


DriveCoupleType = str


def _coupling_value(value) -> float:
    """Normalize optional drive coupling values; None and 0 disable a channel."""
    if value is None:
        return 0.0
    return float(value)


def _normalize_drive_couplings(
    couple_term: Union[float, Sequence[float], np.ndarray, None],
    couple_type: DriveCoupleType,
) -> tuple[float, float]:
    """Return ``(inductive_H, capacitive_F)`` for supported drive coupling layouts."""
    normalized_type = couple_type.lower().replace("_", "").replace("-", "").replace("+", "")
    if normalized_type in {"induc", "ind"}:
        return _coupling_value(couple_term), 0.0
    if normalized_type in {"capac", "cap"}:
        return 0.0, _coupling_value(couple_term)

    pair = np.asarray(couple_term, dtype=object).ravel()
    if pair.size != 2:
        raise ValueError(f"Coupling type {couple_type!r} requires a pair of coupling terms.")
    if normalized_type in {"inducap", "indcap"}:
        return _coupling_value(pair[0]), _coupling_value(pair[1])
    if normalized_type in {"capind", "capacinduc", "capacind"}:
        return _coupling_value(pair[1]), _coupling_value(pair[0])
    raise ValueError(f"Unsupported couple_type: {couple_type}")


def _extract_primary_ec(ec: Union[float, np.ndarray]) -> float:
    ec_array = np.asarray(ec, dtype=float)
    if ec_array.ndim == 0:
        return float(ec_array)
    return float(ec_array[0, 0])


def transmon_effective_capacitance_from_ec(ec: Union[float, np.ndarray]) -> float:
    """Convert the package's ``Ec`` convention into an effective capacitance."""
    ec_value = _extract_primary_ec(ec)
    if not np.isfinite(ec_value) or ec_value <= 0:
        raise ValueError("Ec must be a positive finite value.")
    return e**2 / (2.0 * ec_value * 1e9 * hbar)


def estimate_drive_line_t1_ns(
    *,
    qubit_frequency_ghz: float,
    couple_term: Union[float, Sequence[float], np.ndarray, None],
    couple_type: DriveCoupleType = "induc",
    ec: Union[float, np.ndarray, None] = None,
    effective_capacitance_f: float | None = None,
    line_impedance_ohm: float = 50.0,
) -> float:
    """Estimate drive-line-induced ``T1`` under the weak-coupling LC-mode model.

    The return value is in nanoseconds. Mixed inductive and capacitive couplings
    combine as parallel decay channels; disabled channels return ``np.inf``.
    """
    if qubit_frequency_ghz <= 0:
        raise ValueError("qubit_frequency_ghz must be positive.")
    if line_impedance_ohm <= 0:
        raise ValueError("line_impedance_ohm must be positive.")

    if effective_capacitance_f is None:
        if ec is None:
            raise ValueError("estimate_drive_line_t1_ns requires ec or effective_capacitance_f.")
        effective_capacitance_f = transmon_effective_capacitance_from_ec(ec)
    if not np.isfinite(effective_capacitance_f) or effective_capacitance_f <= 0:
        raise ValueError("effective_capacitance_f must be positive.")

    omega_rad_per_s = 2.0 * pi * float(qubit_frequency_ghz) * 1e9
    induc_drive_h, capac_drive_f = _normalize_drive_couplings(couple_term, couple_type)

    decay_rates_per_s: list[float] = []
    if induc_drive_h > 0:
        t1_induc_s = line_impedance_ohm / (
            omega_rad_per_s**4 * induc_drive_h**2 * effective_capacitance_f
        )
        decay_rates_per_s.append(1.0 / t1_induc_s)
    if capac_drive_f > 0:
        correction = 1.0 + (omega_rad_per_s * capac_drive_f * line_impedance_ohm) ** 2
        t1_capac_s = (
            effective_capacitance_f
            * correction
            / (omega_rad_per_s**2 * capac_drive_f**2 * line_impedance_ohm)
        )
        decay_rates_per_s.append(1.0 / t1_capac_s)

    if not decay_rates_per_s:
        return float(np.inf)

    total_t1_s = 1.0 / float(np.sum(decay_rates_per_s))
    return total_t1_s * 1e9


@dataclass
class TransmonReflectionModel:
    """Weak-drive reflection model for a transmon used as a one-port load."""

    resonance_freq_ghz: float
    external_t1_ns: float
    internal_t1_ns: float | None = None
    pure_dephasing_tphi_ns: float | None = None
    termination: Literal["open", "short"] = "open"
    extra_phase_rad: float = 0.0
    name: str = "transmon_load"

    def __post_init__(self) -> None:
        if self.resonance_freq_ghz <= 0:
            raise ValueError("resonance_freq_ghz must be positive.")
        for value, field_name in (
            (self.external_t1_ns, "external_t1_ns"),
            (self.internal_t1_ns, "internal_t1_ns"),
            (self.pure_dephasing_tphi_ns, "pure_dephasing_tphi_ns"),
        ):
            if value is not None and value <= 0 and not np.isinf(value):
                suffix = " when provided" if field_name != "external_t1_ns" else ""
                raise ValueError(f"{field_name} must be positive or np.inf{suffix}.")
        if self.termination not in {"open", "short"}:
            raise ValueError("termination must be 'open' or 'short'.")

    @classmethod
    def from_qubit(
        cls,
        qubit,
        *,
        couple_term: Union[float, Sequence[float], np.ndarray],
        couple_type: DriveCoupleType = "induc",
        line_impedance_ohm: float = 50.0,
        internal_t1_ns: float | None = None,
        pure_dephasing_tphi_ns: float | None = None,
        resonance_freq_ghz: float | None = None,
        termination: Literal["auto", "open", "short"] = "auto",
        name: str = "transmon_load",
    ) -> "TransmonReflectionModel":
        """Build a load model from a qubit exposing ``Ec`` and ``qubit_f01``."""
        if resonance_freq_ghz is None:
            resonance_freq_ghz = getattr(qubit, "qubit_f01", getattr(qubit, "f01", None))
        if resonance_freq_ghz is None:
            raise AttributeError("qubit must expose qubit_f01 or f01 in GHz.")
        if not hasattr(qubit, "Ec"):
            raise AttributeError("qubit must expose Ec for capacitance estimation.")

        external_t1_ns = estimate_drive_line_t1_ns(
            qubit_frequency_ghz=float(resonance_freq_ghz),
            couple_term=couple_term,
            couple_type=couple_type,
            ec=qubit.Ec,
            line_impedance_ohm=line_impedance_ohm,
        )
        normalized_type = str(couple_type).lower().replace("_", "").replace("-", "").replace("+", "")
        if termination == "auto":
            resolved_termination = "open" if normalized_type in {"capac", "cap"} else "short"
        elif termination in {"open", "short"}:
            resolved_termination = termination
        else:
            raise ValueError("termination must be 'auto', 'open', or 'short'.")
        return cls(
            resonance_freq_ghz=float(resonance_freq_ghz),
            external_t1_ns=float(external_t1_ns),
            internal_t1_ns=internal_t1_ns,
            pure_dephasing_tphi_ns=pure_dephasing_tphi_ns,
            termination=resolved_termination,
            name=name,
        )

    @property
    def background_reflection(self) -> complex:
        """Return the off-resonant reflection coefficient of the line end."""
        return 1.0 + 0.0j if self.termination == "open" else -1.0 + 0.0j

    @property
    def external_decay_rate_per_ns(self) -> float:
        return 0.0 if np.isinf(self.external_t1_ns) else 1.0 / float(self.external_t1_ns)

    @property
    def internal_decay_rate_per_ns(self) -> float:
        if self.internal_t1_ns is None or np.isinf(self.internal_t1_ns):
            return 0.0
        return 1.0 / float(self.internal_t1_ns)

    @property
    def pure_dephasing_rate_per_ns(self) -> float:
        if self.pure_dephasing_tphi_ns is None or np.isinf(self.pure_dephasing_tphi_ns):
            return 0.0
        return 1.0 / float(self.pure_dephasing_tphi_ns)

    @property
    def transverse_decay_rate_per_ns(self) -> float:
        return (
            0.5 * (self.external_decay_rate_per_ns + self.internal_decay_rate_per_ns)
            + self.pure_dephasing_rate_per_ns
        )

    @property
    def hwhm_ghz(self) -> float:
        return float(self.transverse_decay_rate_per_ns / (2.0 * pi))

    @property
    def fwhm_ghz(self) -> float:
        return 2.0 * self.hwhm_ghz

    def adaptive_frequency_grid(self, *, span_hwhm: float = 20.0, points: int = 801) -> np.ndarray:
        """Return a resonance-centered grid that resolves the analytic linewidth."""
        if span_hwhm <= 0.0:
            raise ValueError("span_hwhm must be positive.")
        if points < 3:
            raise ValueError("points must be at least 3.")
        half_span = max(float(span_hwhm) * self.hwhm_ghz, np.finfo(float).eps)
        return np.linspace(
            float(self.resonance_freq_ghz) - half_span,
            float(self.resonance_freq_ghz) + half_span,
            int(points),
        )

    def reflection_coefficient(
        self,
        frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Evaluate the complex load reflection coefficient on a frequency grid."""
        freq = np.asarray(frequencies_ghz, dtype=np.float64)
        delta_omega = 2.0 * pi * (freq - float(self.resonance_freq_ghz))
        gamma2 = float(self.transverse_decay_rate_per_ns)
        kappa_ext = float(self.external_decay_rate_per_ns)
        resonant_factor = np.ones_like(freq, dtype=np.complex128)
        if kappa_ext != 0.0:
            resonant_factor -= kappa_ext / (gamma2 - 1j * delta_omega)
        response = complex(self.background_reflection) * resonant_factor
        if self.extra_phase_rad != 0.0:
            response *= np.exp(1j * float(self.extra_phase_rad))
        return np.asarray(response, dtype=np.complex128).reshape(freq.shape)

    def iq_reflection_coefficient(
        self,
        frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Evaluate the reflection in the IQ convention around the resonance."""
        freq = np.asarray(frequencies_ghz, dtype=np.float64)
        return self.reflection_coefficient(2.0 * float(self.resonance_freq_ghz) - freq)

    def filter_scattered_iq(
        self,
        values: Union[np.ndarray, Sequence[complex]],
        *,
        sample_rate: float,
        lo_freq_ghz: float,
    ) -> np.ndarray:
        """Apply the narrow transmon scattering correction with a causal IIR."""
        signal = np.asarray(values, dtype=np.complex128)
        if signal.ndim != 1:
            raise ValueError("values must be one-dimensional.")
        if sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        if signal.size == 0 or self.external_decay_rate_per_ns == 0.0:
            return np.zeros_like(signal, dtype=np.complex128)

        time_ns = np.arange(signal.size, dtype=np.float64) / float(sample_rate)
        if_ghz = float(self.resonance_freq_ghz) - float(lo_freq_ghz)
        rotating = signal * np.exp(-2j * pi * if_ghz * time_ns)
        gamma2 = float(self.transverse_decay_rate_per_ns)
        dt_ns = 1.0 / float(sample_rate)
        denominator = 1.0 + 0.5 * gamma2 * dt_ns
        state_decay = (1.0 - 0.5 * gamma2 * dt_ns) / denominator
        input_gain = 0.5 * dt_ns / denominator
        state = 0.0 + 0.0j
        filtered = np.zeros_like(rotating, dtype=np.complex128)
        for idx in range(1, rotating.size):
            state = state_decay * state + input_gain * (rotating[idx] + rotating[idx - 1])
            filtered[idx] = state
        scattered_rotating = (
            -complex(self.background_reflection)
            * np.exp(1j * float(self.extra_phase_rad))
            * float(self.external_decay_rate_per_ns)
            * filtered
        )
        return scattered_rotating * np.exp(2j * pi * if_ghz * time_ns)

    def input_impedance(
        self,
        frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
        *,
        line_impedance_ohm: float = 50.0,
    ) -> np.ndarray:
        """Convert the model reflection coefficient to input impedance."""
        if line_impedance_ohm <= 0.0:
            raise ValueError("line_impedance_ohm must be positive.")
        gamma = self.reflection_coefficient(frequencies_ghz)
        return line_impedance_ohm * (1.0 + gamma) / (1.0 - gamma)

    def __call__(
        self,
        frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Alias ``reflection_coefficient`` for load-spec compatibility."""
        return self.reflection_coefficient(frequencies_ghz)


LoadReflectionSpec = Union[
    complex,
    np.ndarray,
    Callable[[np.ndarray], Union[np.ndarray, complex]],
    TransmonReflectionModel,
]
MultiLoadReflectionSpec = Union[
    LoadReflectionSpec,
    Sequence[LoadReflectionSpec],
    np.ndarray,
]


def resolve_load_reflection_response(
    frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    load_reflection: LoadReflectionSpec,
) -> np.ndarray:
    """Evaluate a scalar, array, callable, or model load on one frequency grid."""
    freq = np.asarray(frequencies_ghz, dtype=np.float64)
    if hasattr(load_reflection, "reflection_coefficient"):
        resolved = load_reflection.reflection_coefficient(freq)
    elif callable(load_reflection):
        resolved = load_reflection(freq)
    else:
        resolved = load_reflection

    resolved_array = np.asarray(resolved, dtype=np.complex128)
    if resolved_array.ndim == 0:
        return np.full(freq.shape, resolved_array.item(), dtype=np.complex128)
    if resolved_array.shape != freq.shape:
        raise ValueError("Resolved load_reflection must be scalar or match frequencies_ghz shape.")
    return resolved_array


def calculate_loaded_single_port_response(
    frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    *,
    forward_response: Union[complex, np.ndarray, Sequence[complex]],
    output_reflection_response: Union[complex, np.ndarray, Sequence[complex]],
    load_reflection: LoadReflectionSpec,
    round_trip_delay_ns: float = 0.0,
) -> np.ndarray:
    """Sum the infinite forward/reflection series seen by a one-port load."""
    freq = np.asarray(frequencies_ghz, dtype=np.float64)
    forward = np.asarray(forward_response, dtype=np.complex128)
    output_reflection = np.asarray(output_reflection_response, dtype=np.complex128)
    if forward.shape != freq.shape or output_reflection.shape != freq.shape:
        raise ValueError("forward_response and output_reflection_response must match frequencies_ghz shape.")

    gamma_load = resolve_load_reflection_response(freq, load_reflection)
    loop_delay = np.exp(-2j * pi * freq * float(round_trip_delay_ns))
    loop_gain = output_reflection * gamma_load * loop_delay
    return forward / (1.0 - loop_gain)


@dataclass(frozen=True)
class LoadedMultiportWaveResponse:
    """Traveling-wave and local-field responses for terminated output ports."""

    port_outgoing: np.ndarray
    load_incident: np.ndarray
    load_reflected: np.ndarray
    local_voltage: np.ndarray
    local_current_equivalent_voltage: np.ndarray
    source_operator: np.ndarray
    return_operator: np.ndarray
    load_reflections: np.ndarray
    one_way_phase: np.ndarray
    loop_spectral_radius: np.ndarray
    system_condition_number: np.ndarray
    system_min_singular_value: np.ndarray


def _normalize_output_ports(output_ports: Sequence[int]) -> tuple[int, ...]:
    """Normalize and validate one-based output-port indices."""
    ports = tuple(int(port) for port in output_ports)
    if not ports:
        raise ValueError("output_ports must contain at least one port.")
    if len(set(ports)) != len(ports):
        raise ValueError("output_ports must not contain duplicates.")
    if any(port <= 0 for port in ports):
        raise ValueError("output_ports must contain one-based positive port indices.")
    return ports


def _normalize_per_output_values(
    values: Union[float, Sequence[float], np.ndarray],
    *,
    num_outputs: int,
    name: str,
) -> np.ndarray:
    """Broadcast one scalar or validate one value per output port."""
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim == 0:
        return np.full(num_outputs, float(values_array), dtype=np.float64)
    flat_values = values_array.reshape(-1)
    if len(flat_values) != num_outputs:
        raise ValueError(f"{name} must be scalar or contain one value per output port.")
    return np.asarray(flat_values, dtype=np.float64)


def resolve_multi_load_reflection_response(
    frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    *,
    load_reflections: MultiLoadReflectionSpec,
    num_outputs: int,
) -> np.ndarray:
    """Resolve one load-reflection specification per output port."""
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive.")

    freq = np.asarray(frequencies_ghz, dtype=np.float64)
    response_shape = (num_outputs,) + freq.shape
    if isinstance(load_reflections, np.ndarray):
        resolved_array = np.asarray(load_reflections, dtype=np.complex128)
        if resolved_array.ndim == 0:
            return np.full(response_shape, resolved_array.item(), dtype=np.complex128)
        if resolved_array.shape == freq.shape:
            return np.broadcast_to(resolved_array, response_shape).astype(np.complex128, copy=True)
        if resolved_array.shape == response_shape:
            return np.array(resolved_array, dtype=np.complex128, copy=True)
        raise ValueError(
            "load_reflections ndarray must be scalar, match frequencies_ghz shape, "
            "or have shape (num_outputs, *frequencies_ghz.shape)."
        )

    if isinstance(load_reflections, (list, tuple)):
        if len(load_reflections) != num_outputs:
            raise ValueError("load_reflections must contain one specification per output port.")
        resolved_rows = [
            resolve_load_reflection_response(freq, load_reflection)
            for load_reflection in load_reflections
        ]
        return np.stack(resolved_rows, axis=0).astype(np.complex128, copy=False)

    resolved = resolve_load_reflection_response(freq, load_reflections)
    return np.broadcast_to(resolved, response_shape).astype(np.complex128, copy=True)


def calculate_loaded_multiport_response(
    frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    *,
    forward_responses: np.ndarray,
    output_reflection_matrix: np.ndarray,
    load_reflections: MultiLoadReflectionSpec,
    round_trip_delays_ns: Union[float, Sequence[float], np.ndarray] = 0.0,
) -> np.ndarray:
    """Solve the terminated multi-output forward response ``(I-S_LL Gamma)^-1 S_Ls``."""
    freq = np.asarray(frequencies_ghz, dtype=np.float64)
    forward = np.asarray(forward_responses, dtype=np.complex128)
    s_ll = np.asarray(output_reflection_matrix, dtype=np.complex128)
    if forward.ndim < 1:
        raise ValueError("forward_responses must include an output-port axis.")

    num_outputs = int(forward.shape[0])
    expected_forward_shape = (num_outputs,) + freq.shape
    expected_matrix_shape = (num_outputs, num_outputs) + freq.shape
    if forward.shape != expected_forward_shape:
        raise ValueError(
            "forward_responses must have shape (n_outputs, *frequencies_ghz.shape)."
        )
    if s_ll.shape != expected_matrix_shape:
        raise ValueError(
            "output_reflection_matrix must have shape "
            "(n_outputs, n_outputs, *frequencies_ghz.shape)."
        )

    delays_ns = _normalize_per_output_values(
        round_trip_delays_ns,
        num_outputs=num_outputs,
        name="round_trip_delays_ns",
    )
    gamma = resolve_multi_load_reflection_response(
        freq,
        load_reflections=load_reflections,
        num_outputs=num_outputs,
    )
    gamma_eff = gamma * np.exp(
        -2j * pi * delays_ns.reshape((num_outputs,) + (1,) * freq.ndim) * freq
    )
    forward_flat = forward.reshape(num_outputs, -1)
    s_ll_flat = s_ll.reshape(num_outputs, num_outputs, -1)
    gamma_flat = gamma_eff.reshape(num_outputs, -1)
    loop_matrix = s_ll_flat * gamma_flat[np.newaxis, :, :]
    system = np.moveaxis(
        np.eye(num_outputs, dtype=np.complex128)[:, :, np.newaxis] - loop_matrix,
        -1,
        0,
    )
    rhs = np.moveaxis(forward_flat, -1, 0)[:, :, np.newaxis]
    loaded = np.linalg.solve(system, rhs)[:, :, 0]
    return np.moveaxis(loaded, 0, -1).reshape(expected_forward_shape)


def calculate_loaded_multiport_wave_response(
    frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    *,
    forward_responses: np.ndarray,
    output_reflection_matrix: np.ndarray,
    load_reflections: MultiLoadReflectionSpec,
    round_trip_delays_ns: Union[float, Sequence[float], np.ndarray] = 0.0,
    return_coupling: Literal["full", "diagonal"] = "full",
) -> LoadedMultiportWaveResponse:
    """Solve terminated port waves and convert them to load-plane voltage/current."""
    freq = np.asarray(frequencies_ghz, dtype=np.float64)
    forward = np.asarray(forward_responses, dtype=np.complex128)
    s_ll = np.asarray(output_reflection_matrix, dtype=np.complex128)
    if forward.ndim < 1:
        raise ValueError("forward_responses must include an output-port axis.")
    num_outputs = int(forward.shape[0])
    expected_forward_shape = (num_outputs,) + freq.shape
    expected_matrix_shape = (num_outputs, num_outputs) + freq.shape
    if forward.shape != expected_forward_shape:
        raise ValueError(
            "forward_responses must have shape (n_outputs, *frequencies_ghz.shape)."
        )
    if s_ll.shape != expected_matrix_shape:
        raise ValueError(
            "output_reflection_matrix must have shape "
            "(n_outputs, n_outputs, *frequencies_ghz.shape)."
        )
    if return_coupling not in {"full", "diagonal"}:
        raise ValueError("return_coupling must be 'full' or 'diagonal'.")

    gamma = resolve_multi_load_reflection_response(
        freq,
        load_reflections=load_reflections,
        num_outputs=num_outputs,
    )
    delays_ns = _normalize_per_output_values(
        round_trip_delays_ns,
        num_outputs=num_outputs,
        name="round_trip_delays_ns",
    )
    one_way_phase = np.exp(
        -1j * pi * delays_ns.reshape((num_outputs,) + (1,) * freq.ndim) * freq
    )

    forward_flat = forward.reshape(num_outputs, -1)
    s_ll_flat = s_ll.reshape(num_outputs, num_outputs, -1)
    gamma_flat = gamma.reshape(num_outputs, -1)
    phase_flat = one_way_phase.reshape(num_outputs, -1)
    gamma_eff_flat = gamma_flat * phase_flat**2
    loop_matrix = s_ll_flat * gamma_eff_flat[np.newaxis, :, :]
    system = np.moveaxis(
        np.eye(num_outputs, dtype=np.complex128)[:, :, np.newaxis] - loop_matrix,
        -1,
        0,
    )
    rhs_source = np.moveaxis(forward_flat, -1, 0)[:, :, np.newaxis]
    port_outgoing_flat = np.linalg.solve(system, rhs_source)[:, :, 0]
    port_outgoing = np.moveaxis(port_outgoing_flat, 0, -1).reshape(expected_forward_shape)
    load_incident = one_way_phase * port_outgoing
    load_reflected = gamma * load_incident

    rhs_return = s_ll_flat * phase_flat[np.newaxis, :, :]
    rhs_return = np.moveaxis(rhs_return, -1, 0)
    return_flat = np.linalg.solve(system, rhs_return)
    return_flat = return_flat * phase_flat.T[:, :, np.newaxis]
    return_operator = np.moveaxis(return_flat, 0, -1).reshape(expected_matrix_shape)
    if return_coupling == "diagonal":
        mask = np.eye(num_outputs, dtype=bool).reshape(
            (num_outputs, num_outputs) + (1,) * freq.ndim
        )
        return_operator = np.where(mask, return_operator, 0.0)

    singular_values = np.linalg.svd(system, compute_uv=False)
    min_singular = singular_values[:, -1]
    condition = np.divide(
        singular_values[:, 0],
        min_singular,
        out=np.full_like(min_singular, np.inf, dtype=np.float64),
        where=min_singular > 0.0,
    )
    spectral_radius = np.max(
        np.abs(np.linalg.eigvals(np.moveaxis(loop_matrix, -1, 0))),
        axis=-1,
    )
    diagnostic_shape = freq.shape

    return LoadedMultiportWaveResponse(
        port_outgoing=port_outgoing,
        load_incident=load_incident,
        load_reflected=load_reflected,
        local_voltage=(1.0 + gamma) * load_incident,
        local_current_equivalent_voltage=(1.0 - gamma) * load_incident,
        source_operator=load_incident,
        return_operator=return_operator,
        load_reflections=gamma,
        one_way_phase=one_way_phase,
        loop_spectral_radius=np.asarray(spectral_radius).reshape(diagnostic_shape),
        system_condition_number=np.asarray(condition).reshape(diagnostic_shape),
        system_min_singular_value=np.asarray(min_singular).reshape(diagnostic_shape),
    )


def evaluate_touchstone_multiport_output_block(
    query_frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    *,
    output_ports: Sequence[int],
    file_path: Union[str, Path, None] = None,
    network: TouchstoneNetwork | None = None,
    input_port: int = 1,
    interpolation: TouchstoneInterpolation = "polar",
    out_of_band: OutOfBandPolicy = "edge",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``S[L,s]`` and ``S[L,L]`` for an ordered output-port set."""
    ports = _normalize_output_ports(output_ports)
    if network is None:
        if file_path is None:
            raise ValueError("evaluate_touchstone_multiport_output_block requires either file_path or network.")
        network = load_touchstone_network(file_path)
    elif not isinstance(network, TouchstoneNetwork):
        raise TypeError("network must be a TouchstoneNetwork instance.")

    freq = np.asarray(query_frequencies_ghz, dtype=np.float64)
    forward = np.stack(
        [
            evaluate_touchstone_response(
                freq,
                network=network,
                input_port=input_port,
                output_port=port,
                interpolation=interpolation,
                out_of_band=out_of_band,
            )
            for port in ports
        ],
        axis=0,
    )
    s_ll = np.empty((len(ports), len(ports)) + freq.shape, dtype=np.complex128)
    for out_idx, output_port in enumerate(ports):
        for in_idx, reflection_port in enumerate(ports):
            s_ll[out_idx, in_idx] = evaluate_touchstone_response(
                freq,
                network=network,
                input_port=reflection_port,
                output_port=output_port,
                interpolation=interpolation,
                out_of_band=out_of_band,
            )
    return forward, s_ll


def evaluate_loaded_touchstone_multiport_wave_response(
    query_frequencies_ghz: Union[float, np.ndarray, Sequence[float]],
    *,
    output_ports: Sequence[int],
    load_reflections: MultiLoadReflectionSpec = 1.0 + 0.0j,
    round_trip_delays_ns: Union[float, Sequence[float], np.ndarray] = 0.0,
    return_coupling: Literal["full", "diagonal"] = "full",
    file_path: Union[str, Path, None] = None,
    network: TouchstoneNetwork | None = None,
    input_port: int = 1,
    interpolation: TouchstoneInterpolation = "polar",
    out_of_band: OutOfBandPolicy = "edge",
) -> LoadedMultiportWaveResponse:
    """Evaluate terminated port waves and load-plane fields from Touchstone data."""
    forward, s_ll = evaluate_touchstone_multiport_output_block(
        query_frequencies_ghz,
        output_ports=output_ports,
        file_path=file_path,
        network=network,
        input_port=input_port,
        interpolation=interpolation,
        out_of_band=out_of_band,
    )
    return calculate_loaded_multiport_wave_response(
        query_frequencies_ghz,
        forward_responses=forward,
        output_reflection_matrix=s_ll,
        load_reflections=load_reflections,
        round_trip_delays_ns=round_trip_delays_ns,
        return_coupling=return_coupling,
    )
