"""Circuit-topology helpers extracted from the legacy qubit base layer."""

from typing import Callable, Sequence, Union

import numpy as np
from scipy.constants import e, hbar, pi
from scipy.linalg import block_diag


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
