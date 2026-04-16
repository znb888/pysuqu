"""Circuit-topology helpers extracted from the legacy qubit base layer."""

from typing import Callable, Sequence

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
