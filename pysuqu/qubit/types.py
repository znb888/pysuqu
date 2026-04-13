"""Structured qubit data objects extracted from the qubit package."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from qutip import Qobj


@dataclass(frozen=True)
class FluxSpec:
    """
    Immutable flux specification with full and reduced representations.

    Full representation: Complete flux matrix (for example 5x5 for a 5-node circuit)
    Reduced representation: Minimal flux matrix (for example 3x3 for 3 qubits)
    """

    full: Optional[np.ndarray] = None
    reduced: Optional[np.ndarray] = None

    @classmethod
    def from_full(cls, full_flux: np.ndarray, struct: List[int], nodes: int) -> 'FluxSpec':
        """Create FluxSpec from a full flux matrix."""
        reduced = cls._extract_reduced(full_flux, struct, nodes)
        return cls(full=full_flux.copy(), reduced=reduced)

    @classmethod
    def from_reduced(cls, reduced_flux: np.ndarray, struct: List[int], nodes: int) -> 'FluxSpec':
        """Create FluxSpec from a reduced flux matrix."""
        full = cls._build_full(reduced_flux, struct, nodes)
        return cls(full=full, reduced=reduced_flux.copy())

    @classmethod
    def _extract_reduced(cls, full_flux: np.ndarray, struct: List[int], nodes: int) -> np.ndarray:
        """Extract reduced flux from a full flux matrix."""
        retain_nodes = []
        index = 0
        for item in struct:
            if item == 1:
                retain_nodes.append(index)
                index += 1
            elif item == 2:
                retain_nodes.append(index)
                index += 2

        return np.diag(
            [
                full_flux[retain_nodes[ii]][retain_nodes[ii] + 1]
                if struct[ii] == 2
                else full_flux[retain_nodes[ii]][retain_nodes[ii]]
                for ii in range(len(retain_nodes))
            ]
        )

    @classmethod
    def _build_full(cls, reduced_flux: np.ndarray, struct: List[int], nodes: int) -> np.ndarray:
        """Build a full flux matrix from a reduced flux matrix."""
        full_flux = np.zeros((nodes, nodes))
        retain_nodes = []
        index = 0
        for item in struct:
            if item == 1:
                retain_nodes.append(index)
                index += 1
            elif item == 2:
                retain_nodes.append(index)
                index += 2

        for ii in range(len(retain_nodes)):
            if struct[ii] == 2:
                full_flux[retain_nodes[ii]][retain_nodes[ii] + 1] = reduced_flux[ii, ii]
                full_flux[retain_nodes[ii] + 1][retain_nodes[ii]] = reduced_flux[ii, ii]
            else:
                full_flux[retain_nodes[ii]][retain_nodes[ii]] = reduced_flux[ii, ii]

        return full_flux

    def with_updated_full(self, new_full: np.ndarray, struct: List[int], nodes: int) -> 'FluxSpec':
        """Return a new FluxSpec with an updated full representation."""
        return self.from_full(new_full, struct, nodes)

    def with_updated_reduced(
        self, new_reduced: np.ndarray, struct: List[int], nodes: int
    ) -> 'FluxSpec':
        """Return a new FluxSpec with an updated reduced representation."""
        return self.from_reduced(new_reduced, struct, nodes)


@dataclass(frozen=True)
class SpectrumResult:
    """
    Structured solver output collected from the current Hamiltonian state.

    This keeps the core solver artifacts together for the active solver API.
    """

    hamiltonian: Qobj
    eigenvalues: np.ndarray
    eigenstates: list
    destroy_operators: Optional[list] = None
    number_operators: Optional[list] = None
    phase_operators: Optional[list] = None


@dataclass(frozen=True)
class CouplingResult:
    """Structured coupling-sweep output for extracted helper surfaces."""

    sweep_parameter: str
    sweep_values: list[float]
    coupling_values: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'sweep_values', list(self.sweep_values))
        object.__setattr__(
            self,
            'coupling_values',
            np.asarray(self.coupling_values, dtype=float).copy(),
        )
        object.__setattr__(self, 'metadata', dict(self.metadata))

@dataclass(frozen=True)
class SweepResult:
    """Structured sweep output for extracted helper surfaces."""

    sweep_parameter: str
    sweep_values: list[object]
    series: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            'sweep_values',
            [
                value.copy() if isinstance(value, np.ndarray) else value
                for value in self.sweep_values
            ],
        )
        object.__setattr__(
            self,
            'series',
            {name: np.asarray(values, dtype=float).copy() for name, values in self.series.items()},
        )
        object.__setattr__(self, 'metadata', dict(self.metadata))

@dataclass(frozen=True)
class SensitivityResult:
    """Structured sensitivity output for extracted analysis-helper surfaces."""

    coupler_flux_point: float
    sensitivity_value: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        metadata = dict(self.metadata)
        if metadata.get('qubit_fluxes') is not None:
            metadata['qubit_fluxes'] = list(metadata['qubit_fluxes'])

        object.__setattr__(self, 'coupler_flux_point', float(self.coupler_flux_point))
        object.__setattr__(self, 'sensitivity_value', float(self.sensitivity_value))
        object.__setattr__(self, 'metadata', metadata)

class FluxState:
    """Mutable flux state manager with dual full and reduced representations."""

    def __init__(self, full: np.ndarray = None, reduced: np.ndarray = None):
        self._full = full
        self._reduced = reduced

    @property
    def full(self) -> Optional[np.ndarray]:
        return self._full

    @property
    def reduced(self) -> Optional[np.ndarray]:
        return self._reduced

    def update_from_full(self, new_full: np.ndarray, struct: List[int], nodes: int):
        """Update state from a full flux matrix."""
        spec = FluxSpec.from_full(new_full, struct, nodes)
        self._full = spec.full
        self._reduced = spec.reduced

    def update_from_reduced(self, new_reduced: np.ndarray, struct: List[int], nodes: int):
        """Update state from a reduced flux matrix."""
        spec = FluxSpec.from_reduced(new_reduced, struct, nodes)
        self._full = spec.full
        self._reduced = spec.reduced

    def has_full(self) -> bool:
        return self._full is not None

    def has_reduced(self) -> bool:
        return self._reduced is not None
