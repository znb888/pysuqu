"""Solver primitives extracted from the legacy qubit monolith."""

from typing import List, Union

import numpy as np
from qutip import Qobj, expect, ket2dm

from ..funclib import cal_product_state, truncate_precision
from .types import SpectrumResult


class HamiltonianEvo:
    """Common Hamiltonian solve/update behavior shared by qubit models."""

    def __init__(self, Hamiltonian: Qobj, *args, **kwargs):
        self._Hamiltonian = Hamiltonian
        self._Nlevel = Hamiltonian.dims[0]
        self._qubits_num = len(self._Nlevel)
        self._set_solver_result(Hamiltonian)

    def _build_solver_result(self, hamiltonian: Qobj, eigenvalues, eigenstates) -> SpectrumResult:
        """Collect the current Hamiltonian solve artifacts into a single container."""
        return SpectrumResult(
            hamiltonian=hamiltonian,
            eigenvalues=np.array(eigenvalues),
            eigenstates=list(eigenstates),
            destroy_operators=getattr(self, 'destroyors', None),
            number_operators=getattr(self, 'n_operators', None),
            phase_operators=getattr(self, 'phi_operators', None),
        )

    def _set_solver_result(self, hamiltonian: Qobj):
        """Refresh cached eigensystem fields and the structured solver result."""
        self._energylevels, self._eigenstates = hamiltonian.eigenstates()
        self._solver_result = self._build_solver_result(
            hamiltonian,
            self._energylevels,
            self._eigenstates,
        )

    def get_hamiltonian(self) -> Qobj:
        """Get the Hamiltonian."""
        return truncate_precision(self._Hamiltonian)

    def add_hamiltonian(self, add_term: Qobj) -> Qobj:
        """Add a term to the Hamiltonian."""
        self._Hamiltonian += add_term
        self._set_solver_result(self._Hamiltonian)
        return self._Hamiltonian

    def change_hamiltonian(self, new_hamiltonian: Qobj) -> Qobj:
        """Replace the Hamiltonian."""
        self._Hamiltonian = new_hamiltonian
        self._Nlevel = new_hamiltonian.dims[0]
        self._qubits_num = len(self._Nlevel)
        self._set_solver_result(new_hamiltonian)
        return self._Hamiltonian

    @property
    def qubits_num(self):
        """Number of qubits."""
        return self._qubits_num

    @property
    def Nlevel(self):
        """Number of energy levels."""
        return self._Nlevel

    @property
    def energylevels(self):
        """Energy level array [2*pi*GHz]."""
        return self._energylevels

    @property
    def eigenstates(self):
        """Eigenstates."""
        return [truncate_precision(state) for state in self._eigenstates]

    @property
    def solver_result(self) -> SpectrumResult:
        """Structured view of the latest solver output."""
        return self._solver_result

    def get_energylevel(self, label: int = None, mode='rel') -> Union[float, np.ndarray]:
        """Get energy level(s)."""
        if mode == 'rel':
            el = self._energylevels - self._energylevels[0]
        elif mode == 'abs':
            el = self._energylevels

        if label is None:
            return el
        return el[label]

    def get_eigenstate(self, label=None) -> Union[List[Qobj], Qobj]:
        """Get eigenstate(s)."""
        if label is None:
            return [truncate_precision(state) for state in self._eigenstates]
        return truncate_precision(self._eigenstates[label])

    @staticmethod
    def _raise_placeholder(method_name: str):
        raise NotImplementedError(f"HamiltonianEvo.{method_name}() is not implemented yet.")

    def set_inistate(self, initial_state: Qobj):
        self._raise_placeholder('set_inistate')

    def hamiltonian_evolution(self, *args, **kwargs):
        self._raise_placeholder('hamiltonian_evolution')

    def find_state(
        self,
        state: Union[Qobj, list],
        mode: str = 'brief',
        threshold: float = 1e-2,
        state_space: List[Qobj] = None,
    ) -> Union[int, List[int]]:
        """Find the eigenstate index with the largest overlap."""
        if isinstance(state[0], list):
            state_temp = [cal_product_state(s, self._Nlevel) for s in state]
            state = state_temp[0]
            for i in range(1, len(state_temp)):
                state += state_temp[i]
        elif isinstance(state, list):
            state = cal_product_state(state, self._Nlevel)
        elif isinstance(state, Qobj):
            pass
        else:
            raise ValueError(f"Unknown state type: {type(state)}. Expected Qobj, list, or list of list.")

        num_qubit = len(self._Hamiltonian.dims[0])
        if len(state.dims[0]) != num_qubit:
            raise ValueError(f"State dims {state.dims[0]} do not match system qubits {num_qubit}")

        if state_space is None:
            # Internal hot paths only need the live eigensystem, not precision-truncated clones.
            state_space = self._eigenstates
        total_eig_len = len(state_space)
        if mode == 'brief':
            search_len = min(4 * num_qubit, total_eig_len)
        elif mode == 'full':
            search_len = total_eig_len
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'brev' or 'full'.")
        is_state_ket = state.isket
        idx1 = -1
        val1 = float('-inf')
        idx2 = -1
        val2 = float('-inf')

        for i in range(search_len):
            eig_state = state_space[i]

            if is_state_ket and eig_state.isket:
                fidelity = abs(state.overlap(eig_state)) ** 2
            else:
                fidelity = expect(ket2dm(eig_state), state)

            if fidelity > val1:
                idx2, val2 = idx1, val1
                idx1, val1 = i, fidelity
            elif fidelity > val2:
                idx2, val2 = i, fidelity

        if idx2 == -1:
            return idx1

        if abs(val1 - val2) < threshold:
            return [idx1, idx2]
        return idx1

    def find_state_list(
        self,
        state_list: List[Union[Qobj, list]],
        mode: str = 'brief',
        threshold: float = 1e-2,
        state_space: List[Qobj] = None,
    ) -> Union[int, List[int]]:
        """Find a list of states in state_space by delegating to find_state."""
        if state_space is None:
            state_space = self._eigenstates
        index_list = [self.find_state(state, mode, threshold, state_space) for state in state_list]
        return index_list
