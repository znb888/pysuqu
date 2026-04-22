"""Solver primitives extracted from the legacy qubit monolith."""

from collections import OrderedDict
from copy import copy
from typing import List, Union

import numpy as np
from qutip import Qobj, expect, ket2dm

from ..funclib import cal_product_state, truncate_precision
from .compatibility import (
    _hamiltonian_evo_hamiltonian_evolution,
    _hamiltonian_evo_set_inistate,
)
from .types import SpectrumResult


_HAMILTONIAN_EIGENSYSTEM_CACHE_MAXSIZE = 64
_HAMILTONIAN_EIGENSYSTEM_CACHE = OrderedDict()


def _clone_solver_state(state):
    """Clone one cached eigensystem state for per-instance isolation."""
    if isinstance(state, Qobj):
        return Qobj(state.full(), dims=[list(state.dims[0]), list(state.dims[1])])
    return copy(state)


def _clone_solver_state_list(states):
    """Clone one cached eigensystem state list for per-instance isolation."""
    if states is None:
        return None
    return [_clone_solver_state(state) for state in states]


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

    @classmethod
    def _clear_hamiltonian_eigensystem_cache(cls) -> None:
        """Clear the shared exact-input eigensystem cache used by repeated identical solves."""
        _HAMILTONIAN_EIGENSYSTEM_CACHE.clear()

    def _build_hamiltonian_eigensystem_cache_key(self, hamiltonian: Qobj):
        """Build a stable cache key for one solved Hamiltonian when the active model exposes one."""
        exact_key_getter = getattr(self, '_get_exact_solve_template_cache_key', None)
        if callable(exact_key_getter):
            try:
                return ('exact-input', exact_key_getter())
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        dense_getter = getattr(hamiltonian, 'full', None)
        if not callable(dense_getter):
            return None

        dense_hamiltonian = np.asarray(dense_getter())
        dims = tuple(tuple(int(dim) for dim in side) for side in getattr(hamiltonian, 'dims', ()))
        return (
            'dense-hamiltonian',
            dims,
            dense_hamiltonian.shape,
            dense_hamiltonian.dtype.str,
            dense_hamiltonian.tobytes(),
        )

    def _get_cached_hamiltonian_eigensystem(self, hamiltonian: Qobj):
        """Return one cloned cached eigensystem payload when the current exact input was seen before."""
        cache_key = self._build_hamiltonian_eigensystem_cache_key(hamiltonian)
        if cache_key is None:
            return None

        cached_payload = _HAMILTONIAN_EIGENSYSTEM_CACHE.get(cache_key)
        if cached_payload is None:
            return None

        _HAMILTONIAN_EIGENSYSTEM_CACHE.move_to_end(cache_key)
        cached_energylevels, cached_eigenstates = cached_payload
        return (
            np.array(cached_energylevels, copy=True),
            _clone_solver_state_list(cached_eigenstates),
        )

    def _store_cached_hamiltonian_eigensystem(self, hamiltonian: Qobj, eigenvalues, eigenstates) -> None:
        """Persist one exact-input eigensystem payload with a small LRU-style eviction policy."""
        cache_key = self._build_hamiltonian_eigensystem_cache_key(hamiltonian)
        if cache_key is None:
            return

        _HAMILTONIAN_EIGENSYSTEM_CACHE[cache_key] = (
            np.array(eigenvalues, copy=True),
            tuple(_clone_solver_state_list(eigenstates)),
        )
        _HAMILTONIAN_EIGENSYSTEM_CACHE.move_to_end(cache_key)
        if len(_HAMILTONIAN_EIGENSYSTEM_CACHE) > _HAMILTONIAN_EIGENSYSTEM_CACHE_MAXSIZE:
            _HAMILTONIAN_EIGENSYSTEM_CACHE.popitem(last=False)

    def _solve_hamiltonian_eigensystem(self, hamiltonian: Qobj):
        """Solve one Hamiltonian eigensystem without touching the structured solver container."""
        return hamiltonian.eigenstates()

    def _materialize_exact_core_state_if_needed(self) -> None:
        """Realize deferred exact-template core state before a core-state read or mutation."""
        materialize_exact_core_state = getattr(self, '_materialize_pending_exact_core_state', None)
        if callable(materialize_exact_core_state):
            materialize_exact_core_state()

    def _clear_deferred_exact_restore_state(self) -> None:
        """Drop any deferred exact-template state when the live Hamiltonian is replaced."""
        if hasattr(self, '_pending_exact_core_template'):
            self._pending_exact_core_template = None
        if hasattr(self, '_pending_exact_auxiliary_template'):
            self._pending_exact_auxiliary_template = None
        if hasattr(self, '_active_exact_solve_template'):
            self._active_exact_solve_template = None

    def _set_solver_result(self, hamiltonian: Qobj):
        """Refresh cached eigensystem fields and the structured solver result."""
        cached_eigensystem = self._get_cached_hamiltonian_eigensystem(hamiltonian)
        if cached_eigensystem is None:
            self._energylevels, self._eigenstates = self._solve_hamiltonian_eigensystem(hamiltonian)
            self._store_cached_hamiltonian_eigensystem(
                hamiltonian,
                self._energylevels,
                self._eigenstates,
            )
        else:
            self._energylevels, self._eigenstates = cached_eigensystem
        self._solver_result = self._build_solver_result(
            hamiltonian,
            self._energylevels,
            self._eigenstates,
        )

    def get_hamiltonian(self) -> Qobj:
        """Get the Hamiltonian."""
        self._materialize_exact_core_state_if_needed()
        return truncate_precision(self._Hamiltonian)

    def add_hamiltonian(self, add_term: Qobj) -> Qobj:
        """Add a term to the Hamiltonian."""
        self._materialize_exact_core_state_if_needed()
        self._clear_deferred_exact_restore_state()
        self._Hamiltonian += add_term
        self._set_solver_result(self._Hamiltonian)
        return self._Hamiltonian

    def change_hamiltonian(self, new_hamiltonian: Qobj) -> Qobj:
        """Replace the Hamiltonian."""
        self._clear_deferred_exact_restore_state()
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
        self._materialize_exact_core_state_if_needed()
        return self._energylevels

    @property
    def eigenstates(self):
        """Eigenstates."""
        self._materialize_exact_core_state_if_needed()
        return [truncate_precision(state) for state in self._eigenstates]

    @property
    def solver_result(self) -> SpectrumResult:
        """Structured view of the latest solver output."""
        self._materialize_exact_core_state_if_needed()
        materialize_exact_auxiliary_state = getattr(self, '_materialize_pending_exact_auxiliary_state', None)
        if self._solver_result is None and callable(materialize_exact_auxiliary_state):
            materialize_exact_auxiliary_state()
        return self._solver_result

    def get_energylevel(self, label: int = None, mode='rel') -> Union[float, np.ndarray]:
        """Get energy level(s)."""
        self._materialize_exact_core_state_if_needed()
        if mode == 'rel':
            el = self._energylevels - self._energylevels[0]
        elif mode == 'abs':
            el = self._energylevels

        if label is None:
            return el
        return el[label]

    def get_eigenstate(self, label=None) -> Union[List[Qobj], Qobj]:
        """Get eigenstate(s)."""
        self._materialize_exact_core_state_if_needed()
        if label is None:
            return [truncate_precision(state) for state in self._eigenstates]
        return truncate_precision(self._eigenstates[label])

    def set_inistate(self, initial_state: Qobj):
        return _hamiltonian_evo_set_inistate(self, initial_state)

    def hamiltonian_evolution(self, *args, **kwargs):
        return _hamiltonian_evo_hamiltonian_evolution(self, *args, **kwargs)

    def find_state(
        self,
        state: Union[Qobj, list],
        mode: str = 'brief',
        threshold: float = 1e-2,
        state_space: List[Qobj] = None,
    ) -> Union[int, List[int]]:
        """Find the eigenstate index with the largest overlap."""
        self._materialize_exact_core_state_if_needed()
        state = self._normalize_state_lookup_input(state)

        num_qubit = len(self._Hamiltonian.dims[0])
        if state_space is None:
            # Internal hot paths only need the live eigensystem, not precision-truncated clones.
            state_space = self._eigenstates
        search_len = self._resolve_state_lookup_search_length(mode, num_qubit, len(state_space))
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

    def _normalize_state_lookup_input(self, state: Union[Qobj, list]) -> Qobj:
        """Normalize one lookup target into a Qobj without touching precision-truncated accessors."""
        if isinstance(state, list):
            if state and isinstance(state[0], list):
                state_temp = [cal_product_state(s, self._Nlevel) for s in state]
                normalized_state = state_temp[0]
                for i in range(1, len(state_temp)):
                    normalized_state += state_temp[i]
            else:
                normalized_state = cal_product_state(state, self._Nlevel)
        elif isinstance(state, Qobj):
            normalized_state = state
        else:
            raise ValueError(f"Unknown state type: {type(state)}. Expected Qobj, list, or list of list.")

        num_qubit = len(self._Hamiltonian.dims[0])
        if len(normalized_state.dims[0]) != num_qubit:
            raise ValueError(f"State dims {normalized_state.dims[0]} do not match system qubits {num_qubit}")
        return normalized_state

    @staticmethod
    def _resolve_state_lookup_search_length(mode: str, num_qubit: int, total_eig_len: int) -> int:
        """Resolve the shared state-lookup scan depth used by find_state() hot paths."""
        if mode == 'brief':
            return min(4 * num_qubit, total_eig_len)
        if mode == 'full':
            return total_eig_len
        raise ValueError(f"Unknown mode: '{mode}'. Use 'brev' or 'full'.")

    def find_state_list(
        self,
        state_list: List[Union[Qobj, list]],
        mode: str = 'brief',
        threshold: float = 1e-2,
        state_space: List[Qobj] = None,
    ) -> Union[int, List[int]]:
        """Find a list of states in one shared eigensystem scan on the hot path."""
        self._materialize_exact_core_state_if_needed()
        if state_space is None:
            state_space = self._eigenstates
        normalized_states = [self._normalize_state_lookup_input(state) for state in state_list]
        if not normalized_states:
            return []

        num_qubit = len(self._Hamiltonian.dims[0])
        search_len = self._resolve_state_lookup_search_length(mode, num_qubit, len(state_space))
        best_matches = [
            [-1, float('-inf'), -1, float('-inf')]
            for _ in normalized_states
        ]
        state_is_ket = [state.isket for state in normalized_states]

        for i in range(search_len):
            eig_state = state_space[i]
            eig_density = None
            eig_is_ket = eig_state.isket
            for state_idx, state in enumerate(normalized_states):
                if state_is_ket[state_idx] and eig_is_ket:
                    fidelity = abs(state.overlap(eig_state)) ** 2
                else:
                    if eig_density is None:
                        eig_density = ket2dm(eig_state)
                    fidelity = expect(eig_density, state)

                idx1, val1, idx2, val2 = best_matches[state_idx]
                if fidelity > val1:
                    best_matches[state_idx] = [i, fidelity, idx1, val1]
                elif fidelity > val2:
                    best_matches[state_idx][2] = i
                    best_matches[state_idx][3] = fidelity

        index_list = []
        for idx1, val1, idx2, val2 in best_matches:
            if idx2 == -1:
                index_list.append(idx1)
            elif abs(val1 - val2) < threshold:
                index_list.append([idx1, idx2])
            else:
                index_list.append(idx1)
        return index_list
