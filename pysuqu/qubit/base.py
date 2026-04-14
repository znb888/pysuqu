"""Base qubit classes extracted from the legacy qubit monolith."""

import math
from abc import abstractmethod
from collections import OrderedDict
from copy import copy
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
from qutip import Qobj, destroy, qeye, tensor
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize
from scipy.constants import e, h, hbar, pi

from ..funclib import truncate_hilbert_space
from .circuit import (
    assemble_s_matrix_and_retain_nodes,
    build_retain_nodes,
    convert_elements_to_energy_matrices,
    convert_resistance_to_ej0,
    project_transformed_flux,
    project_transformed_junction_ratio,
    update_full_flux_from_reduced,
)
from .solver import HamiltonianEvo

Phi0 = hbar * pi / e
_QUBIT_EXACT_SOLVE_TEMPLATE_CACHE_MAXSIZE = 64
_QUBIT_EXACT_SOLVE_TEMPLATE_CACHE = OrderedDict()
_PARAMETERIZED_EMATRIX_TEMPLATE_CACHE_MAXSIZE = 64
_PARAMETERIZED_EMATRIX_TEMPLATE_CACHE = OrderedDict()


class QubitBase(HamiltonianEvo):
    """Base class for all qubit implementations."""

    _PARAM_ATTR_MAP = {
        'flux': '_flux',
        'junc_ratio': '_junc_ratio',
        'Nlevel': '_Nlevel',
        'charges': '_charges',
        'cal_mode': '_cal_mode',
    }

    @staticmethod
    def _clone_qobj(obj: Optional[Qobj]) -> Optional[Qobj]:
        """Clone one qutip object for per-instance isolation across cached templates."""
        if obj is None:
            return None
        if hasattr(obj, 'copy'):
            return obj.copy()
        return copy(obj)

    @staticmethod
    def _clone_ndarray(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Clone one ndarray-like payload for per-instance isolation across cached templates."""
        if values is None:
            return None
        return np.array(values, copy=True)

    @classmethod
    def _clone_qobj_list(cls, objects: Optional[list[Qobj] | tuple[Qobj, ...]]) -> Optional[list[Qobj]]:
        """Clone a qutip-object list for per-instance isolation across cached templates."""
        if objects is None:
            return None
        return [cls._clone_qobj(obj) for obj in objects]

    @classmethod
    def _clone_ndarray_mapping(
        cls,
        mapping: Optional[dict[str, np.ndarray]],
    ) -> Optional[dict[str, np.ndarray]]:
        """Clone one mapping of ndarray payloads for per-instance isolation."""
        if mapping is None:
            return None
        return {key: cls._clone_ndarray(value) for key, value in mapping.items()}

    @staticmethod
    def _get_array_cache_key(
        values: np.ndarray,
        *,
        as_int: bool = False,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Normalize one numeric array into a shape-aware exact-input cache key."""
        array = np.asarray(values)
        if as_int:
            flattened = tuple(int(value) for value in array.reshape(-1).tolist())
        else:
            flattened = tuple(float(value) for value in array.reshape(-1).tolist())
        return tuple(int(dim) for dim in array.shape), flattened

    def _get_exact_solve_template_cache_key(self) -> tuple[object, ...]:
        """Return the exact-input constructor solve-template cache key."""
        return (
            self._cal_mode,
            self._get_array_cache_key(self.Ec),
            self._get_array_cache_key(self.El),
            self._get_array_cache_key(self.Ej),
            self._get_array_cache_key(self._charges),
            self._get_array_cache_key(self._Nlevel, as_int=True),
        )

    @classmethod
    def _get_cached_exact_solve_template(cls, cache_key: tuple[object, ...]) -> Optional[dict[str, object]]:
        """Return one cached exact-input constructor solve template when available."""
        template = _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE.get(cache_key)
        if template is not None:
            _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE.move_to_end(cache_key)
        return template

    @classmethod
    def _store_cached_exact_solve_template(
        cls,
        cache_key: tuple[object, ...],
        template: dict[str, object],
    ) -> None:
        """Store one exact-input constructor solve template with LRU-style eviction."""
        _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE[cache_key] = template
        _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE.move_to_end(cache_key)
        if len(_QUBIT_EXACT_SOLVE_TEMPLATE_CACHE) > _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE_MAXSIZE:
            _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE.popitem(last=False)

    @classmethod
    def _clear_exact_solve_template_cache(cls) -> None:
        """Clear the exact-input constructor solve-template cache used by repeated builds."""
        _QUBIT_EXACT_SOLVE_TEMPLATE_CACHE.clear()

    def _restore_cached_exact_solve_template_if_available(self) -> bool:
        """Restore one exact-input solve template when the live state matches a cached build."""
        if getattr(self, '_cal_mode', None) != 'Eigen':
            return False

        cache_key = self._get_exact_solve_template_cache_key()
        cached_template = self._get_cached_exact_solve_template(cache_key)
        if cached_template is None:
            return False

        self._restore_exact_solve_template(cached_template)
        return True

    def _store_current_exact_solve_template(self) -> None:
        """Persist the current exact-input solve template for later same-process reuse."""
        if getattr(self, '_cal_mode', None) != 'Eigen':
            return

        self._store_cached_exact_solve_template(
            self._get_exact_solve_template_cache_key(),
            self._capture_exact_solve_template(),
        )

    def _capture_exact_solve_template(self) -> dict[str, object]:
        """Capture the current solved constructor state for exact-input reuse."""
        return {
            'hamiltonian': self._clone_qobj(self._Hamiltonian),
            'energylevels': np.array(self._energylevels, copy=True),
            'eigenstates': tuple(self._clone_qobj(state) for state in self._eigenstates),
            'destroyors': tuple(self._clone_qobj_list(getattr(self, 'destroyors', None)) or ()),
            'number_operators': tuple(self._clone_qobj_list(getattr(self, 'n_operators', None)) or ()),
            'phase_operators': tuple(self._clone_qobj_list(getattr(self, 'phi_operators', None)) or ()),
            'eigen_hamiltonian': self._clone_qobj(getattr(self, 'eigenHamiltonian', None)),
            'coupling_hamiltonian': self._clone_qobj(getattr(self, 'couplingHamiltonian', None)),
            'highorder_hamiltonian': self._clone_qobj(getattr(self, 'highorderHamiltonian', None)),
        }

    def _restore_exact_solve_template(self, template: dict[str, object]) -> None:
        """Restore one cached constructor solve template onto this instance."""
        self._hamiltonian = self._clone_qobj(template['hamiltonian'])
        self._Hamiltonian = self._hamiltonian
        self._energylevels = np.array(template['energylevels'], copy=True)
        self._eigenstates = self._clone_qobj_list(template['eigenstates'])
        self.destroyors = self._clone_qobj_list(template['destroyors'])
        self.n_operators = self._clone_qobj_list(template['number_operators'])
        self.phi_operators = self._clone_qobj_list(template['phase_operators'])
        self.eigenHamiltonian = self._clone_qobj(template['eigen_hamiltonian'])
        self.couplingHamiltonian = self._clone_qobj(template['coupling_hamiltonian'])
        self.highorderHamiltonian = self._clone_qobj(template['highorder_hamiltonian'])
        self._Nlevel = list(self._Hamiltonian.dims[0])
        self._qubits_num = len(self._Nlevel)
        self._solver_result = self._build_solver_result(
            self._Hamiltonian,
            self._energylevels,
            self._eigenstates,
        )

    def _normalize_flux_input(self, value):
        """Accept scalar, reduced, or full flux forms for active qubit workflows."""
        new_flux = np.asarray(value)
        if not hasattr(self, '_flux'):
            return new_flux

        current_flux = np.asarray(self._flux)
        if current_flux.shape == new_flux.shape:
            return new_flux

        if current_flux.ndim == 0 and new_flux.shape == (1, 1):
            return new_flux[0, 0].item()

        if current_flux.ndim == 2 and current_flux.shape == (1, 1) and new_flux.ndim == 0:
            return np.array([[new_flux.item()]])

        struct = getattr(self, '_ParameterizedQubit__struct', None)
        nodes = getattr(self, '_ParameterizedQubit__nodes', None)
        if not struct or nodes is None:
            raise ValueError(f"Flux shape mismatch: expected {current_flux.shape}, got {new_flux.shape}")

        reduced_size = len(struct)
        reduced_flux = None
        if new_flux.ndim == 0:
            if reduced_size != 1:
                raise ValueError(f"Flux shape mismatch: expected {current_flux.shape}, got {new_flux.shape}")
            reduced_flux = np.array([[new_flux.item()]])
        elif new_flux.ndim == 2 and new_flux.shape == (reduced_size, reduced_size):
            reduced_flux = new_flux

        if reduced_flux is None:
            raise ValueError(f"Flux shape mismatch: expected {current_flux.shape}, got {new_flux.shape}")

        if current_flux.ndim == 0:
            return reduced_flux[0, 0].item()

        if current_flux.ndim == 2 and current_flux.shape == reduced_flux.shape:
            return reduced_flux

        if current_flux.ndim == 2 and current_flux.shape == (nodes, nodes):
            retain_nodes = getattr(self, 'SMatrix_retainNodes', None) or build_retain_nodes(struct)
            return update_full_flux_from_reduced(reduced_flux, current_flux, struct, retain_nodes)

        raise ValueError(f"Flux shape mismatch: expected {current_flux.shape}, got {new_flux.shape}")

    def __init__(
        self,
        Ec: np.ndarray,
        El: np.ndarray,
        Ej: np.ndarray,
        fluxes: np.ndarray = [None],
        charges: Optional[list[float]] = [None],
        junc_ratio: Optional[np.ndarray] = [None],
        trunc_ener_level: list[float] = [None],
        cal_mode: str = 'Eigen',
        *args,
        **kwargs,
    ):
        self.Ec = np.array(Ec)
        self.El = np.array(El)
        self.Ejmax = np.array(Ej)

        self._junc_ratio = junc_ratio if np.any(junc_ratio) else np.ones_like(Ej)

        self._numQubits = len(trunc_ener_level)
        self._charges = np.array(charges) if np.any(charges) else np.array([0] * self._numQubits)
        self._cal_mode = cal_mode
        self._Nlevel = np.array(trunc_ener_level) if np.any(trunc_ener_level) else [10] * self._numQubits

        self._max_spectrum_inputs = (
            np.array(self.Ec, copy=True),
            np.array(self.El, copy=True),
            np.array(self.Ejmax, copy=True),
        )
        self._max_spectrum_cache = None

        if not hasattr(self, '_flux'):
            self._flux = np.array(fluxes) if np.any(fluxes) else np.zeros_like(Ej)
        self._update_Ej()
        if self._restore_cached_exact_solve_template_if_available():
            pass
        else:
            self._hamiltonian = self._generate_hamiltonian(self.Ec, self.El, self.Ej)
            super().__init__(self._hamiltonian, *args, **kwargs)
            self._store_current_exact_solve_template()

    def _materialize_max_spectrum_cache(self) -> tuple[np.ndarray, list[Qobj]]:
        """Build the max-bias spectrum lazily without disturbing the active solver state."""
        if self._max_spectrum_cache is None:
            ec, el, ejmax = self._max_spectrum_inputs
            hamil_max = self._generate_hamiltonian(ec, el, ejmax, transient=True)
            hamilmax = HamiltonianEvo(hamil_max)
            self._max_spectrum_cache = (
                hamilmax.get_energylevel(),
                hamilmax.get_eigenstate(),
            )

        return self._max_spectrum_cache

    @property
    def E_max(self) -> np.ndarray:
        """Sweet-spot energy levels [2*pi*GHz], materialized only when requested."""
        return self._materialize_max_spectrum_cache()[0]

    @property
    def state_max(self) -> list[Qobj]:
        """Sweet-spot eigenstates, materialized only when requested."""
        return self._materialize_max_spectrum_cache()[1]

    def _update_Ej(self):
        """Update Ej based on current flux and junction ratio."""
        flux_to_use = np.asarray(self._flux)
        ratio_to_use = np.asarray(self._junc_ratio)
        self.Ej = self._Ejphi(self.Ejmax, flux_to_use, ratio_to_use)

    def _Ejphi(self, Ej0: float, flux: float, ratio: float) -> float:
        return Ej0 * np.abs(np.cos(pi * flux)) * np.sqrt(
            1 + ((ratio - 1) * np.tan(pi * flux) / (ratio + 1)) ** 2
        )

    def _get_hamiltonian_operator_cache_key(self) -> tuple[tuple[int, ...], tuple[float, ...], int]:
        """Return the cache key for reusable Hamiltonian operator scaffolding."""
        nlevel_key = tuple(int(level) for level in np.asarray(self._Nlevel).reshape(-1).tolist())
        charges_key = tuple(float(charge) for charge in np.asarray(self._charges).reshape(-1).tolist())
        return nlevel_key, charges_key, int(self._numQubits)

    @staticmethod
    @lru_cache(maxsize=64)
    def _build_cached_hamiltonian_operators(
        nlevel_key: tuple[int, ...],
        charges_key: tuple[float, ...],
        num_qubits: int,
    ) -> tuple[tuple[Qobj, ...], ...]:
        """Construct reusable operator tensors once per truncation/charge configuration."""
        destroyors = []
        charge_op = []
        N_level = np.asarray(nlevel_key) + 8
        for ii in range(num_qubits):
            opstr1 = [destroy(N_level[jj]) if ii == jj else qeye(N_level[jj]) for jj in range(num_qubits)]
            opstr2 = [
                charges_key[ii] * qeye(N_level[jj]) if ii == jj else qeye(N_level[jj])
                for jj in range(num_qubits)
            ]
            destroyors.append(tensor(*opstr1))
            charge_op.append(tensor(*opstr2))

        ns_norm = [(-1j) * (des - des.dag()) / 2 for des in destroyors]
        phis_norm = [(des + des.dag()) for des in destroyors]
        number_ops = [des.dag() * des for des in destroyors]
        phis_norm_power_terms = [QubitBase._build_even_operator_powers(phi) for phi in phis_norm]

        return (
            tuple(destroyors),
            tuple(ns_norm),
            tuple(phis_norm),
            tuple(charge_op),
            tuple(number_ops),
            tuple(phis_norm_power_terms),
        )

    def _build_hamiltonian_operators(self) -> tuple[tuple[Qobj, ...], ...]:
        """Return the reusable operator tensors for the current truncation and charges."""
        return self._build_cached_hamiltonian_operators(*self._get_hamiltonian_operator_cache_key())

    def _hamiltonianOperator(self) -> list[object]:
        """Reuse Hamiltonian operator scaffolding until truncation or charge settings change."""
        cache_key = self._get_hamiltonian_operator_cache_key()
        cached_key = getattr(self, '_hamiltonian_operator_cache_key', None)
        cached_operators = getattr(self, '_hamiltonian_operator_cache', None)

        if cached_key != cache_key or cached_operators is None:
            cached_operators = self._build_hamiltonian_operators()
            self._hamiltonian_operator_cache_key = cache_key
            self._hamiltonian_operator_cache = cached_operators

        return cached_operators

    @staticmethod
    def _build_even_operator_powers(operator: Qobj) -> tuple[Qobj, Qobj, Qobj, Qobj]:
        """Build reusable even-power terms for a phase-like operator."""
        operator_squared = operator * operator
        operator_fourth = operator_squared * operator_squared
        operator_sixth = operator_fourth * operator_squared
        operator_eighth = operator_fourth * operator_fourth
        return operator_squared, operator_fourth, operator_sixth, operator_eighth

    @staticmethod
    def _scale_even_operator_powers(
        scale: float,
        operator_powers: tuple[Qobj, Qobj, Qobj, Qobj],
    ) -> tuple[Qobj, Qobj, Qobj, Qobj]:
        """Scale cached even-power scaffolding for the current phase-operator prefactor."""
        scale_squared = scale * scale
        scale_fourth = scale_squared * scale_squared
        scale_sixth = scale_fourth * scale_squared
        scale_eighth = scale_fourth * scale_fourth
        operator_squared, operator_fourth, operator_sixth, operator_eighth = operator_powers
        return (
            scale_squared * operator_squared,
            scale_fourth * operator_fourth,
            scale_sixth * operator_sixth,
            scale_eighth * operator_eighth,
        )

    @staticmethod
    def _get_pair_indices(num_qubits: int) -> tuple[tuple[int, int], ...]:
        """Return the unordered qubit-pair indices used by the Hamiltonian assembly."""
        return tuple(
            (ii, jj)
            for ii in range(num_qubits)
            for jj in range(ii + 1, num_qubits)
        )

    @staticmethod
    def _get_phi_scale_cache_key(phi_scales: list[float]) -> tuple[float, ...]:
        """Return the exact-input cache key for scaled phase-operator reuse."""
        return tuple(float(scale) for scale in phi_scales)

    @staticmethod
    def _get_ns_scale_cache_key(ns_scales: list[float]) -> tuple[float, ...]:
        """Return the exact-input cache key for scaled number-operator reuse."""
        return tuple(float(scale) for scale in ns_scales)

    @staticmethod
    @lru_cache(maxsize=256)
    def _build_cached_scaled_phase_terms(
        nlevel_key: tuple[int, ...],
        charges_key: tuple[float, ...],
        num_qubits: int,
        phi_scale_key: tuple[float, ...],
    ) -> tuple[tuple[Qobj, ...], tuple[tuple[Qobj, Qobj, Qobj, Qobj], ...]]:
        """Build scaled phase operators and even-power terms once per exact scale tuple."""
        _, _, phis_norm, _, _, phis_norm_power_terms = QubitBase._build_cached_hamiltonian_operators(
            nlevel_key,
            charges_key,
            num_qubits,
        )
        phis_op = tuple(
            phi_scale_key[ii] * phis_norm[ii]
            for ii in range(num_qubits)
        )
        phi_power_terms = tuple(
            QubitBase._scale_even_operator_powers(phi_scale_key[ii], phis_norm_power_terms[ii])
            for ii in range(num_qubits)
        )
        return phis_op, phi_power_terms

    @staticmethod
    @lru_cache(maxsize=256)
    def _build_cached_pair_power_terms(
        nlevel_key: tuple[int, ...],
        charges_key: tuple[float, ...],
        num_qubits: int,
        phi_scale_key: tuple[float, ...],
    ) -> tuple[tuple[tuple[int, int], tuple[Qobj, Qobj, Qobj, Qobj]], ...]:
        """Build transient pair-power scaffolding once per exact scale tuple."""
        phis_op, _ = QubitBase._build_cached_scaled_phase_terms(
            nlevel_key,
            charges_key,
            num_qubits,
            phi_scale_key,
        )
        return tuple(
            (
                (ii, jj),
                QubitBase._build_even_operator_powers(phis_op[ii] - phis_op[jj]),
            )
            for ii, jj in QubitBase._get_pair_indices(num_qubits)
        )

    @staticmethod
    @lru_cache(maxsize=256)
    def _build_cached_truncated_operator_views(
        nlevel_key: tuple[int, ...],
        charges_key: tuple[float, ...],
        num_qubits: int,
        phi_scale_key: tuple[float, ...],
        ns_scale_key: tuple[float, ...],
    ) -> tuple[tuple[Qobj, ...], tuple[Qobj, ...], tuple[Qobj, ...]]:
        """Build constructor-side truncated operator views once per exact input scales."""
        destroyors, ns_norm, _, charge_op, _, _ = QubitBase._build_cached_hamiltonian_operators(
            nlevel_key,
            charges_key,
            num_qubits,
        )
        phis_op, _ = QubitBase._build_cached_scaled_phase_terms(
            nlevel_key,
            charges_key,
            num_qubits,
            phi_scale_key,
        )
        ns_op = tuple(
            ns_scale_key[ii] * ns_norm[ii] - charge_op[ii]
            for ii in range(num_qubits)
        )
        trunc_levels = list(nlevel_key)
        return (
            tuple(truncate_hilbert_space(a, trunc_levels) for a in destroyors),
            tuple(truncate_hilbert_space(n, trunc_levels) for n in ns_op),
            tuple(truncate_hilbert_space(phi, trunc_levels) for phi in phis_op),
        )

    def _generate_hamiltonian(
        self,
        Ec: np.ndarray,
        El: np.ndarray,
        Ej: np.ndarray,
        *,
        transient: bool = False,
    ) -> Qobj:
        if self._cal_mode == 'Eigen':
            (
                destroyors,
                ns_norm,
                phis_norm,
                charge_op,
                number_ops,
                phis_norm_power_terms,
            ) = self._hamiltonianOperator()
            ns_scales = [
                ((Ej[ii, ii] + El[ii, ii]) / 2 / Ec[ii, ii]) ** (1 / 4)
                for ii in range(self._numQubits)
            ]
            ns_op = [
                ns_scales[ii] * ns_norm[ii] - charge_op[ii]
                for ii in range(self._numQubits)
            ]
            phi_scales = [
                (2 * Ec[ii, ii] / (Ej[ii, ii] + El[ii, ii])) ** (1 / 4)
                for ii in range(self._numQubits)
            ]
            operator_cache_key = self._get_hamiltonian_operator_cache_key()
            phi_scale_key = self._get_phi_scale_cache_key(phi_scales)
            ns_scale_key = self._get_ns_scale_cache_key(ns_scales)
            phis_op, phi_power_terms = self._build_cached_scaled_phase_terms(
                *operator_cache_key,
                phi_scale_key,
            )
            # Off-diagonal terms only use even powers, so (phi_i - phi_j) and (phi_j - phi_i)
            # can share the same cached matrices even when the coefficient matrices are asymmetric.
            pair_indices = self._get_pair_indices(self._numQubits)
            pair_power_terms = dict(
                self._build_cached_pair_power_terms(
                    *operator_cache_key,
                    phi_scale_key,
                )
            )
            factorial_6 = math.factorial(6)
            factorial_8 = math.factorial(8)
            hamil_c_terms = []
            hamil_high_pair_terms = []

            for ii, jj in pair_indices:
                ec_pair = Ec[ii, jj] + Ec[jj, ii]
                el_pair = El[ii, jj] + El[jj, ii]
                ej_pair = Ej[ii, jj] + Ej[jj, ii]
                if ec_pair == 0 and el_pair == 0 and ej_pair == 0:
                    continue

                delta_squared, delta_fourth, delta_sixth, delta_eighth = pair_power_terms[(ii, jj)]
                hamil_c_terms.append(
                    4 * ec_pair * ns_op[ii] * ns_op[jj]
                    + el_pair * phis_op[ii] * phis_op[jj] / 2
                    - ej_pair * (-delta_squared / 2 + delta_fourth / 24)
                )
                hamil_high_pair_terms.append(
                    ej_pair * (delta_sixth / factorial_6 - delta_eighth / factorial_8)
                )

            hamil_0 = sum(
                [
                    np.sqrt(8 * Ec[ii, ii] * (El[ii, ii] + Ej[ii, ii])) * number_ops[ii]
                    - (Ej[ii, ii] * phi_power_terms[ii][1]) / 24
                    for ii in range(self._numQubits)
                ]
            )
            hamil_c = sum(hamil_c_terms)
            hamil_high = sum(
                [
                    Ej[ii, ii] * phi_power_terms[ii][2] / factorial_6
                    - Ej[ii, ii] * phi_power_terms[ii][3] / factorial_8
                    for ii in range(self._numQubits)
                ]
            )
            hamil_high += sum(hamil_high_pair_terms)

            hamil_all = hamil_0 + hamil_c + hamil_high

            hamil_all = truncate_hilbert_space(hamil_all, self._Nlevel)
            if not transient:
                self.eigenHamiltonian = hamil_0
                self.couplingHamiltonian = hamil_c
                self.highorderHamiltonian = hamil_high
                cached_destroyors, cached_numbers, cached_phases = self._build_cached_truncated_operator_views(
                    *operator_cache_key,
                    phi_scale_key,
                    ns_scale_key,
                )
                self.n_operators = list(cached_numbers)
                self.phi_operators = list(cached_phases)
                self.destroyors = list(cached_destroyors)

        elif self._cal_mode == 'Charge':
            nlist = np.arange(-self._Nlevel[0], self._Nlevel[0] + 1, 1) - self._charges[0]
            hamil_all = 4 * Ec[0, 0] * np.diag(nlist**2) - Ej[0, 0] * (
                np.diag([1] * (len(nlist) - 1), k=1) + np.diag([1] * (len(nlist) - 1), k=-1)
            ) / 2
            hamil_all = Qobj(hamil_all)
            hamil_all = truncate_hilbert_space(hamil_all, self._Nlevel)

        else:
            raise ValueError(f'cal_mode {self._cal_mode} not supported!')

        return hamil_all

    @abstractmethod
    def _recalculate_hamiltonian(self):
        """Recalculate Hamiltonian after parameter changes."""
        pass

    def _resolve_change_parameter_updates(self, updates: dict) -> list[tuple[str, object, str]]:
        """Validate change_para() inputs before mutating state or recomputing."""
        resolved_updates = []
        for param_name, value in updates.items():
            if param_name == 'flux':
                resolved_updates.append(('_flux', self._normalize_flux_input(value), param_name))
                continue

            attr_name = self._PARAM_ATTR_MAP.get(param_name, param_name)
            if hasattr(self, attr_name):
                target_attr = attr_name
            elif hasattr(self, param_name):
                target_attr = param_name
            else:
                raise ValueError(f"Unknown parameter name: {param_name}")

            resolved_updates.append((target_attr, value, param_name))

        return resolved_updates

    def change_para(self, **updates):
        """Update parameters through explicit keyword arguments and recompute Hamiltonian."""
        if not updates:
            print('Nothing to Change!')
            return

        resolved_updates = self._resolve_change_parameter_updates(updates)
        self._last_changed_params = {param_name for _, _, param_name in resolved_updates}

        for target_attr, value, _ in resolved_updates:
            setattr(self, target_attr, value)

        self._recalculate_hamiltonian()

    def get_energy_matrices(self, var_name: Union[str, None] = None) -> Union[dict[str, np.ndarray], np.ndarray]:
        """Get the stored energy matrices through the preferred public accessor."""
        matrices = {'Ec': self.Ec, 'El': self.El, 'Ej_max': self.Ejmax, 'Ej': self.Ej}
        if var_name is None:
            return matrices
        try:
            return matrices[var_name]
        except KeyError as exc:
            raise AttributeError(f"{self.__class__.__name__} has no energy matrix '{var_name}'.") from exc

    def cal_spectroscopy(
        self,
        phi_flux: np.ndarray,
        index: int = 0,
        mode: str = 'brief',
        flux_precision: float = 1e-2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the frequency and sensitivity of the qubit."""
        Ejmax = self.Ejmax[index][index]
        Ec = self.Ec[index][index]
        flux_ori = self._flux[index][index]

        phi_flux = phi_flux + flux_ori
        x = np.pi * phi_flux
        Ej_phi = Ejmax * np.abs(np.cos(x))

        if mode == 'brief':
            omega = np.sqrt(8 * Ej_phi * Ec) - Ec
            sensitivity = -pi * np.sqrt(2 * Ec * Ej_phi) * np.tan(x)
        elif mode == 'full':
            original_flux = copy(self._flux)
            omega = []
            sensitivity = []
            for flux in phi_flux:
                flux_up = flux + flux_precision
                self._flux[index][index] = flux_up
                self.change_para(flux=self._flux)
                state = [1 if i == index else 0 for i in range(self._numQubits)]
                state_index = self.find_state(state)
                if isinstance(state_index, int):
                    omega_up = self.get_energylevel(state_index)
                else:
                    omega_up = self.get_energylevel(state_index[0])

                flux_down = flux - flux_precision
                self._flux[index][index] = flux_down
                self.change_para(flux=self._flux)
                state = [1 if i == index else 0 for i in range(self._numQubits)]
                state_index = self.find_state(state)
                if isinstance(state_index, int):
                    omega_down = self.get_energylevel(state_index)
                else:
                    omega_down = self.get_energylevel(state_index[0])

                omega.append((omega_up + omega_down) / 2)
                sensitivity.append((omega_up - omega_down) / (2 * flux_precision))

            self.change_para(flux=original_flux)
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'brief' or 'full'.")

        return (np.array(omega), np.array(sensitivity))


class ParameterizedQubit(QubitBase):
    '''
    Design for the generation of universal superconducting qubit Hamiltonian.
    '''

    # Extend parameter map for ParameterizedQubit-specific parameters
    _PARAM_ATTR_MAP = {
        **QubitBase._PARAM_ATTR_MAP,
        'capac': '_ParameterizedQubit__capac',
        'resis': '_ParameterizedQubit__resis',
        'induc': '_ParameterizedQubit__induc',
        'struct': '_ParameterizedQubit__struct',
    }
    _ELEMENT_PARAMS = {'capac', 'resis', 'induc', 'struct'}

    @classmethod
    def _get_cached_ematrix_template(
        cls,
        cache_key: tuple[object, ...],
    ) -> Optional[dict[str, object]]:
        """Return one cached exact-input element-matrix template when available."""
        template = _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE.get(cache_key)
        if template is not None:
            _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE.move_to_end(cache_key)
        return template

    @classmethod
    def _store_cached_ematrix_template(
        cls,
        cache_key: tuple[object, ...],
        template: dict[str, object],
    ) -> None:
        """Store one exact-input element-matrix template with LRU-style eviction."""
        _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE[cache_key] = template
        _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE.move_to_end(cache_key)
        if len(_PARAMETERIZED_EMATRIX_TEMPLATE_CACHE) > _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE_MAXSIZE:
            _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE.popitem(last=False)

    @classmethod
    def _clear_ematrix_template_cache(cls) -> None:
        """Clear the exact-input element-matrix template cache used by repeated builds."""
        _PARAMETERIZED_EMATRIX_TEMPLATE_CACHE.clear()

    def __init__(
        self,
        capacitances: list[list],
        junctions_resistance: list[list],
        inductances: list[list] = [None],
        fluxes: Optional[list[list]] = [None],
        charges: Optional[list[float]] = [None],
        trunc_ener_level: list[float] = [None],
        junc_ratio: Optional[list[list]] = [None],
        structure_index: Optional[list] = [None],
        cal_mode: Optional[str] = 'Eigen',
        *args,
        **kwargs,
    ):
        '''
        capacitances: Capacitance matrix
        junctions_resistance: Junction resistance matrix
        inductances: Inductance matrix
        fluxes: Fluxes for each squid
        trunc_ener_level: The num of highest energy levels for each qubit
        junc_ratio: The junc_resistance ratio between the two junctions of squid. ratio > 1
        structure_index: Indicate the structure of whole circuit, 1 stands for grounded transmon, 2 stands for floating transmon
        cal_mode: The method of generation of Hamiltonian. 'Eigen' stands for Eigen-representation Method. 'Charge' stands for Charge-representation Method.

        ATTENTION!:
        1, cal_mode=='Charge' only apply for single qubit without inductance, and it needs higher trunc_level.

        '''
        self.__capac = np.array(capacitances)
        self.__induc = np.array(inductances) if np.any(inductances) else np.ones_like(capacitances) * 1e20
        self.__resis = np.array(junctions_resistance)
        # Only set flux if not already set by subclass
        if not hasattr(self, '_flux'):
            self._flux = np.array(fluxes) if np.any(fluxes) else np.zeros_like(self.__capac)
        self.__nodes = self.__capac.shape[0]
        self._junc_ratio = junc_ratio if np.any(junc_ratio) else np.ones_like(self._flux)
        self._transformed_vars_dirty = True
        if np.any(structure_index):
            self._numQubits = len(structure_index)
            self.__struct = structure_index
        else:
            self._numQubits = self.__nodes
            self.__struct = [1] * self.__nodes
        self._Nlevel = np.array(trunc_ener_level)
        self._charges = np.array(charges) if np.any(charges) else np.array([0] * self._numQubits)

        self.__Ec, self.__El, self.__Ej0 = self._generate_Ematrix()
        super().__init__(
            Ec=self.__Ec,
            El=self.__El,
            Ej=self.__Ej0,
            fluxes=self._flux_transformed,
            charges=self._charges,
            junc_ratio=self._junc_ratio,
            trunc_ener_level=self._Nlevel,
            cal_mode=cal_mode,
            *args,
            **kwargs,
        )

    def _get_ematrix_template_cache_key(self) -> tuple[object, ...]:
        """Return the exact-input cache key for reusable circuit-element matrix templates."""
        return (
            tuple(int(item) for item in np.asarray(self.__struct).reshape(-1).tolist()),
            self._get_array_cache_key(self.__capac),
            self._get_array_cache_key(self.__induc),
            self._get_array_cache_key(self.__resis),
        )

    def _capture_ematrix_template(self) -> dict[str, object]:
        """Capture the current circuit-element matrix bundle for exact-input reuse."""
        return {
            's_matrix': self._clone_ndarray(self.SMatrix),
            'retain_nodes': tuple(int(node) for node in self.SMatrix_retainNodes),
            'maxwell_matrix': self._clone_ndarray_mapping(self.Maxwellmat),
            'ec_matrix': self._clone_ndarray(self.__Ec),
            'el_matrix': self._clone_ndarray(self.__El),
            'ej0_matrix': self._clone_ndarray(self.__Ej0),
        }

    def _restore_ematrix_template(self, template: dict[str, object]) -> None:
        """Restore one cached circuit-element matrix bundle onto this instance."""
        self.SMatrix = self._clone_ndarray(template['s_matrix'])
        self.SMatrix_retainNodes = list(template['retain_nodes'])
        self.Maxwellmat = self._clone_ndarray_mapping(template['maxwell_matrix'])
        self.__Ec = self._clone_ndarray(template['ec_matrix'])
        self.__El = self._clone_ndarray(template['el_matrix'])
        self.__Ej0 = self._clone_ndarray(template['ej0_matrix'])

    def _generate_Ematrix(self):
        cache_key = self._get_ematrix_template_cache_key()
        cached_template = self._get_cached_ematrix_template(cache_key)
        if cached_template is None:
            self.SMatrix, retainNodes = assemble_s_matrix_and_retain_nodes(self.__struct)
            self.SMatrix_retainNodes = retainNodes
            self.Maxwellmat, self.__Ec, self.__El, self.__Ej0 = convert_elements_to_energy_matrices(
                self.__capac,
                self.__induc,
                self.__resis,
                self.SMatrix,
                retainNodes,
                self.__struct,
                convert_resistance_to_ej0,
            )
            self._store_cached_ematrix_template(cache_key, self._capture_ematrix_template())
        else:
            self._restore_ematrix_template(cached_template)
            retainNodes = self.SMatrix_retainNodes

        self._flux_transformed = project_transformed_flux(self._flux, self.__struct, retainNodes)
        self._junc_ratio_transformed = project_transformed_junction_ratio(
            self._junc_ratio,
            self.__struct,
            retainNodes,
        )
        self._transformed_vars_dirty = False

        return self.__Ec, self.__El, self.__Ej0

    def get_element_matrices(self, var_name: Union[str, None] = None) -> Union[dict[str, np.ndarray], np.ndarray]:
        """Get the stored circuit-element matrices through the preferred public accessor."""
        matrices = {
            'capac': self._ParameterizedQubit__capac,
            'induc': self._ParameterizedQubit__induc,
            'resis': self._ParameterizedQubit__resis,
            'flux': self._flux,
        }
        if var_name is None:
            return matrices
        try:
            return matrices[var_name]
        except KeyError as exc:
            raise AttributeError(f"{self.__class__.__name__} has no element matrix '{var_name}'.") from exc

    def _update_transformed_vars(self):
        """
        Update _flux_transformed and _junc_ratio_transformed based on current _flux and _junc_ratio.

        This method should be called after change_para updates _flux or _junc_ratio,
        before _update_Ej recalculates Ej.
        """
        if (
            not getattr(self, '_transformed_vars_dirty', True)
            and hasattr(self, '_flux_transformed')
            and hasattr(self, '_junc_ratio_transformed')
        ):
            return

        self._flux_transformed = project_transformed_flux(
            self._flux,
            self.__struct,
            self.SMatrix_retainNodes,
        )
        self._junc_ratio_transformed = project_transformed_junction_ratio(
            self._junc_ratio,
            self.__struct,
            self.SMatrix_retainNodes,
        )
        self._transformed_vars_dirty = False

    def _update_Ej(self):
        """
        Update Ej based on current flux and junction ratio.

        For ParameterizedQubit and subclasses, this updates the transformed variables
        before calculating Ej to ensure correct shape matching.
        """
        # Update transformed variables first
        self._update_transformed_vars()

        # Use transformed variables for Ej calculation
        flux_to_use = np.asarray(self._flux_transformed)
        ratio_to_use = np.asarray(self._junc_ratio_transformed)

        self.Ej = self._Ejphi(self.Ejmax, flux_to_use, ratio_to_use)

    def _normalize_element_inputs(self):
        """Normalize circuit-element inputs before rebuilding derived matrices."""
        self._ParameterizedQubit__capac = np.array(self._ParameterizedQubit__capac)
        self._ParameterizedQubit__resis = np.array(self._ParameterizedQubit__resis)
        self._ParameterizedQubit__induc = np.array(self._ParameterizedQubit__induc)
        self._ParameterizedQubit__struct = list(self._ParameterizedQubit__struct)
        self._ParameterizedQubit__nodes = self._ParameterizedQubit__capac.shape[0]
        self._numQubits = len(self._ParameterizedQubit__struct)

    def _recalculate_hamiltonian(self):
        """
        Recalculate Hamiltonian after parameter changes.

        Handles flux transformation and Ej recalculation for ParameterizedQubit.
        """
        changed_params = getattr(self, '_last_changed_params', set())
        if changed_params & (self._ELEMENT_PARAMS | {'flux', 'junc_ratio'}):
            self._transformed_vars_dirty = True

        if changed_params & self._ELEMENT_PARAMS:
            self._normalize_element_inputs()
            self.__Ec, self.__El, self.__Ej0 = self._generate_Ematrix()
            self.Ec = self.__Ec
            self.El = self.__El
            self.Ejmax = self.__Ej0

        # Update transformed variables
        self._update_transformed_vars()

        # Recalculate Ej using transformed variables
        junc_ratio_to_use = (
            self._junc_ratio_transformed if hasattr(self, '_junc_ratio_transformed') else self._junc_ratio
        )
        flux_to_use = self._flux_transformed
        ejmax_to_use = self._ParameterizedQubit__Ej0 if hasattr(self, '_ParameterizedQubit__Ej0') else self.Ejmax

        self.Ej = self._Ejphi(ejmax_to_use, flux_to_use, junc_ratio_to_use)

        if self._restore_cached_exact_solve_template_if_available():
            self._last_changed_params = set()
            return

        # Regenerate Hamiltonian
        self._hamiltonian = self._generate_hamiltonian(self.Ec, self.El, self.Ej)
        self.change_hamiltonian(self._hamiltonian)
        self._store_current_exact_solve_template()
        self._last_changed_params = set()


class AbstractQubit(QubitBase):
    """Abstract qubit class for generating qubits from frequency information."""

    @staticmethod
    def _clone_qobj(obj: Qobj) -> Qobj:
        """Clone a qutip object for per-instance isolation across cached templates."""
        if hasattr(obj, 'copy'):
            return obj.copy()
        return copy(obj)

    @classmethod
    def _clone_qobj_list(cls, objects: Optional[list[Qobj]]) -> Optional[list[Qobj]]:
        if objects is None:
            return None
        return [cls._clone_qobj(obj) for obj in objects]

    @staticmethod
    @lru_cache(maxsize=128)
    def _solve_transmon_ec_ej(target_f01: float, target_anh: float) -> tuple[float, float, bool, int, float, float]:
        """Cache the deterministic Transmon parameter inversion used by constructor-style callers."""
        Ec_guess = -target_anh
        Ej_guess = (target_f01 + Ec_guess) ** 2 / (8 * Ec_guess)

        initial_guess = [Ec_guess, Ej_guess]

        def cost_function(params):
            Ec_curr, Ej_curr = params

            if Ec_curr <= 0 or Ej_curr <= 0:
                return 1e9

            f01_calc, anh_calc = AbstractQubit.get_transmon_spectrum_fast(Ec_curr, Ej_curr)

            weight_anh = 10.0
            return (f01_calc - target_f01) ** 2 + weight_anh * (anh_calc - target_anh) ** 2

        result = minimize(cost_function, initial_guess, method='Nelder-Mead', tol=1e-8)
        Ec_final, Ej_final = (float(value) for value in result.x)
        f01_final, anh_final = AbstractQubit.get_transmon_spectrum_fast(Ec_final, Ej_final)
        return (
            Ec_final,
            Ej_final,
            bool(result.success),
            int(result.nit),
            float(f01_final),
            float(anh_final),
        )

    @classmethod
    @lru_cache(maxsize=128)
    def _build_cached_transmon_template(
        cls,
        target_f01: float,
        target_anh: float,
        target_f01_max: float,
        energy_trunc_level: int,
    ) -> dict[str, object]:
        """Build and cache a solved single-qubit Transmon template for repeated constructor callers."""
        if target_anh > 0:
            target_anh = -target_anh

        Ec_final, Ej_final, _, _, _, _ = cls._solve_transmon_ec_ej(target_f01_max, target_anh)
        flux = np.arccos(((target_f01 - target_anh) / (target_f01_max - target_anh)) ** 2) / pi
        template_qubit = QubitBase(
            Ec=np.array([[Ec_final * 2 * pi]]),
            El=np.array([[0.0]]),
            Ej=np.array([[Ej_final * 2 * pi]]),
            fluxes=[[flux]],
            trunc_ener_level=[energy_trunc_level],
        )
        return {
            'Ec': np.array(template_qubit.Ec, copy=True),
            'El': np.array(template_qubit.El, copy=True),
            'Ejmax': np.array(template_qubit.Ejmax, copy=True),
            'Ej': np.array(template_qubit.Ej, copy=True),
            'flux': float(flux),
            'flux_matrix': np.array(template_qubit._flux, copy=True),
            'junc_ratio': np.array(template_qubit._junc_ratio, copy=True),
            'charges': np.array(template_qubit._charges, copy=True),
            'cal_mode': template_qubit._cal_mode,
            'hamiltonian': cls._clone_qobj(template_qubit._Hamiltonian),
            'energylevels': np.array(template_qubit._energylevels, copy=True),
            'eigenstates': tuple(cls._clone_qobj(state) for state in template_qubit._eigenstates),
            'destroyors': tuple(cls._clone_qobj(op) for op in template_qubit.destroyors),
            'number_operators': tuple(cls._clone_qobj(op) for op in template_qubit.n_operators),
            'phase_operators': tuple(cls._clone_qobj(op) for op in template_qubit.phi_operators),
        }

    def _restore_cached_transmon_template(self, template: dict[str, object]) -> None:
        """Restore a solved Transmon template without rerunning the constructor hot path."""
        self.Ec = np.array(template['Ec'], copy=True)
        self.El = np.array(template['El'], copy=True)
        self.Ejmax = np.array(template['Ejmax'], copy=True)
        self.Ej = np.array(template['Ej'], copy=True)
        self._flux = np.array(template['flux_matrix'], copy=True)
        self._junc_ratio = np.array(template['junc_ratio'], copy=True)
        self._charges = np.array(template['charges'], copy=True)
        self._cal_mode = template['cal_mode']
        self._hamiltonian = self._clone_qobj(template['hamiltonian'])
        self._Hamiltonian = self._hamiltonian
        self._energylevels = np.array(template['energylevels'], copy=True)
        self._eigenstates = self._clone_qobj_list(template['eigenstates'])
        self.destroyors = self._clone_qobj_list(template['destroyors'])
        self.n_operators = self._clone_qobj_list(template['number_operators'])
        self.phi_operators = self._clone_qobj_list(template['phase_operators'])
        self._numQubits = 1
        self._Nlevel = list(self._Hamiltonian.dims[0])
        self._qubits_num = len(self._Nlevel)
        self._max_spectrum_inputs = (
            np.array(self.Ec, copy=True),
            np.array(self.El, copy=True),
            np.array(self.Ejmax, copy=True),
        )
        self._max_spectrum_cache = None
        self._solver_result = self._build_solver_result(
            self._Hamiltonian,
            self._energylevels,
            self._eigenstates,
        )

    def __init__(
        self,
        frequency: float = 5e9,
        anharmonicity: float = -250e6,
        frequency_max: float = None,
        qubit_type: str = 'Transmon',
        energy_trunc_level: int = 12,
        is_print: bool = True,
        *args,
        **kwargs,
    ):
        """
        Abstract qubit class. Generating Qubit by frequency information.

        Args:
            frequency (float, optional): Current frequency of qubit. Defaults to 5e9.
            anharmonicity (float, optional): Anharmonicity of qubit. Defaults to -250e6.
            frequency_max (float, optional): Frequency in sweet spot, if None -> equal to frequency.
            qubit_type (str, optional): Type of qubit. Defaults to 'Transmon'. Supported ['Transmon']
            energy_trunc_level (Union[list, np.ndarray], optional): Energy truncated level. Defaults to [12].
        """
        self.qubit_f01 = frequency / 1e9
        self.qubit_anharm = anharmonicity / 1e9
        if frequency_max is None:
            frequency_max = frequency
        self.qubit_f01_max = frequency_max / 1e9
        self.qubit_type = qubit_type
        self.flux = np.arccos(((frequency - anharmonicity) / (frequency_max - anharmonicity)) ** 2) / pi
        self.energy_trunc_level = [energy_trunc_level]
        if qubit_type == 'Transmon':
            template = self._build_cached_transmon_template(
                self.qubit_f01,
                self.qubit_anharm,
                self.qubit_f01_max,
                int(energy_trunc_level),
            )
            self.flux = template['flux']
            self._restore_cached_transmon_template(template)
        else:
            self.cal_Emat_by_type(qubit_type)
            super().__init__(
                Ec=self.Ec,
                El=self.El,
                Ej=self.Ejmax,
                fluxes=[[self.flux]],
                trunc_ener_level=[energy_trunc_level],
                *args,
                **kwargs,
            )
        self.qubit_f01 = self.get_energylevel(1) / 2 / pi
        self.qubit_anharm = self.get_energylevel(2) / 2 / pi - 2 * self.qubit_f01
        if is_print:
            print(f'Qubit F01: {self.qubit_f01:.3f} GHz')
            print(f'Qubit Anharmonicity: {self.qubit_anharm*1e3:.1f} MHz')

    def cal_Emat_by_type(self, qubit_type: str = 'Transmon'):
        if qubit_type == 'Transmon':
            self.El = np.array([[0]])
            Ec, Ejmax = self.optimize_Ec_Ej(self.qubit_f01_max, self.qubit_anharm, is_print=False)
            self.Ec = np.array([[Ec * 2 * pi]])
            self.Ejmax = np.array([[Ejmax * 2 * pi]])
        else:
            raise ValueError(f'Qubit type {qubit_type} is not supported.')

        return self.Ec, self.El, self.Ejmax

    def calculate_sensitivity_at_detuning(
        self,
        delta_freq: float = 0,
        mode: str = 'brief',
        flux_precision: float = 1e-2,
    ) -> tuple[float, float]:
        """
        Calculate sensitivity at specific detuning.

        Args:
            delta_freq (float): detuning frequency [GHz].
            mode (str, optional): calculate mode. Defaults to 'brief'.

        Returns:
            tuple[float, float]: (sensitivity, f_current)
        """

        f_current = self.qubit_f01 - delta_freq

        min_freq = self.qubit_anharm
        if f_current <= min_freq:
            raise ValueError(f"Bias too large, frequency below physical limit! Current: {f_current:.2f}, Limit: {min_freq:.2f}")

        R = (f_current - self.qubit_anharm) / (self.qubit_f01_max - self.qubit_anharm)
        flux_detune = np.arccos(R**2) / pi

        if mode == 'brief':
            term_geometric = np.sqrt(1 - R**4) / R
            sensitivity = (np.pi / 2) * (self.qubit_f01 - self.qubit_anharm) * term_geometric
        elif mode == 'full':
            flux_up = flux_detune + flux_precision
            qubit = QubitBase(
                Ec=self.Ec,
                El=self.El,
                Ej=self.Ejmax,
                fluxes=[[flux_up]],
                trunc_ener_level=self.energy_trunc_level,
            )
            state = [1]
            state_index = qubit.find_state(state)
            if isinstance(state_index, int):
                omega_up = qubit.get_energylevel(state_index)
            else:
                omega_up = qubit.get_energylevel(state_index[0])

            flux_down = flux_detune - flux_precision
            qubit = QubitBase(
                Ec=self.Ec,
                El=self.El,
                Ej=self.Ejmax,
                fluxes=[[flux_down]],
                trunc_ener_level=self.energy_trunc_level,
            )
            state = [1]
            state_index = qubit.find_state(state)
            if isinstance(state_index, int):
                omega_down = qubit.get_energylevel(state_index)
            else:
                omega_down = qubit.get_energylevel(state_index[0])

            sensitivity = (omega_up - omega_down) / (4 * pi * flux_precision)
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'brief' or 'full'.")

        return sensitivity, f_current

    @staticmethod
    def get_transmon_spectrum_fast(Ec, Ej, n_cutoff=15):
        """
        Calculates the first few eigenenergies of a Transmon in the charge basis.

        Hamiltonian: H = 4*Ec*n^2 - Ej/2 * (|n><n+1| + |n+1><n|)
        This forms a tridiagonal matrix, allowing for extremely fast diagonalization.

        Parameters:
            Ec (float): Charging energy (GHz)
            Ej (float): Josephson energy (GHz)
            n_cutoff (int): Number of charge states to include (+/- n_cutoff).
                            15 is usually sufficient for low-energy spectrum.

        Returns:
            f01 (float): Frequency of the 0->1 transition (GHz)
            anh (float): Anharmonicity (f12 - f01) (GHz)
        """
        n_range = np.arange(-n_cutoff, n_cutoff + 1)

        diag = 4.0 * Ec * (n_range**2)
        off_diag = -0.5 * Ej * np.ones(len(n_range) - 1)

        evals = eigh_tridiagonal(
            diag,
            off_diag,
            select='i',
            select_range=(0, 2),
            eigvals_only=True,
        )

        f01 = evals[1] - evals[0]
        f12 = evals[2] - evals[1]
        anh = f12 - f01

        return f01, anh

    def optimize_Ec_Ej(self, target_f01, target_anh, is_print=True):
        """
        Iteratively finds the exact Ec and Ej that yield the target f01 and anharmonicity.
        Uses analytical formulas as the starting point (initial guess).
        """
        target_f01 = float(target_f01)
        target_anh = float(target_anh)
        if target_anh > 0:
            target_anh = -target_anh
            if is_print:
                print("Note: Converted target anharmonicity to negative value.")
        Ec_guess = -target_anh
        Ej_guess = (target_f01 + Ec_guess) ** 2 / (8 * Ec_guess)

        if is_print:
            print(f"--- Optimization Start ---")
            print(f"Target: f01 = {target_f01:.5f} GHz, alpha = {target_anh:.5f} GHz")
            print(f"Initial Guess (Analytical): Ec = {Ec_guess:.4f}, Ej = {Ej_guess:.4f}")
        Ec_final, Ej_final, success, iterations, f01_final, anh_final = self._solve_transmon_ec_ej(
            target_f01,
            target_anh,
        )

        if is_print:
            print(f"--- Optimization Result ---")
            print(f"Success: {success}")
            print(f"Iterations: {iterations}")
            print(f"Optimized Ec = {Ec_final:.6f} GHz")
            print(f"Optimized Ej = {Ej_final:.6f} GHz")
            print(f"Ej/Ec Ratio  = {Ej_final/Ec_final:.2f}")
            print(f"Final Freqs  : f01 = {f01_final:.6f}, alpha = {anh_final:.6f}")
            print(f"Error        : df = {(f01_final-target_f01):.2e}, dalpha = {(anh_final-target_anh):.2e}")
            print(f"---------------------------")

        return Ec_final, Ej_final
