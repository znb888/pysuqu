"""Base qubit classes extracted from the legacy qubit monolith."""

import math
from abc import abstractmethod
from copy import copy
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


class QubitBase(HamiltonianEvo):
    """Base class for all qubit implementations."""

    _PARAM_ATTR_MAP = {
        'flux': '_flux',
        'junc_ratio': '_junc_ratio',
        'Nlevel': '_Nlevel',
        'charges': '_charges',
        'cal_mode': '_cal_mode',
    }

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

        hamil_max = self._generate_hamiltonian(self.Ec, self.El, self.Ejmax)
        hamilmax = HamiltonianEvo(hamil_max)
        self.E_max = hamilmax.get_energylevel()
        self.state_max = hamilmax.get_eigenstate()

        if not hasattr(self, '_flux'):
            self._flux = np.array(fluxes) if np.any(fluxes) else np.zeros_like(Ej)
        self._update_Ej()
        self._hamiltonian = self._generate_hamiltonian(self.Ec, self.El, self.Ej)
        super().__init__(self._hamiltonian, *args, **kwargs)

    def _update_Ej(self):
        """Update Ej based on current flux and junction ratio."""
        flux_to_use = np.asarray(self._flux)
        ratio_to_use = np.asarray(self._junc_ratio)
        self.Ej = self._Ejphi(self.Ejmax, flux_to_use, ratio_to_use)

    def _Ejphi(self, Ej0: float, flux: float, ratio: float) -> float:
        return Ej0 * np.abs(np.cos(pi * flux)) * np.sqrt(
            1 + ((ratio - 1) * np.tan(pi * flux) / (ratio + 1)) ** 2
        )

    def _hamiltonianOperator(self) -> list[Qobj]:
        destroyors = []
        charge_op = []
        N_level = np.asarray(self._Nlevel) + 8
        for ii in range(self._numQubits):
            opstr1 = [destroy(N_level[jj]) if ii == jj else qeye(N_level[jj]) for jj in range(self._numQubits)]
            opstr2 = [
                self._charges[ii] * qeye(N_level[jj]) if ii == jj else qeye(N_level[jj])
                for jj in range(self._numQubits)
            ]
            destroyors.append(tensor(*opstr1))
            charge_op.append(tensor(*opstr2))

        ns_norm = [(-1j) * (des - des.dag()) / 2 for des in destroyors]
        phis_norm = [(des + des.dag()) for des in destroyors]

        return [destroyors, ns_norm, phis_norm, charge_op]

    def _generate_hamiltonian(
        self,
        Ec: np.ndarray,
        El: np.ndarray,
        Ej: np.ndarray,
    ) -> Qobj:
        if self._cal_mode == 'Eigen':
            destroyors, ns_norm, phis_norm, charge_op = self._hamiltonianOperator()
            ns_op = [
                ((Ej[ii, ii] + El[ii, ii]) / 2 / Ec[ii, ii]) ** (1 / 4) * ns_norm[ii] - charge_op[ii]
                for ii in range(self._numQubits)
            ]
            phis_op = [
                (2 * Ec[ii, ii] / (Ej[ii, ii] + El[ii, ii])) ** (1 / 4) * phis_norm[ii]
                for ii in range(self._numQubits)
            ]

            hamil_0 = sum(
                [
                    np.sqrt(8 * Ec[ii, ii] * (El[ii, ii] + Ej[ii, ii])) * destroyors[ii].dag() * destroyors[ii]
                    - (Ej[ii, ii] * phis_op[ii] ** 4) / 24
                    for ii in range(self._numQubits)
                ]
            )
            hamil_c = sum(
                [
                    4 * Ec[ii, jj] * ns_op[ii] * ns_op[jj]
                    + El[ii, jj] * phis_op[ii] * phis_op[jj] / 2
                    - Ej[ii, jj]
                    * (-(phis_op[ii] - phis_op[jj]) ** 2 / 2 + (phis_op[ii] - phis_op[jj]) ** 4 / 24)
                    for ii in range(self._numQubits)
                    for jj in range(self._numQubits)
                    if ii != jj
                ]
            )
            hamil_high = sum(
                [
                    Ej[ii, ii] * (phis_op[ii] ** 6 / math.factorial(6) - phis_op[ii] ** 8 / math.factorial(8))
                    for ii in range(self._numQubits)
                ]
            )
            hamil_high += sum(
                [
                    Ej[ii, jj]
                    * (
                        (phis_op[ii] - phis_op[jj]) ** 6 / math.factorial(6)
                        - (phis_op[ii] - phis_op[jj]) ** 8 / math.factorial(8)
                    )
                    for ii in range(self._numQubits)
                    for jj in range(self._numQubits)
                    if ii != jj
                ]
            )

            self.eigenHamiltonian = hamil_0
            self.couplingHamiltonian = hamil_c
            self.highorderHamiltonian = hamil_high
            hamil_all = hamil_0 + hamil_c + hamil_high

            hamil_all = truncate_hilbert_space(hamil_all, self._Nlevel)
            self.n_operators = [truncate_hilbert_space(n, self._Nlevel) for n in ns_op]
            self.phi_operators = [truncate_hilbert_space(phi, self._Nlevel) for phi in phis_op]
            self.destroyors = [truncate_hilbert_space(a, self._Nlevel) for a in destroyors]

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

    def _generate_Ematrix(self):
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

        self._flux_transformed = project_transformed_flux(self._flux, self.__struct, retainNodes)
        self._junc_ratio_transformed = project_transformed_junction_ratio(
            self._junc_ratio,
            self.__struct,
            retainNodes,
        )

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

        # Regenerate Hamiltonian
        self._hamiltonian = self._generate_hamiltonian(self.Ec, self.El, self.Ej)
        self.change_hamiltonian(self._hamiltonian)
        self._last_changed_params = set()


class AbstractQubit(QubitBase):
    """Abstract qubit class for generating qubits from frequency information."""

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
        if target_anh > 0:
            target_anh = -target_anh
            if is_print:
                print("Note: Converted target anharmonicity to negative value.")

        Ec_guess = -target_anh
        Ej_guess = (target_f01 + Ec_guess) ** 2 / (8 * Ec_guess)

        initial_guess = [Ec_guess, Ej_guess]

        if is_print:
            print(f"--- Optimization Start ---")
            print(f"Target: f01 = {target_f01:.5f} GHz, alpha = {target_anh:.5f} GHz")
            print(f"Initial Guess (Analytical): Ec = {Ec_guess:.4f}, Ej = {Ej_guess:.4f}")

        def cost_function(params):
            Ec_curr, Ej_curr = params

            if Ec_curr <= 0 or Ej_curr <= 0:
                return 1e9

            f01_calc, anh_calc = self.get_transmon_spectrum_fast(Ec_curr, Ej_curr)

            weight_anh = 10.0
            error = (f01_calc - target_f01) ** 2 + weight_anh * (anh_calc - target_anh) ** 2
            return error

        result = minimize(cost_function, initial_guess, method='Nelder-Mead', tol=1e-8)

        Ec_final, Ej_final = result.x

        f01_final, anh_final = self.get_transmon_spectrum_fast(Ec_final, Ej_final)

        if is_print:
            print(f"--- Optimization Result ---")
            print(f"Success: {result.success}")
            print(f"Iterations: {result.nit}")
            print(f"Optimized Ec = {Ec_final:.6f} GHz")
            print(f"Optimized Ej = {Ej_final:.6f} GHz")
            print(f"Ej/Ec Ratio  = {Ej_final/Ec_final:.2f}")
            print(f"Final Freqs  : f01 = {f01_final:.6f}, alpha = {anh_final:.6f}")
            print(f"Error        : df = {(f01_final-target_f01):.2e}, dalpha = {(anh_final-target_anh):.2e}")
            print(f"---------------------------")

        return Ec_final, Ej_final
