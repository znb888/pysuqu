'''
Lib for multiqubit simulation.

Author: Naibin Zhou
USTC
Since 2023-12-05
'''
# import
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
# local lib
from ..funclib import *
from ..funclib.qutiplib import cal_product_state_list
from .circuit import project_transformed_flux, project_transformed_junction_ratio
from .analysis import (
    analyze_multi_qubit_coupler_sensitivity,
    calculate_multi_qubit_coupler_self_sensitivity,
    calculate_multi_qubit_sensitivity_analytical,
    calculate_multi_qubit_sensitivity_numerical,
    get_multi_qubit_frequency_at_coupler_flux,
    plot_multi_qubit_sensitivity_curve,
)
from .base import ParameterizedQubit, e, hbar, pi
from .compatibility import (
    _fgf1v1_qubit_dephasing_by_coupler_thermal,
    _fgfgg1v1v3_coupling_init,
    _grounded_transmon_list_init,
)
from .solver import HamiltonianEvo
from .single import GroundedTransmon, RLINE

RNAN = 1e20
_QCRFGR_PROBE_FREQUENCY_CACHE_MAXSIZE = 128
_QCRFGR_PROBE_FREQUENCY_CACHE = OrderedDict()
_FGF1V1_BASIC_METRIC_CACHE_MAXSIZE = 128
_FGF1V1_BASIC_METRIC_CACHE = OrderedDict()
_FGF1V1_METRIC_STATE_INDEX_CACHE_MAXSIZE = 128
_FGF1V1_METRIC_STATE_INDEX_CACHE = OrderedDict()
_FGF1V1_INSTANCE_BASIC_METRIC_CACHE_MAXSIZE = 16
_QCRFGR_METRIC_STATE_LABELS = ((1, 0), (0, 1), (2, 0))
_QCRFGR_COUPLER_OVERLAP_STATE_LABELS = ((1, 0), (0, 1))
_FGF1V1_METRIC_STATE_LABELS = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 0, 2), (2, 0, 0))
_FGF1V1_QC_OVERLAP_STATE_LABELS = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
_FGF1V1_QQ_OVERLAP_STATE_LABELS = ((0, 0, 1), (1, 0, 0))


@dataclass(frozen=True)
class FGF1V1BasicMetricBundle:
    """Typed derived-metric payload reused across FGF1V1 replay cache layers."""

    qubit1_f01: float
    qubit2_f01: float
    qubit_f01: float
    coupler_f01: float
    qubit1_anharm: float
    qubit2_anharm: float
    qubit_anharm: float
    qr_g: float
    qq_g: float
    qc_g: float
    qq_geff: float

    @classmethod
    def capture(cls, model) -> 'FGF1V1BasicMetricBundle':
        """Capture the current FGF1V1 derived metrics from one live model instance."""
        return cls(
            qubit1_f01=float(model.qubit1_f01),
            qubit2_f01=float(model.qubit2_f01),
            qubit_f01=float(model.qubit_f01),
            coupler_f01=float(model.coupler_f01),
            qubit1_anharm=float(model.qubit1_anharm),
            qubit2_anharm=float(model.qubit2_anharm),
            qubit_anharm=float(model.qubit_anharm),
            qr_g=float(model.qr_g),
            qq_g=float(model.qq_g),
            qc_g=float(model.qc_g),
            qq_geff=float(model.qq_geff),
        )

    @classmethod
    def from_mapping(cls, payload) -> 'FGF1V1BasicMetricBundle':
        """Normalize either the typed bundle or the previous dict-shaped payload."""
        if isinstance(payload, cls):
            return payload
        return cls(
            qubit1_f01=float(payload['qubit1_f01']),
            qubit2_f01=float(payload['qubit2_f01']),
            qubit_f01=float(payload['qubit_f01']),
            coupler_f01=float(payload['coupler_f01']),
            qubit1_anharm=float(payload['qubit1_anharm']),
            qubit2_anharm=float(payload['qubit2_anharm']),
            qubit_anharm=float(payload['qubit_anharm']),
            qr_g=float(payload['qr_g']),
            qq_g=float(payload['qq_g']),
            qc_g=float(payload['qc_g']),
            qq_geff=float(payload['qq_geff']),
        )

    def restore_onto(self, model) -> None:
        """Restore the cached FGF1V1 derived metrics onto one live model instance."""
        model.qubit1_f01 = self.qubit1_f01
        model.qubit2_f01 = self.qubit2_f01
        model.qubit_f01 = self.qubit_f01
        model.coupler_f01 = self.coupler_f01
        model.qubit1_anharm = self.qubit1_anharm
        model.qubit2_anharm = self.qubit2_anharm
        model.qubit_anharm = self.qubit_anharm
        model.qr_g = self.qr_g
        model.qq_g = self.qq_g
        model.qc_g = self.qc_g
        model.qq_geff = self.qq_geff


def _array_cache_key(values) -> tuple[float, ...]:
    """Convert a numeric array-like payload into a stable exact-input cache key."""
    return tuple(float(value) for value in np.asarray(values, dtype=float).reshape(-1).tolist())


def _make_qcrfgr_probe_frequency_cache_key(
    model,
    probe_flux_transformed,
    probe_ratio_transformed,
    qubit_idx: int | None,
) -> tuple[object, ...]:
    """Build the exact-input cache key for repeated identical QCRFGR probe requests."""
    ejmax_to_use = (
        model._ParameterizedQubit__Ej0
        if hasattr(model, '_ParameterizedQubit__Ej0')
        else model.Ejmax
    )
    return (
        tuple(int(level) for level in np.asarray(model._Nlevel).reshape(-1).tolist()),
        tuple(float(charge) for charge in np.asarray(model._charges).reshape(-1).tolist()),
        _array_cache_key(model.Ec),
        _array_cache_key(model.El),
        _array_cache_key(ejmax_to_use),
        _array_cache_key(probe_flux_transformed),
        _array_cache_key(probe_ratio_transformed),
        -1 if qubit_idx is None else int(qubit_idx),
    )


def _get_cached_qcrfgr_probe_frequency(cache_key: tuple[object, ...]) -> float | None:
    """Return the cached QCRFGR probe result when the exact-input key has been seen before."""
    cached_frequency = _QCRFGR_PROBE_FREQUENCY_CACHE.get(cache_key)
    if cached_frequency is not None:
        _QCRFGR_PROBE_FREQUENCY_CACHE.move_to_end(cache_key)
    return cached_frequency


def _store_cached_qcrfgr_probe_frequency(cache_key: tuple[object, ...], frequency: float) -> None:
    """Store one exact-input QCRFGR probe result with a small LRU-style eviction policy."""
    _QCRFGR_PROBE_FREQUENCY_CACHE[cache_key] = frequency
    _QCRFGR_PROBE_FREQUENCY_CACHE.move_to_end(cache_key)
    if len(_QCRFGR_PROBE_FREQUENCY_CACHE) > _QCRFGR_PROBE_FREQUENCY_CACHE_MAXSIZE:
        _QCRFGR_PROBE_FREQUENCY_CACHE.popitem(last=False)


def _clear_qcrfgr_probe_frequency_cache() -> None:
    """Clear the process-level QCRFGR probe cache used by repeated identical fast-path calls."""
    _QCRFGR_PROBE_FREQUENCY_CACHE.clear()


def _normalize_state_labels(state_labels: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    """Convert cached exact-input state labels into the list-of-list form expected by qutip helpers."""
    return [list(state) for state in state_labels]


def _normalize_nlevel_cache_key(nlevel) -> tuple[int, ...]:
    """Convert an `Nlevel` payload into a stable exact-input cache key."""
    return tuple(int(level) for level in np.asarray(nlevel).reshape(-1).tolist())


def _build_tensor_basis_state_index(
    state_label: tuple[int, ...],
    nlevel_key: tuple[int, ...],
) -> int:
    """Return the tensor-basis linear index for one product-state label."""
    state_index = 0
    stride = 1
    for level, dimension in zip(reversed(state_label), reversed(nlevel_key)):
        state_index += int(level) * stride
        stride *= int(dimension)
    return state_index


@lru_cache(maxsize=32)
def _get_cached_qcrfgr_metric_state_sets(
    nlevel_key: tuple[int, ...],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Build and cache the repeated QCRFGR product-state sets used by constructor metrics."""
    nlevel_list = list(nlevel_key)
    metric_states = tuple(
        cal_product_state_list(
            _normalize_state_labels(_QCRFGR_METRIC_STATE_LABELS),
            nlevel_list,
        )
    )
    coupler_overlap_states = tuple(
        cal_product_state_list(
            _normalize_state_labels(_QCRFGR_COUPLER_OVERLAP_STATE_LABELS),
            nlevel_list,
        )
    )
    return metric_states, coupler_overlap_states


def _clear_qcrfgr_metric_state_cache() -> None:
    """Clear the exact-input QCRFGR product-state cache used by repeated constructor metrics."""
    _get_cached_qcrfgr_metric_state_sets.cache_clear()


@lru_cache(maxsize=32)
def _get_cached_fgf1v1_metric_state_sets(
    nlevel_key: tuple[int, ...],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """Build and cache the repeated FGF1V1 product-state sets used by sweep-side metrics."""
    nlevel_list = list(nlevel_key)
    metric_states = tuple(
        cal_product_state_list(
            _normalize_state_labels(_FGF1V1_METRIC_STATE_LABELS),
            nlevel_list,
        )
    )
    qc_overlap_states = tuple(
        cal_product_state_list(
            _normalize_state_labels(_FGF1V1_QC_OVERLAP_STATE_LABELS),
            nlevel_list,
        )
    )
    qq_overlap_states = tuple(
        cal_product_state_list(
            _normalize_state_labels(_FGF1V1_QQ_OVERLAP_STATE_LABELS),
            nlevel_list,
        )
    )
    return metric_states, qc_overlap_states, qq_overlap_states


def _clear_fgf1v1_metric_state_cache() -> None:
    """Clear the exact-input FGF1V1 product-state cache used by repeated metric refreshes."""
    _get_cached_fgf1v1_metric_state_sets.cache_clear()


@lru_cache(maxsize=32)
def _get_cached_fgf1v1_overlap_basis_indices(
    nlevel_key: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build the tensor-basis indices for the repeated FGF1V1 overlap states."""
    qc_overlap_indices = tuple(
        _build_tensor_basis_state_index(state_label, nlevel_key)
        for state_label in _FGF1V1_QC_OVERLAP_STATE_LABELS
    )
    qq_overlap_indices = tuple(
        _build_tensor_basis_state_index(state_label, nlevel_key)
        for state_label in _FGF1V1_QQ_OVERLAP_STATE_LABELS
    )
    return qc_overlap_indices, qq_overlap_indices


def _make_fgf1v1_basic_metric_cache_key(model) -> tuple[object, ...] | None:
    """Build the exact-input FGF1V1 metric cache key for repeated identical replay refreshes."""
    maxwellmat = getattr(model, 'Maxwellmat', None)
    if maxwellmat is None or 'capac' not in maxwellmat or not hasattr(model, '_qrcouple_term'):
        return None

    qrcouple = tuple(float(value) for value in np.asarray(model._qrcouple_term, dtype=float).reshape(-1).tolist())
    return (
        model._get_exact_solve_template_cache_key(),
        qrcouple,
        float(maxwellmat['capac'][0, 0]),
        float(maxwellmat['capac'][1, 1]),
        float(model._capac[0, 1]),
        float(model._capac[1, 0]),
    )


def _get_cached_fgf1v1_basic_metrics(
    cache_key: tuple[object, ...] | None,
) -> FGF1V1BasicMetricBundle | None:
    """Return cached FGF1V1 derived metrics when the exact-input replay key has been seen before."""
    if cache_key is None:
        return None

    cached_metrics = _FGF1V1_BASIC_METRIC_CACHE.get(cache_key)
    if cached_metrics is not None:
        _FGF1V1_BASIC_METRIC_CACHE.move_to_end(cache_key)
        return FGF1V1BasicMetricBundle.from_mapping(cached_metrics)
    return None


def _store_cached_fgf1v1_basic_metrics(
    cache_key: tuple[object, ...] | None,
    metrics: FGF1V1BasicMetricBundle,
) -> None:
    """Store one FGF1V1 exact-input derived-metric bundle with a small LRU-style eviction policy."""
    if cache_key is None:
        return

    _FGF1V1_BASIC_METRIC_CACHE[cache_key] = FGF1V1BasicMetricBundle.from_mapping(metrics)
    _FGF1V1_BASIC_METRIC_CACHE.move_to_end(cache_key)
    if len(_FGF1V1_BASIC_METRIC_CACHE) > _FGF1V1_BASIC_METRIC_CACHE_MAXSIZE:
        _FGF1V1_BASIC_METRIC_CACHE.popitem(last=False)


def _clear_fgf1v1_basic_metric_cache() -> None:
    """Clear the FGF1V1 exact-input metric bundle cache used by repeated replay refreshes."""
    _FGF1V1_BASIC_METRIC_CACHE.clear()


def _get_cached_fgf1v1_metric_state_indices(
    exact_solve_cache_key: tuple[object, ...] | None,
) -> tuple[int, ...] | None:
    """Return cached FGF1V1 metric-state indices for one exact solved eigensystem."""
    if exact_solve_cache_key is None:
        return None

    cached_indices = _FGF1V1_METRIC_STATE_INDEX_CACHE.get(exact_solve_cache_key)
    if cached_indices is not None:
        _FGF1V1_METRIC_STATE_INDEX_CACHE.move_to_end(exact_solve_cache_key)
    return cached_indices


def _store_cached_fgf1v1_metric_state_indices(
    exact_solve_cache_key: tuple[object, ...] | None,
    state_indices,
) -> None:
    """Store one exact-input FGF1V1 metric-state index bundle with LRU-style eviction."""
    if exact_solve_cache_key is None:
        return

    _FGF1V1_METRIC_STATE_INDEX_CACHE[exact_solve_cache_key] = tuple(
        int(index) for index in state_indices
    )
    _FGF1V1_METRIC_STATE_INDEX_CACHE.move_to_end(exact_solve_cache_key)
    if len(_FGF1V1_METRIC_STATE_INDEX_CACHE) > _FGF1V1_METRIC_STATE_INDEX_CACHE_MAXSIZE:
        _FGF1V1_METRIC_STATE_INDEX_CACHE.popitem(last=False)


def _clear_fgf1v1_metric_state_index_cache() -> None:
    """Clear the exact-input FGF1V1 metric-state index cache."""
    _FGF1V1_METRIC_STATE_INDEX_CACHE.clear()


def _get_instance_fgf1v1_basic_metric_cache(model) -> OrderedDict:
    """Return the per-instance FGF1V1 metric cache used by same-instance replay restores."""
    cache = getattr(model, '_fgf1v1_instance_basic_metric_cache', None)
    if cache is None:
        cache = OrderedDict()
        model._fgf1v1_instance_basic_metric_cache = cache
    return cache


def _get_cached_instance_fgf1v1_basic_metrics(
    model,
    exact_solve_cache_key: tuple[object, ...] | None,
) -> FGF1V1BasicMetricBundle | None:
    """Return one same-instance FGF1V1 metric bundle for the current exact replay key."""
    if exact_solve_cache_key is None:
        return None

    cache = getattr(model, '_fgf1v1_instance_basic_metric_cache', None)
    if cache is None:
        return None

    cached_metrics = cache.get(exact_solve_cache_key)
    if cached_metrics is not None:
        cache.move_to_end(exact_solve_cache_key)
        return FGF1V1BasicMetricBundle.from_mapping(cached_metrics)
    return None


def _store_instance_fgf1v1_basic_metrics(
    model,
    exact_solve_cache_key: tuple[object, ...] | None,
    metrics: FGF1V1BasicMetricBundle,
) -> None:
    """Store one per-instance FGF1V1 metric bundle keyed by the exact replay state."""
    if exact_solve_cache_key is None:
        return

    cache = _get_instance_fgf1v1_basic_metric_cache(model)
    cache[exact_solve_cache_key] = FGF1V1BasicMetricBundle.from_mapping(metrics)
    cache.move_to_end(exact_solve_cache_key)
    if len(cache) > _FGF1V1_INSTANCE_BASIC_METRIC_CACHE_MAXSIZE:
        cache.popitem(last=False)


def _capture_fgf1v1_basic_metrics(model) -> FGF1V1BasicMetricBundle:
    """Capture the current FGF1V1 derived metric bundle for exact-input replay reuse."""
    return FGF1V1BasicMetricBundle.capture(model)


def _restore_fgf1v1_basic_metrics(model, metrics) -> None:
    """Restore one cached FGF1V1 derived metric bundle onto the live model instance."""
    FGF1V1BasicMetricBundle.from_mapping(metrics).restore_onto(model)


def _build_fgf1v1_uncached_basic_metric_bundle(model) -> FGF1V1BasicMetricBundle:
    """Compute one fresh FGF1V1 metric bundle without round-tripping through slower public helpers."""
    nlevel_key = _normalize_nlevel_cache_key(model._Nlevel)
    metric_states, _, _ = _get_cached_fgf1v1_metric_state_sets(nlevel_key)
    qc_overlap_indices, qq_overlap_indices = _get_cached_fgf1v1_overlap_basis_indices(nlevel_key)
    exact_solve_cache_key = getattr(model, '_exact_solve_template_cache_key', None)
    state_index = _get_cached_fgf1v1_metric_state_indices(exact_solve_cache_key)
    if state_index is None:
        state_index = tuple(model.find_state_list(metric_states, state_space=model._eigenstates))
        _store_cached_fgf1v1_metric_state_indices(exact_solve_cache_key, state_index)
    relative_levels = model._energylevels - model._energylevels[0]

    qubit1_f01 = float(relative_levels[state_index[1]] / 2 / pi)
    qubit2_f01 = float(relative_levels[state_index[2]] / 2 / pi)
    coupler_f01 = float(relative_levels[state_index[3]] / 2 / pi)
    qubit1_anharm = float(relative_levels[state_index[5]] / 2 / pi - 2 * qubit1_f01)
    qubit2_anharm = float(relative_levels[state_index[6]] / 2 / pi - 2 * qubit2_f01)

    omega_qubit = qubit1_f01 * 1e9 * 2 * pi
    readout_freq = 6.5e9
    omega_res = readout_freq * 2 * pi
    capacitance_matrix = model.Maxwellmat['capac']
    c_q = e**2 / 2 / model.Ec[0, 0] / 1e9 / hbar
    c_r = 1 / 8 / readout_freq / RLINE
    c_qr1, c_qr2 = model._qrcouple_term
    c_q1 = capacitance_matrix[0, 0] - model._capac[0, 1] - c_qr1
    c_q2 = capacitance_matrix[1, 1] - model._capac[1, 0] - c_qr2
    c_eff = abs(c_qr1 * c_q1 - c_qr2 * c_q2) / (c_q1 + c_q2)
    qr_g = float(c_eff * np.sqrt(omega_res * omega_qubit / (c_r * c_q)) / 2)

    hamiltonian = model._Hamiltonian
    q1c_g = abs(hamiltonian[qc_overlap_indices[1], qc_overlap_indices[0]]) / 2 / pi
    q2c_g = abs(hamiltonian[qc_overlap_indices[1], qc_overlap_indices[2]]) / 2 / pi
    qc_g = float((q1c_g + q2c_g) / 2)
    qq_g = float(abs(hamiltonian[qq_overlap_indices[1], qq_overlap_indices[0]]) / 2 / pi)

    delta1 = qubit1_f01 - coupler_f01
    delta2 = qubit2_f01 - coupler_f01
    sum1 = qubit1_f01 + coupler_f01
    sum2 = qubit2_f01 + coupler_f01
    qq_geff = float(qc_g * qc_g * (1 / delta1 + 1 / delta2 - 1 / sum1 - 1 / sum2) / 2 + qq_g)

    return FGF1V1BasicMetricBundle(
        qubit1_f01=qubit1_f01,
        qubit2_f01=qubit2_f01,
        qubit_f01=float((qubit1_f01 + qubit2_f01) / 2),
        coupler_f01=coupler_f01,
        qubit1_anharm=qubit1_anharm,
        qubit2_anharm=qubit2_anharm,
        qubit_anharm=float((qubit1_anharm + qubit2_anharm) / 2),
        qr_g=qr_g,
        qq_g=qq_g,
        qc_g=qc_g,
        qq_geff=qq_geff,
    )


def _calculate_qcrfgr_overlap_coupling(model, overlap_states) -> float:
    """Calculate the overlap-mode qubit-coupler coupling from prebuilt product states."""
    hamil = model.get_hamiltonian()
    return abs(overlap_states[1].dag() * hamil * overlap_states[0]) / 2 / pi


def _validate_readout_couple_mode(couple_mode: str) -> None:
    if couple_mode != 'capac':
        raise ValueError(f'Unsupported couple_mode: {couple_mode}')


class GroundedTransmonList(GroundedTransmon):
    
    """
        A list of grounded transmons with different parameters. 
    """
    
    def __init__(
        self,
        *args, **kwargs
    ):
        _grounded_transmon_list_init(self, *args, **kwargs)
    

class QCRFGRModel(ParameterizedQubit):
    '''
    QCR-FGR Model
    topo: Resonator-FloatingQubit-GroundedTransmon
    
    Inherits cal_coupler_sensitivity() from ParameterizedQubit.
    '''
    def __init__(
            self, 
            capacitance_list: list[float],
            junc_resis_list: list[float],
            qrcouple: list[float],
            flux_list: list[float] = [0,0,0],
            trunc_ener_level: list[float] = [10,8,10],
            *args, **kwargs
        ):
        Cq1g, Cq2g, Cqq, Cc, Cqc = capacitance_list
        self._capac = np.array([
            [Cq1g,  Cqq,  Cqc,],
            [Cqq,  Cq2g,  0,  ],
            [Cqc, 0,    Cc,   ],
        ])
        self._resis = np.ones_like(self._capac)*RNAN
        self._resis[0,1]=self._resis[1,0]=junc_resis_list[0]
        self._resis[2,2]=junc_resis_list[1]
        self._flux = np.zeros_like(self._capac)
        self._flux[0,1]=self._flux[1,0]=flux_list[0]
        self._flux[2,2]=flux_list[1]
        self._qrcouple_term = qrcouple
        super().__init__(
            capacitances=self._capac, 
            junctions_resistance=self._resis,
            fluxes=self._flux, 
            trunc_ener_level=trunc_ener_level, 
            structure_index=[2,1], 
            *args, **kwargs)
        self._refresh_basic_metrics()
        self.print_basic_info()

    def _refresh_basic_metrics(self):
        standard_state, coupler_overlap_states = _get_cached_qcrfgr_metric_state_sets(
            _normalize_nlevel_cache_key(self._Nlevel)
        )
        state_index = [self.find_state(state) for state in standard_state]

        self.qubit_f01 = self.get_energylevel(state_index[0])/2/pi
        self.qubit_anharm = self.get_energylevel(state_index[2])/2/pi-2*self.qubit_f01
        self.coupler_f01 = self.get_energylevel(state_index[1])/2/pi
        self.rq_g = self.get_readout_couple(readout_freq=6.5e9, couple_mode='capac', is_print=False)
        self.qc_g = _calculate_qcrfgr_overlap_coupling(self, coupler_overlap_states)

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_metrics()
        return updated
        
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit frequency: {self.qubit_f01:.3f} GHz')
            print(f'Coupler frequency: {self.coupler_f01:.3f} GHz')
            print(f'Qubit anharmonicity: {self.qubit_anharm*1e3:.3f} MHz')

            print(f'Readout coupling strenth: {self.rq_g/1e6/2/pi:.3f} MHz (res_freq=6.5GHz, Capac couple)')
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")

    def get_coupler_couple(self, mode: str = 'overlap', is_print: bool = True) -> float:
        """
        Get the coupling strenth between qubit and coupler.
        """
        if mode == 'direct':
            Cqsum = e**2/2/self.Ec[0,0]/1e9/hbar
            Ccsum = e**2/2/self.Ec[1,1]/1e9/hbar
            eta = self._capac[1,1]/(self._capac[0,0]+self._capac[1,1])
            self.qc_g = self._capac[0][2]*eta*np.sqrt(self.qubit_f01*self.coupler_f01/Cqsum/Ccsum)/2
        elif mode == 'overlap':
            _, standard_state = _get_cached_qcrfgr_metric_state_sets(
                _normalize_nlevel_cache_key(self._Nlevel)
            )
            self.qc_g = _calculate_qcrfgr_overlap_coupling(self, standard_state)
        
        if is_print:
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")
        return self.qc_g

    def get_readout_couple(
        self, 
        readout_freq:float,
        couple_mode: str = 'capac',
        is_print: bool = True,
    ):
        _validate_readout_couple_mode(couple_mode)
        omega_qubit = self.get_energylevel(1)*1e9
        omega_res = readout_freq*2*pi
        Cq = e**2/2/self.Ec[0,0]/1e9/hbar
        Cr = 1/8/readout_freq/RLINE
        Cqr1, Cqr2 = self._qrcouple_term
        Cq1 = self.Maxwellmat['capac'][0,0]-self._capac[0,1]-Cqr1
        Cq2 = self.Maxwellmat['capac'][1,1]-self._capac[1,0]-Cqr2
        C_eff = abs(Cqr1*Cq1-Cqr2*Cq2)/(Cq1+Cq2)
        g = C_eff*np.sqrt(omega_res*omega_qubit/(Cr*Cq))/2
        if is_print:
            print(f'Capacitance coupling strenth: {g/1e6/2/pi:.3f}MHz')
        return g



class FGF1V1Coupling(ParameterizedQubit):
    '''
    Topo: F-G-F
    
    Inherits cal_coupler_sensitivity() from ParameterizedQubit.
    '''
    
    def __init__(
        self,
        capacitance_list: list[float],
        junc_resis_list: list[float],
        qrcouple: list[float],
        flux_list: list[float] = [0,0,0],
        trunc_ener_level: list[float] = [10,8,10],
        is_print: bool = True,
        *args, **kwargs
    ):
        """
            capacitance_list: set of all capacitances neccesary, [C11, C12, C1q, Cc, C2q, C21, C22, Cqq, Cq1c, Cq2c]
            junc_resis_list: the junc_resistance list of F-G-F
            flux_list: the flux list of F-G-F
        """
        C11g, C12g, C1q, Cc, C2q, C21g, C22g, Cqq, Cq1c, Cq2c = capacitance_list
        self._capac = np.array([
            [C11g,  C1q,  Cq1c,  Cqq,    0],
            [C1q,  C12g,  0,     0,      0],
            [Cq1c, 0,    Cc,    Cq2c,   0],
            [Cqq,  0,    Cq2c,  C21g,  C2q],
            [0,    0,    0,     C2q,  C22g]
        ])
        self._resis = np.ones_like(self._capac)*RNAN
        self._resis[0,1]=self._resis[1,0]=junc_resis_list[0]
        self._resis[2,2]=junc_resis_list[1]
        self._resis[3,4]=self._resis[4,3]=junc_resis_list[2]
        self._flux = np.zeros_like(self._capac)
        self._flux[0,1]=self._flux[1,0]=flux_list[0]
        self._flux[2,2]=flux_list[1]
        self._flux[3,4]=self._flux[4,3]=flux_list[2]
        self._Nlevel = trunc_ener_level
        self._qrcouple_term = qrcouple
        super().__init__(
            capacitances=self._capac,
            junctions_resistance=self._resis,
            fluxes=self._flux,
            trunc_ener_level=self._Nlevel,
            structure_index=[2,1,2],
            *args, **kwargs
        )
        self._refresh_basic_metrics()
        self.print_basic_info(is_print=is_print)

    def _refresh_basic_metrics(self):
        exact_solve_cache_key = getattr(self, '_exact_solve_template_cache_key', None)
        if getattr(self, '_active_exact_solve_template', None) is not None:
            cached_metrics = _get_cached_instance_fgf1v1_basic_metrics(self, exact_solve_cache_key)
            if cached_metrics is not None:
                _restore_fgf1v1_basic_metrics(self, cached_metrics)
                return

        cache_key = _make_fgf1v1_basic_metric_cache_key(self)
        cached_metrics = _get_cached_fgf1v1_basic_metrics(cache_key)
        if cached_metrics is not None:
            _restore_fgf1v1_basic_metrics(self, cached_metrics)
            _store_instance_fgf1v1_basic_metrics(self, exact_solve_cache_key, cached_metrics)
            return

        metrics = _build_fgf1v1_uncached_basic_metric_bundle(self)
        _restore_fgf1v1_basic_metrics(self, metrics)
        _store_cached_fgf1v1_basic_metrics(cache_key, metrics)
        _store_instance_fgf1v1_basic_metrics(self, exact_solve_cache_key, metrics)

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_metrics()
        return updated
    
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit 1 frequency: {self.qubit1_f01:.3f} GHz')
            print(f'Qubit 2 frequency: {self.qubit2_f01:.3f} GHz')
            print(f'Coupler frequency: {self.coupler_f01:.3f} GHz')
            print(f'Qubit anharmonicity: {self.qubit_anharm*1e3:.3f} MHz')
            print(f'Readout coupling strenth: {self.qr_g/1e6/2/pi:.3f} MHz (Read_freq=6.5GHz, Capac couple)')
            print(f"Qubit-Qubit direct coupling: {self.qq_g*1e3:.3f}MHz")
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")
            print(f"Qubit-Qubit effective coupling: {self.qq_geff*1e3:.3f} MHz")

        return True

    @property    
    def capac_keywords(self):
        print("C11, C12, C1q, Cc, C2q, C21, C22, Cqq, Cq1c, Cq2c")
        return ["C11", "C12", "C1q", "Cc", "C2q", "C21", "C22", "Cqq", "Cq1c", "Cq2c"]
    
    def topology(self):
        print('F-G-F')
        return 0
    
    def QubitDephasingbyCouplerThermal(self, coupler_flux):
        return _fgf1v1_qubit_dephasing_by_coupler_thermal(self, coupler_flux)
    
    def get_readout_couple(
        self, 
        readout_freq:float,
        couple_mode: str = 'capac',
        is_print: bool = True,
    ) -> float:
        _validate_readout_couple_mode(couple_mode)
        omega_qubit = self.qubit1_f01*1e9*2*pi
        omega_res = readout_freq*2*pi
        Cq = e**2/2/self.Ec[0,0]/1e9/hbar
        Cr = 1/8/readout_freq/RLINE
        Cqr1, Cqr2 = self._qrcouple_term
        Cq1 = self.Maxwellmat['capac'][0,0]-self._capac[0,1]-Cqr1
        Cq2 = self.Maxwellmat['capac'][1,1]-self._capac[1,0]-Cqr2
        C_eff = abs(Cqr1*Cq1-Cqr2*Cq2)/(Cq1+Cq2)
        self.qr_g = C_eff*np.sqrt(omega_res*omega_qubit/(Cr*Cq))/2
        if is_print:
            print(f'Capacitance coupling strenth: {self.qr_g/1e6/2/pi:.3f}MHz')
        return self.qr_g
    
    def get_qc_couple(self, mode: str = 'overlap', is_print: bool = True) -> float:
        '''
        Get Qubit - Coupler direct coupling strength.
        
        mode: str
            'direct': calculating by couple capacitance.
            'overlap': calculating by state overlap.
        '''
        if mode=='direct':
            Cqsum = e**2/2/self.Ec[0,0]/1e9/hbar
            Ccsum = e**2/2/self.Ec[1,1]/1e9/hbar
            Cqc = e**2/2/self.Ec[0,1]/1e9/hbar
            eta = self._capac[1,1]/(self._capac[0,0]+self._capac[1,1])
            self.qc_g = np.sqrt(self.qubit1_f01*self.coupler_f01*Cqsum*Ccsum)/2/Cqc
        elif mode=='overlap':
            _, standard_state, _ = _get_cached_fgf1v1_metric_state_sets(
                _normalize_nlevel_cache_key(self._Nlevel)
            )
            hamil = self.get_hamiltonian()
            self.q1c_g = abs(standard_state[1].dag()*hamil*standard_state[0])/2/pi
            self.q2c_g = abs(standard_state[1].dag()*hamil*standard_state[2])/2/pi
            self.qc_g = (self.q1c_g+self.q2c_g)/2
        
        if is_print:
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")
        return self.qc_g
    
    def get_qq_dcouple(self, mode: str = 'overlap', is_print: bool = True) -> float:
        '''
        Get Qubit - Qubit direct coupling strength.
        
        mode: str
            'capac': calculating by couple capacitance.
            'overlap': calculating by state overlap.
        '''
        if mode=='capac':
            Cq1sum = e**2/2/self.Ec[0,0]/1e9/hbar
            # Ccsum = e**2/2/self.Ec[1,1]/1e9/hbar
            Cq2sum = e**2/2/self.Ec[2,2]/1e9/hbar
            Cqq = e**2/2/self.Ec[0,2]/1e9/hbar
            # Cqc1 = e**2/2/self.Ec[0,1]/1e9/hbar
            # Cqc2 = e**2/2/self.Ec[1,2]/1e9/hbar
            eta1 = self._capac[1,1]/(self._capac[0,0]+self._capac[1,1])
            eta2 = self._capac[3,3]/(self._capac[4,4]+self._capac[3,3])
            # eta = eta1*eta2
            # eta = Cqc1*Cqc2/Cqq/Ccsum+1
            self.qq_g = np.sqrt(self.qubit1_f01*self.qubit2_f01*Cq1sum*Cq2sum)/2/Cqq
        elif mode=='overlap':
            _, _, standard_state = _get_cached_fgf1v1_metric_state_sets(
                _normalize_nlevel_cache_key(self._Nlevel)
            )
            hamil = self.get_hamiltonian()
            self.qq_g = abs(standard_state[1].dag()*hamil*standard_state[0])/2/pi
        else:
            raise ValueError(f"mode {mode} is not supported for type {type}")
        
        if is_print:
            print(f"Qubit-Qubit direct coupling: {self.qq_g*1e3:.3f}MHz")
        return self.qq_g
    
    def get_qq_ecouple(self, method: str = 'ES', is_print: bool = True) -> float:
        '''
        Get Qubit - Qubit effective coupling strength.
        
        method: str
            'ED': calculating by energy difference.
            'SW': calculating by SW formula.
            'ES': calculating by ES formula.
        '''
        if method=='ED':
            """
            ED requires the two qubit is just the same frequency, so it only works when we design the same frequency qubits. 
            """
            freq_q1 = self.qubit1_f01
            freq_q2 = self.qubit2_f01
            self.qq_geff = (freq_q1-freq_q2)/2
            
        elif method=='SW':
            '''
            SW requires anharmonicity is far lower than the frequency_diff between qubits and coupler, and the g_12 is a second order small 
            quantity of g_1c and g_2c. So it doesn't work when coupler_detune is large. 
            '''
            freq_q1 = self.qubit1_f01
            freq_q2 = self.qubit1_f01
            freq_c = self.coupler_f01
            
            delta1 = freq_q1-freq_c
            delta2 = freq_q2-freq_c
            sum1 = freq_q1+freq_c
            sum2 = freq_q2+freq_c
            Cc = self._capac[2,2]
            C12 = self._capac[0,3]
            C1c = self._capac[0,2]
            C2c = self._capac[2,3]
            C1 = self._capac[0,0]
            C2 = self._capac[1,1]
            eta = C1c*C2c/C12/Cc
            self.qq_geff = (freq_c*eta*(1/(delta1)+1/(delta2)-1/(sum1)-1/(sum2))/4+eta+1)*C12*np.sqrt(freq_q1*freq_q2/C1/C2)/2
        
        elif method=='ES':
            freq_q1 = self.qubit1_f01
            freq_q2 = self.qubit1_f01
            freq_c = self.coupler_f01
            g1 = g2 = self.qc_g
            g12 = self.qq_g

            delta1 = self.qubit1_f01 - self.coupler_f01
            delta2 = self.qubit2_f01 - self.coupler_f01
            sum1 = self.qubit1_f01 + self.coupler_f01
            sum2 = self.qubit2_f01 + self.coupler_f01

            self.qq_geff = g1*g2*(1/delta1+1/delta2-1/sum1-1/sum2)/2+g12

        else:
            raise ValueError(f"method {method} is not supported.")
        
        if is_print:
            print(f"Qubit-Qubit effective coupling: {self.qq_geff*1e3:.3f} MHz")
        
        return self.qq_geff



class FGF2V7Coupling(ParameterizedQubit):
    '''
            G   G
            |   |
    Topo: G-F-G-F-G
            |   |
            G   G
    '''
    
    def __init__(
        self,
        capacitance_list: list[float],
        junc_resis_list: list[float],
        flux_list: list[float] = [0,0],
        trunc_ener_level: list[float] = [8,5],
        is_symmetric: bool = True,
        *args, **kwargs
    ):
        """
            capacitance_list: set of all capacitances neccesary, [C11, C12, C1q, Cc, C2q, C21, C22, Cqq, Cq1c, Cq2c]
            junc_resis_list: the junc_resistance list of F-G-F
            flux_list: the flux list of F-G-F
        """
        if is_symmetric:
            C11g, C12g, C1q, Cc, C2q, C21g, C22g, Cqq, Cq1c, Cq2c = capacitance_list
            self._capac = np.array([
                [C11g,  C1q,  Cq1c,  Cqq,   0,      Cq1c,   0,      0,      0,      0,      0],
                [C1q,  C12g,  0,     0,     0,      0,      Cq2c,   Cq2c,   0,      0,      0],
                [Cq1c, 0,    Cc,    Cq2c,   0,      0,      0,      0,      0,      0,      0],
                [Cqq,  0,    Cq2c,  C21g,   C2q,    0,      0,      0,      Cq2c,   0,      0],
                [0,    0,    0,     C2q,    C22g,   0,      0,      0,      0,      Cq1c,   Cq1c],
                [Cq1c, 0,    0,     0,      0,      Cc,     0,      0,      0,      0,      0],
                [0,    Cq2c, 0,     0,      0,      0,      Cc,     0,      0,      0,      0],
                [0,    Cq2c, 0,     0,      0,      0,      0,      Cc,     0,      0,      0],
                [0,    0,    0,     Cq2c,   0,      0,      0,      0,      Cc,     0,      0],
                [0,    0,    0,     0,      Cq1c,   0,      0,      0,      0,      Cc,     0],
                [0,    0,    0,     0,      Cq1c,   0,      0,      0,      0,      0,      Cc],
            ])
            self._resis = np.ones_like(self._capac)
            self._resis[0,1]=self._resis[1,0]=junc_resis_list[0]
            self._resis[3,4]=self._resis[4,3]=junc_resis_list[0]
            for ii in [2,5,6,7,8,9,10]:
                self._resis[ii,ii]=junc_resis_list[1]
            self._flux = np.zeros_like(self._capac)
            self._flux[0,1]=self._flux[1,0]=flux_list[0]
            self._flux[3,4]=self._flux[4,3]=flux_list[0]
            for ii in [2,5,6,7,8,9,10]:
                self._flux[ii,ii]=flux_list[1]
            self._Nlevel = [trunc_ener_level[0],trunc_ener_level[1],trunc_ener_level[0]]+[trunc_ener_level[1]]*6
        else:
            self._capac = np.array(capacitance_list)
            self._resis = np.array(junc_resis_list)
            self._flux = np.array(flux_list)
            self._Nlevel = [trunc_ener_level[0],trunc_ener_level[1],trunc_ener_level[0]]+[trunc_ener_level[1]]*6
        super().__init__(
            capacitances=self._capac,
            junctions_resistance=self._resis,
            fluxes=self._flux,
            trunc_ener_level=self._Nlevel,
            structure_index=[2,1,2,1,1,1,1,1,1],
            *args, **kwargs
        )
        self._refresh_basic_metrics()
        self.print_basic_info()

    def _refresh_basic_metrics(self):
        self.qubit1_f01 = self.get_energylevel(1)/2/pi
        self.qubit2_f01 = self.get_energylevel(2)/2/pi
        self.qubit_f01 = (self.qubit1_f01+self.qubit2_f01)/2
        el_initial = [self.get_energylevel(ii)/2/pi for ii in range(1,7)]

        anhar_pre = -self.Ec[0,0]/2/pi
        target = 2*self.qubit1_f01+anhar_pre
        indice_200 = np.where(np.abs(el_initial - target) < 80e-3)[0][0]
        target = self.qubit2_f01+self.qubit1_f01
        indice_101 = np.where(np.abs(el_initial - target) < 1e-3)[0][0]

        indice_010 = 13-2*indice_200-indice_101

        self.qubit1_anharm = el_initial[indice_200]-2*self.qubit1_f01
        self.qubit2_anharm = el_initial[indice_200+1]-2*self.qubit2_f01
        self.qubit_anharm = (self.qubit1_anharm+self.qubit2_anharm)/2

        self.coupler_f01 = el_initial[indice_010]

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_metrics()
        return updated
    
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit 1 frequency: {self.qubit1_f01:.3f}GHz')
            print(f'Qubit 2 frequency: {self.qubit2_f01:.3f}GHz')
            print(f'Coupler frequency: {self.coupler_f01:.3f}GHz')
            print(f'Qubit anharmonicity: {self.qubit_anharm*1e3:.3f}MHz')
        
        return True

class FGFGG1V1V3Coupling(ParameterizedQubit):
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        _fgfgg1v1v3_coupling_init(self, *args, **kwargs)


def _qcrfgr_cal_coupler_sensitivity(
    self,
    coupler_flux_point: float,
    method: str = 'numerical',
    flux_step: float = 1e-4,
    qubit_idx: int = 0,
    is_print: bool = True,
    is_plot: bool = False,
) -> float:
    return analyze_multi_qubit_coupler_sensitivity(
        self,
        coupler_flux_point=coupler_flux_point,
        method=method,
        flux_step=flux_step,
        qubit_idx=qubit_idx,
        qubit_fluxes=None,
        is_print=is_print,
        is_plot=is_plot,
    )


def _qcrfgr_cal_sensitivity_numerical(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
) -> float:
    return calculate_multi_qubit_sensitivity_numerical(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
    )


def _qcrfgr_cal_sensitivity_analytical(
    self,
    coupler_flux_point: float,
    qubit_idx: int,
) -> float:
    return calculate_multi_qubit_sensitivity_analytical(
        self,
        coupler_flux_point,
        qubit_idx,
    )


def _qcrfgr_cal_coupler_self_sensitivity(self, coupler_flux: float) -> float:
    return calculate_multi_qubit_coupler_self_sensitivity(self, coupler_flux)


def _qcrfgr_get_qubit_frequency_at_coupler_flux(
    self,
    coupler_flux: float,
    qubit_idx: int,
    flux_offset: float = 0.0,
) -> float:
    return get_multi_qubit_frequency_at_coupler_flux(
        self,
        coupler_flux,
        qubit_idx=qubit_idx,
        flux_offset=flux_offset,
    )


def _qcrfgr_probe_frequency_at_coupler_flux_fast(
    self,
    coupler_flux: float,
    qubit_idx: int | None = None,
    flux_offset: float = 0.0,
) -> float:
    probe_flux = np.array(self._flux, dtype=float, copy=True)
    probe_flux[2, 2] = coupler_flux + flux_offset

    probe_flux_transformed = project_transformed_flux(
        probe_flux,
        self._ParameterizedQubit__struct,
        self.SMatrix_retainNodes,
    )
    probe_ratio_transformed = getattr(self, '_junc_ratio_transformed', None)
    if probe_ratio_transformed is None:
        probe_ratio_transformed = project_transformed_junction_ratio(
            self._junc_ratio,
            self._ParameterizedQubit__struct,
            self.SMatrix_retainNodes,
        )

    cache_key = _make_qcrfgr_probe_frequency_cache_key(
        self,
        probe_flux_transformed,
        probe_ratio_transformed,
        qubit_idx,
    )
    cached_frequency = _get_cached_qcrfgr_probe_frequency(cache_key)
    if cached_frequency is not None:
        return cached_frequency

    ejmax_to_use = (
        self._ParameterizedQubit__Ej0
        if hasattr(self, '_ParameterizedQubit__Ej0')
        else self.Ejmax
    )
    probe_ej = self._Ejphi(ejmax_to_use, probe_flux_transformed, probe_ratio_transformed)
    probe_hamiltonian = self._generate_hamiltonian(
        self.Ec,
        self.El,
        probe_ej,
        transient=True,
    )
    probe_solver = HamiltonianEvo(probe_hamiltonian)
    state_index = probe_solver.find_state([1, 0])
    if isinstance(state_index, list):
        state_index = state_index[0]
    frequency = probe_solver.get_energylevel(state_index) / 2 / pi
    _store_cached_qcrfgr_probe_frequency(cache_key, frequency)
    return frequency


def _qcrfgr_plot_sensitivity_curve(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
    sensitivity: float,
):
    plot_multi_qubit_sensitivity_curve(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
        sensitivity,
    )


def _fgf1_cal_coupler_sensitivity(
    self,
    coupler_flux_point: float,
    method: str = 'numerical',
    flux_step: float = 1e-4,
    qubit_idx: int = None,
    qubit_fluxes=None,
    is_print: bool = True,
    is_plot: bool = False,
) -> float:
    return analyze_multi_qubit_coupler_sensitivity(
        self,
        coupler_flux_point=coupler_flux_point,
        method=method,
        flux_step=flux_step,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
        is_print=is_print,
        is_plot=is_plot,
    )


def _fgf1_cal_sensitivity_numerical(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
    qubit_fluxes=None,
) -> float:
    return calculate_multi_qubit_sensitivity_numerical(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
        qubit_fluxes=qubit_fluxes,
    )


def _fgf1_cal_sensitivity_analytical(
    self,
    coupler_flux_point: float,
    qubit_idx: int,
) -> float:
    return calculate_multi_qubit_sensitivity_analytical(
        self,
        coupler_flux_point,
        qubit_idx,
    )


def _fgf1_cal_coupler_self_sensitivity(self, coupler_flux: float) -> float:
    return calculate_multi_qubit_coupler_self_sensitivity(self, coupler_flux)


def _fgf1_get_qubit_frequency_at_coupler_flux(
    self,
    coupler_flux: float,
    qubit_idx: int = None,
    qubit_fluxes=None,
    flux_offset: float = 0.0,
) -> float:
    return get_multi_qubit_frequency_at_coupler_flux(
        self,
        coupler_flux,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
        flux_offset=flux_offset,
    )


def _fgf1_plot_sensitivity_curve(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
    sensitivity: float,
):
    plot_multi_qubit_sensitivity_curve(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
        sensitivity,
    )


QCRFGRModel.cal_coupler_sensitivity = _qcrfgr_cal_coupler_sensitivity
QCRFGRModel._cal_sensitivity_numerical = _qcrfgr_cal_sensitivity_numerical
QCRFGRModel._cal_sensitivity_analytical = _qcrfgr_cal_sensitivity_analytical
QCRFGRModel._cal_coupler_self_sensitivity = _qcrfgr_cal_coupler_self_sensitivity
QCRFGRModel._get_qubit_frequency_at_coupler_flux = _qcrfgr_get_qubit_frequency_at_coupler_flux
QCRFGRModel._probe_frequency_at_coupler_flux_fast = _qcrfgr_probe_frequency_at_coupler_flux_fast
QCRFGRModel._plot_sensitivity_curve = _qcrfgr_plot_sensitivity_curve

FGF1V1Coupling.cal_coupler_sensitivity = _fgf1_cal_coupler_sensitivity
FGF1V1Coupling._cal_sensitivity_numerical = _fgf1_cal_sensitivity_numerical
FGF1V1Coupling._cal_sensitivity_analytical = _fgf1_cal_sensitivity_analytical
FGF1V1Coupling._cal_coupler_self_sensitivity = _fgf1_cal_coupler_self_sensitivity
FGF1V1Coupling._get_qubit_frequency_at_coupler_flux = _fgf1_get_qubit_frequency_at_coupler_flux
FGF1V1Coupling._plot_sensitivity_curve = _fgf1_plot_sensitivity_curve

__all__ = [
    'FGF1V1Coupling',
    'FGF2V7Coupling',
    'FGFGG1V1V3Coupling',
    'GroundedTransmonList',
    'QCRFGRModel',
    'RNAN',
]
