from collections import OrderedDict
from copy import copy

import numpy as np

from ..funclib import cal_product_state_list
from .circuit import project_transformed_flux
from .plotting import (
    plot_multi_qubit_coupling_strength_vs_flux,
    plot_multi_qubit_energy_vs_flux,
)
from .base import ParameterizedQubit, pi
from .types import CouplingResult, SweepResult


_SINGLE_QUBIT_SWEEP_RELATIVE_ENERGY_CACHE_MAXSIZE = 256


def _validate_single_qubit_sweep_upper_level(qubit, upper_level):
    if isinstance(qubit._Nlevel, (list, tuple, np.ndarray)):
        nlevel_val = qubit._Nlevel[0]
    else:
        nlevel_val = qubit._Nlevel

    if upper_level > nlevel_val:
        raise ValueError('Energy list out of range! Should lower than truncate energy level! ')


def _supports_fast_single_qubit_sweep(qubit) -> bool:
    """Return whether the sweep can reuse the internal single-qubit energy-only path."""
    return isinstance(qubit, ParameterizedQubit) and getattr(qubit, '_numQubits', 0) == 1


def _extract_relative_energylevels(hamiltonian) -> np.ndarray:
    """Return relative eigenenergies while avoiding eigenstate construction when possible."""
    eigenenergies = getattr(hamiltonian, 'eigenenergies', None)
    if callable(eigenenergies):
        energylevels = np.asarray(eigenenergies(), dtype=float)
    elif hasattr(hamiltonian, 'full'):
        energylevels = np.linalg.eigvalsh(np.asarray(hamiltonian.full(), dtype=complex)).real
    else:
        energylevels = np.asarray(hamiltonian.eigenstates()[0], dtype=float)

    return energylevels - energylevels[0]


def _get_single_qubit_sweep_relative_energy_cache(qubit) -> OrderedDict:
    """Return the per-instance exact-input cache used by repeated sweep points."""
    cache = getattr(qubit, '_single_qubit_sweep_relative_energy_cache', None)
    if cache is None:
        cache = OrderedDict()
        qubit._single_qubit_sweep_relative_energy_cache = cache
    return cache


def _get_single_qubit_sweep_relative_energy_cache_key_prefix(qubit) -> tuple[object, ...]:
    """Return the sweep-constant prefix for exact-input single-qubit cache lookups."""
    return (
        getattr(qubit, '_cal_mode', None),
        qubit._get_array_cache_key(qubit.Ec),
        qubit._get_array_cache_key(qubit.El),
        qubit._get_array_cache_key(qubit._charges),
        qubit._get_array_cache_key(qubit._Nlevel, as_int=True),
    )


def _get_single_qubit_sweep_relative_energy_cache_key(
    qubit,
    Ej,
    *,
    cache_key_prefix: tuple[object, ...] | None = None,
) -> tuple[object, ...]:
    """Return the exact-input cache key for one single-qubit sweep-point spectrum."""
    if cache_key_prefix is None:
        cache_key_prefix = _get_single_qubit_sweep_relative_energy_cache_key_prefix(qubit)
    return (*cache_key_prefix, qubit._get_array_cache_key(Ej))


def _extract_single_qubit_scalar_sweep_value(value) -> float | None:
    """Return one scalar sweep value when the fast path can avoid per-point shape normalization."""
    value_array = np.asarray(value, dtype=float)
    if value_array.ndim == 0:
        return float(value_array.item())
    if value_array.size == 1:
        return float(value_array.reshape(-1)[0])
    return None


def _prepare_single_qubit_scalar_sweep_batch(
    qubit,
    flux_origin,
    flux_offsets: list[np.ndarray],
    *,
    cache_key_prefix: tuple[object, ...],
    ejmax_to_use,
    junc_ratio_to_use,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[object, ...], ...]] | None:
    """Return batched scalar-flux sweep inputs when the fast path can avoid per-point updates."""
    flux_origin_scalar = _extract_single_qubit_scalar_sweep_value(flux_origin)
    if flux_origin_scalar is None:
        return None

    ejmax_array = np.asarray(ejmax_to_use, dtype=float)
    junc_ratio_array = np.asarray(junc_ratio_to_use, dtype=float)
    if ejmax_array.size != 1 or junc_ratio_array.size != 1:
        return None

    flux_values = np.empty(len(flux_offsets), dtype=float)
    for index, offset in enumerate(flux_offsets):
        offset_scalar = _extract_single_qubit_scalar_sweep_value(offset)
        if offset_scalar is None:
            return None
        flux_values[index] = flux_origin_scalar + offset_scalar

    transformed_flux_values = flux_values.reshape(-1, 1)
    Ej_values = qubit._Ejphi(
        ejmax_array.reshape(1, 1),
        transformed_flux_values.reshape(-1, 1, 1),
        junc_ratio_array.reshape(1, 1),
    )
    cache_keys = tuple(
        (*cache_key_prefix, ((1, 1), (float(Ej_scalar),)))
        for Ej_scalar in Ej_values.reshape(-1)
    )
    return transformed_flux_values, Ej_values, cache_keys


def _get_single_qubit_relative_energylevels(
    qubit,
    Ej,
    *,
    cache_key: tuple[object, ...] | None = None,
    cache_key_prefix: tuple[object, ...] | None = None,
) -> np.ndarray:
    """Return one exact-input relative spectrum for the fast single-qubit sweep path."""
    cache = _get_single_qubit_sweep_relative_energy_cache(qubit)
    if cache_key is None:
        cache_key = _get_single_qubit_sweep_relative_energy_cache_key(
            qubit,
            Ej,
            cache_key_prefix=cache_key_prefix,
        )
    relative_energylevels = cache.get(cache_key)
    if relative_energylevels is not None:
        cache.move_to_end(cache_key)
        return relative_energylevels

    # The sweep only needs the instantaneous spectrum, so avoid rebuilding/storing the
    # truncated operator views that the steady-state constructor path keeps on the object.
    relative_energylevels = _extract_relative_energylevels(
        qubit._generate_hamiltonian(qubit.Ec, qubit.El, Ej, transient=True)
    )
    cache[cache_key] = np.array(relative_energylevels, copy=True)
    if len(cache) > _SINGLE_QUBIT_SWEEP_RELATIVE_ENERGY_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return relative_energylevels


def _restore_fast_single_qubit_sweep_state(qubit, flux_origin, flux_transformed_origin) -> None:
    """Restore the public sweep inputs without rebuilding the steady-state qubit."""
    qubit._flux = np.array(flux_origin, copy=True)
    if flux_transformed_origin is not None:
        qubit._flux_transformed = np.array(flux_transformed_origin, copy=True)


def _build_single_qubit_sweep_result_generic(
    qubit,
    flux_origin,
    flux_offsets: list[np.ndarray],
    upper_level: float,
) -> SweepResult:
    """Build the single-qubit sweep result through the generic change-by-change path."""
    energy_series = {f'level_{ii}': [] for ii in range(1, upper_level + 1)}

    try:
        for offset in flux_offsets:
            qubit.change_para(flux=flux_origin + offset)
            for ii in range(1, upper_level + 1):
                energy_series[f'level_{ii}'].append(qubit.get_energylevel(ii) / 2 / pi)

        return SweepResult(
            sweep_parameter='flux_offset',
            sweep_values=flux_offsets,
            series=energy_series,
            metadata={
                'flux_origin': np.array(flux_origin, copy=True),
                'upper_level': upper_level,
            },
        )
    finally:
        qubit.change_para(flux=flux_origin)


def _build_single_qubit_sweep_result_fast(
    qubit,
    flux_origin,
    flux_offsets: list[np.ndarray],
    upper_level: float,
) -> SweepResult:
    """Build the sweep result through an energy-only single-qubit update path."""
    energy_series = {f'level_{ii}': [] for ii in range(1, upper_level + 1)}
    level_keys = tuple(energy_series)
    flux_transformed_origin = getattr(qubit, '_flux_transformed', None)
    cache_key_prefix = _get_single_qubit_sweep_relative_energy_cache_key_prefix(qubit)
    junc_ratio_to_use = (
        qubit._junc_ratio_transformed if hasattr(qubit, '_junc_ratio_transformed') else qubit._junc_ratio
    )
    ejmax_to_use = qubit._ParameterizedQubit__Ej0 if hasattr(qubit, '_ParameterizedQubit__Ej0') else qubit.Ejmax
    struct = getattr(qubit, '_ParameterizedQubit__struct', None)
    retain_nodes = getattr(qubit, 'SMatrix_retainNodes', None)
    scalar_batch = _prepare_single_qubit_scalar_sweep_batch(
        qubit,
        flux_origin,
        flux_offsets,
        cache_key_prefix=cache_key_prefix,
        ejmax_to_use=ejmax_to_use,
        junc_ratio_to_use=junc_ratio_to_use,
    )

    try:
        if scalar_batch is not None:
            transformed_flux_values, Ej_values, cache_keys = scalar_batch
            iterator = zip(transformed_flux_values, Ej_values, cache_keys)
        else:
            iterator = ()

        if scalar_batch is not None:
            for transformed_flux, Ej, cache_key in iterator:
                qubit._flux_transformed = np.array(transformed_flux, copy=True)
                relative_energylevels = _get_single_qubit_relative_energylevels(
                    qubit,
                    Ej,
                    cache_key=cache_key,
                )

                for level_index, level_key in enumerate(level_keys, start=1):
                    energy_series[level_key].append(relative_energylevels[level_index] / 2 / pi)
        else:
            for offset in flux_offsets:
                qubit._flux = qubit._normalize_flux_input(flux_origin + offset)
                qubit._flux_transformed = project_transformed_flux(
                    qubit._flux,
                    struct,
                    retain_nodes,
                )
                Ej = qubit._Ejphi(ejmax_to_use, qubit._flux_transformed, junc_ratio_to_use)
                relative_energylevels = _get_single_qubit_relative_energylevels(
                    qubit,
                    Ej,
                    cache_key_prefix=cache_key_prefix,
                )

                for level_index, level_key in enumerate(level_keys, start=1):
                    energy_series[level_key].append(relative_energylevels[level_index] / 2 / pi)

        return SweepResult(
            sweep_parameter='flux_offset',
            sweep_values=flux_offsets,
            series=energy_series,
            metadata={
                'flux_origin': np.array(flux_origin, copy=True),
                'upper_level': upper_level,
            },
        )
    finally:
        _restore_fast_single_qubit_sweep_state(
            qubit,
            flux_origin,
            flux_transformed_origin,
        )


def sweep_single_qubit_energy_vs_flux_base(
    qubit,
    flux_offsets: list[np.ndarray],
    upper_level: float = 2,
) -> SweepResult:
    """Return the structured result for a single-qubit energy-vs-flux sweep."""
    try:
        return sweep_single_qubit_energy_vs_flux_base_result(
            qubit,
            flux_offsets,
            upper_level=upper_level,
        )
    except ValueError as exc:
        print(str(exc))
        return 0


def sweep_single_qubit_energy_vs_flux_base_result(
    qubit,
    flux_offsets: list[np.ndarray],
    upper_level: float = 2,
) -> SweepResult:
    """Return the preferred structured result for a single-qubit energy-vs-flux sweep."""
    _validate_single_qubit_sweep_upper_level(qubit, upper_level)

    flux_origin = copy(qubit._flux)
    if _supports_fast_single_qubit_sweep(qubit):
        return _build_single_qubit_sweep_result_fast(
            qubit,
            flux_origin,
            flux_offsets,
            upper_level,
        )

    return _build_single_qubit_sweep_result_generic(
        qubit,
        flux_origin,
        flux_offsets,
        upper_level,
    )


def sweep_multi_qubit_energy_vs_flux_result(
    qubit,
    coupler_flux: list[float],
    qubits_flux: list[list[float]] | None = None,
    cal_state: list[list[int]] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    is_plot: bool = True,
) -> SweepResult:
    """Return the preferred structured result for a representative multi-qubit energy sweep."""
    flux_origin = copy(qubit._flux)
    standard_state = cal_product_state_list(cal_state, qubit._Nlevel)
    state_index = [qubit.find_state(state) for state in standard_state]
    energy_series = {f'|{state}>': [] for state in cal_state}

    try:
        if qubits_flux is None:
            for c_flux in coupler_flux:
                qubit._flux[2, 2] = c_flux
                qubit.change_para(flux=qubit._flux)
                for state, index in zip(cal_state, state_index):
                    energy_series[f'|{state}>'].append(qubit.get_energylevel(index) / 2 / pi)
        else:
            for c_flux in coupler_flux:
                for q1_flux, q2_flux in zip(qubits_flux[0], qubits_flux[1]):
                    qubit._flux[0, 1] = qubit._flux[1, 0] = q1_flux
                    qubit._flux[3, 4] = qubit._flux[4, 3] = q2_flux
                    qubit._flux[2, 2] = c_flux
                    qubit.change_para(flux=qubit._flux)
                    for state, index in zip(cal_state, state_index):
                        energy_series[f'|{state}>'].append(qubit.get_energylevel(index) / 2 / pi)

        result = SweepResult(
            sweep_parameter='coupler_flux',
            sweep_values=list(coupler_flux),
            series=energy_series,
            metadata={
                'qubits_flux': None if qubits_flux is None else [list(fluxes) for fluxes in qubits_flux]
            },
        )
        if is_plot:
            plot_multi_qubit_energy_vs_flux(coupler_flux, cal_state, result)
        return result
    finally:
        qubit._flux = flux_origin
        qubit.change_para(flux=flux_origin)


def sweep_multi_qubit_energy_vs_flux(
    qubit,
    coupler_flux: list[float],
    qubits_flux: list[list[float]] | None = None,
    cal_state: list[list[int]] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    is_plot: bool = True,
) -> SweepResult:
    """Return the structured result for representative multi-qubit energies across coupler flux bias points."""
    return sweep_multi_qubit_energy_vs_flux_result(
        qubit,
        coupler_flux,
        qubits_flux=qubits_flux,
        cal_state=cal_state,
        is_plot=is_plot,
    )


def sweep_multi_qubit_coupling_strength_vs_flux(
    qubit,
    coupler_flux: list[float],
    method: str = 'ES',
    is_plot: bool = True,
) -> CouplingResult:
    """Return the structured result for representative multi-qubit coupling strengths across coupler flux bias points."""
    return sweep_multi_qubit_coupling_strength_vs_flux_result(
        qubit,
        coupler_flux,
        method=method,
        is_plot=is_plot,
    )


def sweep_multi_qubit_coupling_strength_vs_flux_result(
    qubit,
    coupler_flux: list[float],
    method: str = 'ES',
    is_plot: bool = True,
) -> CouplingResult:
    """Return the preferred structured result for a representative coupling-strength sweep."""
    flux_origin = copy(qubit._flux)
    g_list = []

    try:
        for flux in coupler_flux:
            qubit._flux[2, 2] = flux
            qubit.change_para(flux=qubit._flux)
            g_list.append(qubit.get_qq_ecouple(method=method, is_print=False))

        result = CouplingResult(
            sweep_parameter='coupler_flux',
            sweep_values=list(coupler_flux),
            coupling_values=g_list,
            metadata={'method': method},
        )
        if is_plot:
            plot_multi_qubit_coupling_strength_vs_flux(coupler_flux, result)

        min_point = int(np.argmin(np.abs(result.coupling_values)))
        print(
            f'The turn-off point: {result.sweep_values[min_point]} '
            f'(g={result.coupling_values[min_point]})'
        )
        return result
    finally:
        qubit._flux = flux_origin
        qubit.change_para(flux=flux_origin)
