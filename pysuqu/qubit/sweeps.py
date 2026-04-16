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
from .multi import (
    FGF1V1Coupling,
    _get_cached_fgf1v1_metric_state_sets,
    _get_cached_fgf1v1_overlap_basis_indices,
    _normalize_nlevel_cache_key,
)
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


def _supports_fast_multi_qubit_coupling_sweep(qubit, method: str) -> bool:
    """Return whether the sweep can use the FGF1V1 low-spectrum coupling fast path."""
    return (
        method == 'ES'
        and isinstance(qubit, FGF1V1Coupling)
        and getattr(qubit, '_cal_mode', None) == 'Eigen'
        and hasattr(qubit, '_qrcouple_term')
        and hasattr(qubit, 'Maxwellmat')
        and hasattr(qubit, '_ParameterizedQubit__struct')
        and hasattr(qubit, 'SMatrix_retainNodes')
    )


def _resolve_multi_qubit_coupling_sweep_solver_mode(qubit, method: str, solver_mode: str) -> str:
    """Resolve the requested coupling-sweep execution mode into one concrete path."""
    normalized_mode = str(solver_mode).lower()
    if normalized_mode not in {'auto', 'full', 'fast'}:
        raise ValueError("solver_mode must be one of {'auto', 'full', 'fast'}.")

    fast_supported = _supports_fast_multi_qubit_coupling_sweep(qubit, method)
    if normalized_mode == 'auto':
        return 'fast' if fast_supported else 'full'
    if normalized_mode == 'fast' and not fast_supported:
        raise ValueError(
            "solver_mode='fast' is only supported for FGF1V1Coupling with method='ES' in Eigen mode."
        )
    return normalized_mode


def _solve_low_spectrum_eigensystem(hamiltonian, eigvals: int):
    """Solve only the low spectrum needed by the FGF1V1 coupling sweep fast path."""
    matrix_shape = getattr(hamiltonian, 'shape', (eigvals, eigvals))
    dimension = int(matrix_shape[0]) if matrix_shape else int(eigvals)
    use_sparse = dimension > 128 and eigvals < dimension
    return hamiltonian.eigenstates(
        sparse=use_sparse,
        sort='low',
        eigvals=eigvals,
    )


def _calculate_fgf1v1_es_coupling_fast(
    qubit,
    target_flux,
    *,
    metric_states,
    qc_overlap_indices,
    qq_overlap_indices,
    junc_ratio_to_use,
    ejmax_to_use,
    struct,
    retain_nodes,
    eigvals,
) -> float:
    """Return one FGF1V1 ES coupling value without rebuilding the full steady-state object."""
    flux_transformed = project_transformed_flux(
        target_flux,
        struct,
        retain_nodes,
    )
    Ej = qubit._Ejphi(ejmax_to_use, flux_transformed, junc_ratio_to_use)
    hamiltonian = qubit._generate_hamiltonian(qubit.Ec, qubit.El, Ej, transient=True)
    eigenvalues, eigenstates = _solve_low_spectrum_eigensystem(hamiltonian, eigvals=eigvals)
    eigenstates = list(eigenstates)
    state_index = tuple(qubit.find_state_list(metric_states, state_space=eigenstates))
    relative_levels = np.asarray(eigenvalues, dtype=float) - float(np.asarray(eigenvalues, dtype=float)[0])

    qubit1_f01 = float(relative_levels[state_index[1]] / 2 / pi)
    qubit2_f01 = float(relative_levels[state_index[2]] / 2 / pi)
    coupler_f01 = float(relative_levels[state_index[3]] / 2 / pi)

    q1c_g = abs(hamiltonian[qc_overlap_indices[1], qc_overlap_indices[0]]) / 2 / pi
    q2c_g = abs(hamiltonian[qc_overlap_indices[1], qc_overlap_indices[2]]) / 2 / pi
    qc_g = float((q1c_g + q2c_g) / 2)
    qq_g = float(abs(hamiltonian[qq_overlap_indices[1], qq_overlap_indices[0]]) / 2 / pi)

    delta1 = qubit1_f01 - coupler_f01
    delta2 = qubit2_f01 - coupler_f01
    sum1 = qubit1_f01 + coupler_f01
    sum2 = qubit2_f01 + coupler_f01
    return float(qc_g * qc_g * (1 / delta1 + 1 / delta2 - 1 / sum1 - 1 / sum2) / 2 + qq_g)


def _build_multi_qubit_coupling_strength_vs_flux_result_fast(
    qubit,
    coupler_flux: list[float],
    method: str,
    is_plot: bool,
) -> CouplingResult:
    """Build one FGF1V1 coupling sweep through a low-spectrum transient Hamiltonian path."""
    flux_origin = np.array(qubit._flux, dtype=float, copy=True)
    baseline_coupler_flux = float(flux_origin[2, 2])
    baseline_coupling = None
    if any(np.isclose(flux, baseline_coupler_flux) for flux in coupler_flux):
        if hasattr(qubit, 'qq_geff'):
            baseline_coupling = float(qubit.qq_geff)
        else:
            baseline_coupling = float(qubit.get_qq_ecouple(method=method, is_print=False))

    if (
        not hasattr(qubit, '_junc_ratio_transformed')
        or getattr(qubit, '_junc_ratio_transformed_dirty', False)
    ):
        qubit._update_transformed_vars()

    nlevel_key = _normalize_nlevel_cache_key(qubit._Nlevel)
    metric_states, _, _ = _get_cached_fgf1v1_metric_state_sets(nlevel_key)
    qc_overlap_indices, qq_overlap_indices = _get_cached_fgf1v1_overlap_basis_indices(nlevel_key)
    struct = getattr(qubit, '_ParameterizedQubit__struct')
    retain_nodes = getattr(qubit, 'SMatrix_retainNodes')
    junc_ratio_to_use = (
        qubit._junc_ratio_transformed if hasattr(qubit, '_junc_ratio_transformed') else qubit._junc_ratio
    )
    ejmax_to_use = qubit._ParameterizedQubit__Ej0 if hasattr(qubit, '_ParameterizedQubit__Ej0') else qubit.Ejmax
    eigvals = max(len(metric_states), 4 * int(getattr(qubit, '_numQubits', 0)))

    g_list = []
    cached_couplings = {}
    if baseline_coupling is not None:
        cached_couplings[baseline_coupler_flux] = baseline_coupling

    for flux in coupler_flux:
        flux_key = float(flux)
        if flux_key in cached_couplings:
            g_list.append(cached_couplings[flux_key])
            continue

        target_flux = np.array(flux_origin, copy=True)
        target_flux[2, 2] = flux_key
        coupling = _calculate_fgf1v1_es_coupling_fast(
            qubit,
            target_flux,
            metric_states=metric_states,
            qc_overlap_indices=qc_overlap_indices,
            qq_overlap_indices=qq_overlap_indices,
            junc_ratio_to_use=junc_ratio_to_use,
            ejmax_to_use=ejmax_to_use,
            struct=struct,
            retain_nodes=retain_nodes,
            eigvals=eigvals,
        )
        cached_couplings[flux_key] = coupling
        g_list.append(coupling)

    result = CouplingResult(
        sweep_parameter='coupler_flux',
        sweep_values=list(coupler_flux),
        coupling_values=g_list,
        metadata={
            'method': method,
            'path': 'fgf1v1_low_spectrum_fast',
            'solver_mode': 'fast',
        },
    )
    if is_plot:
        plot_multi_qubit_coupling_strength_vs_flux(coupler_flux, result)

    min_point = int(np.argmin(np.abs(result.coupling_values)))
    print(
        f'The turn-off point: {result.sweep_values[min_point]} '
        f'(g={result.coupling_values[min_point]})'
    )
    return result


def _build_multi_qubit_coupling_strength_vs_flux_result_generic(
    qubit,
    coupler_flux: list[float],
    method: str,
    is_plot: bool,
) -> CouplingResult:
    """Build the coupling sweep through the legacy full-spectrum recompute path."""
    flux_origin = copy(qubit._flux)
    g_list = []
    baseline_coupler_flux = float(np.asarray(flux_origin, dtype=float)[2, 2])
    baseline_coupling = None
    if any(np.isclose(flux, baseline_coupler_flux) for flux in coupler_flux):
        baseline_coupling = qubit.get_qq_ecouple(method=method, is_print=False)

    try:
        for flux in coupler_flux:
            if baseline_coupling is not None and np.isclose(flux, baseline_coupler_flux):
                g_list.append(baseline_coupling)
                continue
            qubit._flux[2, 2] = flux
            qubit.change_para(flux=qubit._flux)
            g_list.append(qubit.get_qq_ecouple(method=method, is_print=False))

        result = CouplingResult(
            sweep_parameter='coupler_flux',
            sweep_values=list(coupler_flux),
            coupling_values=g_list,
            metadata={
                'method': method,
                'path': 'generic_full_spectrum',
                'solver_mode': 'full',
            },
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


def sweep_multi_qubit_coupling_strength_vs_flux(
    qubit,
    coupler_flux: list[float],
    method: str = 'ES',
    solver_mode: str = 'auto',
    is_plot: bool = True,
) -> CouplingResult:
    """Return the structured result for representative multi-qubit coupling strengths across coupler flux bias points."""
    return sweep_multi_qubit_coupling_strength_vs_flux_result(
        qubit,
        coupler_flux,
        method=method,
        solver_mode=solver_mode,
        is_plot=is_plot,
    )


def sweep_multi_qubit_coupling_strength_vs_flux_result(
    qubit,
    coupler_flux: list[float],
    method: str = 'ES',
    solver_mode: str = 'auto',
    is_plot: bool = True,
) -> CouplingResult:
    """Return the preferred structured result for a representative coupling-strength sweep."""
    resolved_solver_mode = _resolve_multi_qubit_coupling_sweep_solver_mode(
        qubit,
        method,
        solver_mode,
    )
    if resolved_solver_mode == 'fast':
        return _build_multi_qubit_coupling_strength_vs_flux_result_fast(
            qubit,
            coupler_flux,
            method=method,
            is_plot=is_plot,
        )
    return _build_multi_qubit_coupling_strength_vs_flux_result_generic(
        qubit,
        coupler_flux,
        method=method,
        is_plot=is_plot,
    )
