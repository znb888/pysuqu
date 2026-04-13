from copy import copy

import numpy as np

from ..funclib import cal_product_state_list
from .plotting import (
    plot_multi_qubit_coupling_strength_vs_flux,
    plot_multi_qubit_energy_vs_flux,
)
from .base import pi
from .types import CouplingResult, SweepResult


def _validate_single_qubit_sweep_upper_level(qubit, upper_level):
    if isinstance(qubit._Nlevel, (list, tuple, np.ndarray)):
        nlevel_val = qubit._Nlevel[0]
    else:
        nlevel_val = qubit._Nlevel

    if upper_level > nlevel_val:
        raise ValueError('Energy list out of range! Should lower than truncate energy level! ')


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
