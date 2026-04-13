from copy import copy
from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .base import pi
from .types import CouplingResult, SensitivityResult


@dataclass(frozen=True)
class SingleQubitSpectrum:
    f01: float
    anharmonicity: float


def analyze_single_qubit_spectrum(qubit) -> SingleQubitSpectrum:
    """Return single-qubit transition metrics from the current eigensystem."""
    f01 = qubit.get_energylevel(1) / 2 / pi
    anharmonicity = qubit.get_energylevel(2) / 2 / pi - 2 * f01
    return SingleQubitSpectrum(f01=f01, anharmonicity=anharmonicity)


def analyze_multi_qubit_coupler_sensitivity(
    qubit,
    coupler_flux_point: float,
    method: str = 'numerical',
    flux_step: float = 1e-4,
    qubit_idx: int | None = None,
    qubit_fluxes: list[float] | None = None,
    is_print: bool = True,
    is_plot: bool = False,
) -> SensitivityResult:
    """Return the structured sensitivity result for multi-qubit helper callers."""
    if method == 'analytical':
        sensitivity = calculate_multi_qubit_sensitivity_analytical(
            qubit,
            coupler_flux_point,
            qubit_idx,
        )
    elif method == 'numerical':
        sensitivity = calculate_multi_qubit_sensitivity_numerical(
            qubit,
            coupler_flux_point,
            flux_step,
            qubit_idx,
            qubit_fluxes=qubit_fluxes,
        )
    else:
        raise ValueError(f"Invalid method: {method}")

    sensitivity_value = float(sensitivity) if hasattr(sensitivity, 'item') else float(sensitivity)
    result = SensitivityResult(
        coupler_flux_point=coupler_flux_point,
        sensitivity_value=sensitivity_value,
        metadata={
            'method': method,
            'flux_step': flux_step,
            'qubit_idx': qubit_idx,
            'qubit_fluxes': None if qubit_fluxes is None else list(qubit_fluxes),
        },
    )

    if is_print:
        qubit_name = _format_qubit_name(qubit_idx)
        print(
            f"{qubit_name} sensitivity @ Phi_c={coupler_flux_point:.4f} Phi0: "
            f"{sensitivity_value:.6f} GHz/Phi0 = {sensitivity_value*1e3:.3f} MHz/Phi0"
        )

    if is_plot:
        plot_multi_qubit_sensitivity_curve(
            qubit,
            coupler_flux_point,
            flux_step,
            qubit_idx,
            result,
        )

    return result


def analyze_multi_qubit_coupler_sensitivity_result(
    qubit,
    coupler_flux_point: float,
    method: str = 'numerical',
    flux_step: float = 1e-4,
    qubit_idx: int | None = None,
    qubit_fluxes: list[float] | None = None,
    is_print: bool = True,
    is_plot: bool = False,
) -> SensitivityResult:
    """Backward name retained for the structured sensitivity helper."""
    return analyze_multi_qubit_coupler_sensitivity(
        qubit,
        coupler_flux_point=coupler_flux_point,
        method=method,
        flux_step=flux_step,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
        is_print=is_print,
        is_plot=is_plot,
    )


def calculate_multi_qubit_sensitivity_numerical(
    qubit,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int | None,
    qubit_fluxes: list[float] | None = None,
) -> float:
    """Calculate sensitivity using central-difference probing."""
    phi_plus = coupler_flux_point + flux_step
    phi_minus = coupler_flux_point - flux_step

    if _is_single_target_multi_qubit(qubit):
        phi_plus = min(phi_plus, 0.5)
        phi_minus = max(phi_minus, 0.0)

    f_plus = get_multi_qubit_frequency_at_coupler_flux(
        qubit,
        phi_plus,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
    )
    f_minus = get_multi_qubit_frequency_at_coupler_flux(
        qubit,
        phi_minus,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
    )
    return (f_plus - f_minus) / (phi_plus - phi_minus)


def calculate_multi_qubit_sensitivity_analytical(
    qubit,
    coupler_flux_point: float,
    qubit_idx: int | None,
) -> float:
    """Calculate sensitivity using the coupler-displacement approximation."""
    g_qc = qubit.qc_g
    omega_q = _get_multi_qubit_probe_frequency(qubit, qubit_idx)
    omega_c = qubit.coupler_f01
    delta_c = omega_c - omega_q

    coupling_ratio = np.abs(g_qc / delta_c)
    if coupling_ratio > 0.1:
        warnings.warn(f"Strong coupling (g/Delta = {coupling_ratio:.3f} > 0.1)")

    d_omega_c_d_phi = calculate_multi_qubit_coupler_self_sensitivity(qubit, coupler_flux_point)
    return (g_qc**2 / delta_c**2) * d_omega_c_d_phi


def calculate_multi_qubit_coupler_self_sensitivity(qubit, coupler_flux: float) -> float:
    """Return the analytical coupler self-sensitivity at a flux point."""
    ec_ghz = qubit.Ec[1, 1]
    ej_max_ghz = qubit.Ejmax[1, 1]
    cos_term = np.cos(np.pi * coupler_flux)

    if np.abs(cos_term) < 1e-10:
        return 0.0

    sin_term = np.sin(np.pi * coupler_flux)
    ej = ej_max_ghz * np.abs(cos_term)
    dej_dphi = -ej_max_ghz * np.pi * sin_term * np.sign(cos_term)
    d_omega_dej = 2 * ec_ghz / np.sqrt(8 * ec_ghz * ej)
    return d_omega_dej * dej_dphi


def get_multi_qubit_frequency_at_coupler_flux(
    qubit,
    coupler_flux: float,
    qubit_idx: int | None = None,
    qubit_fluxes: list[float] | None = None,
    flux_offset: float = 0.0,
) -> float:
    """Probe the requested qubit frequency while restoring the original flux state."""
    original_flux = copy(qubit._flux)

    try:
        qubit._flux[2, 2] = coupler_flux + flux_offset
        if qubit_fluxes is not None:
            _apply_qubit_flux_overrides(qubit, qubit_fluxes)
        qubit.change_para(flux=qubit._flux)
        return _get_multi_qubit_probe_frequency(qubit, qubit_idx)
    finally:
        qubit._flux = original_flux
        qubit.change_para(flux=original_flux)


def find_multi_qubit_coupler_detune(
    g_list: list[float] | CouplingResult,
    flux_list: list[float] | None = None,
    coupler_strength: float | None = None,
) -> float:
    """Interpolate the coupler flux that reaches a requested coupling strength."""
    if coupler_strength is None:
        raise ValueError('coupler_strength is required.')

    if isinstance(g_list, CouplingResult):
        result_flux_list = list(g_list.sweep_values)
        if flux_list is not None and list(flux_list) != result_flux_list:
            raise ValueError('flux_list must match CouplingResult.sweep_values when both are provided.')
        flux_list = result_flux_list
        g_list = list(g_list.coupling_values)

    if flux_list is None:
        raise ValueError('flux_list is required when g_list is not a CouplingResult.')

    if len(g_list) != len(flux_list):
        raise ValueError('g_list and flux_list must have the same length.')
    if len(g_list) < 2:
        raise ValueError('At least two samples are required to interpolate target flux.')

    for sample_g, sample_flux in zip(g_list, flux_list):
        if np.isclose(coupler_strength, sample_g):
            print(f'Flux when g={coupler_strength}: {sample_flux} Phi0')
            return sample_flux

    for index in range(1, len(g_list)):
        g_start = g_list[index - 1]
        g_end = g_list[index]
        lower_bound = min(g_start, g_end)
        upper_bound = max(g_start, g_end)
        if lower_bound <= coupler_strength <= upper_bound:
            if np.isclose(g_end, g_start):
                raise ValueError('Adjacent g_list samples must differ to interpolate target flux.')

            k_coef = (flux_list[index] - flux_list[index - 1]) / (g_end - g_start)
            target_flux = flux_list[index] + k_coef * (coupler_strength - g_end)
            print(f'Flux when g={coupler_strength}: {target_flux} Phi0')
            return target_flux

    raise ValueError(
        f'coupler_strength={coupler_strength} is not bracketed by g_list; cannot interpolate target flux.'
    )


def plot_multi_qubit_sensitivity_curve(
    qubit,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int | None,
    sensitivity: SensitivityResult | float,
) -> None:
    """Plot the probed frequency curve and its tangent line."""
    if isinstance(sensitivity, SensitivityResult):
        sensitivity_value = sensitivity.sensitivity_value
    else:
        sensitivity_value = float(sensitivity)

    flux_range = np.linspace(
        max(0, coupler_flux_point - 5 * flux_step),
        min(0.5, coupler_flux_point + 5 * flux_step),
        20,
    )
    freq_list = [
        get_multi_qubit_frequency_at_coupler_flux(qubit, flux_value, qubit_idx=qubit_idx)
        for flux_value in flux_range
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(flux_range, freq_list, 'bo-', label='Frequency')

    midpoint = len(flux_range) // 2
    tangent_line = freq_list[midpoint] + sensitivity_value * (flux_range - coupler_flux_point)
    plt.plot(
        flux_range,
        tangent_line,
        'r--',
        label=f'Tangent (slope={sensitivity_value:.3f} GHz/Phi0)',
    )

    plt.xlabel('Coupler Flux (Phi/Phi0)')
    plt.ylabel('Qubit Frequency (GHz)')
    plt.title(f'Qubit Frequency vs Coupler Flux @ Phi_c={coupler_flux_point:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def _apply_qubit_flux_overrides(qubit, qubit_fluxes: list[float]) -> None:
    if len(qubit_fluxes) != 2:
        raise ValueError('qubit_fluxes must contain exactly two entries.')

    qubit._flux[0, 1] = qubit._flux[1, 0] = qubit_fluxes[0]
    qubit._flux[3, 4] = qubit._flux[4, 3] = qubit_fluxes[1]


def _format_qubit_name(qubit_idx: int | None) -> str:
    if qubit_idx in [0, 1]:
        return f"Qubit{qubit_idx + 1}"
    return "Qubit"


def _get_multi_qubit_probe_frequency(qubit, qubit_idx: int | None) -> float:
    if hasattr(qubit, 'qubit1_f01') and hasattr(qubit, 'qubit2_f01'):
        if qubit_idx == 0:
            return qubit.qubit1_f01
        if qubit_idx == 1:
            return qubit.qubit2_f01
        return (qubit.qubit1_f01 + qubit.qubit2_f01) / 2
    return qubit.qubit_f01


def _is_single_target_multi_qubit(qubit) -> bool:
    return not (hasattr(qubit, 'qubit1_f01') and hasattr(qubit, 'qubit2_f01'))
