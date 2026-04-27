"""Display-formatting helpers for decoherence reports."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .results import BiasCurrentVoltageResult, XYCurrentVoltageResult
from ..qubit.base import Phi0


def _flatten_numeric(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)

    return arr.ravel()


def _format_frequency_hz(value: float) -> str:
    if not np.isfinite(value):
        return str(value)

    abs_value = abs(value)
    if abs_value >= 1e9:
        return f'{value / 1e9:.3f} GHz'
    if abs_value >= 1e6:
        return f'{value / 1e6:.3f} MHz'
    if abs_value >= 1e3:
        return f'{value / 1e3:.3f} kHz'
    return f'{value:.3f} Hz'


def _format_temperature_k(value: float) -> str:
    if not np.isfinite(value):
        return str(value)

    abs_value = abs(value)
    if abs_value >= 1.0:
        return f'{value:.3f} K'
    if abs_value >= 1e-3:
        return f'{value * 1e3:.3f} mK'
    if abs_value >= 1e-6:
        return f'{value * 1e6:.3f} uK'
    if abs_value >= 1e-9:
        return f'{value * 1e9:.3f} nK'
    return f'{value:.3e} K'


def _format_time_seconds(value: float) -> str:
    if np.isinf(value):
        return 'inf'
    if np.isnan(value):
        return 'nan'

    abs_value = abs(value)
    if abs_value >= 1.0:
        return f'{value:.6f} s'
    if abs_value >= 1e-3:
        return f'{value * 1e3:.3f} ms'
    if abs_value >= 1e-6:
        return f'{value * 1e6:.3f} us'
    if abs_value >= 1e-9:
        return f'{value * 1e9:.3f} ns'
    return f'{value * 1e12:.3f} ps'


def _format_probability(value: float) -> str:
    if not np.isfinite(value):
        return str(value)

    percent = value * 100
    if abs(percent) >= 0.01:
        return f'{percent:.3f}%'

    return f'{value:.3e}'


def _format_summary(value, formatter) -> str:
    flattened = _flatten_numeric(value)
    finite = flattened[np.isfinite(flattened)]

    if finite.size == 0:
        return formatter(float(flattened[0]))

    if finite.size == 1 or np.allclose(finite, finite[0], rtol=1e-9, atol=0.0):
        return formatter(float(finite[0]))

    center = float(np.median(finite))
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    return (
        f'{formatter(center)} '
        f'(median, range {formatter(lower)} to {formatter(upper)})'
    )


def format_electronic_noise_report(
    *,
    noise_type: str,
    noise_prop: str,
    is_spectral: bool,
    input_stage,
    output_stage,
    attenuation_setup,
) -> tuple[str, ...]:
    """Build the stable electronic-noise summary report."""
    total_attenuation_db = float(np.sum(attenuation_setup))
    mode = 'spectral' if is_spectral else 'constant'
    stage_list = ', '.join(f'{float(stage):.2f}' for stage in np.ravel(attenuation_setup))

    return (
        f'--- Electronic Noise Summary ({mode}, {noise_type}, {noise_prop}-sided input) ---',
        f'Input white noise PSD: {_format_summary(input_stage.white_noise, lambda x: f"{x:.3e} A^2/Hz")} @ {_format_summary(input_stage.white_ref_freq, _format_frequency_hz)}',
        f'Input white noise temperature: {_format_summary(input_stage.white_noise_temperature, _format_temperature_k)}',
        f'Output white noise PSD: {_format_summary(output_stage.white_noise, lambda x: f"{x:.3e} A^2/Hz")} @ {_format_summary(output_stage.white_ref_freq, _format_frequency_hz)}',
        f'Output white noise temperature: {_format_summary(output_stage.white_noise_temperature, _format_temperature_k)}',
        f'Total attenuation: {total_attenuation_db:.2f} dB ({stage_list} dB by stage)',
    )


def format_z_tphi1_report(
    *,
    idle_freq: float | None,
    sensitivity: float,
    couple_term: float,
    noise_output,
    tphi1: float,
) -> tuple[str, ...]:
    """Build the Z-noise Tphi1 summary report."""
    idle_label = (
        'current operating point'
        if idle_freq is None
        else _format_frequency_hz(float(idle_freq))
    )
    return (
        '--- Z Noise Tphi1 ---',
        f'Idle frequency: {idle_label}',
        f'Output white noise PSD: {_format_summary(noise_output.white_noise, lambda x: f"{x:.3e} A^2/Hz")} @ {_format_summary(noise_output.white_ref_freq, _format_frequency_hz)}',
        f'Output white noise temperature: {_format_summary(noise_output.white_noise_temperature, _format_temperature_k)}',
        f'Angular flux sensitivity: {sensitivity:.3e} rad/s/Wb, mutual inductance: {couple_term:.3e} H',
        f'Tphi1: {_format_time_seconds(tphi1)}',
    )


def format_coupler_tphi1_report(
    *,
    coupler_flux_point: float | None,
    qubit_idx: int | None,
    qubit_fluxes,
    sensitivity_ghz_per_phi0: float,
    sensitivity_rad_per_wb: float,
    couple_term: float,
    noise_output,
    tphi1: float,
) -> tuple[str, ...]:
    """Build the coupler-flux-noise Tphi1 summary report."""
    flux_label = (
        'provided sensitivity'
        if coupler_flux_point is None
        else f'{float(coupler_flux_point):.6f} Phi0'
    )
    target_label = (
        'average qubit frequency'
        if qubit_idx is None
        else f'Qubit{int(qubit_idx) + 1}'
    )
    lines = [
        '--- Coupler Flux Noise Tphi1 ---',
        f'Coupler flux: {flux_label}',
        f'Target qubit: {target_label}',
    ]
    if qubit_fluxes is not None:
        flux_values = ', '.join(f'{float(value):.6f}' for value in qubit_fluxes)
        lines.append(f'Qubit flux overrides: [{flux_values}] Phi0')

    lines.extend(
        [
            f'Output white noise PSD: {_format_summary(noise_output.white_noise, lambda x: f"{x:.3e} A^2/Hz")} @ {_format_summary(noise_output.white_ref_freq, _format_frequency_hz)}',
            f'Output white noise temperature: {_format_summary(noise_output.white_noise_temperature, _format_temperature_k)}',
            f'Frequency sensitivity: {sensitivity_ghz_per_phi0:.6f} GHz/Phi0 = {sensitivity_ghz_per_phi0 * 1e3:.3f} MHz/Phi0',
            f'Angular flux sensitivity: {sensitivity_rad_per_wb:.3e} rad/s/Wb, mutual inductance: {couple_term:.3e} H',
            f'Tphi1: {_format_time_seconds(tphi1)}',
        ]
    )
    return tuple(lines)


def format_z_tphi2_report(
    *,
    method: str,
    experiment: str,
    idle_freq: float | None,
    sensitivity_factor: float,
    noise_output,
    tphi2: float,
    fit_diagnostics: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Build the Z-noise Tphi2 summary report."""
    idle_label = (
        'current operating point'
        if idle_freq is None
        else _format_frequency_hz(float(idle_freq))
    )
    lines = [
        f'--- Z Noise Tphi2 ({method}, {experiment}) ---',
        f'Idle frequency: {idle_label}',
        f'Sensitivity x coupling: {sensitivity_factor:.3e}',
        f'Output white noise temperature: {_format_summary(noise_output.white_noise_temperature, _format_temperature_k)}',
    ]

    fit_result = getattr(noise_output, 'fit_result', None)
    if fit_result is not None:
        one_over_f = fit_result.fit_diagnostics.get('1f_coef')
        corner_freq = fit_result.fit_diagnostics.get('corner_freq')
        if one_over_f is not None and corner_freq is not None:
            lines.append(
                'Output fit: '
                f'1/f coef {float(one_over_f):.3e}, corner {_format_frequency_hz(float(corner_freq))}'
            )

    if fit_diagnostics is not None and 'tphi1' in fit_diagnostics:
        tphi1_fit = float(fit_diagnostics['tphi1'])
        tphi1_error = fit_diagnostics.get('tphi1_fiterror')
        if tphi1_error is not None:
            lines.append(
                f'Fit Tphi1: {_format_time_seconds(tphi1_fit)} +/- {_format_time_seconds(float(tphi1_error))}'
            )
        else:
            lines.append(f'Fit Tphi1: {_format_time_seconds(tphi1_fit)}')

    tphi2_error = None if fit_diagnostics is None else fit_diagnostics.get('fit_error')
    if tphi2_error is None:
        lines.append(f'Tphi2: {_format_time_seconds(tphi2)}')
    else:
        lines.append(
            f'Tphi2: {_format_time_seconds(tphi2)} +/- {_format_time_seconds(float(tphi2_error))}'
        )

    segments = {} if fit_diagnostics is None else fit_diagnostics.get('segments', {})
    for label, segment in segments.items():
        if not isinstance(segment, Mapping):
            continue

        popt = segment.get('popt')
        if popt is None:
            continue

        popt_array = np.asarray(popt, dtype=float).ravel()
        if popt_array.size < 2:
            continue

        lines.append(f'Segment {label}: Tphi2 {_format_time_seconds(float(popt_array[1]))}')

    return tuple(lines)


def format_xy_t1_report(
    *,
    qubit_freq: float,
    noise_output,
    gamma_up: float,
    gamma_down: float,
    t1: float,
) -> tuple[str, ...]:
    """Build the XY-noise T1 summary report."""
    return (
        '--- XY Noise T1 ---',
        f'Qubit frequency: {_format_frequency_hz(qubit_freq)}',
        f'Output white noise temperature: {_format_summary(noise_output.white_noise_temperature, _format_temperature_k)}',
        f'Gamma_up: {gamma_up:.3e} 1/s, Gamma_down: {gamma_down:.3e} 1/s',
        f'T1: {_format_time_seconds(t1)}',
    )


def format_xy_thermal_excitation_report(
    *,
    measured_t1_us: float | None,
    thermal_excitation: float,
    thermal_excitation_onlyxy: float,
) -> tuple[str, ...]:
    """Build the XY thermal-excitation summary report."""
    t1_label = 'use XY-limited rates only' if measured_t1_us is None else f'{measured_t1_us:.6f} us'
    return (
        '--- XY Thermal Excitation ---',
        f'T1 used for total excitation estimate: {t1_label}',
        f'Thermal excitation probability: {_format_probability(thermal_excitation)}',
        f'XY-only thermal excitation floor: {_format_probability(thermal_excitation_onlyxy)}',
    )


def format_readout_nbar_report(
    *,
    read_freq: float,
    noise_output,
    n_bar: float,
) -> tuple[str, ...]:
    """Build the readout thermal-photon number summary report."""
    return (
        '--- Readout Thermal Population ---',
        f'Readout frequency: {_format_frequency_hz(read_freq)}',
        f'Output white noise temperature: {_format_summary(noise_output.white_noise_temperature, _format_temperature_k)}',
        f'Estimated thermal photon number n_bar: {n_bar:.3e}',
    )


def format_readout_tphi_report(
    *,
    method: str,
    experiment: str,
    read_freq: float,
    n_bar: float,
    kappa_hz: float,
    chi_hz: float,
    tphi: float,
    fit_diagnostics: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Build the readout-induced Tphi summary report."""
    lines = [
        f'--- Readout-Induced Tphi ({method}, {experiment}) ---',
        f'Readout frequency: {_format_frequency_hz(read_freq)}',
        f'n_bar: {n_bar:.6e}, kappa: {_format_frequency_hz(kappa_hz)}, chi: {_format_frequency_hz(chi_hz)}',
    ]

    fit_error = None if fit_diagnostics is None else fit_diagnostics.get('fit_error')
    if fit_error is None:
        lines.append(f'Readout Tphi: {_format_time_seconds(tphi)}')
    else:
        lines.append(
            f'Readout Tphi: {_format_time_seconds(tphi)} +/- {_format_time_seconds(float(fit_error))}'
        )

    if fit_diagnostics is not None and 'tphi2' in fit_diagnostics:
        tphi2_error = fit_diagnostics.get('tphi2_fit_error')
        if tphi2_error is None:
            lines.append(f'Background Tphi2: {_format_time_seconds(float(fit_diagnostics["tphi2"]))}')
        else:
            lines.append(
                'Background Tphi2: '
                f'{_format_time_seconds(float(fit_diagnostics["tphi2"]))} '
                f'+/- {_format_time_seconds(float(tphi2_error))}'
            )

    return tuple(lines)


def format_bias_current_voltage_report(
    *,
    phi_fraction: float,
    results: BiasCurrentVoltageResult,
) -> tuple[str, ...]:
    """Build the stable bias current/voltage display report."""
    phi_bias = results['phi_bias']
    return (
        f'------- Bias Current/Voltage Calculation (phi_fraction={phi_fraction}) -------',
        f'Bias flux: {phi_bias / Phi0:.3f} Phi0 = {phi_bias / Phi0 * 1e3:.3f} mPhi0',
        f'Chip end: {results["chip_current_uA"]:.3f} uA, {results["chip_voltage_mV"]:.3f} mV',
        f'Total attenuation: {results["total_attenuation_dB"]:.2f} dB',
        f'Room end: {results["room_current_mA"]:.3f} mA, {results["room_voltage_mV"]:.3f} mV, {results["room_power_dBm"]:.2f} dBm',
    )


def format_xy_current_voltage_report(
    *,
    phi_fraction: float,
    results: XYCurrentVoltageResult,
) -> tuple[str, ...]:
    """Build the stable XY control-line current/voltage display report."""
    phi_bias = results['phi_bias']
    return (
        f'--- XY Control Line Current/Voltage Calculation (phi_fraction={phi_fraction:.6f}) ---',
        f'Bias flux: {phi_bias / Phi0:.6f} Phi0 = {phi_bias / Phi0 * 1e3:.6f} mPhi0',
        f'Chip end: {results["chip_current_uA"]:.3f} uA, {results["chip_voltage_uV"]:.3f} uV, {results["chip_power_dBm"]:.2f} dBm',
        f'Total attenuation: {results["total_attenuation_dB"]:.2f} dB',
        f'Room end: {results["room_current_mA"]:.6f} mA, {results["room_voltage_mV"]:.3f} mV, {results["room_power_dBm"]:.2f} dBm',
    )
