"""Bounded analysis helpers for the decoherence refactor."""

from __future__ import annotations

import numpy as np

from .electronics import Sii_D2S, T2Sii_Double
from .results import BiasCurrentVoltageResult, XYCurrentVoltageResult
from ..funclib.mathlib import temp2nbar
from ..qubit.base import Phi0

EULER_GAMMA = 0.577216


class ZDephasingAnalyzer:
    """Core Z-side dephasing analysis used behind the facade."""

    def __init__(self, *, couple_term: float):
        self.couple_term = couple_term

    def calculate_tphi1(self, *, noise_output, sensitivity: float) -> float:
        sw = Sii_D2S(noise_output.white_noise, noise_output.white_ref_freq) * (
            2 * sensitivity * self.couple_term
        ) ** 2
        return 4 / sw if sw > 0 else np.inf

    def calculate_bias_current_voltage(
        self,
        *,
        phi_fraction: float,
        attenuation_setup: np.ndarray,
    ) -> BiasCurrentVoltageResult:
        phi_bias = phi_fraction * Phi0

        chip_current_A = phi_bias / self.couple_term / 2
        chip_current_uA = chip_current_A * 1e6
        chip_voltage_mV = chip_current_uA * 50 / 1000

        total_attenuation_dB = np.sum(attenuation_setup)

        chip_current_mA = chip_current_A * 1e3
        room_current_mA = chip_current_mA * 10 ** (total_attenuation_dB / 20)
        room_voltage_mV = room_current_mA * 50

        room_power_W = room_current_mA**2 * 50 / 2 / 1e3
        room_power_dBm = 10 * np.log10(room_power_W)

        return {
            'phi_bias': phi_bias,
            'chip_current_uA': chip_current_uA,
            'chip_voltage_mV': chip_voltage_mV,
            'total_attenuation_dB': total_attenuation_dB,
            'room_current_mA': room_current_mA,
            'room_voltage_mV': room_voltage_mV,
            'room_power_dBm': room_power_dBm,
        }

    def calculate_tphi2_cal(
        self,
        *,
        noise_output,
        sensitivity_factor: float,
        experiment: str,
        delay_list: np.ndarray,
    ) -> float:
        if noise_output.fit_result is None:
            raise ValueError("Analytical 'cal' method requires fitted 1/f noise diagnostics.")

        noise_1fcoef = noise_output.fit_result.fit_diagnostics['1f_coef']
        noise_f = noise_1fcoef * (sensitivity_factor * 2) ** 2

        if experiment == 'SpinEcho':
            return np.sqrt(1 / (np.log(2) * noise_f))

        if experiment == 'Ramsey':
            freq_c = np.min(noise_output.frequency)
            mean_delay = np.mean(delay_list)
            coef = (0.75 + EULER_GAMMA + np.log(2 * np.pi * freq_c * mean_delay)) / np.pi
            return np.sqrt(1 / (coef * noise_f))

        raise ValueError(f"Experiment {experiment} not supported for analytical calc.")


class XYRelaxationAnalyzer:
    """Core XY-side relaxation analysis used behind the facade."""

    def __init__(self, *, couple_term: float):
        self.couple_term = couple_term

    def _calculate_gamma(self, spectral_density: float, *, Ej: float, Ec: float) -> float:
        coupling = 2e9 * np.pi * self.couple_term * Ej * (2 * Ec / Ej) ** 0.25 / Phi0
        return spectral_density * coupling**2

    def calculate_t1(
        self,
        *,
        noise_output,
        qubit_freq: float,
        Ej: float,
        Ec: float,
    ) -> dict[str, float]:
        sxy_positive = T2Sii_Double(noise_output.white_noise_temperature, f=qubit_freq)
        sxy_negative = T2Sii_Double(noise_output.white_noise_temperature, f=-qubit_freq)
        gamma_up = self._calculate_gamma(sxy_negative, Ej=Ej, Ec=Ec)
        gamma_down = self._calculate_gamma(sxy_positive, Ej=Ej, Ec=Ec)
        total_gamma = gamma_up + gamma_down
        return {
            'gamma_up': gamma_up,
            'gamma_down': gamma_down,
            't1': 1 / total_gamma if total_gamma > 0 else np.inf,
        }

    def calculate_thermal_excitation(
        self,
        *,
        gamma_up: float,
        gamma_down: float,
        t1_us: float | None,
    ) -> tuple[float, float]:
        gamma_down_actual = 1 / (t1_us * 1e-6) if t1_us is not None else 1 / 100e-6
        return (
            gamma_up / (gamma_up + gamma_down_actual),
            gamma_up / (gamma_up + gamma_down),
        )

    def calculate_xy_current_voltage(
        self,
        *,
        phi_fraction: float,
        attenuation_setup: np.ndarray,
    ) -> XYCurrentVoltageResult:
        phi_bias = phi_fraction * Phi0

        chip_current_A = phi_bias / self.couple_term / 2
        chip_current_uA = chip_current_A * 1e6
        chip_voltage_uV = chip_current_uA * 50
        chip_power_W = chip_current_A**2 * 50
        chip_power_dBm = 10 * np.log10(chip_power_W / 1e-3)

        total_attenuation_dB = np.sum(attenuation_setup)

        chip_current_mA = chip_current_A * 1e3
        room_current_mA = chip_current_mA * 10 ** (total_attenuation_dB / 20)
        room_voltage_mV = room_current_mA * 50

        room_power_W = room_current_mA**2 * 50 / 2 / 1e3
        room_power_dBm = 10 * np.log10(room_power_W)

        return {
            'phi_bias': phi_bias,
            'chip_current_uA': chip_current_uA,
            'chip_voltage_uV': chip_voltage_uV,
            'chip_power_dBm': chip_power_dBm,
            'total_attenuation_dB': total_attenuation_dB,
            'room_current_mA': room_current_mA,
            'room_voltage_mV': room_voltage_mV,
            'room_power_dBm': room_power_dBm,
        }


class ReadoutCavityAnalyzer:
    """Core readout-cavity analysis used behind the R-side facade."""

    def __init__(self, *, couple_term: float):
        self.couple_term = couple_term

    def calculate_nbar(self, *, noise_output, read_freq: float) -> float:
        return temp2nbar(noise_output.white_noise_temperature, read_freq)

    def calculate_tphi_cal(self, *, n_bar: float, kappa: float, chi: float) -> float:
        eta = kappa**2 / (kappa**2 + 4 * chi**2)
        nbar_th = n_bar * eta
        gamma_phi = nbar_th * (nbar_th + 1) * 4 * chi**2 / kappa
        return 1.0 / gamma_phi if gamma_phi > 1e-15 else np.inf

    @staticmethod
    def calculate_cpmg_integral(*, tau: np.ndarray, N: int, kappa: float, len_pi: float) -> float:
        tau_in = np.atleast_1d(tau)
        num_time_points = tau_in.shape[0]

        num_events = 2 * N + 2
        t = np.zeros((num_events, num_time_points))
        c = np.zeros(num_events)

        c[0] = 1.0
        t[0, :] = 0.0

        j_indices = np.arange(1, N + 1)
        delta_j = (2 * j_indices - 1) / N
        coeffs_j = (-1.0) ** j_indices

        c[1 : N + 1] = coeffs_j
        c[N + 1 : 2 * N + 1] = coeffs_j

        pulse_centers = np.outer(delta_j, tau_in)

        t[1 : N + 1, :] = pulse_centers - len_pi / 2.0
        t[N + 1 : 2 * N + 1, :] = pulse_centers + len_pi / 2.0

        c[-1] = (-1.0) ** (N + 1)
        t[-1, :] = tau_in
        dt_tensor = np.abs(t[:, None, :] - t[None, :, :])
        kernel_tensor = kappa * dt_tensor + np.exp(-kappa * dt_tensor)

        c_matrix = np.outer(c, c)
        c_tensor = c_matrix[:, :, np.newaxis]

        total_sum = np.sum(c_tensor * kernel_tensor, axis=(0, 1))

        prefactor = -1.0 / (8.0 * (kappa * tau_in) ** 2)
        result = prefactor * total_sum

        if np.ndim(tau) == 0:
            return result.item()

        return result

    @staticmethod
    def _calculate_thermal_photon_number(*, n_bar: float, kappa: float, chi: float) -> float:
        eta = kappa**2 / (kappa**2 + 4 * chi**2)
        return n_bar * eta

    def calculate_readcavity_psd(
        self,
        *,
        n_bar: float,
        kappa: float,
        chi: float,
        noise_freq: np.ndarray | None = None,
        read_freq: float = 6.5e9,
    ) -> np.ndarray:
        del read_freq

        if noise_freq is None:
            noise_freq = np.logspace(-2, np.log10(kappa / np.pi), 100)

        nbar_th = self._calculate_thermal_photon_number(n_bar=n_bar, kappa=kappa, chi=chi)
        return (
            2
            * nbar_th
            * (nbar_th + 1)
            * (2 * chi) ** 2
            * (2 * kappa)
            / (noise_freq**2 + kappa**2)
        )

    def calculate_read_dephase(
        self,
        *,
        n_bar: float,
        kappa: float,
        chi: float,
        experiment: str = "Ramsey",
        read_freq: float = 6.5e9,
        delay_list: np.ndarray = np.linspace(10, 10e3, 100) * 1e-9,
        N: int = 100,
        len_pi: float = 100e-9,
    ) -> np.ndarray:
        del read_freq

        nbar_th = self._calculate_thermal_photon_number(n_bar=n_bar, kappa=kappa, chi=chi)
        if experiment == "Ramsey":
            dfactor = (
                8
                * chi**2
                * nbar_th
                * (nbar_th + 1)
                * (kappa * delay_list - 1 + np.exp(-kappa * delay_list))
                / (kappa**2)
            )
            return np.exp(-dfactor / 2)

        if experiment == "SpinEcho":
            dfactor = (
                8
                * chi**2
                * nbar_th
                * (nbar_th + 1)
                * (kappa * delay_list - 3 + 4 * np.exp(-kappa * delay_list / 2) - np.exp(-kappa * delay_list))
                / (kappa**2)
            )
            return np.exp(-dfactor / 2)

        if experiment == "CPMG":
            dfactor = self.calculate_cpmg_integral(tau=delay_list, N=N, kappa=kappa, len_pi=len_pi)
            return np.exp(-dfactor * 4 * chi**2 * nbar_th * (nbar_th + 1))

        raise ValueError(f"Unknown experiment type: {experiment}")


__all__ = ['ReadoutCavityAnalyzer', 'ZDephasingAnalyzer', 'XYRelaxationAnalyzer']
