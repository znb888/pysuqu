import unittest
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.noise import (
    fit_readout_thermal_photon_noise,
    readout_thermal_photon_noise,
)
from pysuqu.funclib.mathlib import temp2nbar
from pysuqu.funclib.noisemodel import thermal_photon_noise, thermal_photon_noise_fit


class ReadoutThermalNoiseBoundaryTests(unittest.TestCase):
    def test_readout_thermal_photon_noise_matches_existing_formula_surface(self):
        freq = np.array([1.0e3, 2.0e4, 5.0e5], dtype=float)
        result = readout_thermal_photon_noise(
            T_cav=55.0,
            kappa=6.0e6,
            chi=1.5e6,
            cavity_freq=6.5e9,
            psd_freq=freq,
            S0=3.0,
        )

        omega_noise = freq * 2 * np.pi
        kappa = 6.0e6 * 2 * np.pi
        chi = 1.5e6 * 2 * np.pi
        nbar = temp2nbar(55.0e-3, 6.5e9)
        eta = kappa**2 / (kappa**2 + 4 * chi**2)
        nbar_eff = nbar * eta
        expected = (
            2 * nbar_eff * (nbar_eff + 1) * (2 * chi) ** 2 * (2 * kappa)
            / (omega_noise**2 + kappa**2)
            + 3.0
        )

        np.testing.assert_allclose(result, expected)

    def test_fit_readout_thermal_photon_noise_reconstructs_fixed_parameter_surface(self):
        freq = np.logspace(3, 6, 64)
        expected_psd = readout_thermal_photon_noise(
            T_cav=48.0,
            kappa=5.5e6,
            chi=1.2e6,
            cavity_freq=6.5e9,
            psd_freq=freq,
            S0=2.5,
        )

        fit_result = fit_readout_thermal_photon_noise(
            psd_freq=freq,
            psd=expected_psd,
            kappa=5.5e6,
            chi=1.2e6,
            S0=2.5,
            bounds=([40.0], [60.0]),
            robust_fit=False,
        )

        self.assertAlmostEqual(fit_result["params"]["T_cav"], 48.0, places=6)
        np.testing.assert_allclose(fit_result["fitted_PSD"], expected_psd)

    def test_legacy_noisemodel_entry_points_delegate_to_decoherence_owned_helpers(self):
        sentinel_psd = np.array([1.0, 2.0, 3.0])
        sentinel_fit = {"params": {"T_cav": 40.0}}

        with patch(
            "pysuqu.decoherence.noise.readout_thermal_photon_noise",
            return_value=sentinel_psd,
        ) as noise_helper:
            result = thermal_photon_noise(
                T_cav=40.0,
                kappa=6.0e6,
                chi=1.5e6,
                cavity_freq=6.5e9,
                PSD_freq=np.array([1.0, 2.0, 3.0]),
                S0=0.0,
            )

        self.assertIs(result, sentinel_psd)
        noise_helper.assert_called_once()

        with patch(
            "pysuqu.decoherence.noise.fit_readout_thermal_photon_noise",
            return_value=sentinel_fit,
        ) as fit_helper:
            result = thermal_photon_noise_fit(
                PSD_freq=np.array([1.0, 2.0, 3.0]),
                PSD=np.array([4.0, 5.0, 6.0]),
            )

        self.assertIs(result, sentinel_fit)
        fit_helper.assert_called_once()


if __name__ == "__main__":
    unittest.main()
