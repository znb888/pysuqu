import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence import ElectronicNoise
from pysuqu.funclib.noisemodel import (
    S_transmission,
    Sii_A2dBm,
    T2Sii_Double,
    T2Sii_Single,
)


class NoiseTransmissionTests(unittest.TestCase):
    @staticmethod
    def _manual_transmission(s_in, frequency_hz, temperatures_k, attenuation_db):
        result = s_in
        for temperature_k, loss_db in zip(temperatures_k, attenuation_db):
            transmission = 10 ** (-loss_db / 10)
            result = (
                transmission * result
                + (1 - transmission) * T2Sii_Double(temperature_k, frequency_hz)
            )
        return result

    def test_zero_db_stage_does_not_add_thermal_noise(self):
        frequency_hz = 5.2e9
        s_in = T2Sii_Double(35_000.0, frequency_hz)

        actual = S_transmission(
            s_in,
            frequency_hz,
            np.array([300.0]),
            np.array([0.0]),
        )

        self.assertEqual(actual, s_in)

    def test_chain_preserves_thermal_equilibrium(self):
        frequency_hz = 6.18e9
        equilibrium_temperature_k = 0.12
        temperatures_k = np.full(4, equilibrium_temperature_k)
        attenuation_db = np.array([3.0, 10.0, 0.0, 27.0])
        s_in = T2Sii_Double(equilibrium_temperature_k, frequency_hz)

        actual = S_transmission(
            s_in,
            frequency_hz,
            temperatures_k,
            attenuation_db,
        )

        self.assertAlmostEqual(actual, s_in, places=38)

    def test_large_final_attenuation_approaches_final_stage_noise(self):
        frequency_hz = 5.2e9
        temperatures_k = np.array([300.0, 0.02])
        attenuation_db = np.array([20.0, 300.0])
        s_in = T2Sii_Double(36_000.0, frequency_hz)

        actual = S_transmission(
            s_in,
            frequency_hz,
            temperatures_k,
            attenuation_db,
        )
        expected = T2Sii_Double(temperatures_k[-1], frequency_hz)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    def test_scalar_positive_and_negative_frequencies_match_manual_chain(self):
        temperatures_k = np.array([300.0, 50.0, 4.0, 0.02])
        attenuation_db = np.array([17.42, 3.27, 16.23, 24.21])

        for frequency_hz in (5.2e9, -5.2e9):
            with self.subTest(frequency_hz=frequency_hz):
                s_in = T2Sii_Double(36_000.0, frequency_hz)
                expected = self._manual_transmission(
                    s_in,
                    frequency_hz,
                    temperatures_k,
                    attenuation_db,
                )
                actual = S_transmission(
                    s_in,
                    frequency_hz,
                    temperatures_k,
                    attenuation_db,
                )

                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

    def test_frequency_array_matches_manual_chain(self):
        frequency_hz = np.array([1e4, 1e7, 5.2e9, 6.18e9])
        temperatures_k = np.array([300.0, 4.0, 0.02])
        attenuation_db = np.array([10.6, 13.73, 24.0])
        s_in = T2Sii_Double(4_500.0, frequency_hz)

        expected = self._manual_transmission(
            s_in,
            frequency_hz,
            temperatures_k,
            attenuation_db,
        )
        actual = S_transmission(
            s_in,
            frequency_hz,
            temperatures_k,
            attenuation_db,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

    def test_electronic_noise_uses_corrected_chain_for_both_input_modes(self):
        temperatures_k = np.array([300.0, 4.0, 0.02])
        attenuation_db = np.array([10.6, 13.73, 24.0])

        spectral_frequency_hz = np.logspace(4, 7, 64)
        spectral_single = T2Sii_Single(4_500.0, spectral_frequency_hz)
        with redirect_stdout(StringIO()):
            spectral_noise = ElectronicNoise(
                psd_freq=spectral_frequency_hz,
                psd_S=spectral_single,
                noise_type='constant',
                noise_prop='single',
                T_setup=temperatures_k,
                attenuation_setup=attenuation_db,
                is_spectral=True,
                is_print=False,
            )
        expected_spectral = self._manual_transmission(
            spectral_noise.input_stage.psd_double,
            spectral_frequency_hz,
            temperatures_k,
            attenuation_db,
        )
        np.testing.assert_allclose(
            spectral_noise.output_stage.psd_double,
            expected_spectral,
            rtol=1e-14,
            atol=0.0,
        )

        constant_frequency_hz = 5.2e9
        constant_single = T2Sii_Single(36_000.0, constant_frequency_hz)
        with redirect_stdout(StringIO()):
            constant_noise = ElectronicNoise(
                psd_freq=constant_frequency_hz,
                psd_S=Sii_A2dBm(constant_single),
                noise_prop='single',
                T_setup=temperatures_k,
                attenuation_setup=attenuation_db,
                is_spectral=False,
                is_print=False,
            )
        expected_constant = self._manual_transmission(
            constant_noise.input_stage.psd_double,
            constant_frequency_hz,
            temperatures_k,
            attenuation_db,
        )
        np.testing.assert_allclose(
            constant_noise.output_stage.psd_double,
            expected_constant,
            rtol=1e-14,
            atol=0.0,
        )


if __name__ == '__main__':
    unittest.main()
