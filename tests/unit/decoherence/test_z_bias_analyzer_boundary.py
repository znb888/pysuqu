import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import ZNoiseDecoherence
from pysuqu.qubit.base import Phi0


class ZNoiseDecoherenceBiasAnalyzerBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return ZNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_cal_bias_current_voltage_keeps_existing_numeric_formula(self):
        phi_fraction = 0.375
        attenuation_setup = np.array([3.0, 7.5, 11.25, 0.5, 0.25, 0.0])
        z_noise = self._construct(attenuation_setup=attenuation_setup)

        phi_bias = phi_fraction * Phi0
        chip_current_A = phi_bias / z_noise.couple_term / 2
        chip_current_uA = chip_current_A * 1e6
        chip_voltage_mV = chip_current_uA * 50 / 1000
        total_attenuation_dB = np.sum(attenuation_setup)
        chip_current_mA = chip_current_A * 1e3
        room_current_mA = chip_current_mA * 10 ** (total_attenuation_dB / 20)
        room_voltage_mV = room_current_mA * 50
        room_power_W = room_current_mA**2 * 50 / 2 / 1e3
        room_power_dBm = 10 * np.log10(room_power_W)

        actual = z_noise.cal_bias_current_voltage(phi_fraction=phi_fraction, is_print=False)

        self.assertAlmostEqual(actual['phi_bias'], phi_bias)
        self.assertAlmostEqual(actual['chip_current_uA'], chip_current_uA)
        self.assertAlmostEqual(actual['chip_voltage_mV'], chip_voltage_mV)
        self.assertAlmostEqual(actual['total_attenuation_dB'], total_attenuation_dB)
        self.assertAlmostEqual(actual['room_current_mA'], room_current_mA)
        self.assertAlmostEqual(actual['room_voltage_mV'], room_voltage_mV)
        self.assertAlmostEqual(actual['room_power_dBm'], room_power_dBm)

    def test_cal_bias_current_voltage_can_delegate_through_explicit_z_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []
        expected = {
            'phi_bias': 1.0,
            'chip_current_uA': 2.0,
            'chip_voltage_mV': 3.0,
            'total_attenuation_dB': 4.0,
            'room_current_mA': 5.0,
            'room_voltage_mV': 6.0,
            'room_power_dBm': 7.0,
        }

        class RecordingAnalyzer:
            def calculate_bias_current_voltage(self, *, phi_fraction, attenuation_setup):
                analyzer_calls.append(
                    {
                        'phi_fraction': phi_fraction,
                        'attenuation_setup': np.array(attenuation_setup, copy=True),
                    }
                )
                return dict(expected)

        def z_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        attenuation_setup = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        z_noise = self._construct(
            attenuation_setup=attenuation_setup,
            z_analyzer_builder=z_analyzer_builder,
        )

        actual = z_noise.cal_bias_current_voltage(phi_fraction=0.125, is_print=False)

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]['couple_term'], z_noise.couple_term)
        self.assertEqual(actual, expected)
        self.assertEqual(len(analyzer_calls), 1)
        self.assertEqual(analyzer_calls[0]['phi_fraction'], 0.125)
        np.testing.assert_array_equal(analyzer_calls[0]['attenuation_setup'], attenuation_setup)


if __name__ == '__main__':
    unittest.main()

