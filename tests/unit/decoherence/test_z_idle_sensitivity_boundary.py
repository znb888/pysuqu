import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import ZNoiseDecoherence
from pysuqu.qubit.base import Phi0


class ZNoiseDecoherenceIdleSensitivityBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return ZNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_get_sensitivity_at_idle_without_idle_freq_uses_cached_operating_point_value(self):
        z_noise = self._construct()
        z_noise.qubit_sensibility = 9.87
        z_noise.qubit.calculate_sensitivity_at_detuning = Mock(
            side_effect=AssertionError('calculate_sensitivity_at_detuning should not be called')
        )

        actual = z_noise.get_sensitivity_at_idle()

        self.assertEqual(actual, 9.87)

    def test_get_sensitivity_at_idle_detuned_path_uses_ghz_detuning_and_flux_conversion(self):
        z_noise = self._construct(qubit_freq=5.3e9)
        idle_freq = 4.82e9
        raw_sensitivity = 0.021
        z_noise.qubit.calculate_sensitivity_at_detuning = Mock(return_value=(raw_sensitivity, idle_freq / 1e9))

        actual = z_noise.get_sensitivity_at_idle(idle_freq=idle_freq)

        expected_detuning = (z_noise.qubit_freq - idle_freq) / 1e9
        expected_sensitivity = raw_sensitivity * 2e9 * np.pi / Phi0
        args, kwargs = z_noise.qubit.calculate_sensitivity_at_detuning.call_args
        self.assertAlmostEqual(args[0], expected_detuning)
        self.assertEqual(kwargs, {'mode': 'brief'})
        self.assertAlmostEqual(actual, expected_sensitivity)


if __name__ == '__main__':
    unittest.main()

