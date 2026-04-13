import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence import TphiResult
from pysuqu.decoherence.dequbit import Euler_gamma, ZNoiseDecoherence


class ZNoiseDecoherenceTphi2AnalyzerBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return ZNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_cal_tphi2_cal_keeps_existing_ramsey_formula(self):
        z_noise = self._construct()
        delay_list = np.array([1.5e-6, 2.0e-6, 2.5e-6])
        sensitivity = 1.23e11
        noise_1fcoef = 3.4e-18
        frequency = np.array([1.0e6, 2.0e6, 3.0e6])
        sensitivity_factor = sensitivity * z_noise.couple_term
        noise_f = noise_1fcoef * (sensitivity_factor * 2) ** 2
        coef = (
            0.75
            + Euler_gamma
            + np.log(2 * np.pi * np.min(frequency) * np.mean(delay_list))
        ) / np.pi

        z_noise.noise = SimpleNamespace(
            output_stage=SimpleNamespace(
                frequency=frequency,
                fit_result=SimpleNamespace(fit_diagnostics={'1f_coef': noise_1fcoef}),
            )
        )

        actual = z_noise.cal_tphi2(
            method='cal',
            experiment='Ramsey',
            delay_list=delay_list,
            sensitivity=sensitivity,
            is_print=False,
            is_plot=False,
        )

        self.assertIsInstance(actual, TphiResult)
        self.assertAlmostEqual(actual.value, np.sqrt(1 / (coef * noise_f)))
        self.assertEqual(actual.metadata['method'], 'cal')
        self.assertEqual(actual.metadata['experiment'], 'Ramsey')
        self.assertEqual(z_noise.tphi2, actual.value)

    def test_cal_tphi2_cal_can_delegate_through_explicit_z_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])

        class RecordingAnalyzer:
            def calculate_tphi2_cal(
                self,
                *,
                noise_output,
                sensitivity_factor,
                experiment,
                delay_list,
            ):
                analyzer_calls.append(
                    {
                        'noise_output': noise_output,
                        'sensitivity_factor': sensitivity_factor,
                        'experiment': experiment,
                        'delay_list': np.array(delay_list, copy=True),
                    }
                )
                return 4.56

        def z_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        z_noise = self._construct(z_analyzer_builder=z_analyzer_builder)

        with patch.object(z_noise, 'get_sensitivity_at_idle', return_value=1.23):
            actual = z_noise.cal_tphi2(
                method='cal',
                experiment='SpinEcho',
                delay_list=delay_list,
                idle_freq=4.8e9,
                is_print=False,
                is_plot=False,
            )

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]['couple_term'], z_noise.couple_term)
        self.assertIsInstance(actual, TphiResult)
        self.assertEqual(actual.value, 4.56)
        self.assertEqual(actual.metadata['method'], 'cal')
        self.assertEqual(actual.metadata['experiment'], 'SpinEcho')
        self.assertEqual(z_noise.tphi2, actual.value)
        self.assertEqual(len(analyzer_calls), 1)
        self.assertIs(analyzer_calls[0]['noise_output'], z_noise.noise.output_stage)
        self.assertEqual(analyzer_calls[0]['sensitivity_factor'], 1.23 * z_noise.couple_term)
        self.assertEqual(analyzer_calls[0]['experiment'], 'SpinEcho')
        np.testing.assert_array_equal(analyzer_calls[0]['delay_list'], delay_list)


if __name__ == '__main__':
    unittest.main()

