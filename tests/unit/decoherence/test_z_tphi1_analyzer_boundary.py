import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import ZNoiseDecoherence
from pysuqu.decoherence.electronics import Sii_D2S
from pysuqu.qubit.base import Phi0


class ZNoiseDecoherenceTphi1AnalyzerBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return ZNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_cal_tphi1_keeps_existing_white_noise_formula(self):
        z_noise = self._construct()
        white_noise = 2.5e-18
        white_ref_freq = 11.0
        raw_sensitivity = 0.02
        expected_sensitivity = raw_sensitivity * 2e9 * np.pi / Phi0
        expected_sw = Sii_D2S(white_noise, white_ref_freq) * (
            2 * expected_sensitivity * z_noise.couple_term
        ) ** 2

        z_noise.noise = SimpleNamespace(
            output_stage=SimpleNamespace(
                white_noise=white_noise,
                white_ref_freq=white_ref_freq,
            )
        )

        actual = z_noise.cal_tphi1(sensitivity=raw_sensitivity, is_print=False)

        self.assertAlmostEqual(actual, 4 / expected_sw)
        self.assertEqual(z_noise.tphi1, actual)

    def test_cal_tphi1_accepts_rad_per_wb_sensitivity_without_conversion(self):
        z_noise = self._construct()
        white_noise = 2.5e-18
        white_ref_freq = 11.0
        sensitivity_rad_per_wb = 7.5e19
        expected_sw = Sii_D2S(white_noise, white_ref_freq) * (
            2 * sensitivity_rad_per_wb * z_noise.couple_term
        ) ** 2

        z_noise.noise = SimpleNamespace(
            output_stage=SimpleNamespace(
                white_noise=white_noise,
                white_ref_freq=white_ref_freq,
            )
        )

        actual = z_noise.cal_tphi1(
            sensitivity=sensitivity_rad_per_wb,
            sensitivity_unit='rad/s/Wb',
            is_print=False,
        )

        self.assertAlmostEqual(actual, 4 / expected_sw)
        self.assertEqual(z_noise.tphi1, actual)

    def test_cal_coupler_tphi1_uses_public_coupler_sensitivity_result(self):
        z_noise = self._construct()
        white_noise = 2.5e-18
        white_ref_freq = 11.0
        sensitivity_ghz_per_phi0 = 0.031
        expected_sensitivity = sensitivity_ghz_per_phi0 * 2e9 * np.pi / Phi0
        expected_sw = Sii_D2S(white_noise, white_ref_freq) * (
            2 * expected_sensitivity * z_noise.couple_term
        ) ** 2

        z_noise.noise = SimpleNamespace(
            output_stage=SimpleNamespace(
                white_noise=white_noise,
                white_ref_freq=white_ref_freq,
            )
        )
        calls = []

        class CouplerModel:
            def cal_coupler_sensitivity(self, **kwargs):
                calls.append(dict(kwargs))
                return SimpleNamespace(sensitivity_value=sensitivity_ghz_per_phi0)

        actual = z_noise.cal_coupler_tphi1(
            CouplerModel(),
            coupler_flux_point=0.42,
            method='numerical',
            flux_step=2e-4,
            qubit_idx=1,
            qubit_fluxes=[0.1, 0.2],
            is_print=False,
        )

        self.assertAlmostEqual(actual, 4 / expected_sw)
        self.assertEqual(z_noise.coupler_tphi1, actual)
        self.assertEqual(z_noise.coupler_tphi1_result.value, actual)
        self.assertEqual(z_noise.coupler_tphi1_result.metadata['source'], 'coupler-flux')
        self.assertEqual(z_noise.coupler_tphi1_result.metadata['coupler_flux_point'], 0.42)
        self.assertEqual(z_noise.coupler_tphi1_result.metadata['qubit_fluxes'], [0.1, 0.2])
        self.assertAlmostEqual(z_noise.coupler_sensitivity_ghz_per_phi0, sensitivity_ghz_per_phi0)
        self.assertAlmostEqual(z_noise.coupler_sensitivity_rad_per_wb, expected_sensitivity)
        self.assertEqual(
            calls,
            [
                {
                    'coupler_flux_point': 0.42,
                    'method': 'numerical',
                    'flux_step': 2e-4,
                    'qubit_idx': 1,
                    'qubit_fluxes': [0.1, 0.2],
                    'is_print': False,
                    'is_plot': False,
                }
            ],
        )

    def test_cal_coupler_tphi1_rejects_missing_sensitivity_source(self):
        z_noise = self._construct()

        with self.assertRaisesRegex(ValueError, 'coupler_model is required'):
            z_noise.cal_coupler_tphi1(is_print=False)

        with self.assertRaisesRegex(ValueError, 'coupler_flux_point is required'):
            z_noise.cal_coupler_tphi1(object(), is_print=False)

    def test_cal_tphi1_can_delegate_through_explicit_z_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []

        class RecordingAnalyzer:
            def calculate_tphi1(self, *, noise_output, sensitivity):
                analyzer_calls.append(
                    {
                        'noise_output': noise_output,
                        'sensitivity': sensitivity,
                    }
                )
                return 9.87

        def z_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        z_noise = self._construct(z_analyzer_builder=z_analyzer_builder)

        with patch.object(z_noise, 'get_sensitivity_at_idle', return_value=1.23):
            actual = z_noise.cal_tphi1(idle_freq=4.8e9, is_print=False)

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]['couple_term'], z_noise.couple_term)
        self.assertEqual(actual, 9.87)
        self.assertEqual(z_noise.tphi1, actual)
        self.assertEqual(len(analyzer_calls), 1)
        self.assertIs(analyzer_calls[0]['noise_output'], z_noise.noise.output_stage)
        self.assertEqual(analyzer_calls[0]['sensitivity'], 1.23)


if __name__ == '__main__':
    unittest.main()
