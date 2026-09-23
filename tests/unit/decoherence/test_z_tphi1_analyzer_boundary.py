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

    def test_cal_tphi1_accepts_structured_sensitivity_result(self):
        z_noise = self._construct()
        white_noise = 2.5e-18
        white_ref_freq = 11.0
        sensitivity_result = SimpleNamespace(sensitivity_value=-0.125)
        expected_sensitivity = sensitivity_result.sensitivity_value * 2e9 * np.pi / Phi0
        expected_sw = Sii_D2S(white_noise, white_ref_freq) * (
            2 * expected_sensitivity * z_noise.couple_term
        ) ** 2

        z_noise.noise = SimpleNamespace(
            output_stage=SimpleNamespace(
                white_noise=white_noise,
                white_ref_freq=white_ref_freq,
            )
        )

        actual = z_noise.cal_tphi1(sensitivity=sensitivity_result, is_print=False)

        self.assertAlmostEqual(actual, 4 / expected_sw)

    def test_cal_coupler_tphi1_uses_coupler_model_sensitivity_and_mutual_inductance(self):
        z_noise = self._construct(couple_term=2.9e-12)
        white_noise = 2.5e-18
        white_ref_freq = 11.0
        sensitivity_value = -0.125
        expected_sensitivity = sensitivity_value * 2e9 * np.pi / Phi0
        expected_sw = Sii_D2S(white_noise, white_ref_freq) * (
            2 * expected_sensitivity * z_noise.couple_term
        ) ** 2
        model_calls = []

        class RecordingCouplerModel:
            def cal_coupler_sensitivity(self, **kwargs):
                model_calls.append(dict(kwargs))
                return SimpleNamespace(
                    sensitivity_value=sensitivity_value,
                    coupler_flux_point=kwargs['coupler_flux_point'],
                )

        z_noise.noise = SimpleNamespace(
            output_stage=SimpleNamespace(
                white_noise=white_noise,
                white_ref_freq=white_ref_freq,
            )
        )

        actual = z_noise.cal_coupler_tphi1(
            RecordingCouplerModel(),
            coupler_flux_point=0.38,
            flux_step=2e-4,
            qubit_idx=1,
            qubit_fluxes=[0.12, 0.13],
            is_print=False,
        )

        self.assertAlmostEqual(actual, 4 / expected_sw)
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(model_calls[0]['coupler_flux_point'], 0.38)
        self.assertEqual(model_calls[0]['flux_step'], 2e-4)
        self.assertEqual(model_calls[0]['qubit_idx'], 1)
        self.assertEqual(model_calls[0]['qubit_fluxes'], [0.12, 0.13])
        self.assertFalse(model_calls[0]['is_print'])
        self.assertEqual(z_noise.coupler_tphi1, actual)
        self.assertAlmostEqual(
            z_noise.coupler_tphi1_result.metadata['sensitivity_ghz_per_phi0'],
            sensitivity_value,
        )
        self.assertAlmostEqual(
            z_noise.coupler_tphi1_result.metadata['sensitivity_rad_per_wb'],
            expected_sensitivity,
        )

    def test_cal_coupler_tphi1_combined_adds_gate_and_off_rates(self):
        z_noise = self._construct()
        calls = []

        def single_tphi(*, sensitivity, sensitivity_unit, is_print):
            calls.append((sensitivity, sensitivity_unit, is_print))
            return {0.031: 2e-6, 0.007: 10e-6}[sensitivity]

        with patch.object(z_noise, 'cal_coupler_tphi1', side_effect=single_tphi):
            result = z_noise.cal_coupler_tphi1_combined(
                gate_sensitivity=0.031,
                off_sensitivity=0.007,
                couplers_per_qubit=4,
                is_print=False,
            )

        expected_rate = 1 / 2e-6 + 3 / 10e-6
        self.assertAlmostEqual(result.value, 1 / expected_rate)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_total_s_inv'], expected_rate)
        self.assertEqual(result.metadata['gate_coupler_count'], 1)
        self.assertEqual(result.metadata['off_coupler_count'], 3)
        self.assertEqual(
            calls,
            [
                (0.031, 'GHz/Phi0', False),
                (0.007, 'GHz/Phi0', False),
            ],
        )
        self.assertIs(z_noise.coupler_tphi1_combined_result, result)

    def test_cal_coupler_tphi1_combined_handles_zero_sensitivity(self):
        z_noise = self._construct()
        with patch.object(z_noise, 'cal_coupler_tphi1', return_value=np.inf):
            result = z_noise.cal_coupler_tphi1_combined(
                gate_sensitivity=0.0,
                off_sensitivity=0.0,
                couplers_per_qubit=4,
                is_print=False,
            )

        self.assertTrue(np.isinf(result.value))
        self.assertEqual(result.fit_diagnostics['gamma_total_s_inv'], 0.0)

    def test_cal_coupler_tphi1_combined_rejects_invalid_inputs(self):
        z_noise = self._construct()
        for value in (0, -1, 1.5, True, np.int64(0)):
            with self.subTest(couplers_per_qubit=value):
                with self.assertRaises(ValueError):
                    z_noise.cal_coupler_tphi1_combined(
                        gate_sensitivity=0.031,
                        off_sensitivity=0.007,
                        couplers_per_qubit=value,
                        is_print=False,
                    )

        for value in (np.nan, np.inf, None, 'bad'):
            with self.subTest(sensitivity=value):
                with self.assertRaises(ValueError):
                    z_noise.cal_coupler_tphi1_combined(
                        gate_sensitivity=value,
                        off_sensitivity=0.007,
                        couplers_per_qubit=2,
                        is_print=False,
                    )


if __name__ == '__main__':
    unittest.main()
