import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np
from scipy.optimize import curve_fit

from tests.support import install_test_stubs

install_test_stubs()

import pysuqu.decoherence.electronics as electronics_module
from pysuqu.decoherence.electronics import ElectronicNoise, S_transmission, Sii2T_Double
from pysuqu.decoherence.results import NoiseFitResult
from pysuqu.funclib.mathlib import find_knee_point, inverse_func


class ElectronicNoisePipelineTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return ElectronicNoise(psd_freq=freq, psd_S=psd, noise_prop='single', **kwargs)

    @staticmethod
    def _legacy_fit_psd_1f(psd: np.ndarray, freq: np.ndarray) -> dict[str, float]:
        tail_len = max(1, len(psd) // 10)
        b_guess = np.mean(psd[-tail_len:])
        idx_1 = np.argmin(np.abs(freq - 1.0))
        if 0.1 <= freq[idx_1] <= 10.0:
            target_idx = idx_1
        else:
            target_idx = 0

        x0 = freq[target_idx]
        y0 = psd[target_idx]
        a_guess = (y0 - b_guess) * x0
        if a_guess <= 0:
            a_guess = 1.0

        popt, _ = curve_fit(inverse_func, freq, psd, p0=[a_guess, b_guess], maxfev=5000)
        a_fit, b_fit = popt
        psd_fit = inverse_func(freq, a_fit, b_fit)
        x_knee, _ = find_knee_point(freq, psd_fit)
        white_region = freq[freq > x_knee]
        if white_region.size > 0:
            white_ref_freq = float(np.mean(white_region))
        else:
            white_ref_freq = float(freq[-1])

        return {
            'white_noise': float(b_fit),
            '1f_coef': float(a_fit),
            'corner_freq': float(x_knee),
            'white_ref_freq': white_ref_freq,
            'white_noise_temperature': float(Sii2T_Double(b_fit, white_ref_freq)),
        }

    def test_refresh_model_exposes_explicit_pipeline_stages_for_spectral_noise(self):
        noise = self._construct(is_spectral=True)

        self.assertIsInstance(noise.input_stage.fit_result, NoiseFitResult)
        self.assertIsInstance(noise.output_stage.fit_result, NoiseFitResult)

        np.testing.assert_allclose(noise.input_stage.frequency, noise.noise_freq)
        np.testing.assert_allclose(noise.output_stage.frequency, noise.noise_freq)
        np.testing.assert_allclose(noise.input_stage.psd_double, noise.psd_double_in)
        np.testing.assert_allclose(noise.output_stage.psd_single, noise.psd_single_out)
        np.testing.assert_allclose(noise.output_stage.psd_smooth, noise.psd_smooth_out)

        self.assertAlmostEqual(noise.input_stage.white_noise, noise.white_noise_in)
        self.assertAlmostEqual(noise.output_stage.white_noise, noise.white_noise_out)
        self.assertAlmostEqual(noise.output_stage.white_noise_temperature, noise.white_noise_temperature_out)
        self.assertAlmostEqual(noise.output_stage.fit_result.value, noise.noise_fitres_out['white_noise'])
        self.assertEqual(noise.output_stage.fit_result.metadata['noise_type'], '1f')
        self.assertEqual(noise.output_stage.fit_result.metadata['noise_prop'], 'double')
        self.assertAlmostEqual(
            noise.output_stage.fit_result.fit_diagnostics['corner_freq'],
            noise.noise_fitres_out['corner_freq'],
        )

    def test_refresh_model_matches_scalar_transmission_for_spectral_noise(self):
        noise = self._construct(is_spectral=True)

        expected_psd_double_out = np.array(
            [
                S_transmission(point_psd, point_freq, noise.T_setup, noise.attenuation_setup)
                for point_psd, point_freq in zip(noise.psd_double_in, noise.noise_freq)
            ]
        )

        np.testing.assert_allclose(noise.output_stage.psd_double, expected_psd_double_out)
        np.testing.assert_allclose(noise.psd_double_out, expected_psd_double_out)

    def test_fit_psd_matches_legacy_curve_fit_reference_for_1f_pipeline_data(self):
        noise = self._construct(is_spectral=True)

        for psd in (noise.input_stage.psd_smooth, noise.output_stage.psd_smooth):
            current = noise.fit_psd(psd, noise.noise_freq, noise_type='1f', noise_prop='double')
            legacy = self._legacy_fit_psd_1f(psd, noise.noise_freq)
            for key in (
                'white_noise',
                '1f_coef',
                'corner_freq',
                'white_ref_freq',
                'white_noise_temperature',
            ):
                self.assertAlmostEqual(current[key], legacy[key], places=12)

    def test_refresh_model_reuses_cached_smoothing_for_identical_pipeline_inputs(self):
        freq, psd = self._sample_noise_inputs()
        original_smooth_data = electronics_module.smooth_data

        ElectronicNoise._smooth_psd_cached.cache_clear()
        ElectronicNoise._build_spectral_pipeline_cached.cache_clear()
        try:
            with patch(
                'pysuqu.decoherence.electronics.smooth_data',
                side_effect=original_smooth_data,
            ) as smooth_mock:
                first = self._construct(is_spectral=True)
                second = self._construct(is_spectral=True)

                first.input_stage.psd_smooth[0] = -1.0
                third = self._construct(is_spectral=True)

                with redirect_stdout(StringIO()):
                    ElectronicNoise(
                        psd_freq=freq,
                        psd_S=psd * 1.01,
                        noise_prop='single',
                        is_spectral=True,
                        is_print=False,
                    )

            self.assertEqual(smooth_mock.call_count, 4)
            np.testing.assert_allclose(second.input_stage.psd_smooth, third.input_stage.psd_smooth)
            np.testing.assert_allclose(second.output_stage.psd_smooth, third.output_stage.psd_smooth)
            self.assertNotEqual(third.input_stage.psd_smooth[0], -1.0)
        finally:
            ElectronicNoise._smooth_psd_cached.cache_clear()
            ElectronicNoise._build_spectral_pipeline_cached.cache_clear()

    def test_refresh_model_reuses_cached_fit_results_for_identical_pipeline_inputs(self):
        freq, psd = self._sample_noise_inputs()
        original_fit_inverse_psd = ElectronicNoise._fit_inverse_psd

        ElectronicNoise._fit_psd_cached.cache_clear()
        ElectronicNoise._build_spectral_pipeline_cached.cache_clear()
        try:
            with patch.object(
                ElectronicNoise,
                '_fit_inverse_psd',
                wraps=original_fit_inverse_psd,
            ) as fit_mock:
                first = self._construct(is_spectral=True)
                second = self._construct(is_spectral=True)

                first.noise_fitres_in['white_noise'] = -1.0
                first.noise_fitres_out['white_noise'] = -2.0
                third = self._construct(is_spectral=True)

                with redirect_stdout(StringIO()):
                    ElectronicNoise(
                        psd_freq=freq,
                        psd_S=psd * 1.01,
                        noise_prop='single',
                        is_spectral=True,
                        is_print=False,
                    )

            self.assertEqual(fit_mock.call_count, 4)
            self.assertAlmostEqual(second.noise_fitres_in['white_noise'], third.noise_fitres_in['white_noise'])
            self.assertAlmostEqual(second.noise_fitres_out['white_noise'], third.noise_fitres_out['white_noise'])
            self.assertNotEqual(third.noise_fitres_in['white_noise'], -1.0)
            self.assertNotEqual(third.noise_fitres_out['white_noise'], -2.0)
        finally:
            ElectronicNoise._fit_psd_cached.cache_clear()
            ElectronicNoise._build_spectral_pipeline_cached.cache_clear()

    def test_refresh_model_reuses_cached_full_spectral_pipeline_without_sharing_arrays(self):
        freq, psd = self._sample_noise_inputs()
        original_transmission = electronics_module.S_transmission

        ElectronicNoise._build_spectral_pipeline_cached.cache_clear()
        try:
            with patch(
                'pysuqu.decoherence.electronics.S_transmission',
                side_effect=original_transmission,
            ) as transmission_mock:
                first = self._construct(is_spectral=True)
                second = self._construct(is_spectral=True)

                first.input_stage.psd_double[0] = -1.0
                first.output_stage.psd_double[0] = -2.0
                third = self._construct(is_spectral=True)

                with redirect_stdout(StringIO()):
                    ElectronicNoise(
                        psd_freq=freq,
                        psd_S=psd,
                        noise_prop='single',
                        attenuation_setup=np.array([40, 1, 10, 10, 20, 11]),
                        is_spectral=True,
                        is_print=False,
                    )

            self.assertEqual(transmission_mock.call_count, 2)
            np.testing.assert_allclose(second.input_stage.psd_double, third.input_stage.psd_double)
            np.testing.assert_allclose(second.output_stage.psd_double, third.output_stage.psd_double)
            self.assertNotEqual(third.input_stage.psd_double[0], -1.0)
            self.assertNotEqual(third.output_stage.psd_double[0], -2.0)
        finally:
            ElectronicNoise._build_spectral_pipeline_cached.cache_clear()

    def test_refresh_model_keeps_constant_pipeline_outputs_explicit_without_fit_results(self):
        freq, _ = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            noise = ElectronicNoise(
                psd_freq=freq,
                psd_S=-140.0,
                noise_prop='single',
                is_spectral=False,
            )

        self.assertEqual(noise.noise_type, 'constant')
        self.assertIsNone(noise.input_stage.fit_result)
        self.assertIsNone(noise.output_stage.fit_result)

        np.testing.assert_allclose(noise.input_stage.frequency, noise.noise_freq)
        np.testing.assert_allclose(noise.output_stage.psd_double, noise.psd_double_out)
        np.testing.assert_allclose(noise.output_stage.psd_single, noise.psd_single_out)
        np.testing.assert_allclose(noise.input_stage.white_noise, noise.white_noise_in)
        np.testing.assert_allclose(noise.output_stage.white_ref_freq, noise.white_ref_freq_out)
        np.testing.assert_allclose(
            noise.output_stage.white_noise_temperature,
            noise.white_noise_temperature_out,
        )

    def test_refresh_model_normalizes_non_spectral_contract_after_state_change(self):
        noise = self._construct(is_spectral=True)

        noise.is_spectral = False
        noise.psd = -140.0
        with redirect_stdout(StringIO()):
            noise.refresh_model(is_print=False)

        self.assertEqual(noise.noise_type, 'constant')
        self.assertIsNone(noise.input_stage.fit_result)
        self.assertIsNone(noise.output_stage.fit_result)
        self.assertFalse(hasattr(noise, 'noise_fitres_in'))
        self.assertFalse(hasattr(noise, 'noise_fitres_out'))
        np.testing.assert_allclose(noise.input_stage.psd_double, noise.psd_double_in)
        np.testing.assert_allclose(noise.output_stage.psd_double, noise.psd_double_out)
        np.testing.assert_allclose(noise.output_stage.psd_single, noise.psd_single_out)


if __name__ == '__main__':
    unittest.main()
