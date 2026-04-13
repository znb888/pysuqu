import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.electronics import ElectronicNoise, S_transmission
from pysuqu.decoherence.results import NoiseFitResult


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


if __name__ == '__main__':
    unittest.main()
