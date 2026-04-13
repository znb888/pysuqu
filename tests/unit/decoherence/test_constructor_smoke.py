import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import RNoiseDecoherence, XYNoiseDecoherence, ZNoiseDecoherence
from pysuqu.decoherence.electronics import ElectronicNoise


class DecoherenceConstructorSmokeTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, cls, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return cls(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_z_xy_r_decoherence_constructors_build_minimum_runtime_state(self):
        z_noise = self._construct(ZNoiseDecoherence)
        xy_noise = self._construct(XYNoiseDecoherence)
        r_noise = self._construct(RNoiseDecoherence)

        self.assertEqual(z_noise.couple_type, 'z')
        self.assertEqual(xy_noise.couple_type, 'xy')
        self.assertEqual(r_noise.couple_type, 'r')

        for model in (z_noise, xy_noise, r_noise):
            self.assertIsInstance(model.noise, ElectronicNoise)
            self.assertEqual(model.noise.noise_type, '1f')
            self.assertEqual(model.noise.psd_single_out.shape, model.psd_freq.shape)
            self.assertTrue(hasattr(model.qubit, 'Ej'))
            self.assertTrue(hasattr(model.qubit, 'Ec'))

        self.assertTrue(np.isfinite(z_noise.qubit_sensibility))
        self.assertTrue(np.isfinite(r_noise.n_bar))

    def test_electronic_noise_constructor_builds_minimum_fit_outputs(self):
        noise = self._construct(ElectronicNoise)

        self.assertEqual(noise.noise_type, '1f')
        self.assertEqual(noise.noise_prop, 'single')
        self.assertEqual(noise.psd_single_out.shape, noise.noise_freq.shape)
        self.assertIn('white_noise', noise.noise_fitres_out)
        self.assertIn('1f_coef', noise.noise_fitres_out)
        self.assertIn('white_ref_freq', noise.noise_fitres_out)
        self.assertGreater(noise.white_noise_out, 0.0)
        self.assertGreater(noise.white_noise_temperature_out, 0.0)


if __name__ == '__main__':
    unittest.main()

