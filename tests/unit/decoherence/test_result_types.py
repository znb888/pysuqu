import unittest

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence import NoiseFitResult, T1Result, TphiResult
from pysuqu.decoherence.electronics import ElectronicNoise


class DecoherenceResultTypesTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def test_tphi_result_preserves_minimum_contract_fields(self):
        metadata = {'method': 'fit', 'experiment': 'Ramsey'}
        fit_diagnostics = {
            'fit_error': 4.2e-8,
            'segments': {'global': {'points': 64}},
        }

        result = TphiResult(
            value=np.float64(3.2e-6),
            metadata=metadata,
            fit_diagnostics=fit_diagnostics,
        )

        self.assertEqual(result.value, 3.2e-6)
        self.assertEqual(result.unit, 's')
        self.assertEqual(result.metadata['method'], 'fit')
        self.assertEqual(result.metadata['experiment'], 'Ramsey')
        self.assertEqual(result.fit_diagnostics['fit_error'], 4.2e-8)
        self.assertEqual(result.fit_diagnostics['segments']['global']['points'], 64)
        self.assertIsNot(result.metadata, metadata)
        self.assertIsNot(result.fit_diagnostics, fit_diagnostics)
        self.assertIsNot(result.fit_diagnostics['segments'], fit_diagnostics['segments'])

    def test_t1_result_preserves_minimum_contract_fields(self):
        metadata = {'method': 'white-noise', 'source': 'xy-control'}
        fit_diagnostics = {'gamma_up': 12.5, 'gamma_down': 18.75}

        result = T1Result(
            value=np.float64(28.31275776497579e-6),
            metadata=metadata,
            fit_diagnostics=fit_diagnostics,
        )

        self.assertEqual(result.value, 28.31275776497579e-6)
        self.assertEqual(result.unit, 's')
        self.assertEqual(result.metadata['method'], 'white-noise')
        self.assertEqual(result.metadata['source'], 'xy-control')
        self.assertEqual(result.fit_diagnostics['gamma_up'], 12.5)
        self.assertEqual(result.fit_diagnostics['gamma_down'], 18.75)
        self.assertIsNot(result.metadata, metadata)
        self.assertIsNot(result.fit_diagnostics, fit_diagnostics)

    def test_noise_fit_result_from_fit_dict_preserves_required_fields(self):
        freq, psd = self._sample_noise_inputs()
        fit_data = ElectronicNoise.fit_psd(psd=psd, freq=freq, noise_type='1f', noise_prop='double')

        result = NoiseFitResult.from_fit_dict(
            fit_data,
            noise_type='1f',
            noise_prop='double',
            metadata={'source': 'fit_psd'},
        )

        self.assertGreater(result.value, 0.0)
        self.assertEqual(result.unit, 'A^2/Hz')
        self.assertEqual(result.metadata['noise_type'], '1f')
        self.assertEqual(result.metadata['noise_prop'], 'double')
        self.assertEqual(result.metadata['source'], 'fit_psd')
        self.assertIn('1f_coef', result.fit_diagnostics)
        self.assertIn('corner_freq', result.fit_diagnostics)
        self.assertIn('white_ref_freq', result.fit_diagnostics)
        self.assertIn('white_noise_temperature', result.fit_diagnostics)
        self.assertGreater(result.fit_diagnostics['white_noise_temperature'], 0.0)

    def test_noise_fit_result_from_fit_dict_requires_minimum_legacy_keys(self):
        with self.assertRaises(ValueError):
            NoiseFitResult.from_fit_dict(
                {'white_noise': 1.0},
                noise_type='1f',
                noise_prop='double',
            )


if __name__ == '__main__':
    unittest.main()

