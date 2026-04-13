import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import Decoherence


class DecoherenceRefreshModelBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def test_refresh_model_rebuilds_default_dependencies_from_current_state(self):
        freq, psd = self._sample_noise_inputs()
        updated_psd = psd * 2

        qubit_instances = [SimpleNamespace(name='qubit-1'), SimpleNamespace(name='qubit-2')]
        noise_instances = [SimpleNamespace(name='noise-1'), SimpleNamespace(name='noise-2')]

        with (
            patch('pysuqu.decoherence.dequbit.AbstractQubit', side_effect=qubit_instances) as qubit_cls,
            patch('pysuqu.decoherence.dequbit.ElectronicNoise', side_effect=noise_instances) as noise_cls,
        ):
            model = Decoherence(
                psd_freq=freq,
                psd_S=psd,
                couple_term=1.0,
                couple_type='z',
                noise_type='white',
                noise_prop='single',
                qubit_freq=4.9e9,
                qubit_freq_max=5.1e9,
                qubit_anharm=-275e6,
                qubit_type='Transmon',
                energy_trunc_level=10,
                is_spectral=True,
            )

            self.assertIs(model.qubit, qubit_instances[0])
            self.assertIs(model.noise, noise_instances[0])

            model.qubit_freq = 6.1e9
            model.qubit_freq_max = 6.4e9
            model.qubit_anharm = -310e6
            model.qubit_type = 'Fluxonium'
            model.energy_trunc_level = 14
            model.psd_S = updated_psd
            model.noise_type = '1f'
            model.noise_prop = 'double'

            model.refresh_model()

        self.assertEqual(qubit_cls.call_count, 2)
        self.assertEqual(noise_cls.call_count, 2)
        self.assertIs(model.qubit, qubit_instances[1])
        self.assertIs(model.noise, noise_instances[1])

        self.assertEqual(qubit_cls.call_args_list[0].kwargs['frequency'], 4.9e9)
        self.assertEqual(qubit_cls.call_args_list[1].kwargs['frequency'], 6.1e9)
        self.assertEqual(qubit_cls.call_args_list[1].kwargs['frequency_max'], 6.4e9)
        self.assertEqual(qubit_cls.call_args_list[1].kwargs['anharmonicity'], -310e6)
        self.assertEqual(qubit_cls.call_args_list[1].kwargs['qubit_type'], 'Fluxonium')
        self.assertEqual(qubit_cls.call_args_list[1].kwargs['energy_trunc_level'], 14)

        np.testing.assert_allclose(noise_cls.call_args_list[0].kwargs['psd_S'], psd)
        np.testing.assert_allclose(noise_cls.call_args_list[1].kwargs['psd_S'], updated_psd)
        self.assertEqual(noise_cls.call_args_list[1].kwargs['noise_type'], '1f')
        self.assertEqual(noise_cls.call_args_list[1].kwargs['noise_prop'], 'double')
        self.assertTrue(noise_cls.call_args_list[1].kwargs['is_spectral'])

    def test_refresh_model_can_rebuild_dependencies_through_explicit_builders(self):
        freq, psd = self._sample_noise_inputs()
        updated_psd = psd * 3

        qubit_calls = []
        noise_calls = []

        def qubit_builder(**kwargs):
            qubit_calls.append(dict(kwargs))
            return SimpleNamespace(role='qubit', kwargs=dict(kwargs))

        def noise_builder(**kwargs):
            recorded = dict(kwargs)
            recorded['psd_freq'] = np.array(kwargs['psd_freq'], copy=True)
            recorded['psd_S'] = np.array(kwargs['psd_S'], copy=True)
            noise_calls.append(recorded)
            return SimpleNamespace(role='noise', kwargs=recorded)

        model = Decoherence(
            psd_freq=freq,
            psd_S=psd,
            couple_term=1.0,
            couple_type='xy',
            noise_type='1f',
            noise_prop='single',
            qubit_freq=5.0e9,
            qubit_freq_max=5.2e9,
            qubit_anharm=-250e6,
            qubit_type='Transmon',
            energy_trunc_level=12,
            is_spectral=True,
            qubit_builder=qubit_builder,
            noise_builder=noise_builder,
        )

        first_qubit = model.qubit
        first_noise = model.noise
        self.assertEqual(len(qubit_calls), 1)
        self.assertEqual(len(noise_calls), 1)
        self.assertEqual(qubit_calls[0]['frequency'], 5.0e9)
        np.testing.assert_allclose(noise_calls[0]['psd_S'], psd)

        model.qubit_freq = 6.0e9
        model.qubit_freq_max = 6.3e9
        model.qubit_type = 'Fluxonium'
        model.psd_S = updated_psd
        model.noise_type = '1f'
        model.refresh_model()

        self.assertEqual(len(qubit_calls), 2)
        self.assertEqual(len(noise_calls), 2)
        self.assertIsNot(model.qubit, first_qubit)
        self.assertIsNot(model.noise, first_noise)
        self.assertEqual(model.qubit.role, 'qubit')
        self.assertEqual(model.noise.role, 'noise')
        self.assertEqual(qubit_calls[1]['frequency'], 6.0e9)
        self.assertEqual(qubit_calls[1]['frequency_max'], 6.3e9)
        self.assertEqual(qubit_calls[1]['qubit_type'], 'Fluxonium')
        np.testing.assert_allclose(noise_calls[1]['psd_freq'], freq)
        np.testing.assert_allclose(noise_calls[1]['psd_S'], updated_psd)
        self.assertEqual(noise_calls[1]['noise_type'], '1f')
        self.assertEqual(noise_calls[1]['noise_prop'], 'single')

    def test_refresh_model_forwards_and_remembers_is_print_preference(self):
        freq, psd = self._sample_noise_inputs()
        qubit_instances = [SimpleNamespace(name='qubit-1'), SimpleNamespace(name='qubit-2'), SimpleNamespace(name='qubit-3')]
        noise_instances = [SimpleNamespace(name='noise-1'), SimpleNamespace(name='noise-2'), SimpleNamespace(name='noise-3')]

        with (
            patch('pysuqu.decoherence.dequbit.AbstractQubit', side_effect=qubit_instances) as qubit_cls,
            patch('pysuqu.decoherence.dequbit.ElectronicNoise', side_effect=noise_instances) as noise_cls,
        ):
            model = Decoherence(
                psd_freq=freq,
                psd_S=psd,
                couple_term=1.0,
                couple_type='z',
                is_print=True,
            )
            model.refresh_model()
            model.refresh_model(is_print=False)

        self.assertTrue(qubit_cls.call_args_list[0].kwargs['is_print'])
        self.assertTrue(noise_cls.call_args_list[0].kwargs['is_print'])
        self.assertTrue(qubit_cls.call_args_list[1].kwargs['is_print'])
        self.assertTrue(noise_cls.call_args_list[1].kwargs['is_print'])
        self.assertFalse(qubit_cls.call_args_list[2].kwargs['is_print'])
        self.assertFalse(noise_cls.call_args_list[2].kwargs['is_print'])


if __name__ == '__main__':
    unittest.main()

