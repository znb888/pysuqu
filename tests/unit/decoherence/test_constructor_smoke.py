import unittest
import inspect
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import Decoherence, RNoiseDecoherence, XYNoiseDecoherence, ZNoiseDecoherence
from pysuqu.decoherence.electronics import ElectronicNoise
from pysuqu.qubit.base import AbstractQubit
import pysuqu.qubit.base as qubit_base


class DecoherenceConstructorSmokeTests(unittest.TestCase):
    @staticmethod
    def _clear_qubit_constructor_caches():
        AbstractQubit._solve_transmon_ec_ej.cache_clear()
        AbstractQubit._build_cached_transmon_template.cache_clear()
        qubit_base.QubitBase._clear_exact_solve_template_cache()

    @staticmethod
    def _clear_builder_signature_cache():
        Decoherence._inspect_builder_signature_cached.cache_clear()

    def setUp(self):
        self._clear_builder_signature_cache()
        self._clear_qubit_constructor_caches()

    def tearDown(self):
        self._clear_builder_signature_cache()
        self._clear_qubit_constructor_caches()

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

    def test_xy_constructor_reuses_cached_transmon_parameter_inversion_for_identical_targets(self):
        AbstractQubit._solve_transmon_ec_ej.cache_clear()
        AbstractQubit._build_cached_transmon_template.cache_clear()
        try:
            with patch('pysuqu.qubit.base.minimize', side_effect=qubit_base.minimize) as minimize_mock:
                first = self._construct(XYNoiseDecoherence)
                second = self._construct(XYNoiseDecoherence)

            self.assertEqual(minimize_mock.call_count, 1)
            np.testing.assert_allclose(first.qubit.Ec, second.qubit.Ec)
            np.testing.assert_allclose(first.qubit.Ej, second.qubit.Ej)
            self.assertAlmostEqual(first.qubit.qubit_f01, second.qubit.qubit_f01)
            self.assertAlmostEqual(first.qubit.qubit_anharm, second.qubit.qubit_anharm)
        finally:
            AbstractQubit._build_cached_transmon_template.cache_clear()
            AbstractQubit._solve_transmon_ec_ej.cache_clear()

    def test_xy_constructor_recomputes_transmon_parameter_inversion_when_targets_change(self):
        AbstractQubit._solve_transmon_ec_ej.cache_clear()
        AbstractQubit._build_cached_transmon_template.cache_clear()
        try:
            with patch('pysuqu.qubit.base.minimize', side_effect=qubit_base.minimize) as minimize_mock:
                baseline = self._construct(XYNoiseDecoherence)
                shifted = self._construct(XYNoiseDecoherence, qubit_freq=5.15e9, qubit_anharm=-280e6)

            self.assertEqual(minimize_mock.call_count, 2)
            self.assertNotAlmostEqual(baseline.qubit.qubit_f01, shifted.qubit.qubit_f01)
            self.assertNotAlmostEqual(baseline.qubit.qubit_anharm, shifted.qubit.qubit_anharm)
        finally:
            AbstractQubit._build_cached_transmon_template.cache_clear()
            AbstractQubit._solve_transmon_ec_ej.cache_clear()

    def test_r_constructor_reuses_cached_transmon_template_with_isolated_solver_state(self):
        AbstractQubit._solve_transmon_ec_ej.cache_clear()
        AbstractQubit._build_cached_transmon_template.cache_clear()
        try:
            with patch.object(
                qubit_base.QubitBase,
                '_generate_hamiltonian',
                autospec=True,
                side_effect=qubit_base.QubitBase._generate_hamiltonian,
            ) as generate_hamiltonian_mock:
                first = self._construct(RNoiseDecoherence)
                second = self._construct(RNoiseDecoherence)

            self.assertEqual(generate_hamiltonian_mock.call_count, 1)
            np.testing.assert_allclose(first.qubit.Ec, second.qubit.Ec)
            np.testing.assert_allclose(first.qubit.Ej, second.qubit.Ej)
            self.assertAlmostEqual(first.qubit.qubit_f01, second.qubit.qubit_f01)
            self.assertAlmostEqual(first.qubit.qubit_anharm, second.qubit.qubit_anharm)
            self.assertIsNot(first.qubit.solver_result, second.qubit.solver_result)
            self.assertIsNot(first.qubit.solver_result.hamiltonian, second.qubit.solver_result.hamiltonian)
        finally:
            AbstractQubit._build_cached_transmon_template.cache_clear()
            AbstractQubit._solve_transmon_ec_ej.cache_clear()

    def test_r_constructor_rebuilds_transmon_template_when_truncation_changes(self):
        AbstractQubit._solve_transmon_ec_ej.cache_clear()
        AbstractQubit._build_cached_transmon_template.cache_clear()
        try:
            with patch.object(
                qubit_base.QubitBase,
                '_generate_hamiltonian',
                autospec=True,
                side_effect=qubit_base.QubitBase._generate_hamiltonian,
            ) as generate_hamiltonian_mock:
                baseline = self._construct(RNoiseDecoherence, energy_trunc_level=12)
                shifted = self._construct(RNoiseDecoherence, energy_trunc_level=14)

            self.assertEqual(generate_hamiltonian_mock.call_count, 2)
            self.assertNotEqual(baseline.qubit.Nlevel, shifted.qubit.Nlevel)
        finally:
            AbstractQubit._build_cached_transmon_template.cache_clear()
            AbstractQubit._solve_transmon_ec_ej.cache_clear()

    def test_xy_constructor_routes_qubit_and_noise_builder_kwargs_independently(self):
        qubit_calls = []
        noise_calls = []

        def qubit_builder(
            *,
            frequency,
            anharmonicity,
            frequency_max,
            qubit_type,
            energy_trunc_level,
            is_print,
            qubit_marker,
        ):
            qubit_calls.append(
                {
                    'frequency': frequency,
                    'anharmonicity': anharmonicity,
                    'frequency_max': frequency_max,
                    'qubit_type': qubit_type,
                    'energy_trunc_level': energy_trunc_level,
                    'is_print': is_print,
                    'qubit_marker': qubit_marker,
                }
            )
            return SimpleNamespace(
                Ej=np.array([[13.5]]),
                Ec=np.array([[0.28]]),
                qubit_f01=frequency / 1e9,
                qubit_anharm=anharmonicity / 1e9,
            )

        def noise_builder(
            *,
            psd_freq,
            psd_S,
            noise_type,
            noise_prop,
            T_setup,
            attenuation_setup,
            is_spectral,
            is_print,
            noise_marker,
        ):
            noise_calls.append(
                {
                    'psd_freq': np.array(psd_freq, copy=True),
                    'psd_S': np.array(psd_S, copy=True),
                    'noise_type': noise_type,
                    'noise_prop': noise_prop,
                    'T_setup': np.array(T_setup, copy=True),
                    'attenuation_setup': np.array(attenuation_setup, copy=True),
                    'is_spectral': is_spectral,
                    'is_print': is_print,
                    'noise_marker': noise_marker,
                }
            )
            return SimpleNamespace(
                noise_type=noise_type,
                output_stage=SimpleNamespace(
                    frequency=np.array(psd_freq, copy=True),
                    white_noise_temperature=0.025,
                    white_noise=1.0,
                    white_ref_freq=1.0,
                ),
            )

        xy_noise = self._construct(
            XYNoiseDecoherence,
            qubit_builder=qubit_builder,
            noise_builder=noise_builder,
            qubit_marker='qubit-only',
            noise_marker='noise-only',
            is_print=False,
        )

        self.assertEqual(len(qubit_calls), 1)
        self.assertEqual(len(noise_calls), 1)
        self.assertEqual(qubit_calls[0]['qubit_marker'], 'qubit-only')
        self.assertEqual(noise_calls[0]['noise_marker'], 'noise-only')
        self.assertFalse(qubit_calls[0]['is_print'])
        self.assertFalse(noise_calls[0]['is_print'])
        self.assertEqual(xy_noise.qubit.qubit_f01, qubit_calls[0]['frequency'] / 1e9)
        self.assertEqual(xy_noise.noise.noise_type, noise_calls[0]['noise_type'])

        xy_noise.refresh_model(
            qubit_marker='qubit-refresh',
            noise_marker='noise-refresh',
            is_print=False,
        )

        self.assertEqual(len(qubit_calls), 2)
        self.assertEqual(len(noise_calls), 2)
        self.assertEqual(qubit_calls[1]['qubit_marker'], 'qubit-refresh')
        self.assertEqual(noise_calls[1]['noise_marker'], 'noise-refresh')
        self.assertFalse(qubit_calls[1]['is_print'])
        self.assertFalse(noise_calls[1]['is_print'])

    def test_refresh_model_caches_builder_signature_metadata_across_rebuilds(self):
        def qubit_builder(
            *,
            frequency,
            anharmonicity,
            frequency_max,
            qubit_type,
            energy_trunc_level,
            is_print,
        ):
            return SimpleNamespace(
                Ej=np.array([[13.5]]),
                Ec=np.array([[0.28]]),
                qubit_f01=frequency / 1e9,
                qubit_anharm=anharmonicity / 1e9,
            )

        def noise_builder(
            *,
            psd_freq,
            psd_S,
            noise_type,
            noise_prop,
            T_setup,
            attenuation_setup,
            is_spectral,
            is_print,
        ):
            return SimpleNamespace(
                noise_type=noise_type,
                output_stage=SimpleNamespace(
                    frequency=np.array(psd_freq, copy=True),
                    white_noise_temperature=0.025,
                    white_noise=1.0,
                    white_ref_freq=1.0,
                ),
            )

        with patch(
            'pysuqu.decoherence.dequbit.inspect.signature',
            side_effect=inspect.signature,
        ) as signature_mock:
            xy_noise = self._construct(
                XYNoiseDecoherence,
                qubit_builder=qubit_builder,
                noise_builder=noise_builder,
                is_print=False,
            )

            xy_noise.refresh_model(is_print=False)
            xy_noise.refresh_model(is_print=False)

        self.assertEqual(signature_mock.call_count, 2)


if __name__ == '__main__':
    unittest.main()
