import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import Decoherence, ZNoiseDecoherence
from pysuqu.funclib.mathlib import (
    PreparedFilteredPSD,
    integrate_filtered_psd_continuous,
    integrate_square_large_span,
    ramsey_transfunc,
)


class SparseRamseyContinuousIntegrationTests(unittest.TestCase):
    @staticmethod
    def _sparse_psd():
        freq = np.logspace(np.log10(5.0), np.log10(3e6), 100)
        psd = 5e-19 / freq + 4e-23
        return freq, psd

    def _construct_base_model(self):
        freq, psd = self._sparse_psd()

        def qubit_builder(**kwargs):
            return SimpleNamespace(builder_kwargs=dict(kwargs))

        def noise_builder(**kwargs):
            return SimpleNamespace(
                builder_kwargs=dict(kwargs),
                output_stage=SimpleNamespace(
                    frequency=np.array(freq, copy=True),
                    psd_single=np.array(psd, copy=True),
                ),
            )

        return Decoherence(
            psd_freq=freq,
            psd_S=psd,
            couple_term=1.0,
            couple_type="z",
            is_spectral=True,
            qubit_builder=qubit_builder,
            noise_builder=noise_builder,
        )

    def _construct_z_model(self):
        freq, psd = self._sparse_psd()
        with redirect_stdout(StringIO()):
            return ZNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True)

    def test_integrate_filtered_psd_continuous_matches_simple_analytic_cases(self):
        x_arr = np.logspace(0.0, 3.0, 30)

        constant_integral = integrate_filtered_psd_continuous(
            x_arr,
            np.full_like(x_arr, 2.5),
            lambda x: 1.0,
        )
        self.assertAlmostEqual(
            constant_integral,
            2.5 * (x_arr[-1] - x_arr[0]),
            delta=2.5 * (x_arr[-1] - x_arr[0]) * 1e-5,
        )

        inverse_integral = integrate_filtered_psd_continuous(
            x_arr,
            1.0 / x_arr,
            lambda x: 1.0,
        )
        self.assertAlmostEqual(
            inverse_integral,
            np.log(x_arr[-1] / x_arr[0]),
            delta=1e-5,
        )

    def test_sparse_ramsey_continuous_dephase_is_monotonic(self):
        freq, psd = self._sparse_psd()
        model = self._construct_base_model()
        delay_list = np.linspace(0.02e-6, 12e-6, 101)

        dephase = model.cal_dephase(
            psd=psd,
            noise_freq=freq,
            sensitivity_factor=7e12,
            experiment="Ramsey",
            delay_list=delay_list,
            integration_method="continuous",
        )

        self.assertTrue(np.all(np.isfinite(dephase)))
        self.assertTrue(np.all(dephase >= 0.0))
        self.assertTrue(np.all(dephase <= 1.0))
        self.assertTrue(np.all(np.diff(dephase) <= 1e-10))
        self.assertLess(dephase[-1], dephase[len(dephase) // 2])

    def test_prepared_continuous_psd_reuses_spline_and_matches_wrapper(self):
        freq, psd = self._sparse_psd()
        prepared = PreparedFilteredPSD.for_continuous(freq, psd)
        self.assertIsNotNone(prepared._log_spline)
        self.assertIsNotNone(prepared._nodes)

        for tau in (0.1e-6, 1.0e-6, 7.5e-6):
            filter_func = lambda x, tau=tau: 1.0 / (1.0 + (2.0 * np.pi * x * tau) ** 2)
            expected = integrate_filtered_psd_continuous(freq, psd, filter_func)
            actual = prepared.integrate_continuous(filter_func)
            self.assertEqual(actual, expected)

        # The prepared object is immutable at the array boundary, so later
        # integrations cannot rebuild or mutate the PSD preprocessing state.
        self.assertFalse(prepared.x_sorted.flags.writeable)
        self.assertFalse(prepared._nodes.flags.writeable)

    def test_prepared_continuous_many_matches_scalar_reference(self):
        freq, psd = self._sparse_psd()
        prepared = PreparedFilteredPSD.for_continuous(freq, psd)
        taus = np.array([0.1e-6, 1.0e-6, 7.5e-6])
        vector = prepared.integrate_continuous_many(
            lambda x: ramsey_transfunc(x, taus), len(taus)
        )
        scalar = np.array([
            prepared.integrate_continuous(
                lambda x, tau=tau: ramsey_transfunc(x, tau)
            )
            for tau in taus
        ])
        np.testing.assert_allclose(vector, scalar, rtol=1e-9, atol=1e-35)

    def test_cal_dephase_vectorized_filters_match_scalar_formula(self):
        freq, psd = self._sparse_psd()
        model = self._construct_base_model()
        delays = np.array([0.1e-6, 1.0e-6, 7.5e-6])
        factor = 7e12
        actual = model.cal_dephase(
            psd=psd,
            noise_freq=freq,
            sensitivity_factor=factor,
            experiment="Ramsey",
            delay_list=delays,
            integration_method="continuous",
        )
        prepared = PreparedFilteredPSD.for_continuous(freq, psd)
        expected = np.array([
            np.exp(
                -prepared.integrate_continuous(
                    lambda x, tau=tau: ramsey_transfunc(x, tau)
                )
                * (factor * 2) ** 2
                / 2
            )
            for tau in delays
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)

    def test_prepared_discrete_psd_matches_legacy_wrapper(self):
        freq, psd = self._sparse_psd()
        freq = freq[::-1]
        psd = psd[::-1]
        prepared = PreparedFilteredPSD.for_discrete(freq, psd)
        filter_func = lambda x: np.cos(2.0 * np.pi * x * 0.7e-6)
        expected = integrate_square_large_span(freq, psd, filter_func, method="log")
        actual = prepared.integrate_discrete(filter_func, method="log")
        self.assertEqual(actual, expected)

    def test_cal_dephase_prepares_continuous_psd_once_per_curve(self):
        model = self._construct_base_model()
        freq, psd = self._sparse_psd()
        with patch(
            "pysuqu.decoherence.dequbit.PreparedFilteredPSD.for_continuous",
            wraps=PreparedFilteredPSD.for_continuous,
        ) as prepare_mock:
            model.cal_dephase(
                psd=psd,
                noise_freq=freq,
                sensitivity_factor=1.0,
                delay_list=np.array([0.1e-6, 0.2e-6, 0.3e-6]),
                integration_method="continuous",
            )
        self.assertEqual(prepare_mock.call_count, 1)

    def test_cal_tphi2_defaults_to_continuous_integration_and_can_request_discrete(self):
        z_noise = self._construct_z_model()
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.95, 0.85, 0.75])
        popt = np.array([10.0e-6, 3.0e-6, 1.0, 0.0])
        pcov = np.diag([0.25e-12, 0.16e-12, 1.0e-4, 1.0e-4])

        with (
            patch.object(z_noise, "cal_dephase", return_value=fit_curve) as cal_dephase_mock,
            patch("pysuqu.decoherence.dequbit.fit_decay", return_value=(popt, pcov)),
        ):
            z_noise.cal_tphi2(
                method="fit",
                experiment="Ramsey",
                delay_list=delay_list,
                is_print=False,
                is_plot=False,
            )

        self.assertEqual(cal_dephase_mock.call_args.kwargs["integration_method"], "continuous")

        with (
            patch.object(z_noise, "cal_dephase", return_value=fit_curve) as cal_dephase_mock,
            patch("pysuqu.decoherence.dequbit.fit_decay", return_value=(popt, pcov)),
        ):
            z_noise.cal_tphi2(
                method="fit",
                experiment="Ramsey",
                delay_list=delay_list,
                is_print=False,
                is_plot=False,
                integration_method="discrete",
            )

        self.assertEqual(cal_dephase_mock.call_args.kwargs["integration_method"], "discrete")


if __name__ == "__main__":
    unittest.main()
