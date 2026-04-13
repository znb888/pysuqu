import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import Decoherence


class DecoherenceDephaseKernelBoundaryTests(unittest.TestCase):
    def _construct(self, noise_frequency=None):
        if noise_frequency is None:
            noise_frequency = np.array([10.0, 20.0, 40.0])

        noise_frequency = np.array(noise_frequency, copy=True)

        def qubit_builder(**kwargs):
            return SimpleNamespace(builder_kwargs=dict(kwargs))

        def noise_builder(**kwargs):
            return SimpleNamespace(
                builder_kwargs=dict(kwargs),
                output_stage=SimpleNamespace(
                    frequency=np.array(noise_frequency, copy=True),
                ),
            )

        return Decoherence(
            psd_freq=noise_frequency,
            psd_S=np.ones_like(noise_frequency),
            couple_term=1.0,
            couple_type="z",
            is_spectral=True,
            qubit_builder=qubit_builder,
            noise_builder=noise_builder,
        )

    def test_generate_transfunc_dispatches_supported_experiments(self):
        model = self._construct()
        calls = []

        def ramsey_transfunc(freq, tau):
            calls.append(("Ramsey", freq, tau))
            return tau + freq

        def echo_transfunc(freq, tau):
            calls.append(("SpinEcho", freq, tau))
            return tau - freq

        def cpmg_transfunc(freq, tau, pulse_count, len_pi):
            calls.append(("CPMG", freq, tau, pulse_count, len_pi))
            return tau + pulse_count + len_pi + freq

        with (
            patch(
                "pysuqu.decoherence.dequbit.ramsey_transfunc",
                side_effect=ramsey_transfunc,
            ),
            patch(
                "pysuqu.decoherence.dequbit.echo_transfunc",
                side_effect=echo_transfunc,
            ),
            patch(
                "pysuqu.decoherence.dequbit.cpmg_transfunc",
                side_effect=cpmg_transfunc,
            ),
        ):
            ramsey_filter = model._generate_transfunc("Ramsey", tau=1.5e-6)
            echo_filter = model._generate_transfunc("SpinEcho", tau=2.5e-6)
            cpmg_filter = model._generate_transfunc(
                "CPMG",
                tau=3.5e-6,
                N=8,
                len_pi=40e-9,
            )

            self.assertEqual(ramsey_filter(3.0), 1.5e-6 + 3.0)
            self.assertEqual(echo_filter(4.0), 2.5e-6 - 4.0)
            self.assertEqual(cpmg_filter(5.0), 3.5e-6 + 8 + 40e-9 + 5.0)

        self.assertEqual(
            calls,
            [
                ("Ramsey", 3.0, 1.5e-6),
                ("SpinEcho", 4.0, 2.5e-6),
                ("CPMG", 5.0, 3.5e-6, 8, 40e-9),
            ],
        )

        with self.assertRaisesRegex(ValueError, "Unknown experiment type"):
            model._generate_transfunc("Unknown")

    def test_cal_dephase_uses_output_stage_frequency_and_applies_shared_exponent_formula(self):
        noise_frequency = np.array([11.0, 22.0, 33.0])
        psd = np.array([1.0e-18, 1.5e-18, 2.0e-18])
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        integrated_weights = [0.125, 0.25, 0.5]
        generated_filters = []

        model = self._construct(noise_frequency=noise_frequency)

        def fake_generate_transfunc(experiment, tau, N, len_pi):
            def transfer(freq):
                return freq + tau

            generated_filters.append(
                {
                    "experiment": experiment,
                    "tau": tau,
                    "N": N,
                    "len_pi": len_pi,
                    "transfer": transfer,
                }
            )
            return transfer

        with (
            patch.object(model, "_generate_transfunc", side_effect=fake_generate_transfunc),
            patch(
                "pysuqu.decoherence.dequbit.integrate_square_large_span",
                side_effect=integrated_weights,
            ) as integrate_mock,
        ):
            actual = model.cal_dephase(
                psd=psd,
                sensitivity_factor=0.75,
                experiment="CPMG",
                delay_list=delay_list,
                N=8,
                len_pi=40e-9,
            )

        expected = np.exp(-np.array(integrated_weights) * (0.75 * 2) ** 2 / 2)
        np.testing.assert_allclose(actual, expected)

        self.assertEqual(
            [
                (entry["experiment"], entry["tau"], entry["N"], entry["len_pi"])
                for entry in generated_filters
            ],
            [
                ("CPMG", 1.0e-6, 8, 40e-9),
                ("CPMG", 2.0e-6, 8, 40e-9),
                ("CPMG", 3.0e-6, 8, 40e-9),
            ],
        )
        self.assertEqual(len(integrate_mock.call_args_list), 3)

        for index, call in enumerate(integrate_mock.call_args_list):
            np.testing.assert_allclose(call.args[0], noise_frequency)
            np.testing.assert_allclose(call.args[1], psd)
            self.assertIs(call.args[2], generated_filters[index]["transfer"])
            self.assertEqual(call.kwargs["method"], "log")


if __name__ == "__main__":
    unittest.main()

