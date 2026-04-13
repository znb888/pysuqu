import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.analysis import ReadoutCavityAnalyzer
from pysuqu.decoherence.dequbit import RNoiseDecoherence
from pysuqu.funclib.mathlib import temp2nbar


class RNoiseDecoherenceReadTphiAnalyzerBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return RNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_readout_cavity_analyzer_keeps_existing_nbar_and_cal_tphi_formulas(self):
        analyzer = ReadoutCavityAnalyzer(couple_term=1.0e6)
        noise_output = type("NoiseOutput", (), {"white_noise_temperature": 0.055})()
        read_freq = 6.5e9
        kappa = 6.0e6 * 2 * np.pi
        chi = 1.5e6 * 2 * np.pi

        expected_n_bar = temp2nbar(noise_output.white_noise_temperature, read_freq)
        actual_n_bar = analyzer.calculate_nbar(
            noise_output=noise_output,
            read_freq=read_freq,
        )

        self.assertAlmostEqual(actual_n_bar, expected_n_bar)

        eta = kappa**2 / (kappa**2 + 4 * chi**2)
        nbar_th = expected_n_bar * eta
        expected_tphi = 1.0 / (nbar_th * (nbar_th + 1) * 4 * chi**2 / kappa)

        actual_tphi = analyzer.calculate_tphi_cal(
            n_bar=actual_n_bar,
            kappa=kappa,
            chi=chi,
        )

        self.assertAlmostEqual(actual_tphi, expected_tphi)

    def test_readout_cavity_analyzer_keeps_existing_psd_and_ramsey_dephase_formulas(self):
        analyzer = ReadoutCavityAnalyzer(couple_term=1.0e6)
        n_bar = 0.321
        kappa = 4.2e6 * 2 * np.pi
        chi = 1.7e6 * 2 * np.pi
        noise_freq = np.array([1.0, 10.0, 100.0])
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])

        eta = kappa**2 / (kappa**2 + 4 * chi**2)
        nbar_th = n_bar * eta
        expected_psd = (
            2
            * nbar_th
            * (nbar_th + 1)
            * (2 * chi) ** 2
            * (2 * kappa)
            / (noise_freq**2 + kappa**2)
        )
        expected_dfactor = (
            8
            * chi**2
            * nbar_th
            * (nbar_th + 1)
            * (kappa * delay_list - 1 + np.exp(-kappa * delay_list))
            / (kappa**2)
        )
        expected_dephase = np.exp(-expected_dfactor / 2)

        actual_psd = analyzer.calculate_readcavity_psd(
            n_bar=n_bar,
            kappa=kappa,
            chi=chi,
            noise_freq=noise_freq,
            read_freq=6.5e9,
        )
        actual_dephase = analyzer.calculate_read_dephase(
            n_bar=n_bar,
            kappa=kappa,
            chi=chi,
            experiment="Ramsey",
            delay_list=delay_list,
            N=100,
            len_pi=100e-9,
            read_freq=6.5e9,
        )

        np.testing.assert_allclose(actual_psd, expected_psd)
        np.testing.assert_allclose(actual_dephase, expected_dephase)

    def test_r_facade_can_delegate_analytical_path_through_explicit_r_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []

        class RecordingAnalyzer:
            def calculate_nbar(self, *, noise_output, read_freq):
                analyzer_calls.append(
                    {
                        "method": "calculate_nbar",
                        "noise_output": noise_output,
                        "read_freq": read_freq,
                    }
                )
                return 0.321

            def calculate_tphi_cal(self, *, n_bar, kappa, chi):
                analyzer_calls.append(
                    {
                        "method": "calculate_tphi_cal",
                        "n_bar": n_bar,
                        "kappa": kappa,
                        "chi": chi,
                    }
                )
                return 9.87e-6

        def r_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        r_noise = self._construct(r_analyzer_builder=r_analyzer_builder)

        actual = r_noise.cal_read_tphi(
            method="cal",
            experiment="SpinEcho",
            chi=1.7e6,
            kappa=4.2e6,
            is_print=False,
            is_plot=False,
        )

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]["couple_term"], r_noise.couple_term)
        self.assertEqual(r_noise.n_bar, 0.321)
        self.assertEqual(actual.value, 9.87e-6)
        self.assertEqual(actual.metadata["method"], "cal")
        self.assertEqual(actual.metadata["experiment"], "SpinEcho")
        self.assertEqual(r_noise.tphi_rc, 9.87e-6)
        self.assertEqual(
            analyzer_calls,
            [
                {
                    "method": "calculate_nbar",
                    "noise_output": r_noise.noise.output_stage,
                    "read_freq": 6.5e9,
                },
                {
                    "method": "calculate_tphi_cal",
                    "n_bar": 0.321,
                    "kappa": 4.2e6 * 2 * np.pi,
                    "chi": 1.7e6 * 2 * np.pi,
                },
            ],
        )

    def test_r_facade_can_delegate_fit_support_path_through_explicit_r_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.97, 0.88, 0.81])
        fit_psd = np.array([1.0, 0.5, 0.25])
        popt = np.array([4.0e-6, 9.0e-6, 1.0, 0.0])
        pcov = np.diag([0.25e-12, 0.49e-12, 1.0e-4, 1.0e-4])

        class RecordingAnalyzer:
            def calculate_nbar(self, *, noise_output, read_freq):
                analyzer_calls.append(
                    {
                        "method": "calculate_nbar",
                        "noise_output": noise_output,
                        "read_freq": read_freq,
                    }
                )
                return 0.321

            def calculate_readcavity_psd(self, *, n_bar, kappa, chi, noise_freq, read_freq):
                analyzer_calls.append(
                    {
                        "method": "calculate_readcavity_psd",
                        "n_bar": n_bar,
                        "kappa": kappa,
                        "chi": chi,
                        "noise_freq": noise_freq,
                        "read_freq": read_freq,
                    }
                )
                return fit_psd

            def calculate_read_dephase(
                self,
                *,
                n_bar,
                kappa,
                chi,
                experiment,
                delay_list,
                N,
                len_pi,
                read_freq,
            ):
                analyzer_calls.append(
                    {
                        "method": "calculate_read_dephase",
                        "n_bar": n_bar,
                        "kappa": kappa,
                        "chi": chi,
                        "experiment": experiment,
                        "delay_list": delay_list,
                        "N": N,
                        "len_pi": len_pi,
                        "read_freq": read_freq,
                    }
                )
                return fit_curve

        def r_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        r_noise = self._construct(r_analyzer_builder=r_analyzer_builder)

        with patch("pysuqu.decoherence.dequbit.fit_decay", return_value=(popt, pcov)):
            actual = r_noise.cal_read_tphi(
                method="fit",
                experiment="SpinEcho",
                chi=1.7e6,
                kappa=4.2e6,
                delay_list=delay_list,
                is_print=False,
                is_plot=False,
            )

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]["couple_term"], r_noise.couple_term)
        self.assertEqual(r_noise.n_bar, 0.321)
        self.assertEqual(actual.value, popt[0])
        self.assertEqual(actual.metadata["method"], "fit")
        self.assertEqual(actual.metadata["experiment"], "SpinEcho")
        self.assertEqual(actual.fit_diagnostics["tphi2"], popt[1])
        self.assertEqual(r_noise.psd_read.tolist(), fit_psd.tolist())
        self.assertEqual(r_noise.dephase.tolist(), fit_curve.tolist())
        self.assertEqual(
            analyzer_calls,
            [
                {
                    "method": "calculate_nbar",
                    "noise_output": r_noise.noise.output_stage,
                    "read_freq": 6.5e9,
                },
                {
                    "method": "calculate_readcavity_psd",
                    "n_bar": 0.321,
                    "kappa": 4.2e6 * 2 * np.pi,
                    "chi": 1.7e6 * 2 * np.pi,
                    "noise_freq": None,
                    "read_freq": 6.5e9,
                },
                {
                    "method": "calculate_read_dephase",
                    "n_bar": 0.321,
                    "kappa": 4.2e6 * 2 * np.pi,
                    "chi": 1.7e6 * 2 * np.pi,
                    "experiment": "SpinEcho",
                    "delay_list": delay_list,
                    "N": 100,
                    "len_pi": 100e-9,
                    "read_freq": 6.5e9,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()

