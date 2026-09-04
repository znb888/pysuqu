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
from pysuqu.funclib.noisemodel import Sii2T_Double, T2Sii_Double


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

    def test_readout_cavity_analyzer_combines_chain_and_heat_temperature_by_psd(self):
        analyzer = ReadoutCavityAnalyzer(couple_term=1.0e6)
        noise_output = type("NoiseOutput", (), {"white_noise_temperature": 0.03277252715495531})()
        read_freq = 6.18e9
        heat_temperature_k = 0.0342

        chain_psd = T2Sii_Double(noise_output.white_noise_temperature, read_freq)
        heat_psd = T2Sii_Double(heat_temperature_k, read_freq)
        effective_temperature = Sii2T_Double(
            0.5 * (chain_psd + heat_psd),
            read_freq,
        )
        expected_n_bar = temp2nbar(effective_temperature, read_freq)

        actual_n_bar = analyzer.calculate_nbar(
            noise_output=noise_output,
            read_freq=read_freq,
            heat_temperature_k=heat_temperature_k,
        )

        self.assertAlmostEqual(actual_n_bar, expected_n_bar)
        self.assertAlmostEqual(effective_temperature, 0.033538178451180393)

    def test_readout_cavity_analyzer_rejects_invalid_heat_temperature(self):
        analyzer = ReadoutCavityAnalyzer(couple_term=1.0e6)
        noise_output = type("NoiseOutput", (), {"white_noise_temperature": 0.03})()

        for heat_temperature_k in (-1.0, np.nan, np.inf):
            with self.subTest(heat_temperature_k=heat_temperature_k):
                with self.assertRaises(ValueError):
                    analyzer.calculate_nbar(
                        noise_output=noise_output,
                        read_freq=6.18e9,
                        heat_temperature_k=heat_temperature_k,
                    )

    def test_r_facade_refreshes_nbar_for_cal_tphi_when_heat_is_provided(self):
        r_noise = self._construct()
        refresh_calls = []

        def refresh_nbar(*, read_freq, is_print, heat_temperature_k):
            refresh_calls.append(
                {
                    "read_freq": read_freq,
                    "is_print": is_print,
                    "heat_temperature_k": heat_temperature_k,
                }
            )
            r_noise.n_bar = 0.5
            return r_noise.n_bar

        r_noise.cal_nbar = refresh_nbar
        with patch.object(r_noise.r_analyzer, "calculate_tphi_cal", return_value=1.23e-6) as tphi_cal:
            actual = r_noise.cal_read_tphi(
                method="cal",
                chi=1.7e6,
                kappa=4.2e6,
                read_freq=6.5e9,
                heat_temperature_k=0.0342,
                is_print=False,
                is_plot=False,
            )

        self.assertEqual(
            refresh_calls,
            [{"read_freq": 6.5e9, "is_print": False, "heat_temperature_k": 0.0342}],
        )
        tphi_cal.assert_called_once_with(
            n_bar=0.5,
            kappa=4.2e6 * 2 * np.pi,
            chi=1.7e6 * 2 * np.pi,
        )
        self.assertEqual(actual.value, 1.23e-6)

    def test_r_facade_refreshes_nbar_for_fit_tphi_when_heat_is_provided(self):
        r_noise = self._construct()
        refresh_calls = []

        def refresh_nbar(*, read_freq, is_print, heat_temperature_k):
            refresh_calls.append((read_freq, is_print, heat_temperature_k))
            r_noise.n_bar = 0.5
            return r_noise.n_bar

        r_noise.cal_nbar = refresh_nbar
        popt = np.array([4.0e-6, 9.0e-6, 1.0, 0.0])
        pcov = np.diag([0.25e-12, 0.49e-12, 1.0e-4, 1.0e-4])
        with patch.object(r_noise, "cal_readcavity_psd") as readcavity_psd, patch.object(
            r_noise,
            "cal_read_dephase",
            return_value=np.array([0.97, 0.88, 0.81]),
        ), patch("pysuqu.decoherence.dequbit.fit_decay", return_value=(popt, pcov)):
            actual = r_noise.cal_read_tphi(
                method="fit",
                chi=1.7e6,
                kappa=4.2e6,
                read_freq=6.5e9,
                delay_list=np.array([1.0e-6, 2.0e-6, 3.0e-6]),
                heat_temperature_k=0.0342,
                is_print=False,
                is_plot=False,
            )

        self.assertEqual(refresh_calls, [(6.5e9, False, 0.0342)])
        readcavity_psd.assert_called_once()
        self.assertEqual(actual.value, popt[0])

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
        )
        actual_dephase = analyzer.calculate_read_dephase(
            n_bar=n_bar,
            kappa=kappa,
            chi=chi,
            experiment="Ramsey",
            delay_list=delay_list,
            N=100,
            len_pi=100e-9,
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

            def calculate_readcavity_psd(self, *, n_bar, kappa, chi, noise_freq):
                analyzer_calls.append(
                    {
                        "method": "calculate_readcavity_psd",
                        "n_bar": n_bar,
                        "kappa": kappa,
                        "chi": chi,
                        "noise_freq": noise_freq,
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
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
