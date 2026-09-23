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
from pysuqu.funclib.noisemodel import (
    S_transmission,
    Sii_A2dBm,
    Sii2T_Double,
    Sii_D2S,
    T2Sii_Double,
)


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

    def test_r_facade_defaults_read_freq_to_scalar_psd_freq(self):
        read_freq = 6.7e9
        with redirect_stdout(StringIO()):
            r_noise = RNoiseDecoherence(
                psd_freq=read_freq,
                psd_S=-140.0,
                is_spectral=False,
                is_print=False,
            )

        self.assertEqual(r_noise.read_freq, read_freq)
        self.assertAlmostEqual(
            r_noise.n_bar,
            temp2nbar(r_noise.noise.output_stage.white_noise_temperature, read_freq),
        )

    def test_r_facade_defaults_read_freq_to_6p5g_for_multi_point_psd_freq(self):
        with redirect_stdout(StringIO()):
            r_noise = RNoiseDecoherence(
                psd_freq=np.array([6.4e9, 6.5e9]),
                psd_S=-140.0,
                is_spectral=False,
                is_print=False,
        )

        self.assertEqual(r_noise.read_freq, 6.5e9)
        np.testing.assert_allclose(
            r_noise.n_bar,
            temp2nbar(r_noise.noise.output_stage.white_noise_temperature, 6.5e9),
        )

    def test_r_facade_accepts_explicit_read_freq_for_multi_point_psd_freq(self):
        read_freq = 6.8e9
        with redirect_stdout(StringIO()):
            r_noise = RNoiseDecoherence(
                psd_freq=np.array([6.4e9, 6.5e9]),
                psd_S=-140.0,
                is_spectral=False,
                read_freq=read_freq,
                is_print=False,
            )

        self.assertEqual(r_noise.read_freq, read_freq)

    def test_readout_cavity_analyzer_combines_chain_and_heat_temperature_by_psd(self):
        analyzer = ReadoutCavityAnalyzer(couple_term=1.0e6)
        noise_output = type(
            "NoiseOutput",
            (),
            {"white_noise_temperature": 0.03277252715495531},
        )()
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

    def test_readout_cavity_analyzer_rejects_invalid_heat_temperature(self):
        analyzer = ReadoutCavityAnalyzer(couple_term=1.0e6)
        noise_output = type("NoiseOutput", (), {"white_noise_temperature": 0.03})()

        for heat_temperature_k in (-1.0, np.nan, np.inf, None):
            if heat_temperature_k is None:
                continue
            with self.subTest(heat_temperature_k=heat_temperature_k):
                with self.assertRaises(ValueError):
                    analyzer.calculate_nbar(
                        noise_output=noise_output,
                        read_freq=6.18e9,
                        heat_temperature_k=heat_temperature_k,
                    )

        with self.assertRaises(ValueError):
            analyzer.calculate_nbar(
                noise_output=noise_output,
                read_freq=6.18e9,
                heat_temperature_k='bad',
            )

    def test_r_facade_refreshes_nbar_when_heat_temperature_is_provided(self):
        r_noise = self._construct()
        with patch.object(r_noise, 'cal_nbar', wraps=r_noise.cal_nbar) as refresh:
            result = r_noise.cal_read_tphi(
                method='cal',
                chi=1.7e6,
                kappa=4.2e6,
                heat_temperature_k=0.0342,
                is_print=False,
                is_plot=False,
            )

        refresh.assert_called_once_with(
            read_freq=None,
            is_print=False,
            heat_temperature_k=0.0342,
        )
        self.assertGreater(result.value, 0.0)


class RNoiseDecoherenceNoiseFloorInferenceTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return RNoiseDecoherence(
                psd_freq=freq,
                psd_S=psd,
                is_spectral=True,
                is_print=False,
                **kwargs,
            )

    def test_room_noise_floor_inverts_configured_temperature_chain(self):
        read_freq = 6.5e9
        t_setup = np.array([0.0, 0.02, 0.04])
        attenuation_setup = np.array([3.0, 10.0, 20.0])
        input_temperature = 42.0
        input_psd = T2Sii_Double(input_temperature, read_freq)
        input_psd_single = Sii_D2S(input_psd, read_freq)
        output_psd = S_transmission(
            input_psd,
            read_freq,
            t_setup,
            attenuation_setup,
        )
        measured_output_temperature = Sii2T_Double(output_psd, read_freq)
        r_noise = self._construct(
            T_setup=t_setup,
            attenuation_setup=attenuation_setup,
        )

        result = r_noise.cal_room_noise_floor(
            measured_output_temperature=measured_output_temperature,
            read_freq=read_freq,
            is_print=False,
        )

        self.assertEqual(result["source"], "temperature")
        self.assertEqual(result["internal_psd_convention"], "double-sided current PSD")
        self.assertEqual(result["room_noise_floor_convention"], "single-sided current PSD")
        self.assertIs(r_noise.room_noise_floor_result, result)
        np.testing.assert_allclose(result["input_psd_double_A2_per_Hz"], input_psd)
        np.testing.assert_allclose(result["input_psd_single_A2_per_Hz"], input_psd_single)
        np.testing.assert_allclose(
            result["input_noise_temperature_K"],
            input_temperature,
        )
        np.testing.assert_allclose(
            result["input_noise_floor_dBm_per_Hz"],
            Sii_A2dBm(input_psd_single),
        )
        np.testing.assert_allclose(
            result["input_noise_floor_single_dBm_per_Hz"],
            Sii_A2dBm(input_psd_single),
        )
        np.testing.assert_allclose(
            result["input_noise_floor_double_dBm_per_Hz"],
            Sii_A2dBm(input_psd),
        )
        self.assertGreater(
            result["input_noise_floor_single_dBm_per_Hz"],
            result["input_noise_floor_double_dBm_per_Hz"],
        )
        np.testing.assert_allclose(
            result["total_attenuation_dB"],
            np.sum(attenuation_setup),
        )

    def test_room_noise_floor_accepts_measured_nbar(self):
        read_freq = 6.5e9
        t_setup = np.array([0.0, 0.02, 0.04])
        attenuation_setup = np.array([3.0, 10.0, 20.0])
        input_temperature = 18.0
        output_psd = S_transmission(
            T2Sii_Double(input_temperature, read_freq),
            read_freq,
            t_setup,
            attenuation_setup,
        )
        measured_output_temperature = Sii2T_Double(output_psd, read_freq)
        measured_nbar = temp2nbar(measured_output_temperature, read_freq)
        r_noise = self._construct(
            T_setup=t_setup,
            attenuation_setup=attenuation_setup,
        )

        result = r_noise.cal_room_noise_floor(
            measured_nbar=measured_nbar,
            read_freq=read_freq,
            is_print=False,
        )

        self.assertEqual(result["source"], "nbar")
        np.testing.assert_allclose(result["output_nbar"], measured_nbar)
        np.testing.assert_allclose(
            result["output_noise_temperature_K"],
            measured_output_temperature,
        )
        np.testing.assert_allclose(
            result["input_noise_temperature_K"],
            input_temperature,
        )

    def test_room_noise_floor_rejects_output_below_chain_floor(self):
        r_noise = self._construct(
            T_setup=np.array([0.0, 0.05]),
            attenuation_setup=np.array([10.0, 20.0]),
        )

        with self.assertRaisesRegex(ValueError, "below the configured chain thermal floor"):
            r_noise.cal_room_noise_floor(
                measured_output_temperature=0.0,
                read_freq=6.5e9,
                is_print=False,
            )

    def test_room_noise_floor_requires_exactly_one_output_measurement(self):
        r_noise = self._construct()

        with self.assertRaisesRegex(ValueError, "exactly one"):
            r_noise.cal_room_noise_floor(is_print=False)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            r_noise.cal_room_noise_floor(
                measured_output_temperature=0.1,
                measured_nbar=0.2,
                is_print=False,
            )


if __name__ == "__main__":
    unittest.main()
