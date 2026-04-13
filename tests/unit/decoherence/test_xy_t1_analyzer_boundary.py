import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.analysis import XYRelaxationAnalyzer
from pysuqu.decoherence.dequbit import XYNoiseDecoherence
from pysuqu.decoherence.electronics import T2Sii_Double
from pysuqu.qubit.base import Phi0


class XYNoiseDecoherenceT1AnalyzerBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return XYNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_xy_relaxation_analyzer_keeps_existing_transition_and_thermal_formulas(self):
        analyzer = XYRelaxationAnalyzer(couple_term=0.65e-12)
        noise_output = SimpleNamespace(white_noise_temperature=0.035)
        qubit_freq = 5.2e9
        ej = 13.5
        ec = 0.28

        expected_sxy_po = T2Sii_Double(noise_output.white_noise_temperature, f=qubit_freq)
        expected_sxy_ne = T2Sii_Double(noise_output.white_noise_temperature, f=-qubit_freq)
        expected_b = 2e9 * np.pi * analyzer.couple_term * ej * (2 * ec / ej) ** 0.25 / Phi0
        expected_gamma_up = expected_sxy_ne * expected_b**2
        expected_gamma_down = expected_sxy_po * expected_b**2
        expected_t1 = 1 / (expected_gamma_up + expected_gamma_down)

        actual = analyzer.calculate_t1(
            noise_output=noise_output,
            qubit_freq=qubit_freq,
            Ej=ej,
            Ec=ec,
        )

        self.assertAlmostEqual(actual['gamma_up'], expected_gamma_up)
        self.assertAlmostEqual(actual['gamma_down'], expected_gamma_down)
        self.assertAlmostEqual(actual['t1'], expected_t1)

        expected_gamma_down_actual = 1 / (37.0 * 1e-6)
        expected_thermal_excitation = expected_gamma_up / (
            expected_gamma_up + expected_gamma_down_actual
        )
        expected_thermal_excitation_onlyxy = expected_gamma_up / (
            expected_gamma_up + expected_gamma_down
        )

        actual_thermal = analyzer.calculate_thermal_excitation(
            gamma_up=actual['gamma_up'],
            gamma_down=actual['gamma_down'],
            t1_us=37.0,
        )

        self.assertEqual(
            actual_thermal,
            (
                expected_thermal_excitation,
                expected_thermal_excitation_onlyxy,
            ),
        )

        expected_default_gamma_down_actual = 1 / 100e-6
        actual_thermal_without_t1 = analyzer.calculate_thermal_excitation(
            gamma_up=actual['gamma_up'],
            gamma_down=actual['gamma_down'],
            t1_us=None,
        )

        self.assertEqual(
            actual_thermal_without_t1,
            (
                expected_gamma_up / (expected_gamma_up + expected_default_gamma_down_actual),
                expected_thermal_excitation_onlyxy,
            ),
        )

    def test_xy_facade_can_delegate_through_explicit_xy_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []

        class RecordingAnalyzer:
            def calculate_t1(self, *, noise_output, qubit_freq, Ej, Ec):
                analyzer_calls.append(
                    {
                        'method': 'calculate_t1',
                        'noise_output': noise_output,
                        'qubit_freq': qubit_freq,
                        'Ej': Ej,
                        'Ec': Ec,
                    }
                )
                return {
                    'gamma_up': 1.25,
                    'gamma_down': 3.75,
                    't1': 0.2,
                }

            def calculate_thermal_excitation(self, *, gamma_up, gamma_down, t1_us):
                analyzer_calls.append(
                    {
                        'method': 'calculate_thermal_excitation',
                        'gamma_up': gamma_up,
                        'gamma_down': gamma_down,
                        't1_us': t1_us,
                    }
                )
                return (0.11, 0.22)

        def xy_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        xy_noise = self._construct(xy_analyzer_builder=xy_analyzer_builder)

        t1_result = xy_noise.cal_t1(is_print=False)
        thermal_result = xy_noise.cal_thermal_exitation(T1=44.0, is_print=False)

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]['couple_term'], xy_noise.couple_term)

        self.assertEqual(t1_result.value, 0.2)
        self.assertEqual(t1_result.fit_diagnostics['gamma_up'], 1.25)
        self.assertEqual(t1_result.fit_diagnostics['gamma_down'], 3.75)
        self.assertEqual(xy_noise.Gamma_up, 1.25)
        self.assertEqual(xy_noise.Gamma_down, 3.75)
        self.assertEqual(xy_noise.T1, 0.2)

        self.assertEqual(thermal_result, (0.11, 0.22))
        self.assertEqual(xy_noise.thermal_exitation, 0.11)
        self.assertEqual(xy_noise.thermal_exitation_onlyxy, 0.22)

        self.assertEqual(
            analyzer_calls,
            [
                {
                    'method': 'calculate_t1',
                    'noise_output': xy_noise.noise.output_stage,
                    'qubit_freq': xy_noise.qubit_freq,
                    'Ej': xy_noise.qubit.Ej[0, 0],
                    'Ec': xy_noise.qubit.Ec[0, 0],
                },
                {
                    'method': 'calculate_thermal_excitation',
                    'gamma_up': 1.25,
                    'gamma_down': 3.75,
                    't1_us': 44.0,
                },
            ],
        )

    def test_xy_facade_auto_derives_transition_rates_before_thermal_excitation_when_t1_is_omitted(self):
        builder_calls = []
        analyzer_calls = []

        class RecordingAnalyzer:
            def calculate_t1(self, *, noise_output, qubit_freq, Ej, Ec):
                analyzer_calls.append(
                    {
                        'method': 'calculate_t1',
                        'noise_output': noise_output,
                        'qubit_freq': qubit_freq,
                        'Ej': Ej,
                        'Ec': Ec,
                    }
                )
                return {
                    'gamma_up': 2.5,
                    'gamma_down': 7.5,
                    't1': 0.1,
                }

            def calculate_thermal_excitation(self, *, gamma_up, gamma_down, t1_us):
                analyzer_calls.append(
                    {
                        'method': 'calculate_thermal_excitation',
                        'gamma_up': gamma_up,
                        'gamma_down': gamma_down,
                        't1_us': t1_us,
                    }
                )
                return (0.33, 0.25)

        def xy_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        xy_noise = self._construct(xy_analyzer_builder=xy_analyzer_builder)

        thermal_result = xy_noise.cal_thermal_exitation(is_print=False)

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]['couple_term'], xy_noise.couple_term)
        self.assertEqual(thermal_result, (0.33, 0.25))
        self.assertEqual(xy_noise.Gamma_up, 2.5)
        self.assertEqual(xy_noise.Gamma_down, 7.5)
        self.assertEqual(xy_noise.T1, 0.1)
        self.assertEqual(xy_noise.thermal_exitation, 0.33)
        self.assertEqual(xy_noise.thermal_exitation_onlyxy, 0.25)
        self.assertEqual(
            analyzer_calls,
            [
                {
                    'method': 'calculate_t1',
                    'noise_output': xy_noise.noise.output_stage,
                    'qubit_freq': xy_noise.qubit_freq,
                    'Ej': xy_noise.qubit.Ej[0, 0],
                    'Ec': xy_noise.qubit.Ec[0, 0],
                },
                {
                    'method': 'calculate_thermal_excitation',
                    'gamma_up': 2.5,
                    'gamma_down': 7.5,
                    't1_us': None,
                },
            ],
        )


if __name__ == '__main__':
    unittest.main()

