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
from pysuqu.decoherence.results import XYCurrentVoltageResult
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

    def test_xy_current_voltage_keeps_existing_numeric_formula(self):
        phi_fraction = 0.015 / (4 * np.pi)
        attenuation_setup = np.array([2.0, 6.5, 10.0, 13.0, 1.0, 0.0])
        xy_noise = self._construct(attenuation_setup=attenuation_setup)

        phi_bias = phi_fraction * Phi0
        chip_current_A = phi_bias / xy_noise.couple_term / 2
        chip_current_uA = chip_current_A * 1e6
        chip_voltage_uV = chip_current_uA * 50
        chip_power_W = chip_current_A**2 * 50
        chip_power_dBm = 10 * np.log10(chip_power_W / 1e-3)
        total_attenuation_dB = np.sum(attenuation_setup)
        chip_current_mA = chip_current_A * 1e3
        room_current_mA = chip_current_mA * 10 ** (total_attenuation_dB / 20)
        room_voltage_mV = room_current_mA * 50
        room_power_W = room_current_mA**2 * 50 / 2 / 1e3
        room_power_dBm = 10 * np.log10(room_power_W)

        actual = xy_noise.cal_xy_current_voltage(phi_fraction=phi_fraction, is_print=False)

        self.assertEqual(set(actual), set(XYCurrentVoltageResult.__annotations__))
        self.assertAlmostEqual(actual['phi_bias'], phi_bias)
        self.assertAlmostEqual(actual['chip_current_uA'], chip_current_uA)
        self.assertAlmostEqual(actual['chip_voltage_uV'], chip_voltage_uV)
        self.assertAlmostEqual(actual['chip_power_dBm'], chip_power_dBm)
        self.assertAlmostEqual(actual['total_attenuation_dB'], total_attenuation_dB)
        self.assertAlmostEqual(actual['room_current_mA'], room_current_mA)
        self.assertAlmostEqual(actual['room_voltage_mV'], room_voltage_mV)
        self.assertAlmostEqual(actual['room_power_dBm'], room_power_dBm)

    def test_xy_current_voltage_can_delegate_through_explicit_xy_analyzer_builder(self):
        builder_calls = []
        analyzer_calls = []
        expected = {
            'phi_bias': 1.0,
            'chip_current_uA': 2.0,
            'chip_voltage_uV': 3.0,
            'chip_power_dBm': 4.0,
            'total_attenuation_dB': 5.0,
            'room_current_mA': 6.0,
            'room_voltage_mV': 7.0,
            'room_power_dBm': 8.0,
        }

        class RecordingAnalyzer:
            def calculate_t1(self, *, noise_output, qubit_freq, Ej, Ec):
                return {
                    'gamma_up': 1.0,
                    'gamma_down': 2.0,
                    't1': 0.5,
                }

            def calculate_thermal_excitation(self, *, gamma_up, gamma_down, t1_us):
                return (0.0, 0.0)

            def calculate_xy_current_voltage(self, *, phi_fraction, attenuation_setup):
                analyzer_calls.append(
                    {
                        'phi_fraction': phi_fraction,
                        'attenuation_setup': np.array(attenuation_setup, copy=True),
                    }
                )
                return dict(expected)

        def xy_analyzer_builder(**kwargs):
            builder_calls.append(dict(kwargs))
            return RecordingAnalyzer()

        attenuation_setup = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        xy_noise = self._construct(
            attenuation_setup=attenuation_setup,
            xy_analyzer_builder=xy_analyzer_builder,
        )

        actual = xy_noise.cal_xy_current_voltage(phi_fraction=0.02, is_print=False)

        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]['couple_term'], xy_noise.couple_term)
        self.assertEqual(set(actual), set(XYCurrentVoltageResult.__annotations__))
        self.assertEqual(actual, expected)
        self.assertEqual(len(analyzer_calls), 1)
        self.assertEqual(analyzer_calls[0]['phi_fraction'], 0.02)
        np.testing.assert_array_equal(analyzer_calls[0]['attenuation_setup'], attenuation_setup)


if __name__ == '__main__':
    unittest.main()
