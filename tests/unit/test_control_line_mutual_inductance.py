import unittest

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import (
    Phi0,
    bias_from_control_frequency,
    control_frequency_from_bias,
    control_frequency_slope,
    estimate_control_line_mutual_inductance,
    estimate_mutual_inductance_from_kf,
    fit_mz,
)


class ControlLineMutualInductanceTests(unittest.TestCase):
    def test_frequency_model_round_trips_on_principal_branch(self):
        bias = np.array([0.0, 0.1, 0.2])
        frequencies = control_frequency_from_bias(
            bias,
            max_frequency=5.0,
            bias_offset=0.0,
            flux_per_bias=0.8,
            minimum_frequency=0.24,
        )

        recovered = bias_from_control_frequency(
            frequencies,
            max_frequency=5.0,
            bias_offset=0.0,
            flux_per_bias=0.8,
            minimum_frequency=0.24,
        )

        np.testing.assert_allclose(recovered, bias)

    def test_model_slope_matches_finite_difference(self):
        bias = 0.17
        slope = control_frequency_slope(bias, 5.0, 0.0, 0.8, 0.24)
        step = 1e-7
        numerical = (
            control_frequency_from_bias(bias + step, 5.0, 0.0, 0.8, 0.24)
            - control_frequency_from_bias(bias - step, 5.0, 0.0, 0.8, 0.24)
        ) / (2 * step)
        self.assertAlmostEqual(slope, numerical, places=6)

    def test_mutual_inductance_converts_ghz_sensitivity_without_extra_two_pi(self):
        expected = 0.3 / 1.5 * 50.0 * Phi0 * 10.0 / 2.0
        result = estimate_control_line_mutual_inductance(
            frequency_slope=0.3,
            sensitivity=1.5,
            frequency_unit='GHz',
            sensitivity_unit='GHz/Phi0',
            attenuation_db=20.0,
        )
        self.assertAlmostEqual(result, expected)

        angular_result = estimate_control_line_mutual_inductance(
            frequency_slope=0.3,
            sensitivity=1.5 * 2 * np.pi * 1e9 / Phi0,
            frequency_unit='GHz',
            sensitivity_unit='rad/s/Wb',
            attenuation_db=20.0,
        )
        self.assertAlmostEqual(angular_result, expected)

    def test_kf_direct_route_agrees_with_frequency_slope_route(self):
        expected = 0.8 / 1.5 * 50.0 * Phi0 * 10.0 / 2.0
        result = estimate_mutual_inductance_from_kf(
            0.8,
            bias_voltage_per_unit=1.5,
            attenuation_db=20.0,
        )
        self.assertAlmostEqual(result, expected)

    def test_fit_mz_uses_explicit_sensitivity_and_pysuqu_ghz_defaults(self):
        coefficients = (5.0, 0.0, 0.8, 0.24)
        selected_bias = bias_from_control_frequency(4.5, *coefficients)
        slope = control_frequency_slope(selected_bias, *coefficients)
        expected = estimate_control_line_mutual_inductance(
            slope,
            sensitivity=1.5,
            attenuation_db=20.0,
        )
        result = fit_mz(
            coefficients,
            sensitivity=1.5,
            att_total=20.0,
            delta_select=0.5,
        )
        self.assertAlmostEqual(result, expected)

        hz_result = fit_mz(
            (5e9, 0.0, 0.8, 0.24e9),
            sensitivity=1.5e9,
            att_total=20.0,
            frequency_unit='Hz',
            sensitivity_unit='Hz/Phi0',
        )
        self.assertAlmostEqual(hz_result, expected)

    def test_fit_mz_accepts_abstract_qubit_sensitivity_tuple_at_same_working_point(self):
        coefficients = (5.0, 0.0, 0.8, 0.24)
        operating_frequency = 4.6
        operating_bias = bias_from_control_frequency(operating_frequency, *coefficients)
        expected = estimate_control_line_mutual_inductance(
            control_frequency_slope(operating_bias, *coefficients),
            sensitivity=1.5,
            attenuation_db=20.0,
        )
        result = fit_mz(
            coefficients,
            sensitivity=(1.5, operating_frequency),
            att_total=20.0,
        )
        self.assertAlmostEqual(result, expected)

    def test_invalid_inputs_fail_with_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, 'sensitivity is required'):
            fit_mz((5.0, 0.0, 0.8, 0.24))
        with self.assertRaisesRegex(ValueError, 'between minimum_frequency'):
            bias_from_control_frequency(5.1, 5.0, 0.0, 0.8, 0.24)
        with self.assertRaisesRegex(ValueError, 'non-zero'):
            estimate_control_line_mutual_inductance(0.2, 0.0)


if __name__ == '__main__':
    unittest.main()
