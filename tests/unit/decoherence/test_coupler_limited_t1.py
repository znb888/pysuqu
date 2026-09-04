import unittest

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu import decoherence
from pysuqu.decoherence import (
    T1Result,
    estimate_coupler_limited_qubit_t1,
    estimate_coupler_limited_qubit_t1_combined,
)
from pysuqu.decoherence.analysis import (
    estimate_coupler_limited_qubit_t1 as analysis_estimator,
    estimate_coupler_limited_qubit_t1_combined as analysis_combined_estimator,
)


class CouplerLimitedQubitT1Tests(unittest.TestCase):
    def test_estimates_participation_weighted_independent_decay(self):
        result = estimate_coupler_limited_qubit_t1(
            coupler_t1_s=20e-6,
            coupler_participation=0.01,
            coupler_count=4,
        )

        self.assertIsInstance(result, T1Result)
        self.assertEqual(result.unit, 's')
        self.assertAlmostEqual(result.value, 500e-6)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_single_s_inv'], 500.0)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_total_s_inv'], 2000.0)
        self.assertEqual(result.metadata['source'], 'coupler_loss')
        self.assertEqual(result.metadata['assumption'], 'independent_identical_couplers')
        self.assertEqual(result.metadata['coupler_count'], 4)

    def test_scales_inversely_with_count_and_participation(self):
        baseline = estimate_coupler_limited_qubit_t1(
            coupler_t1_s=20e-6,
            coupler_participation=0.01,
        )
        scaled = estimate_coupler_limited_qubit_t1(
            coupler_t1_s=20e-6,
            coupler_participation=0.02,
            coupler_count=2,
        )

        self.assertAlmostEqual(scaled.value, baseline.value / 4)

    def test_zero_participation_returns_infinite_limit(self):
        result = estimate_coupler_limited_qubit_t1(
            coupler_t1_s=20e-6,
            coupler_participation=0.0,
            coupler_count=4,
        )

        self.assertTrue(np.isinf(result.value))
        self.assertEqual(result.fit_diagnostics['gamma_single_s_inv'], 0.0)
        self.assertEqual(result.fit_diagnostics['gamma_total_s_inv'], 0.0)

    def test_rejects_invalid_coupler_t1(self):
        for value in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    estimate_coupler_limited_qubit_t1(
                        coupler_t1_s=value,
                        coupler_participation=0.01,
                    )

    def test_rejects_invalid_participation(self):
        for value in (-0.01, 1.01, np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    estimate_coupler_limited_qubit_t1(
                        coupler_t1_s=20e-6,
                        coupler_participation=value,
                    )

    def test_rejects_invalid_coupler_count(self):
        for value in (0, -1, 1.5, True, np.int64(0)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    estimate_coupler_limited_qubit_t1(
                        coupler_t1_s=20e-6,
                        coupler_participation=0.01,
                        coupler_count=value,
                    )

    def test_public_export_is_analysis_estimator(self):
        self.assertIs(decoherence.estimate_coupler_limited_qubit_t1, analysis_estimator)
        self.assertIs(
            decoherence.estimate_coupler_limited_qubit_t1_combined,
            analysis_combined_estimator,
        )
        self.assertIn('estimate_coupler_limited_qubit_t1', decoherence.__all__)
        self.assertIn('estimate_coupler_limited_qubit_t1_combined', decoherence.__all__)

    def test_combined_t1_adds_gate_and_off_rates(self):
        result = estimate_coupler_limited_qubit_t1_combined(
            t1_gate_s=2e-6,
            t1_off_s=10e-6,
            couplers_per_qubit=4,
        )

        expected_rate = 1 / 2e-6 + 3 / 10e-6
        self.assertIsInstance(result, T1Result)
        self.assertAlmostEqual(result.value, 1 / expected_rate)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_gate_s_inv'], 1 / 2e-6)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_off_s_inv'], 1 / 10e-6)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_total_s_inv'], expected_rate)
        self.assertAlmostEqual(result.fit_diagnostics['qubit_t1_s'], result.value)
        self.assertEqual(result.metadata['gate_coupler_count'], 1)
        self.assertEqual(result.metadata['off_coupler_count'], 3)

    def test_combined_t1_n1_is_gate_t1(self):
        result = estimate_coupler_limited_qubit_t1_combined(
            t1_gate_s=2e-6,
            t1_off_s=10e-6,
            couplers_per_qubit=1,
        )

        self.assertAlmostEqual(result.value, 2e-6)
        self.assertAlmostEqual(result.fit_diagnostics['gamma_total_s_inv'], 1 / 2e-6)

    def test_combined_t1_equal_values_scales_by_coupler_count(self):
        result = estimate_coupler_limited_qubit_t1_combined(
            t1_gate_s=12e-6,
            t1_off_s=12e-6,
            couplers_per_qubit=4,
        )

        self.assertAlmostEqual(result.value, 12e-6 / 4)

    def test_combined_t1_allows_infinite_single_coupler_t1(self):
        result = estimate_coupler_limited_qubit_t1_combined(
            t1_gate_s=np.inf,
            t1_off_s=np.inf,
            couplers_per_qubit=4,
        )

        self.assertTrue(np.isinf(result.value))
        self.assertEqual(result.fit_diagnostics['gamma_total_s_inv'], 0.0)

    def test_combined_t1_rejects_invalid_inputs(self):
        invalid_t1 = (0.0, -1.0, -np.inf, np.nan, None, 'bad')
        for value in invalid_t1:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    estimate_coupler_limited_qubit_t1_combined(
                        t1_gate_s=value,
                        t1_off_s=10e-6,
                        couplers_per_qubit=4,
                    )
                with self.assertRaises(ValueError):
                    estimate_coupler_limited_qubit_t1_combined(
                        t1_gate_s=2e-6,
                        t1_off_s=value,
                        couplers_per_qubit=4,
                    )

        for value in (0, -1, 1.5, True, np.int64(0)):
            with self.subTest(couplers_per_qubit=value):
                with self.assertRaises(ValueError):
                    estimate_coupler_limited_qubit_t1_combined(
                        t1_gate_s=2e-6,
                        t1_off_s=10e-6,
                        couplers_per_qubit=value,
                    )


if __name__ == '__main__':
    unittest.main()
