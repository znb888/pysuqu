"""Synthetic contracts for the public predistortion design helpers."""

import unittest

import numpy as np

from pysuqu.funclib.predistortion import (
    apply_derivative_precorrection,
    compute_derivative_basis,
    design_sampled_fir_inverse,
    discretize_zoh_state_space,
    evaluate_discrete_state_space_response,
    propagate_zoh_discrete,
)


class PredistortionApiTests(unittest.TestCase):
    def test_causal_derivative_and_phase_metadata(self):
        values = np.linspace(0.0, 1.0, 12, dtype=np.complex128)
        basis, scales = compute_derivative_basis(
            values,
            0.25,
            [1, 2],
            scheme="backward",
            normalize_to_peak=True,
        )
        np.testing.assert_allclose(basis[0][1:], 1.0 + 0.0j)
        self.assertEqual(scales.shape, (2,))
        corrected, metadata = apply_derivative_precorrection(
            values,
            0.25,
            [0.05j],
            scheme="backward",
            phase_rad=0.2,
        )
        self.assertEqual(corrected.shape, values.shape)
        self.assertEqual(metadata["derivative_orders"], [1])
        self.assertAlmostEqual(metadata["phase_rad"], 0.2)

    def test_zoh_model_response_and_propagation(self):
        model = discretize_zoh_state_space(
            np.array([[-0.3]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[0.0]]),
            0.1,
        )
        frequencies = np.linspace(-0.5, 0.5, 9)
        response = evaluate_discrete_state_space_response(model, frequencies, 5.0)
        self.assertEqual(response.shape, (9, 1, 1))
        propagated = propagate_zoh_discrete(model, np.ones(6))
        self.assertEqual(propagated.shape, (6,))
        self.assertTrue(np.all(np.isfinite(propagated)))

    def test_sampled_fir_inverse_has_finite_kernel(self):
        frequencies = np.fft.fftfreq(32, d=0.2)
        response = 0.8 + 0.1j * frequencies
        design = design_sampled_fir_inverse(
            response,
            sample_rate=5.0,
            num_taps=7,
            regularization=1e-6,
        )
        self.assertEqual(design.kernel.shape, (7,))
        self.assertTrue(np.all(np.isfinite(design.kernel)))
        self.assertGreaterEqual(design.delay_samples, 0)


if __name__ == "__main__":
    unittest.main()
