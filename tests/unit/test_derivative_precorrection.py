import unittest

import numpy as np

import pysuqu.funclib as funclib
from pysuqu.funclib import (
    AttenuatorStage,
    DerivativePrecorrectionDesign,
    DerivativePrecorrectionStage,
    SignalBundle,
    SignalTrace,
    TransmissionChain,
    TransmissionResult,
    apply_derivative_precorrection,
    compute_derivative_basis,
    design_derivative_precorrection,
)
from pysuqu.funclib import transmission


def make_iq_trace(values, *, sample_rate=4.0, plane='awg_iq'):
    values = np.asarray(values, dtype=np.complex128)
    return SignalTrace(
        t_axis=np.arange(len(values), dtype=float) / sample_rate,
        values=values,
        sample_rate=sample_rate,
        domain='iq_complex',
        plane=plane,
        lo_freq=5.0,
        label='template',
        metadata={'source': 'test'},
    )


class DerivativeBasisTests(unittest.TestCase):
    def test_polynomial_derivatives_are_generated_in_requested_order(self):
        t_axis = np.linspace(0.0, 1.0, 11)
        values = t_axis**2

        basis, scales = compute_derivative_basis(
            values,
            sample_period=0.1,
            derivative_orders=(2, 1),
            edge_order=2,
        )

        np.testing.assert_allclose(basis[0], 2.0, atol=1e-12)
        np.testing.assert_allclose(basis[1], 2.0 * t_axis, atol=1e-12)
        np.testing.assert_allclose(scales, [1.0, 1.0])

    def test_peak_normalization_matches_input_or_explicit_reference(self):
        t_axis = np.linspace(0.0, 2 * np.pi, 33)
        values = 2.0 * np.sin(t_axis)

        normalized, scales = compute_derivative_basis(
            values,
            sample_period=t_axis[1] - t_axis[0],
            derivative_orders=(1, 2),
            normalize_to_peak=True,
        )
        explicit, _ = compute_derivative_basis(
            values,
            sample_period=t_axis[1] - t_axis[0],
            derivative_orders=(1,),
            normalize_to_peak=True,
            reference_peak=0.25,
        )

        self.assertTrue(np.all(scales > 0))
        for basis in normalized:
            self.assertAlmostEqual(np.max(np.abs(basis)), 2.0)
        self.assertAlmostEqual(np.max(np.abs(explicit[0])), 0.25)

    def test_constant_waveform_normalizes_zero_derivative_without_nan(self):
        basis, scales = compute_derivative_basis(
            np.ones(8),
            sample_period=0.25,
            derivative_orders=(1,),
            normalize_to_peak=True,
        )

        np.testing.assert_allclose(basis[0], 0.0)
        np.testing.assert_allclose(scales, [0.0])

    def test_short_trace_falls_back_to_first_order_edges(self):
        basis, _ = compute_derivative_basis(
            [0.0, 2.0],
            sample_period=0.5,
            derivative_orders=(1,),
            edge_order=2,
        )

        np.testing.assert_allclose(basis[0], [4.0, 4.0])

    def test_basis_validation_rejects_invalid_inputs(self):
        invalid_calls = (
            lambda: compute_derivative_basis(np.ones((2, 2)), 1.0, (1,)),
            lambda: compute_derivative_basis([1.0, np.nan], 1.0, (1,)),
            lambda: compute_derivative_basis([1.0, 2.0], 0.0, (1,)),
            lambda: compute_derivative_basis([1.0, 2.0], 1.0, (0,)),
            lambda: compute_derivative_basis([1.0, 2.0], 1.0, (1, 1)),
            lambda: compute_derivative_basis([1.0, 2.0], 1.0, (1.5,)),
            lambda: compute_derivative_basis([1.0], 1.0, (1,)),
            lambda: compute_derivative_basis(
                [1.0, 2.0],
                1.0,
                (1,),
                edge_order=3,
            ),
            lambda: compute_derivative_basis(
                [1.0, 2.0],
                1.0,
                (1,),
                reference_peak=-1.0,
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class ApplyDerivativePrecorrectionTests(unittest.TestCase):
    def test_weighted_derivatives_and_phase_match_manual_result(self):
        t_axis = np.linspace(0.0, 1.0, 11)
        values = t_axis**2 + 0.5j * t_axis
        coefficients = np.array([0.2 - 0.1j, -0.03])
        basis, _ = compute_derivative_basis(
            values,
            sample_period=0.1,
            derivative_orders=(1, 2),
        )
        phase = 0.17

        corrected, metadata = apply_derivative_precorrection(
            values,
            sample_period=0.1,
            coefficients=coefficients,
            derivative_orders=(1, 2),
            normalize_to_peak=False,
            phase_rad=phase,
        )

        expected = (values + coefficients[0] * basis[0] + coefficients[1] * basis[1])
        expected *= np.exp(1j * phase)
        np.testing.assert_allclose(corrected, expected)
        self.assertEqual(metadata['derivative_orders'], [1, 2])
        np.testing.assert_allclose(metadata['coefficients_real'], [0.2, -0.03])
        np.testing.assert_allclose(metadata['coefficients_imag'], [-0.1, 0.0])
        self.assertEqual(metadata['normalize_to_peak'], False)

    def test_default_orders_empty_coefficients_and_phase_only(self):
        values = np.array([1.0, 2.0, 3.0])
        corrected, metadata = apply_derivative_precorrection(
            values,
            sample_period=0.5,
            coefficients=[],
            phase_rad=np.pi / 2,
        )

        np.testing.assert_allclose(corrected, 1j * values, atol=1e-15)
        self.assertEqual(metadata['derivative_orders'], [])
        self.assertEqual(metadata['coefficients_real'], [])
        self.assertEqual(metadata['coefficients_imag'], [])

    def test_apply_validation_rejects_mismatches_and_nonfinite_values(self):
        invalid_calls = (
            lambda: apply_derivative_precorrection([1.0, 2.0], 1.0, [0.1], derivative_orders=(1, 2)),
            lambda: apply_derivative_precorrection([1.0, 2.0], 1.0, [np.inf]),
            lambda: apply_derivative_precorrection([1.0, 2.0], 1.0, [], phase_rad=np.nan),
            lambda: apply_derivative_precorrection([1.0, 2.0], np.inf, []),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class DerivativePrecorrectionDesignTests(unittest.TestCase):
    def test_design_recovers_known_derivative_coefficient(self):
        rng = np.random.default_rng(2468)
        values = rng.normal(size=32) + 1j * rng.normal(size=32)
        trace = make_iq_trace(values)
        basis, _ = compute_derivative_basis(
            values,
            sample_period=0.25,
            derivative_orders=(1,),
        )
        expected_coefficient = 0.18 - 0.07j
        predistorted = values + expected_coefficient * basis[0]
        response = np.fft.fft(values) / np.fft.fft(predistorted)

        design = design_derivative_precorrection(
            trace,
            response,
            derivative_orders=(1,),
            normalize_to_peak=False,
            spectral_weight_power=0.0,
            ridge=0.0,
            include_global_phase=False,
        )

        self.assertIsInstance(design, DerivativePrecorrectionDesign)
        np.testing.assert_allclose(
            design.coefficients,
            [expected_coefficient],
            atol=1e-12,
        )
        self.assertLess(design.residual_rms, 1e-12)
        self.assertEqual(design.derivative_orders, (1,))

    def test_global_phase_alignment_corrects_constant_phase_response(self):
        trace = make_iq_trace(np.ones(16))
        phase_error = 0.37
        response = np.full(16, np.exp(1j * phase_error))

        without_phase = design_derivative_precorrection(
            trace,
            response,
            derivative_orders=(1,),
            include_global_phase=False,
        )
        with_phase = design_derivative_precorrection(
            trace,
            response,
            derivative_orders=(1,),
            include_global_phase=True,
        )

        self.assertAlmostEqual(with_phase.phase_rad, -phase_error)
        self.assertLess(with_phase.residual_rms, 1e-12)
        self.assertGreater(without_phase.residual_rms, with_phase.residual_rms)

    def test_design_accepts_multiple_paths_and_records_fit_arrays(self):
        trace = make_iq_trace(np.hanning(16))
        responses = np.vstack([np.ones(16), 0.9 * np.ones(16)])

        design = design_derivative_precorrection(
            trace,
            responses,
            derivative_orders=(1, 2),
        )

        self.assertEqual(design.coefficients.shape, (2,))
        self.assertEqual(design.basis_scales.shape, (2,))
        self.assertEqual(design.spectral_weights.shape, (16,))
        self.assertTrue(np.isfinite(design.residual_rms))

    def test_design_validation_is_explicit(self):
        iq_trace = make_iq_trace(np.ones(8))
        rf_trace = SignalTrace(
            np.arange(8.0),
            np.ones(8),
            1.0,
            'rf_real',
            'awg_rf',
        )
        invalid_calls = (
            lambda: design_derivative_precorrection(object(), np.ones(8), derivative_orders=(1,)),
            lambda: design_derivative_precorrection(rf_trace, np.ones(8), derivative_orders=(1,)),
            lambda: design_derivative_precorrection(iq_trace, [], derivative_orders=(1,)),
            lambda: design_derivative_precorrection(iq_trace, np.ones(8), derivative_orders=()),
            lambda: design_derivative_precorrection(iq_trace, np.ones(8), derivative_orders=(1.5,)),
            lambda: design_derivative_precorrection(iq_trace, np.full(8, np.nan), derivative_orders=(1,)),
            lambda: design_derivative_precorrection(iq_trace, np.ones(8), derivative_orders=(1,), ridge=-1.0),
            lambda: design_derivative_precorrection(
                iq_trace,
                np.ones(8),
                derivative_orders=(1,),
                spectral_weight_power=np.inf,
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_design_payload_validates_manual_construction(self):
        with self.assertRaisesRegex(ValueError, 'basis_scales'):
            DerivativePrecorrectionDesign(
                coefficients=[0.1, 0.2],
                basis_scales=[1.0],
            )
        with self.assertRaisesRegex(ValueError, 'spectral_weights'):
            DerivativePrecorrectionDesign(
                coefficients=[0.1],
                spectral_weights=[1.0, -1.0],
            )


class DerivativePrecorrectionStageTests(unittest.TestCase):
    def test_stage_matches_helper_and_preserves_trace_context(self):
        trace = make_iq_trace(np.linspace(0.0, 1.0, 9) ** 2)
        stage = DerivativePrecorrectionStage(
            coefficients=[0.2 + 0.1j],
            derivative_orders=(1,),
            normalize_to_peak=False,
            phase_rad=0.05,
            name='line_predistortion',
        )
        expected, _ = apply_derivative_precorrection(
            trace.values,
            1.0 / trace.sample_rate,
            stage.coefficients,
            derivative_orders=(1,),
            normalize_to_peak=False,
            phase_rad=0.05,
        )

        output = stage.apply(trace)

        np.testing.assert_allclose(output.values, expected)
        self.assertEqual(output.plane, trace.plane)
        self.assertEqual(output.metadata['source'], 'test')
        self.assertEqual(output.metadata['last_stage'], 'line_predistortion')
        self.assertIn('orders=1', stage.describe())

    def test_stage_from_design_and_chain_history(self):
        design = DerivativePrecorrectionDesign(
            coefficients=np.array([0.1j]),
            derivative_orders=(2,),
            phase_rad=0.2,
            normalize_to_peak=False,
            edge_order=1,
        )
        stage = DerivativePrecorrectionStage.from_design(design, name='fitted')
        chain = TransmissionChain(
            stages=[stage, AttenuatorStage(loss_db=20.0)]
        )

        result = chain.apply(
            make_iq_trace(np.hanning(12)),
            capture_history=True,
        )

        self.assertIsInstance(result, TransmissionResult)
        self.assertEqual(len(result.stage_outputs), 2)
        self.assertEqual(stage.derivative_orders, (2,))
        self.assertEqual(stage.edge_order, 1)
        np.testing.assert_allclose(
            result.output_trace.values,
            0.1 * result.stage_outputs[0].values,
        )

    def test_stage_maps_over_signal_bundles(self):
        stage = DerivativePrecorrectionStage(
            coefficients=[0.1],
            normalize_to_peak=False,
        )
        bundle = SignalBundle(
            {
                'x': make_iq_trace(np.arange(8.0)),
                'y': make_iq_trace(2.0 * np.arange(8.0)),
            }
        )

        output = transmission.BundleTransmissionChain(stages=[stage]).apply(bundle)

        self.assertEqual(output.order, ('x', 'y'))
        self.assertEqual(output['x'].metadata['last_stage'], stage.name)
        self.assertEqual(output['y'].metadata['last_stage'], stage.name)

    def test_stage_rejects_rf_and_invalid_configuration(self):
        rf_trace = SignalTrace(
            np.arange(4.0),
            np.ones(4),
            1.0,
            'rf_real',
            'awg_rf',
        )
        with self.assertRaisesRegex(ValueError, 'expects iq_complex'):
            DerivativePrecorrectionStage(coefficients=[0.1]).apply(rf_trace)

        invalid = (
            lambda: DerivativePrecorrectionStage(coefficients=[0.1], derivative_orders=(1, 2)),
            lambda: DerivativePrecorrectionStage(coefficients=[np.inf]),
            lambda: DerivativePrecorrectionStage(coefficients=[0.1], edge_order=3),
            lambda: DerivativePrecorrectionStage(coefficients=[0.1], phase_rad=np.nan),
            lambda: DerivativePrecorrectionStage.from_design(object()),
        )
        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

    def test_public_exports_include_derivative_precorrection_api(self):
        names = (
            'DerivativePrecorrectionDesign',
            'DerivativePrecorrectionStage',
            'apply_derivative_precorrection',
            'compute_derivative_basis',
            'design_derivative_precorrection',
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, transmission.__all__)
                self.assertIs(getattr(funclib, name), getattr(transmission, name))


if __name__ == '__main__':
    unittest.main()
