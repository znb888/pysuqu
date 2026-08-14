import unittest

import numpy as np
from scipy.signal import lfilter, sosfilt

from pysuqu.funclib import (
    AttenuatorStage,
    BaseTransmissionStage,
    DelayStage,
    FIRFilterStage,
    IIRFilterStage,
    SignalTrace,
    SOSFilterStage,
    TransferFunctionStage,
    TransmissionChain,
    TransmissionResult,
    WaveformGenerator,
)


class SignalTraceTests(unittest.TestCase):
    def test_domain_controls_value_dtype(self):
        t_axis = np.arange(3.0)

        rf_trace = SignalTrace(t_axis, [1, 2, 3], 1.0, 'rf_real', 'awg_rf')
        iq_trace = SignalTrace(t_axis, [1, 2, 3], 1.0, 'iq_complex', 'awg_iq')

        self.assertEqual(rf_trace.values.dtype, np.float64)
        self.assertEqual(iq_trace.values.dtype, np.complex128)

    def test_rejects_invalid_shape_length_rate_domain_and_plane(self):
        cases = (
            (
                (np.ones((2, 2)), np.ones(4), 1.0, 'rf_real', 'awg_rf'),
                '1D t_axis',
            ),
            ((np.arange(2.0), np.ones(3), 1.0, 'rf_real', 'awg_rf'), 'length mismatch'),
            ((np.arange(2.0), np.ones(2), 0.0, 'rf_real', 'awg_rf'), 'must be positive'),
            ((np.arange(2.0), np.ones(2), 1.0, 'invalid', 'awg_rf'), 'signal domain'),
            ((np.arange(2.0), np.ones(2), 1.0, 'rf_real', 'invalid'), 'signal plane'),
        )

        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SignalTrace(*arguments)

    def test_rejects_complex_rf_values(self):
        with self.assertRaisesRegex(ValueError, 'complex values'):
            SignalTrace(
                np.arange(2.0),
                np.array([1.0 + 1.0j, 0.0]),
                1.0,
                'rf_real',
                'awg_rf',
            )


class TransmissionStageTests(unittest.TestCase):
    @staticmethod
    def _rf_trace(values=(1.0, 0.0, 0.0, 0.0)):
        return SignalTrace(
            t_axis=np.arange(4.0),
            values=np.asarray(values),
            sample_rate=1.0,
            domain='rf_real',
            plane='awg_rf',
            metadata={'source': 'test'},
        )

    def test_attenuator_uses_voltage_scaling_and_updates_metadata(self):
        output = AttenuatorStage(loss_db=20.0, name='line_loss').apply(
            self._rf_trace((2.0, -1.0, 0.0, 4.0))
        )

        np.testing.assert_allclose(output.values, [0.2, -0.1, 0.0, 0.4])
        self.assertEqual(output.metadata['source'], 'test')
        self.assertEqual(output.metadata['last_stage'], 'line_loss')
        self.assertEqual(output.metadata['loss_db'], 20.0)

    def test_stage_validates_domain_and_reference_plane(self):
        iq_only = AttenuatorStage(domain='iq_complex')
        baseband_only = AttenuatorStage(allowed_planes=('baseband',))

        with self.assertRaisesRegex(ValueError, 'expects iq_complex'):
            iq_only.apply(self._rf_trace())
        with self.assertRaisesRegex(ValueError, 'does not accept traces on plane awg_rf'):
            baseband_only.apply(self._rf_trace())

    def test_output_plane_override_is_applied(self):
        output = AttenuatorStage(loss_db=0.0, output_plane='qubit_rf').apply(
            self._rf_trace()
        )

        self.assertEqual(output.plane, 'qubit_rf')

    def test_delay_zero_fills_real_and_complex_traces(self):
        real_output = DelayStage(delay_ns=1.0).apply(self._rf_trace((1.0, 2.0, 0.0, 0.0)))
        complex_trace = SignalTrace(
            np.arange(4.0),
            np.array([1.0 + 2.0j, 3.0 + 4.0j, 0.0, 0.0]),
            1.0,
            'iq_complex',
            'awg_iq',
        )
        complex_output = DelayStage(delay_ns=1.0).apply(complex_trace)

        np.testing.assert_allclose(real_output.values, [0.0, 1.0, 2.0, 0.0])
        np.testing.assert_allclose(
            complex_output.values,
            [0.0, 1.0 + 2.0j, 3.0 + 4.0j, 0.0],
        )

    def test_zero_delay_still_applies_output_plane_and_metadata(self):
        output = DelayStage(delay_ns=0.0, output_plane='qubit_rf').apply(
            self._rf_trace()
        )

        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(output.metadata['last_stage'], 'delay')
        self.assertEqual(output.metadata['delay_ns'], 0.0)


class TransferFunctionStageTests(unittest.TestCase):
    @staticmethod
    def _trace(values, *, domain='rf_real', plane='awg_rf', sample_rate=1.0):
        return SignalTrace(
            np.arange(len(values), dtype=float) / sample_rate,
            np.asarray(values),
            sample_rate,
            domain,
            plane,
        )

    def test_impulse_response_matches_linear_convolution_without_wraparound(self):
        values = np.array([0.0, 0.0, 0.0, 1.0])
        impulse = np.array([1.0, 2.0, 3.0])
        stage = TransferFunctionStage.from_impulse_response(impulse)

        output = stage.apply(self._trace(values))

        np.testing.assert_allclose(
            output.values,
            np.convolve(values, impulse)[:4],
            atol=1e-15,
        )
        np.testing.assert_allclose(output.values[:3], 0.0, atol=1e-15)
        self.assertEqual(output.metadata['fft_length'], 8)

    def test_callable_receives_fft_frequency_axis_in_ghz(self):
        captured = []

        def response(freq_axis):
            captured.append(freq_axis.copy())
            return np.ones_like(freq_axis)

        trace = self._trace(
            [1.0 + 1.0j, 2.0, 0.0, 0.0],
            domain='iq_complex',
            plane='awg_iq',
            sample_rate=2.0,
        )
        output = TransferFunctionStage(H=response).apply(trace)

        np.testing.assert_allclose(captured[0], np.fft.fftfreq(8, d=0.5))
        np.testing.assert_allclose(output.values, trace.values)

    def test_scalar_response_is_broadcast_over_the_fft_grid(self):
        trace = self._trace([1.0 + 2.0j, -2.0j], domain='iq_complex', plane='awg_iq')

        output = TransferFunctionStage(H=lambda freq: 0.5).apply(trace)

        np.testing.assert_allclose(output.values, 0.5 * trace.values)

    def test_first_order_filters_accept_real_rf_traces(self):
        trace = self._trace([1.0, 0.5, -0.25, 0.125])

        lowpass = TransferFunctionStage.first_order_lowpass(cutoff_freq=0.25).apply(trace)
        highpass = TransferFunctionStage.first_order_highpass(cutoff_freq=0.25).apply(trace)

        self.assertFalse(np.iscomplexobj(lowpass.values))
        self.assertFalse(np.iscomplexobj(highpass.values))
        self.assertTrue(np.all(np.isfinite(lowpass.values)))
        self.assertTrue(np.all(np.isfinite(highpass.values)))

    def test_real_trace_still_rejects_non_hermitian_response(self):
        trace = self._trace([1.0, 0.0, 0.0, 0.0])
        stage = TransferFunctionStage(H=lambda freq: np.where(freq > 0, 1.0, 0.0))

        with self.assertRaisesRegex(ValueError, 'complex values'):
            stage.apply(trace)

    def test_response_shape_errors_are_explicit(self):
        trace = self._trace([1.0, 0.0, 0.0, 0.0], domain='iq_complex', plane='awg_iq')

        with self.assertRaisesRegex(ValueError, 'scalar or 1D'):
            TransferFunctionStage(H=np.ones((8, 1))).apply(trace)
        with self.assertRaisesRegex(ValueError, 'return length 8'):
            TransferFunctionStage(H=np.ones(7)).apply(trace)

    def test_factory_inputs_are_validated(self):
        for impulse in ([], np.ones((2, 2))):
            with self.subTest(impulse=np.asarray(impulse).shape):
                with self.assertRaisesRegex(ValueError, 'non-empty 1D'):
                    TransferFunctionStage.from_impulse_response(impulse)

        for cutoff in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(cutoff=cutoff):
                with self.assertRaisesRegex(ValueError, 'positive and finite'):
                    TransferFunctionStage.first_order_lowpass(cutoff_freq=cutoff)

    def test_empty_trace_still_applies_output_plane(self):
        trace = self._trace([])

        output = TransferFunctionStage(output_plane='qubit_rf').apply(trace)

        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(output.metadata['fft_length'], 0)


class DigitalFilterStageTests(unittest.TestCase):
    @staticmethod
    def _trace(values, *, sample_rate=2.0, domain='rf_real', plane='awg_rf'):
        return SignalTrace(
            np.arange(len(values), dtype=float) / sample_rate,
            np.asarray(values),
            sample_rate,
            domain,
            plane,
        )

    def test_fir_leading_alignment_matches_legacy_awgenerator(self):
        values = np.array([1.0, -0.5, 0.25, 2.0, 0.0])
        kernel = np.array([0.2, 0.5, 0.3])
        generator = WaveformGenerator(total_time=2.5, sample_rate=2.0)
        expected = generator._apply_fir_filter(values, kernel)

        output = FIRFilterStage(kernel=kernel).apply(self._trace(values))

        np.testing.assert_allclose(output.values, expected)
        self.assertEqual(output.metadata['alignment'], 'leading')
        self.assertEqual(output.metadata['kernel_length'], 3)

    def test_fir_centered_alignment_removes_integer_group_delay(self):
        values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        kernel = np.array([0.25, 0.5, 0.25])
        full = np.convolve(values, kernel, mode='full')

        output = FIRFilterStage(kernel=kernel, alignment='centered').apply(
            self._trace(values)
        )

        np.testing.assert_allclose(output.values, full[1:6])

    def test_empty_fir_kernel_is_identity_but_still_finalizes_trace(self):
        trace = self._trace([1.0, 2.0])

        output = FIRFilterStage(output_plane='qubit_rf').apply(trace)

        np.testing.assert_allclose(output.values, trace.values)
        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(output.metadata['kernel_length'], 0)

    def test_windowed_fir_factories_build_expected_filter_types(self):
        lowpass = FIRFilterStage.lowpass(
            cutoff_freq=0.2,
            sample_rate=2.0,
            num_taps=17,
        )
        highpass = FIRFilterStage.highpass(
            cutoff_freq=0.2,
            sample_rate=2.0,
            num_taps=17,
        )
        bandpass = FIRFilterStage.bandpass(
            cutoff_freq=(0.2, 0.5),
            sample_rate=2.0,
            num_taps=17,
        )
        bandstop = FIRFilterStage.from_windowed_sinc(
            cutoff_freq=(0.2, 0.5),
            sample_rate=2.0,
            num_taps=17,
            filter_kind='notch',
        )

        def endpoint_gains(stage):
            tap_indices = np.arange(len(stage.kernel))
            return (
                abs(np.sum(stage.kernel)),
                abs(np.sum(stage.kernel * (-1.0) ** tap_indices)),
            )

        low_dc, low_nyquist = endpoint_gains(lowpass)
        high_dc, high_nyquist = endpoint_gains(highpass)
        band_dc, band_nyquist = endpoint_gains(bandpass)
        stop_dc, stop_nyquist = endpoint_gains(bandstop)

        self.assertGreater(low_dc, 0.9)
        self.assertLess(low_nyquist, 0.05)
        self.assertLess(high_dc, 0.05)
        self.assertGreater(high_nyquist, 0.9)
        self.assertLess(band_dc, 0.05)
        self.assertLess(band_nyquist, 0.05)
        self.assertGreater(stop_dc, 0.9)
        self.assertGreater(stop_nyquist, 0.9)
        self.assertEqual(lowpass.sample_rate, 2.0)

    def test_iir_and_sos_match_scipy_reference_filters(self):
        values = np.array([1.0, 0.0, 0.5, -0.25, 0.0, 0.0])
        trace = self._trace(values)
        iir = IIRFilterStage.butterworth(
            order=3,
            cutoff_freq=0.25,
            sample_rate=2.0,
        )
        sos = SOSFilterStage.butterworth(
            order=4,
            cutoff_freq=(0.15, 0.45),
            sample_rate=2.0,
            filter_kind='bandpass',
        )

        iir_output = iir.apply(trace)
        sos_output = sos.apply(trace)

        np.testing.assert_allclose(iir_output.values, lfilter(iir.b, iir.a, values))
        np.testing.assert_allclose(sos_output.values, sosfilt(sos.sos, values))
        self.assertEqual(iir_output.metadata['design_sample_rate'], 2.0)
        self.assertEqual(sos_output.metadata['num_sections'], 4)

    def test_factory_designed_filters_reject_trace_sample_rate_mismatch(self):
        stages = (
            FIRFilterStage.lowpass(cutoff_freq=0.2, sample_rate=2.0),
            IIRFilterStage.butterworth(
                order=2,
                cutoff_freq=0.2,
                sample_rate=2.0,
            ),
            SOSFilterStage.butterworth(
                order=2,
                cutoff_freq=0.2,
                sample_rate=2.0,
            ),
        )
        mismatched = self._trace([1.0, 0.0], sample_rate=1.0)

        for stage in stages:
            with self.subTest(stage=stage.name):
                with self.assertRaisesRegex(ValueError, 'designed for sample_rate=2.0'):
                    stage.apply(mismatched)

    def test_direct_coefficients_remain_sample_rate_agnostic(self):
        trace = self._trace([1.0, 2.0], sample_rate=7.5)
        stages = (
            FIRFilterStage(kernel=[1.0]),
            IIRFilterStage(b=[1.0], a=[1.0]),
            SOSFilterStage(sos=[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        )

        for stage in stages:
            with self.subTest(stage=stage.name):
                np.testing.assert_allclose(stage.apply(trace).values, trace.values)

    def test_filter_coefficients_and_alignment_are_validated(self):
        invalid_constructors = (
            lambda: FIRFilterStage(kernel=np.ones((2, 2))),
            lambda: FIRFilterStage(kernel=[1.0], alignment='invalid'),
            lambda: IIRFilterStage(b=[], a=[1.0]),
            lambda: IIRFilterStage(b=[1.0], a=np.ones((1, 1))),
            lambda: SOSFilterStage(sos=np.empty((0, 6))),
            lambda: SOSFilterStage(sos=np.ones((2, 5))),
        )

        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_filter_design_parameters_are_validated(self):
        invalid_designs = (
            lambda: FIRFilterStage.lowpass(cutoff_freq=0.0, sample_rate=2.0),
            lambda: FIRFilterStage.lowpass(cutoff_freq=np.nan, sample_rate=2.0),
            lambda: FIRFilterStage.bandpass(
                cutoff_freq=(0.5, 0.2),
                sample_rate=2.0,
            ),
            lambda: IIRFilterStage.butterworth(
                order=2,
                cutoff_freq=1.0,
                sample_rate=2.0,
            ),
            lambda: SOSFilterStage.butterworth(
                order=2,
                cutoff_freq=0.2,
                sample_rate=np.inf,
            ),
        )

        for design in invalid_designs:
            with self.subTest(design=design):
                with self.assertRaises(ValueError):
                    design()

    def test_empty_iir_and_sos_traces_preserve_output_plane(self):
        trace = self._trace([])
        stages = (
            IIRFilterStage(output_plane='qubit_rf'),
            SOSFilterStage(output_plane='qubit_rf'),
        )

        for stage in stages:
            with self.subTest(stage=stage.name):
                output = stage.apply(trace)
                self.assertEqual(len(output.values), 0)
                self.assertEqual(output.plane, 'qubit_rf')


class TransmissionChainTests(unittest.TestCase):
    @staticmethod
    def _impulse():
        return SignalTrace(
            np.arange(4.0),
            np.array([2.0, 0.0, 0.0, 0.0]),
            1.0,
            'rf_real',
            'awg_rf',
        )

    def test_chain_applies_stages_in_order_and_captures_history(self):
        chain = TransmissionChain(
            name='xy_line',
            stages=[AttenuatorStage(loss_db=20.0), DelayStage(delay_ns=1.0)],
        )

        result = chain.apply(self._impulse(), capture_history=True)

        self.assertIsInstance(result, TransmissionResult)
        np.testing.assert_allclose(result.stage_outputs[0].values, [0.2, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result.output_trace.values, [0.0, 0.2, 0.0, 0.0])
        self.assertEqual(len(result.stage_outputs), 2)
        self.assertIn('attenuator[20.000 dB] -> delay[1.000 ns]', chain.describe())

    def test_append_extend_and_empty_description(self):
        chain = TransmissionChain(name='line')
        self.assertEqual(chain.describe(), 'line (0 stage(s))')

        chain.append(AttenuatorStage(loss_db=3.0))
        chain.extend([DelayStage(delay_ns=0.5)])

        self.assertEqual(len(chain.stages), 2)

    def test_impulse_and_frequency_response_preserve_lti_gain(self):
        chain = TransmissionChain(stages=[AttenuatorStage(loss_db=20.0)])

        impulse = chain.impulse_response(num_samples=8, sample_rate=2.0)
        frequencies, response = chain.frequency_response(num_samples=8, sample_rate=2.0)

        np.testing.assert_allclose(impulse.values, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(response, np.full(8, 0.1))
        np.testing.assert_allclose(frequencies, np.fft.fftfreq(8, d=0.5))

    def test_frequency_response_rejects_non_lti_stage(self):
        chain = TransmissionChain(stages=[BaseTransmissionStage(is_lti=False)])

        with self.assertRaisesRegex(ValueError, 'only defined for LTI'):
            chain.frequency_response(num_samples=4, sample_rate=1.0)

    def test_impulse_response_validates_shape_parameters(self):
        chain = TransmissionChain()

        with self.assertRaisesRegex(ValueError, 'num_samples must be positive'):
            chain.impulse_response(num_samples=0, sample_rate=1.0)
        with self.assertRaisesRegex(ValueError, 'sample_rate must be positive'):
            chain.impulse_response(num_samples=4, sample_rate=0.0)


if __name__ == '__main__':
    unittest.main()
