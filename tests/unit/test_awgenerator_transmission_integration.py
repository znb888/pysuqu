import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pysuqu.funclib import (
    AttenuatorStage,
    BundleTransmissionChain,
    BundleTransmissionResult,
    ChannelSchedule,
    DelayStage,
    EnvelopeParams,
    MIMOTouchstoneStage,
    MixerParams,
    PulseEvent,
    SignalBundle,
    SignalTrace,
    TouchstoneNetwork,
    TransmissionChain,
    TransmissionResult,
    WaveformGenerator,
)


class _FakeScatter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout = {}
        self.show_count = 0

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)

    def show(self):
        self.show_count += 1


def make_schedule(
    name='drive',
    *,
    amplitude=1.0,
    sample_rate=4.0,
    lo_freq=0.5,
    transmission_chain=None,
    fir_kernel=None,
):
    envelope = EnvelopeParams(
        name=f'{name}_square',
        duration=2.0,
        peak_amp=amplitude,
        shape_type='square',
    )
    return ChannelSchedule(
        name=name,
        sampling_rate=sample_rate,
        mixer_config=MixerParams(lo_freq=lo_freq),
        mixer_correction=False,
        events=[PulseEvent(start_time=0.0, envelope=envelope)],
        fir_kernel=fir_kernel,
        transmission_chain=transmission_chain,
    )


class ScheduleAndAwgCompilationTests(unittest.TestCase):
    def setUp(self):
        self.generator = WaveformGenerator(total_time=2.0, sample_rate=4.0)

    def test_clone_preserves_chain_and_copies_mutable_schedule_fields(self):
        chain = TransmissionChain(stages=[AttenuatorStage(loss_db=3.0)])
        schedule = make_schedule(
            transmission_chain=chain,
            fir_kernel=np.array([1.0, 0.5]),
        )

        clone = schedule.clone_with(name='copy')

        self.assertEqual(clone.name, 'copy')
        self.assertIs(clone.transmission_chain, chain)
        self.assertIs(clone.mixer_config, schedule.mixer_config)
        self.assertIsNot(clone.events, schedule.events)
        self.assertIsNot(clone.fir_kernel, schedule.fir_kernel)
        np.testing.assert_allclose(clone.fir_kernel, schedule.fir_kernel)

    def test_awg_traces_match_legacy_iq_and_rf_generation(self):
        schedule = make_schedule(fir_kernel=np.array([0.75, 0.25]))
        legacy_iq, _ = self.generator.generate_channel_waveform(
            schedule,
            return_complex=True,
        )

        iq_trace = self.generator.generate_awg_output(schedule, mode='iq')
        rf_trace = self.generator.generate_awg_output(schedule, mode='rf')

        np.testing.assert_allclose(iq_trace.values, legacy_iq)
        np.testing.assert_allclose(
            rf_trace.values,
            self.generator.generate_rf_waveform(schedule),
        )
        np.testing.assert_allclose(iq_trace.t_axis, self.generator.t_axis)
        self.assertEqual(iq_trace.domain, 'iq_complex')
        self.assertEqual(iq_trace.plane, 'awg_iq')
        self.assertEqual(rf_trace.domain, 'rf_real')
        self.assertEqual(rf_trace.plane, 'awg_rf')
        self.assertEqual(iq_trace.metadata['schedule_name'], 'drive')

    def test_schedule_and_generator_sample_rates_must_match(self):
        schedule = make_schedule(sample_rate=2.0)

        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.generator.generate_channel_waveform(schedule)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.generator.generate_awg_output(schedule)

    def test_awg_bundle_preserves_explicit_order_and_validates_inputs(self):
        schedules = {'q2': make_schedule('second'), 'q1': make_schedule('first')}

        bundle = self.generator.generate_awg_bundle(schedules)

        self.assertEqual(bundle.order, ('q2', 'q1'))
        self.assertEqual(bundle.plane, 'awg_iq')
        with self.assertRaisesRegex(ValueError, 'At least one'):
            self.generator.generate_awg_bundle([])
        with self.assertRaisesRegex(ValueError, 'Duplicate channel name'):
            self.generator.generate_awg_bundle(
                [make_schedule('same'), make_schedule('same')]
            )
        with self.assertRaisesRegex(TypeError, 'ChannelSchedule'):
            self.generator.generate_awg_bundle({'q1': object()})


class SingleChannelPropagationTests(unittest.TestCase):
    def setUp(self):
        self.generator = WaveformGenerator(total_time=2.0, sample_rate=4.0)

    def test_schedule_chain_and_explicit_override_produce_qubit_iq(self):
        configured = TransmissionChain(stages=[AttenuatorStage(loss_db=20.0)])
        override = TransmissionChain(
            stages=[AttenuatorStage(loss_db=20.0 * np.log10(2.0))]
        )
        schedule = make_schedule(transmission_chain=configured)

        configured_output = self.generator.generate_qubit_output(schedule, mode='iq')
        override_output = self.generator.generate_qubit_output(
            schedule,
            chain=override,
            mode='iq',
        )

        np.testing.assert_allclose(configured_output.values, 0.1)
        np.testing.assert_allclose(override_output.values, 0.5, rtol=1e-12)
        self.assertEqual(configured_output.plane, 'qubit_iq')
        self.assertEqual(configured_output.domain, 'iq_complex')

    def test_capture_history_returns_finalized_output_and_stage_traces(self):
        chain = TransmissionChain(
            name='control_line',
            stages=[AttenuatorStage(loss_db=20.0), DelayStage(delay_ns=0.25)],
        )

        result = self.generator.generate_qubit_output(
            make_schedule(),
            chain=chain,
            mode='iq',
            capture_history=True,
        )

        self.assertIsInstance(result, TransmissionResult)
        self.assertEqual(result.input_trace.plane, 'awg_iq')
        self.assertEqual(result.output_trace.plane, 'qubit_iq')
        self.assertEqual(len(result.stage_outputs), 2)
        np.testing.assert_allclose(result.output_trace.values[0], 0.0)
        np.testing.assert_allclose(result.output_trace.values[1:], 0.1)

    def test_stage_without_capture_history_keyword_can_return_history(self):
        class LegacyScaleStage:
            name = 'legacy_scale'
            is_lti = True

            def apply(self, trace):
                return trace.clone(values=2.0 * trace.values)

        result = self.generator.generate_qubit_output(
            make_schedule(),
            chain=LegacyScaleStage(),
            mode='iq',
            capture_history=True,
        )

        self.assertIsInstance(result, TransmissionResult)
        self.assertEqual(len(result.stage_outputs), 1)
        np.testing.assert_allclose(result.output_trace.values, 2.0)

    def test_internal_stage_type_errors_are_not_retried_or_hidden(self):
        class BrokenStage:
            name = 'broken'

            def apply(self, trace, capture_history=False):
                raise TypeError('stage calculation failed')

        with self.assertRaisesRegex(TypeError, 'stage calculation failed'):
            self.generator.generate_qubit_output(
                make_schedule(),
                chain=BrokenStage(),
                mode='iq',
            )

    def test_rf_mode_falls_back_through_iq_chain_then_mixes_carrier(self):
        chain = TransmissionChain(
            stages=[AttenuatorStage(loss_db=20.0, domain='iq_complex')]
        )
        schedule = make_schedule(lo_freq=0.5)

        output = self.generator.generate_qubit_output(
            schedule,
            chain=chain,
            mode='rf',
        )

        expected = 0.1 * np.cos(2 * np.pi * 0.5 * self.generator.t_axis)
        np.testing.assert_allclose(output.values, expected, atol=1e-15)
        self.assertEqual(output.domain, 'rf_real')
        self.assertEqual(output.plane, 'qubit_rf')

    def test_rf_native_chain_is_applied_without_iq_fallback(self):
        chain = TransmissionChain(
            stages=[AttenuatorStage(loss_db=20.0, domain='rf_real')]
        )
        schedule = make_schedule()

        output = self.generator.generate_qubit_output(
            schedule,
            chain=chain,
            mode='rf',
        )

        np.testing.assert_allclose(
            output.values,
            0.1 * self.generator.generate_rf_waveform(schedule),
        )

    def test_iq_to_rf_conversion_preserves_trace_context(self):
        iq_trace = self.generator.generate_qubit_output(make_schedule(), mode='iq')

        rf_trace = self.generator.convert_iq_trace_to_rf(iq_trace)

        self.assertIsInstance(rf_trace, SignalTrace)
        self.assertEqual(rf_trace.plane, 'qubit_rf')
        self.assertEqual(rf_trace.domain, 'rf_real')
        self.assertEqual(rf_trace.metadata['converted_from'], iq_trace.label)
        np.testing.assert_allclose(
            rf_trace.values,
            np.cos(2 * np.pi * iq_trace.lo_freq * iq_trace.t_axis),
            atol=1e-15,
        )


class BundlePropagationTests(unittest.TestCase):
    def setUp(self):
        self.generator = WaveformGenerator(total_time=2.0, sample_rate=4.0)
        self.schedules = {
            'x': make_schedule('x', amplitude=1.0, lo_freq=0.0),
            'y': make_schedule('y', amplitude=2.0, lo_freq=0.0),
        }

    def test_bundle_without_chain_is_finalized_at_qubit_plane(self):
        output = self.generator.generate_qubit_bundle(self.schedules, mode='iq')

        self.assertIsInstance(output, SignalBundle)
        self.assertEqual(output.order, ('x', 'y'))
        self.assertEqual(output.plane, 'qubit_iq')
        np.testing.assert_allclose(output['x'].values, 1.0)
        np.testing.assert_allclose(output['y'].values, 2.0)

    def test_schedule_chains_are_dispatched_per_channel_and_stage_index(self):
        self.schedules['x'].transmission_chain = TransmissionChain(
            stages=[
                AttenuatorStage(loss_db=20.0),
                DelayStage(delay_ns=0.25),
            ]
        )
        self.schedules['y'].transmission_chain = AttenuatorStage(
            loss_db=20.0 * np.log10(2.0)
        )

        output = self.generator.generate_qubit_bundle(self.schedules, mode='iq')

        np.testing.assert_allclose(output['x'].values[0], 0.0)
        np.testing.assert_allclose(output['x'].values[1:], 0.1)
        np.testing.assert_allclose(output['y'].values, 1.0, rtol=1e-12)

    def test_explicit_single_trace_chain_is_mapped_over_every_channel(self):
        chain = TransmissionChain(stages=[AttenuatorStage(loss_db=20.0)])

        result = self.generator.generate_qubit_bundle(
            self.schedules,
            chain=chain,
            mode='iq',
            capture_history=True,
        )

        self.assertIsInstance(result, BundleTransmissionResult)
        self.assertEqual(len(result.stage_outputs), 1)
        np.testing.assert_allclose(result.output_bundle['x'].values, 0.1)
        np.testing.assert_allclose(result.output_bundle['y'].values, 0.2)

    def test_mimo_bundle_chain_can_mix_and_rename_physical_outputs(self):
        matrix = np.array([[1.0, 0.25], [0.5, 1.0]], dtype=complex)
        network = TouchstoneNetwork(
            frequencies=np.array([0.0, 2.0]),
            s_parameters=np.repeat(matrix[np.newaxis, :, :], 2, axis=0),
            path='integration.s2p',
        )
        mimo = MIMOTouchstoneStage(
            network=network,
            input_ports=(1, 2),
            output_ports=(1, 2),
            input_channels=('x', 'y'),
            output_channels=('qubit_x', 'qubit_y'),
        )

        output = self.generator.generate_qubit_bundle(
            self.schedules,
            chain=BundleTransmissionChain(stages=[mimo]),
            mode='rf',
        )

        self.assertEqual(output.order, ('qubit_x', 'qubit_y'))
        self.assertEqual(output.plane, 'qubit_rf')
        np.testing.assert_allclose(output['qubit_x'].values, 1.5, atol=1e-15)
        np.testing.assert_allclose(output['qubit_y'].values, 2.5, atol=1e-15)

    def test_iq_bundle_conversion_uses_each_channel_lo(self):
        schedules = {
            'x': make_schedule('x', lo_freq=0.25),
            'y': make_schedule('y', lo_freq=0.5),
        }
        iq_bundle = self.generator.generate_awg_bundle(schedules, mode='iq')

        rf_bundle = self.generator.convert_iq_bundle_to_rf(iq_bundle)

        self.assertEqual(rf_bundle.plane, 'awg_rf')
        np.testing.assert_allclose(
            rf_bundle['x'].values,
            np.cos(2 * np.pi * 0.25 * self.generator.t_axis),
        )
        np.testing.assert_allclose(
            rf_bundle['y'].values,
            np.cos(2 * np.pi * 0.5 * self.generator.t_axis),
        )


class TransmissionPlottingTests(unittest.TestCase):
    def setUp(self):
        self.generator = WaveformGenerator(total_time=2.0, sample_rate=4.0)
        self.go = SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    def test_plot_trace_renders_iq_components_and_returns_figure(self):
        trace = self.generator.generate_awg_output(make_schedule(), mode='iq')

        with patch(
            'pysuqu.funclib.awgenerator._load_plotly_graph_objects',
            return_value=self.go,
        ):
            figure = self.generator.plot_trace(trace)

        self.assertEqual(len(figure.traces), 3)
        self.assertEqual(figure.traces[0].kwargs['name'], 'drive_awg_iq I')
        self.assertEqual(figure.traces[2].kwargs['name'], 'drive_awg_iq |IQ|')
        self.assertEqual(figure.show_count, 1)
        self.assertIn('[awg_iq]', figure.layout['title'])

    def test_plot_transmission_result_includes_input_stages_and_output(self):
        result = self.generator.generate_qubit_output(
            make_schedule(),
            chain=TransmissionChain(
                stages=[AttenuatorStage(loss_db=3.0), DelayStage(delay_ns=0.25)]
            ),
            mode='iq',
            capture_history=True,
        )

        with patch(
            'pysuqu.funclib.awgenerator._load_plotly_graph_objects',
            return_value=self.go,
        ):
            figure = self.generator.plot_transmission_result(result)

        self.assertEqual(len(figure.traces), 12)
        self.assertTrue(figure.traces[0].kwargs['name'].startswith('input:'))
        self.assertTrue(figure.traces[-1].kwargs['name'].startswith('output:'))

    def test_plot_schedule_routes_qubit_history_to_result_plotter(self):
        chain = TransmissionChain(stages=[AttenuatorStage(loss_db=3.0)])
        sentinel = object()

        with patch.object(
            self.generator,
            'plot_transmission_result',
            return_value=sentinel,
        ) as plot_result:
            output = self.generator.plot_schedule(
                make_schedule(),
                plane='qubit',
                chain=chain,
                capture_history=True,
            )

        self.assertIs(output, sentinel)
        self.assertIsInstance(plot_result.call_args.args[0], TransmissionResult)
        self.assertEqual(plot_result.call_args.kwargs['plot_mode'], 'iq')

    def test_plot_pulse_inherits_generator_sample_rate(self):
        pulse = make_schedule().events[0]
        sentinel = object()

        with patch.object(
            self.generator,
            'plot_schedule',
            return_value=sentinel,
        ) as plot_schedule:
            output = self.generator.plot_pulse(pulse, plane='qubit')

        plotted_schedule = plot_schedule.call_args.args[0]
        self.assertIs(output, sentinel)
        self.assertEqual(plotted_schedule.sampling_rate, self.generator.sample_rate)
        self.assertEqual(plot_schedule.call_args.kwargs['plane'], 'qubit')

    def test_plot_mode_and_plane_validation_are_explicit(self):
        iq_trace = self.generator.generate_awg_output(make_schedule(), mode='iq')
        figure = _FakeFigure()

        with self.assertRaisesRegex(ValueError, 'rf_real'):
            self.generator._plot_trace_to_figure(
                figure,
                iq_trace,
                self.go,
                plot_mode='rf',
            )
        with self.assertRaisesRegex(ValueError, 'plot plane'):
            self.generator.plot_schedule(make_schedule(), plane='invalid')


class SolverTraceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(761923)
        self.generator = WaveformGenerator(total_time=2.0, sample_rate=4.0)
        self.amplitudes = self.rng.uniform(0.15, 0.85, size=2)
        self.lo_freq = self.rng.uniform(2.7, 3.4)
        self.loss_db = self.rng.uniform(2.0, 9.0)
        self.schedules = {
            name: make_schedule(name, amplitude=amplitude, lo_freq=self.lo_freq)
            for name, amplitude in zip(('input_b', 'input_a'), self.amplitudes)
        }
        self.schedule_collections = (
            self.schedules, list(self.schedules.values()), tuple(self.schedules.values()),
        )
        self.query_time = 2.5 * self.generator.dt

    def test_single_trace_preserves_envelope_carrier_and_chain_precedence(self):
        schedule = self.schedules['input_b']
        schedule.transmission_chain = AttenuatorStage(loss_db=self.loss_db)
        override = AttenuatorStage(loss_db=self.rng.uniform(10.0, 16.0))
        for mode in ('rf', 'complex_envelope'):
            for plane in ('awg', 'qubit'):
                for chain in (None, override):
                    with self.subTest(mode=mode, plane=plane, chain=chain):
                        trace = self.generator.get_solver_trace(
                            schedule, mode=mode, plane=plane, chain=chain,
                        )
                        loss = (chain or schedule.transmission_chain).loss_db
                        gain = 1.0 if plane == 'awg' else 10.0 ** (-loss / 20.0)
                        expected = self.amplitudes[0] * gain
                        self.assertIsInstance(trace, SignalTrace)
                        self.assertEqual(trace.domain, 'iq_complex')
                        self.assertEqual(trace.plane, f'{plane}_iq')
                        self.assertEqual(trace.lo_freq, self.lo_freq)
                        np.testing.assert_array_equal(trace.t_axis, self.generator.t_axis)
                        np.testing.assert_allclose(trace.values, expected)
                        callback = self.generator.get_qutip_func(
                            schedule, mode=mode, plane=plane, chain=chain,
                        )
                        if mode == 'rf':
                            expected *= np.cos(2 * np.pi * self.lo_freq * self.query_time)
                        self.assertAlmostEqual(callback(self.query_time), expected)

    def test_bundle_preserves_mimo_outputs_for_mapping_and_sequence_inputs(self):
        matrix = self.rng.uniform(0.05, 0.4, size=(2, 2))
        network = TouchstoneNetwork(
            frequencies=np.array([0.0, self.lo_freq + self.generator.sample_rate]),
            s_parameters=np.repeat(matrix[np.newaxis, :, :], 2, axis=0),
            path='synthetic_solver.s2p',
        )
        outputs = ('output_b', 'output_a')
        chain = BundleTransmissionChain(stages=[MIMOTouchstoneStage(
            network=network,
            input_ports=(1, 2), output_ports=(1, 2),
            input_channels=tuple(self.schedules), output_channels=outputs,
        )])
        expected = matrix @ self.amplitudes
        for schedules in self.schedule_collections:
            for mode in ('rf', 'complex_envelope'):
                with self.subTest(collection=type(schedules).__name__, mode=mode):
                    traces = self.generator.get_solver_trace_bundle(
                        schedules, chain=chain, mode=mode,
                    )
                    if isinstance(schedules, dict):
                        self.assertEqual(tuple(traces), outputs)
                        ordered = tuple(traces.values())
                    else:
                        self.assertIsInstance(traces, tuple)
                        ordered = traces
                    self.assertEqual(len(ordered), len(outputs))
                    callbacks = self.generator.get_qutip_bundle_funcs(
                        schedules, chain=chain, mode=mode,
                    )
                    self.assertEqual(tuple(callbacks), outputs)
                    for index, trace in enumerate(ordered):
                        self.assertEqual(trace.domain, 'iq_complex')
                        self.assertEqual(trace.plane, 'qubit_iq')
                        np.testing.assert_allclose(trace.values, expected[index])
                        value = expected[index]
                        if mode == 'rf':
                            value *= np.cos(2 * np.pi * self.lo_freq * self.query_time)
                        self.assertAlmostEqual(callbacks[outputs[index]](self.query_time), value)

    def test_bundle_awg_plane_preserves_order_and_ignores_qubit_chain(self):
        chain = AttenuatorStage(loss_db=self.loss_db, domain='rf_real')
        for schedules in self.schedule_collections:
            for mode in ('rf', 'complex_envelope'):
                with self.subTest(collection=type(schedules).__name__, mode=mode):
                    traces = self.generator.get_solver_trace_bundle(
                        schedules, mode=mode, plane='awg', chain=chain,
                    )
                    if isinstance(schedules, dict):
                        self.assertEqual(tuple(traces), tuple(self.schedules))
                        traces = tuple(traces.values())
                    for trace, amplitude in zip(traces, self.amplitudes):
                        self.assertEqual(trace.domain, 'iq_complex')
                        self.assertEqual(trace.plane, 'awg_iq')
                        np.testing.assert_allclose(trace.values, amplitude)

    def test_rf_only_chain_returns_sampled_rf_for_single_and_bundle(self):
        chain = AttenuatorStage(loss_db=self.loss_db, domain='rf_real')
        gain = 10.0 ** (-self.loss_db / 20.0)
        single = self.generator.get_solver_trace(self.schedules['input_b'], chain=chain)
        np.testing.assert_allclose(
            single.values,
            gain * self.amplitudes[0] * np.cos(2 * np.pi * self.lo_freq * single.t_axis),
        )
        self.assertEqual(single.domain, 'rf_real')
        for schedules in self.schedule_collections:
            with self.subTest(collection=type(schedules).__name__):
                traces = self.generator.get_solver_trace_bundle(schedules, chain=chain)
                if isinstance(schedules, dict):
                    traces = tuple(traces.values())
                callbacks = self.generator.get_qutip_bundle_funcs(schedules, chain=chain)
                for trace, name, amplitude in zip(traces, self.schedules, self.amplitudes):
                    expected = gain * amplitude * np.cos(2 * np.pi * self.lo_freq * trace.t_axis)
                    self.assertEqual(trace.domain, 'rf_real')
                    self.assertEqual(trace.plane, 'qubit_rf')
                    np.testing.assert_allclose(trace.values, expected)
                    self.assertAlmostEqual(
                        callbacks[name](self.query_time),
                        np.interp(self.query_time, trace.t_axis, expected),
                    )
        for method, value in (
            (self.generator.get_solver_trace, self.schedules['input_b']),
            (self.generator.get_solver_trace_bundle, self.schedules),
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, 'expects rf_real'):
                    method(value, mode='complex_envelope', chain=chain)

    def test_invalid_transmission_is_not_retried_as_rf(self):
        chain = AttenuatorStage(domain='iq_complex', output_plane='baseband')
        for method, value, producer in (
            (self.generator.get_solver_trace, self.schedules['input_b'], 'generate_qubit_output'),
            (self.generator.get_solver_trace_bundle, self.schedules, 'generate_qubit_bundle'),
        ):
            with self.subTest(method=method.__name__):
                with patch.object(
                    self.generator, producer, wraps=getattr(self.generator, producer),
                ) as generate:
                    with self.assertRaisesRegex(ValueError, 'plane baseband'):
                        method(value, chain=chain)
                    self.assertEqual(generate.call_count, 1)

    def test_solver_modes_planes_and_collections_are_validated(self):
        for method, value in (
            (self.generator.get_solver_trace, self.schedules['input_b']),
            (self.generator.get_solver_trace_bundle, self.schedules),
            (self.generator.get_solver_trace_bundle, list(self.schedules.values())),
        ):
            with self.subTest(method=method.__name__, collection=type(value).__name__):
                with self.assertRaisesRegex(ValueError, 'drive mode'):
                    method(value, mode='invalid')
                with self.assertRaisesRegex(ValueError, 'waveform plane'):
                    method(value, plane='invalid')
        for schedules in ({}, [], ()):
            with self.assertRaisesRegex(ValueError, 'At least one'):
                self.generator.get_solver_trace_bundle(schedules)
        with self.assertRaisesRegex(TypeError, 'ChannelSchedule'):
            self.generator.get_solver_trace_bundle({'input': object()})
        schedule = self.schedules['input_b']
        with self.assertRaisesRegex(ValueError, 'Duplicate channel name'):
            self.generator.get_solver_trace_bundle([schedule, schedule])


class QutipCompilationTests(unittest.TestCase):
    def setUp(self):
        self.generator = WaveformGenerator(total_time=2.0, sample_rate=4.0)

    def test_complex_envelope_callback_uses_qubit_side_chain(self):
        schedule = make_schedule(
            transmission_chain=TransmissionChain(
                stages=[AttenuatorStage(loss_db=20.0)]
            )
        )

        callback = self.generator.get_qutip_func(
            schedule,
            mode='complex_envelope',
        )

        self.assertAlmostEqual(callback(0.5), 0.1 + 0.0j)
        self.assertEqual(callback(-0.1), 0.0j)

    def test_rf_callback_mixes_iq_chain_at_solver_query_time(self):
        schedule = make_schedule(lo_freq=0.5)
        chain = TransmissionChain(
            stages=[AttenuatorStage(loss_db=20.0, domain='iq_complex')]
        )
        query_time = 0.125

        callback = self.generator.get_qutip_func(
            schedule,
            mode='rf',
            chain=chain,
        )

        expected = 0.1 * np.cos(2 * np.pi * 0.5 * query_time)
        self.assertAlmostEqual(callback(query_time), expected)

    def test_rf_only_chain_falls_back_to_sampled_rf_trace(self):
        schedule = make_schedule(lo_freq=0.5)
        chain = TransmissionChain(
            stages=[AttenuatorStage(loss_db=20.0, domain='rf_real')]
        )
        output = self.generator.generate_qubit_output(
            schedule,
            chain=chain,
            mode='rf',
        )

        callback = self.generator.get_qutip_func(
            schedule,
            mode='rf',
            chain=chain,
        )

        self.assertAlmostEqual(callback(0.25), output.values[1])

    def test_unrelated_qubit_trace_validation_error_is_not_hidden(self):
        chain = TransmissionChain(
            stages=[
                AttenuatorStage(
                    domain='iq_complex',
                    output_plane='baseband',
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, 'produced plane baseband'):
            self.generator.get_qutip_func(
                make_schedule(),
                mode='rf',
                chain=chain,
            )

    def test_bundle_callbacks_follow_mapped_channel_names(self):
        schedules = {
            'x': make_schedule('x', amplitude=1.0),
            'y': make_schedule('y', amplitude=2.0),
        }
        chain = BundleTransmissionChain(
            stages=[AttenuatorStage(loss_db=20.0)]
        )

        callbacks = self.generator.get_qutip_bundle_funcs(
            schedules,
            mode='complex_envelope',
            chain=chain,
        )

        self.assertEqual(tuple(callbacks), ('x', 'y'))
        self.assertAlmostEqual(callbacks['x'](0.5), 0.1 + 0.0j)
        self.assertAlmostEqual(callbacks['y'](0.5), 0.2 + 0.0j)

    def test_invalid_qutip_modes_and_planes_are_rejected(self):
        schedule = make_schedule()

        with self.assertRaisesRegex(ValueError, 'drive mode'):
            self.generator.get_qutip_func(schedule, mode='invalid')
        with self.assertRaisesRegex(ValueError, 'waveform plane'):
            self.generator.get_qutip_func(schedule, plane='invalid')


if __name__ == '__main__':
    unittest.main()
