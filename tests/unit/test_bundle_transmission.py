import unittest

import numpy as np

from pysuqu.funclib import (
    AttenuatorStage,
    BaseBundleTransmissionStage,
    BundleTransmissionChain,
    BundleTransmissionResult,
    MIMOTouchstoneStage,
    SignalBundle,
    SignalTrace,
    TouchstoneNetwork,
)


def make_trace(
    values,
    *,
    sample_rate=2.0,
    domain='rf_real',
    plane='awg_rf',
    lo_freq=0.0,
    label='signal',
    metadata=None,
):
    return SignalTrace(
        np.arange(len(values), dtype=float) / sample_rate,
        np.asarray(values),
        sample_rate,
        domain,
        plane,
        lo_freq=lo_freq,
        label=label,
        metadata={} if metadata is None else metadata,
    )


def make_network(matrix, frequencies=(0.0, 2.0), path='matrix.s2p'):
    matrix = np.asarray(matrix, dtype=complex)
    s_parameters = np.repeat(matrix[np.newaxis, :, :], len(frequencies), axis=0)
    return TouchstoneNetwork(frequencies, s_parameters, path=path)


class SignalBundleTests(unittest.TestCase):
    def test_properties_order_lookup_clone_and_description(self):
        x = make_trace([1.0, 0.0], lo_freq=5.0)
        y = make_trace([0.5, 0.0], lo_freq=5.0)
        bundle = SignalBundle(
            {'x': x, 'y': y},
            order=('y', 'x'),
            label='drives',
            metadata={'source': 'test'},
        )

        self.assertEqual(bundle.names, ('y', 'x'))
        self.assertIs(bundle['x'], x)
        self.assertEqual(bundle.domain, 'rf_real')
        self.assertEqual(bundle.plane, 'awg_rf')
        self.assertEqual(bundle.sample_rate, 2.0)
        self.assertEqual(bundle.shared_lo_freq, 5.0)
        np.testing.assert_allclose(bundle.t_axis, [0.0, 0.5])
        self.assertEqual(bundle.describe(), 'drives (2 trace(s)): y, x')
        self.assertEqual(bundle.clone(label='copy').label, 'copy')

    def test_shared_lo_frequency_is_none_when_iq_traces_differ(self):
        bundle = SignalBundle(
            {
                'x': make_trace(
                    [1.0j],
                    domain='iq_complex',
                    plane='awg_iq',
                    lo_freq=5.0,
                ),
                'y': make_trace(
                    [1.0j],
                    domain='iq_complex',
                    plane='awg_iq',
                    lo_freq=5.1,
                ),
            }
        )

        self.assertIsNone(bundle.shared_lo_freq)

    def test_rejects_empty_invalid_or_misaligned_traces(self):
        reference = make_trace([1.0, 0.0])
        invalid = (
            lambda: SignalBundle({}),
            lambda: SignalBundle({'x': object()}),
            lambda: SignalBundle({'': reference}),
            lambda: SignalBundle({'x': reference, 'y': reference}, order=('x', 'x')),
            lambda: SignalBundle(
                {
                    'x': reference,
                    'y': make_trace(
                        [1.0, 0.0],
                        domain='iq_complex',
                        plane='awg_rf',
                    ),
                }
            ),
            lambda: SignalBundle(
                {
                    'x': reference,
                    'y': make_trace([1.0, 0.0], plane='qubit_rf'),
                }
            ),
            lambda: SignalBundle(
                {
                    'x': reference,
                    'y': make_trace([1.0, 0.0], sample_rate=1.0),
                }
            ),
            lambda: SignalBundle(
                {
                    'x': reference,
                    'y': SignalTrace(
                        [0.0, 0.6],
                        [1.0, 0.0],
                        2.0,
                        'rf_real',
                        'awg_rf',
                    ),
                }
            ),
        )

        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()


class MIMOTouchstoneStageTests(unittest.TestCase):
    @staticmethod
    def _rf_bundle(values_x=(1.0, -0.5, 0.25, 0.0), values_y=(2.0, 1.0, 0.0, -1.0)):
        return SignalBundle(
            {
                'x': make_trace(values_x, label='x', metadata={'source': 'x'}),
                'y': make_trace(values_y, label='y', metadata={'source': 'y'}),
            },
            label='inputs',
            metadata={'bundle_source': 'test'},
        )

    def test_constant_two_by_two_network_mixes_and_renames_channels(self):
        network = make_network([[1.0, 0.1], [0.2, 1.0]])
        stage = MIMOTouchstoneStage(
            network=network,
            input_ports=(1, 2),
            output_ports=(1, 2),
            input_channels=('x', 'y'),
            output_channels=('qubit_x', 'qubit_y'),
            output_plane='qubit_rf',
        )
        bundle = self._rf_bundle()

        output = stage.apply(bundle)

        np.testing.assert_allclose(
            output['qubit_x'].values,
            bundle['x'].values + 0.1 * bundle['y'].values,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            output['qubit_y'].values,
            0.2 * bundle['x'].values + bundle['y'].values,
            atol=1e-15,
        )
        self.assertEqual(output.names, ('qubit_x', 'qubit_y'))
        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(output.metadata['bundle_source'], 'test')
        self.assertEqual(output.metadata['last_stage'], 'mimo_touchstone')
        self.assertEqual(output['qubit_x'].metadata['source'], 'x')
        self.assertEqual(output['qubit_y'].metadata['source'], 'y')
        self.assertEqual(output['qubit_x'].metadata['fft_length'], 8)

    def test_default_output_names_reuse_inputs_or_fall_back_to_ports(self):
        bundle = self._rf_bundle(values_x=(1.0,), values_y=(2.0,))
        square = MIMOTouchstoneStage(
            network=make_network([[1.0, 0.0], [0.0, 1.0]]),
            input_ports=(1, 2),
            output_ports=(1, 2),
        )
        rectangular = MIMOTouchstoneStage(
            network=make_network([[1.0, 0.5], [0.0, 0.0]]),
            input_ports=(1, 2),
            output_ports=(1,),
        )

        self.assertEqual(square.apply(bundle).names, ('x', 'y'))
        rectangular_output = rectangular.apply(bundle)
        self.assertEqual(rectangular_output.names, ('port_1',))
        np.testing.assert_allclose(rectangular_output['port_1'].values, [2.0])

    def test_absolute_iq_requires_shared_lo_but_relative_mode_does_not(self):
        bundle = SignalBundle(
            {
                'x': make_trace(
                    [1.0 + 0.0j, 0.0],
                    domain='iq_complex',
                    plane='awg_iq',
                    lo_freq=5.0,
                ),
                'y': make_trace(
                    [0.5 + 0.0j, 0.0],
                    domain='iq_complex',
                    plane='awg_iq',
                    lo_freq=5.2,
                ),
            }
        )
        absolute = MIMOTouchstoneStage(
            network=make_network(
                [[1.0, 0.0], [0.0, 1.0]],
                frequencies=(4.0, 6.0),
            ),
            input_ports=(1, 2),
            output_ports=(1, 2),
            out_of_band='error',
        )
        relative = MIMOTouchstoneStage(
            network=make_network(
                [[1.0, 0.0], [0.0, 1.0]],
                frequencies=(-1.0, 1.0),
            ),
            input_ports=(1, 2),
            output_ports=(1, 2),
            frequency_mode='relative',
            out_of_band='error',
        )

        with self.assertRaisesRegex(ValueError, 'share one lo_freq'):
            absolute.apply(bundle)
        relative_output = relative.apply(bundle)
        np.testing.assert_allclose(relative_output['x'].values, bundle['x'].values)
        np.testing.assert_allclose(relative_output['y'].values, bundle['y'].values)

    def test_empty_traces_still_apply_output_mapping_and_metadata(self):
        bundle = self._rf_bundle(values_x=(), values_y=())
        stage = MIMOTouchstoneStage(
            network=make_network([[1.0, 0.0], [0.0, 1.0]]),
            input_ports=(1, 2),
            output_ports=(1, 2),
            output_channels=('a', 'b'),
            output_plane='qubit_rf',
        )

        output = stage.apply(bundle)

        self.assertEqual(output.names, ('a', 'b'))
        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(len(output['a'].values), 0)
        self.assertEqual(output['a'].metadata['fft_length'], 0)

    def test_stage_validates_ports_channels_modes_and_bundle_contract(self):
        network = make_network([[1.0, 0.0], [0.0, 1.0]])
        invalid = (
            lambda: MIMOTouchstoneStage(),
            lambda: MIMOTouchstoneStage(network=object()),
            lambda: MIMOTouchstoneStage(network=network, input_ports=()),
            lambda: MIMOTouchstoneStage(network=network, input_ports=(1.5,)),
            lambda: MIMOTouchstoneStage(network=network, input_ports=(1, 1)),
            lambda: MIMOTouchstoneStage(network=network, output_ports=(1, 1)),
            lambda: MIMOTouchstoneStage(
                network=network,
                input_ports=(1, 2),
                input_channels=('x',),
            ),
            lambda: MIMOTouchstoneStage(
                network=network,
                output_ports=(1, 2),
                output_channels=('x', 'x'),
            ),
            lambda: MIMOTouchstoneStage(network=network, interpolation='invalid'),
            lambda: MIMOTouchstoneStage(network=network, output_ports=(3,)),
        )

        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

        stage = MIMOTouchstoneStage(
            network=network,
            input_ports=(1, 2),
            output_ports=(1, 2),
            input_channels=('x', 'missing'),
        )
        with self.assertRaisesRegex(ValueError, 'could not find input'):
            stage.apply(self._rf_bundle())

        iq_only = MIMOTouchstoneStage(network=network, domain='iq_complex')
        with self.assertRaisesRegex(ValueError, 'expects iq_complex'):
            iq_only.apply(SignalBundle({'x': make_trace([1.0])}))


class BundleTransmissionChainTests(unittest.TestCase):
    @staticmethod
    def _bundle():
        return SignalBundle(
            {
                'x': make_trace([1.0, 0.0], metadata={'channel': 'x'}),
                'y': make_trace([2.0, 0.0], metadata={'channel': 'y'}),
            },
            label='inputs',
        )

    def test_chain_maps_single_trace_stage_then_applies_bundle_stage(self):
        mimo = MIMOTouchstoneStage(
            network=make_network([[1.0, 0.5], [0.0, 1.0]]),
            input_ports=(1, 2),
            output_ports=(1, 2),
            output_channels=('a', 'b'),
        )
        chain = BundleTransmissionChain(
            name='bundle_line',
            stages=[AttenuatorStage(loss_db=20.0), mimo],
        )

        result = chain.apply(self._bundle(), capture_history=True)

        self.assertIsInstance(result, BundleTransmissionResult)
        np.testing.assert_allclose(result.stage_outputs[0]['x'].values, [0.1, 0.0])
        np.testing.assert_allclose(result.stage_outputs[0]['y'].values, [0.2, 0.0])
        np.testing.assert_allclose(result.output_bundle['a'].values, [0.2, 0.0])
        np.testing.assert_allclose(result.output_bundle['b'].values, [0.2, 0.0])
        self.assertEqual(len(result.stage_outputs), 2)
        self.assertIn('attenuator[20.000 dB] -> mimo_touchstone', chain.describe())
        self.assertTrue(chain.is_lti)

    def test_append_extend_and_empty_chain(self):
        bundle = self._bundle()
        chain = BundleTransmissionChain(name='line')

        self.assertIs(chain.apply(bundle), bundle)
        self.assertEqual(chain.describe(), 'line (0 stage(s))')
        chain.append(AttenuatorStage(loss_db=3.0))
        chain.extend([AttenuatorStage(loss_db=2.0)])
        self.assertEqual(len(chain.stages), 2)

    def test_chain_rejects_invalid_input_stage_and_return_types(self):
        bundle = self._bundle()

        class MissingApply:
            name = 'missing_apply'

        class BadSingleStage:
            name = 'bad_single'
            is_lti = True

            def apply(self, trace):
                return bundle

        class BadBundleStage:
            name = 'bad_bundle'
            is_lti = True
            bundle_stage = True

            def apply(self, value):
                return value['x']

        with self.assertRaisesRegex(TypeError, 'SignalBundle input'):
            BundleTransmissionChain().apply(object())
        with self.assertRaisesRegex(TypeError, 'does not define apply'):
            BundleTransmissionChain(stages=[MissingApply()]).apply(bundle)
        with self.assertRaisesRegex(TypeError, 'must return a SignalTrace'):
            BundleTransmissionChain(stages=[BadSingleStage()]).apply(bundle)
        with self.assertRaisesRegex(TypeError, 'must return a SignalBundle'):
            BundleTransmissionChain(stages=[BadBundleStage()]).apply(bundle)

    def test_base_bundle_stage_validates_domain_plane_and_abstract_apply(self):
        bundle = self._bundle()
        domain_stage = BaseBundleTransmissionStage(domain='iq_complex')
        plane_stage = BaseBundleTransmissionStage(allowed_planes=('qubit_rf',))

        with self.assertRaisesRegex(ValueError, 'expects iq_complex'):
            domain_stage._validate_bundle(bundle)
        with self.assertRaisesRegex(ValueError, 'does not accept bundles'):
            plane_stage._validate_bundle(bundle)
        with self.assertRaises(NotImplementedError):
            BaseBundleTransmissionStage().apply(bundle)


if __name__ == '__main__':
    unittest.main()
