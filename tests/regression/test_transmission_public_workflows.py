import unittest

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit import (
    AttenuatorStage,
    ChannelSchedule,
    DelayStage,
    EnvelopeParams,
    MixerParams,
    PulseEvent,
    TouchstoneNetwork,
    TouchstoneStage,
    TransmissionChain,
    TransmissionResult,
    WaveformGenerator,
)


def make_square_schedule(*, transmission_chain=None):
    return ChannelSchedule(
        name='public_drive',
        sampling_rate=2.0,
        mixer_config=MixerParams(lo_freq=5.0),
        mixer_correction=False,
        events=[
            PulseEvent(
                start_time=0.0,
                envelope=EnvelopeParams(
                    duration=2.0,
                    peak_amp=1.0,
                    shape_type='square',
                ),
            )
        ],
        transmission_chain=transmission_chain,
    )


class StableTransmissionWorkflowRegressionTests(unittest.TestCase):
    def test_stable_qubit_exports_propagate_schedule_chain_with_history(self):
        chain = TransmissionChain(
            name='public_line',
            stages=[
                AttenuatorStage(loss_db=20.0),
                DelayStage(delay_ns=0.5),
            ],
        )
        generator = WaveformGenerator(total_time=4.0, sample_rate=2.0)

        result = generator.generate_qubit_output(
            make_square_schedule(transmission_chain=chain),
            mode='iq',
            capture_history=True,
        )

        self.assertIsInstance(result, TransmissionResult)
        self.assertEqual(result.input_trace.plane, 'awg_iq')
        self.assertEqual(result.output_trace.plane, 'qubit_iq')
        self.assertEqual(len(result.stage_outputs), 2)
        np.testing.assert_allclose(
            result.output_trace.values,
            [0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
            atol=1e-15,
        )

    def test_stable_touchstone_export_filters_awg_iq_trace(self):
        frequencies = np.array([4.0, 6.0])
        matrix = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=complex)
        network = TouchstoneNetwork(
            frequencies=frequencies,
            s_parameters=np.repeat(matrix[np.newaxis, :, :], 2, axis=0),
            path='public_constant.s2p',
        )
        chain = TransmissionChain(
            stages=[
                TouchstoneStage(
                    network=network,
                    input_port=1,
                    output_port=2,
                )
            ]
        )
        generator = WaveformGenerator(total_time=4.0, sample_rate=2.0)
        awg_trace = generator.generate_awg_output(make_square_schedule(), mode='iq')

        output = generator.generate_qubit_output(
            make_square_schedule(),
            chain=chain,
            mode='iq',
        )

        np.testing.assert_allclose(output.values, 0.5 * awg_trace.values, atol=1e-15)
        self.assertEqual(output.metadata['input_port'], 1)
        self.assertEqual(output.metadata['output_port'], 2)


if __name__ == '__main__':
    unittest.main()
