import tempfile
import unittest
from pathlib import Path

import numpy as np

from pysuqu.funclib import (
    SignalTrace,
    TouchstoneNetwork,
    TouchstoneStage,
    design_inverse_fir_from_touchstone,
    evaluate_touchstone_response,
    load_touchstone_network,
)


class TouchstoneFileTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

    def _write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding='utf-8')
        return path

    def test_loads_v1_ri_data_and_sorts_frequency_rows(self):
        path = self._write(
            'line.s2p',
            '''
! S11 S21 S12 S22 in the Touchstone 1.x two-port order
# MHz S RI R 75
2000  0.10 0  0.20 0  0.30 0  0.40 0
1000  0.11 0  0.21 0  0.31 0  0.41 0 ! inline comment
''',
        )

        network = load_touchstone_network(path)

        np.testing.assert_allclose(network.frequencies, [1.0, 2.0])
        np.testing.assert_allclose(network.get_response(2, 1), [0.21, 0.20])
        np.testing.assert_allclose(network.get_response(1, 2), [0.31, 0.30])
        self.assertEqual(network.reference, 75.0)
        self.assertEqual(network.n_ports, 2)

    def test_loads_touchstone_2_data_order_and_per_port_reference(self):
        path = self._write(
            'ordered.s2p',
            '''
[Version] 2.0
# GHz S MA R 50
[Number of Ports] 2
[Number of Frequencies] 1
[Reference] 50 75
[Matrix Format] Full
[Two-Port Data Order] 12_21
[Network Data]
1.0  1 0  2 90
3 180  4 -90
[End]
''',
        )

        network = load_touchstone_network(path)

        np.testing.assert_allclose(network.s_parameters[0, 0, 0], 1.0)
        np.testing.assert_allclose(network.s_parameters[0, 0, 1], 2.0j, atol=1e-15)
        np.testing.assert_allclose(network.s_parameters[0, 1, 0], -3.0, atol=1e-15)
        np.testing.assert_allclose(network.s_parameters[0, 1, 1], -4.0j, atol=1e-15)
        np.testing.assert_allclose(network.reference, [50.0, 75.0])

    def test_loads_db_data_and_fortran_exponents(self):
        path = self._write(
            'gain.s1p',
            '# Hz S DB R 50\n1D9 -6.020599913279624 90\n',
        )

        network = load_touchstone_network(path)

        np.testing.assert_allclose(network.frequencies, [1.0])
        np.testing.assert_allclose(network.get_response(1, 1), [0.5j], atol=1e-12)

    def test_rejects_missing_or_non_touchstone_paths(self):
        with self.assertRaises(FileNotFoundError):
            load_touchstone_network(self.root / 'missing.s2p')

        wrong_suffix = self._write('line.txt', '# GHz S RI R 50\n1 1 0\n')
        with self.assertRaisesRegex(ValueError, 'end with .sNp'):
            load_touchstone_network(wrong_suffix)

    def test_rejects_inconsistent_touchstone_2_declarations(self):
        cases = (
            (
                'ports.s2p',
                '[Number of Ports] 3\n# GHz S RI R 50\n',
                'declares 3',
            ),
            (
                'count.s1p',
                '[Number of Frequencies] 2\n# GHz S RI R 50\n1 1 0\n',
                'declares 2 frequencies',
            ),
            (
                'matrix.s2p',
                '[Matrix Format] Lower\n# GHz S RI R 50\n1 1 0 1 0 1 0 1 0\n',
                'matrix format',
            ),
            (
                'reference.s2p',
                '# GHz S RI R 50\n[Reference] 50 60 70\n'
                '1 1 0 1 0 1 0 1 0\n',
                'one value per port',
            ),
            (
                'order.s2p',
                '# GHz S RI R 50\n[Two-Port Data Order] invalid\n'
                '1 1 0 1 0 1 0 1 0\n',
                'Two-Port Data Order',
            ),
        )

        for name, content, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    load_touchstone_network(self._write(name, content))

    def test_rejects_unsupported_options_and_malformed_data(self):
        unsupported = self._write('admittance.s1p', '# GHz Y RI R 50\n1 1 0\n')
        malformed = self._write('malformed.s2p', '# GHz S RI R 50\n1 1 0 2 0\n')
        duplicate = self._write(
            'duplicate.s1p',
            '# GHz S RI R 50\n1 1 0\n1 0.5 0\n',
        )

        with self.assertRaisesRegex(ValueError, 'Only S-parameter'):
            load_touchstone_network(unsupported)
        with self.assertRaisesRegex(ValueError, 'Malformed Touchstone data'):
            load_touchstone_network(malformed)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            load_touchstone_network(duplicate)


class TouchstoneNetworkTests(unittest.TestCase):
    @staticmethod
    def _network(response=(1.0, 1.0), frequencies=(1.0, 2.0)):
        matrix = np.zeros((len(frequencies), 2, 2), dtype=complex)
        matrix[:, 1, 0] = response
        return TouchstoneNetwork(frequencies, matrix)

    def test_network_rejects_invalid_shapes_and_nonfinite_values(self):
        invalid = (
            lambda: TouchstoneNetwork([], np.empty((0, 1, 1))),
            lambda: TouchstoneNetwork([1.0], np.empty((2, 1, 1))),
            lambda: TouchstoneNetwork([1.0], np.empty((1, 1, 2))),
            lambda: TouchstoneNetwork([1.0], np.empty((1, 0, 0))),
            lambda: TouchstoneNetwork([np.nan], np.ones((1, 1, 1))),
            lambda: TouchstoneNetwork([1.0], np.array([[[np.inf]]])),
            lambda: TouchstoneNetwork([2.0, 1.0], np.ones((2, 1, 1))),
            lambda: TouchstoneNetwork([1.0], np.ones((1, 1, 1)), reference=-50),
            lambda: TouchstoneNetwork(
                [1.0],
                np.ones((1, 2, 2)),
                reference=[50.0],
            ),
        )

        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_get_response_validates_one_based_ports(self):
        network = self._network()

        with self.assertRaisesRegex(ValueError, 'input_port'):
            network.get_response(1, 0)
        with self.assertRaisesRegex(ValueError, 'output_port'):
            network.get_response(3, 1)

    def test_cartesian_and_polar_interpolation_are_distinct(self):
        network = self._network(response=(1.0, 1.0j))

        cartesian = evaluate_touchstone_response(
            [1.5],
            network=network,
            interpolation='cartesian',
        )
        polar = evaluate_touchstone_response(
            [1.5],
            network=network,
            interpolation='polar',
        )

        np.testing.assert_allclose(cartesian, [0.5 + 0.5j])
        np.testing.assert_allclose(polar, [np.exp(0.25j * np.pi)])

    def test_out_of_band_policies_and_scalar_shape(self):
        network = self._network(response=(1.0, 2.0))

        edge = evaluate_touchstone_response(
            0.0,
            network=network,
            interpolation='cartesian',
            out_of_band='edge',
        )
        zero = evaluate_touchstone_response(
            [0.0, 3.0],
            network=network,
            out_of_band='zero',
        )

        self.assertEqual(edge.shape, ())
        np.testing.assert_allclose(edge, 1.0)
        np.testing.assert_allclose(zero, [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, 'outside the measured'):
            evaluate_touchstone_response(
                [0.0],
                network=network,
                out_of_band='error',
            )

    def test_response_evaluation_preserves_multidimensional_query_shape(self):
        network = self._network(response=(1.0, 2.0))
        query = np.array([[1.0, 1.25], [1.5, 2.0]])

        response = evaluate_touchstone_response(
            query,
            network=network,
            interpolation='cartesian',
        )

        self.assertEqual(response.shape, query.shape)
        np.testing.assert_allclose(response, [[1.0, 1.25], [1.5, 2.0]])

    def test_response_evaluation_validates_source_and_modes(self):
        network = self._network()

        with self.assertRaisesRegex(ValueError, 'either file_path or network'):
            evaluate_touchstone_response([1.0])
        with self.assertRaises(TypeError):
            evaluate_touchstone_response([1.0], network=object())
        with self.assertRaisesRegex(ValueError, 'interpolation mode'):
            evaluate_touchstone_response(
                [1.0],
                network=network,
                interpolation='invalid',
            )
        with self.assertRaisesRegex(ValueError, 'out_of_band policy'):
            evaluate_touchstone_response(
                [1.0],
                network=network,
                out_of_band='invalid',
            )
        with self.assertRaisesRegex(ValueError, 'frequencies must be finite'):
            evaluate_touchstone_response([np.nan], network=network)


class TouchstoneStageTests(unittest.TestCase):
    @staticmethod
    def _two_port_network(frequencies, response, path=''):
        matrix = np.zeros((len(frequencies), 2, 2), dtype=complex)
        matrix[:, 1, 0] = response
        return TouchstoneNetwork(frequencies, matrix, path=path)

    @staticmethod
    def _trace(values, *, domain='rf_real', plane='awg_rf', sample_rate=2.0, lo_freq=0.0):
        return SignalTrace(
            np.arange(len(values), dtype=float) / sample_rate,
            np.asarray(values),
            sample_rate,
            domain,
            plane,
            lo_freq=lo_freq,
        )

    def test_flat_s21_scales_real_trace_and_records_path_metadata(self):
        network = self._two_port_network([0.0, 2.0], [0.5, 0.5], path='flat.s2p')
        stage = TouchstoneStage(network=network, output_plane='qubit_rf')
        trace = self._trace([1.0, -0.5, 0.25, 0.0])

        output = stage.apply(trace)

        np.testing.assert_allclose(output.values, 0.5 * trace.values, atol=1e-15)
        self.assertFalse(np.iscomplexobj(output.values))
        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(output.metadata['touchstone_file'], 'flat.s2p')
        self.assertEqual(output.metadata['input_port'], 1)
        self.assertEqual(output.metadata['output_port'], 2)

    def test_iq_absolute_mode_offsets_the_fft_axis_by_lo_frequency(self):
        network = self._two_port_network([4.0, 6.0], [1.0, 3.0])
        stage = TouchstoneStage(
            network=network,
            interpolation='cartesian',
            out_of_band='error',
        )
        trace = self._trace(
            [1.0 + 0.0j],
            domain='iq_complex',
            plane='awg_iq',
            lo_freq=5.0,
        )

        response = stage._evaluate_response(trace, np.array([-0.5, 0.0, 0.5]))

        np.testing.assert_allclose(response, [1.5, 2.0, 2.5])

    def test_iq_relative_mode_ignores_lo_frequency(self):
        network = self._two_port_network([-1.0, 1.0], [1.0, 3.0])
        stage = TouchstoneStage(
            network=network,
            interpolation='cartesian',
            frequency_mode='relative',
        )
        trace = self._trace(
            [1.0 + 0.0j],
            domain='iq_complex',
            plane='awg_iq',
            lo_freq=5.0,
        )

        response = stage._evaluate_response(trace, np.array([-0.5, 0.5]))

        np.testing.assert_allclose(response, [1.5, 2.5])

    def test_empty_trace_still_finalizes_plane_and_metadata(self):
        network = self._two_port_network([0.0, 1.0], [1.0, 1.0])
        stage = TouchstoneStage(network=network, output_plane='qubit_rf')

        output = stage.apply(self._trace([]))

        self.assertEqual(output.plane, 'qubit_rf')
        self.assertEqual(output.metadata['fft_length'], 0)
        self.assertEqual(output.metadata['last_stage'], 'touchstone')

    def test_stage_configuration_and_port_selection_are_validated(self):
        network = self._two_port_network([0.0, 1.0], [1.0, 1.0])
        invalid = (
            lambda: TouchstoneStage(),
            lambda: TouchstoneStage(network=object()),
            lambda: TouchstoneStage(network=network, interpolation='invalid'),
            lambda: TouchstoneStage(network=network, out_of_band='invalid'),
            lambda: TouchstoneStage(network=network, frequency_mode='invalid'),
            lambda: TouchstoneStage(network=network, output_port=3),
        )

        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()


class InverseFIRDesignTests(unittest.TestCase):
    @staticmethod
    def _one_port_network(gain):
        response = np.full((2, 1, 1), gain, dtype=complex)
        return TouchstoneNetwork([0.0, 1.0], response)

    def test_flat_response_produces_centered_inverse_impulse(self):
        kernel = design_inverse_fir_from_touchstone(
            lo_freq=0.5,
            sample_rate=1.0,
            num_taps=9,
            network=self._one_port_network(0.5),
            input_port=1,
            output_port=1,
            out_of_band='error',
            window=None,
        )

        expected = np.zeros(9, dtype=complex)
        expected[4] = 2.0
        np.testing.assert_allclose(kernel, expected, atol=1e-12)

    def test_threshold_guard_suppresses_deep_stopband_inverse(self):
        kernel = design_inverse_fir_from_touchstone(
            lo_freq=0.5,
            sample_rate=1.0,
            num_taps=9,
            network=self._one_port_network(0.01),
            input_port=1,
            output_port=1,
            threshold_db=-20.0,
            window='none',
        )

        np.testing.assert_allclose(kernel, 0.0)

    def test_inverse_design_validates_parameters_and_source(self):
        network = self._one_port_network(0.5)
        invalid = (
            dict(lo_freq=0.5, sample_rate=1.0, num_taps=0, network=network),
            dict(lo_freq=0.5, sample_rate=0.0, num_taps=9, network=network),
            dict(lo_freq=np.nan, sample_rate=1.0, num_taps=9, network=network),
            dict(
                lo_freq=0.5,
                sample_rate=1.0,
                num_taps=9,
                network=network,
                threshold_db=np.inf,
            ),
        )

        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    design_inverse_fir_from_touchstone(
                        input_port=1,
                        output_port=1,
                        **kwargs,
                    )

        with self.assertRaisesRegex(ValueError, 'either file_path or network'):
            design_inverse_fir_from_touchstone(
                lo_freq=0.5,
                sample_rate=1.0,
                num_taps=9,
                input_port=1,
                output_port=1,
            )


if __name__ == '__main__':
    unittest.main()
