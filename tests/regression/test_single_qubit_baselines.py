import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit import analyze_single_qubit_spectrum
from pysuqu.qubit.base import ParameterizedQubit
from pysuqu.qubit.single import GroundedTransmon, SingleQubitBase, FloatingTransmon


class SingleQubitConstructorBaselineTests(unittest.TestCase):
    def test_grounded_transmon_normalizes_constructor_inputs(self):
        with mock.patch.object(ParameterizedQubit, '__init__', autospec=True, return_value=None) as qubit_init:
            with mock.patch.object(SingleQubitBase, 'print_basic_info', autospec=True):
                qubit = GroundedTransmon(
                    capacitance=80e-15,
                    junction_resistance=10_000,
                    inductance=1e20,
                    flux=0.125,
                    trunc_ener_level=12,
                    junc_ratio=1.2,
                    qr_couple=[3e-15],
                )

        _, args, kwargs = qubit_init.mock_calls[0]
        self.assertEqual(qubit.qubit_class, 'GroundedTransmon')
        self.assertEqual(qubit._Nlevel, [12])
        self.assertEqual(qubit._flux, 0.125)
        self.assertEqual(kwargs['structure_index'], [1])
        np.testing.assert_allclose(kwargs['capacitances'], np.array([[80e-15]]))
        np.testing.assert_allclose(kwargs['junctions_resistance'], np.array([[10_000.0]]))
        np.testing.assert_allclose(kwargs['inductances'], np.array([[1e20]]))
        np.testing.assert_allclose(kwargs['fluxes'], np.array([[0.125]]))
        np.testing.assert_allclose(kwargs['junc_ratio'], np.array([[1.2]]))

    def test_floating_transmon_builds_matrix_style_inputs(self):
        with mock.patch.object(ParameterizedQubit, '__init__', autospec=True, return_value=None) as qubit_init:
            with mock.patch.object(SingleQubitBase, 'print_basic_info', autospec=True):
                qubit = FloatingTransmon(
                    basic_element=[110e-15, 125e-15, 8e-15, 9_500],
                    flux=0.2,
                    trunc_ener_level=14,
                    junc_ratio=1.1,
                    qr_couple=[10e-15, 2e-15],
                )

        _, args, kwargs = qubit_init.mock_calls[0]
        self.assertEqual(qubit.qubit_class, 'FloatingTransmon')
        self.assertEqual(qubit._Nlevel, [14])
        np.testing.assert_allclose(qubit._flux, np.array([[0.0, 0.2], [0.2, 0.0]]))
        self.assertEqual(kwargs['structure_index'], [2])
        np.testing.assert_allclose(
            kwargs['capacitances'],
            np.array([[110e-15, 8e-15], [8e-15, 125e-15]]),
        )
        np.testing.assert_allclose(
            kwargs['junctions_resistance'],
            np.array([[1e20, 9_500.0], [9_500.0, 1e20]]),
        )
        np.testing.assert_allclose(
            kwargs['fluxes'],
            np.array([[0.0, 0.2], [0.2, 0.0]]),
        )
        np.testing.assert_allclose(
            kwargs['junc_ratio'],
            np.array([[1.0, 1.1], [1.1, 1.0]]),
        )


class SingleQubitRuntimeBaselineTests(unittest.TestCase):
    @staticmethod
    def _construct_grounded_transmon():
        with redirect_stdout(StringIO()):
            return GroundedTransmon(
                capacitance=82e-15,
                junction_resistance=9_800,
                inductance=1e20,
                flux=0.11,
                trunc_ener_level=8,
                junc_ratio=1.07,
                qr_couple=[4.5e-15],
            )

    @staticmethod
    def _construct_floating_transmon():
        with redirect_stdout(StringIO()):
            return FloatingTransmon(
                basic_element=[112e-15, 128e-15, 7.5e-15, 9_600],
                flux=0.11,
                trunc_ener_level=8,
                junc_ratio=1.08,
                qr_couple=[9.5e-15, 1.5e-15],
            )

    def test_grounded_transmon_runtime_baseline_locks_public_outputs(self):
        qubit = self._construct_grounded_transmon()
        spectrum = analyze_single_qubit_spectrum(qubit)

        self.assertAlmostEqual(qubit.f01, 4.774368463086741, places=6)
        self.assertAlmostEqual(qubit.anharmonicity, -0.26713677096567423, places=6)
        self.assertAlmostEqual(qubit.qr_g / (2 * np.pi * 1e6), 70.57922911997767, places=6)
        self.assertAlmostEqual(spectrum.f01, qubit.f01, places=12)
        self.assertAlmostEqual(spectrum.anharmonicity, qubit.anharmonicity, places=12)

    def test_floating_transmon_runtime_baseline_locks_public_outputs(self):
        qubit = self._construct_floating_transmon()
        spectrum = analyze_single_qubit_spectrum(qubit)

        self.assertAlmostEqual(qubit.f01, 5.299814541854555, places=6)
        self.assertAlmostEqual(qubit.anharmonicity, -0.33010364555315697, places=6)
        self.assertAlmostEqual(qubit.qr_g / (2 * np.pi * 1e6), 62.478695239773046, places=6)
        self.assertAlmostEqual(spectrum.f01, qubit.f01, places=12)
        self.assertAlmostEqual(spectrum.anharmonicity, qubit.anharmonicity, places=12)
