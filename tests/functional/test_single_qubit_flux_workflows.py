import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import QubitBase, pi
from pysuqu.qubit.single import FloatingTransmon, GroundedTransmon
from pysuqu.qubit.sweeps import sweep_single_qubit_energy_vs_flux_base


class GroundedTransmonFluxWorkflowTests(unittest.TestCase):
    def test_envs_flux_walks_offsets_and_restores_original_flux(self):
        qubit = GroundedTransmon.__new__(GroundedTransmon)
        qubit._Nlevel = 6
        qubit._flux = 0.2
        qubit._ParameterizedQubit__struct = [1]
        qubit._ParameterizedQubit__nodes = 1
        qubit.SMatrix_retainNodes = [0]
        qubit._recalculate_hamiltonian = mock.Mock()
        qubit.change_para = QubitBase.change_para.__get__(qubit, GroundedTransmon)
        qubit.get_energylevel = mock.Mock(side_effect=[2 * pi, 3 * pi, 4 * pi, 5 * pi])

        result = sweep_single_qubit_energy_vs_flux_base(
            qubit,
            [np.array([[0.1]]), np.array([[-0.05]])],
            upper_level=2,
        )

        np.testing.assert_allclose(result.series['level_1'], np.array([1.0, 2.0]))
        np.testing.assert_allclose(result.series['level_2'], np.array([1.5, 2.5]))
        self.assertEqual(qubit._recalculate_hamiltonian.call_count, 3)
        self.assertEqual(qubit._flux, 0.2)


class FloatingTransmonFluxWorkflowTests(unittest.TestCase):
    def test_envs_flux_updates_matrix_flux_and_restores_scalar_flux(self):
        qubit = FloatingTransmon.__new__(FloatingTransmon)
        qubit._Nlevel = 7
        qubit._flux = np.array([[0.0, 0.25], [0.25, 0.0]])
        qubit._ParameterizedQubit__struct = [2]
        qubit._ParameterizedQubit__nodes = 2
        qubit.SMatrix_retainNodes = [0]
        qubit.Ec = np.array([[1.0]])
        qubit.El = np.array([[2.0]])
        qubit.Ej = np.array([[3.0]])
        qubit._update_Ej = mock.Mock()
        qubit._generate_hamiltonian = mock.Mock(return_value='hamiltonian')
        qubit.change_hamiltonian = mock.Mock()
        qubit._recalculate_hamiltonian = mock.Mock()
        qubit.change_para = QubitBase.change_para.__get__(qubit, FloatingTransmon)
        qubit.get_energylevel = mock.Mock(side_effect=[2 * pi, 3 * pi, 4 * pi, 5 * pi])

        result = sweep_single_qubit_energy_vs_flux_base(
            qubit,
            [
                np.array([[0.0, 0.1], [0.1, 0.0]]),
                np.array([[0.0, -0.15], [-0.15, 0.0]]),
            ],
            upper_level=2,
        )

        np.testing.assert_allclose(result.series['level_1'], np.array([1.0, 2.0]))
        np.testing.assert_allclose(result.series['level_2'], np.array([1.5, 2.5]))
        self.assertEqual(qubit._update_Ej.call_count, 0)
        self.assertEqual(qubit._generate_hamiltonian.call_count, 0)
        self.assertEqual(qubit.change_hamiltonian.call_count, 0)
        self.assertEqual(qubit._recalculate_hamiltonian.call_count, 3)
        np.testing.assert_allclose(qubit._flux, np.array([[0.0, 0.25], [0.25, 0.0]]))

