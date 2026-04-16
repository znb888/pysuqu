import unittest
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import QubitBase, pi
from pysuqu.qubit.single import FloatingTransmon, GroundedTransmon
from pysuqu.qubit.sweeps import (
    sweep_single_qubit_energy_vs_flux_base,
    sweep_single_qubit_energy_vs_flux_base_result,
)


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
    @staticmethod
    def _construct_real_floating_transmon():
        basic_element = [157e-15 - 3.8e-15, 157e-15 - 3.8e-15, 3.8e-15 + 9.8e-15, 8700]
        with redirect_stdout(StringIO()):
            return FloatingTransmon(
                basic_element,
                trunc_ener_level=20,
                cal_mode='Eigen',
            )

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

    def test_scalar_change_para_projects_floating_flux_without_integer_truncation(self):
        qubit = self._construct_real_floating_transmon()
        baseline_flux = np.array(qubit.get_element_matrices('flux'), copy=True)
        baseline_level_1 = float(qubit.get_energylevel(1))
        target_flux = 0.3

        try:
            qubit.change_para(flux=target_flux)

            np.testing.assert_allclose(
                qubit.get_element_matrices('flux'),
                np.array([[0.0, target_flux], [target_flux, 0.0]]),
            )
            self.assertGreater(abs(float(qubit.get_energylevel(1)) - baseline_level_1), 1e-6)
        finally:
            qubit.change_para(flux=baseline_flux)

    def test_structured_sweep_helper_tracks_real_floating_transmon_flux_spectrum(self):
        qubit = self._construct_real_floating_transmon()
        flux_origin = np.array(qubit.get_element_matrices('flux'), copy=True)
        result = sweep_single_qubit_energy_vs_flux_base_result(
            qubit,
            [
                np.array([[0.0, 0.2], [0.2, 0.0]]),
                np.array([[0.0, 0.3], [0.3, 0.0]]),
                np.array([[0.0, 0.4], [0.4, 0.0]]),
            ],
            upper_level=2,
        )

        self.assertEqual(result.sweep_parameter, 'flux_offset')
        self.assertEqual(len(result.sweep_values), 3)
        self.assertGreater(np.ptp(result.series['level_1']), 1e-6)
        self.assertGreater(np.ptp(result.series['level_2']), 1e-6)
        np.testing.assert_allclose(qubit.get_element_matrices('flux'), flux_origin)
