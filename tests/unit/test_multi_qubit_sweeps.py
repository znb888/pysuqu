import inspect
import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.multi import FGF1V1Coupling
from pysuqu.qubit.sweeps import (
    sweep_multi_qubit_coupling_strength_vs_flux,
    sweep_multi_qubit_coupling_strength_vs_flux_result,
    sweep_multi_qubit_energy_vs_flux,
    sweep_multi_qubit_energy_vs_flux_result,
)
from pysuqu.qubit.types import CouplingResult, SweepResult


class MultiQubitSweepWrapperTests(unittest.TestCase):
    def test_sweep_helper_delegates_plotting_branch_to_plotting_module(self):
        qubit = mock.Mock()
        qubit._flux = np.zeros((5, 5), dtype=float)
        qubit._Nlevel = [3, 3, 3]
        qubit.find_state.side_effect = [0, 1, 2]
        qubit.get_energylevel.side_effect = [2 * np.pi, 3 * np.pi, 4 * np.pi]

        with mock.patch(
            'pysuqu.qubit.sweeps.plot_multi_qubit_energy_vs_flux',
        ) as plot:
            energies = sweep_multi_qubit_energy_vs_flux(
                qubit,
                [0.25],
                qubits_flux=None,
                cal_state=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                is_plot=True,
            )

        plot.assert_called_once_with(
            [0.25],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            energies,
        )
        self.assertIsInstance(energies, SweepResult)
        self.assertEqual(energies.sweep_values, [0.25])
        np.testing.assert_allclose(energies.series['|[0, 0, 1]>'], np.array([1.0]))
        np.testing.assert_allclose(energies.series['|[1, 0, 0]>'], np.array([1.5]))
        np.testing.assert_allclose(energies.series['|[0, 1, 0]>'], np.array([2.0]))

    def test_structured_energy_sweep_helper_returns_sweep_result_and_restores_flux(self):
        qubit = mock.Mock()
        qubit._flux = np.zeros((5, 5), dtype=float)
        qubit._Nlevel = [3, 3, 3]
        qubit.find_state.side_effect = [0, 1, 2]
        qubit.get_energylevel.side_effect = [
            2 * np.pi,
            3 * np.pi,
            4 * np.pi,
            5 * np.pi,
            6 * np.pi,
            7 * np.pi,
        ]
        original_flux = qubit._flux.copy()

        result = sweep_multi_qubit_energy_vs_flux_result(
            qubit,
            [0.25, 0.35],
            qubits_flux=None,
            cal_state=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            is_plot=False,
        )

        self.assertIsInstance(result, SweepResult)
        self.assertEqual(result.sweep_parameter, 'coupler_flux')
        self.assertEqual(result.sweep_values, [0.25, 0.35])
        self.assertIsNone(result.metadata['qubits_flux'])
        np.testing.assert_allclose(result.series['|[0, 0, 1]>'], np.array([1.0, 2.5]))
        np.testing.assert_allclose(result.series['|[1, 0, 0]>'], np.array([1.5, 3.0]))
        np.testing.assert_allclose(result.series['|[0, 1, 0]>'], np.array([2.0, 3.5]))
        np.testing.assert_allclose(qubit._flux, original_flux)
        self.assertEqual(qubit.change_para.call_count, 3)

    def test_structured_energy_sweep_helper_passes_sweep_result_to_plotting_module(self):
        qubit = mock.Mock()
        qubit._flux = np.zeros((5, 5), dtype=float)
        qubit._Nlevel = [3, 3, 3]
        qubit.find_state.side_effect = [0, 1, 2]
        qubit.get_energylevel.side_effect = [2 * np.pi, 3 * np.pi, 4 * np.pi]

        with mock.patch('pysuqu.qubit.sweeps.plot_multi_qubit_energy_vs_flux') as plot:
            result = sweep_multi_qubit_energy_vs_flux_result(
                qubit,
                [0.25],
                qubits_flux=None,
                cal_state=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                is_plot=True,
            )

        plot.assert_called_once()
        plot_args = plot.call_args.args
        self.assertEqual(plot_args[0], [0.25])
        self.assertEqual(plot_args[1], [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.assertIs(plot_args[2], result)

    def test_structured_coupling_sweep_helper_returns_coupling_result_and_restores_flux(self):
        qubit = mock.Mock()
        qubit._flux = np.zeros((5, 5), dtype=float)
        qubit.get_qq_ecouple.side_effect = [1.0, -0.25]
        original_flux = qubit._flux.copy()

        result = sweep_multi_qubit_coupling_strength_vs_flux_result(
            qubit,
            [0.25, 0.35],
            method='ES',
            is_plot=False,
        )

        self.assertIsInstance(result, CouplingResult)
        self.assertEqual(result.sweep_parameter, 'coupler_flux')
        self.assertEqual(result.sweep_values, [0.25, 0.35])
        self.assertEqual(result.metadata['method'], 'ES')
        np.testing.assert_allclose(result.coupling_values, np.array([1.0, -0.25]))
        np.testing.assert_allclose(qubit._flux, original_flux)
        self.assertEqual(qubit.change_para.call_count, 3)

    def test_coupling_strength_sweep_helper_delegates_plotting_branch_to_plotting_module(self):
        qubit = mock.Mock()
        qubit._flux = np.zeros((5, 5), dtype=float)
        qubit.get_qq_ecouple.side_effect = [1.0, -0.25]

        with mock.patch(
            'pysuqu.qubit.sweeps.plot_multi_qubit_coupling_strength_vs_flux',
        ) as plot:
            coupling = sweep_multi_qubit_coupling_strength_vs_flux(
                qubit,
                [0.25, 0.35],
                method='ES',
                is_plot=True,
            )

        self.assertIsInstance(coupling, CouplingResult)
        np.testing.assert_allclose(coupling.coupling_values, np.array([1.0, -0.25]))
        self.assertEqual(qubit.change_para.call_count, 3)
        plot.assert_called_once_with([0.25, 0.35], coupling)

    def test_structured_coupling_sweep_helper_passes_coupling_result_to_plotting_module(self):
        qubit = mock.Mock()
        qubit._flux = np.zeros((5, 5), dtype=float)
        qubit.get_qq_ecouple.side_effect = [1.0, -0.25]

        with mock.patch(
            'pysuqu.qubit.sweeps.plot_multi_qubit_coupling_strength_vs_flux',
        ) as plot:
            result = sweep_multi_qubit_coupling_strength_vs_flux_result(
                qubit,
                [0.25, 0.35],
                method='ES',
                is_plot=True,
            )

        plot.assert_called_once()
        plot_args = plot.call_args.args
        self.assertEqual(plot_args[0], [0.25, 0.35])
        self.assertIs(plot_args[1], result)


if __name__ == '__main__':
    unittest.main()
