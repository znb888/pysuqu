import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

import pysuqu.qubit.plotting as plotting_module
from pysuqu.qubit.plotting import plot_multi_qubit_energy_vs_flux
from pysuqu.qubit.types import CouplingResult, SweepResult


class MultiQubitPlottingTests(unittest.TestCase):
    def test_plotting_module_is_canonical_owner_boundary(self):
        self.assertEqual(plotting_module.__name__, 'pysuqu.qubit.plotting')
        self.assertTrue(plotting_module.__file__.endswith('pysuqu\\qubit\\plotting.py'))
        self.assertEqual(plot_multi_qubit_energy_vs_flux.__module__, 'pysuqu.qubit.plotting')
        self.assertEqual(
            plotting_module.plot_multi_qubit_coupling_strength_vs_flux.__module__,
            'pysuqu.qubit.plotting',
        )

    def test_plot_multi_qubit_energy_vs_flux_accepts_sweep_result(self):
        result = SweepResult(
            sweep_parameter='coupler_flux',
            sweep_values=[0.125, 0.25],
            series={
                '|[0, 0, 1]>': [1.0, 2.5],
                '|[1, 0, 0]>': [1.5, 3.0],
            },
        )

        with mock.patch('pysuqu.qubit.plotting.plt') as plt:
            plot_multi_qubit_energy_vs_flux(
                [9.9],
                [[0, 0, 1], [1, 0, 0]],
                result,
            )

        self.assertEqual(plt.plot.call_count, 2)
        first_call = plt.plot.call_args_list[0]
        second_call = plt.plot.call_args_list[1]
        self.assertEqual(first_call.args[0], [0.125, 0.25])
        np.testing.assert_allclose(first_call.args[1], np.array([1.0, 2.5]))
        self.assertEqual(second_call.args[0], [0.125, 0.25])
        np.testing.assert_allclose(second_call.args[1], np.array([1.5, 3.0]))

    def test_plot_multi_qubit_coupling_strength_vs_flux_accepts_coupling_result(self):
        result = CouplingResult(
            sweep_parameter='coupler_flux',
            sweep_values=[0.125, 0.25],
            coupling_values=[1.0, -0.25],
            metadata={'method': 'ES'},
        )

        with mock.patch('pysuqu.qubit.plotting.plt') as plt:
            from pysuqu.qubit.plotting import plot_multi_qubit_coupling_strength_vs_flux

            plot_multi_qubit_coupling_strength_vs_flux([9.9], result)

        plt.plot.assert_called_once()
        plot_call = plt.plot.call_args
        self.assertEqual(plot_call.args[0], [0.125, 0.25])
        np.testing.assert_allclose(plot_call.args[1], np.array([1.0, -0.25]))


if __name__ == '__main__':
    unittest.main()
