import inspect
import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import pi
from pysuqu.qubit.sweeps import (
    sweep_single_qubit_energy_vs_flux_base,
    sweep_single_qubit_energy_vs_flux_base_result,
)
from pysuqu.qubit.types import SweepResult


class SingleQubitSweepWrapperTests(unittest.TestCase):
    def test_single_qubit_sweep_helper_walks_offsets_and_restores_original_flux(self):
        qubit = mock.Mock()
        qubit._Nlevel = [6]
        qubit._flux = np.array([[0.2]])
        original_flux = qubit._flux.copy()

        def apply_params(**params):
            qubit._flux = np.array(params['flux'])

        qubit.change_para.side_effect = apply_params
        qubit.get_energylevel.side_effect = [2 * pi, 3 * pi, 4 * pi, 5 * pi]

        energies = sweep_single_qubit_energy_vs_flux_base(
            qubit,
            [np.array([[0.1]]), np.array([[-0.05]])],
            upper_level=2,
        )

        self.assertIsInstance(energies, SweepResult)
        np.testing.assert_allclose(energies.series['level_1'], np.array([1.0, 2.0]))
        np.testing.assert_allclose(energies.series['level_2'], np.array([1.5, 2.5]))
        self.assertEqual(qubit.change_para.call_count, 3)
        np.testing.assert_allclose(qubit._flux, original_flux)

    def test_structured_single_qubit_sweep_helper_returns_sweep_result_and_restores_flux(self):
        qubit = mock.Mock()
        qubit._Nlevel = [6]
        qubit._flux = np.array([[0.2]])
        original_flux = qubit._flux.copy()

        def apply_params(**params):
            qubit._flux = np.array(params['flux'])

        qubit.change_para.side_effect = apply_params
        qubit.get_energylevel.side_effect = [2 * pi, 3 * pi, 4 * pi, 5 * pi]

        result = sweep_single_qubit_energy_vs_flux_base_result(
            qubit,
            [np.array([[0.1]]), np.array([[-0.05]])],
            upper_level=2,
        )

        self.assertIsInstance(result, SweepResult)
        self.assertEqual(result.sweep_parameter, 'flux_offset')
        self.assertEqual(len(result.sweep_values), 2)
        np.testing.assert_allclose(result.sweep_values[0], np.array([[0.1]]))
        np.testing.assert_allclose(result.sweep_values[1], np.array([[-0.05]]))
        np.testing.assert_allclose(result.series['level_1'], np.array([1.0, 2.0]))
        np.testing.assert_allclose(result.series['level_2'], np.array([1.5, 2.5]))
        np.testing.assert_allclose(result.metadata['flux_origin'], original_flux)
        np.testing.assert_allclose(qubit._flux, original_flux)
        self.assertEqual(qubit.change_para.call_count, 3)

    def test_single_qubit_sweep_helper_matches_structured_result_helper(self):
        qubit = mock.Mock()
        structured_result = SweepResult(
            sweep_parameter='flux_offset',
            sweep_values=[np.array([[0.1]])],
            series={
                'level_1': np.array([1.0]),
                'level_2': np.array([1.5]),
            },
        )

        with mock.patch(
            'pysuqu.qubit.sweeps.sweep_single_qubit_energy_vs_flux_base_result',
            return_value=structured_result,
        ) as sweep:
            result = sweep_single_qubit_energy_vs_flux_base(
                qubit,
                [np.array([[0.1]])],
                upper_level=2,
            )

        sweep.assert_called_once_with(
            qubit,
            [np.array([[0.1]])],
            upper_level=2,
        )
        self.assertIs(result, structured_result)


if __name__ == '__main__':
    unittest.main()

