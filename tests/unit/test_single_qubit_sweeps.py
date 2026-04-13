from copy import copy
import inspect
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import pi
from pysuqu.qubit.single import GroundedTransmon
from pysuqu.qubit.sweeps import (
    sweep_single_qubit_energy_vs_flux_base,
    sweep_single_qubit_energy_vs_flux_base_result,
)
from pysuqu.qubit.types import SweepResult


_GROUNDED_TRANSMON_CONFIG = {
    'capacitance': 80e-15,
    'junction_resistance': 10_000,
    'inductance': 1e20,
    'flux': 0.125,
    'trunc_ener_level': 4,
    'junc_ratio': 1.2,
    'qr_couple': [3e-15],
}


def _construct_grounded_transmon():
    with redirect_stdout(StringIO()):
        return GroundedTransmon(**_GROUNDED_TRANSMON_CONFIG)


def _run_reference_single_qubit_sweep(qubit, flux_offsets, upper_level):
    flux_origin = copy(qubit._flux)
    energy_series = {f'level_{ii}': [] for ii in range(1, upper_level + 1)}

    try:
        for offset in flux_offsets:
            qubit.change_para(flux=flux_origin + offset)
            relative_energylevels = np.asarray(qubit.get_energylevel(), dtype=float)
            for ii in range(1, upper_level + 1):
                energy_series[f'level_{ii}'].append(relative_energylevels[ii] / 2 / pi)

        return SweepResult(
            sweep_parameter='flux_offset',
            sweep_values=flux_offsets,
            series=energy_series,
            metadata={
                'flux_origin': np.array(flux_origin, copy=True),
                'upper_level': upper_level,
            },
        )
    finally:
        qubit.change_para(flux=flux_origin)


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


class SingleQubitSweepFastPathTests(unittest.TestCase):
    def test_structured_helper_matches_reference_grounded_transmon_sweep(self):
        flux_offsets = [
            np.array([[-0.03]], dtype=float),
            np.array([[0.0]], dtype=float),
            np.array([[0.025]], dtype=float),
        ]
        reference_qubit = _construct_grounded_transmon()
        optimized_qubit = _construct_grounded_transmon()

        expected = _run_reference_single_qubit_sweep(reference_qubit, flux_offsets, upper_level=3)
        result = sweep_single_qubit_energy_vs_flux_base_result(
            optimized_qubit,
            flux_offsets,
            upper_level=3,
        )

        self.assertIsInstance(result, SweepResult)
        self.assertEqual(result.sweep_parameter, expected.sweep_parameter)
        self.assertEqual(result.metadata['upper_level'], expected.metadata['upper_level'])
        np.testing.assert_allclose(result.metadata['flux_origin'], expected.metadata['flux_origin'])
        for key, expected_values in expected.series.items():
            np.testing.assert_allclose(result.series[key], expected_values)

        restored_flux = optimized_qubit.get_element_matrices('flux')
        self.assertEqual(np.asarray(restored_flux).shape, ())
        self.assertAlmostEqual(float(restored_flux), _GROUNDED_TRANSMON_CONFIG['flux'])


if __name__ == '__main__':
    unittest.main()
