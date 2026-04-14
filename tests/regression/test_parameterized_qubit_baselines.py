import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit import base as qubit_base_module
from pysuqu.qubit.base import ParameterizedQubit, QubitBase
from pysuqu.qubit.solver import HamiltonianEvo


_BASE_WORKLOAD = {
    'capacitances': [
        [1.8, -0.25, -0.1],
        [-0.25, 1.6, -0.15],
        [-0.1, -0.15, 1.4],
    ],
    'junctions_resistance': [
        [1e20, 11.0, 1e20],
        [11.0, 1e20, 12.0],
        [1e20, 12.0, 13.0],
    ],
    'inductances': [
        [40.0, 90.0, 100.0],
        [90.0, 45.0, 95.0],
        [100.0, 95.0, 50.0],
    ],
    'fluxes': [
        [0.0, 0.17, 0.0],
        [0.17, 0.0, 0.0],
        [0.0, 0.0, 0.09],
    ],
    'trunc_ener_level': [2, 2],
    'junc_ratio': [
        [1.0, 1.15, 0.0],
        [1.15, 1.0, 0.0],
        [0.0, 0.0, 1.05],
    ],
    'structure_index': [2, 1],
}

_UPDATED_ELEMENTS = {
    'capac': [
        [1.9, -0.28, -0.12],
        [-0.28, 1.68, -0.18],
        [-0.12, -0.18, 1.48],
    ],
    'resis': [
        [1e20, 10.5, 1e20],
        [10.5, 1e20, 11.5],
        [1e20, 11.5, 12.5],
    ],
    'induc': [
        [42.0, 92.0, 102.0],
        [92.0, 47.0, 97.0],
        [102.0, 97.0, 52.0],
    ],
}

_SHIFTED_UPDATED_ELEMENTS = {
    'capac': [
        [1.93, -0.28, -0.12],
        [-0.28, 1.68, -0.18],
        [-0.12, -0.18, 1.48],
    ],
    'resis': _UPDATED_ELEMENTS['resis'],
    'induc': _UPDATED_ELEMENTS['induc'],
}


class ParameterizedQubitCircuitRebuildBaselineTests(unittest.TestCase):
    def setUp(self):
        QubitBase._clear_exact_solve_template_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)

    @staticmethod
    def _construct_parameterized_qubit():
        return ParameterizedQubit(**_BASE_WORKLOAD)

    def test_multi_node_parameterized_qubit_rebuild_keeps_preferred_surface_stable(self):
        qubit = self._construct_parameterized_qubit()

        qubit.change_para(**_UPDATED_ELEMENTS)

        self.assertEqual(set(qubit.get_element_matrices()), {'capac', 'induc', 'resis', 'flux'})
        self.assertEqual(set(qubit.get_energy_matrices()), {'Ec', 'El', 'Ej_max', 'Ej'})
        np.testing.assert_allclose(
            qubit.get_element_matrices('capac'),
            np.array(
                [
                    [1.9, -0.28, -0.12],
                    [-0.28, 1.68, -0.18],
                    [-0.12, -0.18, 1.48],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.get_element_matrices('resis'),
            np.array(
                [
                    [1e20, 10.5, 1e20],
                    [10.5, 1e20, 11.5],
                    [1e20, 11.5, 12.5],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.get_element_matrices('induc'),
            np.array(
                [
                    [42.0, 92.0, 102.0],
                    [92.0, 47.0, 97.0],
                    [102.0, 97.0, 52.0],
                ]
            ),
        )
        self.assertEqual(qubit.SMatrix.tolist(), [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(qubit.SMatrix_retainNodes, [0, 2])
        np.testing.assert_allclose(
            qubit.Maxwellmat['capac'],
            np.array(
                [
                    [1.5, 0.28, 0.12],
                    [0.28, 1.22, 0.18],
                    [0.12, 0.18, 1.18],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.Maxwellmat['induc'],
            np.array(
                [
                    [0.044483010595543, -0.010869565217391, -0.009803921568627],
                    [-0.010869565217391, 0.042455439312588, -0.010309278350515],
                    [-0.009803921568627, -0.010309278350515, 0.039343969149912],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.get_energy_matrices('Ec'),
            np.array(
                [
                    [2.285853058550994e-13, 8.489413009749221e-15],
                    [8.489413009749218e-15, 1.059120785461306e-13],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.get_energy_matrices('El'),
            np.array(
                [
                    [2.790457108324908e-08, 2.595156093483211e-10],
                    [2.595156093483211e-10, 4.040857664945163e-08],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.get_energy_matrices('Ej_max'),
            np.array(
                [
                    [83220.12099281019, 0.0],
                    [0.0, 69904.90163396056],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.get_energy_matrices('Ej'),
            np.array(
                [
                    [71692.0026477512, 0.0],
                    [0.0, 67130.92095206138],
                ]
            ),
        )
        np.testing.assert_allclose(
            qubit.solver_result.eigenvalues,
            np.array(
                [
                    -1.069894418873957e-07,
                    2.379770209458227e-04,
                    3.625978214434978e-04,
                    6.006818318312078e-04,
                ]
            ),
        )
        self.assertEqual(qubit.solver_result.hamiltonian.shape, (4, 4))
        self.assertEqual(len(qubit.solver_result.destroy_operators), 2)
        self.assertEqual(len(qubit.solver_result.number_operators), 2)
        self.assertEqual(len(qubit.solver_result.phase_operators), 2)
        self.assertEqual(qubit._last_changed_params, set())

    def test_multi_node_parameterized_qubit_reuses_exact_rebuild_template_with_copy_isolation(self):
        first = self._construct_parameterized_qubit()
        first.change_para(**_UPDATED_ELEMENTS)

        second = self._construct_parameterized_qubit()
        with mock.patch.object(
            QubitBase,
            '_generate_hamiltonian',
            autospec=True,
            wraps=QubitBase._generate_hamiltonian,
        ) as generate_hamiltonian, mock.patch.object(
            HamiltonianEvo,
            '_set_solver_result',
            autospec=True,
            wraps=HamiltonianEvo._set_solver_result,
        ) as set_solver_result:
            second.change_para(**_UPDATED_ELEMENTS)

        self.assertEqual(generate_hamiltonian.call_count, 0)
        self.assertEqual(set_solver_result.call_count, 0)
        np.testing.assert_allclose(first.solver_result.eigenvalues, second.solver_result.eigenvalues)
        self.assertIsNot(first._Hamiltonian, second._Hamiltonian)
        self.assertIsNot(first._eigenstates[0], second._eigenstates[0])
        self.assertIsNot(first.destroyors[0], second.destroyors[0])

        second._energylevels[0] = 123.0
        self.assertNotEqual(first._energylevels[0], second._energylevels[0])

    def test_multi_node_parameterized_qubit_rebuild_cache_invalidates_when_updated_elements_change(self):
        baseline = self._construct_parameterized_qubit()
        baseline.change_para(**_UPDATED_ELEMENTS)

        shifted = self._construct_parameterized_qubit()
        with mock.patch.object(
            QubitBase,
            '_generate_hamiltonian',
            autospec=True,
            wraps=QubitBase._generate_hamiltonian,
        ) as generate_hamiltonian, mock.patch.object(
            HamiltonianEvo,
            '_set_solver_result',
            autospec=True,
            wraps=HamiltonianEvo._set_solver_result,
        ) as set_solver_result:
            shifted.change_para(**_SHIFTED_UPDATED_ELEMENTS)

        self.assertEqual(generate_hamiltonian.call_count, 1)
        self.assertEqual(set_solver_result.call_count, 1)

    def test_multi_node_parameterized_qubit_rebuild_reuses_transformed_projection_state(self):
        with mock.patch.object(
            qubit_base_module,
            'project_transformed_flux',
            wraps=qubit_base_module.project_transformed_flux,
        ) as project_flux, mock.patch.object(
            qubit_base_module,
            'project_transformed_junction_ratio',
            wraps=qubit_base_module.project_transformed_junction_ratio,
        ) as project_ratio:
            qubit = self._construct_parameterized_qubit()
            qubit.change_para(**_UPDATED_ELEMENTS)

        self.assertEqual(project_flux.call_count, 2)
        self.assertEqual(project_ratio.call_count, 2)


if __name__ == '__main__':
    unittest.main()
