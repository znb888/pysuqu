import unittest

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit import circuit
from pysuqu.qubit.base import ParameterizedQubit


class ParameterizedQubitWorkflowTests(unittest.TestCase):
    def test_change_para_rebuilds_public_surface_from_real_constructor(self):
        qubit = ParameterizedQubit(
            capacitances=[[80e-15]],
            junctions_resistance=[[10_000]],
            inductances=[[1e20]],
            fluxes=[[0.125]],
            trunc_ener_level=[1],
            junc_ratio=[[1.2]],
            structure_index=[1],
        )
        updated_capac = np.array([[90e-15]])
        updated_resis = np.array([[9_500]])
        updated_induc = np.array([[9.5e19]])

        expected_s_matrix, expected_retain_nodes = circuit.assemble_s_matrix_and_retain_nodes([1])
        expected_maxwell, expected_ec, expected_el, expected_ej0 = circuit.convert_elements_to_energy_matrices(
            updated_capac,
            updated_induc,
            updated_resis,
            expected_s_matrix,
            expected_retain_nodes,
            [1],
            circuit.convert_resistance_to_ej0,
        )
        expected_flux = circuit.project_transformed_flux(qubit.get_element_matrices('flux'), [1], expected_retain_nodes)
        expected_ratio = circuit.project_transformed_junction_ratio(qubit._junc_ratio, [1], expected_retain_nodes)
        expected_ej = expected_ej0 * np.abs(np.cos(np.pi * expected_flux)) * np.sqrt(
            1 + ((expected_ratio - 1) * np.tan(np.pi * expected_flux) / (expected_ratio + 1)) ** 2
        )

        qubit.change_para(
            capac=updated_capac.tolist(),
            resis=updated_resis.tolist(),
            induc=updated_induc.tolist(),
        )

        np.testing.assert_allclose(qubit.get_element_matrices('capac'), updated_capac)
        np.testing.assert_allclose(qubit.get_element_matrices('resis'), updated_resis)
        np.testing.assert_allclose(qubit.get_element_matrices('induc'), updated_induc)
        np.testing.assert_allclose(qubit.SMatrix, expected_s_matrix)
        self.assertEqual(qubit.SMatrix_retainNodes, expected_retain_nodes)
        np.testing.assert_allclose(qubit.Maxwellmat['capac'], expected_maxwell['capac'])
        np.testing.assert_allclose(qubit.Maxwellmat['induc'], expected_maxwell['induc'])
        np.testing.assert_allclose(qubit.get_energy_matrices('Ec'), expected_ec)
        np.testing.assert_allclose(qubit.get_energy_matrices('El'), expected_el)
        np.testing.assert_allclose(qubit.get_energy_matrices('Ej_max'), expected_ej0)
        np.testing.assert_allclose(qubit.get_energy_matrices('Ej'), expected_ej)
        self.assertEqual(qubit.solver_result.hamiltonian.shape, qubit.get_hamiltonian().shape)
        self.assertEqual(len(qubit.solver_result.destroy_operators), 1)
        self.assertEqual(len(qubit.solver_result.number_operators), 1)
        self.assertEqual(len(qubit.solver_result.phase_operators), 1)
        self.assertEqual(qubit._last_changed_params, set())


if __name__ == '__main__':
    unittest.main()

