import unittest
from pathlib import Path

import numpy as np
from scipy.constants import e, hbar, pi

from tests.support import install_test_stubs

install_test_stubs()

import pysuqu.qubit.base as base_module
from pysuqu.qubit import circuit
from pysuqu.qubit.base import ParameterizedQubit


class QubitCircuitModuleTests(unittest.TestCase):
    def test_circuit_module_converts_scalar_resistance_to_ej0(self):
        resistance = 500.0
        expected = 280e-9 * 1000 / resistance * hbar / 2 / e

        self.assertAlmostEqual(circuit.convert_resistance_to_ej0(resistance), expected)

    def test_circuit_module_converts_element_matrices_to_energy_matrices(self):
        capac = np.array([[2.0, -1.0], [-1.0, 3.0]])
        induc = np.array([[4.0, 20.0], [20.0, 5.0]])
        resis = np.array([[6.0, 7.0], [8.0, 9.0]])
        s_matrix = np.eye(2)
        retain_nodes = [0, 1]
        struct = [1, 1]

        def resistance_to_ej0(value):
            return value * 10.0

        maxwell, ec_matrix, el_matrix, ej0_matrix = circuit.convert_elements_to_energy_matrices(
            capac,
            induc,
            resis,
            s_matrix,
            retain_nodes,
            struct,
            resistance_to_ej0,
        )

        expected_m_cap = np.array([[1.0, 1.0], [1.0, 2.0]])
        expected_m_ind = np.array([[0.3, -0.05], [-0.05, 0.25]])
        expected_ec = e**2 * np.linalg.inv(expected_m_cap) / 2 / hbar / 1e9
        expected_el = (hbar / e) ** 2 * expected_m_ind / 4 / hbar / 1e9
        expected_ej0 = np.diag([60.0, 90.0]) / hbar / 1e9

        np.testing.assert_allclose(maxwell['capac'], expected_m_cap)
        np.testing.assert_allclose(maxwell['induc'], expected_m_ind)
        np.testing.assert_allclose(ec_matrix, expected_ec)
        np.testing.assert_allclose(el_matrix, expected_el)
        np.testing.assert_allclose(ej0_matrix, expected_ej0)

    def test_circuit_module_builds_retain_nodes_and_extracts_reduced_flux(self):
        struct = [1, 2, 1]
        full_flux = np.array(
            [
                [0.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 0.22, 5.0],
                [4.0, 0.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 0.33],
            ]
        )

        retain_nodes = circuit.build_retain_nodes(struct)
        reduced_flux = circuit.extract_reduced_flux(full_flux, struct, retain_nodes)

        self.assertEqual(retain_nodes, [0, 1, 3])
        self.assertEqual(circuit.build_retain_nodes.__module__, 'pysuqu.qubit.circuit')
        np.testing.assert_allclose(reduced_flux, np.diag([0.11, 0.22, 0.33]))

    def test_circuit_module_updates_only_retained_flux_entries(self):
        struct = [1, 2, 1]
        retain_nodes = [0, 1, 3]
        current_full = np.array(
            [
                [0.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 0.22, 5.0],
                [4.0, 0.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 0.33],
            ]
        )
        reduced_flux = np.diag([0.44, 0.55, 0.66])

        updated_full = circuit.update_full_flux_from_reduced(reduced_flux, current_full, struct, retain_nodes)

        expected = current_full.copy()
        expected[0, 0] = 0.44
        expected[1, 2] = 0.55
        expected[2, 1] = 0.55
        expected[3, 3] = 0.66
        np.testing.assert_allclose(updated_full, expected)

    def test_circuit_module_assembles_s_matrix_and_retain_nodes(self):
        struct = [1, 2, 1]

        s_matrix, retain_nodes = circuit.assemble_s_matrix_and_retain_nodes(struct)

        expected_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(s_matrix, expected_matrix)
        self.assertEqual(retain_nodes, [0, 1, 3])

    def test_parameterized_qubit_normalize_flux_input_accepts_reduced_flux_via_circuit_helpers(self):
        qubit = ParameterizedQubit.__new__(ParameterizedQubit)
        qubit._flux = np.array(
            [
                [0.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 0.22, 5.0],
                [4.0, 0.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 0.33],
            ]
        )
        qubit._ParameterizedQubit__struct = [1, 2, 1]
        qubit._ParameterizedQubit__nodes = 4

        normalized_flux = qubit._normalize_flux_input(np.diag([0.44, 0.55, 0.66]))

        expected = qubit._flux.copy()
        expected[0, 0] = 0.44
        expected[1, 2] = 0.55
        expected[2, 1] = 0.55
        expected[3, 3] = 0.66
        np.testing.assert_allclose(normalized_flux, expected)

    def test_circuit_module_projects_transformed_flux_from_scalar_or_matrix_inputs(self):
        struct = [1, 2, 1]
        retain_nodes = [0, 1, 3]
        full_flux = np.array(
            [
                [0.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 0.22, 5.0],
                [4.0, 0.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 0.33],
            ]
        )

        scalar_projection = circuit.project_transformed_flux(np.array(0.44), struct, retain_nodes)
        matrix_projection = circuit.project_transformed_flux(full_flux, struct, retain_nodes)

        np.testing.assert_allclose(scalar_projection, np.array([0.44, 0.44, 0.44]))
        np.testing.assert_allclose(matrix_projection, np.diag([0.11, 0.22, 0.33]))

    def test_circuit_module_projects_transformed_junction_ratio_from_matrix_or_vector_inputs(self):
        struct = [1, 2, 1]
        retain_nodes = [0, 1, 3]
        full_ratio = np.array(
            [
                [1.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 1.22, 5.0],
                [4.0, 1.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 1.33],
            ]
        )
        vector_ratio = np.array([1.1, 1.2, 1.3])

        matrix_projection = circuit.project_transformed_junction_ratio(full_ratio, struct, retain_nodes)
        vector_projection = circuit.project_transformed_junction_ratio(vector_ratio, struct, retain_nodes)

        np.testing.assert_allclose(matrix_projection, np.diag([1.11, 1.22, 1.33]))
        np.testing.assert_allclose(vector_projection, vector_ratio)

    def test_parameterized_qubit_update_transformed_vars_uses_circuit_projection_helpers(self):
        qubit = ParameterizedQubit.__new__(ParameterizedQubit)
        qubit._flux = np.array(
            [
                [0.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 0.22, 5.0],
                [4.0, 0.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 0.33],
            ]
        )
        qubit._junc_ratio = np.array(
            [
                [1.11, 9.0, 8.0, 7.0],
                [6.0, 0.0, 1.22, 5.0],
                [4.0, 1.22, 3.0, 2.0],
                [1.0, 1.0, 1.0, 1.33],
            ]
        )
        qubit._ParameterizedQubit__struct = [1, 2, 1]
        qubit.SMatrix_retainNodes = [0, 1, 3]

        qubit._update_transformed_vars()

        np.testing.assert_allclose(qubit._flux_transformed, np.diag([0.11, 0.22, 0.33]))
        np.testing.assert_allclose(qubit._junc_ratio_transformed, np.diag([1.11, 1.22, 1.33]))

    def test_base_module_no_longer_declares_inline_circuit_helpers(self):
        source = Path(base_module.__file__).read_text(encoding='utf-8')
        circuit_source = Path(circuit.__file__).read_text(encoding='utf-8')

        self.assertNotIn('def _build_retain_nodes', source)
        self.assertNotIn('def _update_full_flux_from_reduced', source)
        self.assertNotIn('def _extract_reduced_flux', source)
        self.assertNotIn('def _R2Ej0', source)
        self.assertNotIn('def _SMatrix_RetainNodes', source)
        self.assertNotIn('self._flux[self.SMatrix_retainNodes[ii]][self.SMatrix_retainNodes[ii] + 1]', source)
        self.assertNotIn('self._junc_ratio[self.SMatrix_retainNodes[ii]][self.SMatrix_retainNodes[ii] + 1]', source)
        self.assertNotIn('M_cap = -self.__capac', source)
        self.assertNotIn('R2Ej = np.vectorize(self._R2Ej0)', source)
        self.assertIn('project_transformed_flux', source)
        self.assertIn('project_transformed_junction_ratio', source)
        self.assertIn('convert_elements_to_energy_matrices', source)
        self.assertIn('convert_resistance_to_ej0', source)
        self.assertIn('assemble_s_matrix_and_retain_nodes', source)
        self.assertIn('def assemble_s_matrix_and_retain_nodes', circuit_source)
        self.assertIn('def convert_resistance_to_ej0', circuit_source)


if __name__ == '__main__':
    unittest.main()
