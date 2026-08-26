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

    def test_transmon_effective_capacitance_helper_matches_legacy_formula(self):
        ec = 2 * np.pi * 0.25
        expected = e**2 / (2.0 * ec * 1e9 * hbar)

        actual = circuit.transmon_effective_capacitance_from_ec(ec)

        self.assertAlmostEqual(actual, expected)

    def test_transmon_reflection_model_applies_termination_background(self):
        frequencies = np.array([4.9, 5.0, 5.1])
        open_model = circuit.TransmonReflectionModel(
            resonance_freq_ghz=5.0,
            external_t1_ns=100.0,
            termination="open",
        )
        short_model = circuit.TransmonReflectionModel(
            resonance_freq_ghz=5.0,
            external_t1_ns=100.0,
            termination="short",
        )

        self.assertEqual(open_model.background_reflection, 1.0 + 0.0j)
        self.assertEqual(short_model.background_reflection, -1.0 + 0.0j)
        np.testing.assert_allclose(
            short_model.reflection_coefficient(frequencies),
            -open_model.reflection_coefficient(frequencies),
        )
        np.testing.assert_allclose(
            open_model.iq_reflection_coefficient(np.array([5.0, 4.9])),
            open_model.reflection_coefficient(np.array([5.0, 5.1])),
        )

    def test_transmon_reflection_model_iir_matches_small_signal_response(self):
        model = circuit.TransmonReflectionModel(
            resonance_freq_ghz=5.0,
            external_t1_ns=10.0,
            internal_t1_ns=20.0,
        )
        sample_rate = 2.0
        lo_freq = 4.7
        signal_freq = 5.05
        time_ns = np.arange(800, dtype=float) / sample_rate
        drive = np.exp(2j * np.pi * (signal_freq - lo_freq) * time_ns)

        scattered = model.filter_scattered_iq(
            drive,
            sample_rate=sample_rate,
            lo_freq_ghz=lo_freq,
        )
        actual = scattered[-1] / drive[-1]
        expected = model.iq_reflection_coefficient(np.array([signal_freq]))[0] - model.background_reflection

        self.assertEqual(scattered[0], 0.0)
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)

    def test_load_reflection_resolver_accepts_model_callable_and_arrays(self):
        frequencies = np.array([4.9, 5.1])
        model = circuit.TransmonReflectionModel(5.0, 100.0)
        model_response = model.reflection_coefficient(frequencies)

        np.testing.assert_allclose(model(frequencies), model_response)
        np.testing.assert_allclose(
            model.input_impedance(frequencies),
            50.0 * (1.0 + model_response) / (1.0 - model_response),
        )
        with self.assertRaises(ValueError):
            model.input_impedance(frequencies, line_impedance_ohm=0.0)

        np.testing.assert_allclose(
            circuit.resolve_load_reflection_response(frequencies, model),
            model_response,
        )
        np.testing.assert_allclose(
            circuit.resolve_load_reflection_response(frequencies, lambda f: 0.2 + 0.01 * f),
            0.2 + 0.01 * frequencies,
        )
        np.testing.assert_allclose(
            circuit.resolve_load_reflection_response(frequencies, 0.35),
            np.full(frequencies.shape, 0.35),
        )
        with self.assertRaises(ValueError):
            circuit.resolve_load_reflection_response(frequencies, np.ones(3))

    def test_loaded_single_port_response_matches_geometric_series(self):
        frequencies = np.array([5.0, 5.1, 5.2])
        forward = np.array([0.8 + 0.1j, 0.7 - 0.2j, 0.6 + 0.05j])
        output_reflection = np.array([0.20 - 0.05j, 0.18 + 0.02j, 0.16 - 0.01j])
        load_reflection = 0.35 + 0.10j
        round_trip_delay_ns = 0.4

        actual = circuit.calculate_loaded_single_port_response(
            frequencies,
            forward_response=forward,
            output_reflection_response=output_reflection,
            load_reflection=load_reflection,
            round_trip_delay_ns=round_trip_delay_ns,
        )
        loop = output_reflection * load_reflection * np.exp(
            -2j * np.pi * frequencies * round_trip_delay_ns
        )
        np.testing.assert_allclose(actual, forward / (1.0 - loop))
        with self.assertRaises(ValueError):
            circuit.calculate_loaded_single_port_response(
                frequencies,
                forward_response=forward[:2],
                output_reflection_response=output_reflection,
                load_reflection=load_reflection,
            )

    def test_transmon_reflection_model_builds_from_qubit_and_resolves_grid(self):
        qubit = type("QubitStub", (), {"qubit_f01": 5.2, "Ec": 2 * np.pi * 0.22})()
        model = circuit.TransmonReflectionModel.from_qubit(
            qubit,
            couple_term=2.5e-15,
            couple_type="capac",
        )

        self.assertEqual(model.termination, "open")
        grid = model.adaptive_frequency_grid(span_hwhm=10.0, points=101)
        self.assertEqual(grid.shape, (101,))
        self.assertAlmostEqual(grid[50], model.resonance_freq_ghz)
        self.assertGreater(model.fwhm_ghz, 0.0)

        with self.assertRaises(ValueError):
            circuit.TransmonReflectionModel(5.0, 100.0, termination="invalid")
        with self.assertRaises(ValueError):
            model.adaptive_frequency_grid(points=2)

    def test_estimate_drive_line_t1_ns_matches_legacy_inductive_formula(self):
        qubit_frequency_ghz = 5.2
        ec = 2 * np.pi * 0.22
        mutual_inductance = 12e-15
        capacitance = circuit.transmon_effective_capacitance_from_ec(ec)
        expected = (
            50.0
            / ((2 * np.pi * qubit_frequency_ghz * 1e9) ** 4 * mutual_inductance**2 * capacitance)
            * 1e9
        )

        actual = circuit.estimate_drive_line_t1_ns(
            qubit_frequency_ghz=qubit_frequency_ghz,
            couple_term=mutual_inductance,
            couple_type="induc",
            ec=ec,
        )

        self.assertAlmostEqual(actual, expected)

    def test_estimate_drive_line_t1_ns_matches_finite_rc_capacitive_formula(self):
        qubit_frequency_ghz = 5.2
        ec = 2 * np.pi * 0.22
        coupling_capacitance = 2.5e-15
        line_impedance = 50.0
        omega = 2 * np.pi * qubit_frequency_ghz * 1e9
        capacitance = circuit.transmon_effective_capacitance_from_ec(ec)
        correction = 1.0 + (omega * coupling_capacitance * line_impedance) ** 2
        expected = (
            capacitance
            * correction
            / (omega**2 * coupling_capacitance**2 * line_impedance)
            * 1e9
        )

        actual = circuit.estimate_drive_line_t1_ns(
            qubit_frequency_ghz=qubit_frequency_ghz,
            couple_term=coupling_capacitance,
            couple_type="capac",
            ec=ec,
            line_impedance_ohm=line_impedance,
        )

        self.assertAlmostEqual(actual, expected)

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
