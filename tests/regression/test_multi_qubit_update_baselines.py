import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from qutip import Qobj, basis, tensor

from pysuqu.qubit import base as base_module
from pysuqu.qubit import multi as multi_module
from pysuqu.qubit.analysis import get_multi_qubit_frequency_at_coupler_flux
from pysuqu.qubit.base import ParameterizedQubit, QubitBase
from pysuqu.qubit.multi import FGF1V1Coupling, QCRFGRModel
from pysuqu.qubit.sweeps import sweep_multi_qubit_coupling_strength_vs_flux_result
from pysuqu.qubit.solver import HamiltonianEvo
from pysuqu.qubit.types import SpectrumResult


class FakeHamiltonian:
    def __init__(self, dims, qubit_frequency_ghz=6.35):
        self.dims = dims
        self.qubit_frequency_ghz = qubit_frequency_ghz

    def eigenstates(self):
        ground = tensor(basis(3, 0), basis(3, 0))
        qubit_excited = tensor(basis(3, 1), basis(3, 0))
        coupler_excited = tensor(basis(3, 0), basis(3, 1))
        return (
            np.array([0.0, 2 * np.pi * self.qubit_frequency_ghz, 2 * np.pi * 9.0]),
            [ground, qubit_excited, coupler_excited],
        )


class MultiQubitUpdateBaselineTests(unittest.TestCase):
    def make_qcrfgr_model(self):
        model = QCRFGRModel.__new__(QCRFGRModel)
        model._flux = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.0, 0.2],
            ],
            dtype=float,
        )
        model._junc_ratio = np.ones_like(model._flux)
        model.SMatrix_retainNodes = [0, 2]
        model._ParameterizedQubit__struct = [2, 1]
        model._ParameterizedQubit__nodes = 3
        model._ParameterizedQubit__capac = np.eye(3)
        model._ParameterizedQubit__resis = np.eye(3)
        model._ParameterizedQubit__induc = np.ones((3, 3))
        model.Ec = np.diag([1.0, 2.0])
        model.El = np.diag([4.0, 5.0])
        model.Ejmax = np.diag([7.0, 8.0])
        baseline_hamiltonian = FakeHamiltonian([[3, 3], [3, 3]], qubit_frequency_ghz=6.2)
        baseline_eigenvalues, baseline_eigenstates = baseline_hamiltonian.eigenstates()
        model._hamiltonian = baseline_hamiltonian
        model._Hamiltonian = baseline_hamiltonian
        model._energylevels = baseline_eigenvalues
        model._eigenstates = baseline_eigenstates
        model._numQubits = 2
        model._Nlevel = [3, 3]
        model._cal_mode = 'Eigen'
        model._charges = np.array([0, 0])
        model._generate_hamiltonian = mock.Mock(
            return_value=FakeHamiltonian([[3, 3], [3, 3]], qubit_frequency_ghz=6.35)
        )
        model._refresh_basic_metrics = mock.Mock(side_effect=lambda: setattr(model, 'qubit_f01', 6.0 + model._flux[2, 2]))
        model.eigenHamiltonian = 'baseline-eigen'
        model.couplingHamiltonian = 'baseline-coupling'
        model.highorderHamiltonian = 'baseline-highorder'
        model.destroyors = ['baseline-destroyor']
        model.n_operators = ['baseline-number']
        model.phi_operators = ['baseline-phase']
        model._solver_result = SpectrumResult(
            hamiltonian=baseline_hamiltonian,
            eigenvalues=np.array(baseline_eigenvalues, copy=True),
            eigenstates=list(baseline_eigenstates),
            destroy_operators=model.destroyors,
            number_operators=model.n_operators,
            phase_operators=model.phi_operators,
        )
        model.change_para = ParameterizedQubit.change_para.__get__(model, QCRFGRModel)
        return model

    def test_qcrfgr_coupler_flux_probe_restores_full_flux_state(self):
        multi_module._clear_qcrfgr_probe_frequency_cache()
        model = self.make_qcrfgr_model()
        original_flux = model._flux.copy()
        original_solver_result = model.solver_result
        original_destroyors = model.destroyors
        original_numbers = model.n_operators
        original_phases = model.phi_operators

        frequency = model._get_qubit_frequency_at_coupler_flux(0.35, qubit_idx=0)

        self.assertAlmostEqual(frequency, 6.35)
        np.testing.assert_allclose(model._flux, original_flux)
        self.assertEqual(model._generate_hamiltonian.call_count, 1)
        self.assertEqual(model._generate_hamiltonian.call_args.kwargs, {'transient': True})
        self.assertEqual(model._refresh_basic_metrics.call_count, 0)
        self.assertIs(model.solver_result, original_solver_result)
        self.assertIs(model.destroyors, original_destroyors)
        self.assertIs(model.n_operators, original_numbers)
        self.assertIs(model.phi_operators, original_phases)
        multi_module._clear_qcrfgr_probe_frequency_cache()


class MultiQubitRuntimeBaselineTests(unittest.TestCase):
    @staticmethod
    def _construct_qcrfgr_model():
        c_j = 9.8e-15
        capacitance_list = [
            70.319e-15,
            90.238e-15,
            6.304e-15 + c_j,
            78e-15,
            12.65e-15,
        ]
        junction_resistance_list = [10007.92, 10007.92 / 6]

        return QCRFGRModel(
            capacitance_list=capacitance_list,
            junc_resis_list=junction_resistance_list,
            qrcouple=[16.812e-15, 0.0159e-15],
            flux_list=[0.11, 0.11],
            trunc_ener_level=[6, 5],
        )

    def test_qcrfgr_runtime_constructor_reuses_exact_solve_template_with_copy_isolation(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)

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
        ) as set_solver_result, mock.patch('builtins.print'):
            first = self._construct_qcrfgr_model()
            second = self._construct_qcrfgr_model()

        self.assertEqual(generate_hamiltonian.call_count, 1)
        self.assertEqual(set_solver_result.call_count, 1)
        self.assertAlmostEqual(first.qubit_f01, second.qubit_f01, places=12)
        self.assertAlmostEqual(first.coupler_f01, second.coupler_f01, places=12)
        self.assertIsNot(first._Hamiltonian, second._Hamiltonian)
        self.assertIsNot(first._eigenstates[0], second._eigenstates[0])
        self.assertIsNot(first.destroyors[0], second.destroyors[0])

        second._energylevels[0] = 123.0
        self.assertNotEqual(first._energylevels[0], second._energylevels[0])

    def test_qcrfgr_runtime_constructor_reuses_exact_ematrix_template_with_copy_isolation(self):
        ParameterizedQubit._clear_ematrix_template_cache()
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)

        with mock.patch.object(
            base_module,
            'assemble_s_matrix_and_retain_nodes',
            wraps=base_module.assemble_s_matrix_and_retain_nodes,
        ) as assemble, mock.patch.object(
            base_module,
            'convert_elements_to_energy_matrices',
            wraps=base_module.convert_elements_to_energy_matrices,
        ) as convert, mock.patch('builtins.print'):
            first = self._construct_qcrfgr_model()
            second = self._construct_qcrfgr_model()

        self.assertEqual(assemble.call_count, 1)
        self.assertEqual(convert.call_count, 1)
        self.assertIsNot(first.SMatrix, second.SMatrix)
        self.assertIsNot(first.Maxwellmat['capac'], second.Maxwellmat['capac'])
        self.assertIsNot(first.Ec, second.Ec)
        self.assertIsNot(first.El, second.El)
        self.assertIsNot(first.Ejmax, second.Ejmax)

        second.SMatrix[0, 0] = -123.0
        second.Maxwellmat['capac'][0, 0] = -456.0
        second.Ec[0, 0] = -789.0
        second.El[0, 0] = -321.0
        second.Ejmax[0, 0] = -654.0
        self.assertNotEqual(first.SMatrix[0, 0], second.SMatrix[0, 0])
        self.assertNotEqual(first.Maxwellmat['capac'][0, 0], second.Maxwellmat['capac'][0, 0])
        self.assertNotEqual(first.Ec[0, 0], second.Ec[0, 0])
        self.assertNotEqual(first.El[0, 0], second.El[0, 0])
        self.assertNotEqual(first.Ejmax[0, 0], second.Ejmax[0, 0])

    def test_qcrfgr_runtime_constructor_reuses_cached_metric_state_products(self):
        QubitBase._clear_exact_solve_template_cache()
        multi_module._clear_qcrfgr_metric_state_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(multi_module._clear_qcrfgr_metric_state_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)

        with mock.patch.object(
            multi_module,
            'cal_product_state_list',
            wraps=multi_module.cal_product_state_list,
        ) as cal_product_state_list, mock.patch('builtins.print'):
            self._construct_qcrfgr_model()
            self._construct_qcrfgr_model()

        self.assertEqual(cal_product_state_list.call_count, 2)

    def test_fgf1v1_runtime_sweep_reuses_cached_metric_state_products(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_metric_state_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_metric_state_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        with mock.patch.object(
            multi_module,
            'cal_product_state_list',
            wraps=multi_module.cal_product_state_list,
        ) as cal_product_state_list, mock.patch('builtins.print'):
            model = self._construct_fgf1v1_model()
            sweep_multi_qubit_coupling_strength_vs_flux_result(
                model,
                [0.095, 0.11, 0.125, 0.14],
                method='ES',
                is_plot=False,
            )

        self.assertEqual(cal_product_state_list.call_count, 3)

    def test_fgf1v1_runtime_constructor_reuses_exact_metric_state_indices(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        multi_module._clear_fgf1v1_metric_state_index_cache()
        HamiltonianEvo._clear_hamiltonian_eigensystem_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(multi_module._clear_fgf1v1_metric_state_index_cache)
        self.addCleanup(HamiltonianEvo._clear_hamiltonian_eigensystem_cache)

        with mock.patch.object(
            HamiltonianEvo,
            'find_state_list',
            autospec=True,
            wraps=HamiltonianEvo.find_state_list,
        ) as find_state_list, mock.patch('builtins.print'):
            first = self._construct_fgf1v1_model()
            first_metrics = {
                'qubit1_f01': first.qubit1_f01,
                'qubit2_f01': first.qubit2_f01,
                'coupler_f01': first.coupler_f01,
                'qr_g': first.qr_g,
                'qq_g': first.qq_g,
                'qc_g': first.qc_g,
                'qq_geff': first.qq_geff,
            }

            self.assertEqual(find_state_list.call_count, 1)

            QubitBase._clear_exact_solve_template_cache()
            multi_module._clear_fgf1v1_basic_metric_cache()

            second = self._construct_fgf1v1_model()

        self.assertEqual(find_state_list.call_count, 1)
        for metric_name, baseline_value in first_metrics.items():
            self.assertAlmostEqual(getattr(second, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_constructor_reuses_cached_solver_eigensystem_after_exact_template_clear(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        multi_module._clear_fgf1v1_metric_state_index_cache()
        HamiltonianEvo._clear_hamiltonian_eigensystem_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(multi_module._clear_fgf1v1_metric_state_index_cache)
        self.addCleanup(HamiltonianEvo._clear_hamiltonian_eigensystem_cache)

        with mock.patch.object(
            HamiltonianEvo,
            '_solve_hamiltonian_eigensystem',
            autospec=True,
            wraps=HamiltonianEvo._solve_hamiltonian_eigensystem,
        ) as solve_hamiltonian_eigensystem, mock.patch('builtins.print'):
            first = self._construct_fgf1v1_model()
            first_metrics = {
                'qubit1_f01': first.qubit1_f01,
                'qubit2_f01': first.qubit2_f01,
                'coupler_f01': first.coupler_f01,
                'qr_g': first.qr_g,
                'qq_g': first.qq_g,
                'qc_g': first.qc_g,
                'qq_geff': first.qq_geff,
            }

            QubitBase._clear_exact_solve_template_cache()
            multi_module._clear_fgf1v1_basic_metric_cache()
            multi_module._clear_fgf1v1_metric_state_index_cache()

            second = self._construct_fgf1v1_model()

        self.assertEqual(solve_hamiltonian_eigensystem.call_count, 1)
        for metric_name, baseline_value in first_metrics.items():
            self.assertAlmostEqual(getattr(second, metric_name), baseline_value, places=12)
        self.assertIsNot(first._eigenstates[0], second._eigenstates[0])
        second._energylevels[0] = 123.0
        self.assertNotEqual(first._energylevels[0], second._energylevels[0])

    @staticmethod
    def _probe_qcrfgr_frequency_reference(model, coupler_flux: float) -> float:
        original_flux = model.get_element_matrices('flux').copy()

        try:
            model._flux[2, 2] = coupler_flux
            model.change_para(flux=model._flux)
            return float(model.qubit_f01)
        finally:
            model._flux = original_flux
            model.change_para(flux=original_flux)

    @staticmethod
    def _construct_fgf1v1_model():
        c_j = 9.8e-15
        c_q1_total = 165e-15
        c_q2_total = 165e-15
        c_qc = 23.2e-15
        c_q_ground = 5.2e-15
        c_qq = 2.1e-15
        c_coupler_total = 142e-15
        c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
        c_12_ground = c_q2_total - c_q_ground
        capacitance_list = [
            c_11_ground,
            c_12_ground,
            c_q_ground + c_j,
            c_coupler_total - 2 * c_qc + 6 * c_j,
            c_q_ground + c_j,
            c_12_ground,
            c_11_ground,
            c_qq,
            c_qc,
            c_qc,
        ]

        return FGF1V1Coupling(
            capacitance_list=capacitance_list,
            junc_resis_list=[7400, 7400 / 6, 7400],
            qrcouple=[18.34e-15, 0.02e-15],
            flux_list=[0.11, 0.11, 0.11],
            trunc_ener_level=[3, 2, 3],
            is_print=False,
        )

    def test_qcrfgr_runtime_baseline_locks_public_outputs(self):
        with mock.patch('builtins.print'):
            model = self._construct_qcrfgr_model()

        self.assertAlmostEqual(model.qubit_f01, 5.488494132366466, places=6)
        self.assertAlmostEqual(model.coupler_f01, 11.480703336567979, places=6)
        self.assertAlmostEqual(model.rq_g / (2 * np.pi * 1e6), 141.00665866929367, places=6)
        self.assertAlmostEqual(model.qc_g * 1e3, 374.0123048975068, places=6)
        np.testing.assert_allclose(
            model.get_element_matrices('flux'),
            np.array(
                [
                    [0.0, 0.11, 0.0],
                    [0.11, 0.0, 0.0],
                    [0.0, 0.0, 0.11],
                ]
            ),
        )

    def test_qcrfgr_runtime_probe_matches_reference_and_preserves_cached_state(self):
        with mock.patch('builtins.print'):
            model = self._construct_qcrfgr_model()
            reference_model = self._construct_qcrfgr_model()

        original_flux = model.get_element_matrices('flux').copy()
        original_solver_result = model.solver_result
        original_qubit_f01 = model.qubit_f01
        original_coupler_f01 = model.coupler_f01
        original_qubit_anharm = model.qubit_anharm
        original_rq_g = model.rq_g
        original_qc_g = model.qc_g
        original_destroyors = model.destroyors
        original_numbers = model.n_operators
        original_phases = model.phi_operators
        original_eigen_hamiltonian = model.eigenHamiltonian
        original_coupling_hamiltonian = model.couplingHamiltonian
        original_highorder_hamiltonian = model.highorderHamiltonian

        expected = self._probe_qcrfgr_frequency_reference(reference_model, 0.125)
        observed = get_multi_qubit_frequency_at_coupler_flux(model, 0.125, qubit_idx=0)

        self.assertAlmostEqual(observed, expected, places=12)
        np.testing.assert_allclose(model.get_element_matrices('flux'), original_flux)
        self.assertIs(model.solver_result, original_solver_result)
        self.assertAlmostEqual(model.qubit_f01, original_qubit_f01, places=12)
        self.assertAlmostEqual(model.coupler_f01, original_coupler_f01, places=12)
        self.assertAlmostEqual(model.qubit_anharm, original_qubit_anharm, places=12)
        self.assertAlmostEqual(model.rq_g, original_rq_g, places=12)
        self.assertAlmostEqual(model.qc_g, original_qc_g, places=12)
        self.assertIs(model.destroyors, original_destroyors)
        self.assertIs(model.n_operators, original_numbers)
        self.assertIs(model.phi_operators, original_phases)
        self.assertIs(model.eigenHamiltonian, original_eigen_hamiltonian)
        self.assertIs(model.couplingHamiltonian, original_coupling_hamiltonian)
        self.assertIs(model.highorderHamiltonian, original_highorder_hamiltonian)

    def test_fgf1v1_runtime_baseline_locks_public_outputs(self):
        model = self._construct_fgf1v1_model()

        self.assertAlmostEqual(model.qubit1_f01, 5.224530151681845, places=6)
        self.assertAlmostEqual(model.qubit2_f01, 5.167776586474884, places=6)
        self.assertAlmostEqual(model.coupler_f01, 9.13495616294588, places=6)
        self.assertAlmostEqual(model.qr_g / (2 * np.pi * 1e6), 131.67677448098613, places=6)
        self.assertAlmostEqual(model.qq_g * 1e3, 29.717909723577513, places=6)
        self.assertAlmostEqual(model.qc_g * 1e3, 280.60475281118494, places=6)
        self.assertAlmostEqual(model.qq_geff * 1e3, 4.231978575913272, places=6)
        np.testing.assert_allclose(
            model.get_element_matrices('flux'),
            np.array(
                [
                    [0.0, 0.11, 0.0, 0.0, 0.0],
                    [0.11, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.11, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.11],
                    [0.0, 0.0, 0.0, 0.11, 0.0],
                ]
            ),
        )

    def test_fgf1v1_overlap_basis_indices_match_overlap_state_matrix_elements(self):
        model = self._construct_fgf1v1_model()
        nlevel_key = multi_module._normalize_nlevel_cache_key(model._Nlevel)
        _, qc_overlap_states, qq_overlap_states = multi_module._get_cached_fgf1v1_metric_state_sets(
            nlevel_key
        )
        qc_overlap_indices, qq_overlap_indices = multi_module._get_cached_fgf1v1_overlap_basis_indices(
            nlevel_key
        )
        hamiltonian = model._Hamiltonian

        self.assertAlmostEqual(
            abs(hamiltonian[qc_overlap_indices[1], qc_overlap_indices[0]]) / 2 / np.pi,
            abs(qc_overlap_states[1].dag() * hamiltonian * qc_overlap_states[0]) / 2 / np.pi,
            places=12,
        )
        self.assertAlmostEqual(
            abs(hamiltonian[qc_overlap_indices[1], qc_overlap_indices[2]]) / 2 / np.pi,
            abs(qc_overlap_states[1].dag() * hamiltonian * qc_overlap_states[2]) / 2 / np.pi,
            places=12,
        )
        self.assertAlmostEqual(
            abs(hamiltonian[qq_overlap_indices[1], qq_overlap_indices[0]]) / 2 / np.pi,
            abs(qq_overlap_states[1].dag() * hamiltonian * qq_overlap_states[0]) / 2 / np.pi,
            places=12,
        )

    def test_fgf1v1_runtime_sweep_restores_derived_metrics_after_flux_round_trip(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        original_flux = model.get_element_matrices('flux').copy()
        baseline_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }

        sweep_result = sweep_multi_qubit_coupling_strength_vs_flux_result(
            model,
            [0.095, 0.11, 0.125, 0.14],
            method='ES',
            is_plot=False,
        )

        self.assertEqual(len(sweep_result.coupling_values), 4)
        self.assertGreater(np.max(np.abs(np.diff(sweep_result.coupling_values))), 0.0)
        np.testing.assert_allclose(model.get_element_matrices('flux'), original_flux)
        for metric_name, baseline_value in baseline_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_cold_flux_change_builds_scaled_phase_views_once(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_truncated_operator_views.cache_clear()
        QubitBase._build_cached_scaled_phase_terms.cache_clear()
        QubitBase._build_cached_scaled_phase_power_terms.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_truncated_operator_views.cache_clear)
        self.addCleanup(QubitBase._build_cached_scaled_phase_terms.cache_clear)
        self.addCleanup(QubitBase._build_cached_scaled_phase_power_terms.cache_clear)

        model = self._construct_fgf1v1_model()
        replay_flux = model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095

        with mock.patch.object(
            QubitBase,
            '_build_cached_scaled_phase_terms',
            wraps=QubitBase._build_cached_scaled_phase_terms,
        ) as scaled_phase_terms:
            model.change_para(flux=replay_flux)

        self.assertEqual(scaled_phase_terms.call_count, 1)
        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)

    def test_fgf1v1_runtime_diagonal_hamiltonian_terms_cache_reuses_exact_inputs_across_cold_rebuilds(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear()
        QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear()
        QubitBase._build_cached_diagonal_hamiltonian_terms.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear)
        self.addCleanup(QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear)
        self.addCleanup(QubitBase._build_cached_diagonal_hamiltonian_terms.cache_clear)

        first = self._construct_fgf1v1_model()
        replay_flux = first.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        first.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': first.qubit1_f01,
            'qubit2_f01': first.qubit2_f01,
            'coupler_f01': first.coupler_f01,
            'qr_g': first.qr_g,
            'qq_g': first.qq_g,
            'qc_g': first.qc_g,
            'qq_geff': first.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()
        QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear()

        second = self._construct_fgf1v1_model()
        second.change_para(flux=replay_flux)

        cache_info = QubitBase._build_cached_diagonal_hamiltonian_terms.cache_info()
        self.assertEqual(cache_info.misses, 2)
        self.assertEqual(cache_info.hits, 2)
        np.testing.assert_allclose(second.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(second, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_pair_charge_bundle_cache_reuses_exact_inputs_across_cold_rebuilds(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear()
        QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear()
        QubitBase._build_cached_pair_number_bundle.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear)
        self.addCleanup(QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear)
        self.addCleanup(QubitBase._build_cached_pair_number_bundle.cache_clear)

        first = self._construct_fgf1v1_model()
        replay_flux = first.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        first.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': first.qubit1_f01,
            'qubit2_f01': first.qubit2_f01,
            'coupler_f01': first.coupler_f01,
            'qr_g': first.qr_g,
            'qq_g': first.qq_g,
            'qc_g': first.qc_g,
            'qq_geff': first.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()
        QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear()

        second = self._construct_fgf1v1_model()
        second.change_para(flux=replay_flux)

        cache_info = QubitBase._build_cached_pair_number_bundle.cache_info()
        self.assertEqual(cache_info.misses, 2)
        self.assertEqual(cache_info.hits, 2)
        np.testing.assert_allclose(second.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(second, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_charge_only_hamiltonian_bundle_cache_reuses_exact_inputs_across_cold_rebuilds(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear()
        QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear)
        self.addCleanup(QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear)

        first = self._construct_fgf1v1_model()
        replay_flux = first.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        first.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': first.qubit1_f01,
            'qubit2_f01': first.qubit2_f01,
            'coupler_f01': first.coupler_f01,
            'qr_g': first.qr_g,
            'qq_g': first.qq_g,
            'qc_g': first.qc_g,
            'qq_geff': first.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()

        second = self._construct_fgf1v1_model()
        second.change_para(flux=replay_flux)

        cache_info = QubitBase._build_cached_charge_only_hamiltonian_terms.cache_info()
        self.assertEqual(cache_info.misses, 2)
        self.assertEqual(cache_info.hits, 4)
        np.testing.assert_allclose(second.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(second, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_truncated_charge_only_hamiltonian_cache_reuses_exact_inputs_across_cold_rebuilds(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear)

        first = self._construct_fgf1v1_model()
        replay_flux = first.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        first.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': first.qubit1_f01,
            'qubit2_f01': first.qubit2_f01,
            'coupler_f01': first.coupler_f01,
            'qr_g': first.qr_g,
            'qq_g': first.qq_g,
            'qc_g': first.qc_g,
            'qq_geff': first.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()

        second = self._construct_fgf1v1_model()
        second.change_para(flux=replay_flux)

        cache_info = QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_info()
        self.assertEqual(cache_info.misses, 2)
        self.assertEqual(cache_info.hits, 2)
        np.testing.assert_allclose(second.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(second, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_warmed_charge_only_hamiltonian_bundle_avoids_live_qobj_add(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear()
        QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear()
        QubitBase._build_cached_pair_number_bundle.cache_clear()
        QubitBase._build_cached_pair_number_contribution.cache_clear()
        QubitBase._build_cached_pair_number_term.cache_clear()
        QubitBase._build_cached_pair_operator_products.cache_clear()
        QubitBase._build_cached_diagonal_hamiltonian_terms.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_charge_only_hamiltonian_terms.cache_clear)
        self.addCleanup(QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear)
        self.addCleanup(QubitBase._build_cached_pair_number_bundle.cache_clear)
        self.addCleanup(QubitBase._build_cached_pair_number_contribution.cache_clear)
        self.addCleanup(QubitBase._build_cached_pair_number_term.cache_clear)
        self.addCleanup(QubitBase._build_cached_pair_operator_products.cache_clear)
        self.addCleanup(QubitBase._build_cached_diagonal_hamiltonian_terms.cache_clear)

        seeded_model = self._construct_fgf1v1_model()
        replay_flux = seeded_model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        seeded_model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': seeded_model.qubit1_f01,
            'qubit2_f01': seeded_model.qubit2_f01,
            'coupler_f01': seeded_model.coupler_f01,
            'qr_g': seeded_model.qr_g,
            'qq_g': seeded_model.qq_g,
            'qc_g': seeded_model.qc_g,
            'qq_geff': seeded_model.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()

        model = self._construct_fgf1v1_model()
        with mock.patch.object(Qobj, '__add__', autospec=True, wraps=Qobj.__add__) as qobj_add:
            model.change_para(flux=replay_flux)

        self.assertEqual(qobj_add.call_count, 0)
        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_warmed_truncated_charge_only_hamiltonian_cache_avoids_live_truncation(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear()
        QubitBase._build_cached_truncated_operator_views.cache_clear()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)
        self.addCleanup(QubitBase._build_cached_truncated_charge_only_hamiltonian.cache_clear)
        self.addCleanup(QubitBase._build_cached_truncated_operator_views.cache_clear)

        seeded_model = self._construct_fgf1v1_model()
        replay_flux = seeded_model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        seeded_model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': seeded_model.qubit1_f01,
            'qubit2_f01': seeded_model.qubit2_f01,
            'coupler_f01': seeded_model.coupler_f01,
            'qr_g': seeded_model.qr_g,
            'qq_g': seeded_model.qq_g,
            'qc_g': seeded_model.qc_g,
            'qq_geff': seeded_model.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()

        model = self._construct_fgf1v1_model()
        with mock.patch.object(base_module, 'truncate_hilbert_space', wraps=base_module.truncate_hilbert_space) as truncate:
            model.change_para(flux=replay_flux)

        self.assertEqual(truncate.call_count, 0)
        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_cold_metric_refresh_avoids_public_helper_roundtrips(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        reference_model = self._construct_fgf1v1_model()
        replay_flux = reference_model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        reference_model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': reference_model.qubit1_f01,
            'qubit2_f01': reference_model.qubit2_f01,
            'coupler_f01': reference_model.coupler_f01,
            'qr_g': reference_model.qr_g,
            'qq_g': reference_model.qq_g,
            'qc_g': reference_model.qc_g,
            'qq_geff': reference_model.qq_geff,
        }

        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()

        model = self._construct_fgf1v1_model()
        with mock.patch.object(
            model,
            'find_state',
            side_effect=AssertionError('cold metric refresh should batch lookup without per-state find_state calls'),
        ), mock.patch.object(
            model,
            'get_readout_couple',
            side_effect=AssertionError('cold metric refresh should skip public readout helper round-trips'),
        ), mock.patch.object(
            model,
            'get_qq_dcouple',
            side_effect=AssertionError('cold metric refresh should skip public qq helper round-trips'),
        ), mock.patch.object(
            model,
            'get_qc_couple',
            side_effect=AssertionError('cold metric refresh should skip public qc helper round-trips'),
        ), mock.patch.object(
            model,
            'get_qq_ecouple',
            side_effect=AssertionError('cold metric refresh should skip public qq-eff helper round-trips'),
        ):
            model.change_para(flux=replay_flux)

        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_exact_input_replay_reuses_cached_metric_bundle(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        replay_flux = model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)
        baseline_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }

        with mock.patch.object(
            model,
            '_generate_hamiltonian',
            side_effect=AssertionError('exact-input replay should reuse the cached solve template'),
        ), mock.patch.object(
            model,
            'find_state',
            side_effect=AssertionError('cached metric replay should skip state lookup'),
        ), mock.patch.object(
            model,
            'get_readout_couple',
            side_effect=AssertionError('cached metric replay should skip readout recomputation'),
        ), mock.patch.object(
            model,
            'get_qq_dcouple',
            side_effect=AssertionError('cached metric replay should skip direct-coupling recomputation'),
        ), mock.patch.object(
            model,
            'get_qc_couple',
            side_effect=AssertionError('cached metric replay should skip qc overlap recomputation'),
        ), mock.patch.object(
            model,
            'get_qq_ecouple',
            side_effect=AssertionError('cached metric replay should skip effective-coupling recomputation'),
        ):
            model.change_para(flux=replay_flux)

        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in baseline_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_metric_bundle_caches_store_typed_payloads(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        replay_flux = model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)

        process_cache_key = multi_module._make_fgf1v1_basic_metric_cache_key(model)
        self.assertIsNotNone(process_cache_key)
        process_payload = multi_module._FGF1V1_BASIC_METRIC_CACHE[process_cache_key]
        self.assertIsInstance(process_payload, multi_module.FGF1V1BasicMetricBundle)
        self.assertIs(
            multi_module._get_cached_fgf1v1_basic_metrics(process_cache_key),
            process_payload,
        )

        exact_solve_cache_key = getattr(model, '_exact_solve_template_cache_key', None)
        self.assertIsNotNone(exact_solve_cache_key)
        instance_cache = getattr(model, '_fgf1v1_instance_basic_metric_cache', None)
        self.assertIsNotNone(instance_cache)
        instance_payload = instance_cache[exact_solve_cache_key]
        self.assertIsInstance(instance_payload, multi_module.FGF1V1BasicMetricBundle)
        self.assertIs(
            multi_module._get_cached_instance_fgf1v1_basic_metrics(model, exact_solve_cache_key),
            instance_payload,
        )
        self.assertEqual(instance_payload, process_payload)

    def test_fgf1v1_runtime_same_instance_revisit_uses_owned_exact_template_cache(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        original_flux = model.get_element_matrices('flux').copy()
        original_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }
        replay_flux = original_flux.copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)

        with mock.patch.object(
            QubitBase,
            '_get_cached_exact_solve_template',
            side_effect=AssertionError('same-instance baseline revisit should bypass the process cache'),
        ), mock.patch.object(
            QubitBase,
            '_restore_exact_solve_template',
            side_effect=AssertionError('same-instance baseline revisit should reuse the owned template'),
        ):
            model.change_para(flux=original_flux)

        np.testing.assert_allclose(model.get_element_matrices('flux'), original_flux)
        for metric_name, baseline_value in original_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_process_replay_restore_materializes_auxiliary_state_on_demand(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        seeded_model = self._construct_fgf1v1_model()
        replay_flux = seeded_model.get_element_matrices('flux').copy()
        replay_flux[2, 2] = 0.095
        seeded_model.change_para(flux=replay_flux)
        seeded_destroyor = seeded_model.destroyors[0]

        model = self._construct_fgf1v1_model()
        model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }

        self.assertIsNone(model._Hamiltonian)
        self.assertIsNotNone(model._pending_exact_core_template)
        self.assertIsNone(model._solver_result)
        self.assertIsNotNone(model._pending_exact_auxiliary_template)

        replay_levels = np.array(model.get_energylevel(), copy=True)

        self.assertGreaterEqual(len(replay_levels), 2)
        self.assertIsNotNone(model._Hamiltonian)
        self.assertIsNone(model._pending_exact_core_template)
        self.assertIsNone(model._solver_result)
        self.assertIsNotNone(model._pending_exact_auxiliary_template)
        self.assertIsNot(model._Hamiltonian, seeded_model._Hamiltonian)
        self.assertIsNot(model._eigenstates[0], seeded_model._eigenstates[0])

        solver_result = model.solver_result

        self.assertIsInstance(solver_result, SpectrumResult)
        self.assertIsNone(model._pending_exact_auxiliary_template)
        self.assertIsNotNone(model.destroyors)
        self.assertIsNot(model.destroyors[0], seeded_destroyor)
        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_same_instance_revisit_reuses_exact_key_prefix(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        original_flux = model.get_element_matrices('flux').copy()
        original_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }
        replay_flux = original_flux.copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)

        observed_labels = []
        original_get_array_cache_key = QubitBase._get_array_cache_key

        def wrapped(values, *, as_int=False):
            if values is model.Ej:
                observed_labels.append('Ej')
            elif values is model.Ec:
                observed_labels.append('Ec')
            elif values is model.El:
                observed_labels.append('El')
            elif values is model._charges:
                observed_labels.append('charges')
            elif values is model._Nlevel:
                observed_labels.append('Nlevel')
            else:
                observed_labels.append('other')
            return original_get_array_cache_key(values, as_int=as_int)

        with mock.patch.object(QubitBase, '_get_array_cache_key', new=staticmethod(wrapped)):
            model.change_para(flux=original_flux)

        self.assertEqual(observed_labels, ['Ej'])
        np.testing.assert_allclose(model.get_element_matrices('flux'), original_flux)
        for metric_name, baseline_value in original_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_same_instance_revisit_reuses_flux_only_replay_preparation(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        original_flux = model.get_element_matrices('flux').copy()
        replay_flux = original_flux.copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }
        model.change_para(flux=original_flux)

        with mock.patch.object(
            base_module,
            'project_transformed_flux',
            side_effect=AssertionError('same-instance revisit should reuse cached transformed flux'),
        ), mock.patch.object(
            model,
            '_Ejphi',
            side_effect=AssertionError('same-instance revisit should reuse cached Ej preparation'),
        ):
            model.change_para(flux=replay_flux)

        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_same_instance_revisit_reuses_instance_metric_bundle_cache(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        original_flux = model.get_element_matrices('flux').copy()
        replay_flux = original_flux.copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }
        model.change_para(flux=original_flux)

        with mock.patch.object(
            multi_module,
            '_make_fgf1v1_basic_metric_cache_key',
            side_effect=AssertionError('same-instance revisit should reuse the direct metric bundle cache'),
        ):
            model.change_para(flux=replay_flux)

        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_same_instance_revisit_reuses_prepared_exact_template(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        model = self._construct_fgf1v1_model()
        original_flux = model.get_element_matrices('flux').copy()
        replay_flux = original_flux.copy()
        replay_flux[2, 2] = 0.095

        model.change_para(flux=replay_flux)
        replay_metrics = {
            'qubit1_f01': model.qubit1_f01,
            'qubit2_f01': model.qubit2_f01,
            'coupler_f01': model.coupler_f01,
            'qr_g': model.qr_g,
            'qq_g': model.qq_g,
            'qc_g': model.qc_g,
            'qq_geff': model.qq_geff,
        }
        model.change_para(flux=original_flux)

        with mock.patch.object(
            QubitBase,
            '_get_exact_solve_template_cache_key',
            side_effect=AssertionError('same-instance revisit should reuse the prepared exact replay key'),
        ), mock.patch.object(
            QubitBase,
            '_get_instance_exact_solve_template',
            side_effect=AssertionError('same-instance revisit should reuse the prepared exact template directly'),
        ):
            model.change_para(flux=replay_flux)

        np.testing.assert_allclose(model.get_element_matrices('flux'), replay_flux)
        for metric_name, baseline_value in replay_metrics.items():
            self.assertAlmostEqual(getattr(model, metric_name), baseline_value, places=12)

    def test_fgf1v1_runtime_warmed_constructor_reuse_preserves_public_sweep_trace(self):
        QubitBase._clear_exact_solve_template_cache()
        ParameterizedQubit._clear_ematrix_template_cache()
        multi_module._clear_fgf1v1_basic_metric_cache()
        self.addCleanup(QubitBase._clear_exact_solve_template_cache)
        self.addCleanup(ParameterizedQubit._clear_ematrix_template_cache)
        self.addCleanup(multi_module._clear_fgf1v1_basic_metric_cache)

        cold_model = self._construct_fgf1v1_model()
        expected = sweep_multi_qubit_coupling_strength_vs_flux_result(
            cold_model,
            [0.095, 0.11, 0.125, 0.14],
            method='ES',
            is_plot=False,
        )

        warmed_model = self._construct_fgf1v1_model()
        observed = sweep_multi_qubit_coupling_strength_vs_flux_result(
            warmed_model,
            [0.095, 0.11, 0.125, 0.14],
            method='ES',
            is_plot=False,
        )

        np.testing.assert_allclose(observed.coupling_values, expected.coupling_values)


if __name__ == '__main__':
    unittest.main()
