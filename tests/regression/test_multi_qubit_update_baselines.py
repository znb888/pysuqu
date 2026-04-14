import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from qutip import basis, tensor

from pysuqu.qubit import base as base_module
from pysuqu.qubit import multi as multi_module
from pysuqu.qubit.analysis import get_multi_qubit_frequency_at_coupler_flux
from pysuqu.qubit.base import ParameterizedQubit, QubitBase
from pysuqu.qubit.multi import FGF1V1Coupling, QCRFGRModel
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


if __name__ == '__main__':
    unittest.main()
