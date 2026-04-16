import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from qutip import basis, tensor

from pysuqu.qubit import base as qubit_base_module
from pysuqu.qubit.base import AbstractQubit, ParameterizedQubit
from pysuqu.qubit import circuit
from pysuqu.qubit.multi import FGF1V1Coupling, FGF2V7Coupling, QCRFGRModel
from pysuqu.qubit.solver import HamiltonianEvo
from pysuqu.qubit.types import SpectrumResult


class ChangeParaContractTests(unittest.TestCase):
    def test_change_para_accepts_keyword_updates(self):
        qubit = ParameterizedQubit.__new__(ParameterizedQubit)
        qubit._flux = np.array([[0.0]])
        qubit._recalculate_hamiltonian = mock.Mock()

        qubit.change_para(flux=[[0.25]])

        np.testing.assert_allclose(qubit._flux, np.array([[0.25]]))
        qubit._recalculate_hamiltonian.assert_called_once()

    def test_change_para_rejects_unknown_parameter_before_state_change_or_recompute(self):
        qubit = ParameterizedQubit.__new__(ParameterizedQubit)
        qubit._flux = np.array([[0.0]])
        qubit._recalculate_hamiltonian = mock.Mock()

        with self.assertRaisesRegex(ValueError, 'Unknown parameter name: typo_flux'):
            qubit.change_para(flux=[[0.25]], typo_flux=1.0)

        np.testing.assert_allclose(qubit._flux, np.array([[0.0]]))
        qubit._recalculate_hamiltonian.assert_not_called()


class SpectrumResultContainerTests(unittest.TestCase):
    def test_hamiltonian_evo_exposes_solver_result_container(self):
        class FakeHamiltonian:
            dims = [[2], [2]]

            def eigenstates(self):
                return np.array([0.0, 1.0]), ['g', 'e']

        evo = HamiltonianEvo(FakeHamiltonian())

        self.assertIsInstance(evo.solver_result, SpectrumResult)
        self.assertIs(evo.solver_result.hamiltonian, evo._Hamiltonian)
        np.testing.assert_allclose(evo.solver_result.eigenvalues, np.array([0.0, 1.0]))
        self.assertEqual(evo.solver_result.eigenstates, ['g', 'e'])
        self.assertIsNone(evo.solver_result.destroy_operators)
        self.assertIsNone(evo.solver_result.number_operators)
        self.assertIsNone(evo.solver_result.phase_operators)

    def test_change_hamiltonian_refreshes_solver_result_and_operator_refs(self):
        class FakeHamiltonian:
            def __init__(self, dims, eigenvalues, eigenstates):
                self.dims = dims
                self._eigenvalues = np.array(eigenvalues)
                self._eigenstates = eigenstates

            def eigenstates(self):
                return self._eigenvalues, self._eigenstates

        evo = HamiltonianEvo(FakeHamiltonian([[1], [1]], [0.0], ['g']))
        evo.destroyors = ['a']
        evo.n_operators = ['n']
        evo.phi_operators = ['phi']

        evo.change_hamiltonian(FakeHamiltonian([[3], [3]], [0.0, 2.0, 4.0], ['g', 'e', 'f']))

        self.assertEqual(evo.Nlevel, [3])
        np.testing.assert_allclose(evo.solver_result.eigenvalues, np.array([0.0, 2.0, 4.0]))
        self.assertEqual(evo.solver_result.eigenstates, ['g', 'e', 'f'])
        self.assertEqual(evo.solver_result.destroy_operators, ['a'])
        self.assertEqual(evo.solver_result.number_operators, ['n'])
        self.assertEqual(evo.solver_result.phase_operators, ['phi'])

    def test_hamiltonian_evo_reuses_identical_cached_eigensystem_with_copy_isolation(self):
        HamiltonianEvo._clear_hamiltonian_eigensystem_cache()
        self.addCleanup(HamiltonianEvo._clear_hamiltonian_eigensystem_cache)

        class FakeHamiltonian:
            dims = [[2], [2]]
            eigenstate_calls = 0

            def full(self):
                return np.array([[0.0, 0.0], [0.0, 1.0]])

            def eigenstates(self):
                type(self).eigenstate_calls += 1
                return np.array([0.0, 1.0]), [['g'], ['e']]

        first = HamiltonianEvo(FakeHamiltonian())
        second = HamiltonianEvo(FakeHamiltonian())

        self.assertEqual(FakeHamiltonian.eigenstate_calls, 1)
        np.testing.assert_allclose(second.solver_result.eigenvalues, np.array([0.0, 1.0]))
        self.assertEqual(second.solver_result.eigenstates, [['g'], ['e']])
        second._energylevels[0] = 123.0
        second._eigenstates[0].append('cache-hit')
        self.assertNotEqual(first._energylevels[0], second._energylevels[0])
        self.assertNotEqual(first._eigenstates[0], second._eigenstates[0])

    def test_find_state_and_find_state_list_use_internal_eigensystem_on_hot_path(self):
        ground = tensor(basis(2, 0), basis(2, 0))
        first_excited = tensor(basis(2, 1), basis(2, 0))
        second_excited = tensor(basis(2, 0), basis(2, 1))
        doubly_excited = tensor(basis(2, 1), basis(2, 1))

        class FakeHamiltonian:
            dims = [[2, 2], [2, 2]]

            def eigenstates(self):
                return np.array([0.0, 1.0, 2.0, 3.0]), [
                    ground,
                    first_excited,
                    second_excited,
                    doubly_excited,
                ]

        evo = HamiltonianEvo(FakeHamiltonian())
        evo.get_eigenstate = mock.Mock(side_effect=AssertionError('hot path should reuse _eigenstates directly'))

        self.assertEqual(evo.find_state([1, 0]), 1)
        self.assertEqual(evo.find_state_list([[1, 0], [0, 1]]), [1, 2])
        evo.get_eigenstate.assert_not_called()

    def test_find_state_list_batches_lookup_without_delegating_to_find_state(self):
        ground = tensor(basis(2, 0), basis(2, 0))
        first_excited = tensor(basis(2, 1), basis(2, 0))
        second_excited = tensor(basis(2, 0), basis(2, 1))
        doubly_excited = tensor(basis(2, 1), basis(2, 1))

        class FakeHamiltonian:
            dims = [[2, 2], [2, 2]]

            def eigenstates(self):
                return np.array([0.0, 1.0, 2.0, 3.0]), [
                    ground,
                    first_excited,
                    second_excited,
                    doubly_excited,
                ]

        evo = HamiltonianEvo(FakeHamiltonian())
        evo.find_state = mock.Mock(
            side_effect=AssertionError('find_state_list should batch lookup without delegating to find_state')
        )

        self.assertEqual(evo.find_state_list([[1, 0], [0, 1]]), [1, 2])
        evo.find_state.assert_not_called()



class FakeHamiltonian:
    def __init__(self, dims=None, eigenvalues=None, eigenstates=None):
        self.dims = dims or [[1], [1]]
        self._eigenvalues = np.array(eigenvalues if eigenvalues is not None else [0.0])
        self._eigenstates = eigenstates if eigenstates is not None else ['g']

    def eigenstates(self):
        return self._eigenvalues, self._eigenstates


class ParameterizedQubitRebuildTests(unittest.TestCase):
    def make_qubit(self):
        qubit = ParameterizedQubit.__new__(ParameterizedQubit)
        qubit._flux = np.array([[0.0]])
        qubit._junc_ratio = np.array([[1.0]])
        qubit.SMatrix_retainNodes = [0]
        qubit._ParameterizedQubit__struct = [1]
        qubit._ParameterizedQubit__nodes = 1
        qubit._ParameterizedQubit__capac = np.array([[1.0]])
        qubit._ParameterizedQubit__resis = np.array([[2.0]])
        qubit._ParameterizedQubit__induc = np.array([[3.0]])
        qubit.Ec = np.array([[10.0]])
        qubit.El = np.array([[20.0]])
        qubit.Ejmax = np.array([[30.0]])
        qubit._hamiltonian = None
        qubit._numQubits = 1
        return qubit

    def make_circuit_rebuild_qubit(self):
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
        qubit._ParameterizedQubit__nodes = 4
        qubit._ParameterizedQubit__capac = np.array(
            [
                [2.0, -0.3, -0.2, -0.1],
                [-0.3, 1.9, -0.25, -0.15],
                [-0.2, -0.25, 1.8, -0.2],
                [-0.1, -0.15, -0.2, 1.7],
            ]
        )
        qubit._ParameterizedQubit__resis = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [11.0, 14.0, 15.0, 16.0],
                [12.0, 15.0, 18.0, 19.0],
                [13.0, 16.0, 19.0, 20.0],
            ]
        )
        qubit._ParameterizedQubit__induc = np.array(
            [
                [40.0, 90.0, 100.0, 110.0],
                [90.0, 45.0, 95.0, 105.0],
                [100.0, 95.0, 50.0, 115.0],
                [110.0, 105.0, 115.0, 55.0],
            ]
        )
        qubit.Ec = np.eye(3)
        qubit.El = np.eye(3)
        qubit.Ejmax = np.eye(3)
        qubit._hamiltonian = None
        qubit._numQubits = 3
        qubit._Nlevel = np.array([1, 1, 1])
        qubit.destroyors = ['a']
        qubit.n_operators = ['n']
        qubit.phi_operators = ['phi']
        qubit.change_hamiltonian = HamiltonianEvo.change_hamiltonian.__get__(qubit, ParameterizedQubit)
        qubit._generate_hamiltonian = mock.Mock(
            return_value=FakeHamiltonian(
                dims=[[2, 2, 2], [2, 2, 2]],
                eigenvalues=[0.0, 1.0],
                eigenstates=['g', 'e'],
            )
        )
        return qubit

    def test_element_updates_rebuild_circuit_matrices(self):
        qubit = self.make_qubit()
        qubit._Ejphi = mock.Mock(return_value=np.array([[300.0]]))
        qubit._generate_hamiltonian = mock.Mock(return_value=FakeHamiltonian())
        qubit.change_hamiltonian = mock.Mock()
        qubit._generate_Ematrix = mock.Mock(
            return_value=(
                np.array([[101.0]]),
                np.array([[202.0]]),
                np.array([[303.0]]),
            )
        )

        qubit.change_para(capac=[[9.0]], resis=[[8.0]], induc=[[7.0]])

        self.assertIsInstance(qubit._ParameterizedQubit__capac, np.ndarray)
        self.assertIsInstance(qubit._ParameterizedQubit__resis, np.ndarray)
        self.assertIsInstance(qubit._ParameterizedQubit__induc, np.ndarray)
        np.testing.assert_allclose(qubit.Ec, np.array([[101.0]]))
        np.testing.assert_allclose(qubit.El, np.array([[202.0]]))
        np.testing.assert_allclose(qubit.Ejmax, np.array([[303.0]]))
        np.testing.assert_allclose(qubit.Ej, np.array([[300.0]]))
        qubit._generate_Ematrix.assert_called_once()
        qubit._generate_hamiltonian.assert_called_once_with(qubit.Ec, qubit.El, qubit.Ej)
        qubit.change_hamiltonian.assert_called_once()
        self.assertIs(qubit.change_hamiltonian.call_args.args[0], qubit._hamiltonian)

    def test_flux_only_update_skips_circuit_rebuild(self):
        qubit = self.make_qubit()
        qubit._Ejphi = mock.Mock(return_value=np.array([[55.0]]))
        qubit._generate_hamiltonian = mock.Mock(return_value=FakeHamiltonian(dims=[[2], [2]], eigenvalues=[0.0, 1.0], eigenstates=['g', 'e']))
        qubit.change_hamiltonian = HamiltonianEvo.change_hamiltonian.__get__(qubit, ParameterizedQubit)
        qubit._generate_Ematrix = mock.Mock()
        qubit.destroyors = ['a']
        qubit.n_operators = ['n']
        qubit.phi_operators = ['phi']

        qubit.change_para(flux=[[0.125]])

        qubit._generate_Ematrix.assert_not_called()
        np.testing.assert_allclose(qubit._flux, np.array([[0.125]]))
        np.testing.assert_allclose(qubit.Ej, np.array([[55.0]]))
        self.assertIsInstance(qubit.solver_result, SpectrumResult)
        self.assertEqual(qubit.solver_result.destroy_operators, ['a'])
        self.assertEqual(qubit.solver_result.number_operators, ['n'])
        self.assertEqual(qubit.solver_result.phase_operators, ['phi'])

    def test_exact_template_restore_defers_auxiliary_state_until_access(self):
        qubit = self.make_qubit()
        restored_hamiltonian = FakeHamiltonian(dims=[[2], [2]], eigenvalues=[0.0, 1.0], eigenstates=['g', 'e'])

        qubit._restore_owned_exact_solve_template(
            qubit_base_module.OwnedExactSolveTemplate(
                core=qubit_base_module.ExactCoreState(
                    hamiltonian=restored_hamiltonian,
                    energylevels=np.array([0.0, 1.0]),
                    eigenstates=['g', 'e'],
                ),
                pending_core=None,
                auxiliary=qubit_base_module.ExactAuxiliaryState(
                    destroyors=None,
                    number_operators=None,
                    phase_operators=None,
                    eigen_hamiltonian=None,
                    coupling_hamiltonian=None,
                    highorder_hamiltonian=None,
                ),
                solver_result=None,
                pending_auxiliary=qubit_base_module.ExactAuxiliaryState(
                    destroyors=['pending-destroyor'],
                    number_operators=['pending-number'],
                    phase_operators=['pending-phase'],
                    eigen_hamiltonian='pending-eigen',
                    coupling_hamiltonian='pending-coupling',
                    highorder_hamiltonian='pending-highorder',
                ),
                nlevel=[2],
                qubits_num=1,
            )
        )

        self.assertIsNone(qubit._solver_result)
        self.assertIsNotNone(qubit._pending_exact_auxiliary_template)
        self.assertEqual(qubit.destroyors, ['pending-destroyor'])
        self.assertEqual(qubit.n_operators, ['pending-number'])
        self.assertEqual(qubit.phi_operators, ['pending-phase'])
        self.assertEqual(qubit.eigenHamiltonian, 'pending-eigen')
        self.assertEqual(qubit.couplingHamiltonian, 'pending-coupling')
        self.assertEqual(qubit.highorderHamiltonian, 'pending-highorder')
        self.assertIsNone(qubit._pending_exact_auxiliary_template)
        self.assertIsInstance(qubit.solver_result, SpectrumResult)
        self.assertIs(qubit.solver_result.hamiltonian, restored_hamiltonian)
        np.testing.assert_allclose(qubit.solver_result.eigenvalues, np.array([0.0, 1.0]))
        self.assertEqual(qubit.solver_result.eigenstates, ['g', 'e'])
        self.assertEqual(qubit.solver_result.destroy_operators, ['pending-destroyor'])
        self.assertEqual(qubit.solver_result.number_operators, ['pending-number'])
        self.assertEqual(qubit.solver_result.phase_operators, ['pending-phase'])

    def test_exact_template_restore_defers_core_state_until_eigensystem_access(self):
        qubit = self.make_qubit()
        restored_hamiltonian = FakeHamiltonian(dims=[[2], [2]], eigenvalues=[0.0, 1.0], eigenstates=['g', 'e'])

        qubit._restore_owned_exact_solve_template(
            qubit_base_module.OwnedExactSolveTemplate(
                core=qubit_base_module.ExactCoreState(
                    hamiltonian=None,
                    energylevels=None,
                    eigenstates=None,
                ),
                pending_core=qubit_base_module.ExactCoreState(
                    hamiltonian=restored_hamiltonian,
                    energylevels=np.array([0.0, 1.0]),
                    eigenstates=['g', 'e'],
                ),
                auxiliary=qubit_base_module.ExactAuxiliaryState(
                    destroyors=None,
                    number_operators=None,
                    phase_operators=None,
                    eigen_hamiltonian=None,
                    coupling_hamiltonian=None,
                    highorder_hamiltonian=None,
                ),
                solver_result=None,
                pending_auxiliary=qubit_base_module.ExactAuxiliaryState(
                    destroyors=['pending-destroyor'],
                    number_operators=['pending-number'],
                    phase_operators=['pending-phase'],
                    eigen_hamiltonian='pending-eigen',
                    coupling_hamiltonian='pending-coupling',
                    highorder_hamiltonian='pending-highorder',
                ),
                nlevel=[2],
                qubits_num=1,
            )
        )

        self.assertIsNone(qubit._Hamiltonian)
        self.assertIsNone(qubit._energylevels)
        self.assertIsNone(qubit._eigenstates)
        self.assertIsNotNone(qubit._pending_exact_core_template)
        self.assertIsNotNone(qubit._pending_exact_auxiliary_template)

        self.assertEqual(qubit.get_energylevel(1), 1.0)

        self.assertIsNotNone(qubit._Hamiltonian)
        self.assertIsNotNone(qubit._energylevels)
        self.assertIsNotNone(qubit._eigenstates)
        self.assertIsNone(qubit._pending_exact_core_template)
        self.assertIsNone(qubit._solver_result)
        self.assertIsNotNone(qubit._pending_exact_auxiliary_template)

        solver_result = qubit.solver_result

        self.assertIsInstance(solver_result, SpectrumResult)
        self.assertIsNotNone(qubit._solver_result)
        self.assertIsNone(qubit._pending_exact_auxiliary_template)
        self.assertIsNot(solver_result.hamiltonian, restored_hamiltonian)
        np.testing.assert_allclose(solver_result.eigenvalues, np.array([0.0, 1.0]))
        self.assertEqual(solver_result.eigenstates, ['g', 'e'])
        self.assertEqual(solver_result.destroy_operators, ['pending-destroyor'])
        self.assertEqual(solver_result.number_operators, ['pending-number'])
        self.assertEqual(solver_result.phase_operators, ['pending-phase'])

    def test_multi_node_flux_only_update_reuses_transformed_junction_ratio_projection(self):
        qubit = self.make_circuit_rebuild_qubit()
        _, qubit.SMatrix_retainNodes = circuit.assemble_s_matrix_and_retain_nodes(
            qubit._ParameterizedQubit__struct
        )
        qubit._update_transformed_vars()
        baseline_ratio_transformed = np.array(qubit._junc_ratio_transformed, copy=True)
        updated_flux = np.array(qubit._flux, copy=True)
        updated_flux[0, 0] = 0.21

        with mock.patch.object(
            qubit_base_module,
            'project_transformed_flux',
            wraps=qubit_base_module.project_transformed_flux,
        ) as project_flux, mock.patch.object(
            qubit_base_module,
            'project_transformed_junction_ratio',
            wraps=qubit_base_module.project_transformed_junction_ratio,
        ) as project_ratio:
            qubit.change_para(flux=updated_flux.tolist())

        expected_flux_transformed = circuit.project_transformed_flux(
            updated_flux,
            qubit._ParameterizedQubit__struct,
            qubit.SMatrix_retainNodes,
        )
        self.assertEqual(project_flux.call_count, 1)
        self.assertEqual(project_ratio.call_count, 0)
        np.testing.assert_allclose(qubit._flux_transformed, expected_flux_transformed)
        np.testing.assert_allclose(qubit._junc_ratio_transformed, baseline_ratio_transformed)

    def test_multi_node_junc_ratio_only_update_reuses_transformed_flux_projection(self):
        qubit = self.make_circuit_rebuild_qubit()
        _, qubit.SMatrix_retainNodes = circuit.assemble_s_matrix_and_retain_nodes(
            qubit._ParameterizedQubit__struct
        )
        qubit._update_transformed_vars()
        baseline_flux_transformed = np.array(qubit._flux_transformed, copy=True)
        updated_ratio = np.array(qubit._junc_ratio, copy=True)
        updated_ratio[0, 0] = 1.41

        with mock.patch.object(
            qubit_base_module,
            'project_transformed_flux',
            wraps=qubit_base_module.project_transformed_flux,
        ) as project_flux, mock.patch.object(
            qubit_base_module,
            'project_transformed_junction_ratio',
            wraps=qubit_base_module.project_transformed_junction_ratio,
        ) as project_ratio:
            qubit.change_para(junc_ratio=updated_ratio.tolist())

        expected_ratio_transformed = circuit.project_transformed_junction_ratio(
            updated_ratio,
            qubit._ParameterizedQubit__struct,
            qubit.SMatrix_retainNodes,
        )
        self.assertEqual(project_flux.call_count, 0)
        self.assertEqual(project_ratio.call_count, 1)
        np.testing.assert_allclose(qubit._flux_transformed, baseline_flux_transformed)
        np.testing.assert_allclose(qubit._junc_ratio_transformed, expected_ratio_transformed)

    def test_flux_only_revisit_reuses_prepared_replay_state_until_non_flux_change_invalidates_it(self):
        qubit = self.make_circuit_rebuild_qubit()
        _, qubit.SMatrix_retainNodes = circuit.assemble_s_matrix_and_retain_nodes(
            qubit._ParameterizedQubit__struct
        )
        qubit._charges = np.array([0.0, 0.0, 0.0])
        qubit._cal_mode = 'Eigen'
        qubit._refresh_basic_metrics = mock.Mock()
        qubit._restore_cached_exact_solve_template_if_available = mock.Mock(return_value=False)

        updated_flux = np.array(qubit._flux, copy=True)
        updated_flux[2, 2] = 0.21
        updated_ratio = np.array(qubit._junc_ratio, copy=True)
        updated_ratio[0, 0] = 1.41

        with mock.patch.object(
            qubit_base_module,
            'project_transformed_flux',
            wraps=qubit_base_module.project_transformed_flux,
        ) as project_flux, mock.patch.object(
            qubit,
            '_Ejphi',
            wraps=ParameterizedQubit._Ejphi.__get__(qubit, ParameterizedQubit),
        ) as cached_ejphi:
            qubit.change_para(flux=updated_flux.tolist())
            qubit.change_para(flux=updated_flux.tolist())

        self.assertEqual(project_flux.call_count, 1)
        self.assertEqual(cached_ejphi.call_count, 1)

        qubit.change_para(junc_ratio=updated_ratio.tolist())

        with mock.patch.object(
            qubit_base_module,
            'project_transformed_flux',
            wraps=qubit_base_module.project_transformed_flux,
        ) as project_flux, mock.patch.object(
            qubit,
            '_Ejphi',
            wraps=ParameterizedQubit._Ejphi.__get__(qubit, ParameterizedQubit),
        ) as recalculated_ejphi:
            qubit.change_para(flux=updated_flux.tolist())

        self.assertEqual(project_flux.call_count, 1)
        self.assertEqual(recalculated_ejphi.call_count, 1)

    def test_hamiltonian_operator_cache_reuses_scaffolding_until_cache_key_changes(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0])
        qubit._Nlevel = np.array([2])
        qubit._numQubits = 1

        first = qubit._hamiltonianOperator()
        second = qubit._hamiltonianOperator()

        self.assertIs(first, second)
        self.assertIs(first[0][0], second[0][0])
        self.assertIs(first[4][0], second[4][0])
        self.assertIs(first[5][0][0], second[5][0][0])

        qubit._charges = np.array([0.5])
        third = qubit._hamiltonianOperator()

        self.assertIsNot(first, third)
        self.assertIsNot(first[0][0], third[0][0])
        self.assertIsNot(first[4][0], third[4][0])
        self.assertIsNot(first[5][0][0], third[5][0][0])

        qubit._Nlevel = np.array([3])
        fourth = qubit._hamiltonianOperator()

        self.assertIsNot(third, fourth)
        self.assertIsNot(third[0][0], fourth[0][0])
        self.assertIsNot(third[4][0], fourth[4][0])
        self.assertIsNot(third[5][0][0], fourth[5][0][0])

    def test_pair_phase_power_cache_reuses_exact_scale_tuple_and_invalidates_when_inputs_change(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        baseline_scales = qubit._get_phi_scale_cache_key([1.0, 2.0])
        first = ParameterizedQubit._build_cached_pair_power_terms(*cache_key, baseline_scales)
        second = ParameterizedQubit._build_cached_pair_power_terms(*cache_key, baseline_scales)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0][0], (0, 1))
        self.assertIs(first, second)
        self.assertIs(first[0][1][0], second[0][1][0])

        changed_scale = ParameterizedQubit._build_cached_pair_power_terms(
            *cache_key,
            qubit._get_phi_scale_cache_key([1.0, 2.5]),
        )

        self.assertIsNot(first, changed_scale)
        self.assertIsNot(first[0][1][0], changed_scale[0][1][0])

        qubit._charges = np.array([0.5, 0.0])
        changed_key = qubit._get_hamiltonian_operator_cache_key()
        changed_charge = ParameterizedQubit._build_cached_pair_power_terms(*changed_key, baseline_scales)

        self.assertIsNot(first, changed_charge)
        self.assertIsNot(first[0][1][0], changed_charge[0][1][0])

    def test_pair_phase_power_cache_reuses_unchanged_pairs_when_unrelated_scale_changes(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0, 0.0])
        qubit._Nlevel = np.array([2, 3, 4])
        qubit._numQubits = 3

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        baseline_scales = qubit._get_phi_scale_cache_key([1.0, 2.0, 3.0])
        shifted_scales = qubit._get_phi_scale_cache_key([1.0, 2.0, 3.5])

        first = ParameterizedQubit._build_cached_pair_power_terms(*cache_key, baseline_scales)
        shifted = ParameterizedQubit._build_cached_pair_power_terms(*cache_key, shifted_scales)

        self.assertEqual([entry[0] for entry in first], [(0, 1), (0, 2), (1, 2)])
        self.assertEqual([entry[0] for entry in shifted], [(0, 1), (0, 2), (1, 2)])
        self.assertIs(first[0][1], shifted[0][1])
        self.assertIs(first[0][1][0], shifted[0][1][0])
        self.assertIsNot(first[1][1], shifted[1][1])
        self.assertIsNot(first[2][1], shifted[2][1])

    def test_pair_interaction_cache_reuses_exact_scale_tuple_and_invalidates_when_inputs_change(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        phi_scale_key = qubit._get_phi_scale_cache_key([1.0, 2.0])
        ns_scale_key = qubit._get_ns_scale_cache_key([0.5, 1.5])
        first = ParameterizedQubit._build_cached_pair_interaction_terms(
            *cache_key,
            phi_scale_key,
            ns_scale_key,
        )
        second = ParameterizedQubit._build_cached_pair_interaction_terms(
            *cache_key,
            phi_scale_key,
            ns_scale_key,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0][0], (0, 1))
        self.assertIs(first, second)
        self.assertIs(first[0][1][0], second[0][1][0])
        self.assertIs(first[0][1][1], second[0][1][1])

        changed_ns = ParameterizedQubit._build_cached_pair_interaction_terms(
            *cache_key,
            phi_scale_key,
            qubit._get_ns_scale_cache_key([0.5, 1.75]),
        )

        self.assertIsNot(first, changed_ns)
        self.assertIsNot(first[0][1][0], changed_ns[0][1][0])

        qubit._charges = np.array([0.5, 0.0])
        changed_key = qubit._get_hamiltonian_operator_cache_key()
        changed_charge = ParameterizedQubit._build_cached_pair_interaction_terms(
            *changed_key,
            phi_scale_key,
            ns_scale_key,
        )

        self.assertIsNot(first, changed_charge)
        self.assertIsNot(first[0][1][0], changed_charge[0][1][0])
        self.assertIsNot(first[0][1][1], changed_charge[0][1][1])

    def test_pair_interaction_cache_reuses_unchanged_pairs_when_unrelated_scale_changes(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0, 0.0])
        qubit._Nlevel = np.array([2, 3, 4])
        qubit._numQubits = 3

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        baseline_phi_scales = qubit._get_phi_scale_cache_key([1.0, 2.0, 3.0])
        shifted_phi_scales = qubit._get_phi_scale_cache_key([1.0, 2.0, 3.5])
        baseline_ns_scales = qubit._get_ns_scale_cache_key([0.5, 1.5, 2.5])
        shifted_ns_scales = qubit._get_ns_scale_cache_key([0.5, 1.5, 2.75])

        first = ParameterizedQubit._build_cached_pair_interaction_terms(
            *cache_key,
            baseline_phi_scales,
            baseline_ns_scales,
        )
        shifted = ParameterizedQubit._build_cached_pair_interaction_terms(
            *cache_key,
            shifted_phi_scales,
            shifted_ns_scales,
        )

        self.assertEqual([entry[0] for entry in first], [(0, 1), (0, 2), (1, 2)])
        self.assertEqual([entry[0] for entry in shifted], [(0, 1), (0, 2), (1, 2)])
        self.assertIs(first[0][1], shifted[0][1])
        self.assertIsNot(first[1][1], shifted[1][1])
        self.assertIsNot(first[2][1], shifted[2][1])

    def test_pair_number_term_cache_reuses_exact_scale_tuple_and_invalidates_when_inputs_change(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        first = ParameterizedQubit._build_cached_pair_number_term(
            *cache_key,
            0,
            1,
            (0.5, 1.5),
        )
        second = ParameterizedQubit._build_cached_pair_number_term(
            *cache_key,
            0,
            1,
            (0.5, 1.5),
        )

        self.assertIs(first, second)

        changed_scale = ParameterizedQubit._build_cached_pair_number_term(
            *cache_key,
            0,
            1,
            (0.5, 1.75),
        )

        self.assertIsNot(first, changed_scale)

        qubit._charges = np.array([0.5, 0.0])
        changed_key = qubit._get_hamiltonian_operator_cache_key()
        changed_charge = ParameterizedQubit._build_cached_pair_number_term(
            *changed_key,
            0,
            1,
            (0.5, 1.5),
        )

        self.assertIsNot(first, changed_charge)

    def test_pair_number_term_zero_charge_path_skips_charge_weighted_qobj_algebra(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        ParameterizedQubit._build_cached_pair_operator_products.cache_clear()
        ParameterizedQubit._build_cached_pair_number_term.cache_clear()
        ns_pair_product = ParameterizedQubit._build_cached_pair_operator_products(
            *cache_key,
            0,
            1,
        )[1]
        qobj_type = type(ns_pair_product)
        add_calls = 0
        sub_calls = 0
        original_add = qobj_type.__add__
        original_sub = qobj_type.__sub__

        def wrapped_add(self, other):
            nonlocal add_calls
            add_calls += 1
            return original_add(self, other)

        def wrapped_sub(self, other):
            nonlocal sub_calls
            sub_calls += 1
            return original_sub(self, other)

        with mock.patch.object(qobj_type, '__add__', wrapped_add), mock.patch.object(
            qobj_type,
            '__sub__',
            wrapped_sub,
        ):
            observed = ParameterizedQubit._build_cached_pair_number_term(
                *cache_key,
                0,
                1,
                (0.5, 1.5),
            )

        expected = (0.5 * 1.5) * ns_pair_product
        np.testing.assert_allclose(observed.full(), expected.full())
        self.assertEqual(add_calls, 0)
        self.assertEqual(sub_calls, 0)

    def test_pair_number_contribution_cache_reuses_exact_scale_tuple_and_invalidates_when_inputs_change(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        first = ParameterizedQubit._build_cached_pair_number_contribution(
            *cache_key,
            0,
            1,
            (0.5, 1.5),
            0.25,
        )
        second = ParameterizedQubit._build_cached_pair_number_contribution(
            *cache_key,
            0,
            1,
            (0.5, 1.5),
            0.25,
        )

        self.assertIs(first, second)

        changed_scale = ParameterizedQubit._build_cached_pair_number_contribution(
            *cache_key,
            0,
            1,
            (0.5, 1.75),
            0.25,
        )
        self.assertIsNot(first, changed_scale)

        changed_ec = ParameterizedQubit._build_cached_pair_number_contribution(
            *cache_key,
            0,
            1,
            (0.5, 1.5),
            0.35,
        )
        self.assertIsNot(first, changed_ec)

        qubit._charges = np.array([0.5, 0.0])
        changed_key = qubit._get_hamiltonian_operator_cache_key()
        changed_charge = ParameterizedQubit._build_cached_pair_number_contribution(
            *changed_key,
            0,
            1,
            (0.5, 1.5),
            0.25,
        )

        self.assertIsNot(first, changed_charge)

    def test_scaled_phase_term_cache_reuses_exact_scale_tuple_and_invalidates_when_inputs_change(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        baseline_scales = qubit._get_phi_scale_cache_key([1.0, 2.0])
        first = ParameterizedQubit._build_cached_scaled_phase_terms(*cache_key, baseline_scales)
        second = ParameterizedQubit._build_cached_scaled_phase_terms(*cache_key, baseline_scales)

        self.assertEqual(len(first[0]), 2)
        self.assertEqual(len(first[1]), 2)
        self.assertIs(first, second)
        self.assertIs(first[0][0], second[0][0])
        self.assertIs(first[1][1][2], second[1][1][2])

        changed_scale = ParameterizedQubit._build_cached_scaled_phase_terms(
            *cache_key,
            qubit._get_phi_scale_cache_key([1.0, 2.5]),
        )

        self.assertIsNot(first, changed_scale)
        self.assertIsNot(first[0][1], changed_scale[0][1])
        self.assertIsNot(first[1][1][0], changed_scale[1][1][0])

        qubit._charges = np.array([0.5, 0.0])
        changed_key = qubit._get_hamiltonian_operator_cache_key()
        changed_charge = ParameterizedQubit._build_cached_scaled_phase_terms(*changed_key, baseline_scales)

        self.assertIsNot(first, changed_charge)
        self.assertIsNot(first[0][0], changed_charge[0][0])
        self.assertIsNot(first[1][0][0], changed_charge[1][0][0])

    def test_truncated_operator_view_cache_reuses_exact_scale_tuple_and_invalidates_when_inputs_change(self):
        qubit = self.make_qubit()
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 3])
        qubit._numQubits = 2

        cache_key = qubit._get_hamiltonian_operator_cache_key()
        phi_scale_key = qubit._get_phi_scale_cache_key([1.0, 2.0])
        ns_scale_key = qubit._get_ns_scale_cache_key([0.5, 1.5])
        first = ParameterizedQubit._build_cached_truncated_operator_views(
            *cache_key,
            phi_scale_key,
            ns_scale_key,
        )
        second = ParameterizedQubit._build_cached_truncated_operator_views(
            *cache_key,
            phi_scale_key,
            ns_scale_key,
        )

        self.assertIs(first, second)
        self.assertIs(first[0][0], second[0][0])
        self.assertIs(first[1][1], second[1][1])
        self.assertIs(first[2][0], second[2][0])

        changed_ns = ParameterizedQubit._build_cached_truncated_operator_views(
            *cache_key,
            phi_scale_key,
            qubit._get_ns_scale_cache_key([0.5, 1.75]),
        )

        self.assertIsNot(first, changed_ns)
        self.assertIsNot(first[1][1], changed_ns[1][1])

        changed_phi = ParameterizedQubit._build_cached_truncated_operator_views(
            *cache_key,
            qubit._get_phi_scale_cache_key([1.0, 2.5]),
            ns_scale_key,
        )

        self.assertIsNot(first, changed_phi)
        self.assertIsNot(first[2][1], changed_phi[2][1])

        qubit._charges = np.array([0.5, 0.0])
        changed_key = qubit._get_hamiltonian_operator_cache_key()
        changed_charge = ParameterizedQubit._build_cached_truncated_operator_views(
            *changed_key,
            phi_scale_key,
            ns_scale_key,
        )

        self.assertIsNot(first, changed_charge)
        self.assertIsNot(first[0][0], changed_charge[0][0])

    def test_transient_hamiltonian_build_preserves_existing_auxiliary_caches(self):
        qubit = self.make_qubit()
        qubit._cal_mode = 'Eigen'
        qubit._charges = np.array([0.0])
        qubit._Nlevel = np.array([2])
        qubit.eigenHamiltonian = object()
        qubit.couplingHamiltonian = object()
        qubit.highorderHamiltonian = object()
        qubit.destroyors = ['baseline-destroyor']
        qubit.n_operators = ['baseline-number']
        qubit.phi_operators = ['baseline-phase']

        original_eigen = qubit.eigenHamiltonian
        original_coupling = qubit.couplingHamiltonian
        original_highorder = qubit.highorderHamiltonian
        original_destroyors = qubit.destroyors
        original_numbers = qubit.n_operators
        original_phases = qubit.phi_operators

        hamiltonian = ParameterizedQubit._generate_hamiltonian(
            qubit,
            np.array([[10.0]]),
            np.array([[20.0]]),
            np.array([[30.0]]),
            transient=True,
        )

        self.assertEqual(hamiltonian.dims, [[2], [2]])
        self.assertIs(qubit.eigenHamiltonian, original_eigen)
        self.assertIs(qubit.couplingHamiltonian, original_coupling)
        self.assertIs(qubit.highorderHamiltonian, original_highorder)
        self.assertIs(qubit.destroyors, original_destroyors)
        self.assertIs(qubit.n_operators, original_numbers)
        self.assertIs(qubit.phi_operators, original_phases)

    def test_transient_hamiltonian_build_skips_unused_pair_phase_and_delta_terms(self):
        qubit = ParameterizedQubit.__new__(ParameterizedQubit)
        qubit._cal_mode = 'Eigen'
        qubit._charges = np.array([0.0, 0.0])
        qubit._Nlevel = np.array([2, 2])
        qubit._numQubits = 2
        qubit.eigenHamiltonian = object()
        qubit.couplingHamiltonian = object()
        qubit.highorderHamiltonian = object()
        qubit.destroyors = ['baseline-destroyor']
        qubit.n_operators = ['baseline-number']
        qubit.phi_operators = ['baseline-phase']

        with mock.patch.object(
            ParameterizedQubit,
            '_build_cached_pair_phase_product',
            side_effect=AssertionError('pair phase product should not be built when pair El is zero'),
        ), mock.patch.object(
            ParameterizedQubit,
            '_build_cached_pair_delta_terms',
            side_effect=AssertionError('pair delta terms should not be built when pair Ej is zero'),
        ):
            hamiltonian = ParameterizedQubit._generate_hamiltonian(
                qubit,
                np.array([[10.0, 0.25], [0.25, 20.0]]),
                np.array([[30.0, 0.0], [0.0, 40.0]]),
                np.array([[50.0, 0.0], [0.0, 60.0]]),
                transient=True,
            )

        self.assertEqual(hamiltonian.dims, [[2, 2], [2, 2]])

    def test_element_updates_rebuild_parameterized_qubit_via_circuit_module_end_to_end(self):
        qubit = self.make_circuit_rebuild_qubit()
        updated_capac = np.array(
            [
                [2.2, -0.35, -0.25, -0.1],
                [-0.35, 2.1, -0.3, -0.2],
                [-0.25, -0.3, 1.95, -0.25],
                [-0.1, -0.2, -0.25, 1.8],
            ]
        )
        updated_resis = np.array(
            [
                [10.5, 11.5, 12.5, 13.5],
                [11.5, 14.5, 15.5, 16.5],
                [12.5, 15.5, 18.5, 19.5],
                [13.5, 16.5, 19.5, 20.5],
            ]
        )
        updated_induc = np.array(
            [
                [42.0, 92.0, 102.0, 112.0],
                [92.0, 47.0, 97.0, 107.0],
                [102.0, 97.0, 52.0, 117.0],
                [112.0, 107.0, 117.0, 57.0],
            ]
        )

        expected_s_matrix, expected_retain_nodes = circuit.assemble_s_matrix_and_retain_nodes(
            qubit._ParameterizedQubit__struct
        )
        expected_maxwell, expected_ec, expected_el, expected_ej0 = circuit.convert_elements_to_energy_matrices(
            updated_capac,
            updated_induc,
            updated_resis,
            expected_s_matrix,
            expected_retain_nodes,
            qubit._ParameterizedQubit__struct,
            circuit.convert_resistance_to_ej0,
        )
        expected_flux_transformed = circuit.project_transformed_flux(
            qubit._flux,
            qubit._ParameterizedQubit__struct,
            expected_retain_nodes,
        )
        expected_ratio_transformed = circuit.project_transformed_junction_ratio(
            qubit._junc_ratio,
            qubit._ParameterizedQubit__struct,
            expected_retain_nodes,
        )
        expected_ej = qubit._Ejphi(expected_ej0, expected_flux_transformed, expected_ratio_transformed)

        qubit.change_para(
            capac=updated_capac.tolist(),
            resis=updated_resis.tolist(),
            induc=updated_induc.tolist(),
        )

        np.testing.assert_allclose(qubit.SMatrix, expected_s_matrix)
        self.assertEqual(qubit.SMatrix_retainNodes, expected_retain_nodes)
        np.testing.assert_allclose(qubit.Maxwellmat['capac'], expected_maxwell['capac'])
        np.testing.assert_allclose(qubit.Maxwellmat['induc'], expected_maxwell['induc'])
        np.testing.assert_allclose(qubit.Ec, expected_ec)
        np.testing.assert_allclose(qubit.El, expected_el)
        np.testing.assert_allclose(qubit.Ejmax, expected_ej0)
        np.testing.assert_allclose(qubit._flux_transformed, expected_flux_transformed)
        np.testing.assert_allclose(qubit._junc_ratio_transformed, expected_ratio_transformed)
        np.testing.assert_allclose(qubit.Ej, expected_ej)

        called_ec, called_el, called_ej = qubit._generate_hamiltonian.call_args.args
        np.testing.assert_allclose(called_ec, expected_ec)
        np.testing.assert_allclose(called_el, expected_el)
        np.testing.assert_allclose(called_ej, expected_ej)
        self.assertIsInstance(qubit.solver_result, SpectrumResult)
        self.assertEqual(qubit.solver_result.destroy_operators, ['a'])
        self.assertEqual(qubit.solver_result.number_operators, ['n'])
        self.assertEqual(qubit.solver_result.phase_operators, ['phi'])
        self.assertEqual(qubit._last_changed_params, set())


class AbstractQubitLazyMaxSpectrumTests(unittest.TestCase):
    def test_max_spectrum_materializes_lazily_without_replacing_active_solver_state(self):
        qubit = AbstractQubit(
            frequency=5e9,
            anharmonicity=-250e6,
            qubit_type='Transmon',
            is_print=False,
        )

        self.assertIsNone(qubit._max_spectrum_cache)
        active_hamiltonian = qubit.solver_result.hamiltonian
        active_levels = np.array(qubit.get_energylevel(), copy=True)

        e_max = qubit.E_max
        state_max = qubit.state_max

        self.assertIsNotNone(qubit._max_spectrum_cache)
        self.assertIs(e_max, qubit.E_max)
        self.assertIs(state_max, qubit.state_max)
        self.assertGreaterEqual(len(e_max), 3)
        self.assertEqual(len(e_max), len(state_max))
        self.assertIs(qubit.solver_result.hamiltonian, active_hamiltonian)
        np.testing.assert_allclose(qubit.get_energylevel(), active_levels)


class MultiQubitRefreshTests(unittest.TestCase):
    def make_model(self, model_cls):
        model = model_cls.__new__(model_cls)
        model._flux = np.array([[0.0]])
        model._junc_ratio = np.array([[1.0]])
        model.SMatrix_retainNodes = [0]
        model._ParameterizedQubit__struct = [1]
        model._ParameterizedQubit__nodes = 1
        model._ParameterizedQubit__capac = np.array([[1.0]])
        model._ParameterizedQubit__resis = np.array([[2.0]])
        model._ParameterizedQubit__induc = np.array([[3.0]])
        model.Ec = np.array([[10.0]])
        model.El = np.array([[20.0]])
        model.Ejmax = np.array([[30.0]])
        model._hamiltonian = None
        model._numQubits = 1
        model._Ejphi = mock.Mock(return_value=np.array([[55.0]]))
        model._generate_hamiltonian = mock.Mock(
            return_value=FakeHamiltonian(
                dims=[[2], [2]],
                eigenvalues=[0.0, 1.0],
                eigenstates=['g', 'e'],
            )
        )
        model._generate_Ematrix = mock.Mock()
        model.destroyors = ['a']
        model.n_operators = ['n']
        model.phi_operators = ['phi']
        return model

    def test_change_para_refreshes_multi_qubit_metrics_without_print_side_effects(self):
        for model_cls in (QCRFGRModel, FGF1V1Coupling, FGF2V7Coupling):
            with self.subTest(model_cls=model_cls.__name__):
                model = self.make_model(model_cls)
                model._refresh_basic_metrics = mock.Mock()
                model.print_basic_info = mock.Mock(side_effect=AssertionError('print_basic_info should not be used as a refresh hook'))

                model.change_para(flux=[[0.125]])

                model._refresh_basic_metrics.assert_called_once()
                model.print_basic_info.assert_not_called()
                self.assertIsInstance(model.solver_result, SpectrumResult)
                np.testing.assert_allclose(model.Ej, np.array([[55.0]]))


if __name__ == '__main__':
    unittest.main()
