import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import ParameterizedQubit
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

