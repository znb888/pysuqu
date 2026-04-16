import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import ParameterizedQubit, QubitBase
from pysuqu.qubit.multi import FGF1V1Coupling
from pysuqu.qubit.sweeps import (
    sweep_multi_qubit_coupling_strength_vs_flux,
    sweep_multi_qubit_energy_vs_flux,
)


class FakeHamiltonian:
    def __init__(self, dims):
        self.dims = dims

    def eigenstates(self):
        return np.array([0.0, 1.0, 2.0]), ['g', 'e', 'f']


class FGF1V1FluxWorkflowTests(unittest.TestCase):
    def setUp(self):
        QubitBase._clear_exact_solve_template_cache()

    def tearDown(self):
        QubitBase._clear_exact_solve_template_cache()

    def make_model(self):
        model = FGF1V1Coupling.__new__(FGF1V1Coupling)
        model._flux = np.array(
            [
                [0.0, 0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.3],
                [0.0, 0.0, 0.0, 0.3, 0.0],
            ],
            dtype=float,
        )
        model._junc_ratio = np.ones_like(model._flux)
        model.SMatrix_retainNodes = [0, 2, 3]
        model._ParameterizedQubit__struct = [2, 1, 2]
        model._ParameterizedQubit__nodes = 5
        model._ParameterizedQubit__capac = np.eye(5)
        model._ParameterizedQubit__resis = np.eye(5)
        model._ParameterizedQubit__induc = np.ones((5, 5))
        model.Ec = np.diag([1.0, 2.0, 3.0])
        model.El = np.diag([4.0, 5.0, 6.0])
        model.Ejmax = np.diag([7.0, 8.0, 9.0])
        model._hamiltonian = None
        model._numQubits = 3
        model._Nlevel = [3, 3, 3]
        model._cal_mode = 'Eigen'
        model._charges = np.array([0, 0, 0])
        model._generate_hamiltonian = mock.Mock(return_value=FakeHamiltonian([[3, 3, 3], [3, 3, 3]]))
        model._refresh_basic_metrics = mock.Mock()
        model.find_state = mock.Mock(side_effect=[0, 1, 2])
        model.get_energylevel = mock.Mock(
            side_effect=[
                2 * np.pi,
                3 * np.pi,
                4 * np.pi,
                5 * np.pi,
                6 * np.pi,
                7 * np.pi,
            ]
        )
        model.get_readout_couple = mock.Mock(return_value=0.0)
        model.get_qq_dcouple = mock.Mock(return_value=0.0)
        model.get_qc_couple = mock.Mock(return_value=0.0)
        model.get_qq_ecouple = mock.Mock(return_value=0.0)
        model.destroyors = ['a']
        model.n_operators = ['n']
        model.phi_operators = ['phi']
        model.change_para = ParameterizedQubit.change_para.__get__(model, FGF1V1Coupling)
        return model

    @staticmethod
    def build_real_stubbed_model():
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

    def test_envs_flux_walks_coupler_biases_and_restores_original_flux(self):
        model = self.make_model()
        original_flux = model._flux.copy()

        energies = sweep_multi_qubit_energy_vs_flux(model, [0.25, 0.35], is_plot=False)

        self.assertEqual(energies.sweep_values, [0.25, 0.35])
        self.assertIsNone(energies.metadata['qubits_flux'])
        np.testing.assert_allclose(energies.series['|[0, 0, 1]>'], np.array([1.0, 2.5]))
        np.testing.assert_allclose(energies.series['|[1, 0, 0]>'], np.array([1.5, 3.0]))
        np.testing.assert_allclose(energies.series['|[0, 1, 0]>'], np.array([2.0, 3.5]))
        np.testing.assert_allclose(model._flux, original_flux)
        self.assertEqual(model._generate_hamiltonian.call_count, 3)
        self.assertEqual(model._refresh_basic_metrics.call_count, 3)

    def test_coupling_strength_vs_coupler_flux_restores_original_flux(self):
        model = self.make_model()
        original_flux = model._flux.copy()
        model.get_qq_ecouple = mock.Mock(side_effect=[1.2, -0.4])

        coupling = sweep_multi_qubit_coupling_strength_vs_flux(model, [0.25, 0.35], is_plot=False)

        np.testing.assert_allclose(coupling.coupling_values, np.array([1.2, -0.4]))
        np.testing.assert_allclose(model._flux, original_flux)
        self.assertEqual(model._generate_hamiltonian.call_count, 3)
        self.assertEqual(model._refresh_basic_metrics.call_count, 3)
        self.assertEqual(model.get_qq_ecouple.call_count, 2)

    def test_coupling_strength_vs_coupler_flux_reuses_original_bias_metric(self):
        model = self.make_model()
        original_flux = model._flux.copy()
        model.get_qq_ecouple = mock.Mock(side_effect=[-0.4, 1.2])

        coupling = sweep_multi_qubit_coupling_strength_vs_flux(model, [0.25, 0.2], is_plot=False)

        np.testing.assert_allclose(coupling.coupling_values, np.array([1.2, -0.4]))
        np.testing.assert_allclose(model._flux, original_flux)
        self.assertEqual(model._generate_hamiltonian.call_count, 2)
        self.assertEqual(model._refresh_basic_metrics.call_count, 2)
        self.assertEqual(model.get_qq_ecouple.call_count, 2)

    def test_fast_es_coupling_sweep_matches_generic_path_for_real_fgf1v1_model(self):
        flux_axis = [0.095, 0.11, 0.125, 0.14]

        fast_model = self.build_real_stubbed_model()
        original_fast_flux = np.array(fast_model._flux, copy=True)
        fast_result = sweep_multi_qubit_coupling_strength_vs_flux(
            fast_model,
            flux_axis,
            method='ES',
            solver_mode='fast',
            is_plot=False,
        )

        slow_model = self.build_real_stubbed_model()
        slow_result = sweep_multi_qubit_coupling_strength_vs_flux(
            slow_model,
            flux_axis,
            method='ES',
            solver_mode='full',
            is_plot=False,
        )

        np.testing.assert_allclose(
            fast_result.coupling_values,
            slow_result.coupling_values,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(fast_model._flux, original_fast_flux)
        np.testing.assert_allclose(slow_model._flux, original_fast_flux)
        self.assertEqual(fast_result.metadata['path'], 'fgf1v1_low_spectrum_fast')
        self.assertEqual(fast_result.metadata['solver_mode'], 'fast')
        self.assertEqual(slow_result.metadata['path'], 'generic_full_spectrum')
        self.assertEqual(slow_result.metadata['solver_mode'], 'full')

    def test_auto_solver_mode_prefers_fast_path_for_supported_fgf1v1_es_sweep(self):
        model = self.build_real_stubbed_model()

        result = sweep_multi_qubit_coupling_strength_vs_flux(
            model,
            [0.095, 0.11],
            method='ES',
            solver_mode='auto',
            is_plot=False,
        )

        self.assertEqual(result.metadata['solver_mode'], 'fast')


if __name__ == '__main__':
    unittest.main()
