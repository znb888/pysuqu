import unittest
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.base import ParameterizedQubit
from pysuqu.qubit.multi import FGF1V1Coupling, QCRFGRModel
from pysuqu.qubit.types import SpectrumResult


class FakeHamiltonian:
    def __init__(self, dims):
        self.dims = dims

    def eigenstates(self):
        return np.array([0.0, 1.0]), ['g', 'e']


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
        model._hamiltonian = None
        model._numQubits = 2
        model._Nlevel = [3, 3]
        model._cal_mode = 'Eigen'
        model._charges = np.array([0, 0])
        model._generate_hamiltonian = mock.Mock(return_value=FakeHamiltonian([[3, 3], [3, 3]]))
        model._refresh_basic_metrics = mock.Mock(side_effect=lambda: setattr(model, 'qubit_f01', 6.0 + model._flux[2, 2]))
        model.destroyors = ['a']
        model.n_operators = ['n']
        model.phi_operators = ['phi']
        model.change_para = ParameterizedQubit.change_para.__get__(model, QCRFGRModel)
        return model

    def test_qcrfgr_coupler_flux_probe_restores_full_flux_state(self):
        model = self.make_qcrfgr_model()
        original_flux = model._flux.copy()

        frequency = model._get_qubit_frequency_at_coupler_flux(0.35, qubit_idx=0)

        self.assertAlmostEqual(frequency, 6.35)
        np.testing.assert_allclose(model._flux, original_flux)
        self.assertEqual(model._generate_hamiltonian.call_count, 2)
        self.assertEqual(model._refresh_basic_metrics.call_count, 2)
        self.assertIsInstance(model.solver_result, SpectrumResult)


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

