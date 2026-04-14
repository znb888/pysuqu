import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu import qubit
from pysuqu.qubit.base import AbstractQubit, ParameterizedQubit, QubitBase
from pysuqu.qubit.gate import GateBase, SingleQubitGate
from pysuqu.qubit.multi import (
    FGF1V1Coupling,
    FGF2V7Coupling,
    FGFGG1V1V3Coupling,
    GroundedTransmonList,
    QCRFGRModel,
)
from pysuqu.qubit.single import FloatingTransmon, GroundedTransmon, SingleQubitBase
from pysuqu.qubit.solver import HamiltonianEvo


class FakeHamiltonian:
    dims = [[1], [1]]

    def eigenstates(self):
        return np.array([0.0]), ['g']


class DummyQubit(QubitBase):
    def _recalculate_hamiltonian(self):
        pass


class DummyParameterizedQubit(ParameterizedQubit):
    pass


class QubitSplitModuleTests(unittest.TestCase):
    @staticmethod
    def _make_dummy_qubit():
        qubit = DummyQubit.__new__(DummyQubit)
        qubit.Ec = np.array([[1.0]])
        qubit.El = np.array([[2.0]])
        qubit.Ejmax = np.array([[3.0]])
        qubit.Ej = np.array([[4.0]])
        return qubit

    @staticmethod
    def _make_dummy_parameterized_qubit(qubit_cls=ParameterizedQubit):
        qubit = qubit_cls.__new__(qubit_cls)
        qubit._ParameterizedQubit__capac = np.array([[1.0]])
        qubit._ParameterizedQubit__induc = np.array([[2.0]])
        qubit._ParameterizedQubit__resis = np.array([[3.0]])
        qubit._flux = np.array([[4.0]])
        return qubit

    @staticmethod
    def _make_hamiltonian_evo():
        return HamiltonianEvo(FakeHamiltonian())

    @staticmethod
    def _make_single_qubit_base_placeholder():
        return SingleQubitBase.__new__(SingleQubitBase)

    @staticmethod
    def _make_fgf1_placeholder():
        return FGF1V1Coupling.__new__(FGF1V1Coupling)

    def test_extracted_classes_live_in_new_modules(self):
        self.assertEqual(HamiltonianEvo.__module__, 'pysuqu.qubit.solver')
        self.assertEqual(QubitBase.__module__, 'pysuqu.qubit.base')
        self.assertEqual(AbstractQubit.__module__, 'pysuqu.qubit.base')
        self.assertEqual(ParameterizedQubit.__module__, 'pysuqu.qubit.base')
        self.assertEqual(GateBase.__module__, 'pysuqu.qubit.gate')
        self.assertEqual(SingleQubitGate.__module__, 'pysuqu.qubit.gate')
        self.assertEqual(SingleQubitBase.__module__, 'pysuqu.qubit.single')
        self.assertEqual(GroundedTransmon.__module__, 'pysuqu.qubit.single')
        self.assertEqual(FloatingTransmon.__module__, 'pysuqu.qubit.single')
        self.assertEqual(QCRFGRModel.__module__, 'pysuqu.qubit.multi')
        self.assertEqual(FGF1V1Coupling.__module__, 'pysuqu.qubit.multi')
        self.assertEqual(FGF2V7Coupling.__module__, 'pysuqu.qubit.multi')
        self.assertEqual(GroundedTransmonList.__module__, 'pysuqu.qubit.multi')
        self.assertEqual(FGFGG1V1V3Coupling.__module__, 'pysuqu.qubit.multi')

    def test_package_export_keeps_same_solver_class(self):
        self.assertIs(qubit.HamiltonianEvo, HamiltonianEvo)

    def test_preferred_energy_matrix_accessor_exposes_expected_matrices(self):
        qubit = self._make_dummy_qubit()

        energy_matrices = qubit.get_energy_matrices()

        self.assertEqual(set(energy_matrices), {'Ec', 'El', 'Ej_max', 'Ej'})
        np.testing.assert_array_equal(energy_matrices['Ec'], qubit.Ec)
        np.testing.assert_array_equal(energy_matrices['Ej_max'], qubit.Ejmax)
        np.testing.assert_array_equal(energy_matrices['Ej'], qubit.Ej)
        np.testing.assert_array_equal(qubit.get_energy_matrices('El'), qubit.El)

    def test_preferred_element_matrix_accessor_exposes_expected_matrices_for_subclasses(self):
        qubit = self._make_dummy_parameterized_qubit(DummyParameterizedQubit)

        element_matrices = qubit.get_element_matrices()

        self.assertEqual(set(element_matrices), {'capac', 'induc', 'resis', 'flux'})
        np.testing.assert_array_equal(element_matrices['capac'], qubit._ParameterizedQubit__capac)
        np.testing.assert_array_equal(element_matrices['flux'], qubit._flux)
        np.testing.assert_array_equal(qubit.get_element_matrices('induc'), qubit._ParameterizedQubit__induc)
        np.testing.assert_array_equal(qubit.get_element_matrices('resis'), qubit._ParameterizedQubit__resis)

    def test_set_inistate_placeholder_fails_explicitly(self):
        evo = self._make_hamiltonian_evo()

        with self.assertRaisesRegex(NotImplementedError, r"HamiltonianEvo\.set_inistate\(\)"):
            evo.set_inistate(object())

    def test_hamiltonian_evolution_placeholder_fails_explicitly(self):
        evo = self._make_hamiltonian_evo()

        with self.assertRaisesRegex(NotImplementedError, r"HamiltonianEvo\.hamiltonian_evolution\(\)"):
            evo.hamiltonian_evolution()

    def test_single_qubit_environment_placeholders_fail_explicitly(self):
        qubit = self._make_single_qubit_base_placeholder()

        for method_name in ("EnvsCapa", "EnvsInduc", "EnvsJuncResis"):
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"SingleQubitBase\.{method_name}\(\)",
                ):
                    getattr(qubit, method_name)()

    def test_single_qubit_readout_photon_placeholder_fails_explicitly(self):
        qubit = self._make_single_qubit_base_placeholder()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"SingleQubitBase\.EnvsReadoutphoton\(\)",
        ):
            qubit.EnvsReadoutphoton()

    def test_single_qubit_readout_parameter_inductive_branch_fails_explicitly(self):
        qubit = self._make_single_qubit_base_placeholder()
        qubit._refresh_basic_metrics = lambda: SimpleNamespace(f01=5.0, anharmonicity=-0.2)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"SingleQubitBase\.get_Readout_parameter\(\) does not implement coupling_mode\['rq'\] == 'induc' yet\.",
        ):
            qubit.get_Readout_parameter(
                rq_coupleterm=1e-15,
                readout_freq=6.5e9,
                coupling_mode={"rq": "induc", "rf": "induc"},
            )

    def test_fgf1_coupler_thermal_dephasing_placeholder_fails_explicitly(self):
        qubit = self._make_fgf1_placeholder()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"FGF1V1Coupling\.QubitDephasingbyCouplerThermal\(\)",
        ):
            qubit.QubitDephasingbyCouplerThermal(0.0)

    def test_grounded_transmon_list_constructor_placeholder_fails_explicitly(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            r"GroundedTransmonList\.__init__\(\)",
        ):
            GroundedTransmonList()

    def test_fgfgg1v1v3_constructor_placeholder_fails_explicitly(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            r"FGFGG1V1V3Coupling\.__init__\(\)",
        ):
            FGFGG1V1V3Coupling([1.0])


if __name__ == '__main__':
    unittest.main()
