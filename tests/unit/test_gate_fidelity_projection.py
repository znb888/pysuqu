import unittest

from tests.support import install_test_stubs

install_test_stubs()

import qutip as qt

from pysuqu.qubit.gate import SingleQubitGate


class _FakeQubit:
    def __init__(self):
        self.qubit_f01 = 0.0
        self._g = (qt.basis(3, 0) + qt.basis(3, 2)).unit()
        self._e = qt.basis(3, 1)

    def get_eigenstate(self, index):
        if index == 0:
            return self._g
        if index == 1:
            return self._e
        raise IndexError(index)


class GateFidelityProjectionTests(unittest.TestCase):
    def _make_gate(self):
        gate = SingleQubitGate.__new__(SingleQubitGate)
        gate.qubit = _FakeQubit()
        gate.pulse_channel = object()
        return gate

    def test_summarize_fidelity_projects_full_dimensional_states_into_eigenbasis(self):
        gate = self._make_gate()
        g_state = gate.qubit.get_eigenstate(0)
        result = qt.Result(states=[g_state], times=[0.0])

        metrics = gate._summarize_fidelity_metrics(
            result=result,
            target_state=g_state,
            is_print=False,
        )

        self.assertAlmostEqual(metrics["fidelity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["leakage"], 0.0, places=9)
        self.assertEqual(metrics["final_state_rot"].shape, (2, 2))

    def test_summarize_fidelity_accepts_two_level_target_states(self):
        gate = self._make_gate()
        e_state = gate.qubit.get_eigenstate(1)
        result = qt.Result(states=[e_state], times=[0.0])

        metrics = gate._summarize_fidelity_metrics(
            result=result,
            target_state=qt.basis(2, 1),
            is_print=False,
        )

        self.assertAlmostEqual(metrics["fidelity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["leakage"], 0.0, places=9)

    def test_calculate_fidelity_forwards_inductive_operator_model(self):
        gate = self._make_gate()
        captured = {}

        def _run_simulation(**kwargs):
            captured.update(kwargs)
            return qt.Result(states=[gate.qubit.get_eigenstate(0)], times=[0.0])

        gate.run_simulation = _run_simulation

        metrics = gate.calculate_fidelity(
            target_state=qt.basis(2, 0),
            is_print=False,
            induc_phi_model="linear",
        )

        self.assertEqual(captured["induc_phi_model"], "linear")
        self.assertAlmostEqual(metrics["fidelity"], 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
