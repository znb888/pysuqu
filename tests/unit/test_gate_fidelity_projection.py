import unittest
from types import SimpleNamespace

from tests.support import install_test_stubs

install_test_stubs()

import numpy as np
import qutip as qt

from pysuqu.funclib.awgenerator import WaveformGenerator
from pysuqu.qubit.gate import SingleQubitGate


class _FakeQubit:
    def __init__(self):
        self.qubit_f01 = 0.0
        self._g = (qt.basis(3, 0) + qt.basis(3, 2)).unit()
        self._e = qt.basis(3, 1)
        self.hamiltonian = qt.qeye(3)
        self.drive = qt.qeye(3)

    def get_eigenstate(self, index):
        if index == 0:
            return self._g
        if index == 1:
            return self._e
        raise IndexError(index)

    def get_hamiltonian(self):
        return self.hamiltonian


class GateFidelityProjectionTests(unittest.TestCase):
    def _make_gate(self):
        gate = SingleQubitGate.__new__(SingleQubitGate)
        gate.qubit = _FakeQubit()
        gate.pulse_channel = object()
        trace_awg = WaveformGenerator.__new__(WaveformGenerator)
        gate.awg = SimpleNamespace(
            t_axis=np.array([0.0, 1.0]),
            get_qutip_func=lambda channel: (lambda t, args=None: 0.0),
            trace_to_qutip_func=trace_awg.trace_to_qutip_func,
            trace_to_qutip_rf_func=trace_awg.trace_to_qutip_rf_func,
        )
        gate.get_drive_hamiltonian = lambda **kwargs: gate.qubit.drive
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

    def test_summarize_fidelity_rejects_target_population_outside_computational_space(self):
        gate = self._make_gate()
        result = qt.Result(states=[gate.qubit.get_eigenstate(0)], times=[0.0])

        with self.assertRaisesRegex(ValueError, "non-computational population"):
            gate._summarize_fidelity_metrics(
                result=result,
                target_state=qt.basis(3, 2),
                is_print=False,
            )

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

    def test_calculate_unitary_fidelity_accepts_explicit_process_matrix(self):
        gate = self._make_gate()
        x_gate = qt.sigmax()

        metrics = gate.calculate_unitary_fidelity(
            target_unitary=x_gate,
            process_unitary=x_gate,
            is_print=False,
        )

        self.assertAlmostEqual(metrics["process_fidelity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["average_gate_fidelity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["average_leakage"], 0.0, places=9)
        self.assertEqual(metrics["unitary"].shape, (2, 2))

    def test_calculate_unitary_fidelity_reports_raw_clipping_for_gain_matrix(self):
        gate = self._make_gate()
        gain_identity = 1.1 * np.eye(2, dtype=complex)

        metrics = gate.calculate_unitary_fidelity(
            target_unitary=np.eye(2),
            process_unitary=gain_identity,
            is_print=False,
        )

        self.assertAlmostEqual(metrics["process_fidelity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["average_gate_fidelity"], 1.0, places=9)
        self.assertGreater(metrics["raw_process_fidelity"], 1.0)
        self.assertGreater(metrics["raw_average_gate_fidelity"], 1.0)
        self.assertTrue(metrics["process_fidelity_clipped"])
        self.assertTrue(metrics["average_gate_fidelity_clipped"])

    def test_calculate_unitary_fidelity_reports_projection_loss_for_non_unitary_process(self):
        gate = self._make_gate()
        lossy_identity = np.sqrt(0.75) * np.eye(2, dtype=complex)

        metrics = gate.calculate_unitary_fidelity(
            target_unitary=np.eye(2),
            process_unitary=lossy_identity,
            is_print=False,
        )

        self.assertAlmostEqual(metrics["process_fidelity"], 0.75, places=9)
        self.assertAlmostEqual(metrics["average_gate_fidelity"], 0.75, places=9)
        self.assertAlmostEqual(metrics["average_leakage"], 0.25, places=9)
        self.assertGreater(metrics["unitarity_error"], 0.0)

    def test_extract_evolution_unitary_projects_two_basis_evolutions(self):
        gate = self._make_gate()

        unitary = gate.extract_evolution_unitary()

        np.testing.assert_allclose(unitary.full(), np.eye(2), atol=1e-12)

    def test_calculate_trace_unitary_fidelity_projects_two_basis_trace_evolutions(self):
        gate = self._make_gate()
        trace = SimpleNamespace(
            t_axis=np.array([0.0, 1.0]),
            values=np.array([0.0, 0.0]),
            domain="rf_real",
            lo_freq=0.0,
        )

        metrics = gate.calculate_trace_unitary_fidelity(
            trace,
            target_unitary=np.eye(2),
            is_print=False,
        )

        self.assertAlmostEqual(metrics["process_fidelity"], 1.0, places=9)
        self.assertAlmostEqual(metrics["average_gate_fidelity"], 1.0, places=9)
        np.testing.assert_allclose(metrics["unitary"].full(), np.eye(2), atol=1e-12)

    def test_resolve_target_unitary_rejects_non_unitary_matrix(self):
        gate = self._make_gate()

        with self.assertRaisesRegex(ValueError, "unitary"):
            gate.calculate_unitary_fidelity(
                target_unitary=np.array([[1.0, 0.0], [0.0, 0.5]]),
                process_unitary=np.eye(2),
                is_print=False,
            )


if __name__ == "__main__":
    unittest.main()
