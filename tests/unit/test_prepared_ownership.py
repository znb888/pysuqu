import unittest

import numpy as np
import qutip as qt

from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.propagation import DriveTerm, DynamicCollapseRate, PreparedPropagation


class PreparedOwnershipTests(unittest.TestCase):
    def test_operator_and_sampled_trace_are_snapshotted(self):
        h0 = qt.sigmax()
        values = np.array([0.1 + 0.02j, 0.3 + 0.04j], dtype=np.complex128)
        trace = SignalTrace(
            t_axis=np.array([0.0, 1.0]), values=values, sample_rate=1.0,
            domain="iq_complex", plane="qubit_iq", lo_freq=0.25,
        )
        prepared = PreparedPropagation(
            h0, [DriveTerm(qt.sigmaz(), trace)], [0.0, 1.0], backend="qutip",
        )

        self.assertIsNot(prepared.static_hamiltonian, h0)
        self.assertIsNot(prepared.drive_terms[0].operator, prepared.static_hamiltonian)
        self.assertIsNot(prepared.drive_terms[0].trace, trace)
        values[:] = 9.0 + 4.0j
        trace.t_axis[:] = [0.0, 2.0]
        np.testing.assert_allclose(
            prepared.drive_terms[0].trace.values,
            [0.1 + 0.02j, 0.3 + 0.04j],
        )
        np.testing.assert_allclose(prepared.drive_terms[0].trace.t_axis, [0.0, 1.0])

    def test_dynamic_collapse_operator_is_owned(self):
        collapse = qt.sigmam()
        trace = SignalTrace(
            t_axis=np.array([0.0, 1.0]), values=np.array([0.2, 0.4]),
            sample_rate=1.0, domain="rf_real", plane="qubit_rf",
        )
        descriptor = DynamicCollapseRate(
            operator=collapse, rate_trace=trace, qutip_coefficient=lambda t, args: 1.0,
        )
        prepared = PreparedPropagation(
            qt.sigmaz(), [], [0.0, 1.0], c_ops=[descriptor], backend="qutip",
        )
        owned = prepared.c_ops[0]
        self.assertIsInstance(owned, DynamicCollapseRate)
        self.assertIsNot(owned.operator, collapse)
        self.assertIsNot(owned.rate_trace, trace)
        trace.values[:] = 0.9
        np.testing.assert_allclose(owned.rate_trace.values, [0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
