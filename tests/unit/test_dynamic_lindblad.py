import unittest

import numpy as np
import qutip as qt

from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.backends.cpp_backend import _native_lindblad_rate_components
from pysuqu.qubit.propagation import (
    DynamicCollapseRate,
    PreparedPropagation,
    UnsupportedBackendError,
)


class DynamicLindbladTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0.0, 1.0, 21)
        self.rates = 0.16 + 0.08 * self.times
        self.trace = SignalTrace(
            self.times,
            self.rates,
            sample_rate=20.0,
            domain="rf_real",
            plane="qubit_rf",
        )

    def _descriptor(self):
        operator = qt.sigmaz()

        def coefficient(t, args=None):
            return np.sqrt(0.16 + 0.08 * float(t))

        return DynamicCollapseRate(operator, self.trace, coefficient, label="synthetic dephasing")

    def test_descriptor_matches_time_dependent_qutip_dissipator(self):
        initial = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
        result = PreparedPropagation(
            qt.qzero(2), [], self.times,
            c_ops=[self._descriptor()], backend="qutip",
            options={"atol": 1e-10, "rtol": 1e-8},
        ).propagate(initial)
        # D[sigmaz] damps the coherence at 2*r(t); the integral is analytic.
        integral = 0.16 * self.times + 0.04 * self.times ** 2
        expected_coherence = 0.5 * np.exp(-2.0 * integral)
        for state, coherence in zip(result.states, expected_coherence):
            self.assertAlmostEqual(float(np.real(state[0, 1])), float(coherence), places=7)
            self.assertAlmostEqual(float(np.imag(state[0, 1])), 0.0, places=7)

    def test_native_rate_payload_is_grid_aligned_and_read_only(self):
        prepared = PreparedPropagation(
            qt.qzero(2), [], self.times,
            c_ops=[self._descriptor()], backend="qutip",
        )
        rates = _native_lindblad_rate_components(prepared)
        np.testing.assert_allclose(rates, self.rates[np.newaxis, :])
        self.assertFalse(rates.flags.writeable)

    def test_native_rate_payload_rejects_mismatched_grid(self):
        descriptor = self._descriptor()
        prepared = PreparedPropagation(
            qt.qzero(2), [], self.times + 0.001,
            c_ops=[descriptor], backend="qutip",
        )
        with self.assertRaises(UnsupportedBackendError):
            _native_lindblad_rate_components(prepared)


if __name__ == "__main__":
    unittest.main()
