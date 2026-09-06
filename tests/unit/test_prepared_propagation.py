import unittest
from unittest.mock import patch

import numpy as np
import qutip as qt

from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.propagation import (
    BackendUnavailable, DriveTerm, PreparedPropagation, PropagationOptions,
)


@unittest.skipUnless(getattr(qt, '__version__', None), 'real QuTiP is required')
class PreparedPropagationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(285791)
        self.times = np.concatenate(([0.0], np.cumsum(rng.uniform(0.03, 0.11, 20))))
        self.angular_rate = float(rng.uniform(0.15, 0.35))
        self.h0 = self.angular_rate * qt.sigmax()
        self.trace = SignalTrace(
            t_axis=self.times, values=rng.uniform(0.03, 0.09, len(self.times)).astype(complex),
            sample_rate=12.0, domain='iq_complex', plane='qubit_iq',
            lo_freq=float(rng.uniform(2.5, 4.5)),
        )
        self.options = {'atol': 1e-10, 'rtol': 1e-8, 'store_states': False}

    def test_constant_hamiltonian_matches_analytic_rotation(self):
        initial = qt.basis(2, 0)
        angle = self.angular_rate * self.times[-1]
        expected = np.cos(angle) * initial - 1j * np.sin(angle) * qt.basis(2, 1)
        for backend in ('qutip', 'qutip_compiled'):
            with self.subTest(backend=backend):
                result = PreparedPropagation(
                    self.h0, [], self.times, backend=backend, options=self.options,
                ).propagate(initial)
                self.assertLess((result.final_state - expected).norm(), 1e-7)

    def test_compiled_rf_retains_the_reference_carrier_on_a_nonuniform_grid(self):
        terms = [DriveTerm(qt.sigmaz(), self.trace)]
        results = [PreparedPropagation(
            self.h0, terms, self.times, backend=backend, options=self.options,
        ).propagate(qt.basis(2, 0)) for backend in ('qutip', 'qutip_compiled')]
        self.assertLess((results[0].final_state - results[1].final_state).norm(), 1e-7)

    def test_reusable_solver_and_batch_keep_initial_states_independent(self):
        prepared = PreparedPropagation(
            self.h0, [], self.times,
            options={**self.options, 'use_solver_class': True, 'profile': 'fast_exact'},
        )
        states = [qt.basis(2, 0), qt.basis(2, 1), (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()]
        batch = prepared.propagate_batch(states)
        angle = self.angular_rate * self.times[-1]
        unitary = np.cos(angle) * qt.qeye(2) - 1j * np.sin(angle) * qt.sigmax()
        self.assertEqual(len(batch.final_states), len(states))
        for state, final in zip(states, batch.final_states):
            self.assertLess((final - unitary * state).norm(), 1e-7)
        self.assertLess((prepared.propagate(states[0]).final_state - batch.final_states[0]).norm(), 1e-10)

    def test_static_collapse_accepts_density_matrices(self):
        rate = 0.13
        result = PreparedPropagation(
            qt.qzero(2), [], self.times, c_ops=[np.sqrt(rate) * qt.destroy(2)],
            options=self.options,
        ).propagate(qt.ket2dm(qt.basis(2, 1)))
        expected = np.diag([1.0 - np.exp(-rate * self.times[-1]), np.exp(-rate * self.times[-1])])
        np.testing.assert_allclose(result.final_state.full(), expected, atol=1e-8)

    def test_expectation_values_are_forwarded_to_qutip(self):
        result = PreparedPropagation(self.h0, [], self.times, options=self.options).propagate(
            qt.basis(2, 0), e_ops=[qt.sigmaz()],
        )
        expected = np.cos(2 * self.angular_rate * self.times)
        np.testing.assert_allclose(result.expect[0], expected, atol=1e-7)

    def test_missing_native_extension_is_explicit_or_uses_auto_fallback(self):
        with patch('pysuqu.qubit.propagation.native_backend_available', return_value=False):
            prepared = PreparedPropagation(self.h0, [], self.times, backend='auto', options=self.options)
            self.assertEqual(prepared.backend, 'qutip_compiled')
            self.assertIsNotNone(prepared.propagate(qt.basis(2, 0)).final_state)
            explicit = PreparedPropagation(self.h0, [], self.times, backend='cpp')
            with self.assertRaises(BackendUnavailable):
                explicit.propagate(qt.basis(2, 0))

    def test_invalid_output_grid_is_rejected_before_solving(self):
        for times in ([], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan], [[0.0, 1.0]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                PreparedPropagation(self.h0, [], times)

    def test_empty_batch_has_a_stable_result_contract(self):
        batch = PreparedPropagation(self.h0, [], self.times).propagate_batch([])
        self.assertEqual(batch.final_states, [])
        self.assertEqual(batch.results, [])
        np.testing.assert_array_equal(batch.times, self.times)


if __name__ == '__main__':
    unittest.main()
