import unittest

import numpy as np
import qutip as qt

from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.propagation import DriveTerm, PreparedPropagation, native_backend_available


@unittest.skipUnless(getattr(qt, '__version__', None) and native_backend_available(),
                     'real QuTiP and a compiled native extension are required')
class NativePropagationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(417859)
        self.times = np.linspace(0.0, rng.uniform(1.2, 1.8), 25)
        self.frequency = float(rng.uniform(0.15, 0.3))
        self.h0 = self.frequency * qt.sigmax()
        self.trace = SignalTrace(
            t_axis=self.times, values=rng.uniform(0.02, 0.07, len(self.times)).astype(complex),
            sample_rate=1.0 / (self.times[1] - self.times[0]),
            domain='iq_complex', plane='qubit_iq', lo_freq=float(rng.uniform(2.0, 3.0)),
        )
        self.options = {'atol': 1e-10, 'rtol': 1e-8, 'store_states': False}

    def test_static_native_solution_matches_analytic_rotation(self):
        result = PreparedPropagation(
            self.h0, [], self.times, backend='cpp', options=self.options,
        ).propagate(qt.basis(2, 0))
        angle = self.frequency * self.times[-1]
        expected = np.cos(angle) * qt.basis(2, 0) - 1j * np.sin(angle) * qt.basis(2, 1)
        self.assertLess((result.final_state - expected).norm(), 1e-7)
        self.assertEqual(result.stats['backend'], 'cpp')
        self.assertEqual(result.stats['approximation'], 'none')

    def test_native_rf_matches_qutip_with_the_same_envelope(self):
        terms = [DriveTerm(qt.sigmaz(), self.trace)]
        results = [PreparedPropagation(
            self.h0, terms, self.times, backend=backend, options=self.options,
        ).propagate(qt.basis(2, 0)) for backend in ('qutip', 'cpp')]
        self.assertLess((results[0].final_state - results[1].final_state).norm(), 2e-6)

    def test_native_batch_retains_trajectory_dimensions_and_column_order(self):
        initial = [qt.basis(2, 1), qt.basis(2, 0)]
        prepared = PreparedPropagation(
            self.h0, [], self.times, backend='cpp', options={**self.options, 'store_states': True},
        )
        batch = prepared.propagate_batch(initial)
        self.assertEqual(len(batch.results), len(initial))
        for state, result in zip(initial, batch.results):
            self.assertEqual(len(result.states), len(self.times))
            self.assertEqual(result.final_state.dims, state.dims)
            for time, sample in zip(self.times, result.states):
                rotation = np.cos(self.frequency * time) * qt.qeye(2) - 1j * np.sin(self.frequency * time) * qt.sigmax()
                self.assertLess((sample - rotation * state).norm(), 1e-7)

    def test_sampled_rf_is_not_mixed_with_the_carrier_twice(self):
        trace = self.trace.clone(values=np.real(self.trace.values), domain='rf_real', plane='qubit_rf')
        results = [PreparedPropagation(
            self.h0, [DriveTerm(qt.sigmaz(), trace)], self.times,
            backend=backend, options=self.options,
        ).propagate(qt.basis(2, 0)) for backend in ('qutip', 'cpp')]
        self.assertLess((results[0].final_state - results[1].final_state).norm(), 2e-6)

    def test_csr_propagation_matches_an_independent_sparse_exponential(self):
        from scipy.sparse import diags
        from scipy.sparse.linalg import expm_multiply

        rng = np.random.default_rng(514937)
        dimension = 48
        diagonal = rng.uniform(-0.3, 0.3, dimension)
        coupling = rng.uniform(0.01, 0.08, dimension - 1)
        matrix = diags([coupling, diagonal, coupling], [-1, 0, 1], dtype=complex)
        prepared = PreparedPropagation(
            qt.Qobj(matrix), [], self.times, backend='cpp',
            options={**self.options, 'matrix_format': 'csr'},
        )
        for index in (0, dimension - 1):
            initial = qt.basis(dimension, index)
            result = prepared.propagate(initial)
            expected = expm_multiply(-1j * matrix * self.times[-1], initial.full()[:, 0])
            np.testing.assert_allclose(result.final_state.full()[:, 0], expected, atol=2e-7)
            self.assertEqual(result.stats['matrix_format'], 'csr')

    def test_unnormalized_inputs_retain_their_norm(self):
        initial = 2.7 * qt.basis(2, 0)
        result = PreparedPropagation(
            self.h0, [], self.times, backend='cpp', options=self.options,
        ).propagate(initial)
        self.assertAlmostEqual(result.final_state.norm(), initial.norm(), places=7)


if __name__ == '__main__':
    unittest.main()
