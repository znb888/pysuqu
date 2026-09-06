import unittest
from unittest.mock import patch

import numpy as np
import qutip as qt
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

from pysuqu import _native
from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.propagation import DriveTerm, PreparedPropagation


@unittest.skipUnless(getattr(qt, '__version__', None) and callable(_native.propagate_lindblad_csr),
                     'real QuTiP and matrix-free native kernels are required')
class KrylovPropagationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(519827)
        self.times = np.array([0.0, 0.13, 0.31, 0.68, 1.17])
        self.options = {'atol': 1e-11, 'rtol': 1e-9, 'store_states': False, 'block_decompose': 'off'}

    def hamiltonian(self, dimension):
        diagonal = self.rng.uniform(-0.25, 0.25, dimension)
        coupling = self.rng.uniform(0.04, 0.13, dimension - 1)
        return diags([coupling, diagonal, coupling], [-1, 0, 1], dtype=complex).tocsr()

    def test_large_constant_sparse_action_matches_scipy(self):
        dimension = 320
        matrix = self.hamiltonian(dimension)
        initial = qt.basis(dimension, 0)
        result = PreparedPropagation(
            qt.Qobj(matrix), [], self.times, backend='cpp',
            options={**self.options, 'matrix_format': 'csr', 'sparse_expm': 'auto'},
        ).propagate(initial)
        expected = expm_multiply(-1j * matrix * self.times[-1], initial.full()[:, 0])
        np.testing.assert_allclose(result.final_state.full()[:, 0], expected, atol=2e-8)
        self.assertIn('krylov', result.stats['integrator'])

    def test_piecewise_constant_actions_preserve_each_source_interval(self):
        dimension = 72
        matrix = self.hamiltonian(dimension)
        control = diags(self.rng.uniform(-0.2, 0.2, dimension), dtype=complex).tocsr()
        values = self.rng.uniform(0.05, 0.18, len(self.times))
        trace = SignalTrace(self.times, values.astype(complex), 8.0, 'iq_complex', 'qubit_iq')
        initial = qt.basis(dimension, 1)
        result = PreparedPropagation(
            qt.Qobj(matrix), [DriveTerm(qt.Qobj(control), trace, mode='complex_envelope')],
            self.times, backend='cpp', options={
                **self.options, 'matrix_format': 'csr', 'coefficient_order': 0, 'sparse_expm': 'on',
            },
        ).propagate(initial)
        expected = initial.full()[:, 0]
        for interval, value in zip(np.diff(self.times), values[:-1]):
            expected = expm_multiply(-1j * (matrix + value * control) * interval, expected)
        np.testing.assert_allclose(result.final_state.full()[:, 0], expected, atol=2e-8)
        self.assertIn('piecewise_krylov', result.stats['integrator'])

    def test_parallel_batches_match_serial_execution(self):
        dimension = 96
        matrix = self.hamiltonian(dimension)
        initial = [qt.basis(dimension, index) for index in (0, 7, 21, 40)]
        results = [PreparedPropagation(
            qt.Qobj(matrix), [], self.times, backend='cpp', options={
                **self.options, 'matrix_format': 'csr', 'sparse_expm': 'on', 'parallel': policy,
            },
        ).propagate_batch(initial) for policy in ('off', 'on')]
        for serial, parallel in zip(results[0].final_states, results[1].final_states):
            self.assertLess((serial - parallel).norm(), 1e-10)
        self.assertEqual(results[0].stats['parallel_workers'], 1)
        self.assertGreaterEqual(results[1].stats['parallel_workers'], 1)

    def test_lindblad_krylov_avoids_a_kronecker_liouvillian(self):
        hamiltonian = 0.21 * qt.sigmax() + 0.13 * qt.sigmaz()
        collapse = [np.sqrt(0.17) * qt.destroy(2), np.sqrt(0.08) * qt.sigmaz()]
        initial = qt.ket2dm((qt.basis(2, 0) + 0.37j * qt.basis(2, 1)).unit())
        with patch('scipy.sparse.kron', side_effect=AssertionError('Unexpected explicit Liouvillian')):
            results = [PreparedPropagation(
                hamiltonian, [], self.times, c_ops=collapse, backend='cpp',
                options={**self.options, 'sparse_expm': mode},
            ).propagate(initial) for mode in ('off', 'on')]
        self.assertLess((results[0].final_state - results[1].final_state).norm(), 2e-8)
        self.assertIn('krylov', results[1].stats['integrator'])
        self.assertAlmostEqual(results[1].final_state.tr(), 1.0, places=8)

    def test_fast_mode_reports_its_approximation(self):
        frequency = float(self.rng.uniform(2.0, 3.0))
        times = np.linspace(0.0, 2.0, 41)
        trace = SignalTrace(times, np.full(len(times), 0.04 + 0j), 20.0, 'iq_complex', 'qubit_iq', frequency)
        hamiltonian = qt.Qobj(np.diag([0.0, 2 * np.pi * frequency]))
        results = [PreparedPropagation(
            hamiltonian, [DriveTerm(qt.sigmax(), trace)], times,
            backend=backend, options=self.options,
        ).propagate(qt.basis(2, 0)) for backend in ('cpp', 'cpp_fast')]
        self.assertEqual(results[1].stats['approximation'], 'rwa')
        self.assertEqual(results[1].stats['active_levels'], 2)
        self.assertLess((results[0].final_state - results[1].final_state).norm(), 2e-3)


if __name__ == '__main__':
    unittest.main()
