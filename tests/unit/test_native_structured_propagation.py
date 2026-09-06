import unittest

import numpy as np
import qutip as qt

from pysuqu import _native
from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.propagation import DriveTerm, PreparedPropagation


@unittest.skipUnless(getattr(qt, '__version__', None) and callable(_native.propagate_banded),
                     'real QuTiP and structured native kernels are required')
class StructuredNativeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(672913)
        self.dimension = 10
        self.times = np.linspace(0.17, 1.43, 31)
        self.energies = rng.uniform(-0.7, 0.7, self.dimension)
        coupling = rng.uniform(0.03, 0.12, self.dimension - 1)
        self.control = qt.Qobj(np.diag(coupling, 1) + np.diag(coupling, -1))
        self.trace = SignalTrace(
            t_axis=self.times,
            values=(rng.uniform(0.1, 0.2) + rng.uniform(0.01, 0.05) * np.sin(self.times)).astype(complex),
            sample_rate=1.0 / (self.times[1] - self.times[0]),
            domain='iq_complex', plane='qubit_iq', lo_freq=0.0,
        )
        self.initial = (qt.basis(self.dimension, 0) + 0.31j * qt.basis(self.dimension, 2)).unit()
        self.h0 = qt.Qobj(np.diag(self.energies))
        self.options = {'atol': 1e-11, 'rtol': 1e-9, 'store_states': True}

    def solve(self, **options):
        return PreparedPropagation(
            self.h0, [DriveTerm(self.control, self.trace, mode='complex_envelope')],
            self.times, backend='cpp', options={**self.options, **options},
        ).propagate(self.initial)

    def test_operator_layouts_represent_the_same_evolution(self):
        reference = self.solve(matrix_format='dense')
        for matrix_format in ('csr', 'banded', 'fused_csr'):
            with self.subTest(matrix_format=matrix_format):
                result = self.solve(matrix_format=matrix_format)
                self.assertEqual(len(result.states), len(self.times))
                for actual, expected in zip(result.states, reference.states):
                    self.assertLess((actual - expected).norm(), 2e-8)
                self.assertEqual(result.stats['approximation'], 'none')

    def test_interaction_frame_returns_lab_states_at_nonzero_initial_time(self):
        lab = self.solve(matrix_format='csr', frame='lab')
        interaction = self.solve(matrix_format='csr', frame='interaction_exact')
        self.assertEqual(interaction.stats['frame'], 'interaction_exact')
        for actual, expected in zip(interaction.states, lab.states):
            self.assertLess((actual - expected).norm(), 2e-8)

    def test_diagonal_propagation_matches_exact_phases(self):
        result = PreparedPropagation(
            self.h0, [], self.times, backend='cpp', options=self.options,
        ).propagate(self.initial)
        for time, state in zip(self.times, result.states):
            expected = np.exp(-1j * self.energies * (time - self.times[0])) * self.initial.full()[:, 0]
            np.testing.assert_allclose(state.full()[:, 0], expected, atol=1e-11)
        self.assertEqual(result.stats['rhs_evaluations'], 0)

    def test_constant_dense_propagation_matches_spectral_solution(self):
        hamiltonian = self.h0 + self.control
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.full())
        coefficients = eigenvectors.conj().T @ self.initial.full()[:, 0]
        result = PreparedPropagation(
            hamiltonian, [], self.times, backend='cpp', options=self.options,
        ).propagate(self.initial)
        for time, state in zip(self.times, result.states):
            expected = eigenvectors @ (np.exp(-1j * eigenvalues * (time - self.times[0])) * coefficients)
            np.testing.assert_allclose(state.full()[:, 0], expected, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
