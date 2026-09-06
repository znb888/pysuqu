import unittest

import numpy as np
import qutip as qt

from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit.propagation import (
    DriveTerm, PreparedPropagation, UnsupportedBackendError,
    clear_native_plan_cache, native_backend_available, native_plan_cache_info,
)


@unittest.skipUnless(getattr(qt, '__version__', None) and native_backend_available(),
                     'real QuTiP and native propagation are required')
class OpenNativeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(728391)
        self.times = np.concatenate(([0.0], np.cumsum(self.rng.uniform(0.04, 0.12, 14))))
        self.rate = float(self.rng.uniform(0.12, 0.27))
        self.options = {'atol': 1e-11, 'rtol': 1e-9, 'store_states': True}

    def test_lindblad_decay_accepts_kets_and_density_matrices(self):
        initial = qt.basis(2, 1)
        for state in (initial, qt.ket2dm(initial)):
            with self.subTest(state_type=state.type):
                result = PreparedPropagation(
                    qt.qzero(2), [], self.times, c_ops=[np.sqrt(self.rate) * qt.destroy(2)],
                    backend='cpp', options=self.options,
                ).propagate(state)
                for time, density in zip(self.times, result.states):
                    population = np.exp(-self.rate * time)
                    np.testing.assert_allclose(density.full(), np.diag([1 - population, population]), atol=1e-8)
                    self.assertAlmostEqual(density.tr(), 1.0, places=9)

    def test_open_system_matches_an_independent_master_equation(self):
        from scipy.integrate import solve_ivp

        hamiltonian = self.rate * qt.sigmax()
        collapse = np.sqrt(self.rate) * qt.destroy(2)
        initial = qt.ket2dm((qt.basis(2, 0) + 0.6j * qt.basis(2, 1)).unit())
        h, c = hamiltonian.full(), collapse.full()
        product = c.conj().T @ c

        def rhs(_time, values):
            density = values.reshape(2, 2)
            return (-1j * (h @ density - density @ h)
                    + c @ density @ c.conj().T - 0.5 * (product @ density + density @ product)).reshape(-1)

        expected = solve_ivp(
            rhs, (self.times[0], self.times[-1]), initial.full().reshape(-1),
            t_eval=self.times, method='DOP853', atol=1e-13, rtol=1e-11,
        )
        self.assertTrue(expected.success)
        result = PreparedPropagation(
            hamiltonian, [], self.times, c_ops=[collapse], backend='cpp', options=self.options,
        ).propagate(initial)
        for actual, reference in zip(result.states, expected.y.T):
            np.testing.assert_allclose(actual.full(), reference.reshape(2, 2), atol=2e-8)

    def test_nonuniform_polynomial_traces_preserve_the_requested_output_grid(self):
        from scipy.interpolate import make_interp_spline

        values = self.rng.uniform(0.08, 0.22, len(self.times))
        output_times = np.unique(np.concatenate((self.times, (self.times[:-1] + self.times[1:]) / 2)))
        trace = SignalTrace(self.times, values.astype(complex), 12.0, 'iq_complex', 'qubit_iq')
        energies = np.array([-0.31, 0.23])
        drive = np.array([0.17, -0.09])
        initial = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
        for order in (0, 1, 2, 3):
            with self.subTest(order=order):
                spline = make_interp_spline(self.times, values, k=order)
                result = PreparedPropagation(
                    qt.Qobj(np.diag(energies)),
                    [DriveTerm(qt.Qobj(np.diag(drive)), trace, mode='complex_envelope')],
                    output_times, backend='cpp', options={**self.options, 'coefficient_order': order},
                ).propagate(initial)
                np.testing.assert_array_equal(result.times, output_times)
                for time, state in zip(output_times, result.states):
                    integral = (
                        np.sum(values[:-1] * np.clip(time - self.times[:-1], 0.0, np.diff(self.times)))
                        if order == 0 else spline.integrate(0.0, time)
                    )
                    phase = energies * time + drive * integral
                    expected = np.exp(-1j * phase) * initial.full()[:, 0]
                    np.testing.assert_allclose(state.full()[:, 0], expected, atol=2e-9)

    def test_exact_interaction_frame_accepts_nondiagonal_static_hamiltonians(self):
        trace = SignalTrace(self.times, np.full(len(self.times), 0.13 + 0j), 12.0, 'iq_complex', 'qubit_iq')
        hamiltonian = 0.23 * qt.sigmax() + 0.17 * qt.sigmaz()
        terms = [DriveTerm(qt.sigmaz(), trace, mode='complex_envelope')]
        outputs = [PreparedPropagation(
            hamiltonian, terms, self.times, backend='cpp', options={**self.options, 'frame': frame},
        ).propagate(qt.basis(2, 0)) for frame in ('lab', 'interaction_exact')]
        for lab, interaction in zip(outputs[0].states, outputs[1].states):
            self.assertLess((lab - interaction).norm(), 2e-8)

    def test_block_decomposition_preserves_relative_amplitudes(self):
        from scipy.linalg import block_diag

        hamiltonian = qt.Qobj(block_diag(0.13 * qt.sigmax().full(), 0.27 * qt.sigmay().full()))
        initial = (qt.basis(4, 0) + 0.37j * qt.basis(4, 3)).unit()
        outputs = [PreparedPropagation(
            hamiltonian, [], self.times, backend='cpp', options={**self.options, 'block_decompose': mode},
        ).propagate(initial) for mode in ('off', 'on')]
        self.assertLess((outputs[0].final_state - outputs[1].final_state).norm(), 2e-8)
        self.assertEqual(outputs[1].stats['block_count'], 2)

    def test_plan_cache_is_bounded_and_does_not_cache_initial_states(self):
        clear_native_plan_cache()
        try:
            for frequency in self.rng.uniform(0.1, 0.3, 5):
                prepared = PreparedPropagation(
                    frequency * qt.sigmax(), [], self.times, backend='cpp',
                    options={**self.options, 'plan_cache_size': 2},
                )
                zero = prepared.propagate(qt.basis(2, 0)).final_state
                one = prepared.propagate(qt.basis(2, 1)).final_state
                self.assertAlmostEqual(abs(zero.overlap(one)), 0.0, places=8)
                self.assertLessEqual(native_plan_cache_info()['entries'], 2)
        finally:
            clear_native_plan_cache()

    def test_callable_collapse_uses_explicit_error_or_auto_fallback(self):
        collapse = [[qt.destroy(2), lambda t, **kwargs: np.sqrt(self.rate)]]
        auto = PreparedPropagation(qt.qzero(2), [], self.times, c_ops=collapse, backend='auto', options=self.options)
        result = auto.propagate(qt.basis(2, 1))
        self.assertEqual(result.stats['backend_fallback'], 'qutip_compiled')
        self.assertIn('backend_fallback_reason', result.stats)
        explicit = PreparedPropagation(qt.qzero(2), [], self.times, c_ops=collapse, backend='cpp')
        with self.assertRaises(UnsupportedBackendError):
            explicit.propagate(qt.basis(2, 1))


if __name__ == '__main__':
    unittest.main()
