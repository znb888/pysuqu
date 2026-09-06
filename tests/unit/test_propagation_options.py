import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from pysuqu.qubit.propagation import PropagationOptions


class PropagationOptionsTests(unittest.TestCase):
    def test_reference_defaults_match_gate_solver_defaults(self):
        from pysuqu.qubit.gate import GateBase

        self.assertEqual(PropagationOptions().qutip_options(), GateBase._default_solver_options())

    def test_mapping_retains_solver_extensions_without_mutating_input(self):
        rng = np.random.default_rng(471283)
        values = {
            'backend': 'qutip_compiled',
            'atol': float(10.0 ** rng.uniform(-10.0, -8.0)),
            'rtol': float(10.0 ** rng.uniform(-8.0, -6.0)),
            'max_step': float(rng.uniform(0.03, 0.15)),
            'extra': {'progress_bar': False},
        }
        original = {**values, 'extra': dict(values['extra'])}
        options = PropagationOptions.from_mapping(values)

        self.assertEqual(values, original)
        self.assertEqual(options.backend, values['backend'])
        self.assertEqual(options.qutip_options()['max_step'], values['max_step'])
        self.assertFalse(options.qutip_options()['progress_bar'])
        options.extra['progress_bar'] = True
        self.assertFalse(values['extra']['progress_bar'])

    def test_backend_override_preserves_representation_and_solver_controls(self):
        options = PropagationOptions(
            backend='qutip_compiled', frame='interaction_exact', matrix_format='csr',
            sparse_kernel='fused', plan_cache_size=7, parallel='off',
            block_decompose='on', sparse_expm='off', extra={'max_step': 0.07},
        )
        changed = PropagationOptions.from_mapping(options, backend='cpp')

        self.assertEqual(changed.backend, 'cpp')
        for name in ('frame', 'matrix_format', 'sparse_kernel', 'plan_cache_size',
                     'parallel', 'block_decompose', 'sparse_expm', 'extra'):
            self.assertEqual(getattr(changed, name), getattr(options, name))
        self.assertIsNot(changed.extra, options.extra)
        self.assertIs(PropagationOptions.from_mapping(options), options)

    def test_profiles_allow_an_explicit_method_override(self):
        self.assertNotIn('method', PropagationOptions().qutip_options())
        self.assertEqual(PropagationOptions(profile='fast_exact').qutip_options()['method'], 'vern7')
        self.assertEqual(
            PropagationOptions(profile='fast_exact', method='dop853').qutip_options()['method'],
            'dop853',
        )

    def test_final_state_is_available_without_storing_the_trajectory(self):
        self.assertTrue(PropagationOptions(store_states=False).qutip_options()['store_final_state'])
        self.assertFalse(
            PropagationOptions(store_states=False, store_final_state=False).qutip_options()['store_final_state']
        )

    def test_native_controls_are_not_forwarded_to_qutip(self):
        options = PropagationOptions.from_mapping({
            'backend': 'cpp', 'active_levels': 4, 'frame': 'auto',
            'matrix_format': 'csr', 'parallel': 'on', 'native_max_steps': 7000,
            'rwa_max_discarded_ratio': 0.2, 'normalize_output': False,
        })
        self.assertEqual(options.qutip_options(), PropagationOptions().qutip_options())

    def test_invalid_numerical_and_representation_options_are_rejected(self):
        invalid = (
            {'atol': 0}, {'rtol': -1}, {'atol': np.nan}, {'rtol': np.inf},
            {'nsteps': 0}, {'coefficient_order': 4}, {'rf_oversample': 0},
            {'matrix_format': 'unknown'}, {'frame': 'unknown'}, {'profile': 'unknown'},
            {'sparse_kernel': 'unknown'}, {'plan_cache_size': -1}, {'plan_cache_size': 1.5},
            {'block_decompose': 'unknown'}, {'sparse_expm': 'unknown'}, {'parallel': 'unknown'},
            {'sparse_threshold': 0}, {'sparse_threshold': 1.1}, {'sparse_threshold': np.nan},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PropagationOptions(**values)

    def test_options_are_frozen(self):
        options = PropagationOptions()
        with self.assertRaises(FrozenInstanceError):
            options.backend = 'cpp'


if __name__ == '__main__':
    unittest.main()
