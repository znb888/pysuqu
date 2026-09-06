from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
import qutip as qt

from pysuqu.funclib import ChannelSchedule, EnvelopeParams, MixerParams, PulseEvent
from pysuqu.qubit.gate import SingleQubitGate
from pysuqu.qubit.propagation import PropagationOptions, native_backend_available
from pysuqu.qubit.solver import HamiltonianEvo


@unittest.skipUnless(getattr(qt, '__version__', None), 'real QuTiP is required')
class PreparedGateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(918473)
        frequency = float(rng.uniform(4.2, 5.8))
        with redirect_stdout(StringIO()):
            self.gate = SingleQubitGate(
                total_time=float(rng.uniform(5.0, 7.0)), sample_rate=4.0,
                qubit_frequency=frequency, energy_trunc_level=3,
            )
        self.channel = ChannelSchedule(
            name='synthetic_drive', sampling_rate=4.0,
            mixer_config=MixerParams(lo_freq=frequency),
            events=[PulseEvent(start_time=0.5, envelope=EnvelopeParams(
                duration=float(rng.uniform(2.0, 3.0)), peak_amp=float(rng.uniform(8e-4, 1.4e-3)),
                shape_type='gaussian', sigma=float(rng.uniform(0.6, 0.9)),
            ))],
        )
        self.options = {'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 100000, 'store_states': False}

    def backends(self):
        return ('qutip_compiled', 'cpp') if native_backend_available() else ('qutip_compiled',)

    def test_gate_schedule_matches_reference_for_each_prepared_backend(self):
        reference = self.gate.run_simulation(channel=self.channel, options=self.options)
        for backend in self.backends():
            with self.subTest(backend=backend):
                result = self.gate.run_simulation(
                    channel=self.channel, backend=backend, options=self.options,
                )
                self.assertLess((result.final_state - reference.final_state).norm(), 3e-6)

    def test_options_container_retains_default_qutip_method(self):
        result = self.gate.run_simulation(
            channel=self.channel, options=PropagationOptions(store_states=False),
        )
        self.assertIsNotNone(result.final_state)

    def test_unitary_payload_matches_reference_and_honors_final_only_storage(self):
        reference = self.gate._extract_evolution_unitary_payload(
            channel=self.channel, frame='lab', options=self.options, store_trajectories=False,
        )
        for backend in self.backends():
            with self.subTest(backend=backend):
                payload = self.gate._extract_evolution_unitary_payload(
                    channel=self.channel, frame='lab', options=self.options,
                    store_trajectories=False, backend=backend,
                )
                self.assertLess((payload['raw_unitary'] - reference['raw_unitary']).norm(), 6e-6)
                for result in payload['basis_results']:
                    self.assertEqual(result.states, [])
                    self.assertIsNotNone(result.final_state)

    def test_trace_modes_match_direct_reference(self):
        trace = self.gate.awg.get_solver_trace(self.channel)
        for mode in ('rf', 'complex_envelope'):
            reference = self.gate.run_trace_simulation(trace, mode=mode, options=self.options)
            for backend in self.backends():
                with self.subTest(mode=mode, backend=backend):
                    result = self.gate.run_trace_simulation(
                        trace, mode=mode, options=self.options, backend=backend,
                    )
                    self.assertLess((result.final_state - reference.final_state).norm(), 3e-6)

    def test_named_multidrive_works_with_a_custom_hamiltonian(self):
        schedules = {'first': self.channel, 'second': self.channel.clone_with(name='second')}
        operators = {'first': 0.15 * qt.sigmax(), 'second': 0.23 * qt.sigmaz()}
        kwargs = dict(
            schedules=schedules, drive_operators=operators, static_hamiltonian=0.19 * qt.sigmaz(),
            initial_state=qt.basis(2, 0), options=self.options,
        )
        reference = self.gate.run_multidrive_simulation(**kwargs)
        for backend in self.backends():
            with self.subTest(backend=backend):
                result = self.gate.run_multidrive_simulation(**kwargs, backend=backend)
                self.assertLess((result.final_state - reference.final_state).norm(), 3e-6)

    def test_eigenstate_phases_are_canonical_after_restore(self):
        solver = HamiltonianEvo(qt.sigmay() + 0.21 * qt.sigmaz())
        original = solver.get_eigenstate(0)
        solver._eigenstates = [1j * state for state in solver._eigenstates]
        solver._eigenstates_canonical = False
        self.assertLess((solver.get_eigenstate(0) - original).norm(), 1e-12)


if __name__ == '__main__':
    unittest.main()
