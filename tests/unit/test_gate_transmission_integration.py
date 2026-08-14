import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.support import install_test_stubs

install_test_stubs()

import numpy as np
import qutip as qt

from pysuqu.funclib import (
    AttenuatorStage,
    ChannelSchedule,
    EnvelopeParams,
    MixerParams,
    PulseEvent,
    TransmissionChain,
)
from pysuqu.qubit.gate import GateBase, SingleQubitGate


def make_schedule(name='drive', *, transmission_chain=None):
    envelope = EnvelopeParams(
        name='square',
        duration=1.0,
        peak_amp=0.5,
        shape_type='square',
    )
    return ChannelSchedule(
        name=name,
        sampling_rate=2.0,
        mixer_config=MixerParams(lo_freq=5.0),
        mixer_correction=False,
        events=[PulseEvent(start_time=0.0, envelope=envelope)],
        fir_kernel=np.array([0.75, 0.25]),
        transmission_chain=transmission_chain,
    )


def make_gate(*, gate_chain=None, channel=None):
    gate = SingleQubitGate.__new__(SingleQubitGate)
    gate.transmission_chain = gate_chain
    gate.pulse_channel = make_schedule() if channel is None else channel
    gate.qubit = SimpleNamespace(
        qubit_f01=0.0,
        get_hamiltonian=lambda: qt.qeye(2),
        get_eigenstate=lambda index: qt.basis(2, index),
    )
    gate.awg = SimpleNamespace(
        t_axis=np.array([0.0, 0.5]),
        get_qutip_func=MagicMock(return_value=lambda t, args=None: 0.0),
        trace_to_qutip_func=MagicMock(return_value=lambda t, args=None: 0.0),
        trace_to_qutip_rf_func=MagicMock(return_value=lambda t, args=None: 0.0),
    )
    gate.get_drive_hamiltonian = MagicMock(return_value=qt.sigmax())
    gate._get_c_ops = MagicMock(return_value=[])
    return gate


class GateTransmissionResolutionTests(unittest.TestCase):
    def test_constructor_accepts_gate_level_default_chain(self):
        chain = TransmissionChain(stages=[AttenuatorStage(loss_db=3.0)])
        channel = make_schedule()

        with patch.object(GateBase, '__init__', return_value=None):
            gate = SingleQubitGate(
                total_time=1.0,
                sample_rate=2.0,
                nco_local=5.0,
                pulse_channel=channel,
                transmission_chain=chain,
            )

        self.assertIs(gate.pulse_channel, channel)
        self.assertIs(gate.transmission_chain, chain)

    def test_chain_priority_is_call_then_channel_then_gate(self):
        gate_chain = object()
        channel_chain = object()
        explicit_chain = object()
        channel = make_schedule(transmission_chain=channel_chain)
        gate = make_gate(gate_chain=gate_chain, channel=channel)

        self.assertIs(gate._resolve_transmission_chain(channel), channel_chain)
        self.assertIs(
            gate._resolve_transmission_chain(channel, explicit_chain),
            explicit_chain,
        )
        channel.transmission_chain = None
        self.assertIs(gate._resolve_transmission_chain(channel), gate_chain)

    def test_chain_resolution_supports_legacy_schedule_like_objects(self):
        gate_chain = object()
        gate = make_gate(gate_chain=gate_chain)

        self.assertIs(gate._resolve_transmission_chain(object()), gate_chain)


class GateSimulationTransmissionTests(unittest.TestCase):
    def test_run_simulation_passes_resolved_chain_to_awg(self):
        gate_chain = object()
        channel_chain = object()
        channel = make_schedule(transmission_chain=channel_chain)
        gate = make_gate(gate_chain=gate_chain, channel=channel)
        sentinel = object()

        with patch('pysuqu.qubit.gate.qt.mesolve', return_value=sentinel) as mesolve:
            result = gate.run_simulation(channel=channel)

        self.assertIs(result, sentinel)
        gate.awg.get_qutip_func.assert_called_once_with(
            channel,
            chain=channel_chain,
        )
        self.assertEqual(mesolve.call_args.args[2].tolist(), [0.0, 0.5])

    def test_run_simulation_explicit_chain_overrides_channel_and_gate(self):
        channel = make_schedule(transmission_chain=object())
        gate = make_gate(gate_chain=object(), channel=channel)
        explicit_chain = object()

        with patch('pysuqu.qubit.gate.qt.mesolve', return_value=object()):
            gate.run_simulation(
                channel=channel,
                transmission_chain=explicit_chain,
            )

        self.assertIs(
            gate.awg.get_qutip_func.call_args.kwargs['chain'],
            explicit_chain,
        )

    def test_evolution_unitary_path_uses_resolved_chain(self):
        channel_chain = object()
        channel = make_schedule(transmission_chain=channel_chain)
        gate = make_gate(channel=channel)
        result = SimpleNamespace(
            states=[qt.basis(2, 0)],
            times=[0.0],
        )

        with patch('pysuqu.qubit.gate.qt.mesolve', return_value=result):
            gate._extract_evolution_unitary_payload(
                channel=channel,
                frame='lab',
            )

        gate.awg.get_qutip_func.assert_called_once_with(
            channel,
            chain=channel_chain,
        )

    def test_fidelity_forwards_explicit_chain_to_simulation(self):
        gate = make_gate()
        explicit_chain = object()
        solver_result = object()
        gate.run_simulation = MagicMock(return_value=solver_result)
        gate._summarize_fidelity_metrics = MagicMock(return_value={'fidelity': 1.0})

        metrics = gate.calculate_fidelity(
            transmission_chain=explicit_chain,
            is_print=False,
        )

        self.assertEqual(metrics['fidelity'], 1.0)
        self.assertIs(
            gate.run_simulation.call_args.kwargs['transmission_chain'],
            explicit_chain,
        )

    def test_unitary_fidelity_forwards_chain_to_payload_extraction(self):
        gate = make_gate()
        explicit_chain = object()
        payload = {
            'unitary': qt.qeye(2),
            'raw_unitary': qt.qeye(2),
            'average_leakage': 0.0,
            'unitarity_error': 0.0,
        }
        gate._extract_evolution_unitary_payload = MagicMock(return_value=payload)

        gate.calculate_unitary_fidelity(
            target_unitary=qt.qeye(2),
            transmission_chain=explicit_chain,
            is_print=False,
        )

        self.assertIs(
            gate._extract_evolution_unitary_payload.call_args.kwargs[
                'transmission_chain'
            ],
            explicit_chain,
        )

    def test_parameter_scan_clones_all_schedule_level_line_settings(self):
        chain = TransmissionChain(stages=[AttenuatorStage(loss_db=3.0)])
        schedule = make_schedule('scan_drive', transmission_chain=chain)
        gate = make_gate(channel=schedule)
        captured_channels = []

        def calculate_fidelity(**kwargs):
            captured_channels.append(kwargs['channel'])
            return {'fidelity': kwargs['channel'].events[0].envelope.peak_amp, 'leakage': 0.0}

        gate.calculate_fidelity = calculate_fidelity
        with (
            patch('pysuqu.qubit.gate.tqdm', side_effect=lambda values: values),
            patch('builtins.print'),
        ):
            best_value, best_fidelity = gate.scan_parameter_by_fidelity(
                'peak_amp',
                np.array([0.25, 0.75]),
                target_state=qt.basis(2, 1),
                update_best=False,
                is_plot=False,
            )

        self.assertEqual(best_value, 0.75)
        self.assertEqual(best_fidelity, 0.75)
        self.assertEqual(len(captured_channels), 2)
        for candidate in captured_channels:
            self.assertEqual(candidate.name, 'scan_drive')
            self.assertEqual(candidate.sampling_rate, 2.0)
            self.assertFalse(candidate.mixer_correction)
            self.assertIs(candidate.transmission_chain, chain)
            np.testing.assert_allclose(candidate.fir_kernel, [0.75, 0.25])


class TraceFidelityFacadeTests(unittest.TestCase):
    def test_trace_fidelity_runs_trace_simulation_and_reuses_summary(self):
        gate = make_gate()
        trace = object()
        solver_result = object()
        gate.run_trace_simulation = MagicMock(return_value=solver_result)
        gate._summarize_fidelity_metrics = MagicMock(
            return_value={'fidelity': 0.9, 'leakage': 0.1}
        )

        metrics = gate.calculate_trace_fidelity(
            trace,
            initial_state_input=1,
            options={'nsteps': 10},
            args={'phase': 0.2},
            induc_phi_model='linear',
            is_print=False,
        )

        self.assertEqual(metrics['fidelity'], 0.9)
        gate.run_trace_simulation.assert_called_once_with(
            trace,
            initial_state_input=1,
            couple_term=0.5e-12,
            couple_type='induc',
            options={'nsteps': 10},
            args={'phase': 0.2},
            induc_phi_model='linear',
        )
        self.assertIs(
            gate._summarize_fidelity_metrics.call_args.kwargs['result'],
            solver_result,
        )

    def test_trace_fidelity_accepts_precomputed_result(self):
        gate = make_gate()
        solver_result = object()
        gate.run_trace_simulation = MagicMock()
        gate._summarize_fidelity_metrics = MagicMock(return_value={'fidelity': 1.0})

        gate.calculate_trace_fidelity(
            object(),
            result=solver_result,
            is_print=False,
        )

        gate.run_trace_simulation.assert_not_called()
        self.assertIs(
            gate._summarize_fidelity_metrics.call_args.kwargs['result'],
            solver_result,
        )


class MultiDriveGateFacadeTests(unittest.TestCase):
    def _make_gate_base(self):
        gate = GateBase.__new__(GateBase)
        gate.qubit = SimpleNamespace(get_hamiltonian=lambda: qt.qeye(2))
        gate.awg = SimpleNamespace(
            t_axis=np.array([0.0, 0.5, 1.0]),
            get_qutip_bundle_funcs=MagicMock(
                return_value={
                    'x': lambda t, args=None: 1.0,
                    'y': lambda t, args=None: 2.0,
                }
            ),
        )
        return gate

    def test_build_multidrive_hamiltonian_uses_bundle_callbacks(self):
        gate = self._make_gate_base()
        schedules = {'x': make_schedule('x'), 'y': make_schedule('y')}
        operators = {'x': qt.sigmax(), 'y': qt.sigmay()}
        chain = object()

        h_total, drive_funcs = gate.build_multidrive_hamiltonian(
            schedules,
            operators,
            transmission_chain=chain,
            mode='complex_envelope',
        )

        self.assertEqual(len(h_total), 3)
        self.assertEqual(tuple(drive_funcs), ('x', 'y'))
        self.assertIs(h_total[1][0], operators['x'])
        self.assertIs(h_total[2][0], operators['y'])
        gate.awg.get_qutip_bundle_funcs.assert_called_once_with(
            schedules,
            mode='complex_envelope',
            chain=chain,
            plane='qubit',
        )

    def test_run_multidrive_uses_gate_state_collapse_and_solver_defaults(self):
        gate = self._make_gate_base()
        gate._parse_initial_state = MagicMock(return_value=qt.basis(2, 0))
        collapse = qt.sigmax()
        gate._get_c_ops = MagicMock(return_value=[collapse])
        gate.decoherence_params = {'Tphi2': 25.0}
        h_total = [qt.qeye(2)]
        gate.build_multidrive_hamiltonian = MagicMock(return_value=(h_total, {}))
        sentinel = object()

        with patch('pysuqu.qubit.gate.qt.mesolve', return_value=sentinel) as mesolve:
            result = gate.run_multidrive_simulation(
                {'x': make_schedule('x')},
                {'x': qt.sigmax()},
                initial_state=1,
                options={'nsteps': 20},
                args={'custom': 3},
            )

        self.assertIs(result, sentinel)
        gate._parse_initial_state.assert_called_once_with(1)
        self.assertIs(mesolve.call_args.kwargs['c_ops'][0], collapse)
        self.assertEqual(mesolve.call_args.kwargs['options']['nsteps'], 20)
        self.assertEqual(mesolve.call_args.kwargs['args']['Tphi2'], 25.0)
        self.assertEqual(mesolve.call_args.kwargs['args']['custom'], 3)

    def test_custom_static_hamiltonian_requires_qobj_initial_state(self):
        gate = self._make_gate_base()

        with self.assertRaisesRegex(TypeError, 'qutip.Qobj'):
            gate.run_multidrive_simulation(
                {'x': make_schedule('x')},
                {'x': qt.sigmax()},
                initial_state=0,
                static_hamiltonian=qt.qeye(2),
            )


if __name__ == '__main__':
    unittest.main()
