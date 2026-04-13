'''
 Author: Zhou Naibin
 USTC
 Since 2025-12-19

The unit requirements in this program are as follows: 
 All input parameters
adopt the International System of Units. 
 As for output parameters, the unit of
energy is in gigahertz (GHz).
'''
# import
import numpy as np
import qutip as qt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union, List, Tuple, Dict, Optional, Literal
from dataclasses import replace
from tqdm import tqdm
from copy import copy

# local lib
from .base import AbstractQubit, Phi0, e, pi
from ..funclib.awgenerator import *

# --- Base Class ---
class GateBase:
    def __init__(
        self,
        total_time: float,
        sample_rate: float,
        qubit_frequency: float = 5.0,
        qubit_anharmonicity: float = -0.25,
        qubit_freqmax: float = None,
        qubit_type: str = 'Transmon',
        energy_trunc_level: int = 12,
    ):
        """
        Initialize the basic gate simulation environment and hardware.

        Args:
            total_time (float): Total duration of the simulation [ns].
            sample_rate (float): AWG sampling rate [GS/s].
            qubit_frequency (float): Qubit linear frequency (f01) [GHz].
            qubit_anharmonicity (float): Qubit anharmonicity (alpha) [GHz].
            qubit_freqmax (float, optional): Maximum qubit frequency (for flux-tunable qubits) [GHz].
            qubit_type (str): The physical model of the qubit (e.g., 'Transmon').
            energy_trunc_level (Union[list, np.ndarray]): Number of energy levels to simulate (truncation dimension).
        """
        # Initialize AbstractQubit (Assumed external class)
        self.qubit = AbstractQubit(
            frequency=qubit_frequency * 1e9,
            anharmonicity=qubit_anharmonicity * 1e9,
            frequency_max=qubit_freqmax,
            qubit_type=qubit_type,
            energy_trunc_level=energy_trunc_level
        )
        # Initialize WaveformGenerator
        self.awg = WaveformGenerator(
            total_time=total_time,
            sample_rate=sample_rate,
            anharmonicity=self.qubit.qubit_anharm
        )
        print('AWG initialized. ')

class SingleQubitGate(GateBase):
    def __init__(
        self, 
        total_time: float, 
        sample_rate: float, 
        qubit_frequency: float = 5.0, 
        qubit_anharmonicity: float = -0.25, 
        nco_local: Optional[float] = None,
        qubit_freqmax: Optional[float] = None, 
        qubit_type: str = 'Transmon', 
        energy_trunc_level: int = 10, 
        pulse_channel: ChannelSchedule = None,
    ):
        """
        Initialize a single-qubit gate simulation.

        Args:
            total_time (float): Total duration of the simulation [ns].
            sample_rate (float): AWG sampling rate [GS/s].
            qubit_frequency (float): Qubit linear frequency (f01) [GHz].
            qubit_anharmonicity (float): Qubit anharmonicity (alpha) [GHz].
            qubit_freqmax (float, optional): Maximum qubit frequency [GHz].
            qubit_type (str): Qubit model type.
            energy_trunc_level (List[int]): Hilbert space dimension for the qubit.
        """
        super().__init__(
            total_time, sample_rate, qubit_frequency, 
            qubit_anharmonicity, qubit_freqmax, 
            qubit_type, energy_trunc_level
        )
        if nco_local is None:
            self.nco_local = self.qubit.qubit_f01
        else:
            self.nco_local = nco_local
        if pulse_channel is None:
            self.pulse_channel = ChannelSchedule(
                mixer_config=MixerParams(lo_freq=self.nco_local)
            )
        else:
            self.pulse_channel = pulse_channel

    def load_pulse(self, pulses: Union[PulseEvent, List[PulseEvent]]) -> ChannelSchedule:
        """
        Inject/Load pulse event into the gate instance.
        
        Args:
            params: The PulseEvent object to be used as the active configuration.  
        """
        print(f"Pulses loaded into: {self.pulse_channel.name}")
        if isinstance(pulses, list):
            for p in pulses:
                self.pulse_channel.events.append(p)
                print(f"  - {p.name} (Start_time={p.start_time:.4e}, If_freq={p.if_freq*1e3:.4f}, Frame_change={p.frame_change:.4f})")
            self.pulse_channel.events.sort(key=lambda x: x.start_time)
        else:
            self.pulse_channel.events.append(pulses)
            print(f"  - {pulses.name} (Start_time={pulses.start_time:.4e}, If_freq={pulses.if_freq*1e3:.4f}, Frame_change={pulses.frame_change:.4f})")
        return self.pulse_channel
    
    def pulse_reload(self, pulses: Union[PulseEvent, List[PulseEvent]] = None) -> ChannelSchedule:
        """
        Reload pulse event into the gate instance.
        
        Args:
            params: The PulseEvent object to be used as the active configuration.  
        """
        self.pulse_channel.events = []
        print(f"Pulse channel {self.pulse_channel.name} cleaned. ")
        if pulses is not None:
            self.load_pulse(pulses)
            print(f"All new pulses loaded into: {self.pulse_channel.name}")
        else:
            print(f"No pulse event in {self.pulse_channel.name}. ")
        return self.pulse_channel
    
    def load_channel(self, channel: ChannelSchedule, is_print: bool = True) -> ChannelSchedule:
        """
        Load pulse channel into the gate instance.
        
        Args:
            channel: The ChannelSchedule object to be used as the active configuration.  
        """
        self.pulse_channel = channel
        print(f"Channel loaded: {channel.name}")
        if is_print:
            self.pulse_channel.display()
        return self.pulse_channel
    
    def channel_reload(self, channel: ChannelSchedule = None, is_print: bool = True) -> ChannelSchedule:
        """
        Reload pulse channel into the gate instance.
        
        Args:
            channel: The ChannelSchedule object to be used as the active configuration.  
        """
        if channel is None:
            self.pulse_channel = ChannelSchedule(
                mixer_config=MixerParams(lo_freq=self.nco_local)
            )
        else:
            self.pulse_channel = channel
        print(f"Channel reloaded: {self.pulse_channel.name}")
        if is_print:
            self.pulse_channel.display()
        return self.pulse_channel
    
    def load_decoherence(
        self, 
        T1: float = None, 
        Tphi1: float = None, 
        Tphi2: float = None
    ) -> None:
        """
        Load decoherence parameters.
        
        Args:
            T1 (float): Energy relaxation time [ns]. Decay: exp(-t/T1).
            Tphi1 (float): Pure dephasing time (exponential) [ns]. Decay: exp(-t/Tphi1).
            Tphi2 (float): Pure dephasing time (gaussian) [ns]. Decay: exp(-(t/Tphi2)^2).
        """
        self.decoherence_params = {
            "T1": T1,
            "Tphi1": Tphi1,
            "Tphi2": Tphi2
        }
        print(f"Decoherence loaded: {self.decoherence_params}")

    def clean_decoherence(self) -> None:
        """Clean decoherence parameters."""
        delattr(self, 'decoherence_params')
        print(f"Decoherence cleaned. ")

    def get_drive_hamiltonian(self, couple_term: float, couple_type: Literal['induc', 'capac'] = 'induc', R_line: float = 50.0) -> qt.Qobj:
        """
        Construct drive hamiltonian (e.g., a + a.dag()).
        
        Args:
            couple_term(float): Couple strength of control line, H if couple_type is 'induc', F if couple_type is 'capac'.
            couple_type(str): Couple type of control line. 
        """
        if couple_type == 'induc':
            phi_op = self.qubit.phi_operators[0]
            return self.qubit.Ej[0][0]*2*pi*couple_term*(phi_op-phi_op**3/6)/Phi0/R_line
        elif couple_type == 'capac':
            n_op = self.qubit.n_operators[0]
            return couple_term*n_op/(e**2/2/self.qubit.Ec[0][0]+couple_term)
        else:
            raise TypeError(f"Couple Type {couple_type} not supported! ")
    
    def _get_c_ops(self) -> list:
        """Construct collapse operators based on loaded parameters."""
        c_ops = []
        if not hasattr(self, 'decoherence_params'):
            return c_ops

        T1 = self.decoherence_params.get("T1")
        Tphi1 = self.decoherence_params.get("Tphi1")
        Tphi2 = self.decoherence_params.get("Tphi2")
        
        a = self.qubit.destroyors[0]
        n = a.dag() * a

        if T1 is not None and T1 > 0:
            rate_T1 = 1.0 / T1
            c_ops.append(np.sqrt(rate_T1) * a)

        if Tphi1 is not None and Tphi1 > 0:
            rate_phi1 = 2.0 / Tphi1
            c_ops.append(np.sqrt(rate_phi1) * n)

        if Tphi2 is not None and Tphi2 > 0:
            def coeff_tphi2(t, args):
                return np.sqrt(2 * t) / args['Tphi2']
            
            c_ops.append([n, coeff_tphi2])

        return c_ops

    def _parse_initial_state(self, state_input: Union[int, List[complex]]) -> qt.Qobj:
        """Helper: Convert index or coeff list to Qobj state vector."""
        if isinstance(state_input, int):
            return self.qubit.get_eigenstate(state_input)
        
        coeffs = np.array(state_input, dtype=complex)
        coeffs = coeffs / np.linalg.norm(coeffs)

        psi = sum(c * self.qubit.get_eigenstate(i) for i, c in enumerate(coeffs))
        return psi.unit()

    def visualize_pulse(self, params: Union[ChannelSchedule, PulseEvent] = None, plot_mode: Literal['iq', 'rf'] = 'iq') -> None:
        """
        Visualizes the generated I/Q envelopes.
        Supports both single PulseEvent or pulse in channel (Sequence).
        """
        if params is None:
            self.awg.plot_schedule(self.pulse_channel,plot_mode=plot_mode)
        elif isinstance(params, PulseEvent):
            self.awg.plot_pulse(params, plot_mode=plot_mode)
        elif isinstance(params, ChannelSchedule):
            self.awg.plot_schedule(params, plot_mode=plot_mode)
        else:
            raise TypeError("Do not support type of params.")
        
    def run_simulation(
        self, 
        channel: Union[ChannelSchedule, None] = None,
        initial_state_input: Union[int, List[complex]] = 0,
        **kwargs
    ) -> qt.Result:
        """
        Evolve qubit state under pulse drive.
        
        Args:
            params: ChannelSchedule.
            initial_state_input: int index or list of coefficients.
        Returns:
            qutip.Result object.
        """
        # [Safety Check] Determine which params to use
        if channel is None:
            if self.pulse_channel is None:
                raise ValueError("No channel loaded! Please call .load_channel() first or pass 'pulse_channel' argument.")
            channel = self.pulse_channel
        
        psi0 = self._parse_initial_state(initial_state_input)
        c_term = kwargs.get('couple_term', 0.5e-12)
        c_type = kwargs.get('couple_type', 'induc')
        
        H_static = self.qubit.get_hamiltonian()
        H_drive = self.get_drive_hamiltonian(couple_term=c_term, couple_type=c_type)
        drive_func = self.awg.get_qutip_func(channel)
        c_ops = self._get_c_ops()
        
        H_total = [H_static, [H_drive, drive_func]]
        opts = {
            "nsteps": 5000,
            "atol": 1e-8, 
            "rtol": 1e-6,
            "store_states": True
        }
        solver_args = {}
        if hasattr(self, 'decoherence_params') and self.decoherence_params.get("Tphi2"):
            solver_args['Tphi2'] = self.decoherence_params["Tphi2"]
        
        result = qt.mesolve(
            H_total, psi0, self.awg.t_axis, 
            c_ops=c_ops, e_ops=[], options=opts, args=solver_args
        )
        return result

    def plot_bloch_evolution(self, result: qt.Result, rotation_omega: float) -> None:
        """
        Plot state evolution on Bloch sphere in Rotating Frame.
        Note: Uses QuTiP's matplotlib backend (standard for Bloch spheres).
        """
        b = qt.Bloch()
        b.view = [-45, 30]
        b.point_marker = ['o']; b.point_size = [20]
        
        x, y, z = [], [], []
        
        phase_correction = np.exp(- 1j * rotation_omega * np.array(result.times))
        
        for i, state in enumerate(result.states):
            if state.isket:
                rho = state * state.dag()
            else:
                rho = state

            r00 = rho[0, 0]
            r01 = rho[0, 1]
            r10 = rho[1, 0]
            r11 = rho[1, 1]
            
            r01_rot = r01 * phase_correction[i]
            r10_rot = r10 * np.conj(phase_correction[i])
            
            rho_rot = qt.Qobj([[r00, r01_rot], [r10_rot, r11]])
            
            x.append(qt.expect(qt.sigmax(), rho_rot))
            y.append(qt.expect(qt.sigmay(), rho_rot))
            z.append(qt.expect(qt.sigmaz(), rho_rot))
            
        b.add_points([x, y, z], meth='l')
        b.add_points([x[0], y[0], z[0]], meth='s')
        b.add_points([x[-1], y[-1], z[-1]], meth='s')
        b.show()
        
        return (x, y, z)

    def calculate_fidelity(
        self, 
        channel: ChannelSchedule = None, 
        target_state: Union[qt.Qobj, List[complex]] = qt.basis(2,1), 
        couple_term: float = 0.5e-12,  # 10 fH (typical mutual inductance)
        couple_type: Literal['induc', 'capac'] = 'induc',
        initial_state_input: Union[int, List[complex]] = 0,
        result: qt.Result = None,
        is_print: bool = True,
    ) -> Dict[str, float]:
        """
        Compute State Fidelity (prob), Leakage, and Phase Error.
        
        Args:
            target_state: Ideal 2D qubit state (e.g., basis(2,1)).
        Returns:
            Dictionary containing metrics.
        """
        if channel is None:
            if self.pulse_channel is None:
                raise ValueError("No channel loaded! Please call .load_channel() first or pass 'channel' argument.")
            channel = self.pulse_channel
        if result is None:
            res = self.run_simulation(channel=channel, initial_state_input=initial_state_input, couple_term=couple_term, couple_type=couple_type)
        else:
            res = result
        
        final_state = res.states[-1]
        t_final = res.times[-1]
        phase_factor = 2 * pi * self.qubit.qubit_f01 * t_final
        
        if final_state.isket:
            rho = final_state * final_state.dag()
        else:
            rho = final_state
        
        N = rho.shape[0]
        U = qt.qdiags(np.exp(-1j * phase_factor * np.arange(N)), 0)
        rho_rot = U * rho * U.dag()
        
        r00 = rho_rot[0, 0]
        r11 = rho_rot[1, 1]
        leakage = 1.0 - np.real(r00 + r11)
        
        if isinstance(target_state, list):
            coefs = np.array(target_state)
        elif isinstance(target_state, qt.Qobj):
            coefs = target_state.full().flatten().tolist()
        t_state = self._parse_initial_state(coefs)
        rho_target = t_state * t_state.dag()
        fid_val = qt.fidelity(rho_rot, rho_target)**2
        
        actual_rho01_phase = np.angle(rho_rot[0, 1])
        target_rho01_phase = np.angle(rho_target[0, 1])
            
        phase_err = np.degrees(actual_rho01_phase - target_rho01_phase)
        phase_err = (phase_err + 180) % 360 - 180
        
        if fid_val > 1.0: fid_val = 1.0

        if is_print:
            print(f"Fid: {fid_val*100:.5f}%, Leak: {leakage:.5e}, PhaseErr: {phase_err:.2f}")
            
        return {
            "fidelity": fid_val,
            "leakage": leakage,
            "phase_error_deg": phase_err,
            "final_state_rot": rho_rot
        }

    def scan_parameter_by_fidelity(
        self, 
        param_name: str, 
        scan_range: np.ndarray, 
        target_state: qt.Qobj,
        pulse_index: int = 0,
        update_best: bool = True,
        initial_state_input: Union[int, List[complex]] = 0,
        is_plot: bool = True,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Generic scanner for any attribute in PulseEvent (Amp, Drag, Freq, etc.).
        
        Args:
            param_name (str): Name of the attribute to scan (e.g., 'gate_amp', 'drag_coeff').
            scan_range (np.ndarray): Array of values to iterate over.
            target_state (qt.Qobj): Ideal target state (Qubit subspace) for fidelity calculation.
            pulse_index (int): Index of the pulse to scan (default is 0).
            update_best (bool): If True, automatically updates self.pulse_channel with the best value found.
            initial_state_input (Union[int, List[complex]]): Initial state index (0) or superposition coeffs.
            **kwargs: Additional physics arguments passed to get_drive_operator (e.g., couple_term, couple_type).
            
        Returns:
            Tuple[float, float]: (best_parameter_value, max_fidelity)
        """
        if self.pulse_channel is None:
            raise ValueError("No channel loaded! Call .load_channel() first.")
            
        if hasattr(self.pulse_channel.events[pulse_index].envelope, param_name):
            p_in_enve = True
        elif hasattr(self.pulse_channel.events[pulse_index], param_name):
            p_in_enve = False
        else:
            raise AttributeError(f"No attribute '{param_name}' found.")

        fids, leaks = [], []
        print(f"Scanning '{param_name}' ({len(scan_range)} points)...")

        for val in tqdm(scan_range):
            if p_in_enve:
                curr_envelope = replace(self.pulse_channel.events[pulse_index].envelope, **{param_name: val})
                curr_event = replace(self.pulse_channel.events[pulse_index], envelope=curr_envelope)
            else:
                curr_event = replace(self.pulse_channel.events[pulse_index], **{param_name: val})
            
            unite_events = copy(self.pulse_channel.events)
            unite_events[pulse_index] = curr_event
            # Preserve mixer_config from original channel
            curr_schedule = ChannelSchedule(
                events=unite_events,
                mixer_config=self.pulse_channel.mixer_config
            )
            
            metrics = self.calculate_fidelity(
                target_state=target_state, 
                channel=curr_schedule, 
                initial_state_input=initial_state_input, 
                is_print=False,
                **kwargs 
            )
            fids.append(metrics['fidelity'])
            leaks.append(metrics['leakage'])

        fids = np.array(fids)
        best_idx = np.argmax(fids)
        best_val = scan_range[best_idx]
        max_fid = fids[best_idx]

        if update_best:
            print(f"-> Updating best value: {param_name} = {best_val:.5g}")
            if p_in_enve:
                curr_envelope = replace(self.pulse_channel.events[pulse_index].envelope, **{param_name: best_val})
                curr_event = replace(self.pulse_channel.events[pulse_index], envelope=curr_envelope)
            else:
                curr_event = replace(self.pulse_channel.events[pulse_index], **{param_name: best_val})
            self.pulse_channel.events[pulse_index] = curr_event
        
        if is_plot:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=scan_range, y=fids, name="Fidelity", line=dict(color='blue')),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(x=scan_range, y=leaks, name="Leakage", line=dict(color='gray', dash='dot')),
                secondary_y=True
            )

            fig.add_vline(x=best_val, line_width=1, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f"Scan: {param_name} (Best: {best_val:.4g}, Fid: {max_fid:.4f})",
                xaxis_title=param_name,
                hovermode="x unified"
            )
            fig.update_yaxes(title_text="Fidelity", secondary_y=False)
            fig.update_yaxes(title_text="Leakage", secondary_y=True)
            fig.show()
        
        return best_val, max_fid

