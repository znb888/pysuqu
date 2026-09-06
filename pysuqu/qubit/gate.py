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
from typing import Any, Union, List, Tuple, Dict, Optional, Literal, Sequence
from dataclasses import replace
from tqdm import tqdm
from copy import copy

# local lib
from .base import AbstractQubit, Phi0, e, pi
from .solver import HamiltonianEvo
from .propagation import (
    DriveTerm,
    PreparedPropagation,
    PropagationOptions,
    UnsupportedBackendError,
    _PROFILE_METHODS,
)
from ..funclib.awgenerator import *
from ..funclib import truncate_hilbert_space
from ..funclib.transmission import TransmissionChain


def _load_plotly_helpers():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotly is required for gate plotting helpers"
        ) from exc
    return go, make_subplots


def _as_complex_scalar(value) -> complex:
    if isinstance(value, qt.Qobj):
        data = value.full()
        if data.size != 1:
            raise ValueError("Expected a scalar qutip.Qobj.")
        return complex(data.reshape(-1)[0])
    return complex(value)

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

    @staticmethod
    def _default_solver_options(
        overrides: Optional[Dict[str, Any]] = None,
        *,
        include_internal: bool = True,
    ) -> Dict[str, Any]:
        """Shared QuTiP solver defaults used by gate-level simulation helpers."""
        if isinstance(overrides, PropagationOptions):
            resolved_qutip_options = overrides.qutip_options()
            override_values = dict(overrides.extra)
            override_values.update(
                {
                    "backend": overrides.backend,
                    # A profile is meaningful even when the caller leaves
                    # method unset; resolve it before forwarding to QuTiP.
                    "method": resolved_qutip_options.get("method"),
                    "atol": overrides.atol,
                    "rtol": overrides.rtol,
                    "nsteps": overrides.nsteps,
                    "store_states": overrides.store_states,
                    "coefficient_order": overrides.coefficient_order,
                    "use_solver_class": overrides.use_solver_class,
                    "matrix_format": overrides.matrix_format,
                    "frame": overrides.frame,
                    "sparse_kernel": overrides.sparse_kernel,
                    "plan_cache_size": overrides.plan_cache_size,
                    "block_decompose": overrides.block_decompose,
                    "sparse_expm": overrides.sparse_expm,
                    "parallel": overrides.parallel,
                    "profile": overrides.profile,
                    "sparse_threshold": overrides.sparse_threshold,
                }
            )
            if overrides.store_final_state is not None:
                override_values["store_final_state"] = overrides.store_final_state
            if overrides.rf_oversample is not None:
                override_values["rf_oversample"] = overrides.rf_oversample
        else:
            override_values = dict(overrides or {})
            if "profile" in override_values:
                profile = override_values["profile"]
                if profile not in _PROFILE_METHODS:
                    raise ValueError(
                        "profile must be 'reference', 'fast', 'fast_exact', 'fallback', or 'stiff'"
                    )
                profile_method = _PROFILE_METHODS[profile]
                if override_values.get("method") is None and profile_method is not None:
                    override_values["method"] = profile_method
        options = {
            "nsteps": 5000,
            "atol": 1e-8,
            "rtol": 1e-6,
            "store_states": True,
        }
        if override_values:
            if include_internal:
                options.update(override_values)
            else:
                options.update(
                    {
                        key: value
                        for key, value in override_values.items()
                        if key not in {
                            'active_levels',
                            'active_levels_tol',
                            'backend',
                            'coefficient_order',
                            'matrix_format',
                            'frame',
                            'profile',
                            'sparse_threshold',
                            'sparse_kernel',
                            'plan_cache_size',
                            'block_decompose',
                            'sparse_expm',
                            'parallel',
                            'native_max_steps',
                            'rf_oversample',
                            'rwa_max_discarded_ratio',
                            'use_solver_class',
                        }
                    }
                )
        if options.get("store_states") is False and "store_final_state" not in options:
            options["store_final_state"] = True
        if options.get("method") is None:
            options.pop("method", None)
        return options

    @staticmethod
    def _options_mapping(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize a solver options mapping or ``PropagationOptions`` instance."""
        if isinstance(options, PropagationOptions):
            resolved_qutip_options = options.qutip_options()
            values = dict(options.extra)
            values.update(
                {
                    "backend": options.backend,
                    "method": resolved_qutip_options.get("method"),
                    "atol": options.atol,
                    "rtol": options.rtol,
                    "nsteps": options.nsteps,
                    "store_states": options.store_states,
                    "coefficient_order": options.coefficient_order,
                    "use_solver_class": options.use_solver_class,
                    "matrix_format": options.matrix_format,
                    "frame": options.frame,
                    "sparse_kernel": options.sparse_kernel,
                    "plan_cache_size": options.plan_cache_size,
                    "block_decompose": options.block_decompose,
                    "sparse_expm": options.sparse_expm,
                    "parallel": options.parallel,
                    "profile": options.profile,
                    "sparse_threshold": options.sparse_threshold,
                }
            )
            if options.store_final_state is not None:
                values["store_final_state"] = options.store_final_state
            if options.rf_oversample is not None:
                values["rf_oversample"] = options.rf_oversample
            return values
        return dict(options or {})

    def _default_solver_args(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge explicit solver args with gate-owned decoherence kwargs when available."""
        solver_args = dict(args or {})
        decoherence_params = getattr(self, 'decoherence_params', None)
        if decoherence_params and decoherence_params.get("Tphi2") is not None:
            solver_args.setdefault('Tphi2', decoherence_params["Tphi2"])
        return solver_args

    def _resolve_multidrive_initial_state(
        self,
        initial_state: Union[qt.Qobj, int, List[complex]],
        *,
        allow_gate_parser: bool,
    ) -> qt.Qobj:
        """Resolve an initial state for a multi-drive simulation."""
        if isinstance(initial_state, qt.Qobj):
            return initial_state

        if allow_gate_parser:
            parse_initial_state = getattr(self, '_parse_initial_state', None)
            if callable(parse_initial_state):
                return parse_initial_state(initial_state)

        raise TypeError(
            "initial_state must be a qutip.Qobj when using a custom static_hamiltonian. "
            "Integer and list inputs require the gate-owned qubit Hamiltonian."
        )

    def build_multidrive_hamiltonian(
        self,
        schedules: Union[
            Dict[str, ChannelSchedule],
            List[ChannelSchedule],
            Tuple[ChannelSchedule, ...],
        ],
        drive_operators,
        transmission_chain: Optional[Any] = None,
        mode: Literal['rf', 'complex_envelope'] = 'rf',
        plane: Literal['awg', 'qubit'] = 'qubit',
        channel_order: Optional[Tuple[str, ...]] = None,
        static_hamiltonian: Optional[qt.Qobj] = None,
    ) -> Tuple[list, Dict[str, Any]]:
        """Build a multi-drive Hamiltonian from an aligned schedule bundle."""
        resolved_static = (
            self.qubit.get_hamiltonian()
            if static_hamiltonian is None
            else static_hamiltonian
        )
        drive_funcs = self.awg.get_qutip_bundle_funcs(
            schedules,
            mode=mode,
            chain=transmission_chain,
            plane=plane,
        )
        resolved_order = channel_order
        if resolved_order is None and isinstance(drive_operators, dict):
            resolved_order = tuple(drive_funcs)

        solver = HamiltonianEvo(resolved_static)
        h_total = solver.build_time_dependent_hamiltonian(
            drive_operators=drive_operators,
            drive_funcs=drive_funcs,
            channel_order=resolved_order,
            static_hamiltonian=resolved_static,
        )
        return h_total, drive_funcs

    def run_multidrive_simulation(
        self,
        schedules: Union[
            Dict[str, ChannelSchedule],
            List[ChannelSchedule],
            Tuple[ChannelSchedule, ...],
        ],
        drive_operators,
        initial_state: Union[qt.Qobj, int, List[complex]] = 0,
        transmission_chain: Optional[Any] = None,
        mode: Literal['rf', 'complex_envelope'] = 'rf',
        plane: Literal['awg', 'qubit'] = 'qubit',
        channel_order: Optional[Tuple[str, ...]] = None,
        static_hamiltonian: Optional[qt.Qobj] = None,
        tlist: Optional[np.ndarray] = None,
        c_ops: Optional[list] = None,
        e_ops: Optional[list] = None,
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        backend: str = 'qutip',
    ) -> qt.Result:
        """Run a gate-level simulation driven by a propagated schedule bundle."""
        use_gate_hamiltonian = static_hamiltonian is None
        psi0 = self._resolve_multidrive_initial_state(
            initial_state,
            allow_gate_parser=use_gate_hamiltonian,
        )

        if backend not in {'qutip', 'reference', 'qutip_reference'}:
            resolved_static_hamiltonian = (
                self.qubit.get_hamiltonian()
                if static_hamiltonian is None
                else static_hamiltonian
            )
            trace_bundle = self.awg.get_solver_trace_bundle(
                schedules,
                mode=mode,
                chain=transmission_chain,
                plane=plane,
            )
            drive_funcs = trace_bundle
            if isinstance(drive_operators, dict):
                if not isinstance(trace_bundle, dict):
                    raise TypeError("Mapping drive_operators require mapping schedules.")
                resolved_order = channel_order or tuple(trace_bundle.keys())
            else:
                if isinstance(trace_bundle, dict):
                    raise TypeError("Sequence drive_operators require sequence schedules.")
                resolved_order = channel_order or tuple(range(len(drive_operators)))
            normalized_terms = HamiltonianEvo._normalize_drive_terms(
                drive_operators,
                drive_funcs,
                channel_order=resolved_order,
            )
            terms = [
                DriveTerm(operator, trace, mode=mode)
                for _name, operator, trace in normalized_terms
            ]
            get_c_ops = getattr(self, '_get_c_ops', None)
            resolved_c_ops = (
                get_c_ops()
                if c_ops is None and static_hamiltonian is None and callable(get_c_ops)
                else ([] if c_ops is None else list(c_ops))
            )
            prepared = PreparedPropagation(
                resolved_static_hamiltonian,
                terms,
                self.awg.t_axis if tlist is None else tlist,
                c_ops=resolved_c_ops,
                options=self._default_solver_options(options),
                args=(
                    self._default_solver_args(args)
                    if static_hamiltonian is None
                    else dict(args or {})
                ),
                backend=backend,
            )
            return prepared.propagate(psi0, e_ops=e_ops or [])

        h_total, _ = self.build_multidrive_hamiltonian(
            schedules=schedules,
            drive_operators=drive_operators,
            transmission_chain=transmission_chain,
            mode=mode,
            plane=plane,
            channel_order=channel_order,
            static_hamiltonian=static_hamiltonian,
        )

        if c_ops is None and use_gate_hamiltonian:
            get_c_ops = getattr(self, '_get_c_ops', None)
            resolved_c_ops = get_c_ops() if callable(get_c_ops) else []
        else:
            resolved_c_ops = [] if c_ops is None else c_ops

        resolved_args = (
            self._default_solver_args(args)
            if use_gate_hamiltonian
            else dict(args or {})
        )
        return qt.mesolve(
            h_total,
            psi0,
            self.awg.t_axis if tlist is None else tlist,
            c_ops=resolved_c_ops,
            e_ops=[] if e_ops is None else e_ops,
            options=self._default_solver_options(options, include_internal=False),
            args=resolved_args,
        )

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
        transmission_chain: Optional[TransmissionChain] = None,
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
            transmission_chain: Gate-level default control-line model.
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
        self.transmission_chain = transmission_chain

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

    def _get_inductive_drive_operator(
        self,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
    ) -> qt.Qobj:
        """Return the cached inductive-drive operator in the truncated Hilbert space."""
        cache = getattr(self, '_inductive_drive_operator_cache', None)
        if cache is None:
            cache = {}
            self._inductive_drive_operator_cache = cache

        if induc_phi_model in cache:
            return cache[induc_phi_model]

        phi_op = self.qubit.phi_operators[0]
        if induc_phi_model == 'linear':
            drive_op = phi_op
        elif induc_phi_model == 'exact':
            operator_cache_key = self.qubit._get_hamiltonian_operator_cache_key()
            phi_scales = [
                (2 * self.qubit.Ec[ii, ii] / (self.qubit.Ej[ii, ii] + self.qubit.El[ii, ii])) ** 0.25
                for ii in range(self.qubit._numQubits)
            ]
            phi_scale_key = self.qubit._get_phi_scale_cache_key(phi_scales)
            phis_big, _ = type(self.qubit)._build_cached_scaled_phase_terms(
                *operator_cache_key,
                phi_scale_key,
            )
            drive_big = ((1j * phis_big[0]).expm() - (-1j * phis_big[0]).expm()) / (2j)
            drive_op = truncate_hilbert_space(drive_big, list(self.qubit._Nlevel))
            drive_op = 0.5 * (drive_op + drive_op.dag())
        else:
            raise ValueError(
                f"Unsupported induc_phi_model '{induc_phi_model}'. "
                "Choose from 'exact' or 'linear'."
            )

        cache[induc_phi_model] = drive_op
        return drive_op

    def get_drive_hamiltonian(
        self,
        couple_term: float,
        couple_type: Literal['induc', 'capac'] = 'induc',
        R_line: float = 50.0,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
    ) -> qt.Qobj:
        """
        Construct the drive Hamiltonian for one control line.
        
        Args:
            couple_term(float): Couple strength of control line, H if couple_type is 'induc', F if couple_type is 'capac'.
            couple_type(str): Couple type of control line.
            induc_phi_model(str): Inductive drive operator model, 'exact' for sin(phi)
                as a matrix function or 'linear' for the small-angle phi approximation.
        """
        if couple_type == 'induc':
            drive_op = self._get_inductive_drive_operator(induc_phi_model=induc_phi_model)
            return self.qubit.Ej[0][0] * 2 * pi * couple_term * drive_op / Phi0 / R_line
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
            def coeff_tphi2(t, *callback_args, **callback_kwargs):
                solver_args = callback_kwargs
                if callback_args and isinstance(callback_args[0], dict):
                    solver_args = callback_args[0]
                return np.sqrt(2 * t) / solver_args['Tphi2']
            
            c_ops.append([n, coeff_tphi2])

        return c_ops

    def _parse_initial_state(self, state_input: Union[qt.Qobj, int, List[complex]]) -> qt.Qobj:
        """Helper: Convert index or coeff list to Qobj state vector."""
        if isinstance(state_input, qt.Qobj):
            return state_input
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

    def visualize_signal(
        self,
        channel: Union[ChannelSchedule, None] = None,
        plot_mode: Literal['iq', 'rf'] = 'iq',
        plane: Literal['awg', 'qubit'] = 'qubit',
        transmission_chain: Optional[TransmissionChain] = None,
        capture_history: bool = False,
    ):
        """Visualize AWG or propagated qubit-side electrical waveforms."""
        if channel is None:
            if self.pulse_channel is None:
                raise ValueError(
                    "No channel loaded! Please call .load_channel() first or "
                    "pass 'channel' argument."
                )
            channel = self.pulse_channel

        active_chain = self._resolve_transmission_chain(
            channel,
            transmission_chain=transmission_chain,
        )
        return self.awg.plot_schedule(
            channel,
            plot_mode=plot_mode,
            plane=plane,
            chain=active_chain,
            capture_history=capture_history,
        )

    def _resolve_transmission_chain(
        self,
        channel: ChannelSchedule,
        transmission_chain: Optional[TransmissionChain] = None,
    ) -> Optional[TransmissionChain]:
        """Resolve call, channel, and gate-level chains in descending priority."""
        if transmission_chain is not None:
            return transmission_chain
        channel_chain = getattr(channel, 'transmission_chain', None)
        if channel_chain is not None:
            return channel_chain
        return getattr(self, 'transmission_chain', None)

    def prepare_trace_propagator(
        self,
        trace,
        *,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        mode: Literal['rf', 'complex_envelope'] = 'rf',
        c_ops: Optional[Sequence[Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        backend: str = 'qutip_compiled',
    ) -> PreparedPropagation:
        """Prepare a reusable propagation context for one solver-facing trace.

        The returned object owns the static Hamiltonian, drive operator, trace
        grid, and solver options.  Reusing it for several initial states avoids
        rebuilding ``QobjEvo`` and, for the native backend, keeps all data in
        contiguous arrays across the language boundary.
        """
        trace_domain = getattr(trace, 'domain', None)
        if trace_domain not in {'rf_real', 'iq_complex'}:
            raise ValueError(
                "prepare_trace_propagator expects an rf_real or iq_complex trace."
            )
        H_static = self.qubit.get_hamiltonian()
        H_drive = self.get_drive_hamiltonian(
            couple_term=couple_term,
            couple_type=couple_type,
            induc_phi_model=induc_phi_model,
        )
        resolved_options = self._default_solver_options(options)
        resolved_args = self._default_solver_args(args)
        get_c_ops = getattr(self, '_get_c_ops', None)
        resolved_c_ops = get_c_ops() if c_ops is None and callable(get_c_ops) else (
            [] if c_ops is None else list(c_ops)
        )
        return PreparedPropagation(
            H_static,
            [DriveTerm(H_drive, trace, mode=mode)],
            np.asarray(trace.t_axis, dtype=np.float64),
            c_ops=resolved_c_ops,
            options=resolved_options,
            args=resolved_args,
            backend=backend,
        )

    def prepare_propagator(
        self,
        channel: Optional[ChannelSchedule] = None,
        *,
        transmission_chain: Optional[TransmissionChain] = None,
        mode: Literal['rf', 'complex_envelope'] = 'rf',
        plane: Literal['awg', 'qubit'] = 'qubit',
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        c_ops: Optional[Sequence[Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        backend: str = 'qutip_compiled',
    ) -> PreparedPropagation:
        """Compile a channel and return a reusable prepared propagator."""
        if channel is None:
            if getattr(self, 'pulse_channel', None) is None:
                raise ValueError("No channel loaded! Please pass a ChannelSchedule.")
            channel = self.pulse_channel
        active_chain = self._resolve_transmission_chain(
            channel,
            transmission_chain=transmission_chain,
        )
        trace = self.awg.get_solver_trace(
            channel,
            mode=mode,
            chain=active_chain,
            plane=plane,
        )
        return self.prepare_trace_propagator(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            mode=mode,
            c_ops=c_ops,
            options=options,
            args=args,
            induc_phi_model=induc_phi_model,
            backend=backend,
        )

    def run_simulation(
        self, 
        channel: Union[ChannelSchedule, None] = None,
        initial_state_input: Union[qt.Qobj, int, List[complex]] = 0,
        transmission_chain: Optional[TransmissionChain] = None,
        c_ops: Optional[Sequence[qt.Qobj]] = None,
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

        backend = kwargs.pop('backend', 'qutip')
        mode = kwargs.pop('mode', 'rf')
        plane = kwargs.pop('plane', 'qubit')
        options = kwargs.pop('options', None)
        args = kwargs.pop('args', None)

        psi0 = self._parse_initial_state(initial_state_input)
        c_term = kwargs.get('couple_term', 0.5e-12)
        c_type = kwargs.get('couple_type', 'induc')
        induc_phi_model = kwargs.get('induc_phi_model', 'exact')

        if backend not in {'qutip', 'reference', 'qutip_reference'}:
            prepared = self.prepare_propagator(
                channel,
                transmission_chain=transmission_chain,
                mode=mode,
                plane=plane,
                couple_term=c_term,
                couple_type=c_type,
                c_ops=c_ops,
                options=options,
                args=args,
                induc_phi_model=induc_phi_model,
                backend=backend,
            )
            return prepared.propagate(psi0)

        active_chain = self._resolve_transmission_chain(
            channel,
            transmission_chain=transmission_chain,
        )
        
        H_static = self.qubit.get_hamiltonian()
        H_drive = self.get_drive_hamiltonian(
            couple_term=c_term,
            couple_type=c_type,
            induc_phi_model=induc_phi_model,
        )
        drive_options = {'chain': active_chain}
        if mode != 'rf':
            drive_options['mode'] = mode
        if plane != 'qubit':
            drive_options['plane'] = plane
        drive_func = self.awg.get_qutip_func(channel, **drive_options)
        resolved_c_ops = self._get_c_ops() if c_ops is None else list(c_ops)
        
        H_total = [H_static, [H_drive, drive_func]]
        opts = self._default_solver_options(options, include_internal=False)
        solver_args = self._default_solver_args(args)
        
        result = qt.mesolve(
            H_total, psi0, self.awg.t_axis,
            c_ops=resolved_c_ops, e_ops=[], options=opts, args=solver_args
        )
        return result

    def run_trace_simulation(
        self,
        trace,
        initial_state_input: Union[qt.Qobj, int, List[complex]] = 0,
        *,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        c_ops: Optional[Sequence[qt.Qobj]] = None,
        backend: str = 'qutip',
        mode: Literal['rf', 'complex_envelope'] = 'rf',
    ) -> qt.Result:
        """
        Evolve the qubit directly under one solver-facing waveform trace.

        ``trace`` is duck-typed and must provide ``t_axis``, ``values``, and
        ``domain``. ``rf_real`` traces are interpolated directly; ``iq_complex``
        traces are analytically mixed to RF using ``trace.lo_freq``.
        """
        if backend not in {'qutip', 'reference', 'qutip_reference'}:
            prepared = self.prepare_trace_propagator(
                trace,
                couple_term=couple_term,
                couple_type=couple_type,
                options=options,
                args=args,
                induc_phi_model=induc_phi_model,
                c_ops=c_ops,
                backend=backend,
                mode=mode,
            )
            return prepared.propagate(self._parse_initial_state(initial_state_input))

        H_total, t_axis, resolved_options, resolved_args = self._prepare_trace_simulation(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            options=options,
            args=args,
            induc_phi_model=induc_phi_model,
            mode=mode,
        )
        psi0 = self._parse_initial_state(initial_state_input)
        resolved_c_ops = self._get_c_ops() if c_ops is None else list(c_ops)

        return qt.mesolve(
            H_total,
            psi0,
            t_axis,
            c_ops=resolved_c_ops,
            e_ops=[],
            options=resolved_options,
            args=resolved_args,
        )

    def _prepare_trace_simulation(
        self,
        trace,
        *,
        couple_term: float,
        couple_type: Literal['induc', 'capac'],
        options: Optional[Dict[str, Any]],
        args: Optional[Dict[str, Any]],
        induc_phi_model: Literal['exact', 'linear'],
        mode: Literal['rf', 'complex_envelope'],
    ) -> tuple[list[Any], np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Prepare one trace-dependent solver context reusable across initial states."""
        trace_domain = getattr(trace, 'domain', None)
        if trace_domain not in {'rf_real', 'iq_complex'}:
            raise ValueError(
                "run_trace_simulation expects an rf_real or iq_complex trace. "
                "Use a trace with t_axis, values, domain, and optional lo_freq first."
            )
        if mode not in {'rf', 'complex_envelope'}:
            raise ValueError("mode must be either 'rf' or 'complex_envelope'.")

        H_static = self.qubit.get_hamiltonian()
        H_drive = self.get_drive_hamiltonian(
            couple_term=couple_term,
            couple_type=couple_type,
            induc_phi_model=induc_phi_model,
        )
        if mode == 'complex_envelope':
            drive_func = self.awg.trace_to_qutip_func(trace)
        elif trace_domain == 'iq_complex':
            drive_func = self.awg.trace_to_qutip_rf_func(trace)
        else:
            drive_func = self.awg.trace_to_qutip_func(trace)
        resolved_options = self._default_solver_options(options, include_internal=False)
        resolved_args = self._default_solver_args(args)
        return (
            [H_static, [H_drive, drive_func]],
            np.asarray(trace.t_axis, dtype=np.float64),
            resolved_options,
            resolved_args,
        )

    @staticmethod
    def _result_final_state(result: qt.Result) -> qt.Qobj:
        """Return the final state from full-trajectory or final-state-only results."""
        final_state = getattr(result, 'final_state', None)
        if final_state is not None:
            return final_state
        states = getattr(result, 'states', None)
        if states:
            return states[-1]
        raise ValueError("Solver result does not contain a final state.")

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

    def _computational_basis_states(self) -> Tuple[qt.Qobj, qt.Qobj]:
        """Return the computational eigenstates used by gate-level projections."""
        return (
            self.qubit.get_eigenstate(0),
            self.qubit.get_eigenstate(1),
        )

    def _project_ket_to_qubit_coefficients(
        self,
        state: qt.Qobj,
        basis_states: Tuple[qt.Qobj, qt.Qobj],
    ) -> np.ndarray:
        """Project one ket onto the computational eigenstate subspace."""
        if not state.isket:
            raise ValueError(
                "Evolution-unitary extraction requires ket outputs. "
                "Use coherent evolution without collapse operators."
            )
        return np.asarray(
            [_as_complex_scalar(basis_state.dag() * state) for basis_state in basis_states],
            dtype=complex,
        )

    def _project_operator_to_qubit_subspace(
        self,
        operator: qt.Qobj,
        basis_states: Tuple[qt.Qobj, qt.Qobj],
    ) -> np.ndarray:
        """Project one full-space operator into the computational eigenstate subspace."""
        return np.asarray(
            [
                [_as_complex_scalar(left.dag() * operator * right) for right in basis_states]
                for left in basis_states
            ],
            dtype=complex,
        )

    @staticmethod
    def _strip_global_phase(matrix: np.ndarray, atol: float = 1e-15) -> np.ndarray:
        """Remove the determinant phase from one 2x2 matrix when it is well-defined."""
        det_val = np.linalg.det(matrix)
        if abs(det_val) <= atol:
            return matrix
        return np.exp(-0.5j * np.angle(det_val)) * matrix

    @staticmethod
    def _nearest_unitary(matrix: np.ndarray) -> np.ndarray:
        """Return the polar-decomposition nearest unitary for coherent-error diagnostics."""
        left, _, right_dag = np.linalg.svd(matrix)
        return left @ right_dag

    @staticmethod
    def _clip_fidelity_value(value: float, atol: float = 1e-12) -> Tuple[float, bool]:
        """Clip a fidelity-like scalar to [0, 1] and report nontrivial clipping."""
        raw_value = float(np.real(value))
        clipped_value = min(max(raw_value, 0.0), 1.0)
        was_clipped = raw_value < -atol or raw_value > 1.0 + atol
        return clipped_value, was_clipped

    @staticmethod
    def _ensure_target_has_no_noncomputational_support(
        total_probability: float,
        projected_probability: float,
        *,
        atol: float = 1e-10,
    ) -> None:
        """Reject target states that carry intended population outside |0>, |1>."""
        outside_probability = float(np.real(total_probability - projected_probability))
        if outside_probability > atol:
            raise ValueError(
                "target_state contains non-computational population "
                f"({outside_probability:.3e}). The fidelity target must live entirely "
                "inside the |0>, |1> computational subspace."
            )

    def _summarize_fidelity_metrics(
        self,
        *,
        result: qt.Result,
        target_state: Union[qt.Qobj, List[complex]],
        is_print: bool = True,
    ) -> Dict[str, float]:
        """Project the final state into the qubit eigenbasis before evaluating fidelity."""
        final_state = self._result_final_state(result)
        times = getattr(result, 'times', None)
        if times is None or len(times) == 0:
            raise ValueError("Solver result does not contain a time axis.")
        t_final = times[-1]

        if final_state.isket:
            rho = final_state * final_state.dag()
        else:
            rho = final_state

        eig0, eig1 = self._computational_basis_states()
        pop0 = _as_complex_scalar(eig0.dag() * rho * eig0)
        pop1 = _as_complex_scalar(eig1.dag() * rho * eig1)
        rho01 = _as_complex_scalar(eig0.dag() * rho * eig1)

        phase_factor = 2 * pi * self.qubit.qubit_f01 * t_final
        rho_qubit_rot = qt.Qobj(
            [
                [pop0, rho01 * np.exp(-1j * phase_factor)],
                [np.conj(rho01) * np.exp(1j * phase_factor), pop1],
            ]
        )
        # Numerical integration can leave a tiny negative leakage (for example -4e-16)
        # even when the projected population is exactly one.  Leakage is a probability.
        leakage = max(0.0, float(1.0 - np.real(pop0 + pop1)))

        rho_target = self._resolve_target_qubit_density(target_state)
        fid_val = qt.fidelity(rho_qubit_rot, rho_target)**2

        actual_rho01 = complex(rho_qubit_rot[0, 1])
        target_rho01 = complex(rho_target[0, 1])
        if abs(actual_rho01) <= 1e-12 or abs(target_rho01) <= 1e-12:
            phase_err = 0.0
        else:
            actual_rho01_phase = np.angle(actual_rho01)
            target_rho01_phase = np.angle(target_rho01)
            phase_err = np.degrees(actual_rho01_phase - target_rho01_phase)
            phase_err = (phase_err + 180) % 360 - 180

        if fid_val > 1.0:
            fid_val = 1.0

        if is_print:
            print(f"Fid: {fid_val*100:.5f}%, Leak: {leakage:.5e}, PhaseErr: {phase_err:.2f}")

        return {
            "fidelity": fid_val,
            "leakage": leakage,
            "phase_error_deg": phase_err,
            "final_state_rot": rho_qubit_rot
        }

    def _resolve_target_qubit_density(
        self,
        target_state: Union[qt.Qobj, List[complex]],
    ) -> qt.Qobj:
        """Resolve target states in the 2D computational eigenbasis."""
        if isinstance(target_state, list):
            coeffs = np.asarray(target_state, dtype=complex).reshape(-1)
            if coeffs.size < 2:
                raise ValueError("target_state coefficient lists must provide at least two entries.")
            coeffs = coeffs[:2]
            norm = np.linalg.norm(coeffs)
            if norm <= 0:
                raise ValueError("target_state coefficient list must not be all zeros.")
            coeffs = coeffs / norm
            ket = coeffs[0] * qt.basis(2, 0) + coeffs[1] * qt.basis(2, 1)
            return ket * ket.dag()

        if not isinstance(target_state, qt.Qobj):
            raise TypeError("target_state must be either a qutip.Qobj or a coefficient list.")

        if target_state.isket:
            if target_state.shape == (2, 1):
                ket = target_state.unit()
                return ket * ket.dag()

            eig0, eig1 = self._computational_basis_states()
            coeffs = np.asarray(
                [
                    _as_complex_scalar(eig0.dag() * target_state),
                    _as_complex_scalar(eig1.dag() * target_state),
                ],
                dtype=complex,
            )
            total_probability = float(np.real(_as_complex_scalar(target_state.dag() * target_state)))
            projected_probability = float(np.real(np.vdot(coeffs, coeffs)))
            self._ensure_target_has_no_noncomputational_support(
                total_probability,
                projected_probability,
            )
            norm = np.linalg.norm(coeffs)
            if norm <= 0:
                raise ValueError("target_state has no support in the computational eigenstate subspace.")
            coeffs = coeffs / norm
            ket = coeffs[0] * qt.basis(2, 0) + coeffs[1] * qt.basis(2, 1)
            return ket * ket.dag()

        if target_state.isoper:
            if target_state.shape == (2, 2):
                trace_val = np.trace(target_state.full())
                if abs(trace_val) <= 0:
                    raise ValueError("target_state density matrix must have non-zero trace.")
                return qt.Qobj(target_state.full() / trace_val)

            projected = self._project_operator_to_qubit_subspace(
                target_state,
                self._computational_basis_states(),
            )
            trace_val = np.trace(projected)
            full_trace = np.trace(target_state.full())
            self._ensure_target_has_no_noncomputational_support(full_trace, trace_val)
            if abs(trace_val) <= 0:
                raise ValueError("target_state density has no support in the computational eigenstate subspace.")
            return qt.Qobj(projected / trace_val)

        raise TypeError("target_state must be a ket/density qutip.Qobj or a coefficient list.")

    def _resolve_target_qubit_unitary(
        self,
        target_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None],
        *,
        make_su2: bool = False,
        unitary_atol: float = 1e-8,
    ) -> qt.Qobj:
        """Resolve a target gate into the same 2D computational subspace as the extracted U."""
        if target_unitary is None:
            target_matrix = np.eye(2, dtype=complex)
        elif isinstance(target_unitary, qt.Qobj):
            if not target_unitary.isoper:
                raise TypeError("target_unitary must be an operator qutip.Qobj or a 2x2 matrix.")
            if target_unitary.shape == (2, 2):
                target_matrix = np.asarray(target_unitary.full(), dtype=complex)
            else:
                target_matrix = self._project_operator_to_qubit_subspace(
                    target_unitary,
                    self._computational_basis_states(),
                )
        else:
            target_matrix = np.asarray(target_unitary, dtype=complex)
            if target_matrix.shape != (2, 2):
                raise ValueError("target_unitary array-like inputs must have shape (2, 2).")

        identity = np.eye(2, dtype=complex)
        if not np.allclose(target_matrix.conj().T @ target_matrix, identity, atol=unitary_atol):
            raise ValueError("target_unitary must be unitary within the computational subspace.")

        if make_su2:
            target_matrix = self._strip_global_phase(target_matrix)
        return qt.Qobj(target_matrix, dims=[[2], [2]])

    def _extract_evolution_unitary_payload(
        self,
        *,
        channel: ChannelSchedule = None,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        transmission_chain: Optional[TransmissionChain] = None,
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = True,
        backend: str = 'qutip',
    ) -> Dict[str, Any]:
        """Extract the projected 2x2 process matrix and diagnostics for one pulse schedule."""
        if channel is None:
            if self.pulse_channel is None:
                raise ValueError("No channel loaded! Please call .load_channel() first or pass 'channel' argument.")
            channel = self.pulse_channel

        if frame not in {'rotating', 'lab'}:
            raise ValueError("frame must be either 'rotating' or 'lab'.")

        basis_states = self._computational_basis_states()
        active_chain = self._resolve_transmission_chain(channel, transmission_chain=transmission_chain)

        propagation_options = self._options_mapping(options)
        if not store_trajectories or backend not in {'qutip', 'reference', 'qutip_reference'}:
            propagation_options.update(store_states=False, store_final_state=True)
        prepared = None
        if backend in {'qutip', 'reference', 'qutip_reference'}:
            h_static = self.qubit.get_hamiltonian()
            h_drive = self.get_drive_hamiltonian(
                couple_term=couple_term, couple_type=couple_type,
                induc_phi_model=induc_phi_model,
            )
            drive_func = self.awg.get_qutip_func(channel, chain=active_chain)
            h_total = [h_static, [h_drive, drive_func]]
            solver_options = self._default_solver_options(propagation_options, include_internal=False)
            solver_args = self._default_solver_args(args)
        else:
            prepared = self.prepare_propagator(
                channel,
                transmission_chain=active_chain,
                couple_term=couple_term,
                couple_type=couple_type,
                c_ops=[],
                options=propagation_options,
                args=args,
                induc_phi_model=induc_phi_model,
                backend=backend,
            )

        columns = []
        leakages = []
        results = []
        if prepared is not None and prepared.backend in {'cpp', 'cpp_rwa'}:
            # A unitary is defined by two columns.  Keep both columns in one
            # native integration so coefficient evaluation and matrix assembly
            # happen once instead of once per basis state.
            batch = prepared.propagate_batch(basis_states)
            results = batch.results
            final_states = batch.final_states
        else:
            final_states = []
            for initial_state in basis_states:
                if prepared is None:
                    result = qt.mesolve(
                        h_total, initial_state, self.awg.t_axis,
                        c_ops=[], e_ops=[], options=solver_options, args=solver_args,
                    )
                else:
                    result = prepared.propagate(initial_state)
                results.append(result)
                final_states.append(self._result_final_state(result))

        for final_state in final_states:
            coeffs = self._project_ket_to_qubit_coefficients(final_state, basis_states)
            columns.append(coeffs)
            subspace_probability = float(np.real(np.vdot(coeffs, coeffs)))
            leakages.append(max(0.0, 1.0 - subspace_probability))

        raw_matrix = np.column_stack(columns)
        t_final = self.awg.t_axis[-1] if prepared is None else prepared.tlist[-1]
        if frame == 'rotating':
            phase_factor = 2 * pi * self.qubit.qubit_f01 * t_final
            raw_matrix = np.diag([1.0, np.exp(1j * phase_factor)]) @ raw_matrix

        unitary_matrix = raw_matrix
        if unitarize:
            unitary_matrix = self._nearest_unitary(unitary_matrix)
        if make_su2:
            unitary_matrix = self._strip_global_phase(unitary_matrix)

        raw_unitary = qt.Qobj(raw_matrix, dims=[[2], [2]])
        unitary = qt.Qobj(unitary_matrix, dims=[[2], [2]])
        identity = np.eye(2, dtype=complex)
        gram = raw_matrix.conj().T @ raw_matrix
        survival_probability = float(np.real(np.trace(gram)) / 2.0)
        unitarity_error = float(np.linalg.norm(gram - identity))

        return {
            "unitary": unitary,
            "raw_unitary": raw_unitary,
            "leakage_by_basis": leakages,
            "average_leakage": float(np.mean(leakages)),
            "survival_probability": survival_probability,
            "unitarity_error": unitarity_error,
            "frame": frame,
            "basis_results": results,
        }

    def _extract_trace_unitary_payload(
        self,
        trace,
        *,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = True,
        backend: str = 'qutip',
        sample_times: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Extract a projected 2x2 process matrix from one solver-facing waveform trace."""
        if frame not in {'rotating', 'lab'}:
            raise ValueError("frame must be either 'rotating' or 'lab'.")

        basis_inputs = ([1.0, 0.0], [0.0, 1.0])
        basis_states = self._computational_basis_states()
        columns = []
        leakages = []
        results = []
        solver_options = self._options_mapping(options)
        if not store_trajectories:
            solver_options.update(store_states=False, store_final_state=True)
        elif backend not in {'qutip', 'reference', 'qutip_reference'}:
            solver_options.update(store_states=False, store_final_state=True)

        # Sequence workflows need states at gate boundaries.  Keep the ordinary path on the
        # prepared/native backend, but extend the QuTiP integration grid when boundary samples
        # are requested so the projected matrices refer to the exact requested times.
        boundary_times = None
        trace_t_axis = np.asarray(trace.t_axis, dtype=np.float64)
        solver_t_axis = trace_t_axis
        if sample_times is not None:
            boundary_times = np.asarray(sample_times, dtype=np.float64).reshape(-1)
            if boundary_times.size and (
                not np.all(np.isfinite(boundary_times))
                or not np.all(np.diff(boundary_times) > 0.0)
            ):
                raise ValueError("sample_times must be a strictly increasing finite sequence.")
            if boundary_times.size and (
                boundary_times[0] < trace_t_axis[0]
                or boundary_times[-1] > trace_t_axis[-1]
            ):
                raise ValueError("sample_times must lie inside the trace time axis.")
            solver_t_axis = np.unique(np.concatenate((trace_t_axis, boundary_times)))
            solver_options.update(store_states=True, store_final_state=True)

        prepared = self.prepare_trace_propagator(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            options=solver_options,
            args=args,
            induc_phi_model=induc_phi_model,
            c_ops=[],
            backend=backend,
        )

        if boundary_times is not None and not np.array_equal(solver_t_axis, trace_t_axis):
            if prepared.backend in {'cpp', 'cpp_rwa'}:
                raise UnsupportedBackendError(
                    "sample_times with non-grid boundaries requires a QuTiP backend; "
                    "use backend='qutip' or omit sample_times."
                )
            prepared = PreparedPropagation(
                prepared.static_hamiltonian,
                prepared.drive_terms,
                solver_t_axis,
                c_ops=prepared.c_ops,
                options=prepared.options,
                args=prepared.args,
                backend=backend,
            )

        if prepared.backend in {'cpp', 'cpp_rwa'}:
            batch = prepared.propagate_batch(
                [self._parse_initial_state(item) for item in basis_inputs]
            )
            results = batch.results
            final_states = batch.final_states
        else:
            final_states = []
            for initial_state_input in basis_inputs:
                result = prepared.propagate(self._parse_initial_state(initial_state_input))
                results.append(result)
                final_states.append(self._result_final_state(result))

        for final_state in final_states:
            coeffs = self._project_ket_to_qubit_coefficients(final_state, basis_states)
            columns.append(coeffs)
            subspace_probability = float(np.real(np.vdot(coeffs, coeffs)))
            leakages.append(max(0.0, 1.0 - subspace_probability))

        boundary_raw_unitaries = []
        boundary_unitaries = []
        boundary_leakages = []
        if boundary_times is not None:
            for boundary_time in boundary_times:
                boundary_columns = []
                boundary_leakage_values = []
                for result in results:
                    states = getattr(result, 'states', None)
                    if states is None or len(states) == 0:
                        raise ValueError(
                            "The solver did not return states required for boundary sampling."
                        )
                    index = int(np.argmin(np.abs(np.asarray(result.times) - boundary_time)))
                    if not np.isclose(float(result.times[index]), boundary_time, atol=1e-10, rtol=0.0):
                        raise ValueError("A requested boundary time was not represented in solver output.")
                    coeffs = self._project_ket_to_qubit_coefficients(states[index], basis_states)
                    boundary_columns.append(coeffs)
                    subspace_probability = float(np.real(np.vdot(coeffs, coeffs)))
                    boundary_leakage_values.append(max(0.0, 1.0 - subspace_probability))

                boundary_raw_matrix = np.column_stack(boundary_columns)
                if frame == 'rotating':
                    phase_factor = 2 * pi * self.qubit.qubit_f01 * boundary_time
                    boundary_raw_matrix = np.diag(
                        [1.0, np.exp(1j * phase_factor)]
                    ) @ boundary_raw_matrix
                boundary_matrix = boundary_raw_matrix
                if unitarize:
                    boundary_matrix = self._nearest_unitary(boundary_matrix)
                if make_su2:
                    boundary_matrix = self._strip_global_phase(boundary_matrix)
                boundary_raw_unitaries.append(qt.Qobj(boundary_raw_matrix, dims=[[2], [2]]))
                boundary_unitaries.append(qt.Qobj(boundary_matrix, dims=[[2], [2]]))
                boundary_leakages.append(boundary_leakage_values)

        raw_matrix = np.column_stack(columns)
        t_final = prepared.tlist[-1]
        if frame == 'rotating':
            phase_factor = 2 * pi * self.qubit.qubit_f01 * t_final
            raw_matrix = np.diag([1.0, np.exp(1j * phase_factor)]) @ raw_matrix

        unitary_matrix = raw_matrix
        if unitarize:
            unitary_matrix = self._nearest_unitary(unitary_matrix)
        if make_su2:
            unitary_matrix = self._strip_global_phase(unitary_matrix)

        raw_unitary = qt.Qobj(raw_matrix, dims=[[2], [2]])
        unitary = qt.Qobj(unitary_matrix, dims=[[2], [2]])
        identity = np.eye(2, dtype=complex)
        gram = raw_matrix.conj().T @ raw_matrix
        survival_probability = float(np.real(np.trace(gram)) / 2.0)
        unitarity_error = float(np.linalg.norm(gram - identity))

        payload = {
            "unitary": unitary,
            "raw_unitary": raw_unitary,
            "leakage_by_basis": leakages,
            "average_leakage": float(np.mean(leakages)),
            "survival_probability": survival_probability,
            "unitarity_error": unitarity_error,
            "frame": frame,
            "basis_results": results,
        }
        if boundary_times is not None:
            payload.update(
                boundary_times=boundary_times,
                boundary_unitaries=boundary_unitaries,
                boundary_raw_unitaries=boundary_raw_unitaries,
                boundary_leakage_by_basis=boundary_leakages,
                boundary_average_leakage=[float(np.mean(values)) for values in boundary_leakages],
            )
        return payload

    def extract_trace_channel_payload(
        self,
        trace,
        *,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        frame: Literal['rotating', 'lab'] = 'rotating',
        c_ops: Optional[Sequence[qt.Qobj]] = None,
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        store_trajectories: bool = True,
        backend: str = 'qutip',
    ) -> Dict[str, Any]:
        """Reconstruct the projected computational channel from four physical inputs."""
        if frame not in {'rotating', 'lab'}:
            raise ValueError("frame must be either 'rotating' or 'lab'.")
        basis_states = self._computational_basis_states()
        input_coefficients = (
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0j],
        )
        projected_outputs: list[np.ndarray] = []
        results = []
        solver_options = self._options_mapping(options)
        if not store_trajectories:
            solver_options.update(store_states=False, store_final_state=True)
        elif backend not in {'qutip', 'reference', 'qutip_reference'}:
            solver_options.update(store_states=False, store_final_state=True)
        resolved_c_ops = [] if c_ops is None else list(c_ops)
        prepared = self.prepare_trace_propagator(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            options=solver_options,
            args=args,
            induc_phi_model=induc_phi_model,
            c_ops=resolved_c_ops,
            backend=backend,
        )

        input_states = [self._parse_initial_state(coefficients) for coefficients in input_coefficients]
        if prepared.backend in {'cpp', 'cpp_rwa'}:
            batch = prepared.propagate_batch(input_states)
            results = batch.results
            final_states = batch.final_states
        else:
            final_states = []
            for input_state in input_states:
                result = prepared.propagate(input_state)
                results.append(result)
                final_states.append(self._result_final_state(result))

        for result, final_state in zip(results, final_states):
            final_density = final_state * final_state.dag() if final_state.isket else final_state
            projected = self._project_operator_to_qubit_subspace(final_density, basis_states)
            if frame == 'rotating':
                t_final = float(prepared.tlist[-1])
                rotation = np.diag([1.0, np.exp(2j * pi * self.qubit.qubit_f01 * t_final)])
                projected = rotation @ projected @ rotation.conj().T
            projected_outputs.append(np.asarray(projected, dtype=np.complex128))

        e00, e11, eplus, eplus_i = projected_outputs
        symmetric = 2.0 * eplus - e00 - e11
        antisymmetric = 2.0 * eplus_i - e00 - e11
        e01 = 0.5 * (symmetric + 1j * antisymmetric)
        e10 = 0.5 * (symmetric - 1j * antisymmetric)
        survival_probability = float(np.real(np.trace(e00) + np.trace(e11)) / 2.0)
        return {
            "basis_operator_outputs": ((e00, e01), (e10, e11)),
            "survival_probability": survival_probability,
            "frame": frame,
            "input_results": results,
        }

    def score_trace_channel_payload(
        self,
        payload: Dict[str, Any],
        *,
        target_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None],
    ) -> Dict[str, float]:
        """Score a projected trace-decreasing channel against one target unitary."""
        target = np.asarray(
            self._resolve_target_qubit_unitary(target_unitary).full(),
            dtype=np.complex128,
        )
        outputs = payload["basis_operator_outputs"]
        entanglement_sum = 0.0 + 0.0j
        for row in range(2):
            for column in range(2):
                target_row = target[:, row]
                target_column = target[:, column]
                entanglement_sum += np.vdot(
                    target_row,
                    np.asarray(outputs[row][column], dtype=np.complex128) @ target_column,
                )
        raw_entanglement_fidelity = float(np.real(entanglement_sum) / 4.0)
        raw_survival = float(payload["survival_probability"])
        raw_average_fidelity = float(
            (2.0 * raw_entanglement_fidelity + raw_survival) / 3.0
        )
        entanglement_fidelity = min(max(raw_entanglement_fidelity, 0.0), 1.0)
        survival = min(max(raw_survival, 0.0), 1.0)
        average_fidelity = min(max(raw_average_fidelity, 0.0), 1.0)
        choi = np.block(
            [
                [np.asarray(outputs[0][0]), np.asarray(outputs[0][1])],
                [np.asarray(outputs[1][0]), np.asarray(outputs[1][1])],
            ]
        )
        choi_hermiticity_error = float(np.linalg.norm(choi - choi.conj().T))
        choi_min_eigenvalue = float(
            np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)))
        )
        trace_effect = np.asarray(
            [
                [np.trace(outputs[0][0]), np.trace(outputs[1][0])],
                [np.trace(outputs[0][1]), np.trace(outputs[1][1])],
            ],
            dtype=np.complex128,
        )
        trace_effect_hermiticity_error = float(
            np.linalg.norm(trace_effect - trace_effect.conj().T)
        )
        trace_effect_eigenvalues = np.linalg.eigvalsh(
            0.5 * (trace_effect + trace_effect.conj().T)
        )
        trace_effect_min_eigenvalue = float(np.min(trace_effect_eigenvalues))
        trace_effect_max_eigenvalue = float(np.max(trace_effect_eigenvalues))
        trace_nonincreasing_violation = max(0.0, trace_effect_max_eigenvalue - 1.0)
        physicality_warning = bool(
            choi_hermiticity_error > 1e-8
            or choi_min_eigenvalue < -1e-8
            or trace_effect_hermiticity_error > 1e-8
            or trace_effect_min_eigenvalue < -1e-8
            or trace_nonincreasing_violation > 1e-8
            or raw_survival < -1e-8
            or raw_survival > 1.0 + 1e-8
            or raw_entanglement_fidelity < -1e-8
            or raw_entanglement_fidelity > 1.0 + 1e-8
            or raw_average_fidelity < -1e-8
            or raw_average_fidelity > 1.0 + 1e-8
        )
        return {
            "average_gate_fidelity": average_fidelity,
            "entanglement_fidelity": entanglement_fidelity,
            "survival_probability": survival,
            "average_leakage": max(0.0, 1.0 - survival),
            "raw_average_gate_fidelity": raw_average_fidelity,
            "raw_entanglement_fidelity": raw_entanglement_fidelity,
            "raw_survival_probability": raw_survival,
            "choi_hermiticity_error": choi_hermiticity_error,
            "choi_min_eigenvalue": choi_min_eigenvalue,
            "trace_effect_hermiticity_error": trace_effect_hermiticity_error,
            "trace_effect_min_eigenvalue": trace_effect_min_eigenvalue,
            "trace_effect_max_eigenvalue": trace_effect_max_eigenvalue,
            "trace_nonincreasing_violation": trace_nonincreasing_violation,
            "physicality_warning": physicality_warning,
        }

    def calculate_trace_channel_fidelity(
        self,
        trace,
        *,
        target_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None],
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        frame: Literal['rotating', 'lab'] = 'rotating',
        c_ops: Optional[Sequence[qt.Qobj]] = None,
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        store_trajectories: bool = True,
        backend: str = 'qutip',
    ) -> Dict[str, Any]:
        """Reconstruct and score a dissipative trace channel."""
        payload = self.extract_trace_channel_payload(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            frame=frame,
            c_ops=c_ops,
            options=options,
            args=args,
            induc_phi_model=induc_phi_model,
            store_trajectories=store_trajectories,
            backend=backend,
        )
        return {
            **self.score_trace_channel_payload(payload, target_unitary=target_unitary),
            "channel_payload": payload,
        }

    def _score_unitary_fidelity(
        self,
        *,
        actual_unitary: qt.Qobj,
        raw_unitary: qt.Qobj,
        target_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None],
        payload: Optional[Dict[str, Any]],
        frame: Literal['rotating', 'lab'],
        make_su2: bool,
        is_print: bool,
    ) -> Dict[str, Any]:
        """Score one effective 2x2 process matrix against a target unitary."""
        target = self._resolve_target_qubit_unitary(target_unitary, make_su2=make_su2)
        actual_matrix = np.asarray(actual_unitary.full(), dtype=complex)
        target_matrix = np.asarray(target.full(), dtype=complex)

        d = target_matrix.shape[0]
        overlap_operator = target_matrix.conj().T @ actual_matrix
        trace_overlap = np.trace(overlap_operator)
        trace_norm = float(np.real(np.trace(overlap_operator.conj().T @ overlap_operator)))
        raw_process_fidelity = float(abs(trace_overlap) ** 2 / (d ** 2))
        raw_average_gate_fidelity = float(
            (trace_norm + abs(trace_overlap) ** 2) / (d * (d + 1))
        )

        process_fidelity, process_fidelity_clipped = self._clip_fidelity_value(raw_process_fidelity)
        average_gate_fidelity, average_gate_fidelity_clipped = self._clip_fidelity_value(
            raw_average_gate_fidelity
        )

        raw_matrix = np.asarray(raw_unitary.full(), dtype=complex)
        gram = raw_matrix.conj().T @ raw_matrix
        average_leakage = (
            payload["average_leakage"]
            if payload is not None
            else max(0.0, 1.0 - float(np.real(np.trace(gram)) / d))
        )
        unitarity_error = (
            payload["unitarity_error"]
            if payload is not None
            else float(np.linalg.norm(gram - np.eye(d, dtype=complex)))
        )

        if is_print:
            clip_note = (
                ", clipped"
                if process_fidelity_clipped or average_gate_fidelity_clipped
                else ""
            )
            print(
                f"Favg: {average_gate_fidelity*100:.5f}%, "
                f"Fpro: {process_fidelity*100:.5f}%, "
                f"Leak: {average_leakage:.5e}, "
                f"UnitaryErr: {unitarity_error:.5e}"
                f"{clip_note}"
            )

        observed_backends = set()
        if payload is not None:
            for result in payload.get("basis_results", ()):
                stats = getattr(result, "stats", None)
                if not isinstance(stats, dict):
                    continue
                if stats.get("backend"):
                    observed_backends.add(str(stats["backend"]))
                if stats.get("backend_fallback"):
                    observed_backends.add(str(stats["backend_fallback"]))

        return {
            "fidelity": average_gate_fidelity,
            "average_gate_fidelity": average_gate_fidelity,
            "process_fidelity": process_fidelity,
            "raw_average_gate_fidelity": raw_average_gate_fidelity,
            "raw_process_fidelity": raw_process_fidelity,
            "average_gate_fidelity_clipped": average_gate_fidelity_clipped,
            "process_fidelity_clipped": process_fidelity_clipped,
            "trace_overlap": trace_overlap,
            "average_leakage": average_leakage,
            "unitarity_error": unitarity_error,
            "unitary": actual_unitary,
            "raw_unitary": raw_unitary,
            "target_unitary": target,
            "frame": frame,
            "propagation_backends": tuple(sorted(observed_backends)),
        }

    def extract_evolution_unitary(
        self,
        channel: ChannelSchedule = None,
        *,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        transmission_chain: Optional[TransmissionChain] = None,
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = True,
        backend: str = 'qutip',
    ) -> qt.Qobj:
        """
        Extract the 2x2 evolution matrix projected onto the computational eigenstate subspace.

        The default ``frame='rotating'`` removes the idle ``|1>`` precession so the returned
        matrix can be compared directly with common single-qubit target gates such as X, Y, or X90.
        Collapse operators are intentionally not used here because a single U matrix only describes
        the coherent evolution.
        """
        payload = self._extract_evolution_unitary_payload(
            channel=channel,
            couple_term=couple_term,
            couple_type=couple_type,
            transmission_chain=transmission_chain,
            frame=frame,
            options=options,
            args=args,
            induc_phi_model=induc_phi_model,
            unitarize=unitarize,
            make_su2=make_su2,
            store_trajectories=store_trajectories,
            backend=backend,
        )
        return payload["unitary"]

    def extract_trace_unitary(
        self,
        trace,
        *,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = True,
        backend: str = 'qutip',
    ) -> qt.Qobj:
        """
        Extract the 2x2 computational-subspace process matrix from a drive trace.

        ``trace`` is a sampled solver-facing waveform, not a matrix trace. This is the trace
        analogue of ``extract_evolution_unitary(...)`` for workflows that already synthesized
        the qubit-plane drive outside ``ChannelSchedule``.
        """
        payload = self._extract_trace_unitary_payload(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            frame=frame,
            options=options,
            args=args,
            induc_phi_model=induc_phi_model,
            unitarize=unitarize,
            make_su2=make_su2,
            store_trajectories=store_trajectories,
            backend=backend,
        )
        return payload["unitary"]

    def calculate_unitary_fidelity(
        self,
        target_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None] = None,
        channel: ChannelSchedule = None,
        *,
        process_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None] = None,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        transmission_chain: Optional[TransmissionChain] = None,
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = True,
        is_print: bool = True,
        backend: str = 'qutip',
    ) -> Dict[str, Any]:
        """
        Compute gate-level fidelity against a target unitary in the 2D computational subspace.

        If ``process_unitary`` is not supplied, the method propagates the computational basis
        states ``|0>`` and ``|1>`` without collapse operators, projects both final kets back
        onto the computational eigenstate subspace, and assembles the resulting columns into
        an effective 2x2 process matrix ``A``. The default ``frame='rotating'`` removes the
        idle ``|1>`` precession before comparison.

        The reported ``process_fidelity`` is ``|Tr(U_target^dag A)|^2 / d^2`` and
        ``average_gate_fidelity`` is ``(Tr(A^dag A) + |Tr(U_target^dag A)|^2) / (d(d+1))``.
        With the default ``unitarize=False``, leakage remains as loss of norm in ``A`` and
        lowers the fidelity. Setting ``unitarize=True`` replaces ``A`` by its nearest unitary,
        which is useful for coherent-error diagnostics but no longer includes amplitude loss
        in the fidelity value.

        If ``process_unitary`` is supplied by the caller, it is assumed to already be the
        intended 2x2 effective process matrix. In that path ``average_leakage`` is inferred
        from ``Tr(A^dag A)`` only, so externally computed leakage should be tracked by the
        caller if the supplied matrix has already been normalized or unitarized.

        The compatibility fields ``process_fidelity`` and ``average_gate_fidelity`` are clipped
        into ``[0, 1]``. The unclipped values are also returned as
        ``raw_process_fidelity`` and ``raw_average_gate_fidelity``; clipping usually indicates
        a non-physical effective matrix, gain in the supplied process matrix, or numerical
        error that should be checked together with ``unitarity_error``.
        """
        payload = None
        if process_unitary is None:
            payload = self._extract_evolution_unitary_payload(
                channel=channel,
                couple_term=couple_term,
                couple_type=couple_type,
                transmission_chain=transmission_chain,
                frame=frame,
                options=options,
                args=args,
                induc_phi_model=induc_phi_model,
                unitarize=unitarize,
                make_su2=make_su2,
                store_trajectories=store_trajectories,
                backend=backend,
            )
            actual_unitary = payload["unitary"]
            raw_unitary = payload["raw_unitary"]
        else:
            actual_matrix = np.asarray(
                process_unitary.full() if isinstance(process_unitary, qt.Qobj) else process_unitary,
                dtype=complex,
            )
            if actual_matrix.shape != (2, 2):
                raise ValueError("process_unitary must have shape (2, 2).")
            if unitarize:
                actual_matrix = self._nearest_unitary(actual_matrix)
            if make_su2:
                actual_matrix = self._strip_global_phase(actual_matrix)
            actual_unitary = qt.Qobj(actual_matrix, dims=[[2], [2]])
            raw_unitary = actual_unitary

        return self._score_unitary_fidelity(
            actual_unitary=actual_unitary,
            raw_unitary=raw_unitary,
            target_unitary=target_unitary,
            payload=payload,
            frame=frame,
            make_su2=make_su2,
            is_print=is_print,
        )

    def calculate_trace_unitary_fidelity(
        self,
        trace,
        target_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None] = None,
        *,
        process_unitary: Union[qt.Qobj, np.ndarray, List[List[complex]], None] = None,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = True,
        is_print: bool = True,
        backend: str = 'qutip',
    ) -> Dict[str, Any]:
        """
        Compute gate-level unitary fidelity for one already-synthesized drive trace.

        ``trace`` is a sampled waveform time trace, not a matrix trace. When
        ``process_unitary`` is not supplied, the method propagates ``|0>`` and ``|1>`` through
        ``run_trace_simulation(...)``, projects both final kets into the computational
        subspace, and compares the resulting effective 2x2 process matrix with
        ``target_unitary``. Like ``calculate_unitary_fidelity(...)``, this path is coherent
        and does not include collapse operators in the extracted process matrix.
        """
        payload = None
        if process_unitary is None:
            payload = self._extract_trace_unitary_payload(
                trace,
                couple_term=couple_term,
                couple_type=couple_type,
                frame=frame,
                options=options,
                args=args,
                induc_phi_model=induc_phi_model,
                unitarize=unitarize,
                make_su2=make_su2,
                store_trajectories=store_trajectories,
                backend=backend,
            )
            actual_unitary = payload["unitary"]
            raw_unitary = payload["raw_unitary"]
        else:
            actual_matrix = np.asarray(
                process_unitary.full() if isinstance(process_unitary, qt.Qobj) else process_unitary,
                dtype=complex,
            )
            if actual_matrix.shape != (2, 2):
                raise ValueError("process_unitary must have shape (2, 2).")
            if unitarize:
                actual_matrix = self._nearest_unitary(actual_matrix)
            if make_su2:
                actual_matrix = self._strip_global_phase(actual_matrix)
            actual_unitary = qt.Qobj(actual_matrix, dims=[[2], [2]])
            raw_unitary = actual_unitary

        return self._score_unitary_fidelity(
            actual_unitary=actual_unitary,
            raw_unitary=raw_unitary,
            target_unitary=target_unitary,
            payload=payload,
            frame=frame,
            make_su2=make_su2,
            is_print=is_print,
        )

    def calculate_trace_sequence_unitary_fidelity(
        self,
        trace,
        target_unitaries,
        *,
        boundary_times: Optional[Sequence[float]] = None,
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        frame: Literal['rotating', 'lab'] = 'rotating',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        unitarize: bool = False,
        make_su2: bool = False,
        store_trajectories: bool = False,
        max_prefix_condition_number: float = 1e8,
        is_print: bool = False,
        backend: str = 'qutip',
    ) -> Dict[str, Any]:
        """Score a short sequence of the same primitive gate family.

        ``target_unitaries`` contains one ideal 2x2 gate per pulse, for example
        ``R_phi(pi/2)`` with different ``phi`` values.  The complete waveform is propagated
        once (two computational-basis states), so residual tails and filter memory between
        neighbouring pulses are retained.  When ``boundary_times`` are supplied, the method
        also reports prefix and conditional gate fidelities at those boundaries.  The latter
        uses ``A_j @ pinv(A_{j-1})`` (implemented as a least-squares solve) and is marked
        ``ill_conditioned`` when the previous projected prefix is nearly singular because of
        leakage.  This is a finite-memory diagnostic, not a long-sequence RB decay fit.
        """
        if isinstance(target_unitaries, qt.Qobj):
            target_items = [target_unitaries]
        else:
            target_array = np.asarray(target_unitaries, dtype=object)
            if target_array.shape == (2, 2):
                target_items = [target_unitaries]
            elif target_array.ndim == 3 and target_array.shape[1:] == (2, 2):
                target_items = [target_array[index] for index in range(target_array.shape[0])]
            else:
                target_items = list(target_unitaries)
        if not target_items:
            raise ValueError("target_unitaries must contain at least one 2x2 unitary.")
        if not np.isfinite(max_prefix_condition_number) or max_prefix_condition_number <= 0.0:
            raise ValueError("max_prefix_condition_number must be a positive finite number.")

        targets = [
            self._resolve_target_qubit_unitary(item, make_su2=make_su2)
            for item in target_items
        ]
        target_matrices = [np.asarray(target.full(), dtype=np.complex128) for target in targets]
        target_sequence_matrix = np.eye(2, dtype=np.complex128)
        for target_matrix in target_matrices:
            target_sequence_matrix = target_matrix @ target_sequence_matrix

        payload = self._extract_trace_unitary_payload(
            trace,
            couple_term=couple_term,
            couple_type=couple_type,
            frame=frame,
            options=options,
            args=args,
            induc_phi_model=induc_phi_model,
            unitarize=unitarize,
            make_su2=make_su2,
            store_trajectories=store_trajectories,
            sample_times=boundary_times,
            backend=backend,
        )
        sequence_score = self._score_unitary_fidelity(
            actual_unitary=payload["unitary"],
            raw_unitary=payload["raw_unitary"],
            target_unitary=target_sequence_matrix,
            payload=payload,
            frame=frame,
            make_su2=make_su2,
            is_print=is_print,
        )
        result = {
            **sequence_score,
            "sequence_length": len(targets),
            "sequence_unitary": payload["unitary"],
            "target_sequence_unitary": qt.Qobj(target_sequence_matrix, dims=[[2], [2]]),
            "sequence_payload": payload,
            "boundary_metrics": [],
        }

        if boundary_times is None:
            return result
        boundary_values = np.asarray(boundary_times, dtype=np.float64).reshape(-1)
        if len(boundary_values) != len(targets):
            raise ValueError(
                "boundary_times must have exactly one entry per target unitary."
            )

        previous_actual = np.eye(2, dtype=np.complex128)
        previous_ideal = np.eye(2, dtype=np.complex128)
        boundary_metrics = []
        for index, target in enumerate(targets):
            raw_prefix = np.asarray(
                payload["boundary_raw_unitaries"][index].full(), dtype=np.complex128
            )
            prefix = np.asarray(
                payload["boundary_unitaries"][index].full(), dtype=np.complex128
            )
            condition_number = float(np.linalg.cond(previous_actual))
            conditional_status = "ok"
            conditional_score = None
            if not np.isfinite(condition_number) or condition_number > max_prefix_condition_number:
                conditional_status = "ill_conditioned"
            else:
                conditional_raw = np.linalg.lstsq(
                    previous_actual.T,
                    raw_prefix.T,
                    rcond=None,
                )[0].T
                conditional_matrix = conditional_raw
                if unitarize:
                    conditional_matrix = self._nearest_unitary(conditional_matrix)
                if make_su2:
                    conditional_matrix = self._strip_global_phase(conditional_matrix)
                conditional_score = self._score_unitary_fidelity(
                    actual_unitary=qt.Qobj(conditional_matrix, dims=[[2], [2]]),
                    raw_unitary=qt.Qobj(conditional_raw, dims=[[2], [2]]),
                    target_unitary=target,
                    payload=None,
                    frame=frame,
                    make_su2=make_su2,
                    is_print=False,
                )

            ideal_prefix = target_matrices[index] @ previous_ideal
            prefix_score = self._score_unitary_fidelity(
                actual_unitary=qt.Qobj(prefix, dims=[[2], [2]]),
                raw_unitary=qt.Qobj(raw_prefix, dims=[[2], [2]]),
                target_unitary=ideal_prefix,
                payload={
                    "average_leakage": payload["boundary_average_leakage"][index],
                    "unitarity_error": float(
                        np.linalg.norm(raw_prefix.conj().T @ raw_prefix - np.eye(2))
                    ),
                },
                frame=frame,
                make_su2=make_su2,
                is_print=False,
            )
            boundary_metrics.append(
                {
                    "index": index,
                    "time": float(boundary_values[index]),
                    "conditional_status": conditional_status,
                    "conditional_condition_number": condition_number,
                    "conditional_average_gate_fidelity": (
                        None
                        if conditional_score is None
                        else conditional_score["average_gate_fidelity"]
                    ),
                    "conditional_process_fidelity": (
                        None
                        if conditional_score is None
                        else conditional_score["process_fidelity"]
                    ),
                    "conditional_average_leakage": (
                        None
                        if conditional_score is None
                        else conditional_score["average_leakage"]
                    ),
                    "prefix_average_gate_fidelity": prefix_score["average_gate_fidelity"],
                    "prefix_process_fidelity": prefix_score["process_fidelity"],
                    "prefix_average_leakage": prefix_score["average_leakage"],
                }
            )
            previous_actual = raw_prefix
            previous_ideal = ideal_prefix

        result["boundary_metrics"] = boundary_metrics
        return result

    def calculate_fidelity(
        self, 
        channel: ChannelSchedule = None, 
        target_state: Union[qt.Qobj, List[complex]] = qt.basis(2,1), 
        couple_term: float = 0.5e-12,  # 10 fH (typical mutual inductance)
        couple_type: Literal['induc', 'capac'] = 'induc',
        initial_state_input: Union[int, List[complex]] = 0,
        result: qt.Result = None,
        is_print: bool = True,
        transmission_chain: Optional[TransmissionChain] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        backend: str = 'qutip',
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute single-input-state fidelity, leakage, and phase error for one schedule.

        This is a state-transfer metric, not a full gate fidelity. The method evolves one
        ``initial_state_input`` through ``run_simulation(...)`` using the provided
        ``ChannelSchedule`` and any decoherence previously loaded through
        ``load_decoherence(...)``. The final state is projected onto the computational
        eigenstate subspace, rotated into the default qubit rotating frame, and compared with
        ``target_state``.

        The projected final state is not renormalized by ``pop0 + pop1``. Therefore leakage
        out of the computational subspace lowers the returned ``fidelity`` and is also
        reported separately as ``leakage``. A full-space ``target_state`` is accepted only
        when it has no population outside the computational subspace; otherwise a
        ``ValueError`` is raised instead of silently discarding target leakage.
        """
        if channel is None:
            if self.pulse_channel is None:
                raise ValueError("No channel loaded! Please call .load_channel() first or pass 'channel' argument.")
            channel = self.pulse_channel
        if result is None:
            res = self.run_simulation(
                channel=channel,
                initial_state_input=initial_state_input,
                transmission_chain=transmission_chain,
                couple_term=couple_term,
                couple_type=couple_type,
                induc_phi_model=induc_phi_model,
                backend=backend,
                options=options,
                args=args,
            )
        else:
            res = result
        return self._summarize_fidelity_metrics(
            result=res,
            target_state=target_state,
            is_print=is_print,
        )

    def calculate_trace_fidelity(
        self,
        trace,
        *,
        target_state: Union[qt.Qobj, List[complex]] = qt.basis(2, 1),
        couple_term: float = 0.5e-12,
        couple_type: Literal['induc', 'capac'] = 'induc',
        initial_state_input: Union[int, List[complex]] = 0,
        result: qt.Result = None,
        is_print: bool = True,
        options: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        induc_phi_model: Literal['exact', 'linear'] = 'exact',
        backend: str = 'qutip',
    ) -> Dict[str, float]:
        """Compute state-transfer fidelity for an already propagated drive trace."""
        if result is None:
            result = self.run_trace_simulation(
                trace,
                initial_state_input=initial_state_input,
                couple_term=couple_term,
                couple_type=couple_type,
                options=options,
                args=args,
                induc_phi_model=induc_phi_model,
                **({'backend': backend} if backend != 'qutip' else {}),
            )

        return self._summarize_fidelity_metrics(
            result=result,
            target_state=target_state,
            is_print=is_print,
        )

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
            curr_schedule = self.pulse_channel.clone_with(events=unite_events)
            
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
            go, make_subplots = _load_plotly_helpers()
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
