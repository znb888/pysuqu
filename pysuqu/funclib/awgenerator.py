"""
Lib for arbitrary wave generator. 

"""

import os
import csv
import sys
import importlib.util
import numpy as np
from scipy.signal import windows, convolve
from scipy.special import i0, i1
from scipy.interpolate import interp1d
from scipy.io import wavfile
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple, Callable, Optional, Union, List
from pathlib import Path


def _load_plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotly is required for awgenerator plotting helpers"
        ) from exc
    return go


def _load_transmission_module():
    """Load transmission even when this file is imported outside its package."""
    try:
        from . import transmission as transmission_module
        return transmission_module
    except ImportError:
        module_name = "_pysuqu_funclib_transmission_fallback"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached

        module_path = Path(__file__).with_name("transmission.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load transmission module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


@dataclass
class _PerChannelBundleStage:
    """Dispatch one single-trace stage per bundle channel."""

    channel_stages: Dict[str, Any]
    name: str = "per_channel_stage"
    bundle_stage: bool = True

    @property
    def is_lti(self) -> bool:
        return all(
            stage is None or getattr(stage, "is_lti", False)
            for stage in self.channel_stages.values()
        )

    def describe(self) -> str:
        active = {
            name: stage
            for name, stage in self.channel_stages.items()
            if stage is not None
        }
        if not active:
            return f"{self.name}[identity]"

        parts = []
        for channel_name, stage in active.items():
            describe = getattr(stage, "describe", None)
            stage_name = getattr(stage, "name", stage.__class__.__name__)
            stage_desc = str(describe()) if callable(describe) else str(stage_name)
            parts.append(f"{channel_name}:{stage_desc}")
        return f"{self.name}[" + ", ".join(parts) + "]"

    def apply(self, bundle):
        transmission = _load_transmission_module()
        output_traces = {}
        for channel_name in bundle.order:
            trace = bundle[channel_name]
            stage = self.channel_stages.get(channel_name)
            output = trace if stage is None else stage.apply(trace)
            if not isinstance(output, transmission.SignalTrace):
                raise TypeError(
                    f"{self.describe()} expected channel {channel_name!r} "
                    "stage to return a SignalTrace."
                )
            output_traces[channel_name] = output

        return bundle.clone(traces=output_traces, order=bundle.order)

# --- Pulse Configuration ---
@dataclass
class MixerParams:
    """
    Physical IQ Mixer calibration parameters.
    Used to correct hardware imperfections.

    Attributes:
        lo_freq (float): Local Oscillator frequency [GHz].
        gain_ratio (float): Amplitude balance between I and Q ports (alpha). 1.0 = perfect.
        phase_error (float): Phase orthogonality error [rad] (phi). 0.0 = perfect 90 deg.
        lo_leakage_i (float): DC offset to cancel LO leakage on I port [V].
        lo_leakage_q (float): DC offset to cancel LO leakage on Q port [V].
    """
    lo_freq: float = 0.0          # 5 GHz
    gain_ratio: float = 1.0         # I/Q Amplitude imbalance
    phase_error: float = 0.0        # Orthogonality correction
    lo_leakage_i: float = 0.0
    lo_leakage_q: float = 0.0
    
    def __repr__(self):
        return (
            f"<MixerParams lo_freq={self.lo_freq:.2f}GHz, "
            f"gain_ratio={self.gain_ratio}, "
            f"phase_error={self.phase_error}rad, "
            f"lo_leakage_i={self.lo_leakage_i}V, "
            f"lo_leakage_q={self.lo_leakage_q}V>"
        )

@dataclass
class EnvelopeParams:
    """
    Defines the shape of the pulse envelope (Baseband).
    Does NOT contain frequency or absolute time information.

    Attributes:
        name (str): Identifier for this shape.
        duration (float): Length of the pulse [ns].
        peak_amp (float): Maximum amplitude [a.u. or V].
        shape_type (str): 'gaussian', 'cosine', 'square', or 'custom'.
        sigma (Optional[float]): Width parameter for gaussian [ns].
        drag_coeff (float): DRAG correction coefficient. 0.0 = disabled.
        custom_func (Optional[Callable]): Custom envelope function f(t) -> complex.
    """
    name: str = "envelope"
    duration: float = 20.0
    peak_amp: float = 1.0
    shape_type: Literal['gaussian', 'cosine', 'square', 'triangle', 'blackman_harris', 'slepian', 'custom', 'constant'] = 'cosine'
    sigma: Optional[float] = 2.5
    drag_coeff: float = 0.0
    kaiser_beta: float = None
    custom_func: Optional[Union[np.ndarray, Callable]] = None
    
    def __repr__(self):
        return (f"<EnvelopeParams t={self.duration}ns, "
                f"amp={self.peak_amp/1e-3:.1f}uV, "
                f"shape={self.shape_type}>")

@dataclass
class PulseEvent:
    """
    Represents a specific playback event.
    Combines an Envelope + Modulation settings + Start Time.

    Attributes:
        start_time (float): Absolute start time in the sequence [ns].
        envelope (EnvelopeParams): The baseband shape configuration.
        if_freq (float): Intermediate Frequency (NCO) for digital mixing [GHz].
                         Final Freq = MixerParams.lo_freq + this if_freq.
        phase_offset (float): Digital phase rotation [rad].
        frame_change (float): Virtual Z-gate (frame update) applied after this pulse [rad].
    """
    start_time: float
    envelope: EnvelopeParams
    name: str = "base_pulse"
    if_freq: float = 0.0            # Digital modulation freq
    phase_offset: float = 0.0       # Initial phase
    frame_change: float = 0.0       # Virtual Z (post-pulse)

    def __repr__(self):
        return (f"<PulseEvent start_time={self.start_time}ns, "
                f"freq={self.if_freq/1e-3:.1f}MHz, "
                f"shape={self.envelope.name}>")

@dataclass
class ChannelSchedule:
    """
    The Master Container.
    Represents the full waveform for a single DAC channel.
    Implements the 'Adder' logic by containing a list of overlapping events.

    Attributes:
        name (str): Channel name (e.g., "XY_Q1").
        sampling_rate (float): DAC sampling rate [GSa/s].
        mixer_config (MixerParams): Hardware mixer settings associated with this channel.
        events (List[PulseEvent]): List of pulse events to be added together.
        fir_kernel (Optional[np.ndarray]): Legacy FIR correction applied at the AWG.
        transmission_chain (Optional[Any]): Line model applied after AWG compilation.
    """
    name: str = "Channel_default"
    sampling_rate: float = 2.0
    mixer_config: MixerParams = field(default_factory=MixerParams)
    mixer_correction: bool = True
    events: List[PulseEvent] = field(default_factory=list)
    fir_kernel: Optional[np.ndarray] = None
    transmission_chain: Optional[Any] = None

    def add_pulse(self, start_time: float, envelope: EnvelopeParams, 
                  freq: float = 0.0, phase: float = 0.0):
        """Helper to quickly add a pulse event."""
        event = PulseEvent(
            start_time=start_time,
            envelope=envelope,
            if_freq=freq,
            phase_offset=phase
        )
        self.events.append(event)
        self.events.sort(key=lambda x: x.start_time)

    def clone_with(self, **changes):
        """Return a copy while preserving schedule-level metadata by default."""
        field_values = {
            "name": self.name,
            "sampling_rate": self.sampling_rate,
            "mixer_config": self.mixer_config,
            "mixer_correction": self.mixer_correction,
            "events": list(self.events),
            "fir_kernel": (
                None
                if self.fir_kernel is None
                else np.array(self.fir_kernel, copy=True)
            ),
            "transmission_chain": self.transmission_chain,
        }
        field_values.update(changes)
        return ChannelSchedule(**field_values)

    def display(self) -> None:
        """Visual summary of the channel configuration."""
        print(f"\n{'='*15} Channel Schedule: {self.name} {'='*15}")
        print(f"Mixer LO: {self.mixer_config.lo_freq:.3f} GHz | SR: {self.sampling_rate:.2f} GSa/s")
        print(f"Corrections: Gain={self.mixer_config.gain_ratio}, Phase={self.mixer_config.phase_error}")
        if self.fir_kernel is not None and len(self.fir_kernel) > 0:
            print(f"Legacy FIR kernel length: {len(self.fir_kernel)}")
        if self.transmission_chain is not None:
            describe = getattr(self.transmission_chain, 'describe', None)
            if callable(describe):
                print(f"Transmission chain: {describe()}")
            else:
                stage_count = len(getattr(self.transmission_chain, 'stages', []))
                chain_name = getattr(self.transmission_chain, 'name', 'chain')
                print(f"Transmission chain: {chain_name} ({stage_count} stage(s))")
        print("-" * 65)
        print(f"{'Start (ns)':<12} | {'Freq (MHz)':<12} | {'Phase (rad)':<12} | {'Envelope'}")
        print("-" * 65)
        
        if not self.events:
            print("  <No Events>")
        
        for ev in self.events:
            env_desc = f"{ev.envelope.shape_type} ({ev.envelope.duration}ns)"
            print(f"{ev.name} | {ev.start_time:<12.2f} | {ev.if_freq/1e-3:<12.2f} | {ev.phase_offset:<12.3f} | {env_desc}")
        print("=" * 65 + "\n")

def import_waveform(
    time_array: np.ndarray,
    complex_wave: np.ndarray,
    target_type: Literal['envelope', 'event', 'schedule'] = 'schedule',
    name: str = "imported_wave",
    mixer_config: Optional['MixerParams'] = None,
    # Event level params
    if_freq: float = 0.0,
    frame_change: float = 0.0,
    # Schedule level params
    sampling_rate: Optional[float] = None
) -> Union['EnvelopeParams', 'PulseEvent', 'ChannelSchedule']:
    """
    Import an arbitrary complex waveform and convert it to the specified hierarchy level.

    Args:
        time_array (np.ndarray): Time axis [ns]. Must be monotonic.
        complex_wave (np.ndarray): Complex values (I + 1j*Q). Length must match time_array.
        target_type (str): Output format.
            - 'envelope': Returns EnvelopeParams (shape only, t starts at 0).
            - 'event': Returns PulseEvent (includes absolute start time & IF freq).
            - 'schedule': Returns ChannelSchedule (includes Mixer config & sampling rate).
        name (str): Label for the generated object.
        mixer_config (MixerParams): Hardware mixer config. 
                                    Only used if target_type='schedule'.
                                    If None, defaults to Ideal Mixer.
        if_freq (float): Digital IF frequency [GHz] to associate with the event. 
                         (Used if target_type is 'event' or 'schedule').
        frame_change (float): Virtual Z frame update [rad].
        sampling_rate (float): DAC sampling rate [GSa/s]. If None, inferred from time_array.

    Returns:
        The object corresponding to `target_type`.
    """
    if len(time_array) != len(complex_wave):
        raise ValueError(f"Length mismatch: time={len(time_array)}, wave={len(complex_wave)}")

    dt = time_array[1] - time_array[0]
    duration = time_array[-1] - time_array[0] + dt

    start_time_abs = time_array[0]
    t_local_axis = time_array - start_time_abs

    interp_real = interp1d(t_local_axis, np.real(complex_wave), 
                           kind='linear', bounds_error=False, fill_value=0.0)
    interp_imag = interp1d(t_local_axis, np.imag(complex_wave), 
                           kind='linear', bounds_error=False, fill_value=0.0)

    def custom_envelope_func(t: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """Inner interpolation function."""
        return interp_real(t) + 1j * interp_imag(t)

    env_params = EnvelopeParams(
        name=f"{name}_env",
        duration=duration,
        peak_amp=1.0,
        shape_type='custom',
        custom_func=custom_envelope_func,
        drag_coeff=0.0
    )
    
    if target_type == 'envelope':
        return env_params

    event = PulseEvent(
        start_time=start_time_abs,
        envelope=env_params,
        if_freq=if_freq,
        phase_offset=0.0,
        frame_change=frame_change
    )

    if target_type == 'event':
        return event

    if sampling_rate is None:
        sampling_rate = 1.0 / dt
    if mixer_config is None:
        mixer_config = MixerParams() 

    schedule = ChannelSchedule(
        name=name,
        sampling_rate=sampling_rate,
        mixer_config=mixer_config,
        events=[event]
    )
    return schedule


def import_waveform_from_file(
    file_path: str,
    file_type: Optional[Literal['csv', 'wav', 'auto']] = 'auto',
    target_type: Literal['envelope', 'event', 'schedule'] = 'schedule',
    name: Optional[str] = None,
    mixer_config: Optional['MixerParams'] = None,
    if_freq: float = 0.0,
    frame_change: float = 0.0,
    sampling_rate: Optional[float] = None,
    # CSV specific params
    time_column: str = 'time',
    real_column: str = 'real',
    imag_column: str = 'imag',
    time_unit: Literal['ns', 'us', 'ms', 's'] = 'ns',
    # WAV specific params
    wav_channel: int = 0,
    wav_amplitude_scale: float = 1.0,
) -> Union['EnvelopeParams', 'PulseEvent', 'ChannelSchedule']:
    """
    Import waveform from file (CSV or WAV) and convert to QuSim hierarchy object.
    
    Supports:
        - CSV files with time series and I/Q values
        - WAV audio files (real-valued waveforms)
    
    Args:
        file_path (str): Path to the waveform file.
        file_type (str): File format.
            - 'csv': Comma-separated values file.
            - 'wav': Audio waveform file.
            - 'auto': Auto-detect from file extension (default).
        target_type (str): Output format ('envelope', 'event', or 'schedule').
        name (str): Label for the generated object. Default: filename without extension.
        mixer_config (MixerParams): Hardware mixer config (for 'schedule' type).
        if_freq (float): Digital IF frequency [GHz].
        frame_change (float): Virtual Z frame update [rad].
        sampling_rate (float): DAC sampling rate [GSa/s]. If None, inferred from data.
        
        # CSV specific parameters
        time_column (str): Column name for time axis in CSV.
        real_column (str): Column name for real part (I) in CSV.
        imag_column (str): Column name for imaginary part (Q) in CSV.
        time_unit (str): Time unit in CSV file ('ns', 'us', 'ms', 's').
        
        # WAV specific parameters
        wav_channel (int): Audio channel index (0 = left/mono).
        wav_amplitude_scale (float): Scale factor for WAV amplitude.
    
    Returns:
        Union[EnvelopeParams, PulseEvent, ChannelSchedule]: QuSim hierarchy object.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or data is invalid.
    
    Examples:
        # Import CSV with I/Q data
        schedule = import_waveform_from_file('waveform.csv', target_type='schedule')
        
        # Import WAV file
        schedule = import_waveform_from_file('pulse.wav', target_type='schedule')
        
        # Import CSV with custom column names
        schedule = import_waveform_from_file(
            'data.csv',
            time_column='t',
            real_column='I',
            imag_column='Q',
            time_unit='us'
        )
    """
    # Auto-detect file type
    if file_type == 'auto':
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            file_type = 'csv'
        elif ext in ['.wav', '.wave']:
            file_type = 'wav'
        else:
            raise ValueError(f"Cannot auto-detect file type for extension '{ext}'. "
                           f"Please specify file_type='csv' or 'wav'.")
    
    # Set default name
    if name is None:
        name = Path(file_path).stem
    
    # Load data based on file type
    if file_type == 'csv':
        time_array, complex_wave = _load_waveform_csv(
            file_path,
            time_column=time_column,
            real_column=real_column,
            imag_column=imag_column,
            time_unit=time_unit
        )
    elif file_type == 'wav':
        time_array, complex_wave = _load_waveform_wav(
            file_path,
            channel=wav_channel,
            amplitude_scale=wav_amplitude_scale,
            sampling_rate=sampling_rate
        )
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'wav'.")
    
    # Use existing import_waveform function
    return import_waveform(
        time_array=time_array,
        complex_wave=complex_wave,
        target_type=target_type,
        name=name,
        mixer_config=mixer_config,
        if_freq=if_freq,
        frame_change=frame_change,
        sampling_rate=sampling_rate
    )


def _load_waveform_csv(
    file_path: str,
    time_column: str = 'time',
    real_column: str = 'real',
    imag_column: str = 'imag',
    time_unit: Literal['ns', 'us', 'ms', 's'] = 'ns'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load waveform data from CSV file.
    
    Args:
        file_path (str): Path to CSV file.
        time_column (str): Column name for time axis.
        real_column (str): Column name for real part (I).
        imag_column (str): Column name for imaginary part (Q).
        time_unit (str): Time unit ('ns', 'us', 'ms', 's').
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (time_array in ns, complex_wave)
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Time unit conversion to ns
    time_converters = {
        'ns': 1.0,
        'us': 1e3,
        'ms': 1e6,
        's': 1e9
    }
    time_scale = time_converters[time_unit]
    
    time_data = []
    real_data = []
    imag_data = []
    
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        # Try to detect delimiter
        sample = f.read(4096)
        f.seek(0)
        
        # Detect delimiter
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','  # Default to comma
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Validate columns
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no headers: {file_path}")
        
        # Normalize column names (strip whitespace)
        fieldnames = [name.strip().lower() for name in reader.fieldnames]
        time_col = time_column.strip().lower()
        real_col = real_column.strip().lower()
        imag_col = imag_column.strip().lower()
        
        if time_col not in fieldnames:
            raise ValueError(f"Time column '{time_column}' not found. Available: {reader.fieldnames}")
        if real_col not in fieldnames:
            raise ValueError(f"Real column '{real_column}' not found. Available: {reader.fieldnames}")
        if imag_col not in fieldnames:
            raise ValueError(f"Imag column '{imag_column}' not found. Available: {reader.fieldnames}")
        
        # Map to actual column names (case-insensitive)
        col_map = {name.strip().lower(): name for name in reader.fieldnames}
        
        for row in reader:
            time_data.append(float(row[col_map[time_col]]) * time_scale)
            real_data.append(float(row[col_map[real_col]]))
            imag_data.append(float(row[col_map[imag_col]]))
    
    time_array = np.array(time_data)
    complex_wave = np.array(real_data) + 1j * np.array(imag_data)
    
    # Validate data
    if len(time_array) == 0:
        raise ValueError(f"CSV file contains no data rows: {file_path}")
    
    if len(time_array) != len(complex_wave):
        raise ValueError(f"Length mismatch in CSV data: time={len(time_array)}, wave={len(complex_wave)}")
    
    # Check monotonic time
    if not np.all(np.diff(time_array) >= 0):
        raise ValueError(f"Time array must be monotonically increasing: {file_path}")
    
    return time_array, complex_wave


def _load_waveform_wav(
    file_path: str,
    channel: int = 0,
    amplitude_scale: float = 1.0,
    sampling_rate: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load waveform data from WAV audio file.
    
    WAV files contain real-valued audio data. The function converts this to
    a complex waveform with Q=0 (real-only signal).
    
    Args:
        file_path (str): Path to WAV file.
        channel (int): Audio channel index (0 = left/mono).
        amplitude_scale (float): Scale factor for amplitude.
        sampling_rate (float): Expected sampling rate [GSa/s]. If None, uses WAV file's rate.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (time_array in ns, complex_wave)
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ImportError: If scipy.io.wavfile is not available.
    """
    # wavfile already imported at module level
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"WAV file not found: {file_path}")
    
    # Read WAV file
    sample_rate, audio_data = wavfile.read(file_path)
    
    # Convert sample rate from Hz to GSa/s
    sample_rate_ghz = sample_rate / 1e9  # Hz -> GSa/s
    
    if sampling_rate is None:
        sampling_rate = sample_rate_ghz
    
    # Handle multi-channel audio
    if audio_data.ndim > 1:
        if channel >= audio_data.shape[1]:
            raise ValueError(f"Channel {channel} out of range. WAV file has {audio_data.shape[1]} channels.")
        audio_data = audio_data[:, channel]
    
    # Convert to float and normalize
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float64) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float64) / 2147483648.0
    elif audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_data = audio_data.astype(np.float64)
    else:
        audio_data = audio_data.astype(np.float64)
    
    # Apply amplitude scale
    audio_data = audio_data * amplitude_scale
    
    # Create time axis in ns
    num_samples = len(audio_data)
    duration_ns = num_samples / sampling_rate  # ns
    time_array = np.linspace(0, duration_ns, num_samples, endpoint=False)
    
    # Convert to complex (real signal, Q=0)
    complex_wave = audio_data.astype(np.complex128)
    
    return time_array, complex_wave

# --- Waveform Generator ---
class WaveformGenerator:
    def __init__(self, total_time: float, sample_rate: float, anharmonicity: float = -0.25):
        """
        Waveform Generator for hierarchical Pulse/Schedule architecture.

        Args:
            total_time (float): Total simulation duration [ns].
            sample_rate (float): DAC Sampling rate [GSa/s] (e.g., 2.0).
            anharmonicity (float): Default anharmonicity for DRAG [GHz].
        """
        self.total_time = total_time
        self.sample_rate = sample_rate
        self.anharmonicity = anharmonicity
        
        self.num_samples = int(np.round(total_time * sample_rate))
        self.t_axis = np.linspace(0, total_time, self.num_samples, endpoint=False)
        self.dt = 1.0 / sample_rate

    def _validate_schedule_sampling_rate(self, schedule: 'ChannelSchedule') -> None:
        if schedule.sampling_rate is None:
            return
        if not np.isclose(schedule.sampling_rate, self.sample_rate):
            raise ValueError(
                f"ChannelSchedule sampling_rate ({schedule.sampling_rate} GSa/s) "
                f"does not match WaveformGenerator sample_rate ({self.sample_rate} GSa/s)."
            )

    @staticmethod
    def _resolve_transmission_chain(
        schedule: 'ChannelSchedule',
        chain: Optional[Any] = None,
    ) -> Optional[Any]:
        return chain if chain is not None else schedule.transmission_chain

    @staticmethod
    def _normalize_schedule_collection(
        schedules: Union[
            Dict[str, 'ChannelSchedule'],
            List['ChannelSchedule'],
            Tuple['ChannelSchedule', ...],
        ],
    ) -> Dict[str, 'ChannelSchedule']:
        if isinstance(schedules, dict):
            schedule_map = dict(schedules)
        else:
            schedule_map = {}
            for index, schedule in enumerate(schedules):
                if not isinstance(schedule, ChannelSchedule):
                    raise TypeError("Schedule collections must contain ChannelSchedule objects.")
                name = schedule.name or f"channel_{index}"
                if name in schedule_map:
                    raise ValueError(
                        f"Duplicate channel name {name!r} in schedule collection. "
                        "Pass a dict to disambiguate channels explicitly."
                    )
                schedule_map[name] = schedule

        if not schedule_map:
            raise ValueError("At least one ChannelSchedule is required.")
        for name, schedule in schedule_map.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Bundle channel names must be non-empty strings.")
            if not isinstance(schedule, ChannelSchedule):
                raise TypeError("Schedule collections must contain ChannelSchedule objects.")
        return schedule_map

    @staticmethod
    def _coerce_bundle_chain(chain: Any):
        transmission = _load_transmission_module()
        if chain is None:
            return None
        if isinstance(chain, transmission.BundleTransmissionChain):
            return chain
        if isinstance(chain, transmission.TransmissionChain):
            return transmission.BundleTransmissionChain(
                name=getattr(chain, 'name', 'bundle_chain'),
                stages=list(chain.stages),
            )
        if hasattr(chain, 'stages'):
            return transmission.BundleTransmissionChain(
                name=getattr(chain, 'name', 'bundle_chain'),
                stages=list(chain.stages),
            )
        return transmission.BundleTransmissionChain(stages=[chain])

    @staticmethod
    def _expand_single_trace_chain_stages(chain: Any) -> List[Any]:
        transmission = _load_transmission_module()
        if chain is None:
            return []
        if isinstance(chain, transmission.BundleTransmissionChain):
            raise TypeError(
                "ChannelSchedule.transmission_chain must be a single-trace stage or "
                "TransmissionChain when auto-resolving a bundle. Pass MIMO chains "
                "explicitly via generate_qubit_bundle(..., chain=...)."
            )
        if isinstance(chain, transmission.TransmissionChain):
            return list(chain.stages)
        if hasattr(chain, 'stages'):
            stages = list(chain.stages)
            if any(getattr(stage, 'bundle_stage', False) for stage in stages):
                raise TypeError(
                    "ChannelSchedule.transmission_chain contains bundle stages. "
                    "Pass MIMO chains explicitly via generate_qubit_bundle(..., chain=...)."
                )
            return stages
        if getattr(chain, 'bundle_stage', False):
            raise TypeError(
                "ChannelSchedule.transmission_chain cannot be a bundle-only stage. "
                "Pass it explicitly via generate_qubit_bundle(..., chain=...)."
            )
        return [chain]

    def _resolve_bundle_chain(
        self,
        schedule_map: Dict[str, 'ChannelSchedule'],
        chain: Optional[Any] = None,
    ):
        transmission = _load_transmission_module()
        if chain is not None:
            return self._coerce_bundle_chain(chain)

        per_channel_stages = {
            name: self._expand_single_trace_chain_stages(schedule.transmission_chain)
            for name, schedule in schedule_map.items()
        }
        stage_count = max(
            (len(stages) for stages in per_channel_stages.values()),
            default=0,
        )
        if stage_count == 0:
            return None

        derived_stages = []
        for stage_index in range(stage_count):
            channel_stages = {
                name: stages[stage_index] if stage_index < len(stages) else None
                for name, stages in per_channel_stages.items()
            }
            derived_stages.append(
                _PerChannelBundleStage(
                    channel_stages=channel_stages,
                    name=f"per_channel_stage_{stage_index + 1}",
                )
            )

        return transmission.BundleTransmissionChain(
            name="auto_schedule_bundle_chain",
            stages=derived_stages,
        )

    def _build_signal_trace(
        self,
        values: np.ndarray,
        domain: Literal['iq_complex', 'rf_real'],
        plane: str,
        schedule: 'ChannelSchedule',
        label_suffix: str,
    ):
        transmission = _load_transmission_module()
        return transmission.SignalTrace(
            t_axis=self.t_axis.copy(),
            values=np.array(values, copy=True),
            sample_rate=self.sample_rate,
            domain=domain,
            plane=plane,
            lo_freq=schedule.mixer_config.lo_freq,
            label=f"{schedule.name}_{label_suffix}",
            metadata={"schedule_name": schedule.name},
        )

    @staticmethod
    def _apply_transmission(chain: Any, payload: Any, capture_history: bool):
        try:
            return chain.apply(payload, capture_history=capture_history)
        except TypeError as exc:
            message = str(exc).lower()
            unsupported_keyword = (
                "capture_history" in message
                and ("unexpected keyword" in message or "keyword argument" in message)
            )
            if not unsupported_keyword:
                raise
            output = chain.apply(payload)
            if not capture_history:
                return output

            transmission = _load_transmission_module()
            if isinstance(payload, transmission.SignalTrace):
                if not isinstance(output, transmission.SignalTrace):
                    raise TypeError("Transmission stages must return a SignalTrace.")
                return transmission.TransmissionResult(
                    input_trace=payload,
                    output_trace=output,
                    stage_outputs=[output],
                )
            if isinstance(payload, transmission.SignalBundle):
                if not isinstance(output, transmission.SignalBundle):
                    raise TypeError("Bundle transmission stages must return a SignalBundle.")
                return transmission.BundleTransmissionResult(
                    input_bundle=payload,
                    output_bundle=output,
                    stage_outputs=[output],
                )
            raise TypeError("Unsupported transmission payload type.")

    def _finalize_qubit_trace(
        self,
        trace,
        *,
        schedule: 'ChannelSchedule',
        mode: Literal['iq', 'rf'],
        input_plane: str,
    ):
        transmission = _load_transmission_module()
        if not isinstance(trace, transmission.SignalTrace):
            raise TypeError("Transmission chains must return a SignalTrace.")

        expected_domain = 'iq_complex' if mode == 'iq' else 'rf_real'
        output_plane = 'qubit_iq' if mode == 'iq' else 'qubit_rf'
        if trace.domain != expected_domain:
            raise ValueError(
                f"Transmission chain produced domain {trace.domain} for mode={mode!r}, "
                f"expected {expected_domain}."
            )
        if trace.plane not in (input_plane, output_plane):
            raise ValueError(
                f"Transmission chain produced plane {trace.plane} for mode={mode!r}. "
                f"Expected either {input_plane} or {output_plane}."
            )
        return trace.clone(plane=output_plane, label=f"{schedule.name}_{output_plane}")

    def _finalize_qubit_bundle(
        self,
        bundle,
        *,
        mode: Literal['iq', 'rf'],
        input_plane: str,
    ):
        transmission = _load_transmission_module()
        if not isinstance(bundle, transmission.SignalBundle):
            raise TypeError("Bundle transmission chains must return a SignalBundle.")

        expected_domain = 'iq_complex' if mode == 'iq' else 'rf_real'
        output_plane = 'qubit_iq' if mode == 'iq' else 'qubit_rf'
        if bundle.domain != expected_domain:
            raise ValueError(
                f"Transmission chain produced bundle domain {bundle.domain} for "
                f"mode={mode!r}, expected {expected_domain}."
            )
        if bundle.plane not in (input_plane, output_plane):
            raise ValueError(
                f"Transmission chain produced bundle plane {bundle.plane} for "
                f"mode={mode!r}. Expected either {input_plane} or {output_plane}."
            )

        traces = {
            name: trace.clone(plane=output_plane, label=f"{name}_{output_plane}")
            for name, trace in bundle.traces.items()
        }
        return bundle.clone(
            traces=traces,
            order=bundle.order,
            label=f"{bundle.label}_{output_plane}",
        )

    def _compile_modulated_complex_wave(self, schedule: 'ChannelSchedule') -> np.ndarray:
        self._validate_schedule_sampling_rate(schedule)

        full_complex_wave = np.zeros(self.num_samples, dtype=np.complex128)
        global_phase_accum = 0.0
        for event in sorted(schedule.events, key=lambda item: item.start_time):
            start_idx = int(np.round(event.start_time * self.sample_rate))
            duration_samples = int(np.round(event.envelope.duration * self.sample_rate))
            if start_idx >= self.num_samples:
                continue
            end_idx = min(start_idx + duration_samples, self.num_samples)
            actual_len = end_idx - start_idx
            if actual_len <= 0:
                continue

            t_local = np.linspace(0, actual_len * self.dt, actual_len, endpoint=False)
            baseband_pulse = self._generate_baseband(t_local, event.envelope)
            t_absolute = self.t_axis[start_idx:end_idx]
            phase_term = event.phase_offset + global_phase_accum + (
                2 * np.pi * event.if_freq * t_absolute
            )
            full_complex_wave[start_idx:end_idx] += (
                baseband_pulse * np.exp(1j * phase_term)
            )
            global_phase_accum += event.frame_change

        return full_complex_wave

    def _compile_awg_iq_complex(self, schedule: 'ChannelSchedule') -> np.ndarray:
        full_complex_wave = self._compile_modulated_complex_wave(schedule)
        if schedule.mixer_correction:
            I_dac, Q_dac = self._apply_mixer_correction(
                full_complex_wave,
                schedule.mixer_config,
            )
        else:
            I_dac = np.real(full_complex_wave)
            Q_dac = np.imag(full_complex_wave)

        if schedule.fir_kernel is not None and len(schedule.fir_kernel) > 0:
            I_dac = self._apply_fir_filter(I_dac, schedule.fir_kernel)
            Q_dac = self._apply_fir_filter(Q_dac, schedule.fir_kernel)
        return I_dac + 1j * Q_dac

    def generate_channel_waveform(self, schedule: 'ChannelSchedule',
                                  return_complex: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the final DAC waveforms (I and Q) for a specific Channel Schedule.
        Performs: Envelope Gen -> Modulation -> Superposition -> Mixer Correction.

        Args:
            schedule (ChannelSchedule): The schedule containing events and mixer config.
            return_complex (bool): If True, returns (Complex_Wave, None).
                                   If False, returns (I_dac, Q_dac).
            is_mixercorrect (bool): Whether to correct the mixer error.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (I_array, Q_array) calibrated for the DAC.
        """
        awg_iq = self._compile_awg_iq_complex(schedule)
        if return_complex:
            return awg_iq, None
        return np.real(awg_iq), np.imag(awg_iq)

    def generate_rf_waveform(self, schedule: 'ChannelSchedule') -> np.ndarray:
        """
        Simulate the final RF waveform output by the physical IQ mixer.

        Args:
            schedule (ChannelSchedule): The schedule config to generate.

        Returns:
            np.ndarray: The real-valued RF voltage array.
        """
        I_dac, Q_dac = self.generate_channel_waveform(schedule)

        w_lo = 2 * np.pi * schedule.mixer_config.lo_freq
        rf_wave = I_dac * np.cos(w_lo * self.t_axis) - Q_dac * np.sin(w_lo * self.t_axis)
        
        return rf_wave

    def generate_awg_output(
        self,
        schedule: 'ChannelSchedule',
        mode: Literal['iq', 'rf'] = 'iq',
    ):
        """Return the AWG-side sampled signal as a SignalTrace."""
        if mode == 'iq':
            return self._build_signal_trace(
                self._compile_awg_iq_complex(schedule),
                domain='iq_complex',
                plane='awg_iq',
                schedule=schedule,
                label_suffix='awg_iq',
            )
        if mode == 'rf':
            return self._build_signal_trace(
                self.generate_rf_waveform(schedule),
                domain='rf_real',
                plane='awg_rf',
                schedule=schedule,
                label_suffix='awg_rf',
            )
        raise ValueError(f"Unsupported AWG output mode: {mode}")

    def generate_awg_bundle(
        self,
        schedules: Union[
            Dict[str, 'ChannelSchedule'],
            List['ChannelSchedule'],
            Tuple['ChannelSchedule', ...],
        ],
        mode: Literal['iq', 'rf'] = 'iq',
    ):
        """Return multiple aligned AWG traces as one SignalBundle."""
        transmission = _load_transmission_module()
        schedule_map = self._normalize_schedule_collection(schedules)
        traces = {
            name: self.generate_awg_output(schedule, mode=mode).clone(
                label=f"{name}_awg_{mode}"
            )
            for name, schedule in schedule_map.items()
        }
        return transmission.SignalBundle(
            traces=traces,
            order=tuple(schedule_map),
            label=f"awg_{mode}_bundle",
        )

    def generate_qubit_output(
        self,
        schedule: 'ChannelSchedule',
        chain: Optional[Any] = None,
        mode: Literal['iq', 'rf'] = 'rf',
        capture_history: bool = False,
    ):
        """Return one solver-facing trace after its optional transmission chain."""
        transmission = _load_transmission_module()
        resolved_chain = self._resolve_transmission_chain(schedule, chain)
        awg_trace = self.generate_awg_output(schedule, mode=mode)

        if resolved_chain is None:
            output_trace = self._finalize_qubit_trace(
                awg_trace,
                schedule=schedule,
                mode=mode,
                input_plane=awg_trace.plane,
            )
            if capture_history:
                return transmission.TransmissionResult(
                    input_trace=awg_trace,
                    output_trace=output_trace,
                    stage_outputs=[],
                )
            return output_trace

        chain_input = awg_trace
        try:
            applied = self._apply_transmission(
                resolved_chain,
                chain_input,
                capture_history,
            )
        except ValueError as exc:
            if not (mode == 'rf' and self._chain_requires_iq_fallback(exc)):
                raise
            chain_input = self.generate_awg_output(schedule, mode='iq')
            applied = self._apply_transmission(
                resolved_chain,
                chain_input,
                capture_history,
            )

        if capture_history:
            output_trace = applied.output_trace
            if mode == 'rf' and output_trace.domain == 'iq_complex':
                output_trace = self.convert_iq_trace_to_rf(
                    output_trace,
                    output_plane='qubit_rf',
                )
            output_trace = self._finalize_qubit_trace(
                output_trace,
                schedule=schedule,
                mode=mode,
                input_plane=chain_input.plane,
            )
            stage_outputs = [
                stage_trace.clone(label=f"{schedule.name}_{stage_trace.label}")
                for stage_trace in applied.stage_outputs
            ]
            return transmission.TransmissionResult(
                input_trace=applied.input_trace,
                output_trace=output_trace,
                stage_outputs=stage_outputs,
            )

        output_trace = applied
        if mode == 'rf' and output_trace.domain == 'iq_complex':
            output_trace = self.convert_iq_trace_to_rf(
                output_trace,
                output_plane='qubit_rf',
            )
        return self._finalize_qubit_trace(
            output_trace,
            schedule=schedule,
            mode=mode,
            input_plane=chain_input.plane,
        )

    def generate_qubit_bundle(
        self,
        schedules: Union[
            Dict[str, 'ChannelSchedule'],
            List['ChannelSchedule'],
            Tuple['ChannelSchedule', ...],
        ],
        chain: Optional[Any] = None,
        mode: Literal['iq', 'rf'] = 'rf',
        capture_history: bool = False,
    ):
        """Return multiple solver-facing traces after an optional bundle chain."""
        transmission = _load_transmission_module()
        schedule_map = self._normalize_schedule_collection(schedules)
        resolved_chain = self._resolve_bundle_chain(schedule_map, chain)
        awg_bundle = self.generate_awg_bundle(schedule_map, mode=mode)

        if resolved_chain is None:
            output_bundle = self._finalize_qubit_bundle(
                awg_bundle,
                mode=mode,
                input_plane=awg_bundle.plane,
            )
            if capture_history:
                return transmission.BundleTransmissionResult(
                    input_bundle=awg_bundle,
                    output_bundle=output_bundle,
                    stage_outputs=[],
                )
            return output_bundle

        chain_input = awg_bundle
        try:
            applied = self._apply_transmission(
                resolved_chain,
                chain_input,
                capture_history,
            )
        except ValueError as exc:
            if not (mode == 'rf' and self._chain_requires_iq_fallback(exc)):
                raise
            chain_input = self.generate_awg_bundle(schedule_map, mode='iq')
            applied = self._apply_transmission(
                resolved_chain,
                chain_input,
                capture_history,
            )

        if capture_history:
            output_bundle = applied.output_bundle
            if mode == 'rf' and output_bundle.domain == 'iq_complex':
                output_bundle = self.convert_iq_bundle_to_rf(output_bundle)
            output_bundle = self._finalize_qubit_bundle(
                output_bundle,
                mode=mode,
                input_plane=chain_input.plane,
            )
            return transmission.BundleTransmissionResult(
                input_bundle=applied.input_bundle,
                output_bundle=output_bundle,
                stage_outputs=[
                    stage_bundle.clone(label=f"{stage_bundle.label}_{index}")
                    for index, stage_bundle in enumerate(
                        applied.stage_outputs,
                        start=1,
                    )
                ],
            )

        output_bundle = applied
        if mode == 'rf' and output_bundle.domain == 'iq_complex':
            output_bundle = self.convert_iq_bundle_to_rf(output_bundle)
        return self._finalize_qubit_bundle(
            output_bundle,
            mode=mode,
            input_plane=chain_input.plane,
        )

    @staticmethod
    def _chain_requires_iq_fallback(exc: Exception) -> bool:
        message = str(exc).lower()
        return "expects iq_complex" in message and "received rf_real" in message

    @staticmethod
    def _chain_requires_rf_fallback(exc: Exception) -> bool:
        message = str(exc).lower()
        return "expects rf_real" in message and "received iq_complex" in message

    def _generate_baseband(self, t: np.ndarray, env_params: 'EnvelopeParams') -> np.ndarray:
        """
        Generates the complex baseband envelope (I + i*DRAG).
        """
        if env_params.shape_type == 'gaussian':
            # Default sigma = duration / 4 if not specified
            sigma = env_params.sigma if env_params.sigma else (env_params.duration / 4.0)
            mu = env_params.duration / 2.0
            shape = np.exp(-0.5 * ((t - mu) / sigma)**2)
            
        elif env_params.shape_type == 'cosine':
            # 0.5 * (1 - cos(2pi * t / T))
            shape = 0.5 * (1 - np.cos(2 * np.pi * t / env_params.duration))
            
        elif env_params.shape_type == 'square':
            shape = np.ones_like(t)

        elif env_params.shape_type == 'triangle':
            shape = 1.0 - 2 * np.abs(t - env_params.duration / 2.0) / env_params.duration
            shape = np.clip(shape, 0.0, None)
        
        elif env_params.shape_type == 'blackman_harris':
            if len(t) > 0:
                shape = windows.blackmanharris(len(t))
            else:
                shape = np.array([])
            
        elif env_params.shape_type == 'slepian':
            if len(t) > 0:
                shape = windows.dpss(len(t), NW=env_params.sigma)
            else:
                shape = np.array([])
        
        elif env_params.shape_type == 'custom':
            if callable(env_params.custom_func):
                # Custom function usually returns complex directly
                return env_params.peak_amp * env_params.custom_func(t)
            elif isinstance(env_params.custom_func, (np.ndarray, list)):
                raw_data = np.array(env_params.custom_func)
                # Resample if length mismatches (e.g. sample rate changed)
                if len(raw_data) != len(t):
                    old_x = np.linspace(0, 1, len(raw_data))
                    new_x = np.linspace(0, 1, len(t))
                    # Interp real and imag separately
                    r = np.interp(new_x, old_x, np.real(raw_data))
                    i = np.interp(new_x, old_x, np.imag(raw_data))
                    shape = r + 1j * i
                else:
                    shape = raw_data
            else:
                return np.zeros_like(t, dtype=complex)
        elif env_params.shape_type == 'constant':
            shape = np.ones_like(t)
        else:
            raise ValueError(f"Unknown shape: {env_params.shape_type}")

        kaiser_beta = getattr(env_params, 'kaiser_beta', None)
        if kaiser_beta is not None and kaiser_beta > 0 and len(t) > 0:
            # Generate Kaiser window of the same length
            k_window = np.kaiser(len(t), kaiser_beta)
            shape = shape * k_window
        shape *= env_params.peak_amp

        if env_params.drag_coeff != 0 and env_params.shape_type != 'square':
            deriv = np.gradient(shape, self.dt)
            drag_q = env_params.drag_coeff * deriv / self.anharmonicity
            
            return shape + 1j * drag_q
        
        return shape + 0j

    def _apply_mixer_correction(self, complex_wave: np.ndarray, 
                                mixer: 'MixerParams') -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Inverse Mixer Transformation Matrix to pre-distort the waveform.
        
        Hardware Model:
            I_out = I_in
            Q_out = g * (Q_in * cos(phi) + I_in * sin(phi)) + offsets
            
        We need to invert this to find the DAC values (I_in, Q_in) that produce the Ideal Complex Wave.
        """
        I_ideal = np.real(complex_wave)
        Q_ideal = np.imag(complex_wave)
        
        g = mixer.gain_ratio
        phi = mixer.phase_error
        
        # Correction Matrix (Inverse of the hardware model)
        # To get orthogonal output, we must pre-skew the inputs.
        # Matrix:
        # | I_dac |   | 1            -tan(phi)      | | I_ideal |
        # | Q_dac | = | 0            1/(g*cos(phi)) | | Q_ideal |
        
        # Optimization: pre-calculate factors
        # Note: If phi is small, cos(phi) ~ 1. 
        # But let's use exact trig for robustness.
        
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        
        if abs(c_phi) < 1e-6: c_phi = 1e-6 
        
        m12 = -np.tan(phi)
        m22 = 1.0 / (g * c_phi)
        
        I_dac = I_ideal + m12 * Q_ideal
        Q_dac = m22 * Q_ideal
        
        I_dac += mixer.lo_leakage_i
        Q_dac += mixer.lo_leakage_q
        
        return I_dac, Q_dac

    def _apply_fir_filter(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply FIR filter using convolution with 'full' mode and truncation.
        
        Logic:
            y = x * h (length N + M - 1)
            y_out = y[:N] (Truncate tail to match original time duration)
        
        Args:
            signal: Input signal (1D array).
            kernel: FIR coefficients (Impulse Response).

        Returns:
            Filtered signal of the same length as input.
        """
        if len(kernel) == 0:
            return signal
            
        filtered_full = convolve(signal, kernel, mode='full', method='auto')
        
        return filtered_full[:len(signal)]
    
    def plot_schedule(self, schedule: 'ChannelSchedule', plot_mode: Literal['iq', 'rf'] = 'iq'):
        """
        Visualize the generated Channel Schedule.
        
        Args:
            schedule: The schedule object.
            plot_mode: 'iq' (Envelope) or 'rf' (Modulated Carrier approx).
        """
        I_dac, Q_dac = self.generate_channel_waveform(schedule)
        t = self.t_axis
        go = _load_plotly_graph_objects()

        fig = go.Figure()
        
        if plot_mode == 'iq':
            fig.add_trace(go.Scatter(x=t, y=I_dac, mode='lines', name='I (DAC)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=t, y=Q_dac, mode='lines', name='Q (DAC)', line=dict(color='orange')))
            
            # Compute Magnitude for reference
            mag = np.sqrt(I_dac**2 + Q_dac**2)
            fig.add_trace(go.Scatter(x=t, y=mag, mode='lines', name='Magnitude', 
                                     line=dict(color='black', width=1, dash='dot'), opacity=0.5))
            
            title = f"Control Waveform: {schedule.name} (I/Q)"
            
        elif plot_mode == 'rf':
            w_lo = 2 * np.pi * schedule.mixer_config.lo_freq # rad/ns
            rf_wave = I_dac * np.cos(w_lo * t) - Q_dac * np.sin(w_lo * t) # -sin for standard IQ mixer
            
            fig.add_trace(go.Scatter(x=t, y=rf_wave, mode='lines', name='RF Output', 
                                     line=dict(color='grey', width=1)))
            title = f"Simulated RF Output: {schedule.name} (LO={schedule.mixer_config.lo_freq}G)"

        fig.update_layout(
            title=title,
            xaxis_title="Time (ns)",
            yaxis_title="Amplitude (V)",
            template="plotly_white",
            hovermode="x unified"
        )
        fig.show()
    
    def plot_pulse(self, pulse: 'PulseEvent', plot_mode: Literal['iq', 'rf'] = 'iq'):
        """
        Visualize single Pulse Event.
        """
        channel = ChannelSchedule()
        channel.events.append(pulse)
        self.plot_schedule(channel, plot_mode)
    
    def get_qutip_func(
        self,
        schedule: 'ChannelSchedule',
        mode: Literal['rf', 'complex_envelope'] = 'rf',
        chain: Optional[Any] = None,
        plane: Literal['awg', 'qubit'] = 'qubit',
    ) -> Callable:
        """
        Returns a closure function `func(t, args)` compatible with QuTiP solvers.
        
        Args:
            schedule (ChannelSchedule): The schedule to compile.
            mode (str): 
                - 'rf': Returns the real-valued physical RF signal: I(t)*cos(wt) - Q(t)*sin(wt).
                        Use this for Lab Frame simulations.
                - 'complex_envelope': Returns I(t) + 1j*Q(t). 
                                      Use this for Rotating Wave Approximation (RWA) simulations.
                                      Note: This includes IF modulation, but NOT LO modulation.
        
        Returns:
            Callable[[float, dict], float|complex]: The time-dependent coefficient function.
        """
        if plane not in ('awg', 'qubit'):
            raise ValueError(f"Unsupported waveform plane: {plane}")
        if mode not in ('rf', 'complex_envelope'):
            raise ValueError(f"Unsupported QuTiP drive mode: {mode}")

        if mode == 'complex_envelope':
            trace = (
                self.generate_awg_output(schedule, mode='iq')
                if plane == 'awg'
                else self.generate_qubit_output(schedule, chain=chain, mode='iq')
            )
            return self._trace_to_qutip_func(trace)

        # Mix the carrier at solver query time whenever the chain accepts IQ. This
        # preserves the continuous LO even when the sampled envelope is below the
        # RF Nyquist rate.
        try:
            trace = (
                self.generate_awg_output(schedule, mode='iq')
                if plane == 'awg'
                else self.generate_qubit_output(schedule, chain=chain, mode='iq')
            )
            return self.trace_to_qutip_rf_func(trace)
        except ValueError as exc:
            if plane == 'awg' or not self._chain_requires_rf_fallback(exc):
                raise

        trace = self.generate_qubit_output(schedule, chain=chain, mode='rf')
        return self._trace_to_qutip_func(trace)

    @staticmethod
    def _trace_to_qutip_func(trace) -> Callable:
        """Convert one sampled trace into a QuTiP-compatible interpolation function."""
        t_axis = np.asarray(trace.t_axis, dtype=np.float64)
        values = np.asarray(trace.values)

        if np.iscomplexobj(values):
            real_values = np.asarray(np.real(values), dtype=np.float64)
            imag_values = np.asarray(np.imag(values), dtype=np.float64)

            def qutip_drive_func(t: float, args=None):
                i_val = np.interp(t, t_axis, real_values, left=0.0, right=0.0)
                q_val = np.interp(t, t_axis, imag_values, left=0.0, right=0.0)
                return i_val + 1j * q_val

            return qutip_drive_func

        real_values = np.asarray(values, dtype=np.float64)

        def qutip_drive_func(t: float, args=None):
            return np.interp(t, t_axis, real_values, left=0.0, right=0.0)

        return qutip_drive_func

    def trace_to_qutip_func(self, trace) -> Callable:
        """Public wrapper for converting one sampled trace into a QuTiP drive callable."""
        return self._trace_to_qutip_func(trace)

    def trace_to_qutip_rf_func(self, trace) -> Callable:
        """
        Convert a qubit/AWG trace into a real RF QuTiP callback.

        ``rf_real`` traces are interpolated directly. ``iq_complex`` traces are
        interpolated as envelopes and multiplied by the LO carrier at the solver
        query time, which keeps the carrier continuous even when the AWG grid is
        below the RF Nyquist rate.
        """
        if getattr(trace, 'domain', None) != 'iq_complex':
            return self._trace_to_qutip_func(trace)

        t_axis = np.asarray(trace.t_axis, dtype=np.float64)
        values = np.asarray(trace.values, dtype=np.complex128)
        real_values = np.asarray(np.real(values), dtype=np.float64)
        imag_values = np.asarray(np.imag(values), dtype=np.float64)
        lo_freq = float(getattr(trace, 'lo_freq', 0.0))

        def qutip_drive_func(t: float, args=None):
            i_val = np.interp(t, t_axis, real_values, left=0.0, right=0.0)
            q_val = np.interp(t, t_axis, imag_values, left=0.0, right=0.0)
            carrier_phase = 2 * np.pi * lo_freq * t
            return i_val * np.cos(carrier_phase) - q_val * np.sin(carrier_phase)

        return qutip_drive_func

    def convert_iq_trace_to_rf(
        self,
        trace,
        *,
        lo_freq: Optional[float] = None,
        output_plane: Optional[str] = None,
    ):
        """Mix one IQ SignalTrace onto its LO carrier."""
        transmission = _load_transmission_module()
        if not isinstance(trace, transmission.SignalTrace):
            raise TypeError("convert_iq_trace_to_rf expects a SignalTrace input.")
        if trace.domain != 'iq_complex':
            raise ValueError("convert_iq_trace_to_rf requires an iq_complex SignalTrace.")

        carrier_freq = float(trace.lo_freq if lo_freq is None else lo_freq)
        if output_plane is None:
            plane_map = {'awg_iq': 'awg_rf', 'qubit_iq': 'qubit_rf'}
            try:
                resolved_plane = plane_map[trace.plane]
            except KeyError as exc:
                raise ValueError(
                    "output_plane must be provided when converting an IQ trace "
                    f"from plane {trace.plane!r}."
                ) from exc
        else:
            resolved_plane = output_plane

        carrier_phase = 2 * np.pi * carrier_freq * trace.t_axis
        rf_values = (
            np.real(trace.values) * np.cos(carrier_phase)
            - np.imag(trace.values) * np.sin(carrier_phase)
        )
        return trace.clone(
            values=np.asarray(rf_values, dtype=np.float64),
            domain='rf_real',
            plane=resolved_plane,
            lo_freq=carrier_freq,
            label=f"{trace.label}_rf",
            metadata={**trace.metadata, "converted_from": trace.label},
        )

    def convert_iq_bundle_to_rf(
        self,
        bundle,
        *,
        output_plane: Optional[str] = None,
    ):
        """Mix every trace in one IQ SignalBundle onto its own LO carrier."""
        transmission = _load_transmission_module()
        if not isinstance(bundle, transmission.SignalBundle):
            raise TypeError("convert_iq_bundle_to_rf expects a SignalBundle input.")
        if bundle.domain != 'iq_complex':
            raise ValueError("convert_iq_bundle_to_rf requires an iq_complex SignalBundle.")

        if output_plane is None:
            first_plane = bundle[bundle.order[0]].plane
            plane_map = {'awg_iq': 'awg_rf', 'qubit_iq': 'qubit_rf'}
            try:
                resolved_plane = plane_map[first_plane]
            except KeyError as exc:
                raise ValueError(
                    "output_plane must be provided when converting an IQ bundle "
                    f"from plane {first_plane!r}."
                ) from exc
        else:
            resolved_plane = output_plane

        converted_traces = {
            name: self.convert_iq_trace_to_rf(trace, output_plane=resolved_plane)
            for name, trace in bundle.traces.items()
        }
        return bundle.clone(
            traces=converted_traces,
            order=bundle.order,
            label=f"{bundle.label}_rf",
        )

    def get_qutip_bundle_funcs(
        self,
        schedules: Union[
            Dict[str, 'ChannelSchedule'],
            List['ChannelSchedule'],
            Tuple['ChannelSchedule', ...],
        ],
        mode: Literal['rf', 'complex_envelope'] = 'rf',
        chain: Optional[Any] = None,
        plane: Literal['awg', 'qubit'] = 'qubit',
    ) -> Dict[str, Callable]:
        """Compile a bundle into QuTiP callbacks keyed by output channel."""
        if plane not in ('awg', 'qubit'):
            raise ValueError(f"Unsupported waveform plane: {plane}")
        if mode not in ('rf', 'complex_envelope'):
            raise ValueError(f"Unsupported QuTiP drive mode: {mode}")

        if mode == 'complex_envelope':
            trace_mode = 'iq'
        else:
            try:
                bundle = (
                    self.generate_awg_bundle(schedules, mode='iq')
                    if plane == 'awg'
                    else self.generate_qubit_bundle(schedules, chain=chain, mode='iq')
                )
                return {
                    name: self.trace_to_qutip_rf_func(bundle[name])
                    for name in bundle.order
                }
            except ValueError as exc:
                if plane == 'awg' or not self._chain_requires_rf_fallback(exc):
                    raise
                trace_mode = 'rf'

        bundle = (
            self.generate_awg_bundle(schedules, mode=trace_mode)
            if plane == 'awg'
            else self.generate_qubit_bundle(schedules, chain=chain, mode=trace_mode)
        )
        return {
            name: self._trace_to_qutip_func(bundle[name])
            for name in bundle.order
        }


class WaveformDerivatives:
    """
    Generator for waveforms and their analytical derivatives.
    All methods return a tuple of (waveform, derivative).
    """
    
    @staticmethod
    def blackman_harris(t: np.ndarray, carrier_freq: float, amp_list: List[float] = [0.35875, 0.48829, 0.14128, 0.01168]) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the Blackman-Harris window and its analytical derivative.
        Formula: w(t) = a0 - a1*cos(wt) + a2*cos(2wt) - a3*cos(3wt)
        """
        w = 2 * np.pi * carrier_freq
        theta = w * t
        a0, a1, a2, a3 = amp_list

        wave = a0 - a1*np.cos(theta) + a2*np.cos(2*theta) - a3*np.cos(3*theta)
        dw_dt = (a1*np.sin(theta) - 2*a2*np.sin(2*theta) + 3*a3*np.sin(3*theta)) * w
        
        return wave, dw_dt

    @staticmethod
    def kaiser(t: np.ndarray, carrier_freq: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the Kaiser window and its analytical derivative (based on Bessel functions).
        Core variable: z = 2t/T - 1 (Range: -1 to 1)
        """
        z = 2 * t * carrier_freq - 1
        eps = 1e-16 
        arg_sq = 1 - z**2
        arg_sq = np.maximum(arg_sq, eps) # Ensure non-negative
        arg = np.sqrt(arg_sq)            # sqrt(1 - z^2)
        
        u = beta * arg # Argument for the Bessel function
        
        denom = i0(beta)
        wave = i0(u) / denom
        
        dw_dt = -1.0 * i1(u) * (beta * z) / arg * (2 * carrier_freq) / denom

        dw_dt[arg < 1e-8] = 0.0
        
        return wave, dw_dt

    @staticmethod
    def kaiser_windowed_bh(t: np.ndarray, T: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        [Hybrid Waveform] Kaiser window * Blackman-Harris window.
        Uses the product rule: (uv)' = u'v + uv'
        """
        # Get individual waveforms and derivatives
        u, du = WaveformDerivatives.kaiser(t, T, beta)
        v, dv = WaveformDerivatives.blackman_harris(t, T)
        
        # Combine using product rule
        wave = u * v
        deriv = du * v + u * dv
        
        return wave, deriv


