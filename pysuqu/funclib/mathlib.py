'''
Lib for math utility functions.

 Author: Zhou Naibin
 USTC
 Since 2025-04-13
'''
import inspect
from functools import lru_cache

import numpy as np
from scipy.constants import hbar, pi, Boltzmann
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from typing import Callable, Optional, Tuple

kb = Boltzmann


def _fit_cache_token(values: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    return array.dtype.str, array.shape, array.tobytes()


def _fit_cache_array(token: tuple[str, tuple[int, ...], bytes]) -> np.ndarray:
    dtype_str, shape, payload = token
    return np.frombuffer(payload, dtype=np.dtype(dtype_str)).reshape(shape).copy()


def _fit_param_count(decay_func: Callable, p0: Optional[np.ndarray]) -> int:
    if p0 is not None:
        return len(p0)

    signature = inspect.signature(decay_func)
    return max(len(signature.parameters) - 1, 0)


def _normalize_fit_bounds(
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    *,
    param_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        lower_array = np.full(param_count, -np.inf, dtype=float)
        upper_array = np.full(param_count, np.inf, dtype=float)
    else:
        lower_array = np.broadcast_to(np.asarray(bounds[0], dtype=float), (param_count,)).copy()
        upper_array = np.broadcast_to(np.asarray(bounds[1], dtype=float), (param_count,)).copy()

    return lower_array, upper_array


@lru_cache(maxsize=32)
def _fit_decay_cached(
    decay_func: Callable,
    t_token: tuple[str, tuple[int, ...], bytes],
    p1_token: tuple[str, tuple[int, ...], bytes],
    p0_token: Optional[tuple[str, tuple[int, ...], bytes]],
    lower_token: tuple[str, tuple[int, ...], bytes],
    upper_token: tuple[str, tuple[int, ...], bytes],
) -> tuple[
    tuple[str, tuple[int, ...], bytes],
    tuple[str, tuple[int, ...], bytes],
]:
    t_array = _fit_cache_array(t_token)
    p1_array = _fit_cache_array(p1_token)
    p0_array = None if p0_token is None else _fit_cache_array(p0_token)
    lower_array = _fit_cache_array(lower_token)
    upper_array = _fit_cache_array(upper_token)

    popt, pcov = curve_fit(
        decay_func,
        t_array,
        p1_array,
        p0=p0_array,
        bounds=(lower_array, upper_array),
    )
    return _fit_cache_token(popt), _fit_cache_token(pcov)

def fft_analysis(
    t_total: np.ndarray, 
    signal: np.ndarray,
    is_db: bool = False,
    is_positivefreq: bool = True
) -> tuple:
    """
    Perform FFT analysis on time-domain signal
    
    Args:
        t_total (np.ndarray): Time array [ns]
        signal (np.ndarray): Time-domain signal [V]
        is_db (bool): Convert to dB scale (normalized to peak)
        is_positivefreq (bool): Return only positive frequencies
    
    Returns:
        tuple: (frequency_axis, spectrum)
        
    Spectrum unit:
        - Linear scale: V·ns (V·1e-9 s)
        - dB scale: relative to maximum amplitude (0 dB)
    """
    t_total = np.asarray(t_total)
    signal = np.asarray(signal)

    if t_total.ndim != 1 or signal.ndim != 1:
        raise ValueError("t_total and signal must be 1D arrays.")
    if len(t_total) != len(signal):
        raise ValueError("t_total and signal must have the same length.")
    if len(t_total) < 2:
        raise ValueError("fft_analysis requires at least 2 samples.")

    dt = t_total[1] - t_total[0]
    N = len(signal)
    
    spectrum = np.fft.fft(signal) * dt
    freq_axis = np.fft.fftfreq(N, dt) 
    
    if is_db:
        norm_spectrum = np.abs(spectrum) / np.max(np.abs(spectrum))
        processed_spectrum = 20 * np.log10(norm_spectrum + 1e-16)
    else:
        processed_spectrum = np.abs(spectrum)
    
    if is_positivefreq:
        mask = freq_axis >= 0
        return freq_axis[mask], processed_spectrum[mask]
    else:
        return freq_axis, processed_spectrum

def thermal_photon(
    temperature: float,
    frequency: float,
    return_zero_if_negligible: bool = False
) -> float:
    """
    Calculate thermal photon number using Bose-Einstein distribution
    
    Args:
        temperature (float): Temperature in kelvin (K)
        frequency (float): Frequency in gigahertz (GHz)
        return_zero_if_negligible (bool): Return 0 when n_th < 1e-6 to avoid floating errors
        
    Returns:
        float: Thermal photon number (n_th)
    
    Notes:
        This helper expects temperature in kelvin and converts the supplied
        frequency from GHz to Hz before evaluating the Bose-Einstein formula.
        
    Mathematical Formulation:
        n_th = 1 / [exp(ħω/(k_B T)) - 1]
        where ω = 2πf (angular frequency)
    """
    omega = 2 * np.pi * frequency * 1e9
    
    x = hbar * omega / (Boltzmann * temperature)
    
    if x > 700:
        return 0.0
    
    n_th = 1.0 / (np.exp(x) - 1.0)
    
    if return_zero_if_negligible and n_th < 1e-6:
        return 0.0
    
    return n_th

def nbar2temp(nbar: float, ff: float = 6.7e9) -> float:
    r"""
    Converts average photon number (n_bar) to effective temperature.
    Based on Bose-Einstein statistics.

    Math:
        T = \frac{\hbar \omega}{k_B \ln(1 + 1/\bar{n})}

    Args:
        nbar (float): Average photon number.
        ff (float): Frequency [Hz]. Default 6.7 GHz.

    Returns:
        float: Temperature [K].
    """
    return hbar * 2 * np.pi * ff / kb / np.log(1 + 1 / nbar)

def temp2nbar(T: float, ff: float = 6.7e9) -> float:
    r"""
    Converts temperature to average photon number (n_bar).

    Math:
        \bar{n} = \frac{1}{e^{\hbar \omega / k_B T} - 1}

    Args:
        T (float): Temperature [K].
        ff (float): Frequency [Hz]. Default 6.7 GHz.

    Returns:
        float: Average photon number.
    """
    return 1 / (np.exp(hbar * 2 * np.pi * ff / T / kb) - 1)

def smooth_data(data, method='savgol', window=11, polyorder=3, sigma=2):
    """
    Smooth the input data using the specified method.

    Args:
        y (np.ndarray): The original y-axis data points.
        method (str): The smoothing algorithm to use. Options are
            'savgol' (recommended), 'moving_avg', or 'gaussian'.
        window (int): The size of the smoothing window (must be an odd integer).
        polyorder (int): The polynomial order for the Savitzky-Golay filter.
        sigma (float): The standard deviation for the Gaussian filter kernel.
    """
    data = np.asarray(data)

    if method == 'savgol':
        if data.ndim != 1:
            raise ValueError("smooth_data expects a 1D array.")
        if data.size == 0:
            raise ValueError("smooth_data requires at least 1 sample.")
        if window % 2 == 0:
            window += 1
        max_window = data.size if data.size % 2 == 1 else data.size - 1
        if max_window < 1:
            raise ValueError("smooth_data requires at least 1 sample.")
        window = min(window, max_window)
        if window <= polyorder:
            raise ValueError("savgol window must be greater than polyorder.")
        return savgol_filter(data, window_length=window, polyorder=polyorder)

    if method == 'moving_avg':
        return uniform_filter1d(data, size=window)

    if method == 'gaussian':
        return gaussian_filter1d(data, sigma=sigma)

    raise ValueError("Unknown method. Choose 'savgol', 'moving_avg', or 'gaussian'.")
    
def inverse_func(x, a, b):
    return a / x + b

def find_knee_point(x, y):
    """
    Locate the knee point using the geometric distance method.

    Principle: On a normalized curve, identify the point with the maximum
    perpendicular distance to the line segment connecting the start and end points.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("find_knee_point requires at least 2 samples.")
    if np.allclose(x, x[0]):
        raise ValueError("x must vary to locate a knee point.")

    x_norm = (x - x.min()) / (x.max() - x.min())
    if np.allclose(y, y[0]):
        y_norm = np.zeros_like(y, dtype=float)
    else:
        y_norm = (y - y.min()) / (y.max() - y.min())
    
    start_point = np.array([x_norm[0], y_norm[0]])
    end_point = np.array([x_norm[-1], y_norm[-1]])
    vec_line = end_point - start_point
    
    vec_points = np.stack([x_norm - start_point[0], y_norm - start_point[1]], axis=1)
    
    cross_product = vec_points[:, 0] * vec_line[1] - vec_points[:, 1] * vec_line[0]
    distances = np.abs(cross_product) / np.linalg.norm(vec_line)
    knee_idx = np.argmax(distances)
    
    return x[knee_idx], knee_idx

def ramsey_transfunc(freq: np.ndarray, tau: float) -> np.ndarray:
    """
    Calculate transfer function of Ramsey experiment.
    """
    return tau*np.sinc(2*pi*freq*tau/2)

def echo_transfunc(freq: np.ndarray, tau: float) -> np.ndarray:
    """
    Calculate transfer function of SpinEcho experiment.
    """
    return tau*np.sin(2*pi*freq*tau/4)*np.sinc(2*pi*freq*tau/4)

def cpmg_transfunc(freq: np.ndarray, tau: float, N: int, len_pi: float = 30e-9) -> np.ndarray:
    """
    Calculate transfer function of CPMG experiment.
    
    Args:
        freq (np.ndarray): frequency array.
        tau (np.ndarray): characteristic time.
        N (int): number of pi pulse.
        len_pi (float, optional): duration of pi pulse. Defaults to 30e-9.
    """
    delta_place = [(2*n-1)/N for n in range(1,N+1)]
    return np.abs(1+(-1)**N*np.exp(1j*2*pi*freq*tau)+2*np.sum([(-1)**n*np.exp(1j*2*pi*delta_place[n-1]*freq*tau)*np.cos(2*pi*freq*len_pi/2) for n in range(1,N+1)], axis=0))/4/pi/freq/tau

def integrate_square_large_span(x_arr: np.ndarray, y_arr: np.ndarray, z_func: Callable, method: str = 'simpson') -> float:
    """
    Performs numerical integration on discrete data spanning large orders of magnitude.
    Formula: Integral = sum( y * z(x) * dx )

    Parameters:
        x_arr (np.ndarray): The x-axis data. For 'log' method, must be strictly positive.
        y_arr (np.ndarray): The y-axis data corresponding to x_arr.
        z_func (callable): A function that takes x and returns z. Should support vectorized input.
        method (str): 
            - 'simpson': Uses Simpson's rule. Robust for non-uniform grids.
            - 'log': Performs integration in logarithmic space. 
                     Math: int f(x) dx = int (f(x) * x) d(ln(x)).
                     Recommended if x is log-spaced over many decades (e.g., 1e-2 to 1e6).

    Returns:
        float: The calculated integral value.
    """
    x_arr = np.asarray(x_arr)
    y_arr = np.asarray(y_arr)
    
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]
    
    try:
        z_values = z_func(x_sorted) ** 2
    except TypeError:
        z_values = np.array([z_func(val) ** 2 for val in x_sorted])
        
    integrand = y_sorted * z_values
    
    if method == 'simpson':
        return simpson(y=integrand, x=x_sorted)
    elif method == 'log':
        if np.any(x_sorted <= 0):
            raise ValueError("x must be positive for log-space integration.")

        integrand_log = integrand * x_sorted
        x_log = np.log(x_sorted)

        return simpson(y=integrand_log, x=x_log)
    elif method == 'spline':
        spline = CubicSpline(x_sorted, integrand)
        return spline.integrate(x_sorted[0], x_sorted[-1])
    else:
        raise ValueError(f"Method {method} not supported. Choose in ['simpson', 'log', 'spline'].")

def exp_decay(t: np.ndarray, tau: float, A: float = 1, B: float = 0) -> np.ndarray:
    """
    Calculate exponential decay function.
    """
    return A*np.exp(-t/tau)+B

def gaussian_decay(t: np.ndarray, tau: float, A: float = 1, B: float = 0) -> np.ndarray:
    """
    Calculate gaussian decay function.
    """
    return A*np.exp(-(t/tau)**2)+B

def tphi_decay(t: np.ndarray, tau1: float, tau2: float, A: float = 1, B: float = 0) -> np.ndarray:
    """
    Calculate decoherence time decay function.
    """
    return A*np.exp(-t/tau1-(t/tau2)**2)+B

def ramsey_decay(t: np.ndarray, gamma: float, omega0: float, A: float = 1, B: float = 0) -> np.ndarray:
    """
    Calculate Ramsey decay function.
    """
    return A*np.exp(-gamma*t-1j*omega0*t)+B

def lorentzian_decay(t: np.ndarray, tau: float, A: float = 1, B: float = 0) -> np.ndarray:
    """
    Calculate lorentzian decay function.
    """
    return A/(1+(t/tau)**2)+B

def fit_decay(t: np.ndarray, p1: np.ndarray, decay_func: Callable, p0: Optional[np.ndarray] = None, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = (-np.inf, np.inf)) -> np.ndarray:
    """
    Fit decay function to data.
    """
    t_array = np.ascontiguousarray(np.asarray(t, dtype=float))
    p1_array = np.ascontiguousarray(np.asarray(p1, dtype=float))
    p0_array = None if p0 is None else np.ascontiguousarray(np.asarray(p0, dtype=float))
    param_count = _fit_param_count(decay_func, p0_array)
    lower_array, upper_array = _normalize_fit_bounds(bounds, param_count=param_count)

    try:
        popt_token, pcov_token = _fit_decay_cached(
            decay_func,
            _fit_cache_token(t_array),
            _fit_cache_token(p1_array),
            None if p0_array is None else _fit_cache_token(p0_array),
            _fit_cache_token(lower_array),
            _fit_cache_token(upper_array),
        )
        return _fit_cache_array(popt_token), _fit_cache_array(pcov_token)
    except TypeError:
        popt, pcov = curve_fit(
            decay_func,
            t_array,
            p1_array,
            p0=p0_array,
            bounds=(lower_array, upper_array),
        )
        return popt, pcov

def generate_chirp_envelope(
    t_arr: np.ndarray, 
    f_start: float = -0.1, 
    f_end: float = 0.1
) -> np.ndarray:
    """
    Generates a linear chirp (frequency sweep) complex envelope.

    Args:
        t_arr (np.ndarray): Local time array [ns].
        f_start (float): Starting frequency offset [GHz]. Default -100 MHz.
        f_end (float): Ending frequency offset [GHz]. Default +100 MHz.

    Returns:
        np.ndarray: Complex envelope (I + iQ).
    """
    amp = np.ones_like(t_arr)

    duration = t_arr[-1]
    if duration == 0: 
        return amp.astype(complex)
        
    slope = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t_arr + 0.5 * slope * t_arr**2)

    return amp * np.exp(1j * phase)




















######################################################################################
## Functions below are mostly likely not used anymore. Just leave here for backup.  ##
######################################################################################

def generate_envelope(
    total_time: float,
    gate_time: float,
    gate_amp: float,
    sample_rate: float = 2,
    time_start: float = 0,
    mode: str = 'cosine',
    truncate: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Generate a time-localized envelope waveform (supports Gaussian and raised cosine types)

    Args:
        total_time (float): Total duration (nanoseconds) of the output waveform
        gate_time (float): Envelope duration (nanoseconds) - width of the non-zero waveform
        gate_amp (float): Peak amplitude of the envelope (unitless and normalized)
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 2 (2GS/s)
        time_start (float, optional): Start time (nanoseconds) of the envelope relative to the beginning of the total duration. Defaults to 0
        mode (str, optional): Envelope type. 
            Options:
            - 'cosine': Raised cosine window (1-cos format)
            - 'gaussian': Gaussian envelope (with automatic 1% truncation)
            Defaults to 'cosine'
        truncate (bool, optional): Whether to allow automatic truncation of 
            out-of-bound envelopes. Defaults to True
        verbose (bool, optional): Whether to print truncation warnings. 
            Defaults to False

    Raises:
        ValueError: When `time_start` is negative or envelope exceeds 
            `total_time` with truncate=False
        ValueError: When invalid mode is specified

    Returns:
        tuple: (t_total, waveform) where:
            - t_total (np.ndarray): Time array of the full waveform
            - waveform (np.ndarray): Generated envelope waveform
    """
    
    time_end = time_start + gate_time

    if time_end > total_time:
        if not truncate:
            raise ValueError(f"包络结束时间 {time_end}s 超出总时长 {total_time}s，需设置 truncate=True")
        if verbose:
            print(f"警告：包络被截断（原结束时间 {time_end}s，实际截断到 {total_time}s）")
    if time_start < 0:
        if not truncate:
            raise ValueError(f"包络起始时间 {time_start}s < 0，需设置 truncate=True")
        if verbose:
            print(f"警告：包络被截断（原起始时间 {time_start}s，实际截断到 0s）")

    num_enve_samples = int(np.round(gate_time * sample_rate))
    t_envelope = np.linspace(0, gate_time, num_enve_samples, endpoint=False)
    total_samples = int(np.round(total_time * sample_rate))
    t_total = np.linspace(0, total_time, total_samples, endpoint=False)

    if mode == 'gaussian':
        sigma = gate_time / 2 / np.pi
        envelope = gate_amp * np.exp(-((t_envelope - gate_time/2)**2) / (2 * sigma**2))
    elif mode == 'cosine':
        envelope = gate_amp * 0.5 * (1 - np.cos(2 * np.pi * t_envelope / gate_time))
    else:
        raise ValueError("Mode wrong! Now only support mode in ['cosine','gaussian']. ")
    
    waveform = np.zeros_like(t_total)
    start_idx = int(np.round(time_start * sample_rate))
    end_idx = start_idx + num_enve_samples
    
    if start_idx < 0:
        num_trunc =  - start_idx
        start_idx = 0
        envelope = envelope[num_trunc:]
    if end_idx > total_samples:
        end_idx = total_samples
        num_trunc = end_idx - start_idx
        envelope = envelope[:num_trunc]
    
    waveform[start_idx:end_idx] = envelope
    
    return t_total, waveform

def calculate_drivevolt(t:float, args:dict)->float:
    """
    Calculate drive voltage with time-dependent envelope modulation using linear interpolation.

    The drive voltage is computed as the product of a pre-generated envelope function 
    and a sinusoidal carrier wave. Linear interpolation is used to evaluate the envelope 
    at arbitrary time points.

    Args:
        t (float): Time point at which to calculate the voltage (in nanoseconds)
        args (dict): Parameter dictionary containing:
            * omega_d (float): Driving angular frequency (rad/ns)
            * phi0 (float): Initial phase offset of the carrier wave (radians)
            * time_start (float): Start time of the envelope modulation (ns)
            * envelope_I (np.ndarray): Precomputed envelope amplitude array of I component
            * envelope_Q (np.ndarray, optional): Precomputed envelope amplitude array of Q component
            * t_total (np.ndarray): Time array corresponding to the envelope values (ns)

    Returns:
        float: Instantaneous drive voltage value at time t (V)
    
    Implementation Details:
        1. Uses linear interpolation (np.interp) for envelope evaluation
        2. Carrier wave: sin(ω_d*(t - t_start) + φ0)
        3. Handles out-of-bound times by returning 0 (left/right fill)
    """
    omega_d = args['omega_d']
    phi0 = args['phi0']
    time_start = args['time_start']
    t_total = args['t_total']
    
    try:
        envelope_I = args['envelope_I']
    except Exception as e:
        envelope_I = args['envelope']

    try:
        envelope_Q = args['envelope_Q']
    except Exception as e:
        envelope_Q = np.zeros_like(t_total)
    
    envelopeI_val = np.interp(t, t_total, envelope_I, left=0.0, right=0.0)
    envelopeQ_val = np.interp(t, t_total, envelope_Q, left=0.0, right=0.0)
    
    return envelopeI_val * np.sin(omega_d * (t - time_start) + phi0) + envelopeQ_val * np.cos(omega_d * (t - time_start) + phi0)

def drag_envelope(
    t_total: np.ndarray, 
    envelope_I: np.ndarray, 
    anharmonicity: float, 
    drag_alpha: float = 0.5
) -> np.ndarray:
    """
    Generate DRAG compensation envelope for quantum control pulses.

    Implements the DRAG (Derivative Removal by Adiabatic Gate) technique to suppress leakage to higher energy levels. The quadrature component is calculated by taking the time derivative of the in-phase envelope, scaled by the anharmonicity and DRAG coefficient.

    Args:
        t_total (np.ndarray): Time array [ns] corresponding to envelope points
        envelope_I (np.ndarray): In-phase envelope amplitudes
        anharmonicity (float): Qubit anharmonicity Δ = ω₁₂ - ω₀₁ [GHz]
        drag_alpha (float): Dimensionless scaling factor (default: 0.5)

    Returns:
        np.ndarray: Quadrature envelope [V] for DRAG correction

    Mathematical Formulation:
        envelope_Q = α_drag * da/dt / (2πΔ)
        where Δ is in GHz, dt in ns
    """
    # Validate input dimensions
    if len(t_total) != len(envelope_I):
        raise ValueError("Time array and envelope must have same length")
    
    # Compute numerical derivative (handles non-uniform time spacing)
    da_dt = np.gradient(envelope_I, t_total)
    
    # Calculate DRAG envelope with unit conversion:
    # GHz⁻¹ = 1e9 ns, so Δ (GHz) * 1e9 (ns⁻¹) converts to angular frequency
    envelope_Q = drag_alpha * da_dt / anharmonicity
    
    return envelope_Q

def generate_drivevolt(
    total_time: float,
    gate_time: float,
    gate_amp: float,
    sample_rate: float,
    time_start: float = 0.0,
    mode: str = 'cosine',
    omega_d: float = 5.0,
    phi0: float = 0.0,
    is_drag: bool = True, 
    drag_coeff: float = 0.0,
    anharmonicity: float = -0.250
) -> np.ndarray:
    """
    Generate IQ-modulated drive waveform with optional DRAG compensation.

    Args:
        total_time (float): Total waveform duration (ns)
        gate_time (float): Envelope duration (ns)
        gate_amp (float): Peak amplitude of I-envelope (V)
        sample_rate (float): Sampling rate (GS/s)
        time_start (float): Envelope start time (ns)
        mode (str): Envelope type ('cosine', 'gaussian')
        omega_d (float): Driving angular frequency (rad/ns)
        phi0 (float): Carrier phase offset (radians)
        drag_coeff (float): DRAG scaling coefficient (default=0.0)
        anharmonicity (float): Qubit anharmonicity Δ (GHz)(default=-0.250)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - t_total: Time array (ns)
            - envelope_I: In-phase envelope (V)
            - envelope_Q: Quadrature envelope (V)

    Mathematical Formulation:
        envelope_Q = drag_coeff * dI/dt / (2πΔ)  (when anharmonicity > 0)
    """
    # 生成I包络
    t_total, envelope_I = generate_envelope(
        total_time=total_time,
        gate_time=gate_time,
        gate_amp=gate_amp,
        sample_rate=sample_rate,
        time_start=time_start,
        mode=mode,
    )
    if is_drag:
        envelope_Q = drag_envelope(
            t_total=t_total,
            envelope_I=envelope_I,
            anharmonicity=anharmonicity,
            drag_alpha=drag_coeff
        )
    
    return np.array([
        I*np.sin(omega_d*(t-time_start)+phi0) + 
        Q*np.cos(omega_d*(t-time_start)+phi0) 
        for t, I, Q in zip(t_total, envelope_I, envelope_Q)
    ])


