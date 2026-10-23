"""
Lib for electrical noise model.

Author: Naibin Zhou
since 2025-12-10
"""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import h, hbar, Boltzmann
from typing import List, Literal, Mapping, Optional, Sequence, Tuple, Union

kb = Boltzmann

PSDUnitMode = Literal[
    "dBm/Hz",
    "W/Hz",
    "V^2/Hz",
    "V/sqrt(Hz)",
    "A^2/Hz",
    "A/sqrt(Hz)",
]
PSDInput = Union[float, Sequence[float], NDArray[np.float64]]
PSDOutput = Union[float, NDArray[np.float64]]

def T2Sii_Double(T: Union[float, int, np.ndarray], f: float, R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Calculates the double-sided current noise spectral density (S_II) for a given temperature.
    Based on the Fluctuation-Dissipation Theorem (Callen-Welton).

    Math:
        S_{II}(\omega) = \frac{hf}{2R} (\coth(\frac{hf}{2k_B T}) + 1) / 2  (Structure varies by convention)
    
    Args:
        T (float or np.ndarray): Temperature in Kelvin [K].
        f (float): Frequency in Hertz [Hz].

    Returns:
        Union[float, np.ndarray]: Current noise spectral density [A^2/Hz].
    """
    if isinstance(T, (int, float)):
        if T == 0:
            return 0
        else:
            # Logic: np.coth(x) = 1 / np.tanh(x)
            x = h * f / (2 * kb * T)
            Si = h * f / R * (1 / np.tanh(x) + 1) / 4
            return Si

    if isinstance(T, np.ndarray):
        Si = np.zeros_like(T, dtype=float)
        mask = T != 0
        
        if np.any(mask):
            val_T = T[mask]
            x = h * f / (2 * kb * val_T)
            
            Si[mask] = h * f / R * (1 / np.tanh(x) + 1) / 4
            
        return Si

def Sii2T_Double(S: Union[float, int, np.ndarray], f: float, R: float = 50) -> Union[float, np.ndarray]:
    """
    Inverse function of T2SDouble. Calculates the noise temperature corresponding 
    to a given current noise spectral density.

    Args:
        S (float or np.ndarray): Current noise spectral density [A^2/Hz].
        f (float): Frequency in Hertz [Hz].

    Returns:
        Union[float, np.ndarray]: Noise temperature in Kelvin [K].
    """
    if isinstance(S, (int, float)):
        if S == 0:
            return 0
        else:
            # Inversion of the Bose-Einstein factor derived from S formula
            T = -h * f / kb / np.log(1 - 2 * h * f / 4 / S / R)
            return T
            
    if isinstance(S, np.ndarray):
        T = np.array([
            -h * f / kb / np.log(1 - 2 * h * f / 4 / s / R) 
            if s != 0 else 0 
            for s in S
        ])
        return T

def T2Sii_Single(T: Union[float, int, np.ndarray], f: float, R: float = 50) -> Union[float, np.ndarray]:
    """
    Calculates single-sided spectral density by summing positive and negative frequency components.
    
    Args:
        T (float or np.ndarray): Temperature [K].
        f (float): Frequency [Hz].

    Returns:
        Union[float, np.ndarray]: Single-sided noise spectral density.
    """
    # Sum of S(f) and S(-f) captures quantum asymmetry/vacuum fluctuations
    return T2Sii_Double(T, f, R) + T2Sii_Double(T, -f,  R)

def Sii2T_Single(S: Union[float, int, np.ndarray], f: float, R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Inverse function of T2Sii_Single. Calculates the effective Noise Temperature 
    from the Single-sided Current Noise Spectral Density.
    
    Based on the inverse of the Quantum Nyquist formula:
        S_{single} = (hf / 2R) * coth(hf / 2k_B T)
        => T = hf / (2 * k_B * arctanh(hf / 2RS))
    
    Args:
        S (float or np.ndarray): Single-sided current noise spectral density [A^2/Hz].
        f (float): Frequency [Hz].
        R (float): Load resistance [Ohm]. Default is 50.

    Returns:
        Union[float, np.ndarray]: Noise Temperature [K]. 
                                  Returns 0 if S is below the quantum vacuum limit.
    """

    if f == 0:
        return S * R / kb

    S_vac = h * f / (2 * R)

    if isinstance(S, (int, float)):
        if S <= S_vac:
            return 0.0
        
        val = np.arctanh(S_vac / S)
        T = (h * f) / (2 * kb * val)
        return T

    elif isinstance(S, np.ndarray):
        S = np.array(S, dtype=float)
        T_out = np.zeros_like(S)
        
        valid_mask = S > S_vac
        
        if np.any(valid_mask):
            val = np.arctanh(S_vac / S[valid_mask])
            T_out[valid_mask] = (h * f) / (2 * kb * val)
            
        return T_out

def Sii_D2S(S: Union[float, int, np.ndarray], f: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Convert Double-sided Current Noise Density to Single-sided.
    
    Optimization:
        Uses direct algebraic relationship instead of converting to Temperature first.
        This avoids 'inf'/'nan' errors caused by logarithms when input noise 
        is close to or below the vacuum limit due to data fluctuations.
        
    Math:
        S_{single} = 2 * S_{double} - \frac{hf}{2R}
        (Note: hf/2R is the vacuum noise term that cancels out partially)

    Args:
        S: Double-sided noise spectral density [A^2/Hz]
        f: Frequency [Hz]
        R: Resistance [Ohm]
    """
    S = np.asarray(S)
    f = np.asarray(f)

    vacuum_term = h * f / (2 * R)
    S_single = 2 * S - vacuum_term
    
    return S_single

def Sii_S2D(S: Union[float, int, np.ndarray], f: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Convert Single-sided Current Noise Density to Double-sided Current Noise Density.
    
    This function performs the inverse algebraic operation of Sii_D2S.
    
    Math:
        S_{double} = \frac{1}{2} S_{single} + \frac{hf}{4R}
    
    Args:
        S (Union[float, int, np.ndarray]): Single-sided noise spectral density [A^2/Hz].
        f (Union[float, np.ndarray]): Frequency [Hz].
        R (float): Resistance [Ohm]. Default is 50.

    Returns:
        Union[float, np.ndarray]: Double-sided noise spectral density [A^2/Hz].
    """
    S = np.asarray(S)
    f = np.asarray(f)
    
    vacuum_term_quarter = h * f / (4 * R)
    
    return 0.5 * S + vacuum_term_quarter

def Sii_A2dBm(Sii: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    """
    Converts Current Spectral Density (A^2/Hz) to Power Spectral Density in dBm/Hz.

    Args:
        Sii (float or np.ndarray): Current spectral density [A^2/Hz].

    Returns:
        float or np.ndarray: Power spectral density [dBm/Hz].
    """
    # P = I^2 * R. Convert to mW (1e-3 factor) then to dB.
    return 10 * np.log10(Sii * R / 1e-3)

def Sii_dBm2A(Sii_dBm: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    """
    Converts Power Spectral Density (dBm/Hz) back to Current Spectral Density (A^2/Hz).

    Args:
        Sii_dBm (float or np.ndarray): Power spectral density [dBm/Hz].

    Returns:
        float or np.ndarray: Current spectral density [A^2/Hz].
    """
    # Inverse of 10*log10(P/1mW)
    return 10**(Sii_dBm / 10) * 1e-3 / R

def Sii_dBm2temp(Sii_dBm: Union[float, np.ndarray], f: float = 6.7e9, R: float = 50) -> Union[float, np.ndarray]:
    """
    Converts Noise Power (dBm/Hz) directly to Noise Temperature (K).
    Wrapper combining Sii_dBm2A and Sii2T_Double.

    Args:
        Sii_dBm (float or np.ndarray): Power spectral density [dBm/Hz].
        f (float): Frequency [Hz]. Defaults to 6.7e9 for backward compatibility.
        R (float): Load resistance [Ohm]. Default is 50.
    """
    return Sii2T_Double(Sii_dBm2A(Sii_dBm, R=R), f=f, R=R)

def Svv_V2dBm(Svv: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Converts Voltage Spectral Density (V^2/Hz) to Power Spectral Density in dBm/Hz.
    
    Math:
        P = \frac{V^2}{R} \implies P_{dBm} = 10 \log_{10}\left( \frac{S_{vv}/R}{1\text{mW}} \right)

    Args:
        Svv (float or np.ndarray): Voltage spectral density [V^2/Hz].

    Returns:
        Union[float, np.ndarray]: Power spectral density [dBm/Hz].
    """
    # Convert V^2/Hz to W/Hz (divide by R), then normalize to 1mW (1e-3)
    return 10 * np.log10((Svv / R) / 1e-3)

def Svv_dBm2V(Svv_dBm: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Converts Power Spectral Density (dBm/Hz) back to Voltage Spectral Density (V^2/Hz).

    Math:
        S_{vv} = P_{Linear} \times R

    Args:
        Svv_dBm (float or np.ndarray): Power spectral density [dBm/Hz].

    Returns:
        Union[float, np.ndarray]: Voltage spectral density [V^2/Hz].
    """
    # Convert dBm to mW (10^(x/10)), convert mW to W (1e-3), then convert W to V^2 (multiply by R)
    return 10**(Svv_dBm / 10) * 1e-3 * R

def S_V2I(Svv: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Converts Voltage Spectral Density (V^2/Hz) to Current Spectral Density (A^2/Hz).

    Math:
        S_{ii} = \frac{S_{vv}}{R}

    Args:
        Svv (float or np.ndarray): Voltage spectral density [V^2/Hz].

    Returns:
        Union[float, np.ndarray]: Current spectral density [A^2/Hz].
    """
    return Svv / R**2

def S_I2V(Sii: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    r"""
    Converts Current Spectral Density (A^2/Hz) to Voltage Spectral Density (V^2/Hz).

    Math:
        S_{vv} = S_{ii} \times R

    Args:
        Sii (float or np.ndarray): Current spectral density [A^2/Hz].

    Returns:
        Union[float, np.ndarray]: Voltage spectral density [V^2/Hz].
    """
    return Sii * R**2

def cal_Teff_byS(Splus: float, Sminus: float, ff: float = 6.7e9) -> float:
    r"""
    Calculates effective temperature using the Sideband Asymmetry method.
    Using the ratio of noise at positive and negative frequencies.

    Math:
        \frac{S(\omega)}{S(-\omega)} = e^{\frac{\hbar \omega}{k_B T}}

    Args:
        Splus (float): Noise density at +f.
        Sminus (float): Noise density at -f.
        ff (float): Frequency [Hz].

    Returns:
        float: Effective Temperature [K].
    """
    return hbar * 2 * np.pi * ff / kb / np.log(Splus / Sminus)

def cal_Teff_byST(T: float, ff: float = 6.7e9) -> float:
    """
    Verification helper: Calculates T_eff from a given T to check consistency.
    """
    Splus = T2Sii_Double(T, ff)
    Sminus = T2Sii_Double(T, -ff)
    return cal_Teff_byS(Splus, Sminus, ff)

def S_transmission(S_in: Union[float, np.ndarray], 
                   ff: float = 6.7e9, 
                   T_setup: np.ndarray = np.array([290, 45, 3.5, 0.9, 0.1, 0.01]), 
                   attenuationindB: np.ndarray = np.array([40, 1, 10, 10, 20, 10])) -> Union[float, np.ndarray]:
    """
    Calculates the output noise spectral density (double-sided) after passing through a cascade of attenuators
    at different temperature stages (Friis formula for passive components).

    Model:
        S_{out} = S_{in} * A + S_{thermal} * (1 - A)
        Where A is the power transmission factor (0 < A <= 1). This rule is
        applied to every stage, including the first stage in the chain.

    Args:
        S_in (float or np.ndarray): Input noise spectral density [A^2/Hz].
        ff (float): Frequency [Hz].
        T_setup (np.ndarray): Array of temperatures for each stage [K].
        attenuationindB (np.ndarray): Array of power attenuation values in dB
            for each stage. They are converted with ``A = 10**(-dB / 10)``.

    Returns:
        Union[float, np.ndarray]: Output noise spectral density [A^2/Hz].
    """
    attenuation = 10**(-attenuationindB / 10)

    S_transmission = (
        S_in * attenuation[0]
        + (1 - attenuation[0]) * T2Sii_Double(T_setup[0], ff)
    )

    for ii in range(1, len(T_setup)):
        S_transmission = attenuation[ii] * S_transmission + (1 - attenuation[ii]) * T2Sii_Double(T_setup[ii], ff)
        
    return S_transmission


def find_psd_peaks(
    frequency: Union[Sequence[float], np.ndarray],
    psd,
    *,
    labels: Optional[Sequence[str]] = None,
    is_plot: bool = False,
    ax=None,
    title: Optional[str] = "PSD peak detection",
    frequency_unit: str = "Hz",
    psd_unit: str = "V^2/Hz",
    prominence: float = 3.0,
    distance: int = 8,
    width: int = 1,
    smooth_window: int = 9,
    baseline_window: int = 0,
    show_peak_frequencies: bool = False,
    peak_frequency_unit: Optional[str] = None,
    peak_frequency_decimals: int = 2,
    show_common_peak_lines: bool = False,
    common_peak_tolerance: float = 0.02,
):
    """Find prominent peaks in one or more noise PSD traces.

    ``frequency`` is a shared frequency array. ``psd`` may be a one-dimensional
    array, a two-dimensional array with shape ``(n_traces, n_frequency)``, or a
    mapping from trace labels to one-dimensional arrays. Detection is performed
    on an equally-spaced ``log10(frequency)`` grid, which also handles
    logarithmically spaced or mildly irregular input data.

    Args:
        frequency: Frequency samples. Values must be finite, positive, and
            strictly increasing after sorting.
        psd: Linear PSD values. A mapping or 2-D array enables multi-trace
            detection. All traces use the same frequency array.
        labels: Optional labels for rows of a 2-D PSD array.
        is_plot: If true, plot each PSD and mark the detected peaks.
        ax: Optional Matplotlib axes. A new axes is created when omitted.
        title: Plot title; set to ``None`` to omit it.
        frequency_unit, psd_unit: Axis-label units shown when plotting.
        prominence: Minimum peak prominence in dB after baseline removal.
        distance, width: Minimum peak distance and width in samples on the
            internal uniform log-frequency grid.
        smooth_window: Savitzky-Golay window length; set to 0 or 1 to disable.
        baseline_window: Median-filter window for a slowly varying baseline;
            set to 0 to detect against the smoothed PSD directly.
        show_peak_frequencies: Annotate every detected peak on the plot when
            ``is_plot`` is true.
        peak_frequency_unit: Annotation unit, one of ``Hz``, ``kHz``, ``MHz``,
            or ``GHz``. ``None`` selects a readable unit for each peak.
        peak_frequency_decimals: Number of decimal places in peak annotations.
        show_common_peak_lines: For multiple traces, replace markers for peaks
            shared by at least two traces with one vertical dashed line and a
            frequency annotation.
        common_peak_tolerance: Matching tolerance for common peaks in decades
            of frequency (``0.02`` is approximately 4.7 percent).

    Returns:
        A one-dimensional NumPy array for one trace, or a dictionary mapping
        trace labels to peak-frequency arrays for multiple traces.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks, savgol_filter

    if peak_frequency_unit is not None:
        peak_frequency_unit = str(peak_frequency_unit)
        if peak_frequency_unit not in {"Hz", "kHz", "MHz", "GHz"}:
            raise ValueError("peak_frequency_unit must be one of Hz, kHz, MHz, or GHz.")
    if int(peak_frequency_decimals) < 0:
        raise ValueError("peak_frequency_decimals must be non-negative.")
    if not np.isfinite(common_peak_tolerance) or common_peak_tolerance < 0:
        raise ValueError("common_peak_tolerance must be a finite, non-negative value.")

    freq = np.asarray(frequency, dtype=float).ravel()
    if freq.size < 3:
        raise ValueError("frequency must contain at least three samples.")
    if not np.all(np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("frequency must contain only finite, positive values.")
    order = np.argsort(freq)
    freq = freq[order]
    if np.any(np.diff(freq) <= 0):
        raise ValueError("frequency values must be unique.")

    if isinstance(psd, Mapping):
        trace_items = [(str(name), values) for name, values in psd.items()]
        if not trace_items:
            raise ValueError("psd mapping must contain at least one trace.")
        multi_trace = True
    else:
        psd_array = np.asarray(psd, dtype=float)
        if psd_array.ndim == 1:
            trace_items = [("psd", psd_array)]
            multi_trace = False
        elif psd_array.ndim == 2:
            if psd_array.shape[1] != freq.size:
                raise ValueError(
                    "A 2-D psd array must have shape (n_traces, n_frequency)."
                )
            if psd_array.shape[0] == 0:
                raise ValueError("A 2-D psd array must contain at least one trace.")
            if labels is None:
                trace_labels = [f"psd_{index}" for index in range(psd_array.shape[0])]
            else:
                trace_labels = [str(label) for label in labels]
                if len(trace_labels) != psd_array.shape[0]:
                    raise ValueError("labels must have one entry per PSD trace.")
            trace_items = list(zip(trace_labels, psd_array))
            multi_trace = True
        else:
            raise ValueError("psd must be 1-D, 2-D, or a mapping of 1-D arrays.")

    def detect(values):
        values = np.asarray(values, dtype=float).ravel()
        if values.size != freq.size:
            raise ValueError("Each PSD trace must have the same length as frequency.")
        values = values[order]
        valid = np.isfinite(values) & (values > 0)
        if np.count_nonzero(valid) < 3:
            raise ValueError("Each PSD trace needs at least three positive finite samples.")
        x = np.log10(freq[valid])
        z = 10.0 * np.log10(values[valid])
        x_uniform = np.linspace(x[0], x[-1], max(256, x.size * 8))
        z_uniform = PchipInterpolator(x, z)(x_uniform)

        window = int(smooth_window)
        if window > 1:
            window = min(window, z_uniform.size if z_uniform.size % 2 else z_uniform.size - 1)
            if window >= 3:
                if window % 2 == 0:
                    window -= 1
                z_work = savgol_filter(z_uniform, window, 2)
            else:
                z_work = z_uniform
        else:
            z_work = z_uniform

        baseline_size = int(baseline_window)
        if baseline_size > 1:
            baseline_size = min(baseline_size, z_work.size)
            if baseline_size % 2 == 0:
                baseline_size -= 1
            baseline = median_filter(z_work, size=max(3, baseline_size), mode="nearest")
            signal = z_work - baseline
        else:
            signal = z_work

        peak_indices, _ = find_peaks(
            signal,
            prominence=float(prominence),
            distance=max(1, int(distance)),
            width=max(1, int(width)),
        )
        return 10.0 ** x_uniform[peak_indices], valid, values

    peak_results = {}
    plot_items = []
    for name, values in trace_items:
        peaks, valid, sorted_values = detect(values)
        peak_results[name] = peaks
        plot_items.append((name, valid, sorted_values, peaks))

    common_peaks = np.array([], dtype=float)
    shared_peak_masks = [np.zeros(item[3].size, dtype=bool) for item in plot_items]
    if show_common_peak_lines and multi_trace and len(plot_items) > 1:
        candidates = sorted(
            (float(np.log10(peak)), trace_index, peak_index)
            for trace_index, item in enumerate(plot_items)
            for peak_index, peak in enumerate(item[3])
        )
        clusters = []
        for log_peak, trace_index, peak_index in candidates:
            matching = [
                cluster for cluster in clusters
                if abs(log_peak - cluster["center"]) <= common_peak_tolerance
                and trace_index not in cluster["traces"]
            ]
            if matching:
                cluster = min(matching, key=lambda item: abs(log_peak - item["center"]))
                cluster["entries"].append((log_peak, trace_index, peak_index))
                cluster["traces"].add(trace_index)
                cluster["center"] = float(np.mean([entry[0] for entry in cluster["entries"]]))
            else:
                clusters.append({
                    "center": log_peak,
                    "traces": {trace_index},
                    "entries": [(log_peak, trace_index, peak_index)],
                })
        shared_clusters = [cluster for cluster in clusters if len(cluster["traces"]) >= 2]
        common_peaks = np.asarray([10.0 ** cluster["center"] for cluster in shared_clusters])
        for cluster in shared_clusters:
            for _, trace_index, peak_index in cluster["entries"]:
                shared_peak_masks[trace_index][peak_index] = True

    if is_plot:
        import matplotlib.pyplot as plt

        created_axes = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))
        for trace_index, (name, valid, sorted_values, peaks) in enumerate(plot_items):
            ax.loglog(freq[valid], sorted_values[valid], label=name, linewidth=1.4)
            if peaks.size:
                peak_values = np.interp(
                    np.log10(peaks), np.log10(freq[valid]), np.log10(sorted_values[valid])
                )
                marker_mask = ~shared_peak_masks[trace_index]
                if np.any(marker_mask):
                    ax.scatter(
                        peaks[marker_mask], 10.0 ** peak_values[marker_mask],
                        marker="o", zorder=3,
                    )
        ax.set_xlabel(f"Frequency ({frequency_unit})")
        ax.set_ylabel(f"{psd_unit}")
        if title:
            ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        if ax.lines:
            ax.legend()

        if show_peak_frequencies or common_peaks.size:
            offsets = (12, -18, 30, -36, 48, -54, 66, -72)
            unit_scales = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

            def format_peak_frequency(value):
                unit = peak_frequency_unit
                if unit is None:
                    magnitude = abs(float(value))
                    unit = (
                        "GHz" if magnitude >= 1e9 else
                        "MHz" if magnitude >= 1e6 else
                        "kHz" if magnitude >= 1e3 else "Hz"
                    )
                scaled = float(value) / unit_scales[unit]
                return f"{scaled:.{int(peak_frequency_decimals)}f} {unit}"

            if common_peaks.size:
                for index, common_peak in enumerate(common_peaks):
                    ax.axvline(
                        common_peak, color="0.25", linestyle="--",
                        linewidth=1.0, alpha=0.7, zorder=1,
                    )
                    ax.annotate(
                        format_peak_frequency(common_peak),
                        xy=(common_peak, 0.98),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -6 - 18 * (index % 3)),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=8,
                        rotation=90,
                        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.8},
                    )

            ax.figure.canvas.draw()
            lane_last_x = []
            for trace_index, (_, valid, sorted_values, peaks) in enumerate(plot_items):
                annotation_mask = ~shared_peak_masks[trace_index]
                if not peaks.size or not show_peak_frequencies or not np.any(annotation_mask):
                    continue
                annotated_peaks = peaks[annotation_mask]
                x_pixels = ax.transData.transform(
                    np.column_stack((annotated_peaks, np.ones(annotated_peaks.size)))
                )[:, 0]
                for peak, x_pixel in zip(annotated_peaks, x_pixels):
                    lane = next(
                        (index for index, previous_x in enumerate(lane_last_x)
                         if x_pixel - previous_x >= 55.0),
                        len(lane_last_x),
                    )
                    if lane == len(lane_last_x):
                        lane_last_x.append(x_pixel)
                    else:
                        lane_last_x[lane] = x_pixel
                    offset = offsets[(lane + 2 * trace_index) % len(offsets)]
                    peak_value = np.interp(
                        np.log10(peak), np.log10(freq[valid]), np.log10(sorted_values[valid])
                    )
                    ax.annotate(
                        format_peak_frequency(peak),
                        xy=(peak, 10.0 ** peak_value),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if offset > 0 else "top",
                        fontsize=8,
                        rotation=35,
                        arrowprops={"arrowstyle": "-", "lw": 0.6, "alpha": 0.55},
                        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
                    )
        ax.figure.tight_layout()
        if created_axes:
            plt.show()

    return peak_results if multi_trace else peak_results["psd"]


PSD_MODES = (
    "rpwr_over_bandwidth",
    "rpwr_raw",
    "xy_stddev2_over_bandwidth",
    "rstddev2_over_bandwidth",
)

PSD_MODE_LABELS = {
    "rpwr_over_bandwidth": "rpwr / bandwidth",
    "rpwr_raw": "rpwr (raw)",
    "xy_stddev2_over_bandwidth": "(xstddev^2 + ystddev^2) / bandwidth",
    "rstddev2_over_bandwidth": "rstddev^2 / bandwidth",
}


def process_psd(
    frequency: Union[float, np.ndarray],
    bandwidth: Union[float, np.ndarray],
    mode: str = "xy_stddev2_over_bandwidth",
    *,
    rpwr: Optional[Union[float, np.ndarray]] = None,
    xstddev: Optional[Union[float, np.ndarray]] = None,
    ystddev: Optional[Union[float, np.ndarray]] = None,
    rstddev: Optional[Union[float, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one PSD representation from Zurich sweeper arrays.

    Args:
        frequency: Frequency samples in Hz. Returned unchanged as a detached
            one-dimensional array so callers can keep frequency and PSD aligned.
        bandwidth: Resolution bandwidth for each sample in Hz.
        mode: One of ``PSD_MODES``:
            ``rpwr_over_bandwidth`` computes ``rpwr / bandwidth``;
            ``rpwr_raw`` returns ``rpwr`` without normalization;
            ``xy_stddev2_over_bandwidth`` computes
            ``(xstddev**2 + ystddev**2) / bandwidth``;
            ``rstddev2_over_bandwidth`` computes ``rstddev**2 / bandwidth``.
        rpwr, xstddev, ystddev, rstddev: Raw sweeper arrays. Only the fields
            required by the selected mode are required.

    Returns:
        A ``(frequency, values)`` tuple. Values retain non-finite results from
        invalid input; callers can decide whether to filter or diagnose them.

    Raises:
        ValueError: If the mode is unknown, a required field is missing, or
            input arrays do not have the same length.
    """
    if mode not in PSD_MODES:
        choices = ", ".join(PSD_MODES)
        raise ValueError(f"Unknown PSD mode {mode!r}. Use one of: {choices}")

    freq = np.asarray(frequency, dtype=float).ravel()
    bw = np.asarray(bandwidth, dtype=float).ravel()
    if freq.size != bw.size:
        raise ValueError(
            f"frequency and bandwidth must have the same length; got {freq.size} and {bw.size}."
        )

    if mode == "xy_stddev2_over_bandwidth":
        if xstddev is None or ystddev is None:
            raise ValueError(f"PSD mode {mode!r} requires xstddev and ystddev arrays.")
        x_values = np.asarray(xstddev, dtype=float).ravel()
        y_values = np.asarray(ystddev, dtype=float).ravel()
        if x_values.size != freq.size or y_values.size != freq.size:
            raise ValueError(
                f"PSD mode {mode!r} requires xstddev and ystddev to have length {freq.size}."
            )
    else:
        field_name, field_values = {
            "rpwr_over_bandwidth": ("rpwr", rpwr),
            "rpwr_raw": ("rpwr", rpwr),
            "rstddev2_over_bandwidth": ("rstddev", rstddev),
        }[mode]
        if field_values is None:
            raise ValueError(f"PSD mode {mode!r} requires a {field_name} array.")
        raw_values = np.asarray(field_values, dtype=float).ravel()
        if raw_values.size != freq.size:
            raise ValueError(
                f"PSD mode {mode!r} requires {field_name} to have length {freq.size}; "
                f"got {raw_values.size}."
            )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if mode == "rpwr_over_bandwidth":
            values = raw_values / bw
        elif mode == "rpwr_raw":
            values = raw_values.copy()
        elif mode == "xy_stddev2_over_bandwidth":
            values = (x_values**2 + y_values**2) / bw
        else:
            values = raw_values**2 / bw

    return freq.copy(), np.asarray(values, dtype=float).copy()


def compare_psd(
    frequency: Union[float, np.ndarray],
    psd_a: Union[float, np.ndarray],
    psd_b: Union[float, np.ndarray],
) -> dict[str, dict[str, Union[float, int]]]:
    """Compare two PSD arrays using pointwise relative error statistics.

    Returns a mapping with ``all`` and ``nonzero_frequency`` entries, each
    holding the point count plus median, 95th-percentile, and maximum
    relative error against the larger magnitude of the two traces.
    """
    freq = np.asarray(frequency, dtype=float).ravel()
    first = np.asarray(psd_a, dtype=float).ravel()
    second = np.asarray(psd_b, dtype=float).ravel()
    if not (freq.size == first.size == second.size):
        raise ValueError(
            "frequency and both PSD arrays must have the same length; "
            f"got {freq.size}, {first.size}, and {second.size}."
        )

    both_finite = np.isfinite(freq) & np.isfinite(first) & np.isfinite(second)
    denominator = np.maximum(
        np.maximum(np.abs(first), np.abs(second)),
        np.finfo(float).tiny,
    )
    relative_error = np.abs(first - second) / denominator
    comparison: dict[str, dict[str, Union[float, int]]] = {}
    for name, mask in (
        ("all", both_finite),
        ("nonzero_frequency", both_finite & (freq != 0)),
    ):
        values = relative_error[mask]
        if values.size:
            comparison[name] = {
                "points": int(values.size),
                "median_relative_error": float(np.median(values)),
                "p95_relative_error": float(np.percentile(values, 95)),
                "max_relative_error": float(np.max(values)),
            }
    return comparison


def plot_psd(
    frequency: Union[float, np.ndarray],
    psd_by_mode: Mapping[str, Union[np.ndarray, None]],
    *,
    ax=None,
    title: Optional[str] = None,
    ylabel: str = "PSD",
    include_raw: bool = False,
):
    """Plot one frequency trace for each available PSD mode.

    Matplotlib is imported lazily so using the numerical noise-model helpers
    does not require a plotting backend. The raw ``rpwr`` trace is excluded by
    default because it is V^2 rather than V^2/Hz; pass ``include_raw=True`` to
    show it as a diagnostic. The function returns ``(figure, ax)`` and leaves
    display control to the caller.
    """
    import matplotlib.pyplot as plt

    freq = np.asarray(frequency, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots()
    figure = ax.figure

    for mode, values in psd_by_mode.items():
        if mode == "rpwr_raw" and not include_raw:
            continue
        if values is None:
            continue
        psd = np.asarray(values, dtype=float).ravel()
        if psd.size != freq.size:
            raise ValueError(
                f"PSD mode {mode!r} has length {psd.size}; expected {freq.size}."
            )
        mask = np.isfinite(freq) & np.isfinite(psd) & (freq > 0) & (psd > 0)
        if not np.any(mask):
            continue
        order = np.argsort(freq[mask])
        label = PSD_MODE_LABELS.get(mode, mode)
        ax.loglog(freq[mask][order], psd[mask][order], label=label, linewidth=1.5)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    if ax.lines:
        ax.legend()
    return figure, ax


def plot_psd_groups(
    data,
    group_ids=None,
    psd_key: str = "psd",
    *,
    ax=None,
    title: Optional[str] = "Group PSD",
    ylabel: str = "PSD (V$^2$/Hz)",
    group_labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence] = None,
    cmap="tab20",
    legend: bool = True,
    legend_loc: Optional[str] = None,
    legend_ncol: Optional[int] = None,
    legend_outside_threshold: int = 10,
    legend_kwargs: Optional[Mapping] = None,
):
    """Plot selected processed H5 groups on one PSD axes.

    Args:
        data: Mapping returned by a group-processing workflow. Each item must
            provide ``freq``, the selected PSD under ``psd_key`` (or in
            ``psd_by_mode``), and may provide ``group_id``/``label`` metadata.
        group_ids: Optional group ids or sample paths to include. ``None`` plots
            every group in insertion order.
        psd_key: Item key for the selected PSD, or a key in ``psd_by_mode``.
        ax: Existing Matplotlib axes. A new axes is created when omitted.
        title: Plot title. Set to ``None`` to omit it.
        ylabel: PSD axis label.
        group_labels: Optional labels in selected-group order.
        colors: Optional explicit color sequence, one color per plotted group.
        cmap: Colormap name or Matplotlib colormap used when ``colors`` is not
            provided. The default uses ``tab20`` up to 20 groups and switches
            to ``viridis`` for larger collections to avoid color repetition.
        legend: Whether to add a legend.
        legend_loc: Explicit Matplotlib legend location. When omitted, small
            collections use ``"best"`` and large collections move outside the
            axes.
        legend_ncol: Explicit number of legend columns.
        legend_outside_threshold: Move the automatic legend outside the axes
            when the number of plotted groups exceeds this value.
        legend_kwargs: Additional keyword arguments passed to ``ax.legend``.

    Returns:
        ``(figure, ax)`` with one line per selected group. The legend uses each
        group's H5 ``label`` and falls back to ``group_id``.
    """
    import matplotlib.pyplot as plt

    if not data:
        raise ValueError("No processed H5 groups are available for plotting.")

    if group_ids is None:
        selected = list(data.items())
    else:
        if isinstance(group_ids, (str, int)):
            group_ids = [group_ids]
        wanted = {str(group_id) for group_id in group_ids}
        selected = [
            (sample_path, item)
            for sample_path, item in data.items()
            if str(item.get("group_id")) in wanted
            or str(sample_path) in wanted
        ]
        if not selected:
            raise ValueError(f"No requested groups found: {sorted(wanted)}")

    if legend_outside_threshold < 0:
        raise ValueError("legend_outside_threshold must be non-negative.")

    plot_items = []
    for sid, (sample_path, item) in enumerate(selected):
        if psd_key in item:
            psd = item[psd_key]
        else:
            psd = item.get("psd_by_mode", {}).get(psd_key)
        if psd is None:
            group_name = item.get("group_id", sample_path)
            raise ValueError(f"PSD key {psd_key!r} is unavailable for group {group_name}.")

        frequency = np.asarray(item["freq"], dtype=float).ravel()
        values = np.asarray(psd, dtype=float).ravel()
        if frequency.shape != values.shape:
            group_name = item.get("group_id", sample_path)
            raise ValueError(f"Frequency and PSD lengths differ for group {group_name}.")

        mask = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0) & (values > 0)
        if not np.any(mask):
            continue
        order = np.argsort(frequency[mask])
        if group_labels is not None and sid < len(group_labels):
            group_label = group_labels[sid]
        else:
            group_label = item.get("label") or f"group={item.get('group_id', sample_path)}"
        plot_items.append(
            (
                frequency[mask][order],
                values[mask][order],
                group_label,
            )
        )

    n_groups = len(plot_items)
    if colors is not None:
        line_colors = list(colors)
        if len(line_colors) < n_groups:
            raise ValueError(
                f"colors must provide at least one color per plotted group; "
                f"got {len(line_colors)} for {n_groups} groups."
            )
    else:
        if cmap == "tab20" and n_groups > 20:
            cmap = "viridis"
        color_map = plt.get_cmap(cmap)
        sample_points = np.linspace(0.0, 1.0, max(n_groups, 1))
        line_colors = [color_map(point) for point in sample_points]

    legend_options = dict(legend_kwargs or {})
    has_explicit_legend_layout = any(
        key in legend_options for key in ("loc", "bbox_to_anchor")
    ) or legend_loc is not None
    auto_external_legend = (
        legend
        and n_groups > legend_outside_threshold
        and not has_explicit_legend_layout
    )

    if ax is None:
        figsize = (12, 5) if auto_external_legend else (9, 5)
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    for (frequency, values, group_label), line_color in zip(plot_items, line_colors):
        ax.loglog(
            frequency,
            values,
            label=group_label,
            color=line_color,
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if legend and ax.lines:
        if legend_loc is not None:
            legend_options.setdefault("loc", legend_loc)
        if legend_ncol is not None:
            legend_options.setdefault("ncol", legend_ncol)
        if auto_external_legend:
            legend_options.setdefault("loc", "upper left")
            legend_options.setdefault("bbox_to_anchor", (1.02, 1.0))
            legend_options.setdefault("borderaxespad", 0.0)
            legend_options.setdefault("fontsize", "small")
            if legend_ncol is None:
                legend_options.setdefault(
                    "ncol",
                    max(1, min(3, int(np.ceil(len(ax.lines) / 16.0)))),
                )
        else:
            legend_options.setdefault("loc", "best")
        ax.legend(**legend_options)

    if auto_external_legend:
        figure.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
    else:
        figure.tight_layout()
    return figure, ax


def compare_psd_groups(
    data,
    modes=("xy_stddev2_over_bandwidth", "rpwr_over_bandwidth"),
):
    """Compare multiple PSD representations in one panel per H5 group."""
    import matplotlib.pyplot as plt

    if not data:
        raise ValueError("No processed H5 groups are available for plotting.")
    if isinstance(modes, str):
        modes = (modes,)

    figure, axes = plt.subplots(
        nrows=len(data),
        ncols=1,
        squeeze=False,
        figsize=(9, max(4.0, 3.8 * len(data))),
    )
    for axis, (sample_path, item) in zip(axes[:, 0], data.items()):
        available = item.get("psd_by_mode", {})
        if not available:
            available = {
                "xy_stddev2_over_bandwidth": item.get("psd_xy"),
                "rpwr_over_bandwidth": item.get("psd_rpwr"),
            }
        selected = {mode: available.get(mode) for mode in modes}
        check = item.get("crosscheck", {}).get("nonzero_frequency", {})
        title = f"group={item.get('group_id', sample_path)}"
        if item.get("label"):
            title += f" | label={item['label']}"
        if check:
            title += f" | p95 relative error={check['p95_relative_error']:.2%}"
        plot_psd(
            item["freq"],
            selected,
            ax=axis,
            title=title,
            ylabel="PSD (V$^2$/Hz)",
        )
    figure.tight_layout()
    return figure, axes


def convert_noise_psd_to_load(
    psd: PSDInput,
    R_m: float = 50.0,
    R_s: float = 50.0,
    R_l: float = 50.0,
    mode: PSDUnitMode = "dBm/Hz",
) -> PSDOutput:
    """Convert a measured noise spectrum to the spectrum at the actual load.

    The source is modeled as a Thevenin voltage-noise source with a series
    resistance R_s. The measurement instrument presents a resistance R_m,
    while the actual load presents a resistance R_l.

    Args:
        psd: Measured spectrum as a scalar or array-like value. Its unit is
            selected by ``mode``.
        R_m: Measurement instrument input resistance in ohms.
        R_s: Source output resistance in ohms.
        R_l: Actual load resistance in ohms.
        mode: Unit and quantity represented by ``psd``. Supported values are
            ``dBm/Hz`` and ``W/Hz`` for absorbed power; ``V^2/Hz`` and
            ``V/sqrt(Hz)`` for load voltage; and ``A^2/Hz`` and ``A/sqrt(Hz)``
            for load current.

    Returns:
        The corresponding spectrum at R_l in the same unit as ``psd``. A
        scalar input produces a float; an array-like input produces a NumPy
        float64 array with the same shape.

    Raises:
        ValueError: If a resistance is non-positive or non-finite, ``mode`` is
            unsupported, or a linear PSD or ASD input contains negative values.

    Notes:
        This corrects only for resistive loading. It does not remove the
        instrument noise floor or correct cable loss, amplifier noise, or
        frequency-dependent complex impedances.
    """
    resistances = np.asarray([R_m, R_s, R_l], dtype=np.float64)

    if not np.all(np.isfinite(resistances)) or np.any(resistances <= 0.0):
        raise ValueError("R_m, R_s, and R_l must be finite and positive.")

    supported_modes: Tuple[PSDUnitMode, ...] = (
        "dBm/Hz",
        "W/Hz",
        "V^2/Hz",
        "V/sqrt(Hz)",
        "A^2/Hz",
        "A/sqrt(Hz)",
    )
    if mode not in supported_modes:
        raise ValueError(
            f"Unsupported mode {mode!r}. Expected one of {supported_modes}."
        )

    measured = np.asarray(psd, dtype=np.float64)

    if mode != "dBm/Hz" and np.any(measured < 0.0):
        raise ValueError("Linear PSD and ASD values cannot be negative.")

    power_ratio: float = R_l * (R_s + R_m) ** 2 / (R_m * (R_s + R_l) ** 2)
    voltage_asd_ratio: float = R_l * (R_s + R_m) / (R_m * (R_s + R_l))
    current_asd_ratio: float = (R_s + R_m) / (R_s + R_l)

    if mode == "dBm/Hz":
        converted = measured + 10.0 * np.log10(power_ratio)
    elif mode == "W/Hz":
        converted = measured * power_ratio
    elif mode == "V^2/Hz":
        converted = measured * voltage_asd_ratio**2
    elif mode == "V/sqrt(Hz)":
        converted = measured * voltage_asd_ratio
    elif mode == "A^2/Hz":
        converted = measured * current_asd_ratio**2
    else:  # mode == "A/sqrt(Hz)"
        converted = measured * current_asd_ratio

    if converted.ndim == 0:
        return float(converted)

    return converted.astype(np.float64, copy=False)

def thermal_photon_noise(T_cav:float, kappa:float, chi:float, cavity_freq:float, PSD_freq:np.ndarray, S0:float = 0.0) -> np.ndarray:
    """
    Compatibility wrapper around the decoherence-owned thermal photon PSD helper.

    Args:
        T_cav (float): Cavity temperature [mK].
        kappa (float): Photon number decay rate of cavity[Hz].
        chi (float): Dispersive shift of qubit [Hz].
        cavity_freq (float): Cavity frequency [Hz].
        PSD_freq (np.ndarray): Frequency array [Hz].
        S0 (float, optional): Background noise [omega^2/Hz]. Defaults to 0.0.

    Returns:
        np.ndarray: Thermal photon number noise PSD [omega^2/Hz].
    """
    from ..decoherence.noise import readout_thermal_photon_noise

    return readout_thermal_photon_noise(
        T_cav=T_cav,
        kappa=kappa,
        chi=chi,
        cavity_freq=cavity_freq,
        psd_freq=PSD_freq,
        S0=S0,
    )


def thermal_photon_noise_fit(
    PSD_freq: np.ndarray,
    PSD: np.ndarray,
    init_guess: Optional[dict] = None,
    T_cav: Optional[float] = None,
    kappa: Optional[float] = None,
    chi: Optional[float] = None,
    S0: Optional[float] = None,
    cavity_freq: float = 6.5e9,
    bounds: Optional[tuple] = None,
    valid_mask: Optional[np.ndarray] = None,
    robust_fit: bool = True
) -> dict:
    """
    Compatibility wrapper around the decoherence-owned thermal photon fit helper.

    Args:
        PSD_freq (np.ndarray): Frequency array [Hz].
        PSD (np.ndarray): Single-sided PSD array of qubit f01 caused by thermal photon noise [omega^2/Hz].
        init_guess (dict, optional): Initial guesses for the free parameters. Keys should be a subset of {'T_cav', 'kappa', 'chi', 'S0'}. Defaults to None.
        T_cav (float, optional): Cavity temperature [mK]. If None, it will be fitted.
        kappa (float, optional): Photon decay rate of cavity [Hz]. If None, it will be fitted.
        chi (float, optional): Dispersive shift of qubit [Hz]. If None, it will be fitted.
        S0 (float, optional): Background noise floor [omega^2/Hz]. If None, it will be fitted.
        cavity_freq (float, optional): Cavity frequency [Hz]. Defaults to 6.5e9.
        bounds (tuple, optional): Optimization bounds. Defaults to None.
        valid_mask (np.ndarray, optional): Boolean array to filter out specific noise peaks. True means keep, False means ignore. Defaults to None.
        robust_fit (bool, optional): Use 'soft_l1' loss function to automatically downweight outliers/peaks. Defaults to True.

    Returns:
        dict: Fitting results including 'params', 'fit_values', 'errors', 'popt', 'pcov', and 'fitted_PSD'.
    """
    from ..decoherence.noise import fit_readout_thermal_photon_noise

    return fit_readout_thermal_photon_noise(
        psd_freq=PSD_freq,
        psd=PSD,
        init_guess=init_guess,
        T_cav=T_cav,
        kappa=kappa,
        chi=chi,
        S0=S0,
        cavity_freq=cavity_freq,
        bounds=bounds,
        valid_mask=valid_mask,
        robust_fit=robust_fit,
    )



