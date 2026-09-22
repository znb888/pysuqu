"""
Lib for electrical noise model.

Author: Naibin Zhou
since 2025-12-10
"""

import numpy as np
from scipy.constants import h, hbar, Boltzmann
from typing import Mapping, Optional, Sequence, Union

kb = Boltzmann

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



