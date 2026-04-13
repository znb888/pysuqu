"""
Lib for calculation of decoherence of superconducting qubit.


"""
# import
import inspect
import warnings

import numpy as np
import scipy as sp
from typing import Optional, Callable, Tuple, List, Union
# local lib
from ..funclib.mathlib import *
from ..qubit.base import AbstractQubit, Phi0
from .analysis import ReadoutCavityAnalyzer, XYRelaxationAnalyzer, ZDephasingAnalyzer
from .electronics import ElectronicNoise
from .formatting import (
    format_bias_current_voltage_report,
    format_readout_nbar_report,
    format_readout_tphi_report,
    format_xy_current_voltage_report,
    format_xy_t1_report,
    format_xy_thermal_excitation_report,
    format_z_tphi1_report,
    format_z_tphi2_report,
)
from .plotting import plot_read_tphi_fit, plot_z_tphi2_fit
from .results import T1Result, TphiResult

Euler_gamma = 0.577216

class Decoherence:
    """
    Base class for superconducting qubit decoherence models.
    
    Purpose:
        Provides a unified framework for decoherence calculations, handling initialization
        of qubit and electronic noise models. All specific noise types inherit from this class.
    
    Args:
        psd_freq (np.ndarray | float): Frequency array for noise power spectral density (PSD) [Hz]
        psd_S (np.ndarray | float): Noise power spectral density values [units depend on noise type]
        couple_term (float): Coupling strength parameter (interpretation varies by subclass)
        couple_type (str): Coupling type identifier ('z' | 'xy' | 'r')
        noise_type (str): Noise type, default '1f' ('1f' | 'white' | 'constant')
        noise_prop (str): Noise propagation mode, default 'single'
        T_setup (np.ndarray): Temperature setup [K], default [290, 45, 3.5, 0.9, 0.1, 0.01]
        attenuation_setup (np.ndarray): Attenuation setup [dB], default [10, 3, 10, 10, 0, 0]
        qubit_freq (float): Qubit frequency [Hz], default 5e9
        qubit_freq_max (float): Maximum frequency [Hz], default equals qubit_freq
        qubit_anharm (float): Anharmonicity [Hz], default -250e6
        qubit_type (str): Qubit type, default 'Transmon'
        energy_trunc_level (int): Energy truncation level, default 12
        is_spectral (bool): Whether to use spectral analysis, default True
    
    Note:
        This class is typically not instantiated directly. Use subclasses:
        ZNoiseDecoherence, XYNoiseDecoherence, or RNoiseDecoherence.
    """
    def __init__(
        self,
        psd_freq: Union[np.ndarray, float],
        psd_S: Union[np.ndarray, float],
        couple_term: float,
        couple_type: str,
        noise_type: str = '1f',
        noise_prop: str = 'single',
        T_setup: np.ndarray = np.array([290, 45, 3.5, 0.9, 0.1, 0.01]),
        attenuation_setup: np.ndarray = np.array([10, 3, 10, 10, 0, 0]),
        qubit_freq_max: float = None,
        qubit_anharm: float = -250e6,
        qubit_freq: float = 5e9,
        qubit_type: str = 'Transmon',
        energy_trunc_level: int = 12,
        is_spectral: bool = True,
        is_print: bool = True,
        qubit_builder: Optional[Callable[..., object]] = None,
        noise_builder: Optional[Callable[..., object]] = None,
        *args, **kwargs
    ):
        self.couple_term = couple_term
        self.couple_type = couple_type
        self.psd_freq = psd_freq
        self.psd_S = psd_S
        self.noise_prop = noise_prop
        self.T_setup = T_setup
        self.attenuation_setup = attenuation_setup
        
        self.qubit_freq = qubit_freq
        self.qubit_freq_max = qubit_freq_max if qubit_freq_max is not None else qubit_freq
        self.qubit_anharm = qubit_anharm
        self.qubit_type = qubit_type
        self.energy_trunc_level = energy_trunc_level
        self.is_spectral = is_spectral
        self._default_is_print = bool(is_print)
        self._qubit_builder = qubit_builder or AbstractQubit
        self._noise_builder = noise_builder or ElectronicNoise
        if self.is_spectral:
            self.noise_type = noise_type
        else:
            self.noise_type = 'constant'
        
        # Initialize models
        self.refresh_model(*args, is_print=is_print, **kwargs)

    @staticmethod
    def _builder_accepts_kwarg(builder, kwarg: str) -> bool:
        try:
            signature = inspect.signature(builder)
        except (TypeError, ValueError):
            return True

        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True

        return kwarg in signature.parameters

    def _resolve_is_print(self, is_print: Optional[bool]) -> bool:
        if is_print is None:
            return self._default_is_print

        return bool(is_print)

    def _build_qubit(self, *args, is_print: Optional[bool] = None, **kwargs):
        resolved_is_print = self._resolve_is_print(is_print)
        builder_kwargs = dict(kwargs)
        if self._builder_accepts_kwarg(self._qubit_builder, 'is_print'):
            builder_kwargs['is_print'] = resolved_is_print

        return self._qubit_builder(
            frequency=self.qubit_freq,
            anharmonicity=self.qubit_anharm,
            frequency_max=self.qubit_freq_max,
            qubit_type=self.qubit_type,
            energy_trunc_level=self.energy_trunc_level,
            *args, **builder_kwargs
        )

    def _build_noise(self, *args, is_print: Optional[bool] = None, **kwargs):
        resolved_is_print = self._resolve_is_print(is_print)
        builder_kwargs = dict(kwargs)
        if self._builder_accepts_kwarg(self._noise_builder, 'is_print'):
            builder_kwargs['is_print'] = resolved_is_print

        return self._noise_builder(
            psd_freq=self.psd_freq,
            psd_S=self.psd_S,
            noise_type=self.noise_type,
            noise_prop=self.noise_prop,
            T_setup=self.T_setup,
            attenuation_setup=self.attenuation_setup,
            is_spectral=self.is_spectral,
            *args, **builder_kwargs
        )

    def refresh_model(self, *args, is_print: Optional[bool] = None, **kwargs):
        """Rebuild the qubit and noise dependencies from the current model state."""
        resolved_is_print = self._resolve_is_print(is_print)
        self.qubit = self._build_qubit(*args, is_print=resolved_is_print, **kwargs)
        self.noise = self._build_noise(*args, is_print=resolved_is_print, **kwargs)

    @staticmethod
    def _emit_report(lines: tuple[str, ...], *, is_print: bool) -> None:
        if is_print:
            for line in lines:
                print(line)

    def _generate_transfunc(self, experiment: str = 'Ramsey', tau: float = 100e-9, N: int = 100, len_pi: float = 100e-9) -> Callable:
        """Generate transfer function for qubit experiments (filter function)."""
        if experiment == 'Ramsey':
            return lambda f: ramsey_transfunc(f, tau)
        elif experiment == 'SpinEcho':
            return lambda f: echo_transfunc(f, tau)
        elif experiment == 'CPMG':
            return lambda f: cpmg_transfunc(f, tau, N, len_pi)
        else:
            raise ValueError(f"Unknown experiment type: {experiment}")

    def cal_dephase(self, psd: np.ndarray, sensitivity_factor: float, noise_freq: np.ndarray = None, experiment: str = 'Ramsey', delay_list: np.ndarray = np.linspace(10, 10e3, 100)*1e-9, *, N: int = 100, len_pi: float = 100e-9) -> np.ndarray:
        """
        Generic calculation for dephasing probability decay P(t).
        
        Args:
            psd: Noise power spectral density.
            sensitivity_factor: The pre-factor determining coupling strength (e.g. dOmega/dPhi * M or chi).
                                This should be the FULL coefficient entering the integral, excluding the factor of 2 if applicable.
                                Formula used: decay = exp( - integral( S(w) * |F(w)|^2 ) * (2 * factor)^2 / 2 )
        """
        p1_list = []
        if noise_freq is None:
            noise_freq = self.noise.output_stage.frequency
        
        for tau in delay_list:
            trans_func = self._generate_transfunc(experiment, tau, N, len_pi)
            dfactor = integrate_square_large_span(noise_freq, psd, trans_func, method='log')
            exponent = -dfactor * (sensitivity_factor * 2)**2 / 2
            p1_list.append(np.exp(exponent))
            
        return np.array(p1_list)


class ZNoiseDecoherence(Decoherence):
    """
    Longitudinal (Z) noise decoherence handler (flux noise).
    
    Physical Scenario:
        - Flux noise couples to qubit via mutual inductance
        - Primarily affects pure dephasing time Tphi (T2*)
        - Suitable for Ramsey, SpinEcho, CPMG experiment analysis
    
    Args:
        psd_freq (np.ndarray): Noise frequency array [Hz] (required)
        psd_S (np.ndarray): Noise power spectral density [1/Hz] (required)
        couple_term (float): Mutual inductance M [H], default 1.5e-12 (required)
            Physical meaning: Mutual inductance between flux control line and qubit
        couple_type (str): Fixed to 'z' (auto-set, do not provide)
        noise_type (str): Noise type, default '1f' ('1f' | 'white' | 'constant')
        noise_prop (str): Noise propagation mode, default 'single'
        T_setup (np.ndarray): Temperature setup [K], default [290, 45, 3.5, 0.9, 0.1, 0.01]
        attenuation_setup (np.ndarray): Attenuation setup [dB], default [10, 3, 10, 10, 0, 0]
        qubit_freq (float): Qubit frequency [Hz], default 5e9
        qubit_freq_max (float): Maximum qubit frequency [Hz], default equals qubit_freq
        qubit_anharm (float): Anharmonicity [Hz], default -250e6
        qubit_type (str): Qubit type, default 'Transmon'
        energy_trunc_level (int): Energy truncation level, default 12
        is_spectral (bool): Whether to use spectral analysis, default True
    
    Key Methods:
        cal_tphi1(idle_freq, is_print) -> float
            Calculate pure dephasing time Tphi1 [s] in white noise limit
            
        cal_tphi2(method, experiment, delay_list, idle_freq, cut_point, is_plot, is_print) -> TphiResult
            Calculate Tphi2 [s] dominated by 1/f noise
            - method: 'fit' (numerical fitting) or 'cal' (analytical formula)
            - experiment: 'Ramsey' | 'SpinEcho' | 'CPMG'
            - idle_freq: Idle frequency point [Hz] for sensitivity calculation
            - cut_point: List of frequency boundaries for segmented analysis

        cal_bias_current_voltage(phi_fraction, is_print) -> dict
            Calculate bias current and voltage at chip and room temperature ends
            - phi_fraction: Flux bias as fraction of Phi0 (0.25 for qubit, 0.5 for coupler)
            - Returns dictionary with all calculated values
    
    Example Usage:
        ```python
        z_noise = ZNoiseDecoherence(
            psd_freq=freq_array,          # Noise frequency array [Hz]
            psd_S=psd_array,              # Noise PSD [1/Hz]
            couple_term=1.5e-12,          # Mutual inductance [H]
            qubit_freq=5e9,               # Qubit frequency [Hz]
            qubit_anharm=-250e6,          # Anharmonicity [Hz]
        )
        tphi1 = z_noise.cal_tphi1(idle_freq=5e9)
        tphi2_result = z_noise.cal_tphi2(experiment='Ramsey', idle_freq=5e9)
        # Calculate bias current/voltage
        chip_results = z_noise.cal_bias_current_voltage(phi_fraction=0.25)  # Qubit
        coupler_results = z_noise.cal_bias_current_voltage(phi_fraction=0.5)  # Coupler
        ```
    """
    def __init__(
        self,
        *args,
        couple_term: float = 1.5e-12,
        kappa: float = 6e6,
        z_analyzer_builder: Optional[Callable[..., object]] = None,
        **kwargs,
    ):
        couple_type='z'
        self._z_analyzer_builder = z_analyzer_builder or ZDephasingAnalyzer
        super().__init__(*args, couple_term=couple_term, couple_type=couple_type, **kwargs)
        self.z_analyzer = self._build_z_analyzer()
        self._update_sensitivity()

    def _build_z_analyzer(self):
        return self._z_analyzer_builder(couple_term=self.couple_term)

    def refresh_model(self, *args, is_print: Optional[bool] = None, **kwargs):
        super().refresh_model(*args, is_print=is_print, **kwargs)
        # Ensure sensitivity is updated whenever model is refreshed
        if hasattr(self, 'qubit'): 
            self._update_sensitivity()

    def _update_sensitivity(self):
        """Calculates flux sensitivity dOmega/dPhi."""
        sense, _ = self.qubit.calculate_sensitivity_at_detuning(mode='brief')
        self.qubit_sensibility = sense * 2e9 * pi / Phi0

    def get_sensitivity_at_idle(self, idle_freq: float = None) -> float:
        if idle_freq is None:
            return self.qubit_sensibility
        else:
            sense, _ = self.qubit.calculate_sensitivity_at_detuning((self.qubit_freq - idle_freq) / 1e9, mode='brief')
            return sense * 2e9 * pi / Phi0

    def cal_tphi1(self, idle_freq: float = None, sensitivity: float = None, is_print: bool = True) -> float:
        """Calculate pure dephasing Tphi1 (White noise limit)."""
        noise_output = self.noise.output_stage
        if sensitivity is None:
            sens = self.get_sensitivity_at_idle(idle_freq)
        else:
            sens = sensitivity * 2e9 * pi / Phi0

        self.tphi1 = self.z_analyzer.calculate_tphi1(
            noise_output=noise_output,
            sensitivity=sens,
        )
        if is_print:
            self._emit_report(
                format_z_tphi1_report(
                    idle_freq=idle_freq,
                    sensitivity=sens,
                    couple_term=self.couple_term,
                    noise_output=noise_output,
                    tphi1=self.tphi1,
                ),
                is_print=True,
            )
        return self.tphi1

    def cal_bias_current_voltage(self, phi_fraction: float = 0.25, is_print: bool = True) -> dict:
        """
        Calculate bias current and voltage at chip and room temperature ends.
        
        Args:
            phi_fraction (float): Fraction of Phi0 for bias flux, default 0.25 (for qubit)
                Use 0.25 for qubit, 0.5 for coupler
            is_print (bool): Whether to print the results
        
        Returns:
            dict: Dictionary containing all calculated values:
                - phi_bias: Bias flux [Wb]
                - chip_current_uA: Chip end current [uA]
                - chip_voltage_mV: Chip end voltage [mV]
                - total_attenuation_dB: Total attenuation [dB]
                - room_current_mA: Room temperature end current [mA]
                - room_voltage_mV: Room temperature end voltage [mV]
                - room_power_dBm: Room temperature end power [dBm]
        """
        results = self.z_analyzer.calculate_bias_current_voltage(
            phi_fraction=phi_fraction,
            attenuation_setup=self.attenuation_setup,
        )
        
        if is_print:
            for line in format_bias_current_voltage_report(phi_fraction=phi_fraction, results=results):
                print(line)
        
        return results

    def _build_tphi2_result(
        self,
        *,
        value: float,
        method: str,
        experiment: str,
        fit_diagnostics: Optional[dict[str, object]] = None,
    ) -> TphiResult:
        diagnostics = {'segments': {}}
        if fit_diagnostics is not None:
            diagnostics.update(fit_diagnostics)

        return TphiResult(
            value=value,
            metadata={
                'method': method,
                'experiment': experiment,
                'source': 'z-control',
            },
            fit_diagnostics=diagnostics,
        )

    def cal_tphi2(self, 
              method: str = 'fit', 
              experiment: str = 'Ramsey', 
              delay_list: np.ndarray = np.linspace(10, 10e3, 100) * 1e-9, 
              *, 
              sensitivity: Optional[float] = None,
              N: int = 100, 
              len_pi: float = 100e-9, 
              idle_freq: Optional[float] = None, 
              p0: Optional[List[float]] = None, 
              bounds: Tuple = ([0, 0, 0, -1], [np.inf, np.inf, 2, 1]), 
              cut_point: List[float] = [],
              is_print: bool = True, 
              is_plot: bool = True) -> TphiResult:
        """
        Calculate Tphi2 (pure dephasing time), typically dominated by 1/f noise.

        Allows filtering the noise power spectral density (PSD) via cut_point 
        to analyze contributions from specific frequency bands.

        Args:
            method (str): Calculation method, 'fit' for simulation fitting or 'cal' for analytical formula.
            experiment (str): Experiment type, 'Ramsey' or 'SpinEcho'.
            delay_list (np.ndarray): Array of delay times in seconds.
            N (int): Number of pulses (for CPMG/Echo), default is 100 (if applicable).
            len_pi (float): Length of pi pulse in seconds.
            idle_freq (float): Idle frequency point. If None, uses current operating point.
            p0 (List[float]): Initial guess for fitting parameters. 
                            Format: [T1, Tphi, scale, offset]. Default is None (sets internal default).
            bounds (Tuple): Bounds for curve fitting (lower_bound, upper_bound).
            cut_point (List[float]): Frequency range [min_freq, max_freq] to slice the PSD. 
                                    If empty, uses full spectrum.
            is_print (bool): Whether to print the result.
            is_plot (bool): Whether to plot the decay curve (only for 'fit' method).

        Returns:
            TphiResult: The calculated or fitted Tphi2 value in seconds.
        """
        if p0 is None:
            p0 = [100e-6, 3e-6, 1., 0.]

        if sensitivity is None:
            sens = self.get_sensitivity_at_idle(idle_freq)
        else:
            sens = sensitivity
        sensitivity_factor = sens * self.couple_term

        if method == 'fit':
            if len(p0) != len(bounds[0]):
                raise ValueError(f"Length mismatch: p0 ({len(p0)}) vs bounds ({len(bounds[0])})")

            noise_output = self.noise.output_stage
            noise_freq_all = noise_output.frequency
            psd_all = noise_output.psd_single

            def _compute_segment(f_mask):
                if not np.any(f_mask):
                    return np.inf, np.zeros_like(delay_list)

                _freq = noise_freq_all[f_mask]
                _psd = psd_all[f_mask]

                _p1_list = self.cal_dephase(
                    psd=_psd,
                    sensitivity_factor=sensitivity_factor,
                    noise_freq=_freq,
                    experiment=experiment,
                    delay_list=delay_list,
                    N=N,
                    len_pi=len_pi,
                )

                try:
                    _popt, _pcov = fit_decay(delay_list, _p1_list, tphi_decay, p0=p0, bounds=bounds)
                    return _popt, _p1_list, _pcov
                except RuntimeError:
                    return 0.0, _p1_list, np.inf

            popt, self.dephase, pcov = _compute_segment(np.ones(len(noise_freq_all), dtype=bool))
            self.tphi2 = popt[1]
            self.tphi2_fiterror = np.sqrt(pcov[1][1])

            segment_results = {}
            if cut_point:
                if min(cut_point) < np.min(noise_freq_all):
                    raise ValueError(f"Cut point is too low. The minimum frequency is {np.min(noise_freq_all)}")
                if max(cut_point) > np.max(noise_freq_all):
                    raise ValueError(f"Cut point is too high. The maximum frequency is {np.max(noise_freq_all)}")

                sorted_cuts = sorted(cut_point)
                boundaries = [np.min(noise_freq_all)] + sorted_cuts + [np.max(noise_freq_all)]

                for i in range(len(boundaries) - 1):
                    f_start, f_end = boundaries[i], boundaries[i+1]
                    mask = (noise_freq_all >= f_start) & (noise_freq_all <= f_end)

                    popt_seg, _, pcov_seg = _compute_segment(mask)

                    label = f"{f_start:.2e}-{f_end:.2e} Hz"
                    segment_results[label] = dict(
                        popt=np.array(popt_seg, copy=True),
                        pcov=np.array(pcov_seg, copy=True),
                    )

            if is_plot:
                plot_z_tphi2_fit(
                    delay_list=delay_list,
                    dephase=self.dephase,
                    popt=popt,
                    tphi2=self.tphi2,
                    tphi2_fiterror=self.tphi2_fiterror,
                    experiment=experiment,
                    segment_results=segment_results,
                )


            diagnostics = {
                'tphi1': float(popt[0]),
                'tphi1_fiterror': float(np.sqrt(pcov[0][0])),
                'fit_error': float(self.tphi2_fiterror),
                'segments': segment_results,
            }
            result = self._build_tphi2_result(
                value=self.tphi2,
                method='fit',
                experiment=experiment,
                fit_diagnostics=diagnostics,
            )
            if is_print:
                self._emit_report(
                    format_z_tphi2_report(
                        method='fit',
                        experiment=experiment,
                        idle_freq=idle_freq,
                        sensitivity_factor=sensitivity_factor,
                        noise_output=noise_output,
                        tphi2=self.tphi2,
                        fit_diagnostics=diagnostics,
                    ),
                    is_print=True,
                )
            return result

        if method == 'cal':
            if cut_point:
                warnings.warn(
                    "'cal' method usually assumes full 1/f spectrum. 'cut_point' is ignored in analytical mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            if not self.noise_type.startswith('1f'):
                raise ValueError("Analytical 'cal' method only works for 1/f type noise.")

            noise_output = self.noise.output_stage
            self.tphi2 = self.z_analyzer.calculate_tphi2_cal(
                noise_output=noise_output,
                sensitivity_factor=sensitivity_factor,
                experiment=experiment,
                delay_list=delay_list,
            )

            result = self._build_tphi2_result(
                value=self.tphi2,
                method='cal',
                experiment=experiment,
            )
            if is_print:
                self._emit_report(
                    format_z_tphi2_report(
                        method='cal',
                        experiment=experiment,
                        idle_freq=idle_freq,
                        sensitivity_factor=sensitivity_factor,
                        noise_output=noise_output,
                        tphi2=self.tphi2,
                    ),
                    is_print=True,
                )
            return result

        raise ValueError(f"Method {method} not supported.")

class XYNoiseDecoherence(Decoherence):
    """
    Transverse (XY) noise decoherence handler.
    
    Physical Scenario:
        - XY control line noise couples to qubit via capacitance/inductance
        - Primarily affects relaxation time T1 (energy decay)
        - Also calculates thermal excitation probability (from environment temperature)
    
    Parameters:
        psd_freq (np.ndarray): Noise frequency array [Hz] (required)
        psd_S (np.ndarray): Noise power spectral density [1/Hz] (required)
        couple_term (float): Coupling coefficient [H], default 0.8e-12 (required)
            Physical meaning: Coupling strength between XY control line and qubit
        couple_type (str): Fixed to 'xy' (auto-set, do not provide)
        noise_type (str): Noise type, default '1f' ('1f' | 'white' | 'constant')
        noise_prop (str): Noise propagation mode, default 'single'
        T_setup (np.ndarray): Temperature setup [K], default [290, 45, 3.5, 0.9, 0.1, 0.01]
        attenuation_setup (np.ndarray): Attenuation setup [dB], default [10, 3, 10, 10, 0, 0]
        qubit_freq (float): Qubit frequency [Hz], default 5e9
        qubit_freq_max (float): Maximum qubit frequency [Hz], default equals qubit_freq
        qubit_anharm (float): Anharmonicity [Hz], default -250e6
        qubit_type (str): Qubit type, default 'Transmon'
        energy_trunc_level (int): Energy truncation level, default 12
        is_spectral (bool): Whether to use spectral analysis, default True
    
    Key Methods:
        cal_t1() -> T1Result
            Calculate T1 relaxation time [s] limited by XY noise
            Based on Fermi's golden rule, considering absorption and emission processes

        cal_thermal_exitation(T1) -> (float, float)
            Calculate thermal excitation probability
            - T1: Actual measured T1 time [μs] for more accurate calculation
            - Returns: (total thermal excitation probability, XY-only thermal excitation probability)
            
        cal_xy_current_voltage(phi_fraction, is_print) -> dict
            Calculate XY control line current and voltage at chip and room temperature ends
            - phi_fraction: Flux bias as fraction of Phi0, default 0.010/(4*pi) for XY control
            - Returns dictionary with all calculated values
    
    Example Usage:
        ```python
        xy_noise = XYNoiseDecoherence(
            psd_freq=freq_array,                      # Noise frequency array [Hz]
            psd_S=psd_array,                          # Noise PSD [1/Hz]
            couple_term=0.65e-12,                     # Coupling coefficient [H]
            qubit_freq=5e9,                           # Qubit frequency [Hz]
            qubit_anharm=-250e6,                      # Anharmonicity [Hz]
            attenuation_setup=np.array([14.42, 3.27, 21.23, 3.47, 0.30, 19.21]),
        )
        t1_result = xy_noise.cal_t1()
        p_thermal, p_thermal_xy = xy_noise.cal_thermal_exitation(T1=50)
        
        # Calculate XY control line current/voltage
        xy_results = xy_noise.cal_xy_current_voltage()
        ```
    """
    def __init__(
        self,
        *args,
        couple_term: float = 0.65e-12,
        xy_analyzer_builder: Optional[Callable[..., object]] = None,
        **kwargs,
    ):
        couple_type='xy'
        self._xy_analyzer_builder = xy_analyzer_builder or XYRelaxationAnalyzer
        super().__init__(*args, couple_term=couple_term, couple_type=couple_type, **kwargs)
        self.xy_analyzer = self._build_xy_analyzer()

    def _build_xy_analyzer(self):
        return self._xy_analyzer_builder(couple_term=self.couple_term)

    def _build_t1_result(
        self,
        *,
        value: float,
        fit_diagnostics: Optional[dict[str, object]] = None,
    ) -> T1Result:
        diagnostics = {}
        if fit_diagnostics is not None:
            diagnostics.update(fit_diagnostics)

        return T1Result(
            value=value,
            metadata={
                'method': 'cal',
                'source': 'xy-control',
            },
            fit_diagnostics=diagnostics,
        )

    def cal_t1(self, is_print: bool = True) -> T1Result:
        """Calculate T1 limited by XY control line noise."""
        analysis = self.xy_analyzer.calculate_t1(
            noise_output=self.noise.output_stage,
            qubit_freq=self.qubit_freq,
            Ej=self.qubit.Ej[0,0],
            Ec=self.qubit.Ec[0,0],
        )
        self.Gamma_up = analysis['gamma_up']
        self.Gamma_down = analysis['gamma_down']
        self.T1 = analysis['t1']
        result = self._build_t1_result(
            value=self.T1,
            fit_diagnostics={
                'gamma_up': float(self.Gamma_up),
                'gamma_down': float(self.Gamma_down),
            },
        )
        if is_print:
            self._emit_report(
                format_xy_t1_report(
                    qubit_freq=self.qubit_freq,
                    noise_output=self.noise.output_stage,
                    gamma_up=self.Gamma_up,
                    gamma_down=self.Gamma_down,
                    t1=self.T1,
                ),
                is_print=True,
            )
        return result

    def cal_thermal_exitation(self, T1: float = None, is_print: bool = True) -> Tuple[float, float]:
        """
        Calculate thermal excitation probability.
        
        Args:
            T1 (float, optional): T1 relaxation time in [us]. Defaults to None.
        
        Returns:
            Tuple[float, float]: (Probability of thermal excitation, Probability of thermal excitation only XY)
        """
        if not hasattr(self, 'Gamma_up'):
            try:
                self.cal_t1(is_print=is_print)
            except Exception as exception:
                warnings.warn(
                    f"Auto-calculation of T1 failed: {exception}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return 0.0, 0.0

        if T1 is not None:
            thermal_excitation = self.xy_analyzer.calculate_thermal_excitation(
                gamma_up=self.Gamma_up,
                gamma_down=self.Gamma_down,
                t1_us=T1,
            )
        else:
            thermal_excitation = self.xy_analyzer.calculate_thermal_excitation(
                gamma_up=self.Gamma_up,
                gamma_down=self.Gamma_down,
                t1_us=None,
            )

        self.thermal_exitation, self.thermal_exitation_onlyxy = thermal_excitation
        if is_print:
            self._emit_report(
                format_xy_thermal_excitation_report(
                    measured_t1_us=T1,
                    thermal_excitation=self.thermal_exitation,
                    thermal_excitation_onlyxy=self.thermal_exitation_onlyxy,
                ),
                is_print=True,
            )

        return self.thermal_exitation, self.thermal_exitation_onlyxy

    def cal_xy_current_voltage(self, phi_fraction: float = 0.010 / (4 * np.pi), is_print: bool = True) -> dict:
        """
        Calculate XY control line current and voltage at chip and room temperature ends.
        
        Args:
            phi_fraction (float): Fraction of Phi0 for bias flux, default 0.010/(4*pi) for XY control
                This corresponds to Phi = 0.010*Phi0/2/np.pi/2 in the calculation script
            is_print (bool): Whether to print the results
        
        Returns:
            dict: Dictionary containing all calculated values:
                - phi_bias: Bias flux [Wb]
                - chip_current_uA: Chip end current [uA]
                - chip_voltage_uV: Chip end voltage [uV]
                - chip_power_dBm: Chip end power [dBm]
                - total_attenuation_dB: Total attenuation [dB]
                - room_current_mA: Room temperature end current [mA]
                - room_voltage_mV: Room temperature end voltage [mV]
                - room_power_dBm: Room temperature end power [dBm]
        """
        results = self.xy_analyzer.calculate_xy_current_voltage(
            phi_fraction=phi_fraction,
            attenuation_setup=self.attenuation_setup,
        )
        
        if is_print:
            for line in format_xy_current_voltage_report(phi_fraction=phi_fraction, results=results):
                print(line)
        
        return results


class RNoiseDecoherence(Decoherence):
    """
    Resonator (R) induced decoherence handler (photon shot noise).
    
    Physical Scenario:
        - Thermal photons in readout cavity affect qubit frequency via dispersive coupling
        - Primarily affects dephasing time Tphi (photon number fluctuation → frequency fluctuation)
        - Suitable for analyzing readout cavity design impact on coherence time
    
    Parameters:
        psd_freq (np.ndarray): Noise frequency array [Hz] (required)
        psd_S (np.ndarray): Noise power spectral density [1/Hz] (required)
        couple_term (float): Dispersive shift chi [Hz], default 1e6 (required)
            Physical meaning: Qubit frequency shift per photon
        kappa (float): Cavity linewidth [Hz], default 6e6.
        couple_type (str): Fixed to 'r' (auto-set, do not provide)
        noise_type (str): Noise type, default '1f' ('1f' | 'white' | 'constant')
        noise_prop (str): Noise propagation mode, default 'single'
        T_setup (np.ndarray): Temperature setup [K], default [290, 45, 3.5, 0.9, 0.1, 0.01]
        attenuation_setup (np.ndarray): Attenuation setup [dB], default [10, 3, 10, 10, 0, 0]
        qubit_freq (float): Qubit frequency [Hz], default 5e9
        qubit_freq_max (float): Maximum qubit frequency [Hz], default equals qubit_freq
        qubit_anharm (float): Anharmonicity [Hz], default -250e6
        qubit_type (str): Qubit type, default 'Transmon'
        energy_trunc_level (int): Energy truncation level, default 12
        is_spectral (bool): Whether to use spectral analysis, default True
    
    Key Methods:
        cal_readcavity_psd(kappa, chi, read_freq, noise_freq) -> np.ndarray
            Calculate effective frequency noise PSD induced by cavity
            
        cal_read_dephase(kappa, chi, experiment, read_freq, delay_list, N, len_pi) -> np.ndarray
            Calculate dephasing decay curve P(t)
            
        cal_read_tphi(kappa, method, chi, read_freq, experiment, delay_list, is_plot, is_print) -> TphiResult
            Calculate Tphi [s] limited by readout cavity noise
            - kappa: Cavity linewidth [Hz] (required for calculation)
            - chi: Dispersive shift [Hz], defaults to couple_term
            - method: 'cal' (analytical) or 'fit' (numerical fitting)
            - experiment: 'Ramsey' | 'SpinEcho' | 'CPMG'

    Example Usage:
        ```python
        r_noise = RNoiseDecoherence(
            psd_freq=freq_array,          # Noise frequency array [Hz]
            psd_S=psd_array,              # Noise PSD [1/Hz]
            couple_term=1e6,              # Dispersive shift chi [Hz]
            qubit_freq=5e9,               # Qubit frequency [Hz]
            qubit_anharm=-250e6,          # Anharmonicity [Hz]
        )
        tphi_result = r_noise.cal_read_tphi(
            kappa=5e6,                    # Cavity linewidth [Hz]
            experiment='Ramsey',
            is_plot=True
        )
        ```
    """
    def __init__(
        self,
        *args,
        couple_term: float = 1e6,
        kappa: float = 6e6,
        r_analyzer_builder: Optional[Callable[..., object]] = None,
        **kwargs,
    ):
        couple_type='r'
        self.chi = couple_term
        self.kappa = kappa
        self._r_analyzer_builder = r_analyzer_builder or ReadoutCavityAnalyzer
        super().__init__(*args, couple_term=couple_term, couple_type=couple_type, **kwargs)
        self.r_analyzer = self._build_r_analyzer()
        self.n_bar = self.cal_nbar(is_print=self._default_is_print)

    def _build_r_analyzer(self):
        return self._r_analyzer_builder(couple_term=self.couple_term)

    def cal_nbar(self, kappa:float=None, chi:float=None, read_freq:float=6.5e9, is_print:bool=True) -> float:
        """
        Calculate the photon number n_bar in the readout cavity.
        
        Args:
            kappa (float): Cavity linewidth [rad/s] (required)
            chi (float): Dispersive shift [rad/s] (required)
            read_freq (float): Readout frequency [Hz] (required)
        
        Returns:
            float: The photon number n_bar in the readout cavity.
        """
        if kappa is None:
            kappa = self.kappa
        if chi is None:
            chi = self.chi

        self.n_bar = self.r_analyzer.calculate_nbar(
            noise_output=self.noise.output_stage,
            read_freq=read_freq,
        )
        if is_print:
            self._emit_report(
                format_readout_nbar_report(
                    read_freq=read_freq,
                    noise_output=self.noise.output_stage,
                    n_bar=self.n_bar,
                ),
                is_print=True,
            )
        return self.n_bar

    def cal_readcavity_psd(self, kappa: float = None, chi: float = None, read_freq: float = 6.5e9, noise_freq: np.ndarray = None) -> np.ndarray:
        """
        Calculate single-sided PSD of effective qubit frequency noise induced by thermal photons.
        
        Args:
            kappa (float): Cavity linewidth [rad/s] (required)
            chi (float, optional): Dispersive shift [rad/s], defaults to couple_term*2*pi
            read_freq (float, optional): Readout frequency [Hz], default 6.5e9
            noise_freq (np.ndarray, optional): Noise frequency array [Hz], default log-spaced from 10^-2 to 2*kappa
        
        Returns:
            np.ndarray: The single-sided PSD of effective qubit frequency noise [omega^2/Hz].
        """
        if chi is None:
            chi = self.couple_term*2*pi
        if kappa is None:
            kappa = self.kappa*2*pi

        self.psd_read = self.r_analyzer.calculate_readcavity_psd(
            n_bar=self.n_bar,
            kappa=kappa,
            chi=chi,
            noise_freq=noise_freq,
            read_freq=read_freq,
        )
        return self.psd_read

    @staticmethod
    def _cal_CPMG_integral(tau: np.ndarray, N: int, kappa: float, len_pi: float) -> float:
        return ReadoutCavityAnalyzer.calculate_cpmg_integral(
            tau=tau,
            N=N,
            kappa=kappa,
            len_pi=len_pi,
        )
    
    def cal_read_dephase(self, kappa: float = None, chi: float = None, experiment: str = 'Ramsey', read_freq: float = 6.5e9, delay_list: np.ndarray = np.linspace(10, 10e3, 100)*1e-9, N: int = 100, len_pi: float = 100e-9) -> float:
        """
        Calculate dephasing time Tphi limited by readout cavity thermal photons.
        
        Args:
            kappa (float): Cavity linewidth [rad/s] (required)
            chi (float, optional): Dispersive shift [rad/s], defaults to couple_term*2*pi
            experiment (str, optional): The experiment type. Defaults to 'Ramsey'.
            read_freq (float, optional): Readout frequency [Hz], default 6.5e9
            delay_list (np.ndarray, optional): Delay time array [s], default linspace from 10 to 10e3
            N (int, optional): Number of pulse blocks for CPMG, default 100
            len_pi (float, optional): Finite width of each pulse [s], default 100e-9
        
        Returns:
            np.ndarray: The dephasing probability array P1.
        
        """
        if chi is None:
            chi = self.couple_term*2*pi
        if kappa is None:
            kappa = self.kappa*2*pi

        return self.r_analyzer.calculate_read_dephase(
            n_bar=self.n_bar,
            kappa=kappa,
            chi=chi,
            experiment=experiment,
            read_freq=read_freq,
            delay_list=delay_list,
            N=N,
            len_pi=len_pi,
        )

    def _build_read_tphi_result(
        self,
        *,
        value: float,
        method: str,
        experiment: str,
        fit_diagnostics: Optional[dict[str, object]] = None,
    ) -> TphiResult:
        diagnostics = {}
        if fit_diagnostics is not None:
            diagnostics.update(fit_diagnostics)

        return TphiResult(
            value=value,
            metadata={
                'method': method,
                'experiment': experiment,
                'source': 'readout-cavity',
            },
            fit_diagnostics=diagnostics,
        )

    def cal_read_tphi(self, kappa: float = None, method: str = 'cal', chi: float = None, 
                      read_freq: float = 6.5e9, noise_freq: np.ndarray = None, 
                      experiment: str = 'Ramsey', 
                      delay_list: np.ndarray = np.linspace(10, 10e3, 100)*1e-9, 
                      N: int = 100, len_pi: float = 100e-9,
                      p0: Optional[np.ndarray] = [100e-6, 3e-6, 1., 0.],bounds=([0, 0, 0, -1], [np.inf, np.inf, 2, 1]), is_plot: bool = True, is_print: bool = True) -> TphiResult:
        """
        Calculate dephasing time Tphi limited by readout cavity thermal photons.
        
        Args:
            kappa (float): The noise characteristic frequency (or decay rate) in [Hz].
            method (str, optional): The method to use for calculation. Defaults to 'cal'.
            chi (float, optional): The dispersive shift in [Hz]. Defaults to None.
            read_freq (float, optional): The readout frequency. Defaults to 6.5e9.
            noise_freq (np.ndarray, optional): The noise frequency array. Defaults to None.
            experiment (str, optional): The experiment type. Defaults to 'Ramsey'.
            delay_list (np.ndarray, optional): The delay list array. Defaults to np.linspace(10, 10e3, 100)*1e-9.
            N (int, optional): The number of pulse blocks (or pulses). Defaults to 100.
            len_pi (float, optional): The finite width of each pulse. Defaults to 100e-9.
            p0 (Optional[np.ndarray], optional): The initial guess for fitting. Defaults to [100e-6, 3e-6, 1., 0.].
            bounds (tuple, optional): The bounds for fitting. Defaults to ([0, 0, 0, -1], [np.inf, np.inf, 2, 1]).
            is_plot (bool, optional): Whether to plot the results. Defaults to True.
            is_print (bool, optional): Whether to print the results. Defaults to True.
        
        Returns:
            TphiResult: The calculated dephasing time Tphi in [s].
        """
        
        chi_hz = self.couple_term if chi is None else chi
        kappa_hz = self.kappa if kappa is None else kappa
        chi = chi_hz * 2*pi
        kappa = kappa_hz * 2*pi

        if method == 'cal':
            self.tphi_rc = self.r_analyzer.calculate_tphi_cal(
                n_bar=self.n_bar,
                kappa=kappa,
                chi=chi,
            )
            result = self._build_read_tphi_result(
                value=self.tphi_rc,
                method='cal',
                experiment=experiment,
            )
            if is_print:
                self._emit_report(
                    format_readout_tphi_report(
                        method='cal',
                        experiment=experiment,
                        read_freq=read_freq,
                        n_bar=self.n_bar,
                        kappa_hz=kappa_hz,
                        chi_hz=chi_hz,
                        tphi=self.tphi_rc,
                    ),
                    is_print=True,
                )
            return result

        elif method == 'fit':
            self.cal_readcavity_psd(kappa=kappa, chi=chi, read_freq=read_freq, noise_freq=noise_freq)
            
            p1_list = self.cal_read_dephase(
                kappa=kappa,
                chi=chi,
                experiment=experiment, 
                read_freq=read_freq, 
                delay_list=delay_list,
                N=N,
                len_pi=len_pi,
            )
            
            popt, pcov = fit_decay(
                delay_list, 
                p1_list, 
                tphi_decay, 
                p0=p0, 
                bounds=bounds
            )
            self.tphi_rc = popt[0]
            self.tphi2 = popt[1]
            self.dephase = p1_list

            if is_plot:
                plot_read_tphi_fit(
                    delay_list=delay_list,
                    dephase=p1_list,
                    popt=popt,
                )
            diagnostics = {
                'tphi2': float(self.tphi2),
                'fit_error': float(np.sqrt(pcov[0][0])),
                'tphi2_fit_error': float(np.sqrt(pcov[1][1])),
            }
            result = self._build_read_tphi_result(
                value=self.tphi_rc,
                method='fit',
                experiment=experiment,
                fit_diagnostics=diagnostics,
            )
            if is_print:
                self._emit_report(
                    format_readout_tphi_report(
                        method='fit',
                        experiment=experiment,
                        read_freq=read_freq,
                        n_bar=self.n_bar,
                        kappa_hz=kappa_hz,
                        chi_hz=chi_hz,
                        tphi=self.tphi_rc,
                        fit_diagnostics=diagnostics,
                    ),
                    is_print=True,
                )
            return result

        else:
            raise ValueError(f"Method {method} not supported.")
