'''
Electronics influenced qubit performance calculation.

Author: Naibin Zhou
USTC
Since 2025-12-11
'''
# import
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from typing import Union
# local lib
from ..funclib.noisemodel import *
from ..funclib.mathlib import smooth_data, inverse_func, find_knee_point
from .formatting import format_electronic_noise_report
from .results import NoiseFitResult, NoisePipelineStage

class ElectronicNoise():
    
    def __init__(
        self,
        psd_freq: Union[np.ndarray, float],
        psd_S: Union[np.ndarray, float],
        noise_type: str = '1f',
        noise_prop: str = 'single',
        T_setup: np.ndarray = np.array([290,45,3.5,0.9,0.1,0.01]),
        attenuation_setup: np.ndarray = np.array([40,1,10,10,20,10]),
        is_spectral: bool = True,
        is_print: bool = True,
        *args,**kwargs
    ):
        """
        Electronic noise model.

        Args:
            psd_freq (np.ndarray): frequency array in Hz.
            psd_S (np.ndarray): power spectral density array in A^2/Hz.
            noise_type (str, optional): spectrum type of noise. Defaults to '1f'. Choose in ['1f', 'constant', '1f_bump']
            noise_prop (str, optional): single-sided or double-sided noise spectrum. Default to 'single'
            T_setup (np.ndarray, optional): . Defaults to np.array([290,45,3.5,0.9,0.1,0.01]).
            attenuation_setup (np.ndarray, optional): . Defaults to np.array([40,1,10,10,20,10]).
            is_spectral (bool, optional): Define the form of psd input, if True, psd_S is the power spectral density array in A^2/Hz,else psd_S is the mean PSD in dBm/Hz. Defaults to True.
        """
        self.noise_freq = psd_freq
        self.psd = psd_S
        self.noise_prop = noise_prop
        self.T_setup = T_setup
        self.attenuation_setup = attenuation_setup
        self.is_spectral = is_spectral
        if self.is_spectral:
            self.noise_type = noise_type
        else:
            self.noise_type = 'constant'
        
        self.refresh_model(is_print=is_print)

    @staticmethod
    def _build_noise_stage(
        *,
        stage: str,
        frequency,
        psd_double,
        psd_single,
        psd_smooth,
        white_noise,
        white_ref_freq,
        white_noise_temperature,
        noise_type: str,
        fit_data: dict | None = None,
    ) -> NoisePipelineStage:
        fit_result = None
        if fit_data is not None:
            fit_result = NoiseFitResult.from_fit_dict(
                fit_data,
                noise_type=noise_type,
                noise_prop='double',
                metadata={
                    'source': 'ElectronicNoise.refresh_model',
                    'stage': stage,
                },
            )

        return NoisePipelineStage(
            frequency=frequency,
            psd_double=psd_double,
            psd_single=psd_single,
            psd_smooth=psd_smooth,
            white_noise=white_noise,
            white_ref_freq=white_ref_freq,
            white_noise_temperature=white_noise_temperature,
            fit_result=fit_result,
        )

    def _sync_pipeline_aliases(
        self,
        *,
        input_stage: NoisePipelineStage,
        output_stage: NoisePipelineStage,
        noise_fitres_in: dict | None = None,
        noise_fitres_out: dict | None = None,
    ) -> None:
        self.input_stage = input_stage
        self.output_stage = output_stage

        self.psd_double_in = self.input_stage.psd_double
        self.psd_smooth_in = self.input_stage.psd_smooth
        self.psd_double_out = self.output_stage.psd_double
        self.psd_single_out = self.output_stage.psd_single
        self.psd_smooth_out = self.output_stage.psd_smooth

        self.white_noise_in = self.input_stage.white_noise
        self.white_noise_out = self.output_stage.white_noise
        self.white_ref_freq_in = self.input_stage.white_ref_freq
        self.white_ref_freq_out = self.output_stage.white_ref_freq
        self.white_noise_temperature_in = self.input_stage.white_noise_temperature
        self.white_noise_temperature_out = self.output_stage.white_noise_temperature

        if noise_fitres_in is not None:
            self.noise_fitres_in = dict(noise_fitres_in)
        elif hasattr(self, 'noise_fitres_in'):
            delattr(self, 'noise_fitres_in')

        if noise_fitres_out is not None:
            self.noise_fitres_out = dict(noise_fitres_out)
        elif hasattr(self, 'noise_fitres_out'):
            delattr(self, 'noise_fitres_out')
        
    def refresh_model(self, is_print: bool = True):
        """
        Refresh the model with new data.
        """
        noise_fitres_in = None
        noise_fitres_out = None

        if self.is_spectral:
            if self.noise_prop == 'single':
                psd_double_in = Sii_S2D(self.psd, self.noise_freq)
            else:
                psd_double_in = self.psd
            
            psd_single_in = Sii_D2S(psd_double_in, self.noise_freq)
            psd_smooth_in = smooth_data(psd_double_in)
            psd_double_out = np.array([S_transmission(p,f,self.T_setup,self.attenuation_setup) for p, f in zip(psd_double_in, self.noise_freq)])
            psd_single_out = Sii_D2S(psd_double_out, self.noise_freq)
            psd_smooth_out = smooth_data(psd_double_out)
            
            noise_fitres_in = self.fit_psd(psd_smooth_in, self.noise_freq, noise_type=self.noise_type, noise_prop='double')
            noise_fitres_out = self.fit_psd(psd_smooth_out, self.noise_freq, noise_type=self.noise_type, noise_prop='double')

            white_noise_in = noise_fitres_in['white_noise']
            white_noise_out = noise_fitres_out['white_noise']
            white_ref_freq_in = noise_fitres_in['white_ref_freq']
            white_ref_freq_out = noise_fitres_out['white_ref_freq']
            white_noise_temperature_in = noise_fitres_in['white_noise_temperature']
            white_noise_temperature_out = noise_fitres_out['white_noise_temperature']
            
        else:
            if self.noise_prop == 'single':
                psd_double_in = Sii_S2D(Sii_dBm2A(self.psd), self.noise_freq)
            else:
                psd_double_in = Sii_dBm2A(self.psd)
            
            psd_single_in = Sii_D2S(psd_double_in, self.noise_freq)
            psd_smooth_in = psd_double_in
            psd_double_out = S_transmission(psd_double_in,self.noise_freq,self.T_setup,self.attenuation_setup)
            psd_single_out = Sii_D2S(psd_double_out, self.noise_freq)
            psd_smooth_out = psd_double_out

            white_noise_in = psd_smooth_in
            white_noise_out = psd_smooth_out
            white_ref_freq_in = self.noise_freq
            white_ref_freq_out = self.noise_freq
            white_noise_temperature_in = Sii2T_Double(white_noise_in, white_ref_freq_in)
            white_noise_temperature_out = Sii2T_Double(white_noise_out, white_ref_freq_out)

        input_stage = self._build_noise_stage(
            stage='input',
            frequency=self.noise_freq,
            psd_double=psd_double_in,
            psd_single=psd_single_in,
            psd_smooth=psd_smooth_in,
            white_noise=white_noise_in,
            white_ref_freq=white_ref_freq_in,
            white_noise_temperature=white_noise_temperature_in,
            noise_type=self.noise_type,
            fit_data=noise_fitres_in,
        )
        output_stage = self._build_noise_stage(
            stage='output',
            frequency=self.noise_freq,
            psd_double=psd_double_out,
            psd_single=psd_single_out,
            psd_smooth=psd_smooth_out,
            white_noise=white_noise_out,
            white_ref_freq=white_ref_freq_out,
            white_noise_temperature=white_noise_temperature_out,
            noise_type=self.noise_type,
            fit_data=noise_fitres_out,
        )
        self._sync_pipeline_aliases(
            input_stage=input_stage,
            output_stage=output_stage,
            noise_fitres_in=noise_fitres_in,
            noise_fitres_out=noise_fitres_out,
        )

        if is_print:
            for line in format_electronic_noise_report(
                noise_type=self.noise_type,
                noise_prop=self.noise_prop,
                is_spectral=self.is_spectral,
                input_stage=self.input_stage,
                output_stage=self.output_stage,
                attenuation_setup=self.attenuation_setup,
            ):
                print(line)
            
        return True

    @staticmethod
    def fit_psd(psd: np.ndarray, freq: np.ndarray, noise_type: str = '1f', noise_prop: str = 'single') -> dict:
        """
        Fit the power spectral density to the noise model.

        Returns:
            dict: A dictionary containing extracted noise parameters.
                Keys: 'white_noise', 'white_noise_temp', '1f_coef', 'corner_freq', etc.
        """
        if len(psd) != len(freq):
            raise ValueError(f"Length of psd {len(psd)} doesn't match length of freq {len(freq)} !")

        result = {}

        abs_freq = np.abs(freq)

        valid_mask = (psd > 0) & (abs_freq > 1e-9)
        sort_idx = np.argsort(abs_freq[valid_mask])
        f_data = abs_freq[valid_mask][sort_idx]
        s_data = psd[valid_mask][sort_idx]
        lg_f = np.log10(f_data)
        lg_s = np.log10(s_data)

        if noise_type == '1f':
            tail_len = max(1, len(psd) // 10)
            b_guess = np.mean(psd[-tail_len:])
            idx_1 = np.argmin(np.abs(freq - 1.0))
   
            if 0.1 <= freq[idx_1] <= 10.0:
                target_idx = idx_1
            else:
                target_idx = 0

            x0 = freq[target_idx]
            y0 = psd[target_idx]
           
            a_guess = (y0 - b_guess) * x0
            if a_guess <= 0:
                a_guess = 1.0
            p0 = [a_guess, b_guess]
            popt, pcov = curve_fit(inverse_func, freq, psd, p0=p0, maxfev=5000)
            a_fit, b_fit = popt

            psd_fit = inverse_func(freq, a_fit, b_fit)
            x_knee, knee_idx = find_knee_point(freq, psd_fit)

            white_noise_freq = np.mean(freq[freq > x_knee])

            result['white_noise'] = b_fit
            result['1f_coef'] = a_fit
            result['corner_freq'] = x_knee
            result['white_ref_freq'] = white_noise_freq

        elif noise_type == '1f_bump':
            fit_len = max(5, int(len(f_data) * 0.15))
            f_1f_region = f_data[:fit_len]
            s_1f_region = s_data[:fit_len]

            lg_A_estimates = np.log10(s_1f_region) + 1.0 * np.log10(f_1f_region)
            lg_A_fit = np.median(lg_A_estimates)
            result['1f_coef'] = 10**lg_A_fit

            window_len = min(51, len(lg_s) if len(lg_s) % 2 != 0 else len(lg_s)-1)
            if window_len > 3:
                lg_s_smooth = savgol_filter(lg_s, window_len, 3)
            else:
                lg_s_smooth = lg_s

            ds = np.gradient(lg_s_smooth, lg_f)
            search_start_idx = int(len(ds) * 0.3)
            
            if search_start_idx < len(ds) - 2:
                local_min_slope_idx = np.argmin(ds[search_start_idx:]) + search_start_idx
            else:
                local_min_slope_idx = 0
            
            knee_search_f = lg_f[local_min_slope_idx:]
            knee_search_s = lg_s_smooth[local_min_slope_idx:]
            
            knee_lg_f, _ = find_knee_point(knee_search_f, knee_search_s)
            
            corner_freq = 10**knee_lg_f
            result['corner_freq'] = corner_freq

            mask_white = f_data > corner_freq
            
            if np.any(mask_white):
                white_noise_val = np.mean(s_data[mask_white])
                white_ref_freq = np.sqrt(f_data[mask_white][0] * f_data[mask_white][-1])
            else:
                white_noise_val = np.mean(s_data[-5:])
                white_ref_freq = f_data[-1]

            result['white_noise'] = white_noise_val
            result['white_ref_freq'] = white_ref_freq

        elif noise_type == 'constant':
            white_noise = np.mean(psd)
            result['white_noise'] = white_noise
            result['1f_coef'] = 0.0
            result['corner_freq'] = 0.0
            result['white_ref_freq'] = np.median(freq)
            
        else:
            raise ValueError("Noise_type must be in ['1f', 'constant', '1f_bump']")

        white_noise_val = result['white_noise']
        ref_freq = result['white_ref_freq']

        if noise_prop == 'double':
            temp = Sii2T_Double(white_noise_val, ref_freq)
        elif noise_prop == 'single':
            temp = Sii2T_Single(white_noise_val, ref_freq)
        else:
            raise ValueError(f"Noise prop {noise_prop} not supported! ")
        
        result['white_noise_temperature'] = temp

        return result
        


