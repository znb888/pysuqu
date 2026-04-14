'''
Electronics influenced qubit performance calculation.

Author: Naibin Zhou
USTC
Since 2025-12-11
'''
# import
from functools import lru_cache
from types import MappingProxyType
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
    """Electronic control-line noise model with explicit pipeline stages.

    The public contract is centered on two immutable stage snapshots:
    `input_stage` captures the PSD exactly as it enters the attenuation chain and
    `output_stage` captures the PSD after transmission through the configured
    temperature and attenuation setup. Legacy attributes such as
    `psd_double_in`, `psd_double_out`, and `white_noise_out` are preserved as
    aliases of the corresponding stage fields for compatibility.
    """

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
        Build the electronic-noise pipeline from PSD input data.

        Args:
            psd_freq: PSD sample frequencies in Hz.
            psd_S: Noise input. When `is_spectral=True`, this is the PSD sampled
                at `psd_freq` in `A^2/Hz`. When `is_spectral=False`, this is the
                mean white-noise floor in `dBm/Hz`.
            noise_type: Spectral model to fit for spectral inputs. Choose from
                `'1f'`, `'constant'`, or `'1f_bump'`. Non-spectral inputs always
                use the public `'constant'` contract.
            noise_prop: Whether `psd_S` is a `'single'`- or `'double'`-sided
                spectrum.
            T_setup: Temperature chain in K for each electronics stage.
            attenuation_setup: Attenuation chain in dB for each electronics
                stage.
            is_spectral: Controls whether `psd_S` is interpreted as a full PSD
                array (`True`) or as a mean white-noise level (`False`).
            is_print: Whether to emit the formatted pipeline summary during
                construction.

        Side Effects:
            Calls `refresh_model()` immediately, which populates the public
            `input_stage` and `output_stage` snapshots.
        """
        self.noise_freq = psd_freq
        self.psd = psd_S
        self.noise_prop = noise_prop
        self.T_setup = T_setup
        self.attenuation_setup = attenuation_setup
        self.is_spectral = is_spectral
        self.noise_type = self._resolve_noise_type(
            noise_type=noise_type,
            is_spectral=self.is_spectral,
        )

        self.refresh_model(is_print=is_print)

    @staticmethod
    def _resolve_noise_type(*, noise_type: str, is_spectral: bool) -> str:
        if not is_spectral:
            return 'constant'

        return noise_type

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

    @staticmethod
    def _fit_inverse_psd(freq: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
        """Fit `a / f + b` directly for positive PSD data.

        The public `1f` branch is linear in the unknown coefficients once the
        frequency samples are treated as fixed, so a direct least-squares solve
        avoids repeated nonlinear-optimizer overhead on the hot XY constructor
        path. Fall back to the historical `curve_fit` solver if the direct
        solution becomes non-finite or physically implausible.
        """
        design = np.column_stack((1.0 / freq, np.ones_like(freq)))
        coeffs, _, _, _ = np.linalg.lstsq(design, psd, rcond=None)
        a_fit, b_fit = (float(value) for value in coeffs)
        if np.isfinite(a_fit) and np.isfinite(b_fit) and a_fit >= 0.0 and b_fit > 0.0:
            return a_fit, b_fit

        tail_len = max(1, len(psd) // 10)
        b_guess = float(np.mean(psd[-tail_len:]))
        idx_1 = int(np.argmin(np.abs(freq - 1.0)))
        if 0.1 <= freq[idx_1] <= 10.0:
            target_idx = idx_1
        else:
            target_idx = 0

        x0 = float(freq[target_idx])
        y0 = float(psd[target_idx])
        a_guess = (y0 - b_guess) * x0
        if a_guess <= 0.0:
            a_guess = 1.0
        popt, _ = curve_fit(inverse_func, freq, psd, p0=[a_guess, b_guess], maxfev=5000)
        return float(popt[0]), float(popt[1])

    @staticmethod
    @lru_cache(maxsize=32)
    def _smooth_psd_cached(
        data_bytes: bytes,
        dtype_str: str,
        method: str,
        window: int,
        polyorder: int,
        sigma: float,
    ) -> np.ndarray:
        data = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str))
        smoothed = np.asarray(
            smooth_data(
                data,
                method=method,
                window=window,
                polyorder=polyorder,
                sigma=sigma,
            )
        )
        smoothed.setflags(write=False)
        return smoothed

    @staticmethod
    def _smooth_psd(
        data: np.ndarray,
        *,
        method: str = 'savgol',
        window: int = 11,
        polyorder: int = 3,
        sigma: float = 2.0,
    ) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(data))
        cached = ElectronicNoise._smooth_psd_cached(
            array.tobytes(),
            array.dtype.str,
            method,
            int(window),
            int(polyorder),
            float(sigma),
        )
        # Preserve per-instance isolation even when the smoothing result is cached.
        return np.array(cached, copy=True)

    @staticmethod
    def _pack_array_cache_key(array: np.ndarray) -> tuple[bytes, str]:
        packed = np.ascontiguousarray(np.asarray(array))
        return packed.tobytes(), packed.dtype.str

    @staticmethod
    def _freeze_cached_array(array: np.ndarray) -> np.ndarray:
        frozen = np.array(np.asarray(array), copy=True)
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _fit_psd_uncached(
        psd: np.ndarray,
        freq: np.ndarray,
        *,
        noise_type: str = '1f',
        noise_prop: str = 'single',
    ) -> dict[str, float]:
        result = {}

        abs_freq = np.abs(freq)

        valid_mask = (psd > 0) & (abs_freq > 1e-9)
        sort_idx = np.argsort(abs_freq[valid_mask])
        f_data = abs_freq[valid_mask][sort_idx]
        s_data = psd[valid_mask][sort_idx]
        lg_f = np.log10(f_data)
        lg_s = np.log10(s_data)

        if noise_type == '1f':
            a_fit, b_fit = ElectronicNoise._fit_inverse_psd(f_data, s_data)
            psd_fit = inverse_func(f_data, a_fit, b_fit)
            x_knee, _ = find_knee_point(f_data, psd_fit)
            white_region = f_data[f_data > x_knee]
            if white_region.size > 0:
                white_noise_freq = float(np.mean(white_region))
            else:
                white_noise_freq = float(f_data[-1])

            result['white_noise'] = float(b_fit)
            result['1f_coef'] = float(a_fit)
            result['corner_freq'] = float(x_knee)
            result['white_ref_freq'] = white_noise_freq

        elif noise_type == '1f_bump':
            fit_len = max(5, int(len(f_data) * 0.15))
            f_1f_region = f_data[:fit_len]
            s_1f_region = s_data[:fit_len]

            lg_A_estimates = np.log10(s_1f_region) + 1.0 * np.log10(f_1f_region)
            lg_A_fit = np.median(lg_A_estimates)
            result['1f_coef'] = float(10**lg_A_fit)

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
            result['corner_freq'] = float(corner_freq)

            mask_white = f_data > corner_freq

            if np.any(mask_white):
                white_noise_val = float(np.mean(s_data[mask_white]))
                white_ref_freq = float(np.sqrt(f_data[mask_white][0] * f_data[mask_white][-1]))
            else:
                white_noise_val = float(np.mean(s_data[-5:]))
                white_ref_freq = float(f_data[-1])

            result['white_noise'] = white_noise_val
            result['white_ref_freq'] = white_ref_freq

        elif noise_type == 'constant':
            white_noise = float(np.mean(psd))
            result['white_noise'] = white_noise
            result['1f_coef'] = 0.0
            result['corner_freq'] = 0.0
            result['white_ref_freq'] = float(np.median(freq))

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

        result['white_noise_temperature'] = float(temp)

        return result

    @staticmethod
    @lru_cache(maxsize=32)
    def _fit_psd_cached(
        psd_bytes: bytes,
        psd_dtype_str: str,
        freq_bytes: bytes,
        freq_dtype_str: str,
        noise_type: str,
        noise_prop: str,
    ) -> dict[str, float]:
        psd = np.frombuffer(psd_bytes, dtype=np.dtype(psd_dtype_str))
        freq = np.frombuffer(freq_bytes, dtype=np.dtype(freq_dtype_str))
        return ElectronicNoise._fit_psd_uncached(
            psd,
            freq,
            noise_type=noise_type,
            noise_prop=noise_prop,
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _build_spectral_pipeline_cached(
        freq_bytes: bytes,
        freq_dtype_str: str,
        psd_bytes: bytes,
        psd_dtype_str: str,
        t_setup_bytes: bytes,
        t_setup_dtype_str: str,
        attenuation_bytes: bytes,
        attenuation_dtype_str: str,
        noise_type: str,
        noise_prop: str,
    ) -> MappingProxyType:
        freq = np.frombuffer(freq_bytes, dtype=np.dtype(freq_dtype_str))
        psd = np.frombuffer(psd_bytes, dtype=np.dtype(psd_dtype_str))
        t_setup = np.frombuffer(t_setup_bytes, dtype=np.dtype(t_setup_dtype_str))
        attenuation_setup = np.frombuffer(
            attenuation_bytes,
            dtype=np.dtype(attenuation_dtype_str),
        )

        if noise_prop == 'single':
            psd_double_in = Sii_S2D(psd, freq)
        else:
            psd_double_in = np.asarray(psd)

        psd_single_in = Sii_D2S(psd_double_in, freq)
        psd_smooth_in = ElectronicNoise._smooth_psd(psd_double_in)
        psd_double_out = S_transmission(
            psd_double_in,
            freq,
            t_setup,
            attenuation_setup,
        )
        psd_single_out = Sii_D2S(psd_double_out, freq)
        psd_smooth_out = ElectronicNoise._smooth_psd(psd_double_out)

        noise_fitres_in = ElectronicNoise.fit_psd(
            psd_smooth_in,
            freq,
            noise_type=noise_type,
            noise_prop='double',
        )
        noise_fitres_out = ElectronicNoise.fit_psd(
            psd_smooth_out,
            freq,
            noise_type=noise_type,
            noise_prop='double',
        )

        return MappingProxyType(
            {
                'psd_double_in': ElectronicNoise._freeze_cached_array(psd_double_in),
                'psd_single_in': ElectronicNoise._freeze_cached_array(psd_single_in),
                'psd_smooth_in': ElectronicNoise._freeze_cached_array(psd_smooth_in),
                'psd_double_out': ElectronicNoise._freeze_cached_array(psd_double_out),
                'psd_single_out': ElectronicNoise._freeze_cached_array(psd_single_out),
                'psd_smooth_out': ElectronicNoise._freeze_cached_array(psd_smooth_out),
                'noise_fitres_in': MappingProxyType(dict(noise_fitres_in)),
                'noise_fitres_out': MappingProxyType(dict(noise_fitres_out)),
            }
        )

    @staticmethod
    def _build_spectral_pipeline(
        *,
        freq: np.ndarray,
        psd: np.ndarray,
        t_setup: np.ndarray,
        attenuation_setup: np.ndarray,
        noise_type: str,
        noise_prop: str,
    ) -> MappingProxyType:
        freq_bytes, freq_dtype_str = ElectronicNoise._pack_array_cache_key(freq)
        psd_bytes, psd_dtype_str = ElectronicNoise._pack_array_cache_key(psd)
        t_setup_bytes, t_setup_dtype_str = ElectronicNoise._pack_array_cache_key(t_setup)
        attenuation_bytes, attenuation_dtype_str = ElectronicNoise._pack_array_cache_key(
            attenuation_setup
        )
        return ElectronicNoise._build_spectral_pipeline_cached(
            freq_bytes,
            freq_dtype_str,
            psd_bytes,
            psd_dtype_str,
            t_setup_bytes,
            t_setup_dtype_str,
            attenuation_bytes,
            attenuation_dtype_str,
            noise_type,
            noise_prop,
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
        """Rebuild the pipeline stages from the current PSD and setup state.

        After a successful refresh, `input_stage` and `output_stage` are
        replaced with new `NoisePipelineStage` snapshots whose `frequency`
        fields remain in Hz and whose PSD fields remain in `A^2/Hz`. The legacy
        alias attributes (`psd_double_in`, `psd_double_out`, `white_noise_out`,
        and related fields) are kept synchronized with those stage snapshots for
        compatibility. If the model has been switched into non-spectral mode,
        the public `noise_type` contract is normalized back to `constant`
        before stage rebuilding so the facade state and pipeline snapshots stay
        aligned.

        Args:
            is_print: Whether to emit the formatted pipeline summary.

        Returns:
            bool: Always `True` after the stages and compatibility aliases are
            refreshed successfully.
        """
        self.noise_type = self._resolve_noise_type(
            noise_type=self.noise_type,
            is_spectral=self.is_spectral,
        )
        noise_fitres_in = None
        noise_fitres_out = None

        if self.is_spectral:
            spectral_pipeline = self._build_spectral_pipeline(
                freq=self.noise_freq,
                psd=self.psd,
                t_setup=self.T_setup,
                attenuation_setup=self.attenuation_setup,
                noise_type=self.noise_type,
                noise_prop=self.noise_prop,
            )

            psd_double_in = spectral_pipeline['psd_double_in']
            psd_single_in = spectral_pipeline['psd_single_in']
            psd_smooth_in = spectral_pipeline['psd_smooth_in']
            psd_double_out = spectral_pipeline['psd_double_out']
            psd_single_out = spectral_pipeline['psd_single_out']
            psd_smooth_out = spectral_pipeline['psd_smooth_out']
            noise_fitres_in = dict(spectral_pipeline['noise_fitres_in'])
            noise_fitres_out = dict(spectral_pipeline['noise_fitres_out'])

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
        psd_array = np.ascontiguousarray(np.asarray(psd))
        freq_array = np.ascontiguousarray(np.asarray(freq))

        if len(psd_array) != len(freq_array):
            raise ValueError(f"Length of psd {len(psd)} doesn't match length of freq {len(freq)} !")
        cached_result = ElectronicNoise._fit_psd_cached(
            psd_array.tobytes(),
            psd_array.dtype.str,
            freq_array.tobytes(),
            freq_array.dtype.str,
            noise_type,
            noise_prop,
        )
        return dict(cached_result)
        


