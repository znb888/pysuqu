"""
Lib for electrical noise model.

Author: Naibin Zhou
since 2025-12-10
"""

import numpy as np
from scipy.constants import h, hbar, Boltzmann
from typing import Union, Optional

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
    at different temperature stages (Friss formula for passive components).

    Model:
        S_{out} = S_{in} * A + S_{thermal} * (1 - A)
        Where A is the attenuation factor (0 < A < 1).

    Args:
        S_in (float or np.ndarray): Input noise spectral density [A^2/Hz].
        ff (float): Frequency [Hz].
        T_setup (np.ndarray): Array of temperatures for each stage [K].
        attenuationindB (np.ndarray): Array of attenuation values in dB for each stage.

    Returns:
        Union[float, np.ndarray]: Output noise spectral density [A^2/Hz].
    """
    attenuation = 10**(-attenuationindB / 10)

    S_transmission = S_in * attenuation[0] + T2Sii_Double(T_setup[0], ff)

    for ii in range(1, len(T_setup)):
        S_transmission = attenuation[ii] * S_transmission + (1 - attenuation[ii]) * T2Sii_Double(T_setup[ii], ff)
        
    return S_transmission

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



