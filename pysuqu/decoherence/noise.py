"""Decoherence-owned noise helpers."""

from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from ..funclib.mathlib import temp2nbar


def readout_thermal_photon_noise(
    T_cav: float,
    kappa: float,
    chi: float,
    cavity_freq: float,
    psd_freq: np.ndarray,
    S0: float = 0.0,
) -> np.ndarray:
    """Calculate readout-cavity thermal photon noise PSD for decoherence paths."""
    omega_noise = psd_freq * 2 * np.pi
    kappa = kappa * 2 * np.pi
    chi = chi * 2 * np.pi

    nbar = temp2nbar(T_cav * 1e-3, cavity_freq)
    eta = kappa**2 / (kappa**2 + 4 * chi**2)
    nbar_eff = nbar * eta

    return (
        2 * nbar_eff * (nbar_eff + 1) * (2 * chi) ** 2 * (2 * kappa)
        / (omega_noise**2 + kappa**2)
        + S0
    )


def fit_readout_thermal_photon_noise(
    psd_freq: np.ndarray,
    psd: np.ndarray,
    init_guess: Optional[dict] = None,
    T_cav: Optional[float] = None,
    kappa: Optional[float] = None,
    chi: Optional[float] = None,
    S0: Optional[float] = None,
    cavity_freq: float = 6.5e9,
    bounds: Optional[tuple] = None,
    valid_mask: Optional[np.ndarray] = None,
    robust_fit: bool = True,
) -> dict:
    """Fit the readout-cavity thermal photon PSD surface."""
    parameter_defaults = {
        "T_cav": 40.0,
        "kappa": 6e6,
        "chi": 1.5e6,
        "S0": 0.0,
    }
    bounds_defaults = {
        "T_cav": (20, 100),
        "kappa": (0.5e6, 20e6),
        "chi": (0.1e6, 4e6),
        "S0": (-1e-3, 5e4),
    }

    fixed = {
        "T_cav": T_cav,
        "kappa": kappa,
        "chi": chi,
        "S0": S0,
    }
    free_names = [name for name, value in fixed.items() if value is None]

    if not free_names:
        raise ValueError("All parameters are fixed; nothing to fit.")

    defaults = parameter_defaults.copy()
    if init_guess is not None:
        unknown = set(init_guess.keys()) - set(parameter_defaults.keys())
        if unknown:
            raise ValueError(f"Unknown keys in init_guess: {unknown}")
        defaults.update(init_guess)

    p0 = [defaults[name] for name in free_names]
    if bounds is None:
        bounds = (
            [bounds_defaults[name][0] for name in free_names],
            [bounds_defaults[name][1] for name in free_names],
        )

    if valid_mask is not None:
        fit_freq = psd_freq[valid_mask]
        fit_psd = psd[valid_mask]
    else:
        fit_freq = psd_freq
        fit_psd = psd

    def fit_wrapper(frequency, *free_values):
        params = dict(fixed)
        for name, value in zip(free_names, free_values):
            params[name] = value

        return readout_thermal_photon_noise(
            T_cav=params["T_cav"],
            kappa=params["kappa"],
            chi=params["chi"],
            cavity_freq=cavity_freq,
            psd_freq=frequency,
            S0=params["S0"],
        )

    loss_method = "soft_l1" if robust_fit else "linear"

    try:
        popt, pcov = curve_fit(
            fit_wrapper,
            fit_freq,
            fit_psd,
            p0=p0,
            maxfev=10000,
            bounds=bounds,
            loss=loss_method,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"curve_fit failed to converge: {exc}") from exc

    if pcov is not None and not np.isinf(pcov).all():
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.full_like(popt, np.inf)

    fit_values = dict(zip(free_names, popt))
    errors = dict(zip(free_names, perr))
    params = {**fixed, **fit_values}
    params = {
        name: (value if value is not None else fit_values.get(name))
        for name, value in params.items()
    }

    return {
        "params": params,
        "fit_values": fit_values,
        "errors": errors,
        "popt": popt,
        "pcov": pcov,
        "fitted_PSD": fit_wrapper(psd_freq, *popt),
    }


__all__ = [
    "fit_readout_thermal_photon_noise",
    "readout_thermal_photon_noise",
]
