"""Plotting helpers for decoherence result visualization."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from ..funclib.mathlib import tphi_decay


def plot_z_tphi2_fit(
    *,
    delay_list: np.ndarray,
    dephase: np.ndarray,
    popt: np.ndarray,
    fitted_dephase: np.ndarray | None = None,
    tphi2: float,
    tphi2_fiterror: float,
    experiment: str,
    segment_results: Mapping[str, Mapping[str, np.ndarray]],
    plot_title: str = 'Global Dephasing',
) -> None:
    """Render the Round G fit plot for longitudinal dephasing."""
    n_cols = 2 if segment_results else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(delay_list * 1e6, dephase, 'o', alpha=0.5, label='Sim (Global)')
    ax0.plot(
        delay_list * 1e6,
        tphi_decay(delay_list, *popt) if fitted_dephase is None else fitted_dephase,
        'r--',
        label=rf'Fit Tphi2: {tphi2 * 1e6:.2f} +/- {tphi2_fiterror * 1e6:.2f} us',
    )
    ax0.set_xlabel('Delay (us)')
    ax0.set_ylabel('Dephase factor')
    ax0.set_title(f'{plot_title} ({experiment})')
    ax0.legend()

    if segment_results:
        ax1 = axes[1]
        labels = list(segment_results.keys())
        values = [entry['popt'][1] * 1e6 for entry in segment_results.values()]

        bars = ax1.bar(labels, values, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Tphi2 (us)')
        ax1.set_title('Tphi2 Contribution by Frequency Band')
        ax1.tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
            )

        ax1.axhline(tphi2 * 1e6, color='r', linestyle='--', label='Global Tphi')
        ax1.legend()

    fig.tight_layout()
    plt.show()


def plot_z_tphi2_multipsd_fit(
    *,
    delay_list: np.ndarray,
    series: Mapping[str, Mapping[str, object]],
    experiment: str,
    plot_title: str = 'Multi-PSD Dephasing',
) -> None:
    """Render simulated and fitted Tphi2 curves for several PSD inputs."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    delay_us = np.asarray(delay_list) * 1e6

    for index, (label, entry) in enumerate(series.items()):
        color = f'C{index % 10}'
        ax.plot(
            delay_us,
            entry['dephase'],
            linestyle='none',
            marker='o',
            markersize=3.5,
            alpha=0.4,
            color=color,
        )
        ax.plot(
            delay_us,
            entry['fitted_dephase'],
            linestyle='--',
            linewidth=1.6,
            color=color,
            label=(
                f"{label}: Tphi2 = {entry['tphi2'] * 1e6:.2f} "
                f"+/- {entry['tphi2_fiterror'] * 1e6:.2f} us"
            ),
        )

    ax.set_xlabel('Delay (us)')
    ax.set_ylabel('Dephase factor')
    ax.set_title(f'{plot_title} ({experiment})')
    ax.legend(loc='upper right', borderaxespad=0.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_read_tphi_fit(
    *,
    delay_list: np.ndarray,
    dephase: np.ndarray,
    popt: np.ndarray,
) -> None:
    """Render the Round G fit plot for readout-cavity dephasing."""
    plt.plot(delay_list * 1e6, dephase, label='Dephase')
    plt.plot(delay_list * 1e6, tphi_decay(delay_list, *popt), label='Fitted Tphi')
    plt.xlabel('Delay (us)')
    plt.ylabel('Dephase factor')
    plt.legend()
    plt.show()
