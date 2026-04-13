import matplotlib.pyplot as plt

from .types import CouplingResult, SweepResult


def plot_multi_qubit_energy_vs_flux(
    coupler_flux: list[float],
    cal_state: list[list[int]],
    energy_result: SweepResult,
) -> None:
    """Plot representative multi-qubit energies across coupler flux bias points."""
    x_values = energy_result.sweep_values
    energy_series = energy_result.series

    plt.figure()
    for state in cal_state:
        plt.plot(x_values, energy_series[f'|{state}>'], label=rf'$ E_|{state}> $')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'Coupler $flux(\Phi/\Phi_0)$')
    plt.ylabel('Energy(GHz)')
    plt.title("Frequency vs Coupler Flux")
    plt.show()


def plot_multi_qubit_coupling_strength_vs_flux(
    coupler_flux: list[float],
    coupling_result: CouplingResult,
) -> None:
    """Plot representative multi-qubit coupling strengths across coupler flux bias points."""
    x_values = coupling_result.sweep_values
    g_values = coupling_result.coupling_values

    plt.figure()
    plt.plot(x_values, g_values)
    plt.xlabel(r'Coupler $flux(\Phi/\Phi_0)$')
    plt.ylabel(r'Coupling Strength($MHz$)')
    plt.title("Coupling Strength vs Coupler Flux")
    plt.show()
