"""Compatibility-only qubit placeholders.

These helpers keep historical method names available while making their status
explicit. They should not grow primary implementation logic; stable workflows
belong in the split qubit modules and package-level exports.
"""

from .experimental import raise_qubit_feature_boundary


def _hamiltonian_evo_set_inistate(self, initial_state):
    raise_qubit_feature_boundary(
        "HamiltonianEvo",
        "set_inistate",
        boundary="compatibility",
        replacement="passing the initial state directly into your QuTiP solve flow",
        details="The old state-setting hook is retained only as a clear failure boundary.",
    )


def _hamiltonian_evo_hamiltonian_evolution(self, *args, **kwargs):
    raise_qubit_feature_boundary(
        "HamiltonianEvo",
        "hamiltonian_evolution",
        boundary="compatibility",
        replacement="an explicit QuTiP solve flow built from HamiltonianEvo.get_hamiltonian()",
        details="The old monolithic evolution hook is retained only as a clear failure boundary.",
    )


def _single_qubit_base_envs_readout_photon(self):
    raise_qubit_feature_boundary(
        "SingleQubitBase",
        "EnvsReadoutphoton",
        boundary="compatibility",
        details="The legacy environment-helper family has no stable split-module implementation yet.",
    )


def _single_qubit_base_envs_capa(self):
    raise_qubit_feature_boundary(
        "SingleQubitBase",
        "EnvsCapa",
        boundary="compatibility",
        details="The legacy environment-helper family has no stable split-module implementation yet.",
    )


def _single_qubit_base_envs_induc(self):
    raise_qubit_feature_boundary(
        "SingleQubitBase",
        "EnvsInduc",
        boundary="compatibility",
        details="The legacy environment-helper family has no stable split-module implementation yet.",
    )


def _single_qubit_base_envs_junc_resis(self):
    raise_qubit_feature_boundary(
        "SingleQubitBase",
        "EnvsJuncResis",
        boundary="compatibility",
        details="The legacy environment-helper family has no stable split-module implementation yet.",
    )


def _raise_single_qubit_readout_inductive_boundary():
    raise_qubit_feature_boundary(
        "SingleQubitBase",
        "get_Readout_parameter",
        boundary="experimental",
        replacement="coupling_mode={'rq': 'capac', ...} or a custom circuit model",
        details="The inductive readout-coupling branch is intentionally not advertised as stable.",
    )


def _fgf1v1_qubit_dephasing_by_coupler_thermal(self, coupler_flux):
    raise_qubit_feature_boundary(
        "FGF1V1Coupling",
        "QubitDephasingbyCouplerThermal",
        boundary="experimental",
        replacement="pysuqu.decoherence noise-analysis workflows",
        details="The coupler-thermal dephasing shortcut still needs a validated model boundary.",
    )


def _grounded_transmon_list_init(self, *args, **kwargs):
    raise_qubit_feature_boundary(
        "GroundedTransmonList",
        "__init__",
        boundary="experimental",
        replacement="a list of GroundedTransmon instances or a custom ParameterizedQubit subclass",
    )


def _fgfgg1v1v3_coupling_init(self, *args, **kwargs):
    raise_qubit_feature_boundary(
        "FGFGG1V1V3Coupling",
        "__init__",
        boundary="experimental",
        replacement="QCRFGRModel, FGF1V1Coupling, or FGF2V7Coupling",
    )


__all__ = [
    "_fgf1v1_qubit_dephasing_by_coupler_thermal",
    "_fgfgg1v1v3_coupling_init",
    "_grounded_transmon_list_init",
    "_hamiltonian_evo_hamiltonian_evolution",
    "_hamiltonian_evo_set_inistate",
    "_raise_single_qubit_readout_inductive_boundary",
    "_single_qubit_base_envs_capa",
    "_single_qubit_base_envs_induc",
    "_single_qubit_base_envs_junc_resis",
    "_single_qubit_base_envs_readout_photon",
]
