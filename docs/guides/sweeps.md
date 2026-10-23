# Sweep API boundaries

Sweep helpers are public qubit operations, not an implementation detail. They
are available directly from `pysuqu.qubit` as well as
`pysuqu.qubit.sweeps`:

```python
from pysuqu.qubit import (
    sweep_single_qubit_energy_vs_flux,
    sweep_single_qubit_energy_vs_flux_base,
    sweep_multi_qubit_energy_vs_flux,
    sweep_multi_qubit_coupling_strength_vs_flux,
)

energies = sweep_single_qubit_energy_vs_flux(
    qubit, [-0.03, 0.0, 0.025], upper_level=3
)
couplings = sweep_multi_qubit_coupling_strength_vs_flux(
    model, coupler_flux, solver_mode="auto", is_plot=False
)
```

The helpers own the loop, restore the original flux state, select the exact
single-qubit or supported FGF1V1 fast path, and return immutable structured
`SweepResult` or `CouplingResult` values. Use the corresponding
`*_result` function when the explicit structured name is clearer in a larger
pipeline. The lower-level `change_para()` API remains available for custom
update workflows.

`solver_mode="auto"` selects the validated fast path only when the model and
method support it; `"full"` forces the complete recomputation path and
`"fast"` rejects unsupported models instead of silently changing semantics.
No default sweep drops levels or computes only an active low-energy model.
Partial-spectrum helpers, if added later, must remain explicit opt-in APIs.

For a one-node qubit, scalar offsets such as ``[-0.03, 0.0, 0.025]`` are
accepted directly and normalized to the reduced ``(1, 1)`` flux shape.  Matrix
offsets remain supported for multi-node models and continue to undergo the
same shape validation as before.

Cache keys are based on exact model inputs and are invalidated by supported
parameter changes. Do not mutate model arrays in place while a sweep is
running. Independent sweep points can be parallelized by the caller with
separate qubit instances or processes; a shared mutable qubit instance is not a
safe parallel execution boundary.
