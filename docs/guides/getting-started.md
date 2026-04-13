# Getting Started

This guide is for users who want to install `pysuqu`, import the public API,
and run a minimal simulation example.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

If you want build artifacts as part of release validation:

```bash
python -m build
```

## Recommended Imports

For qubit modeling, prefer package-level imports:

```python
from pysuqu.qubit import (
    AbstractQubit,
    ParameterizedQubit,
    GroundedTransmon,
    FloatingTransmon,
    QCRFGRModel,
    FGF1V1Coupling,
    FGF2V7Coupling,
    HamiltonianEvo,
    Phi0,
    pi,
)
```

For decoherence analysis, prefer:

```python
from pysuqu.decoherence import (
    Decoherence,
    ElectronicNoise,
    NoiseFitResult,
    NoisePipelineStage,
    T1Result,
    TphiResult,
    ZNoiseDecoherence,
    XYNoiseDecoherence,
    RNoiseDecoherence,
)
```

## Minimal Example

```python
import numpy as np

from pysuqu.qubit import AbstractQubit
from pysuqu.decoherence import ZNoiseDecoherence

qubit = AbstractQubit(
    frequency=5e9,
    anharmonicity=-250e6,
    frequency_max=6e9,
    qubit_type="Transmon",
    energy_trunc_level=12,
)

energies = qubit.get_energylevel()
print(energies[:3])

psd_freq = np.logspace(-4, 8, 100000)
psd_s = 5e-16 / psd_freq + 4e-21

result = ZNoiseDecoherence(
    psd_freq=psd_freq,
    psd_S=psd_s,
    qubit_freq=5e9,
    qubit_anharm=-250e6,
).cal_tphi2(method="cal", idle_freq=5e9, is_print=False)

print(result)
```

## Units

- Use SI units for inputs unless an API explicitly documents otherwise.
- Frequency parameters are typically provided in `Hz`.
- Public docs and printed summaries may also display `GHz`, `MHz`, `s`, or
  `us` for readability.

## Repository Policy

This public repository ships a fresh `demo/` directory with public tutorial
notebooks and synthetic data only. The old private notebook set is still
excluded.

## Public Demo Notebooks

- [../../demo/demo_01_single_qubit_basics.ipynb](../../demo/demo_01_single_qubit_basics.ipynb)
- [../../demo/demo_02_decoherence_with_synthetic_noise.ipynb](../../demo/demo_02_decoherence_with_synthetic_noise.ipynb)
- [../../demo/demo_03_waveform_and_gate_basics.ipynb](../../demo/demo_03_waveform_and_gate_basics.ipynb)
- [../../demo/demo_04_multiqubit_coupler_workflow.ipynb](../../demo/demo_04_multiqubit_coupler_workflow.ipynb)

## Next References

- [../architecture/module-map.md](../architecture/module-map.md)
- [../architecture/refactor-status.md](../architecture/refactor-status.md)
- [code-style.md](code-style.md)

