# Getting Started

This guide is for users who want to install `pysuqu`, import the public API,
and run a minimal simulation example.

## Install

For most users, install the published package from PyPI:

```bash
pip install pysuqu
```

If you want the latest repository version for development:

```bash
git clone https://github.com/znb888/pysuqu.git
cd pysuqu
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

## Experimental / Compatibility Boundary

- Prefer the stable package exports under `pysuqu.qubit` for normal work.
- Exploratory placeholder surfaces live under `pysuqu.qubit.experimental`.
- Retained legacy placeholders now raise `QubitFeatureBoundaryError` so the
  boundary fails explicitly instead of looking like a partial implementation.
- Standard single-qubit, multi-qubit, and decoherence workflows should not
  need anything from the experimental module.

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

This public repository ships `demo/` notebooks and synthetic data only. The
old private notebook set is still excluded.

## Public Demo Notebooks

- [../../demo/README.md](../../demo/README.md) for the public notebook index
- [../../demo/demo_01_single_qubit_basics.ipynb](../../demo/demo_01_single_qubit_basics.ipynb)
- [../../demo/demo_02_decoherence.ipynb](../../demo/demo_02_decoherence.ipynb)
- [../../demo/demo_03_waveform_and_gate_basics.ipynb](../../demo/demo_03_waveform_and_gate_basics.ipynb)
- [../../demo/demo_04_dynamic_simulation.ipynb](../../demo/demo_04_dynamic_simulation.ipynb)

## Next References

- [../architecture/module-map.md](../architecture/module-map.md)
- [../architecture/refactor-status.md](../architecture/refactor-status.md)
- [code-style.md](code-style.md)

