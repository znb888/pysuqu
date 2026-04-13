# Module Map

Updated for the public `pysuqu 2.0.0` repository.

## Package Layout

```text
pysuqu/
  qubit/
    __init__.py
    analysis.py
    base.py
    circuit.py
    gate.py
    multi.py
    plotting.py
    single.py
    solver.py
    sweeps.py
    types.py
  decoherence/
    __init__.py
    analysis.py
    dequbit.py
    electronics.py
    formatting.py
    noise.py
    plotting.py
    results.py
  funclib/
    __init__.py
    awgenerator.py
    mathlib.py
    noisemodel.py
    qutiplib.py
```

## Recommended Import Surfaces

### Qubit APIs

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

### Decoherence APIs

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

## Responsibilities

### `pysuqu.qubit`

- `base.py`: core qubit abstractions and parameterized qubit foundations
- `circuit.py`: circuit input handling and matrix preparation
- `solver.py`: Hamiltonian assembly and solver helpers
- `single.py`: single-qubit models
- `multi.py`: multi-qubit and coupler models
- `gate.py`: gate-level scheduling and simulation helpers
- `analysis.py`, `sweeps.py`, `plotting.py`, `types.py`: analysis and support
  utilities

### `pysuqu.decoherence`

- `dequbit.py`: top-level decoherence facades
- `electronics.py`: electronic noise modeling and pipelines
- `analysis.py`: numerical analyzers
- `results.py`: typed public result objects
- `plotting.py` and `formatting.py`: presentation helpers
- `noise.py`: supporting noise-domain calculations

### `pysuqu.funclib`

- mathematical helpers
- waveform-generation helpers
- noise-model conversion helpers
- qutip integration helpers

