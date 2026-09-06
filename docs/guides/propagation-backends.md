# Propagation Backends

`PreparedPropagation` reuses Hamiltonian, sampled control, and solver preparation
across initial states. Gate and multi-drive simulations accept the same backend
selection. The default gate path remains `qutip`.

## Installation

Python 3.9 or newer and QuTiP 5 are required. Install a native wheel where available:

```bash
python -m pip install --upgrade pysuqu
python -c "from pysuqu.qubit import native_backend_available; print(native_backend_available())"
```

The platform-independent wheel supports QuTiP without a compiler. To build the
C++17 extension from source, use a C++ compiler and Python development headers:

```bash
PYSUQU_BUILD_NATIVE=1 python -m pip install --no-binary=pysuqu --no-cache-dir pysuqu
```

In PowerShell:

```powershell
$env:PYSUQU_BUILD_NATIVE = '1'
python -m pip install --no-binary=pysuqu --no-cache-dir pysuqu
```

For an editable checkout, set the same environment variable and run
`python -m pip install -e .`. The source distribution includes `native/dynamics.cpp`
and an optional CMake build definition.

## Backend Selection

| Backend | Equation and execution |
| --- | --- |
| `qutip` | Reference callbacks with linear trace interpolation |
| `qutip_compiled` | Reused QuTiP coefficients; interpolation order 0 through 3 |
| `cpp` | Native propagation of the full specified equation |
| `auto` | Native when supported, otherwise compiled QuTiP |
| `cpp_fast` | Explicit rotating-wave approximation for supported IQ models |

`cpp_fast` changes the equation. Its result statistics identify the approximation
and active levels; compare its output against `cpp` before using it. `auto` never
selects this approximation.

Native propagation supports dense, CSR, banded, and fused sparse operators,
nonuniform trace and output grids, exact interaction frames, independent state
batches, and static collapse operators. Constant and piecewise-constant problems
can use checked Krylov exponential actions. Open-system evolution applies the
Lindblad equation directly to density matrices without constructing a Kronecker
Liouvillian.

An explicit native request raises `BackendUnavailable` when the extension is
absent, or `UnsupportedBackendError` for unsupported inputs. `auto` can fall back
for time-dependent collapse operators and expectation operators (`e_ops`), recording
`backend_fallback` and `backend_fallback_reason` in result statistics. When no
extension is installed, `auto` selects `qutip_compiled` at construction.

## Repeated State Propagation

```python
import numpy as np
import qutip as qt
from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit import DriveTerm, PreparedPropagation, PropagationOptions

rng = np.random.default_rng(863591)
times = np.linspace(0.0, 3.0, 121)
amplitude = rng.uniform(0.12, 0.22)
envelope = amplitude * np.sin(np.pi * times / times[-1]) ** 2
trace = SignalTrace(times, envelope.astype(complex), 40.0,
                    'iq_complex', 'qubit_iq', 0.9)
options = PropagationOptions(atol=1e-10, rtol=1e-8, store_states=False)
prepared = PreparedPropagation(
    0.23 * qt.sigmaz(), [DriveTerm(qt.sigmax(), trace)], times,
    backend='auto', options=options,
)
batch = prepared.propagate_batch([qt.basis(2, 0), qt.basis(2, 1)])
unitary = np.column_stack([state.full()[:, 0] for state in batch.final_states])
assert np.linalg.norm(unitary.conj().T @ unitary - np.eye(2)) < 1e-6
```

Hamiltonians and drive amplitudes use angular-frequency units consistent with
the time axis; IQ `lo_freq` uses cycles per time unit. `DriveTerm(mode='rf')`
reconstructs the real RF signal from I/Q and its carrier. Use
`mode='complex_envelope'` only when the Hamiltonian already represents that envelope.
Ensure that complex coefficients and their operators form the intended Hermitian
Hamiltonian. RF reconstruction keeps the analytic carrier for native propagation
and the default linear compiled-IQ path.

The trace grid specifies the sampled control; `tlist` specifies returned times.
Interpolation order and any RF oversampling must agree when comparing backends.
The reference callback path always uses linear interpolation. Avoid mutating
operators, traces, or options after preparing a context; construct a new context
when the model changes.

With `store_states=False`, use `final_state` (or batch `final_states`) to avoid
storing every output state. A batch shares preparation, but still evolves each
initial state. `parallel='on'` allows native worker scheduling; `off` forces serial
execution, and `auto` chooses according to workload size.

## Numerical Controls

`PropagationOptions` accepts `atol`, `rtol`, `coefficient_order`, `matrix_format`,
`frame`, `sparse_kernel`, `block_decompose`, `sparse_expm`, and `parallel`.
`frame='interaction_exact'` is an exact change of representation, with output
transformed back to the original basis. It is separate from `cpp_fast`.

`clear_native_plan_cache()` and `native_plan_cache_info()` manage the process-local
bounded preparation cache. `plan_cache_size=0` disables it. Initial states are
never cached. QuTiP-specific options can be supplied through `extra`.

## Benchmarks And Tutorial

Run from a checkout with real QuTiP installed:

```bash
python -m benchmarks.propagation_workflow --case gate --dimension 8
python -m benchmarks.propagation_workflow --case sparse --dimension 320
python -m benchmarks.propagation_workflow --case lindblad --dimension 8
```

Each run uses a reproducible synthetic model and the same dimension, grid,
tolerances, output policy, and initial states for every backend. JSON output
separates cold preparation plus propagation, warm wall time, and native integration
time when available. Errors include state distance and norm or trace conservation;
the process fails if either exceeds `--max-error`. Speedups are workload-specific,
and timing includes Python result construction. No approximate backend is included.

The [native propagation tutorial](../../demo/demo_06_native_propagation.ipynb)
compares RF dynamics, repeated state batches, and dissipative evolution. It also
runs on the platform-independent wheel using QuTiP.
