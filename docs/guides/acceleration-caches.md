# Acceleration Caches

`pysuqu` keeps reusable numerical preparation separate from mutable user inputs.
The optional caches in 2.1.2 are bounded and content keyed, so changing a trace,
network, schedule, or model creates a fresh entry.

## Native Preparation

`PreparedPropagation` shares operator conversion, sparse layouts, interpolation
segments, and solver setup across `propagate` and `propagate_batch`. Use
`clear_native_plan_cache()` between independent workloads and inspect
`native_plan_cache_info()` when measuring memory. Initial states are never retained
by a preparation cache.

## Gate Runs

Gate preparation reuse is opt in:

```python
result = gate.run_simulation(channel=schedule, backend="cpp", reuse_prepared=True)
gate.clear_prepared_cache()
print(gate.prepared_cache_info())
```

The cache fingerprints operators, traces, options, arguments, and schedule content.
Opaque callbacks and mutable transmission-chain objects bypass it because their
future values cannot be proven from a stable content key.

## Waveform And Touchstone Caches

Pass `WaveformCache(max_entries=32)` to `WaveformGenerator(cache=...)` for
deterministic AWG generation. Returned arrays are copies, and callable envelopes
are bypassed. `TouchstoneStage` and `MIMOTouchstoneStage` accept
`response_cache_size`; zero disables caching. Cached response arrays are read-only,
and the key includes network data, frequencies, interpolation, and channel options.

## Build Tuning

Portable native builds remain the default. Explicitly opt into host-specific
options only for local measurements:

```powershell
$env:PYSUQU_BUILD_NATIVE = '1'
$env:PYSUQU_NATIVE_MARCH_NATIVE = '1'
$env:PYSUQU_NATIVE_PGO = 'generate'
$env:PYSUQU_NATIVE_PGO_DIR = 'build/pgo'
python -m pip install -e .
```

`PYSUQU_NATIVE_PGO` accepts `off`, `generate`, or `use`. `-march=native` is
supported by GCC/Clang; MSVC rejects it explicitly. PGO data belongs to the local
build machine and should not be packaged into a wheel. The deterministic portable
path remains available for every supported platform.

## Dephasing Batches

`PreparedFilteredPSD.for_continuous(...)` prepares a positive log-grid once.
`integrate_continuous_many` uses `quad_vec` when the installed SciPy provides it
and falls back to scalar integration otherwise. `Decoherence.cal_dephase` uses the
batch path for Ramsey, SpinEcho, and CPMG delay arrays while retaining the legacy
continuous and discrete contracts.

Use matching PSD arrays, delay grids, tolerances, and integration methods when
comparing results. Cache timing is workload-specific; validate state, trace, and
observable errors before claiming a speedup.
