# Transmission Chains

`pysuqu` can propagate generated control waveforms from the AWG reference
plane to the qubit reference plane. The same APIs support scalar loss and
delay, digital filters, arbitrary transfer functions, Touchstone S-parameters,
derivative precorrection, and multi-line MIMO models.

For a complete runnable tutorial, see
[`demo_05_transmission_chain_touchstone.ipynb`](../../demo/demo_05_transmission_chain_touchstone.ipynb).

## Units And Domains

- Time values use `ns`.
- Sample rates use samples/ns, numerically equivalent to `GSa/s`.
- Frequencies use `GHz` in waveform and transmission APIs.
- `iq_complex` traces contain complex envelopes and carry an `lo_freq`.
- `rf_real` traces contain sampled real-valued RF waveforms.
- Reference planes are `baseband`, `awg_iq`, `awg_rf`, `qubit_iq`, and
  `qubit_rf`.

## Build A Single-Line Chain

```python
from pysuqu.qubit import (
    AttenuatorStage,
    ChannelSchedule,
    DelayStage,
    EnvelopeParams,
    MixerParams,
    PulseEvent,
    TransmissionChain,
    WaveformGenerator,
)

generator = WaveformGenerator(total_time=24.0, sample_rate=2.0)
channel = ChannelSchedule(
    name="XY_Q1",
    sampling_rate=2.0,
    mixer_config=MixerParams(lo_freq=5.0),
    events=[
        PulseEvent(
            start_time=4.0,
            if_freq=0.05,
            envelope=EnvelopeParams(
                duration=8.0,
                peak_amp=0.03,
                shape_type="cosine",
            ),
        )
    ],
    transmission_chain=TransmissionChain(
        name="xy_line",
        stages=[
            AttenuatorStage(loss_db=3.0),
            DelayStage(delay_ns=1.5),
        ],
    ),
)

result = generator.generate_qubit_output(
    channel,
    mode="iq",
    capture_history=True,
)

print(result.input_trace.plane)
print(result.output_trace.plane)
print([trace.metadata["last_stage"] for trace in result.stage_outputs])
```

`generate_awg_output()` returns the unpropagated AWG-side trace.
`generate_qubit_output()` applies the configured chain and finalizes the trace
at `qubit_iq` or `qubit_rf`.

## Touchstone Models

Use `TouchstoneStage` for one selected `Sij` path:

```python
from pysuqu.qubit import TouchstoneStage

package_path = TouchstoneStage(
    file_path="package.s2p",
    input_port=1,
    output_port=2,
    interpolation="polar",
    out_of_band="edge",
)
```

Ports are one-based. The example selects `S21`. Use `MIMOTouchstoneStage` and
`BundleTransmissionChain` when several aligned input lines must be propagated
through one S-parameter submatrix.

The public parser accepts full-matrix `.sNp` data. Choose an explicit
`out_of_band` policy when the simulation spectrum extends beyond the measured
frequency span.

## Derivative Precorrection

`DerivativePrecorrectionStage` adds weighted time derivatives before the
physical line model. Coefficients can be supplied directly or fitted against a
frequency response:

```python
import numpy as np

from pysuqu.qubit import (
    DerivativePrecorrectionStage,
    design_derivative_precorrection,
)

awg_trace = generator.generate_awg_output(channel, mode="iq")
response_on_fft_grid = np.ones(len(awg_trace.values), dtype=complex)

design = design_derivative_precorrection(
    awg_trace,
    response_on_fft_grid,
    derivative_orders=(1, 2),
)
stage = DerivativePrecorrectionStage.from_design(design)
```

The response array must already be evaluated on the FFT grid used for the
template trace. For an IQ trace, physical-frequency evaluation normally uses
`trace.lo_freq + numpy.fft.fftfreq(...)`.

## Gate And Multi-Drive Integration

`WaveformGenerator.get_solver_trace()` returns a sampled `SignalTrace` using
the same waveform and transmission rules as `get_qutip_func()`. For example,
using the generator and channel above:

```python
trace = generator.get_solver_trace(channel, mode="rf")
traces = generator.get_solver_trace_bundle({"drive": channel}, mode="rf")
```

The default reference plane is `qubit`; select `plane="awg"` to obtain the
trace before its transmission chain. In `rf` mode, IQ-compatible chains retain
the complex envelope and `lo_freq` so a solver can mix the carrier continuously.
RF-only chains return sampled `rf_real` traces. `mode="complex_envelope"`
requires an IQ-compatible chain and leaves the LO unmixed.

The bundle method applies transmission to the complete collection, including
MIMO mixing. Dictionary inputs return traces keyed by physical output channel;
list and tuple inputs return a tuple in the propagated output order. Existing
`get_qutip_bundle_funcs()` calls continue to return callbacks keyed by output
channel for all three input forms.

`SingleQubitGate` resolves transmission chains in this order:

1. Explicit `transmission_chain=` argument.
2. `ChannelSchedule.transmission_chain`.
3. The gate-level chain supplied to `SingleQubitGate(...)`.

`GateBase.build_multidrive_hamiltonian()` and
`GateBase.run_multidrive_simulation()` accept schedule collections, named drive
operators, and an optional `BundleTransmissionChain`. MIMO stages may rename
the output channels; drive-operator keys must match those propagated output
names.

## Diagnostics

- `TransmissionChain.describe()` reports the ordered stages.
- `capture_history=True` returns every intermediate trace or bundle.
- `WaveformGenerator.plot_trace()` plots one sampled trace.
- `WaveformGenerator.plot_transmission_result()` compares the full history.
- `SingleQubitGate.visualize_signal()` selects the AWG or qubit plane.
