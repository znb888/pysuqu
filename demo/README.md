# Public Demo Notebooks

This directory contains public tutorial notebooks for `pysuqu`.

## Contents

- `demo_01_single_qubit_basics.ipynb`
  - single-qubit construction, public spectrum helpers, and flux sweeps
- `demo_02_decoherence.ipynb`
  - control-line T1 loss, electronic-noise pipeline, and dephasing analysis
- `demo_03_waveform_and_gate_basics.ipynb`
  - waveform generation, schedule import, and a minimal gate simulation path
- `demo_04_dynamic_simulation.ipynb`
  - single-qubit gate dynamics, fidelity diagnostics, and local calibration checks
- `demo_05_transmission_chain_touchstone.ipynb`
  - single-line and MIMO transmission chains, synthetic Touchstone models,
    qubit-plane traces, and multi-drive gate simulation

## Data Policy

The `data/` files in this directory are fully synthetic and contain no private
measurements, notebook outputs, or old demo assets.

Demo 05 creates temporary synthetic Touchstone files under
`tmp/demo_05_touchstone/` when it runs.

## Running

From the repository root:

```bash
jupyter lab
```

Then open the notebooks under `demo/`.

