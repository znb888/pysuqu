# Public Demo Notebooks

This directory contains public tutorial notebooks for `pysuqu`.

## Contents

- `demo_01_single_qubit_basics.ipynb`
  - single-qubit construction, public spectrum helpers, and flux sweeps
- `demo_02_decoherence_with_synthetic_noise.ipynb`
  - electronic-noise pipeline and dephasing analysis using synthetic PSD data
- `demo_03_waveform_and_gate_basics.ipynb`
  - waveform generation, schedule import, and a minimal gate simulation path
- `demo_04_multiqubit_coupler_workflow.ipynb`
  - coupler-bias probing and multi-qubit sensitivity analysis

## Data Policy

The `data/` files in this directory are fully synthetic and contain no private
measurements, notebook outputs, or old demo assets.

## Running

From the repository root:

```bash
jupyter lab
```

Then open the notebooks under `demo/`.

