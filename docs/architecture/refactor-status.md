# Refactor Status

This public repository exposes the current stabilized `pysuqu` package layout.

## Current State

- The `qubit` package is published through a stable package-level export surface.
- The `decoherence` package is published through a stable package-level export
  surface.
- Transmission-chain stages, waveform propagation, Touchstone models, and
  precorrection tools are available from the stable `pysuqu.qubit` surface.
- The public repository keeps the package, public tests, and public
  documentation, including curated demo notebooks.

## Stable Public Direction

- Prefer `pysuqu.qubit` for user-facing qubit imports.
- Prefer `pysuqu.decoherence` for user-facing decoherence imports.
- Prefer `pysuqu.qubit` for gate and transmission workflows; the same objects
  remain available from `pysuqu.funclib` for focused waveform code.
- Keep typed result objects as the preferred public return contracts.

## Repository Boundary

- Historical private notebooks are excluded.
- Curated notebooks under `demo/` use synthetic or public-safe inputs.
- Internal migration logs and archive reports are excluded from this public
  repository snapshot.

