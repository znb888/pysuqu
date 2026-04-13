# Refactor Status

This public repository exposes the stabilized `pysuqu 2.0.1` package layout.

## Current State

- The `qubit` package is published through a stable package-level export surface.
- The `decoherence` package is published through a stable package-level export
  surface.
- The public repository keeps the package, public tests, and public
  documentation only.

## Stable Public Direction

- Prefer `pysuqu.qubit` for user-facing qubit imports.
- Prefer `pysuqu.decoherence` for user-facing decoherence imports.
- Keep typed result objects as the preferred public return contracts.

## Repository Boundary

- Private notebooks are excluded.
- The old private notebook tree is excluded.
- Internal migration logs and archive reports are excluded from this public
  repository snapshot.

