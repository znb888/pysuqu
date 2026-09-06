# Documentation

This directory is the public documentation index for `pysuqu`.

The repository root `README.md` is the package homepage. This `docs/` index is
the place to decide which public guide or reference to open next.

## Start Here

- [guides/getting-started.md](guides/getting-started.md) for installation,
  imports, and a minimal usage example
- [guides/transmission-chains.md](guides/transmission-chains.md) for AWG-to-qubit
  propagation, Touchstone models, MIMO chains, and derivative precorrection
- [guides/propagation-backends.md](guides/propagation-backends.md) for prepared
  propagation, native acceleration, backend selection, and reproducible benchmarks
- [architecture/module-map.md](architecture/module-map.md) for package layout
  and recommended import surfaces
- [architecture/refactor-status.md](architecture/refactor-status.md) for the
  current public module status
- [guides/code-style.md](guides/code-style.md) for development conventions
- [releases/2.1.0.md](releases/2.1.0.md) for prepared propagation and native acceleration
- [releases/2.0.4.md](releases/2.0.4.md) for the transmission-chain release
  summary and validation record
- [../demo/README.md](../demo/README.md) for the public tutorial notebooks

## Public Layout

```text
docs/
  README.md
  architecture/
  assets/
  guides/
  releases/
```

## Directory Roles

- `guides/` contains public setup notes, walkthroughs, and contributor-facing
  conventions.
- `architecture/` contains stable package maps and status notes for the public
  modules.
- `assets/` contains shared public visuals used by the documentation.
- `releases/` records concise public summaries for tagged releases.

## Public Scope

Examples and benchmarks use reproducible synthetic inputs. Numerical guides
state their model assumptions and backend limitations.
