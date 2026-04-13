# Code Style

This guide keeps only the conventions that remain active for the public
`pysuqu` repository.

## Principles

- Document why and where the boundary is, not the obvious mechanics.
- New public functions should document input units, output units, and return
  types.
- Prefer small focused helpers over large functions with heavy inline comments.
- Mark compatibility-only entry points explicitly as legacy behavior.

## Docstrings

Public functions and classes should at least describe:

- purpose
- important parameters
- return value
- units or shape contracts
- important exceptions when needed

## Naming

- Keep public API names stable and descriptive.
- Avoid reintroducing old module-shell names as new primary entry points.
- Prefer typed result names such as `TphiResult`, `T1Result`, and
  `NoiseFitResult`.

## pysuqu-Specific Expectations

- Physical quantities should always state their units.
- Compatibility helpers should stay clearly separated from the preferred public
  path.
- Package-level imports should remain the recommended public surface when
  practical.

