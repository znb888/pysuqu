# Code Style

This guide keeps only the conventions that still apply to the public
`pysuqu` repository.

## Principles

- Document why and where the boundary is, not the obvious mechanics.
- New public functions should document input units, output units, and return
  types.
- Prefer small focused helpers over large functions with heavy inline comments.
- Mark compatibility-only entry points explicitly as legacy behavior.

## Minimum Docstring Expectations

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
- Prefer names that expose intent directly, for example
  `get_energy_matrices()`.

## pysuqu-Specific Expectations

- Physical quantities should always state their units, especially `Hz`, `GHz`,
  `2π·GHz`, `s`, and `us`.
- Compatibility helpers should stay clearly separated from the preferred public
  path.
- Compatibility return shapes such as `_legacy` flows should not be mixed into
  primary API documentation.
- Package-level imports should remain the recommended public surface when
  practical.

## Recommended Docstring Template

```python
def function_name(param: float) -> float:
    """Short summary.

    Args:
        param: Input value in SI units.

    Returns:
        Output value in GHz.
    """
```

## Avoid

- Using `print()` as the main mechanism for status refresh or API signaling.
- Mixing calculation, formatting, and plotting inside one long method.
- Continuing to promote old module-shell entry points as the main public path.
- Leaving unit conventions only in notebooks instead of the API surface.
