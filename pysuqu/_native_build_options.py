"""Opt-in compiler tuning controls for the optional native extension.

The default build stays portable and deterministic.  CPU-specific and
profile-guided options must be requested explicitly through environment
variables so a wheel never silently targets the build host.
"""

from __future__ import annotations

import os
from pathlib import Path


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def native_build_options() -> dict[str, object]:
    """Return validated native build options derived from the environment."""
    compiler = os.environ.get("PYSUQU_NATIVE_COMPILER", "").strip().lower()
    if compiler not in {"", "msvc", "gnu", "clang"}:
        raise ValueError("PYSUQU_NATIVE_COMPILER must be msvc, gnu, clang, or empty")
    pgo = os.environ.get("PYSUQU_NATIVE_PGO", "off").strip().lower()
    if pgo not in {"off", "generate", "use"}:
        raise ValueError("PYSUQU_NATIVE_PGO must be off, generate, or use")
    profile_dir = os.environ.get("PYSUQU_NATIVE_PGO_DIR", "").strip()
    return {
        "compiler_family": compiler,
        "march_native": _truthy("PYSUQU_NATIVE_MARCH_NATIVE"),
        "pgo": pgo,
        "profile_dir": str(Path(profile_dir).resolve()) if profile_dir else "",
    }


__all__ = ["native_build_options"]
