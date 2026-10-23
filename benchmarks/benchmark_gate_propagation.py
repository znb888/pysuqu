"""Measure single-gate propagation backends.

Examples
--------
    python benchmarks/benchmark_gate_propagation.py --levels 10 --total-time 100
    PYSUQU_BUILD_NATIVE=1 python benchmarks/benchmark_gate_propagation.py --native

The benchmark reports medians and state overlaps.  The exact native IQ/RF path
is selected with ``frame="interaction_exact"`` (or ``frame="auto"``); the
separate ``cpp_fast`` backend remains an explicitly approximate RWA comparison.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysuqu.funclib.awgenerator import (
    ChannelSchedule,
    EnvelopeParams,
    MixerParams,
    PulseEvent,
)
from pysuqu.qubit.gate import SingleQubitGate
from pysuqu.qubit.propagation import native_backend_available


def _build_gate(levels: int, total_time: float, sample_rate: float):
    with redirect_stdout(StringIO()):
        gate = SingleQubitGate(
            total_time=total_time,
            sample_rate=sample_rate,
            qubit_frequency=5.0,
            energy_trunc_level=levels,
        )
    schedule = ChannelSchedule(
        sampling_rate=sample_rate,
        mixer_config=MixerParams(lo_freq=5.0),
        events=[
            PulseEvent(
                start_time=2.0,
                envelope=EnvelopeParams(
                    duration=min(20.0, total_time),
                    peak_amp=1e-3,
                    shape_type="gaussian",
                    sigma=5.0,
                ),
                if_freq=0.1,
            )
        ],
    )
    return gate, schedule


def _run(
    gate,
    schedule,
    backend: str,
    repetitions: int,
    method: Optional[str],
    fast_active_levels: Optional[int] = None,
    matrix_format: str = "auto",
    frame: str = "lab",
    profile: str = "reference",
):
    options = {
        "profile": profile,
        "matrix_format": matrix_format,
        "frame": frame,
        "atol": 1e-8,
        "rtol": 1e-6,
        "store_states": False,
        "store_final_state": True,
    }
    if method is not None:
        options["method"] = method
    if backend == "cpp_fast":
        # cpp_fast owns its rotating frame and cannot be combined with the
        # exact interaction representation option.
        options["frame"] = "lab"
        options.update(
            {
                "atol": 1e-6,
                "rtol": 1e-4,
                "rwa_max_discarded_ratio": 0.5,
            }
        )
        if fast_active_levels is not None:
            options["active_levels"] = max(
                2,
                min(int(fast_active_levels), gate.qubit._Nlevel[0]),
            )
    elapsed = []
    final = None
    for _ in range(repetitions):
        started = time.perf_counter()
        result = gate.run_simulation(
            channel=schedule,
            backend=backend,
            options=options,
        )
        elapsed.append(time.perf_counter() - started)
        final = getattr(result, "final_state", None)
        if final is None:
            final = result.states[-1]
    return {
        "backend": backend,
        "median_seconds": statistics.median(elapsed),
        "min_seconds": min(elapsed),
        "samples": len(gate.awg.t_axis),
        "norm": float(final.norm()),
        "stats": dict(getattr(result, "stats", {}) or {}),
    }, final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--total-time", type=float, default=100.0)
    parser.add_argument("--sample-rate", type=float, default=2.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--method",
        default=None,
        help="explicit QuTiP integrator method (default: profile/reference choice)",
    )
    parser.add_argument(
        "--profile",
        choices=("reference", "fast", "fast_exact", "fallback", "stiff"),
        default="reference",
        help="prepared QuTiP solver profile",
    )
    parser.add_argument(
        "--matrix-format",
        choices=("auto", "dense", "csr", "banded", "fused_csr"),
        default="auto",
        help="native operator layout (default: auto)",
    )
    parser.add_argument(
        "--frame",
        choices=("lab", "interaction_exact", "auto"),
        default="lab",
        help="representation frame for the native backend (default: lab)",
    )
    parser.add_argument(
        "--fast-active-levels",
        type=int,
        default=None,
        help="optional active-level reduction for cpp_fast (default: full Hilbert space)",
    )
    parser.add_argument("--native", action="store_true", help="include the optional C++ backend")
    args = parser.parse_args()

    gate, schedule = _build_gate(args.levels, args.total_time, args.sample_rate)
    backends = ["qutip", "qutip_compiled"]
    if args.native and native_backend_available():
        backends.extend(["cpp", "cpp_fast"])

    reports = []
    finals = {}
    for backend in backends:
        report, final = _run(
            gate,
            schedule,
            backend,
            args.repetitions,
            args.method,
            args.fast_active_levels,
            args.matrix_format,
            args.frame,
            args.profile,
        )
        reports.append(report)
        finals[backend] = np.asarray(final.full(), dtype=np.complex128)

    reference = finals["qutip"]
    for report in reports:
        report["speedup_vs_qutip"] = reports[0]["median_seconds"] / report["median_seconds"]
        report["overlap_vs_qutip"] = float(
            abs(np.vdot(reference[:, 0], finals[report["backend"]][:, 0]))
        )
    print(json.dumps({"configuration": vars(args), "results": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
