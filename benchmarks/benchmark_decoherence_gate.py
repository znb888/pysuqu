"""Benchmark gate-level propagation with gate-owned decoherence parameters."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import statistics
import time

import numpy as np


def _build_gate(*, dynamic: bool):
    from pysuqu.funclib.awgenerator import EnvelopeParams, PulseEvent
    from pysuqu.qubit.gate import SingleQubitGate

    with redirect_stdout(io.StringIO()):
        gate = SingleQubitGate(
            total_time=64.0,
            sample_rate=2.0,
            energy_trunc_level=3,
        )
        pulse = PulseEvent(
            start_time=12.0,
            envelope=EnvelopeParams(
                name="x90",
                duration=20.0,
                peak_amp=0.2,
                shape_type="cosine",
            ),
            name="x90",
            if_freq=0.05,
        )
        channel = gate.load_pulse(pulse)
        gate.load_decoherence(
            T1=500.0,
            Tphi1=700.0,
            Tphi2=300.0 if dynamic else None,
        )
    return gate, channel


def _final_state(result):
    final = getattr(result, "final_state", None)
    if final is not None:
        return final
    states = getattr(result, "states", None) or []
    if not states:
        raise RuntimeError("propagation result does not contain a final state")
    return states[-1]


def _run_once(*, backend: str, dynamic: bool, options: dict):
    gate, channel = _build_gate(dynamic=dynamic)
    with redirect_stdout(io.StringIO()):
        result = gate.run_simulation(
            channel=channel,
            initial_state_input=0,
            backend=backend,
            options=options,
        )
    return _final_state(result), dict(getattr(result, "stats", {}) or {})


def _run_warm_once(gate, channel, *, backend: str, options: dict):
    with redirect_stdout(io.StringIO()):
        result = gate.run_simulation(
            channel=channel,
            initial_state_input=0,
            backend=backend,
            options=options,
        )
    return _final_state(result), dict(getattr(result, "stats", {}) or {})


def benchmark_decoherence_gate(
    *,
    samples: int = 5,
    warmups: int = 2,
    dynamic: bool = False,
    include_fresh: bool = True,
) -> dict[str, object]:
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be >= 1 and warmups must be >= 0")

    options = {
        "atol": 1e-12,
        "rtol": 1e-10,
        "nsteps": 50000,
        "store_states": False,
        "store_final_state": True,
        "normalize_output": False,
        "matrix_format": "csr",
        "frame": "lab",
        "block_decompose": "off",
        "sparse_expm": "off",
    }
    warm_gate, warm_channel = _build_gate(dynamic=dynamic)
    for backend in ("qutip", "cpp"):
        for _ in range(warmups):
            _run_warm_once(warm_gate, warm_channel, backend=backend, options=options)

    timings: dict[str, list[float]] = {"qutip": [], "cpp": []}
    finals = {}
    stats = {}
    for backend in ("qutip", "cpp"):
        for _ in range(samples):
            started = time.perf_counter()
            final, observed = _run_warm_once(
                warm_gate,
                warm_channel,
                backend=backend,
                options=options,
            )
            timings[backend].append(time.perf_counter() - started)
            finals[backend] = final.full()
            stats[backend] = observed

    fresh_timings: dict[str, list[float]] = {"qutip": [], "cpp": []}
    if include_fresh:
        for backend in ("qutip", "cpp"):
            for _ in range(samples):
                started = time.perf_counter()
                final, _observed = _run_once(
                    backend=backend,
                    dynamic=dynamic,
                    options=options,
                )
                fresh_timings[backend].append(time.perf_counter() - started)
                finals.setdefault(f"fresh_{backend}", final.full())

    qutip_final = np.asarray(finals["qutip"])
    cpp_final = np.asarray(finals["cpp"])
    max_abs_difference = float(np.max(np.abs(cpp_final - qutip_final)))

    def summarize(values):
        return {
            "samples": values,
            "median_seconds": statistics.median(values),
            "mean_seconds": statistics.fmean(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }

    warm_summary = {name: summarize(values) for name, values in timings.items()}
    fresh_summary = {name: summarize(values) for name, values in fresh_timings.items() if values}
    return {
        "benchmark": "decoherence_gate",
        "case": "T1+Tphi1+Tphi2" if dynamic else "T1+Tphi1",
        "samples": samples,
        "warmups": warmups,
        "options": options,
        "warm": warm_summary,
        "fresh": fresh_summary,
        "warm_speedup_qutip_over_cpp": (
            warm_summary["qutip"]["median_seconds"]
            / warm_summary["cpp"]["median_seconds"]
        ),
        "fresh_speedup_qutip_over_cpp": (
            fresh_summary["qutip"]["median_seconds"]
            / fresh_summary["cpp"]["median_seconds"]
            if fresh_summary
            else None
        ),
        "max_abs_final_state_difference": max_abs_difference,
        "qutip_stats": stats["qutip"],
        "cpp_stats": stats["cpp"],
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--dynamic", action="store_true", help="Include gate-owned Tphi2.")
    parser.add_argument("--no-fresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    result = benchmark_decoherence_gate(
        samples=args.samples,
        warmups=args.warmups,
        dynamic=args.dynamic,
        include_fresh=not args.no_fresh,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"case: {result['case']}")
        print(f"warm speedup: {result['warm_speedup_qutip_over_cpp']:.3f}x")
        if result["fresh_speedup_qutip_over_cpp"] is not None:
            print(f"fresh speedup: {result['fresh_speedup_qutip_over_cpp']:.3f}x")
        print(f"max abs final-state difference: {result['max_abs_final_state_difference']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
