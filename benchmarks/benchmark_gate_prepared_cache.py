"""Measure exact ``run_simulation(reuse_prepared=True)`` reuse."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from benchmark_gate_propagation import _build_gate


def _measure(gate, schedule, options, *, reuse_prepared, repeats):
    samples = []
    final = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = gate.run_simulation(
            channel=schedule,
            backend="cpp",
            options=options,
            reuse_prepared=reuse_prepared,
        )
        samples.append(time.perf_counter() - started)
        final = result.final_state
    return statistics.median(samples), final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--total-time", type=float, default=32.0)
    parser.add_argument("--sample-rate", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    gate, schedule = _build_gate(args.levels, args.total_time, args.sample_rate)
    options = {
        "profile": "reference",
        "matrix_format": "auto",
        "frame": "lab",
        "store_states": False,
        "store_final_state": True,
    }
    uncached, reference = _measure(
        gate, schedule, options, reuse_prepared=False, repeats=args.repeats
    )
    gate.clear_prepared_cache()
    cached, actual = _measure(
        gate, schedule, options, reuse_prepared=True, repeats=args.repeats
    )
    print({
        "uncached_median_seconds": uncached,
        "cached_median_seconds": cached,
        "speedup": uncached / cached,
        "max_abs_difference": float(np.max(np.abs(reference.full() - actual.full()))),
        "cache": gate.prepared_cache_info(),
    })


if __name__ == "__main__":
    main()
