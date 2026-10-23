"""Benchmark scalar adaptive PSD integration against the strict batch path.

The batch path uses SciPy ``quad_vec`` with the same log-log spline,
breakpoints and tolerances as the scalar ``quad`` implementation.  It is only
intended for vectorizable built-in filter functions; this benchmark reports
the raw integral agreement so speedups cannot hide a semantic change.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from pysuqu.funclib.mathlib import (
    PreparedFilteredPSD,
    cpmg_transfunc,
    echo_transfunc,
    ramsey_transfunc,
)


def _measure(fn, repeats: int) -> tuple[float, list[float]]:
    samples = []
    value = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - started)
    return value, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfreq", type=int, default=2000)
    parser.add_argument("--ndelay", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--experiment", choices=("Ramsey", "SpinEcho", "CPMG"), default="Ramsey")
    parser.add_argument("--N", type=int, default=100, help="CPMG pulse count")
    parser.add_argument("--epsrel", type=float, default=1e-5)
    args = parser.parse_args()

    frequency = np.geomspace(1e1, 1e9, args.nfreq)
    psd = 1e-12 / frequency + 4e-23
    delays = np.linspace(10e-9, 10e-6, args.ndelay)
    prepared = PreparedFilteredPSD.for_continuous(frequency, psd)
    if args.experiment == "Ramsey":
        vector_filter = lambda x: ramsey_transfunc(x, delays)
        scalar_filter = lambda x, tau: ramsey_transfunc(x, tau)
    elif args.experiment == "SpinEcho":
        vector_filter = lambda x: echo_transfunc(x, delays)
        scalar_filter = lambda x, tau: echo_transfunc(x, tau)
    else:
        vector_filter = lambda x: cpmg_transfunc(x, delays, args.N, 100e-9)
        scalar_filter = lambda x, tau: cpmg_transfunc(x, tau, args.N, 100e-9)

    def scalar_run():
        return np.asarray([
            prepared.integrate_continuous(
                lambda x, tau=tau: scalar_filter(x, tau), epsrel=args.epsrel
            )
            for tau in delays
        ])

    def vector_run():
        return prepared.integrate_continuous_many(
            vector_filter, delays.size, epsrel=args.epsrel
        )

    scalar, scalar_times = _measure(scalar_run, args.repeats)
    vector, vector_times = _measure(vector_run, args.repeats)
    max_abs = float(np.max(np.abs(scalar - vector)))
    max_rel = float(
        np.max(np.abs(scalar - vector) / np.maximum(np.abs(scalar), 1e-300))
    )
    payload = {
        "experiment": args.experiment,
        "nfreq": args.nfreq,
        "ndelay": args.ndelay,
        "epsrel": args.epsrel,
        "N": args.N,
        "scalar_median_seconds": statistics.median(scalar_times),
        "batch_median_seconds": statistics.median(vector_times),
        "speedup": statistics.median(scalar_times) / statistics.median(vector_times),
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "scalar_samples": scalar_times,
        "batch_samples": vector_times,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
