"""Timing harness for the public GroundedTransmon flux-sweep workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import time
from contextlib import redirect_stdout
from io import StringIO

import numpy as np


_BASE_QUBIT_CONFIG = {
    'capacitance': 80e-15,
    'junction_resistance': 10_000,
    'inductance': 1e20,
    'flux': 0.125,
    'trunc_ener_level': 4,
    'junc_ratio': 1.2,
    'qr_couple': [3e-15],
}

_FLUX_OFFSETS = [np.array([[value]], dtype=float) for value in np.linspace(-0.06, 0.06, 81)]
_UPPER_LEVEL = 3


def _ensure_runtime_backends(use_test_stubs: bool) -> str:
    if use_test_stubs:
        from tests.support import install_test_stubs

        install_test_stubs()
    else:
        try:
            importlib.import_module('qutip')
        except ModuleNotFoundError:
            from tests.support import install_test_stubs

            install_test_stubs()

        try:
            importlib.import_module('plotly.graph_objects')
        except ModuleNotFoundError:
            from tests.support import install_plotly_stub

            install_plotly_stub()

    qutip = importlib.import_module('qutip')
    return 'test_stub' if hasattr(qutip, '_last_mesolve_call') else 'real_qutip'


def _construct_grounded_transmon(grounded_transmon_cls):
    with redirect_stdout(StringIO()):
        return grounded_transmon_cls(**_BASE_QUBIT_CONFIG)


def _run_grounded_flux_sweep_workload(qubit, sweep_helper) -> dict[str, float]:
    result = sweep_helper(qubit, _FLUX_OFFSETS, upper_level=_UPPER_LEVEL)
    level_1 = np.asarray(result.series['level_1'], dtype=float)
    level_2 = np.asarray(result.series['level_2'], dtype=float)
    level_3 = np.asarray(result.series['level_3'], dtype=float)
    restored_flux = float(np.asarray(qubit.get_element_matrices('flux'), dtype=float).reshape(-1)[0])

    return {
        'checksum': float(level_1.sum() + level_2.sum() + level_3.sum()),
        'baseline_f01_ghz': float(level_1[len(level_1) // 2]),
        'min_level_1_ghz': float(level_1.min()),
        'max_level_1_ghz': float(level_1.max()),
        'max_level_3_ghz': float(level_3.max()),
        'restored_flux': restored_flux,
    }


def benchmark_grounded_transmon_flux_sweep(
    *,
    samples: int = 7,
    warmups: int = 2,
    iterations: int = 3,
    use_test_stubs: bool = False,
) -> dict[str, object]:
    if samples < 1:
        raise ValueError('samples must be >= 1')
    if warmups < 0:
        raise ValueError('warmups must be >= 0')
    if iterations < 1:
        raise ValueError('iterations must be >= 1')

    backend = _ensure_runtime_backends(use_test_stubs=use_test_stubs)
    from pysuqu.qubit.single import GroundedTransmon
    from pysuqu.qubit.sweeps import sweep_single_qubit_energy_vs_flux_base_result

    qubit = _construct_grounded_transmon(GroundedTransmon)

    for _ in range(warmups):
        for _ in range(iterations):
            _run_grounded_flux_sweep_workload(qubit, sweep_single_qubit_energy_vs_flux_base_result)

    sample_seconds = []
    workload_signature = None
    for _ in range(samples):
        sample_signature = None
        start = time.perf_counter()
        for _ in range(iterations):
            sample_signature = _run_grounded_flux_sweep_workload(
                qubit,
                sweep_single_qubit_energy_vs_flux_base_result,
            )
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_signature = sample_signature

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0

    return {
        'benchmark': 'grounded_transmon_flux_sweep',
        'workflow': (
            'pysuqu.qubit.sweeps.sweep_single_qubit_energy_vs_flux_base_result '
            'on a public GroundedTransmon'
        ),
        'backend': backend,
        'samples': samples,
        'warmups': warmups,
        'iterations_per_sample': iterations,
        'flux_point_count': len(_FLUX_OFFSETS),
        'upper_level': _UPPER_LEVEL,
        'sample_seconds': sample_seconds,
        'mean_seconds': mean_seconds,
        'median_seconds': median_seconds,
        'min_seconds': min(sample_seconds),
        'max_seconds': max(sample_seconds),
        'stdev_seconds': stdev_seconds,
        'workload_checksum': workload_signature['checksum'],
        'workload_signature': workload_signature,
        'python_version': platform.python_version(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=7, help='Number of measured samples to collect.')
    parser.add_argument('--warmups', type=int, default=2, help='Number of warmup samples to discard.')
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of workload executions to average within each sample.',
    )
    parser.add_argument(
        '--use-test-stubs',
        action='store_true',
        help='Force the qutip test stub backend instead of a real qutip installation.',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit machine-readable JSON instead of a human-readable summary.',
    )
    return parser


def _format_summary(result: dict[str, object]) -> str:
    sample_seconds = ', '.join(f'{sample:.6f}' for sample in result['sample_seconds'])
    signature = result['workload_signature']
    return '\n'.join(
        [
            f"benchmark: {result['benchmark']}",
            f"workflow: {result['workflow']}",
            f"backend: {result['backend']}",
            (
                'config: '
                f"samples={result['samples']}, warmups={result['warmups']}, "
                f"iterations={result['iterations_per_sample']}, "
                f"flux_point_count={result['flux_point_count']}, upper_level={result['upper_level']}"
            ),
            f"mean_seconds: {result['mean_seconds']:.6f}",
            f"median_seconds: {result['median_seconds']:.6f}",
            f"min_seconds: {result['min_seconds']:.6f}",
            f"max_seconds: {result['max_seconds']:.6f}",
            f"stdev_seconds: {result['stdev_seconds']:.6f}",
            f"workload_checksum: {result['workload_checksum']:.12f}",
            (
                'workload_signature: '
                f"baseline_f01_ghz={signature['baseline_f01_ghz']:.12f}, "
                f"min_level_1_ghz={signature['min_level_1_ghz']:.12f}, "
                f"max_level_1_ghz={signature['max_level_1_ghz']:.12f}, "
                f"max_level_3_ghz={signature['max_level_3_ghz']:.12f}, "
                f"restored_flux={signature['restored_flux']:.12f}"
            ),
            f"sample_seconds: [{sample_seconds}]",
            f"python_version: {result['python_version']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_grounded_transmon_flux_sweep(
        samples=args.samples,
        warmups=args.warmups,
        iterations=args.iterations,
        use_test_stubs=args.use_test_stubs,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_summary(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
