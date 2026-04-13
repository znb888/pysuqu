"""Timing harness for the public ParameterizedQubit rebuild workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import time


_BASE_WORKLOAD = {
    'capacitances': [
        [1.8, -0.25, -0.1],
        [-0.25, 1.6, -0.15],
        [-0.1, -0.15, 1.4],
    ],
    'junctions_resistance': [
        [1e20, 11.0, 1e20],
        [11.0, 1e20, 12.0],
        [1e20, 12.0, 13.0],
    ],
    'inductances': [
        [40.0, 90.0, 100.0],
        [90.0, 45.0, 95.0],
        [100.0, 95.0, 50.0],
    ],
    'fluxes': [
        [0.0, 0.17, 0.0],
        [0.17, 0.0, 0.0],
        [0.0, 0.0, 0.09],
    ],
    'trunc_ener_level': [2, 2],
    'junc_ratio': [
        [1.0, 1.15, 0.0],
        [1.15, 1.0, 0.0],
        [0.0, 0.0, 1.05],
    ],
    'structure_index': [2, 1],
}

_UPDATED_ELEMENTS = {
    'capac': [
        [1.9, -0.28, -0.12],
        [-0.28, 1.68, -0.18],
        [-0.12, -0.18, 1.48],
    ],
    'resis': [
        [1e20, 10.5, 1e20],
        [10.5, 1e20, 11.5],
        [1e20, 11.5, 12.5],
    ],
    'induc': [
        [42.0, 92.0, 102.0],
        [92.0, 47.0, 97.0],
        [102.0, 97.0, 52.0],
    ],
}


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


def _run_parameterized_qubit_rebuild_workload(parameterized_qubit_cls) -> float:
    qubit = parameterized_qubit_cls(**_BASE_WORKLOAD)
    qubit.change_para(**_UPDATED_ELEMENTS)
    return float(qubit.get_energy_matrices('Ej')[0, 0])


def benchmark_parameterized_qubit_rebuild(
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
    from pysuqu.qubit import ParameterizedQubit

    for _ in range(warmups):
        for _ in range(iterations):
            _run_parameterized_qubit_rebuild_workload(ParameterizedQubit)

    sample_seconds = []
    workload_checksum = 0.0
    for _ in range(samples):
        sample_checksum = 0.0
        start = time.perf_counter()
        for _ in range(iterations):
            sample_checksum = _run_parameterized_qubit_rebuild_workload(ParameterizedQubit)
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_checksum = sample_checksum

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0

    return {
        'benchmark': 'parameterized_qubit_element_rebuild',
        'workflow': 'pysuqu.qubit.ParameterizedQubit constructor + element rebuild via change_para()',
        'backend': backend,
        'samples': samples,
        'warmups': warmups,
        'iterations_per_sample': iterations,
        'sample_seconds': sample_seconds,
        'mean_seconds': mean_seconds,
        'median_seconds': median_seconds,
        'min_seconds': min(sample_seconds),
        'max_seconds': max(sample_seconds),
        'stdev_seconds': stdev_seconds,
        'workload_checksum': workload_checksum,
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
    return '\n'.join(
        [
            f"benchmark: {result['benchmark']}",
            f"workflow: {result['workflow']}",
            f"backend: {result['backend']}",
            (
                'config: '
                f"samples={result['samples']}, warmups={result['warmups']}, "
                f"iterations={result['iterations_per_sample']}"
            ),
            f"mean_seconds: {result['mean_seconds']:.6f}",
            f"median_seconds: {result['median_seconds']:.6f}",
            f"min_seconds: {result['min_seconds']:.6f}",
            f"max_seconds: {result['max_seconds']:.6f}",
            f"stdev_seconds: {result['stdev_seconds']:.6f}",
            f"workload_checksum: {result['workload_checksum']:.12f}",
            f"sample_seconds: [{sample_seconds}]",
            f"python_version: {result['python_version']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_parameterized_qubit_rebuild(
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
