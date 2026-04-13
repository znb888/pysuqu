"""Timing harness for the public XYNoiseDecoherence T1 workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import time

import numpy as np


_PSD_FREQ = np.logspace(0, 6, 2048)
_PSD_S = 1e-18 / np.maximum(_PSD_FREQ, 1.0) + 1e-20


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


def _run_xy_noise_t1_workload(xy_noise_cls) -> dict[str, float]:
    model = xy_noise_cls(
        psd_freq=_PSD_FREQ,
        psd_S=_PSD_S,
        noise_prop='single',
        is_spectral=True,
        is_print=False,
    )
    result = model.cal_t1(is_print=False)
    gamma_up = float(result.fit_diagnostics['gamma_up'])
    gamma_down = float(result.fit_diagnostics['gamma_down'])
    t1_seconds = float(result.value)
    white_noise_temperature = float(model.noise.output_stage.white_noise_temperature)
    return {
        't1_seconds': t1_seconds,
        'gamma_up': gamma_up,
        'gamma_down': gamma_down,
        'white_noise_temperature_kelvin': white_noise_temperature,
        'checksum': t1_seconds + gamma_up + gamma_down + white_noise_temperature,
    }


def benchmark_xy_noise_t1_workflow(
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
    from pysuqu.decoherence import XYNoiseDecoherence

    for _ in range(warmups):
        for _ in range(iterations):
            _run_xy_noise_t1_workload(XYNoiseDecoherence)

    sample_seconds = []
    workload_signature = {}
    workload_checksum = 0.0
    for _ in range(samples):
        sample_signature = {}
        start = time.perf_counter()
        for _ in range(iterations):
            sample_signature = _run_xy_noise_t1_workload(XYNoiseDecoherence)
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_signature = sample_signature
        workload_checksum = sample_signature['checksum']

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0

    return {
        'benchmark': 'xy_noise_t1_workflow',
        'workflow': 'pysuqu.decoherence.XYNoiseDecoherence constructor + cal_t1()',
        'backend': backend,
        'samples': samples,
        'warmups': warmups,
        'iterations_per_sample': iterations,
        'noise_points': len(_PSD_FREQ),
        'sample_seconds': sample_seconds,
        'mean_seconds': mean_seconds,
        'median_seconds': median_seconds,
        'min_seconds': min(sample_seconds),
        'max_seconds': max(sample_seconds),
        'stdev_seconds': stdev_seconds,
        'workload_signature': workload_signature,
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
    workload_signature = result['workload_signature']
    return '\n'.join(
        [
            f"benchmark: {result['benchmark']}",
            f"workflow: {result['workflow']}",
            f"backend: {result['backend']}",
            (
                'config: '
                f"samples={result['samples']}, warmups={result['warmups']}, "
                f"iterations={result['iterations_per_sample']}, noise_points={result['noise_points']}"
            ),
            f"mean_seconds: {result['mean_seconds']:.6f}",
            f"median_seconds: {result['median_seconds']:.6f}",
            f"min_seconds: {result['min_seconds']:.6f}",
            f"max_seconds: {result['max_seconds']:.6f}",
            f"stdev_seconds: {result['stdev_seconds']:.6f}",
            f"t1_seconds: {workload_signature['t1_seconds']:.12f}",
            f"gamma_up: {workload_signature['gamma_up']:.12f}",
            f"gamma_down: {workload_signature['gamma_down']:.12f}",
            (
                'white_noise_temperature_kelvin: '
                f"{workload_signature['white_noise_temperature_kelvin']:.12f}"
            ),
            f"workload_checksum: {result['workload_checksum']:.12f}",
            f"sample_seconds: [{sample_seconds}]",
            f"python_version: {result['python_version']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_xy_noise_t1_workflow(
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
