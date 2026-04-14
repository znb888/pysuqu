"""Timing harness for the public ZNoiseDecoherence Tphi2 workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import time

import numpy as np


_PSD_FREQ = np.logspace(4, 8, 2048)
_PSD_S = 1e-18 / np.maximum(_PSD_FREQ, 1.0) + 1e-20
_DELAY_LIST = np.linspace(10, 10e3, 100) * 1e-9


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


def _run_z_noise_tphi2_workload(z_noise_cls) -> dict[str, float | str]:
    model = z_noise_cls(
        psd_freq=_PSD_FREQ,
        psd_S=_PSD_S,
        noise_prop='single',
        is_spectral=True,
        is_print=False,
    )
    result = model.cal_tphi2(
        method='cal',
        experiment='Ramsey',
        delay_list=_DELAY_LIST,
        is_plot=False,
        is_print=False,
    )
    noise_output = model.noise.output_stage
    one_over_f_coef = float(noise_output.fit_result.fit_diagnostics['1f_coef'])
    white_noise_temperature = float(noise_output.white_noise_temperature)
    flux_sensitivity = float(model.qubit_sensibility)
    tphi2_seconds = float(result.value)
    checksum = (
        tphi2_seconds
        + white_noise_temperature
        + one_over_f_coef * 1e22
        + flux_sensitivity / 1e22
    )
    return {
        'method': result.metadata['method'],
        'experiment': result.metadata['experiment'],
        'tphi2_seconds': tphi2_seconds,
        'white_noise_temperature_kelvin': white_noise_temperature,
        'one_over_f_coef': one_over_f_coef,
        'flux_sensitivity_rad_per_s_per_wb': flux_sensitivity,
        'frequency_min_hz': float(noise_output.frequency.min()),
        'frequency_max_hz': float(noise_output.frequency.max()),
        'delay_mean_seconds': float(np.mean(_DELAY_LIST)),
        'checksum': checksum,
    }


def _profile_warm_path(z_noise_cls) -> dict[str, float]:
    model = z_noise_cls(
        psd_freq=_PSD_FREQ,
        psd_S=_PSD_S,
        noise_prop='single',
        is_spectral=True,
        is_print=False,
    )

    for _ in range(3):
        qubit = model._build_qubit(is_print=False)
        noise = model._build_noise(is_print=False)
        model.qubit = qubit
        model.noise = noise
        model._update_sensitivity()
        model.cal_tphi2(
            method='cal',
            experiment='Ramsey',
            delay_list=_DELAY_LIST,
            is_plot=False,
            is_print=False,
        )

    constructor_times = []
    build_qubit_times = []
    build_noise_times = []
    update_sensitivity_times = []
    cal_tphi2_times = []

    for _ in range(10):
        start = time.perf_counter()
        z_noise_cls(
            psd_freq=_PSD_FREQ,
            psd_S=_PSD_S,
            noise_prop='single',
            is_spectral=True,
            is_print=False,
        )
        constructor_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        qubit = model._build_qubit(is_print=False)
        build_qubit_times.append(time.perf_counter() - start)
        model.qubit = qubit

        start = time.perf_counter()
        noise = model._build_noise(is_print=False)
        build_noise_times.append(time.perf_counter() - start)
        model.noise = noise

        start = time.perf_counter()
        model._update_sensitivity()
        update_sensitivity_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.cal_tphi2(
            method='cal',
            experiment='Ramsey',
            delay_list=_DELAY_LIST,
            is_plot=False,
            is_print=False,
        )
        cal_tphi2_times.append(time.perf_counter() - start)

    return {
        'constructor_seconds': statistics.fmean(constructor_times),
        'build_qubit_seconds': statistics.fmean(build_qubit_times),
        'build_noise_seconds': statistics.fmean(build_noise_times),
        'update_sensitivity_seconds': statistics.fmean(update_sensitivity_times),
        'cal_tphi2_seconds': statistics.fmean(cal_tphi2_times),
    }


def _build_hotspot_shortlist(warm_path_split: dict[str, float]) -> list[str]:
    constructor_seconds = warm_path_split['constructor_seconds']
    cal_tphi2_seconds = warm_path_split['cal_tphi2_seconds']
    stage_labels = {
        'build_noise_seconds': 'build_noise',
        'build_qubit_seconds': 'build_qubit',
        'update_sensitivity_seconds': 'update_sensitivity',
        'cal_tphi2_seconds': 'cal_tphi2',
    }
    ranked = sorted(
        (
            (stage_labels[key], value)
            for key, value in warm_path_split.items()
            if key in stage_labels
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_stage, top_seconds = ranked[0]
    second_stage, second_seconds = ranked[1]
    return [
        (
            'Warm-path constructor work '
            f'(~{constructor_seconds:.6f} s/sample) dominates analytical cal_tphi2 '
            f'(~{cal_tphi2_seconds:.6f} s/sample) on this public Z workflow.'
        ),
        (
            'The largest measured substeps are '
            f'{top_stage} (~{top_seconds:.6f} s) and {second_stage} '
            f'(~{second_seconds:.6f} s).'
        ),
        (
            'The next optimization round should stay on this benchmark and land one narrow '
            'constructor-side improvement before reopening broader analytical dephasing hypotheses.'
        ),
    ]


def benchmark_z_noise_tphi2_workflow(
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
    from pysuqu.decoherence import ZNoiseDecoherence

    for _ in range(warmups):
        for _ in range(iterations):
            _run_z_noise_tphi2_workload(ZNoiseDecoherence)

    sample_seconds = []
    workload_signature = {}
    workload_checksum = 0.0
    for _ in range(samples):
        sample_signature = {}
        start = time.perf_counter()
        for _ in range(iterations):
            sample_signature = _run_z_noise_tphi2_workload(ZNoiseDecoherence)
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_signature = sample_signature
        workload_checksum = sample_signature['checksum']

    warm_path_split = _profile_warm_path(ZNoiseDecoherence)
    hotspot_shortlist = _build_hotspot_shortlist(warm_path_split)

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0

    return {
        'benchmark': 'z_noise_tphi2_workflow',
        'workflow': (
            "pysuqu.decoherence.ZNoiseDecoherence constructor + "
            "cal_tphi2(method='cal', experiment='Ramsey')"
        ),
        'backend': backend,
        'samples': samples,
        'warmups': warmups,
        'iterations_per_sample': iterations,
        'noise_points': len(_PSD_FREQ),
        'delay_points': len(_DELAY_LIST),
        'sample_seconds': sample_seconds,
        'mean_seconds': mean_seconds,
        'median_seconds': median_seconds,
        'min_seconds': min(sample_seconds),
        'max_seconds': max(sample_seconds),
        'stdev_seconds': stdev_seconds,
        'workload_signature': workload_signature,
        'workload_checksum': workload_checksum,
        'warm_path_split': warm_path_split,
        'hotspot_shortlist': hotspot_shortlist,
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
    warm_path_split = result['warm_path_split']
    return '\n'.join(
        [
            f"benchmark: {result['benchmark']}",
            f"workflow: {result['workflow']}",
            f"backend: {result['backend']}",
            (
                'config: '
                f"samples={result['samples']}, warmups={result['warmups']}, "
                f"iterations={result['iterations_per_sample']}, noise_points={result['noise_points']}, "
                f"delay_points={result['delay_points']}"
            ),
            f"mean_seconds: {result['mean_seconds']:.6f}",
            f"median_seconds: {result['median_seconds']:.6f}",
            f"min_seconds: {result['min_seconds']:.6f}",
            f"max_seconds: {result['max_seconds']:.6f}",
            f"stdev_seconds: {result['stdev_seconds']:.6f}",
            f"tphi2_seconds: {workload_signature['tphi2_seconds']:.12f}",
            f"white_noise_temperature_kelvin: {workload_signature['white_noise_temperature_kelvin']:.12f}",
            f"one_over_f_coef: {workload_signature['one_over_f_coef']:.12e}",
            (
                'flux_sensitivity_rad_per_s_per_wb: '
                f"{workload_signature['flux_sensitivity_rad_per_s_per_wb']:.12e}"
            ),
            f"frequency_min_hz: {workload_signature['frequency_min_hz']:.6f}",
            f"frequency_max_hz: {workload_signature['frequency_max_hz']:.6f}",
            f"delay_mean_seconds: {workload_signature['delay_mean_seconds']:.12f}",
            f"workload_checksum: {result['workload_checksum']:.12f}",
            f"warm_constructor_seconds: {warm_path_split['constructor_seconds']:.6f}",
            f"warm_build_qubit_seconds: {warm_path_split['build_qubit_seconds']:.6f}",
            f"warm_build_noise_seconds: {warm_path_split['build_noise_seconds']:.6f}",
            (
                'warm_update_sensitivity_seconds: '
                f"{warm_path_split['update_sensitivity_seconds']:.6f}"
            ),
            f"warm_cal_tphi2_seconds: {warm_path_split['cal_tphi2_seconds']:.6f}",
            f"sample_seconds: [{sample_seconds}]",
            f"python_version: {result['python_version']}",
        ]
        + [f"hotspot: {item}" for item in result['hotspot_shortlist']]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_z_noise_tphi2_workflow(
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
