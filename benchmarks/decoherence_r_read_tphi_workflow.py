"""Timing harness for the public RNoiseDecoherence readout-Tphi workflow."""

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
_DELAY_LIST = np.linspace(10, 10e3, 100) * 1e-9
_READ_FREQ = 6.5e9
_FIT_P0 = [100e-6, 3e-6, 1.0, 0.0]
_FIT_BOUNDS = ([0, 0, 0, -1], [np.inf, np.inf, 2, 1])


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


def _validate_method(method: str) -> str:
    if method not in {'cal', 'fit'}:
        raise ValueError("method must be 'cal' or 'fit'")

    return method


def _build_readout_noise_freq_hz(*, kappa_hz: float) -> np.ndarray:
    return np.logspace(-2, np.log10(2 * kappa_hz), 100)


def _fit_readout_decay(delay_list: np.ndarray, dephase: np.ndarray) -> None:
    from pysuqu.funclib.mathlib import fit_decay, tphi_decay

    fit_decay(
        delay_list,
        dephase,
        tphi_decay,
        p0=_FIT_P0,
        bounds=_FIT_BOUNDS,
    )


def _run_r_noise_read_tphi_workload(
    r_noise_cls,
    *,
    method: str,
) -> dict[str, float | str | int]:
    model = r_noise_cls(
        psd_freq=_PSD_FREQ,
        psd_S=_PSD_S,
        noise_prop='single',
        is_spectral=True,
        is_print=False,
    )
    result = model.cal_read_tphi(
        method=method,
        experiment='Ramsey',
        read_freq=_READ_FREQ,
        delay_list=_DELAY_LIST,
        is_plot=False,
        is_print=False,
    )
    n_bar = float(model.n_bar)
    tphi_seconds = float(result.value)
    white_noise_temperature = float(model.noise.output_stage.white_noise_temperature)
    workload_signature: dict[str, float | str | int] = {
        'method': result.metadata['method'],
        'experiment': result.metadata['experiment'],
        'source': result.metadata['source'],
        'tphi_seconds': tphi_seconds,
        'n_bar': n_bar,
        'white_noise_temperature_kelvin': white_noise_temperature,
        'read_freq_hz': float(_READ_FREQ),
        'kappa_hz': float(model.kappa),
        'chi_hz': float(model.couple_term),
        'frequency_min_hz': float(model.psd_freq.min()),
        'frequency_max_hz': float(model.psd_freq.max()),
        'checksum': (
            tphi_seconds
            + n_bar
            + white_noise_temperature
            + float(model.kappa) / 1e6
            + float(model.couple_term) / 1e6
        ),
    }
    if method == 'fit':
        readout_noise_freq_hz = _build_readout_noise_freq_hz(kappa_hz=float(model.kappa))
        readout_psd_mean = float(np.mean(model.psd_read))
        dephase_terminal = float(model.dephase[-1])
        tphi2_seconds = float(result.fit_diagnostics['tphi2'])
        fit_error_seconds = float(result.fit_diagnostics['fit_error'])
        tphi2_fit_error_seconds = float(result.fit_diagnostics['tphi2_fit_error'])
        workload_signature.update(
            {
                'tphi2_seconds': tphi2_seconds,
                'fit_error_seconds': fit_error_seconds,
                'tphi2_fit_error_seconds': tphi2_fit_error_seconds,
                'readout_psd_points': int(len(model.psd_read)),
                'dephase_points': int(len(model.dephase)),
                'readout_psd_mean': readout_psd_mean,
                'dephase_terminal': dephase_terminal,
                'readout_frequency_min_hz': float(readout_noise_freq_hz.min()),
                'readout_frequency_max_hz': float(readout_noise_freq_hz.max()),
                'checksum': (
                    workload_signature['checksum']
                    + tphi2_seconds
                    + fit_error_seconds * 1e9
                    + tphi2_fit_error_seconds * 1e9
                    + readout_psd_mean * 1e18
                    + dephase_terminal
                ),
            }
        )

    return workload_signature


def _profile_warm_path(
    r_noise_cls,
    *,
    method: str,
) -> dict[str, float]:
    model = r_noise_cls(
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
        model.r_analyzer = model._build_r_analyzer()
        model.cal_nbar(read_freq=_READ_FREQ, is_print=False)
        model.cal_read_tphi(
            method=method,
            experiment='Ramsey',
            read_freq=_READ_FREQ,
            delay_list=_DELAY_LIST,
            is_plot=False,
            is_print=False,
        )
        model.cal_readcavity_psd(read_freq=_READ_FREQ)
        dephase = model.cal_read_dephase(
            experiment='Ramsey',
            read_freq=_READ_FREQ,
            delay_list=_DELAY_LIST,
        )
        if method == 'fit':
            _fit_readout_decay(_DELAY_LIST, dephase)

    constructor_times = []
    build_qubit_times = []
    build_noise_times = []
    build_r_analyzer_times = []
    cal_nbar_times = []
    cal_read_tphi_times = []
    cal_readcavity_psd_times = []
    cal_read_dephase_times = []
    fit_decay_times = []

    for _ in range(10):
        start = time.perf_counter()
        r_noise_cls(
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
        model.r_analyzer = model._build_r_analyzer()
        build_r_analyzer_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.cal_nbar(read_freq=_READ_FREQ, is_print=False)
        cal_nbar_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.cal_read_tphi(
            method=method,
            experiment='Ramsey',
            read_freq=_READ_FREQ,
            delay_list=_DELAY_LIST,
            is_plot=False,
            is_print=False,
        )
        cal_read_tphi_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.cal_readcavity_psd(read_freq=_READ_FREQ)
        cal_readcavity_psd_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        dephase = model.cal_read_dephase(
            experiment='Ramsey',
            read_freq=_READ_FREQ,
            delay_list=_DELAY_LIST,
        )
        cal_read_dephase_times.append(time.perf_counter() - start)

        if method == 'fit':
            start = time.perf_counter()
            _fit_readout_decay(_DELAY_LIST, dephase)
            fit_decay_times.append(time.perf_counter() - start)

    return {
        'constructor_seconds': statistics.fmean(constructor_times),
        'build_qubit_seconds': statistics.fmean(build_qubit_times),
        'build_noise_seconds': statistics.fmean(build_noise_times),
        'build_r_analyzer_seconds': statistics.fmean(build_r_analyzer_times),
        'cal_nbar_seconds': statistics.fmean(cal_nbar_times),
        'cal_read_tphi_seconds': statistics.fmean(cal_read_tphi_times),
        'cal_readcavity_psd_seconds': statistics.fmean(cal_readcavity_psd_times),
        'cal_read_dephase_seconds': statistics.fmean(cal_read_dephase_times),
        'fit_decay_seconds': statistics.fmean(fit_decay_times) if fit_decay_times else 0.0,
    }


def _build_hotspot_shortlist(
    warm_path_split: dict[str, float],
    *,
    method: str,
) -> list[str]:
    constructor_seconds = warm_path_split['constructor_seconds']
    cal_read_tphi_seconds = warm_path_split['cal_read_tphi_seconds']
    constructor_ranked = sorted(
        (
            ('build_qubit', warm_path_split['build_qubit_seconds']),
            ('build_noise', warm_path_split['build_noise_seconds']),
            ('cal_read_tphi', warm_path_split['cal_read_tphi_seconds']),
            ('cal_nbar', warm_path_split['cal_nbar_seconds']),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_constructor_stage, top_constructor_seconds = constructor_ranked[0]
    second_constructor_stage, second_constructor_seconds = constructor_ranked[1]
    if method == 'cal':
        readout_ranked = sorted(
            (
                ('cal_readcavity_psd', warm_path_split['cal_readcavity_psd_seconds']),
                ('cal_read_tphi', warm_path_split['cal_read_tphi_seconds']),
                ('cal_read_dephase', warm_path_split['cal_read_dephase_seconds']),
                ('cal_nbar', warm_path_split['cal_nbar_seconds']),
                ('build_r_analyzer', warm_path_split['build_r_analyzer_seconds']),
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_readout_stage, top_readout_seconds = readout_ranked[0]
        next_module = (
            'pysuqu.decoherence.analysis.py'
            if top_readout_stage != 'build_r_analyzer'
            else 'pysuqu.decoherence.noise.py'
        )
        return [
            (
                'Warm-path constructor work '
                f'(~{constructor_seconds:.6f} s/sample) dominates analytical cal_read_tphi '
                f'(~{cal_read_tphi_seconds:.6f} s/sample) on this public R workflow.'
            ),
            (
                'The largest warmed constructor substeps are '
                f'{top_constructor_stage} (~{top_constructor_seconds:.6f} s) and '
                f'{second_constructor_stage} (~{second_constructor_seconds:.6f} s), while the largest '
                f'readout-specific helper is {top_readout_stage} (~{top_readout_seconds:.6f} s).'
            ),
            (
                'Because the first measured readout-specific work currently sits behind the '
                f'R-side analyzer helpers, the next narrow follow-up should target {next_module} '
                'before opening a separate readout-noise helper hypothesis.'
            ),
        ]

    helper_ranked = sorted(
        (
            ('fit_decay', warm_path_split['fit_decay_seconds']),
            ('cal_readcavity_psd', warm_path_split['cal_readcavity_psd_seconds']),
            ('cal_read_dephase', warm_path_split['cal_read_dephase_seconds']),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_helper_stage, top_helper_seconds = helper_ranked[0]
    second_helper_stage, second_helper_seconds = helper_ranked[1]
    if cal_read_tphi_seconds <= constructor_seconds:
        next_round = (
            'Because warmed constructor work now outweighs the fit-path branch on this public '
            'workflow, the next honest step is a refresh round that preserves the fit-support win '
            'before reopening constructor-side or analyzer-side follow-up work.'
        )
    elif top_helper_stage == 'fit_decay':
        next_round = (
            'The next optimization round should stay on this public fit benchmark and reduce the '
            'direct `fit_decay(...)` support path in `pysuqu.funclib.mathlib.py` before '
            'claiming an analyzer-side speedup.'
        )
    else:
        next_round = (
            'The next optimization round should stay on this public fit benchmark and reduce one '
            'measured `ReadoutCavityAnalyzer` helper in `pysuqu.decoherence.analysis.py`, '
            f'starting with `{top_helper_stage}(...)`.'
        )
    return [
        (
            'Warm-path constructor work '
            f'(~{constructor_seconds:.6f} s/sample) and fit-path cal_read_tphi '
            f'(~{cal_read_tphi_seconds:.6f} s/sample) are now both directly measured on this public '
            'R workflow.'
        ),
        (
            'Within the fit branch, the largest warmed substeps are '
            f'{top_helper_stage} (~{top_helper_seconds:.6f} s) and '
            f'{second_helper_stage} (~{second_helper_seconds:.6f} s).'
        ),
        next_round,
    ]


def benchmark_r_noise_read_tphi_workflow(
    *,
    method: str = 'cal',
    samples: int = 7,
    warmups: int = 2,
    iterations: int = 3,
    use_test_stubs: bool = False,
) -> dict[str, object]:
    method = _validate_method(method)
    if samples < 1:
        raise ValueError('samples must be >= 1')
    if warmups < 0:
        raise ValueError('warmups must be >= 0')
    if iterations < 1:
        raise ValueError('iterations must be >= 1')

    backend = _ensure_runtime_backends(use_test_stubs=use_test_stubs)
    from pysuqu.decoherence import RNoiseDecoherence

    for _ in range(warmups):
        for _ in range(iterations):
            _run_r_noise_read_tphi_workload(RNoiseDecoherence, method=method)

    sample_seconds = []
    workload_signature = {}
    workload_checksum = 0.0
    for _ in range(samples):
        sample_signature = {}
        start = time.perf_counter()
        for _ in range(iterations):
            sample_signature = _run_r_noise_read_tphi_workload(RNoiseDecoherence, method=method)
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_signature = sample_signature
        workload_checksum = sample_signature['checksum']

    warm_path_split = _profile_warm_path(RNoiseDecoherence, method=method)
    hotspot_shortlist = _build_hotspot_shortlist(warm_path_split, method=method)

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0

    return {
        'benchmark': 'r_noise_read_tphi_workflow',
        'method': method,
        'workflow': (
            "pysuqu.decoherence.RNoiseDecoherence constructor + "
            f"cal_read_tphi(method='{method}', experiment='Ramsey')"
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
    parser.add_argument(
        '--method',
        choices=('cal', 'fit'),
        default='cal',
        help='Public readout-Tphi path to benchmark.',
    )
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
            f"method: {result['method']}",
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
            f"tphi_seconds: {workload_signature['tphi_seconds']:.12e}",
            f"n_bar: {workload_signature['n_bar']:.12f}",
            (
                'white_noise_temperature_kelvin: '
                f"{workload_signature['white_noise_temperature_kelvin']:.12f}"
            ),
            f"read_freq_hz: {workload_signature['read_freq_hz']:.6f}",
            f"kappa_hz: {workload_signature['kappa_hz']:.6f}",
            f"chi_hz: {workload_signature['chi_hz']:.6f}",
            f"frequency_min_hz: {workload_signature['frequency_min_hz']:.6f}",
            f"frequency_max_hz: {workload_signature['frequency_max_hz']:.6f}",
            f"workload_checksum: {result['workload_checksum']:.12f}",
            f"warm_constructor_seconds: {warm_path_split['constructor_seconds']:.6f}",
            f"warm_build_qubit_seconds: {warm_path_split['build_qubit_seconds']:.6f}",
            f"warm_build_noise_seconds: {warm_path_split['build_noise_seconds']:.6f}",
            f"warm_build_r_analyzer_seconds: {warm_path_split['build_r_analyzer_seconds']:.6f}",
            f"warm_cal_nbar_seconds: {warm_path_split['cal_nbar_seconds']:.6f}",
            f"warm_cal_read_tphi_seconds: {warm_path_split['cal_read_tphi_seconds']:.6f}",
            (
                'warm_cal_readcavity_psd_seconds: '
                f"{warm_path_split['cal_readcavity_psd_seconds']:.6f}"
            ),
            f"warm_cal_read_dephase_seconds: {warm_path_split['cal_read_dephase_seconds']:.6f}",
            f"warm_fit_decay_seconds: {warm_path_split['fit_decay_seconds']:.6f}",
            f"sample_seconds: [{sample_seconds}]",
            f"python_version: {result['python_version']}",
        ]
        + (
            [
                f"tphi2_seconds: {workload_signature['tphi2_seconds']:.12e}",
                f"fit_error_seconds: {workload_signature['fit_error_seconds']:.12e}",
                (
                    'tphi2_fit_error_seconds: '
                    f"{workload_signature['tphi2_fit_error_seconds']:.12e}"
                ),
                f"readout_psd_points: {workload_signature['readout_psd_points']}",
                f"dephase_points: {workload_signature['dephase_points']}",
                f"readout_psd_mean: {workload_signature['readout_psd_mean']:.12e}",
                f"dephase_terminal: {workload_signature['dephase_terminal']:.12f}",
                f"readout_frequency_min_hz: {workload_signature['readout_frequency_min_hz']:.6f}",
                f"readout_frequency_max_hz: {workload_signature['readout_frequency_max_hz']:.6f}",
            ]
            if result['method'] == 'fit'
            else []
        )
        + [f"hotspot: {item}" for item in result['hotspot_shortlist']]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_r_noise_read_tphi_workflow(
        method=args.method,
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
