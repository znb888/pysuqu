"""Timing harness for the public QCRFGRModel frequency-probe workflow."""

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


_BASE_QCRFGR_CONFIG = {
    'capacitance_list': [
        70.319e-15,
        90.238e-15,
        6.304e-15 + 9.8e-15,
        78e-15,
        12.65e-15,
    ],
    'junc_resis_list': [10007.92, 10007.92 / 6],
    'qrcouple': [16.812e-15, 0.0159e-15],
    'flux_list': [0.11, 0.11],
    'trunc_ener_level': [6, 5],
}

_PROBE_FLUX_POINTS = [0.105, 0.11, 0.125, 0.14]
_QUBIT_INDEX = 0


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


def _construct_qcrfgr_model(qcrfgr_cls):
    with redirect_stdout(StringIO()):
        return qcrfgr_cls(**_BASE_QCRFGR_CONFIG)


def _run_qcrfgr_frequency_probe_workload(qcrfgr_cls, probe_helper) -> dict[str, object]:
    model = _construct_qcrfgr_model(qcrfgr_cls)
    original_flux = np.array(model.get_element_matrices('flux'), dtype=float, copy=True)
    baseline_qubit_f01 = float(model.qubit_f01)
    baseline_coupler_f01 = float(model.coupler_f01)
    probe_values = [
        float(probe_helper(model, coupler_flux, qubit_idx=_QUBIT_INDEX))
        for coupler_flux in _PROBE_FLUX_POINTS
    ]
    restored_flux = np.array(model.get_element_matrices('flux'), dtype=float, copy=True)

    min_probe = min(probe_values)
    max_probe = max(probe_values)
    return {
        'checksum': float(sum(probe_values)),
        'baseline_qubit_f01_ghz': baseline_qubit_f01,
        'baseline_coupler_f01_ghz': baseline_coupler_f01,
        'min_probed_frequency_ghz': float(min_probe),
        'max_probed_frequency_ghz': float(max_probe),
        'probe_span_mhz': float((max_probe - min_probe) * 1e3),
        'probe_values_ghz': probe_values,
        'restored_coupler_flux': float(restored_flux[2, 2]),
        'restored_flux_matches_original': bool(np.allclose(restored_flux, original_flux)),
    }


def _profile_warm_path(qcrfgr_cls, probe_helper) -> dict[str, float]:
    from pysuqu.qubit.base import ParameterizedQubit, QubitBase

    for _ in range(3):
        _run_qcrfgr_frequency_probe_workload(qcrfgr_cls, probe_helper)

    tracked_times = {
        'refresh_basic_metrics_seconds': 0.0,
        'generate_ematrix_seconds': 0.0,
        'restore_exact_template_check_seconds': 0.0,
        'restore_exact_template_seconds': 0.0,
    }
    originals = {}

    def _wrap_method(owner, attr: str, label: str) -> None:
        original = getattr(owner, attr)
        originals[(owner, attr)] = original

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                tracked_times[label] += time.perf_counter() - start

        setattr(owner, attr, wrapped)

    _wrap_method(qcrfgr_cls, '_refresh_basic_metrics', 'refresh_basic_metrics_seconds')
    _wrap_method(ParameterizedQubit, '_generate_Ematrix', 'generate_ematrix_seconds')
    _wrap_method(
        QubitBase,
        '_restore_cached_exact_solve_template_if_available',
        'restore_exact_template_check_seconds',
    )
    _wrap_method(
        QubitBase,
        '_restore_exact_solve_template',
        'restore_exact_template_seconds',
    )

    constructor_times = []
    probe_times = []
    measurements = 10
    try:
        for _ in range(measurements):
            start = time.perf_counter()
            model = _construct_qcrfgr_model(qcrfgr_cls)
            constructor_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            for coupler_flux in _PROBE_FLUX_POINTS:
                probe_helper(model, coupler_flux, qubit_idx=_QUBIT_INDEX)
            probe_times.append(time.perf_counter() - start)
    finally:
        for (owner, attr), original in originals.items():
            setattr(owner, attr, original)

    return {
        'constructor_seconds': statistics.fmean(constructor_times),
        'probe_seconds': statistics.fmean(probe_times),
        'refresh_basic_metrics_seconds': tracked_times['refresh_basic_metrics_seconds'] / measurements,
        'generate_ematrix_seconds': tracked_times['generate_ematrix_seconds'] / measurements,
        'restore_exact_template_check_seconds': (
            tracked_times['restore_exact_template_check_seconds'] / measurements
        ),
        'restore_exact_template_seconds': tracked_times['restore_exact_template_seconds'] / measurements,
    }


def _build_hotspot_shortlist(warm_path_split: dict[str, float]) -> list[str]:
    constructor_seconds = warm_path_split['constructor_seconds']
    probe_seconds = warm_path_split['probe_seconds']
    stage_labels = {
        'refresh_basic_metrics_seconds': '_refresh_basic_metrics()',
        'generate_ematrix_seconds': '_generate_Ematrix()',
        'restore_exact_template_seconds': '_restore_exact_solve_template()',
        'restore_exact_template_check_seconds': '_restore_cached_exact_solve_template_if_available()',
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
    next_surface = 'pysuqu.qubit.multi.py' if top_stage == '_refresh_basic_metrics()' else 'pysuqu.qubit.base.py'
    return [
        (
            'Warm-path constructor work '
            f'(~{constructor_seconds:.6f} s/sample) still exceeds the four-point probe path '
            f'(~{probe_seconds:.6f} s/sample) on this public QCRFGR workflow.'
        ),
        (
            'The largest warmed constructor substeps are '
            f'{top_stage} (~{top_seconds:.6f} s) and {second_stage} '
            f'(~{second_seconds:.6f} s).'
        ),
        (
            'The next optimization round should stay on this benchmark and reduce one narrow '
            f'constructor-side owner in `{next_surface}` before reopening the already-cheap '
            'fast probe boundary.'
        ),
    ]


def benchmark_qcrfgr_frequency_probe_workflow(
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
    from pysuqu.qubit.analysis import get_multi_qubit_frequency_at_coupler_flux
    from pysuqu.qubit.multi import QCRFGRModel

    for _ in range(warmups):
        for _ in range(iterations):
            _run_qcrfgr_frequency_probe_workload(
                QCRFGRModel,
                get_multi_qubit_frequency_at_coupler_flux,
            )

    sample_seconds = []
    workload_signature = None
    for _ in range(samples):
        sample_signature = None
        start = time.perf_counter()
        for _ in range(iterations):
            sample_signature = _run_qcrfgr_frequency_probe_workload(
                QCRFGRModel,
                get_multi_qubit_frequency_at_coupler_flux,
            )
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_signature = sample_signature

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0
    warm_path_split = _profile_warm_path(QCRFGRModel, get_multi_qubit_frequency_at_coupler_flux)
    hotspot_shortlist = _build_hotspot_shortlist(warm_path_split)

    return {
        'benchmark': 'qcrfgr_frequency_probe_workflow',
        'workflow': (
            'pysuqu.qubit.multi.QCRFGRModel construction + '
            'pysuqu.qubit.analysis.get_multi_qubit_frequency_at_coupler_flux'
        ),
        'backend': backend,
        'samples': samples,
        'warmups': warmups,
        'iterations_per_sample': iterations,
        'probe_point_count': len(_PROBE_FLUX_POINTS),
        'probe_flux_points': list(_PROBE_FLUX_POINTS),
        'qubit_index': _QUBIT_INDEX,
        'sample_seconds': sample_seconds,
        'mean_seconds': mean_seconds,
        'median_seconds': median_seconds,
        'min_seconds': min(sample_seconds),
        'max_seconds': max(sample_seconds),
        'stdev_seconds': stdev_seconds,
        'workload_checksum': workload_signature['checksum'],
        'workload_signature': workload_signature,
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
    signature = result['workload_signature']
    warm_path_split = result['warm_path_split']
    probe_values = ', '.join(f'{value:.12f}' for value in signature['probe_values_ghz'])
    return '\n'.join(
        [
            f"benchmark: {result['benchmark']}",
            f"workflow: {result['workflow']}",
            f"backend: {result['backend']}",
            (
                'config: '
                f"samples={result['samples']}, warmups={result['warmups']}, "
                f"iterations={result['iterations_per_sample']}, "
                f"probe_point_count={result['probe_point_count']}, "
                f"qubit_index={result['qubit_index']}"
            ),
            f"mean_seconds: {result['mean_seconds']:.6f}",
            f"median_seconds: {result['median_seconds']:.6f}",
            f"min_seconds: {result['min_seconds']:.6f}",
            f"max_seconds: {result['max_seconds']:.6f}",
            f"stdev_seconds: {result['stdev_seconds']:.6f}",
            f"workload_checksum: {result['workload_checksum']:.12f}",
            (
                'workload_signature: '
                f"baseline_qubit_f01_ghz={signature['baseline_qubit_f01_ghz']:.12f}, "
                f"baseline_coupler_f01_ghz={signature['baseline_coupler_f01_ghz']:.12f}, "
                f"min_probed_frequency_ghz={signature['min_probed_frequency_ghz']:.12f}, "
                f"max_probed_frequency_ghz={signature['max_probed_frequency_ghz']:.12f}, "
                f"probe_span_mhz={signature['probe_span_mhz']:.6f}, "
                f"restored_coupler_flux={signature['restored_coupler_flux']:.12f}, "
                f"restored_flux_matches_original={signature['restored_flux_matches_original']}"
            ),
            f"warm_constructor_seconds: {warm_path_split['constructor_seconds']:.6f}",
            f"warm_probe_seconds: {warm_path_split['probe_seconds']:.6f}",
            (
                'warm_refresh_basic_metrics_seconds: '
                f"{warm_path_split['refresh_basic_metrics_seconds']:.6f}"
            ),
            f"warm_generate_ematrix_seconds: {warm_path_split['generate_ematrix_seconds']:.6f}",
            (
                'warm_restore_exact_template_check_seconds: '
                f"{warm_path_split['restore_exact_template_check_seconds']:.6f}"
            ),
            (
                'warm_restore_exact_template_seconds: '
                f"{warm_path_split['restore_exact_template_seconds']:.6f}"
            ),
            f"probe_values_ghz: [{probe_values}]",
            f"sample_seconds: [{sample_seconds}]",
            f"hotspot: {' '.join(result['hotspot_shortlist'])}",
            f"python_version: {result['python_version']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_qcrfgr_frequency_probe_workflow(
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
