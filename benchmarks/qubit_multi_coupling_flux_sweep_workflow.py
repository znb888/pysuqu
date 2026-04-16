"""Timing harness for the public FGF1V1Coupling coupling-flux sweep workflow."""

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


_COUPLER_FLUX_POINTS = [0.095, 0.11, 0.125, 0.14]
_COUPLING_METHOD = 'ES'
_COLD_CONSTRUCTOR_PROBE_SAMPLES = 5
_COLD_REPLAY_PROBE_COLD_SAMPLES = 5
_COLD_REPLAY_PROBE_REPLAY_ROUND_TRIPS = 250


def _build_fgf1v1_config() -> dict[str, object]:
    c_j = 9.8e-15
    c_q1_total = 165e-15
    c_q2_total = 165e-15
    c_qc = 23.2e-15
    c_q_ground = 5.2e-15
    c_qq = 2.1e-15
    c_coupler_total = 142e-15
    c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
    c_12_ground = c_q2_total - c_q_ground
    capacitance_list = [
        c_11_ground,
        c_12_ground,
        c_q_ground + c_j,
        c_coupler_total - 2 * c_qc + 6 * c_j,
        c_q_ground + c_j,
        c_12_ground,
        c_11_ground,
        c_qq,
        c_qc,
        c_qc,
    ]
    return {
        'capacitance_list': capacitance_list,
        'junc_resis_list': [7400, 7400 / 6, 7400],
        'qrcouple': [18.34e-15, 0.02e-15],
        'flux_list': [0.11, 0.11, 0.11],
        'trunc_ener_level': [3, 2, 3],
        'is_print': False,
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


def _construct_fgf1v1_model(fgf1v1_cls):
    with redirect_stdout(StringIO()):
        return fgf1v1_cls(**_build_fgf1v1_config())


def _clear_exact_solve_template_cache() -> None:
    from pysuqu.qubit.base import QubitBase
    from pysuqu.qubit.multi import _clear_fgf1v1_basic_metric_cache

    QubitBase._clear_exact_solve_template_cache()
    _clear_fgf1v1_basic_metric_cache()


def _build_replay_flux_inputs(model) -> tuple[np.ndarray, np.ndarray, float, float]:
    original_flux = np.array(model.get_element_matrices('flux'), dtype=float, copy=True)
    baseline_coupler_flux = float(original_flux[2, 2])
    target_coupler_flux = next(
        flux_point
        for flux_point in _COUPLER_FLUX_POINTS
        if not np.isclose(flux_point, baseline_coupler_flux)
    )
    replay_flux = np.array(original_flux, copy=True)
    replay_flux[2, 2] = target_coupler_flux
    return original_flux, replay_flux, baseline_coupler_flux, target_coupler_flux


def _run_coupling_flux_sweep_workload(
    fgf1v1_cls,
    sweep_helper,
    *,
    clear_exact_solve_cache: bool = False,
) -> dict[str, object]:
    if clear_exact_solve_cache:
        _clear_exact_solve_template_cache()

    model = _construct_fgf1v1_model(fgf1v1_cls)
    original_flux = np.array(model.get_element_matrices('flux'), dtype=float, copy=True)
    baseline_qubit1_f01 = float(model.qubit1_f01)
    baseline_qubit2_f01 = float(model.qubit2_f01)
    baseline_coupler_f01 = float(model.coupler_f01)
    baseline_qr_g = float(model.qr_g)
    baseline_qq_g = float(model.qq_g)
    baseline_qc_g = float(model.qc_g)
    baseline_qq_geff = float(model.qq_geff)

    with redirect_stdout(StringIO()):
        result = sweep_helper(
            model,
            _COUPLER_FLUX_POINTS,
            method=_COUPLING_METHOD,
            is_plot=False,
        )

    coupling_values = np.asarray(result.coupling_values, dtype=float)
    restored_flux = np.array(model.get_element_matrices('flux'), dtype=float, copy=True)
    turnoff_index = int(np.argmin(np.abs(coupling_values)))
    restored_signature = {
        'qubit1_f01_matches_baseline': bool(np.isclose(model.qubit1_f01, baseline_qubit1_f01)),
        'qubit2_f01_matches_baseline': bool(np.isclose(model.qubit2_f01, baseline_qubit2_f01)),
        'coupler_f01_matches_baseline': bool(np.isclose(model.coupler_f01, baseline_coupler_f01)),
        'qr_g_matches_baseline': bool(np.isclose(model.qr_g, baseline_qr_g)),
        'qq_g_matches_baseline': bool(np.isclose(model.qq_g, baseline_qq_g)),
        'qc_g_matches_baseline': bool(np.isclose(model.qc_g, baseline_qc_g)),
        'qq_geff_matches_baseline': bool(np.isclose(model.qq_geff, baseline_qq_geff)),
    }

    return {
        'checksum': float(coupling_values.sum()),
        'baseline_qubit1_f01_ghz': baseline_qubit1_f01,
        'baseline_qubit2_f01_ghz': baseline_qubit2_f01,
        'baseline_coupler_f01_ghz': baseline_coupler_f01,
        'baseline_qr_g_hz': baseline_qr_g,
        'baseline_qq_g_ghz': baseline_qq_g,
        'baseline_qc_g_ghz': baseline_qc_g,
        'baseline_qq_geff_ghz': baseline_qq_geff,
        'min_coupling_ghz': float(coupling_values.min()),
        'max_coupling_ghz': float(coupling_values.max()),
        'turnoff_coupler_flux': float(_COUPLER_FLUX_POINTS[turnoff_index]),
        'turnoff_coupling_ghz': float(coupling_values[turnoff_index]),
        'coupling_values_ghz': [float(value) for value in coupling_values.tolist()],
        'restored_coupler_flux': float(restored_flux[2, 2]),
        'restored_flux_matches_original': bool(np.allclose(restored_flux, original_flux)),
        'restored_signature': restored_signature,
    }


def _measure_cache_isolation_drift(fgf1v1_cls, sweep_helper) -> dict[str, object]:
    cold_signature = _run_coupling_flux_sweep_workload(
        fgf1v1_cls,
        sweep_helper,
        clear_exact_solve_cache=True,
    )
    warmed_signature = _run_coupling_flux_sweep_workload(
        fgf1v1_cls,
        sweep_helper,
        clear_exact_solve_cache=False,
    )
    cold_values = np.asarray(cold_signature['coupling_values_ghz'], dtype=float)
    warmed_values = np.asarray(warmed_signature['coupling_values_ghz'], dtype=float)

    return {
        'cold_coupling_values_ghz': [float(value) for value in cold_values.tolist()],
        'warmed_coupling_values_ghz': [float(value) for value in warmed_values.tolist()],
        'warmed_matches_cold': bool(np.allclose(warmed_values, cold_values)),
    }


def _start_stage_tracking(fgf1v1_cls) -> tuple[dict[str, float], dict[tuple[object, str], object]]:
    from pysuqu.qubit.base import QubitBase

    tracked_times = {
        'change_para_seconds': 0.0,
        'refresh_basic_metrics_seconds': 0.0,
        'generate_hamiltonian_seconds': 0.0,
        'get_qq_ecouple_seconds': 0.0,
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

    _wrap_method(QubitBase, 'change_para', 'change_para_seconds')
    _wrap_method(fgf1v1_cls, '_refresh_basic_metrics', 'refresh_basic_metrics_seconds')
    _wrap_method(QubitBase, '_generate_hamiltonian', 'generate_hamiltonian_seconds')
    _wrap_method(fgf1v1_cls, 'get_qq_ecouple', 'get_qq_ecouple_seconds')
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
    return tracked_times, originals


def _restore_stage_tracking(originals: dict[tuple[object, str], object]) -> None:
    for (owner, attr), original in originals.items():
        setattr(owner, attr, original)


def _normalize_stage_tracking(tracked_times: dict[str, float], divisor: int) -> dict[str, float]:
    return {
        'change_para_seconds': tracked_times['change_para_seconds'] / divisor,
        'refresh_basic_metrics_seconds': tracked_times['refresh_basic_metrics_seconds'] / divisor,
        'generate_hamiltonian_seconds': tracked_times['generate_hamiltonian_seconds'] / divisor,
        'get_qq_ecouple_seconds': tracked_times['get_qq_ecouple_seconds'] / divisor,
        'restore_exact_template_check_seconds': (
            tracked_times['restore_exact_template_check_seconds'] / divisor
        ),
        'restore_exact_template_seconds': tracked_times['restore_exact_template_seconds'] / divisor,
    }


def _profile_warm_path(fgf1v1_cls, sweep_helper) -> dict[str, float]:
    for _ in range(2):
        _run_coupling_flux_sweep_workload(fgf1v1_cls, sweep_helper)

    constructor_times = []
    sweep_times = []
    measurements = 5
    tracked_times, originals = _start_stage_tracking(fgf1v1_cls)
    try:
        for _ in range(measurements):
            start = time.perf_counter()
            model = _construct_fgf1v1_model(fgf1v1_cls)
            constructor_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            with redirect_stdout(StringIO()):
                sweep_helper(
                    model,
                    _COUPLER_FLUX_POINTS,
                    method=_COUPLING_METHOD,
                    is_plot=False,
                )
            sweep_times.append(time.perf_counter() - start)
    finally:
        _restore_stage_tracking(originals)

    return {
        'constructor_seconds': statistics.fmean(constructor_times),
        'sweep_seconds': statistics.fmean(sweep_times),
        **_normalize_stage_tracking(tracked_times, measurements),
    }


def _start_constructor_tracking(
    fgf1v1_cls,
) -> tuple[dict[str, float], dict[tuple[object, str], object]]:
    from pysuqu.qubit.base import ParameterizedQubit, QubitBase
    from pysuqu.qubit.solver import HamiltonianEvo

    tracked_times = {
        'parameterized_init_seconds': 0.0,
        'generate_ematrix_seconds': 0.0,
        'update_ej_seconds': 0.0,
        'restore_exact_template_check_seconds': 0.0,
        'generate_hamiltonian_seconds': 0.0,
        'refresh_basic_metrics_seconds': 0.0,
        'set_solver_result_seconds': 0.0,
        'store_exact_template_seconds': 0.0,
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

    _wrap_method(ParameterizedQubit, '__init__', 'parameterized_init_seconds')
    _wrap_method(ParameterizedQubit, '_generate_Ematrix', 'generate_ematrix_seconds')
    _wrap_method(ParameterizedQubit, '_update_Ej', 'update_ej_seconds')
    _wrap_method(
        QubitBase,
        '_restore_cached_exact_solve_template_if_available',
        'restore_exact_template_check_seconds',
    )
    _wrap_method(QubitBase, '_generate_hamiltonian', 'generate_hamiltonian_seconds')
    _wrap_method(fgf1v1_cls, '_refresh_basic_metrics', 'refresh_basic_metrics_seconds')
    _wrap_method(HamiltonianEvo, '_set_solver_result', 'set_solver_result_seconds')
    _wrap_method(QubitBase, '_store_current_exact_solve_template', 'store_exact_template_seconds')
    return tracked_times, originals


def _measure_cold_constructor_probe(fgf1v1_cls) -> dict[str, float]:
    constructor_times = []
    constructor_stage_totals = {
        'parameterized_init_seconds': 0.0,
        'generate_ematrix_seconds': 0.0,
        'update_ej_seconds': 0.0,
        'restore_exact_template_check_seconds': 0.0,
        'generate_hamiltonian_seconds': 0.0,
        'refresh_basic_metrics_seconds': 0.0,
        'set_solver_result_seconds': 0.0,
        'store_exact_template_seconds': 0.0,
    }

    for _ in range(_COLD_CONSTRUCTOR_PROBE_SAMPLES):
        _clear_exact_solve_template_cache()
        tracked_times, originals = _start_constructor_tracking(fgf1v1_cls)
        try:
            start = time.perf_counter()
            _construct_fgf1v1_model(fgf1v1_cls)
            constructor_times.append(time.perf_counter() - start)
        finally:
            _restore_stage_tracking(originals)
            for key in constructor_stage_totals:
                constructor_stage_totals[key] += tracked_times[key]

    constructor_seconds = statistics.fmean(constructor_times)
    parameterized_init_seconds = (
        constructor_stage_totals['parameterized_init_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    generate_ematrix_seconds = (
        constructor_stage_totals['generate_ematrix_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    update_ej_seconds = (
        constructor_stage_totals['update_ej_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    restore_exact_template_check_seconds = (
        constructor_stage_totals['restore_exact_template_check_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    generate_hamiltonian_seconds = (
        constructor_stage_totals['generate_hamiltonian_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    refresh_basic_metrics_seconds = (
        constructor_stage_totals['refresh_basic_metrics_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    set_solver_result_seconds = (
        constructor_stage_totals['set_solver_result_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    store_exact_template_seconds = (
        constructor_stage_totals['store_exact_template_seconds'] / _COLD_CONSTRUCTOR_PROBE_SAMPLES
    )
    parameterized_init_other_seconds = max(
        parameterized_init_seconds
        - generate_ematrix_seconds
        - update_ej_seconds
        - restore_exact_template_check_seconds
        - generate_hamiltonian_seconds
        - set_solver_result_seconds
        - store_exact_template_seconds,
        0.0,
    )
    fgf1v1_init_glue_seconds = max(
        constructor_seconds
        - parameterized_init_seconds
        - refresh_basic_metrics_seconds,
        0.0,
    )
    constructor_other_seconds = parameterized_init_other_seconds + fgf1v1_init_glue_seconds

    return {
        'samples': _COLD_CONSTRUCTOR_PROBE_SAMPLES,
        'constructor_seconds': constructor_seconds,
        'parameterized_init_seconds': parameterized_init_seconds,
        'generate_ematrix_seconds': generate_ematrix_seconds,
        'update_ej_seconds': update_ej_seconds,
        'restore_exact_template_check_seconds': restore_exact_template_check_seconds,
        'generate_hamiltonian_seconds': generate_hamiltonian_seconds,
        'refresh_basic_metrics_seconds': refresh_basic_metrics_seconds,
        'set_solver_result_seconds': set_solver_result_seconds,
        'store_exact_template_seconds': store_exact_template_seconds,
        'parameterized_init_other_seconds': parameterized_init_other_seconds,
        'fgf1v1_init_glue_seconds': fgf1v1_init_glue_seconds,
        'constructor_other_seconds': constructor_other_seconds,
    }


def _build_hotspot_shortlist(
    warm_path_split: dict[str, float],
    cache_isolation_check: dict[str, object],
    cold_constructor_probe: dict[str, float],
    cold_replay_probe: dict[str, object],
) -> list[str]:
    if not cache_isolation_check['warmed_matches_cold']:
        cold_values = ', '.join(
            f'{value:.12f}'
            for value in cache_isolation_check['cold_coupling_values_ghz']
        )
        warmed_values = ', '.join(
            f'{value:.12f}'
            for value in cache_isolation_check['warmed_coupling_values_ghz']
        )
        return [
            (
                'Repeated identical FGF1V1 constructor reuse currently changes the public coupling '
                'sweep output on the same flux points, so the exact-solve restore path is not yet a '
                'trustworthy warmed benchmark owner for this workflow.'
            ),
            (
                'The cache-isolated sweep returns '
                f'[{cold_values}], while the immediate warmed replay returns [{warmed_values}] '
                'on the same public inputs.'
            ),
            (
                'The next round should stay on this benchmark and repair exact-template cache '
                'isolation on the `FGF1V1Coupling` constructor/change-by-flux boundary before '
                'claiming a performance follow-up.'
            ),
        ]

    constructor_stage_labels = {
        'generate_hamiltonian_seconds': 'fresh constructor `_generate_hamiltonian()`',
        'refresh_basic_metrics_seconds': 'fresh constructor `_refresh_basic_metrics()`',
        'generate_ematrix_seconds': 'fresh constructor `_generate_Ematrix()`',
        'update_ej_seconds': 'fresh constructor `_update_Ej()`',
        'restore_exact_template_check_seconds': (
            'fresh constructor `_restore_cached_exact_solve_template_if_available()`'
        ),
        'set_solver_result_seconds': 'fresh constructor `HamiltonianEvo._set_solver_result()`',
        'store_exact_template_seconds': 'fresh constructor `_store_current_exact_solve_template()`',
        'parameterized_init_other_seconds': 'remaining `ParameterizedQubit.__init__()` glue',
        'fgf1v1_init_glue_seconds': 'remaining `FGF1V1Coupling.__init__()` glue',
    }
    constructor_stage_surfaces = {
        'generate_hamiltonian_seconds': '`pysuqu.qubit.base.py`',
        'refresh_basic_metrics_seconds': '`pysuqu.qubit.multi.py`',
        'generate_ematrix_seconds': '`pysuqu.qubit.base.py`',
        'update_ej_seconds': '`pysuqu.qubit.base.py`',
        'restore_exact_template_check_seconds': '`pysuqu.qubit.base.py`',
        'set_solver_result_seconds': '`pysuqu.qubit.solver.py`',
        'store_exact_template_seconds': '`pysuqu.qubit.base.py`',
        'parameterized_init_other_seconds': '`pysuqu.qubit.base.py`',
        'fgf1v1_init_glue_seconds': '`pysuqu.qubit.multi.py`',
    }
    transition_stage_labels = {
        'cold_refresh_basic_metrics_seconds': 'cold `_refresh_basic_metrics()`',
        'cold_generate_hamiltonian_seconds': 'cold `_generate_hamiltonian()`',
        'cold_restore_exact_template_check_seconds': (
            'cold `_restore_cached_exact_solve_template_if_available()`'
        ),
        'cold_get_qq_ecouple_seconds': 'cold `get_qq_ecouple()`',
        'cold_restore_exact_template_seconds': 'cold `_restore_exact_solve_template()`',
    }
    transition_stage_surfaces = {
        'cold_refresh_basic_metrics_seconds': '`pysuqu.qubit.multi.py`',
        'cold_generate_hamiltonian_seconds': '`pysuqu.qubit.base.py`',
        'cold_restore_exact_template_check_seconds': '`pysuqu.qubit.base.py`',
        'cold_get_qq_ecouple_seconds': '`pysuqu.qubit.multi.py`',
        'cold_restore_exact_template_seconds': '`pysuqu.qubit.base.py`',
    }
    replay_stage_labels = {
        'replay_refresh_basic_metrics_seconds': 'replay `_refresh_basic_metrics()`',
        'replay_change_para_seconds': 'replay `change_para()`',
        'replay_restore_exact_template_check_seconds': (
            'replay `_restore_cached_exact_solve_template_if_available()`'
        ),
        'replay_get_qq_ecouple_seconds': 'replay `get_qq_ecouple()`',
    }
    replay_stage_surfaces = {
        'replay_refresh_basic_metrics_seconds': '`pysuqu.qubit.multi.py`',
        'replay_change_para_seconds': '`pysuqu.qubit.base.py`',
        'replay_restore_exact_template_check_seconds': '`pysuqu.qubit.base.py`',
        'replay_get_qq_ecouple_seconds': '`pysuqu.qubit.multi.py`',
    }
    ranked_constructor_stages = sorted(
        (
            (key, cold_constructor_probe[key])
            for key in constructor_stage_labels
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_key, top_seconds = ranked_constructor_stages[0]
    second_key, second_seconds = ranked_constructor_stages[1]
    top_stage = constructor_stage_labels[top_key]
    second_stage = constructor_stage_labels[second_key]
    ranked_transition_stages = sorted(
        (
            (key, cold_replay_probe[key])
            for key in transition_stage_labels
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    transition_top_key, transition_top_seconds = ranked_transition_stages[0]
    transition_top_stage = transition_stage_labels[transition_top_key]
    ranked_replay_stages = sorted(
        (
            (key, cold_replay_probe[key])
            for key in replay_stage_labels
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    replay_top_key, replay_top_seconds = ranked_replay_stages[0]
    replay_top_stage = replay_stage_labels[replay_top_key]
    constructor_seconds = cold_constructor_probe['constructor_seconds']
    cold_transition_seconds = cold_replay_probe['cold_target_transition_seconds']
    replay_transition_seconds = cold_replay_probe['replay_transition_seconds']
    if constructor_seconds >= cold_transition_seconds and constructor_seconds >= replay_transition_seconds:
        dominant_boundary = 'fresh constructor'
        next_surface = constructor_stage_surfaces[top_key]
        first_line = (
            'Focused constructor/cold/replay probing shows fresh construction '
            f'(~{constructor_seconds:.6f} s) still exceeds the first off-baseline transition '
            f'(~{cold_transition_seconds:.6f} s) and alternating same-instance replay '
            f'(~{replay_transition_seconds:.6f} s/transition) on this public FGF1V1 workflow.'
        )
        second_line = (
            'Within fresh construction, the largest measured stages are '
            f'{top_stage} (~{top_seconds:.6f} s) and {second_stage} '
            f'(~{second_seconds:.6f} s).'
        )
    elif cold_transition_seconds >= replay_transition_seconds:
        dominant_boundary = 'cold off-baseline transition'
        next_surface = transition_stage_surfaces[transition_top_key]
        first_line = (
            'Focused constructor/cold/replay probing shows the first off-baseline transition '
            f'(~{cold_transition_seconds:.6f} s) still exceeds fresh construction '
            f'(~{constructor_seconds:.6f} s) and alternating same-instance replay '
            f'(~{replay_transition_seconds:.6f} s/transition) on this public FGF1V1 workflow.'
        )
        second_line = (
            'Within fresh construction, the largest measured stages are '
            f'{top_stage} (~{top_seconds:.6f} s) and {second_stage} '
            f'(~{second_seconds:.6f} s), while the transition still leads through '
            f'{transition_top_stage} (~{transition_top_seconds:.6f} s).'
        )
    else:
        dominant_boundary = 'same-instance replay'
        next_surface = replay_stage_surfaces[replay_top_key]
        first_line = (
            'Focused constructor/cold/replay probing shows alternating same-instance replay '
            f'(~{replay_transition_seconds:.6f} s/transition) now exceeds fresh construction '
            f'(~{constructor_seconds:.6f} s) and the first off-baseline transition '
            f'(~{cold_transition_seconds:.6f} s) on this public FGF1V1 workflow.'
        )
        second_line = (
            'Within fresh construction, the largest measured stages are '
            f'{top_stage} (~{top_seconds:.6f} s) and {second_stage} '
            f'(~{second_seconds:.6f} s), but replay now leads through '
            f'{replay_top_stage} (~{replay_top_seconds:.6f} s).'
        )

    return [
        first_line,
        second_line,
        (
            'The next optimization round should stay on this benchmark and reduce one narrow '
            f'{dominant_boundary}-side owner near {next_surface} before opening a different '
            'multi-qubit hypothesis.'
        ),
    ]


def _measure_cold_replay_probe(fgf1v1_cls) -> dict[str, object]:
    cold_constructor_times = []
    cold_target_transition_times = []
    cold_target_couplings = []
    cold_stage_totals = {
        'change_para_seconds': 0.0,
        'refresh_basic_metrics_seconds': 0.0,
        'generate_hamiltonian_seconds': 0.0,
        'get_qq_ecouple_seconds': 0.0,
        'restore_exact_template_check_seconds': 0.0,
        'restore_exact_template_seconds': 0.0,
    }

    baseline_coupling = None
    baseline_coupler_flux = None
    target_coupler_flux = None

    for _ in range(_COLD_REPLAY_PROBE_COLD_SAMPLES):
        _clear_exact_solve_template_cache()
        tracked_times, originals = _start_stage_tracking(fgf1v1_cls)
        try:
            start = time.perf_counter()
            model = _construct_fgf1v1_model(fgf1v1_cls)
            cold_constructor_times.append(time.perf_counter() - start)

            original_flux, replay_flux, baseline_coupler_flux, target_coupler_flux = (
                _build_replay_flux_inputs(model)
            )
            if baseline_coupling is None:
                baseline_coupling = float(model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False))

            start = time.perf_counter()
            model.change_para(flux=replay_flux)
            cold_target_transition_times.append(time.perf_counter() - start)
            cold_target_couplings.append(
                float(model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False))
            )
        finally:
            _restore_stage_tracking(originals)
            for key in cold_stage_totals:
                cold_stage_totals[key] += tracked_times[key]

    _clear_exact_solve_template_cache()
    replay_model = _construct_fgf1v1_model(fgf1v1_cls)
    original_flux, replay_flux, baseline_coupler_flux, target_coupler_flux = _build_replay_flux_inputs(
        replay_model
    )
    if baseline_coupling is None:
        baseline_coupling = float(replay_model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False))

    replay_model.change_para(flux=replay_flux)
    seeded_target_coupling = float(replay_model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False))
    replay_model.change_para(flux=original_flux)
    replay_model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False)

    replay_transition_times = []
    replay_target_transition_times = []
    replay_restore_transition_times = []
    replay_target_couplings = []
    replay_restored_couplings = []
    tracked_times, originals = _start_stage_tracking(fgf1v1_cls)
    try:
        for _ in range(_COLD_REPLAY_PROBE_REPLAY_ROUND_TRIPS):
            start = time.perf_counter()
            replay_model.change_para(flux=replay_flux)
            replay_target_transition_times.append(time.perf_counter() - start)
            replay_transition_times.append(replay_target_transition_times[-1])
            replay_target_couplings.append(
                float(replay_model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False))
            )

            start = time.perf_counter()
            replay_model.change_para(flux=original_flux)
            replay_restore_transition_times.append(time.perf_counter() - start)
            replay_transition_times.append(replay_restore_transition_times[-1])
            replay_restored_couplings.append(
                float(replay_model.get_qq_ecouple(method=_COUPLING_METHOD, is_print=False))
            )
    finally:
        _restore_stage_tracking(originals)

    cold_stage_means = _normalize_stage_tracking(tracked_times=cold_stage_totals, divisor=_COLD_REPLAY_PROBE_COLD_SAMPLES)
    replay_stage_means = _normalize_stage_tracking(
        tracked_times=tracked_times,
        divisor=_COLD_REPLAY_PROBE_REPLAY_ROUND_TRIPS * 2,
    )
    cold_target_coupling = statistics.fmean(cold_target_couplings)
    replay_target_coupling = statistics.fmean(replay_target_couplings)
    replay_restored_coupling = statistics.fmean(replay_restored_couplings)

    return {
        'baseline_coupler_flux': baseline_coupler_flux,
        'target_coupler_flux': target_coupler_flux,
        'cold_samples': _COLD_REPLAY_PROBE_COLD_SAMPLES,
        'replay_round_trips': _COLD_REPLAY_PROBE_REPLAY_ROUND_TRIPS,
        'replay_transition_count': _COLD_REPLAY_PROBE_REPLAY_ROUND_TRIPS * 2,
        'baseline_coupling_ghz': baseline_coupling,
        'seeded_target_coupling_ghz': seeded_target_coupling,
        'cold_constructor_seconds': statistics.fmean(cold_constructor_times),
        'cold_target_transition_seconds': statistics.fmean(cold_target_transition_times),
        'cold_change_para_seconds': cold_stage_means['change_para_seconds'],
        'cold_refresh_basic_metrics_seconds': cold_stage_means['refresh_basic_metrics_seconds'],
        'cold_generate_hamiltonian_seconds': cold_stage_means['generate_hamiltonian_seconds'],
        'cold_get_qq_ecouple_seconds': cold_stage_means['get_qq_ecouple_seconds'],
        'cold_restore_exact_template_check_seconds': (
            cold_stage_means['restore_exact_template_check_seconds']
        ),
        'cold_restore_exact_template_seconds': cold_stage_means['restore_exact_template_seconds'],
        'cold_target_coupling_ghz': cold_target_coupling,
        'replay_transition_seconds': statistics.fmean(replay_transition_times),
        'replay_target_transition_seconds': statistics.fmean(replay_target_transition_times),
        'replay_restore_transition_seconds': statistics.fmean(replay_restore_transition_times),
        'replay_change_para_seconds': replay_stage_means['change_para_seconds'],
        'replay_refresh_basic_metrics_seconds': replay_stage_means['refresh_basic_metrics_seconds'],
        'replay_generate_hamiltonian_seconds': replay_stage_means['generate_hamiltonian_seconds'],
        'replay_get_qq_ecouple_seconds': replay_stage_means['get_qq_ecouple_seconds'],
        'replay_restore_exact_template_check_seconds': (
            replay_stage_means['restore_exact_template_check_seconds']
        ),
        'replay_restore_exact_template_seconds': replay_stage_means['restore_exact_template_seconds'],
        'replay_target_coupling_ghz': replay_target_coupling,
        'replay_restored_coupling_ghz': replay_restored_coupling,
        'replay_target_matches_cold': bool(np.isclose(replay_target_coupling, cold_target_coupling)),
        'replay_restored_matches_baseline': bool(np.isclose(replay_restored_coupling, baseline_coupling)),
    }


def benchmark_fgf1v1_coupling_flux_sweep_workflow(
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
    from pysuqu.qubit.multi import FGF1V1Coupling
    from pysuqu.qubit.sweeps import sweep_multi_qubit_coupling_strength_vs_flux

    for _ in range(warmups):
        for _ in range(iterations):
            _run_coupling_flux_sweep_workload(
                FGF1V1Coupling,
                sweep_multi_qubit_coupling_strength_vs_flux,
                clear_exact_solve_cache=True,
            )

    sample_seconds = []
    workload_signature = None
    for _ in range(samples):
        sample_signature = None
        start = time.perf_counter()
        for _ in range(iterations):
            sample_signature = _run_coupling_flux_sweep_workload(
                FGF1V1Coupling,
                sweep_multi_qubit_coupling_strength_vs_flux,
                clear_exact_solve_cache=True,
            )
        elapsed_seconds = (time.perf_counter() - start) / iterations
        sample_seconds.append(elapsed_seconds)
        workload_signature = sample_signature

    mean_seconds = statistics.fmean(sample_seconds)
    median_seconds = statistics.median(sample_seconds)
    stdev_seconds = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0
    warm_path_split = _profile_warm_path(
        FGF1V1Coupling,
        sweep_multi_qubit_coupling_strength_vs_flux,
    )
    cold_constructor_probe = _measure_cold_constructor_probe(FGF1V1Coupling)
    cold_replay_probe = _measure_cold_replay_probe(FGF1V1Coupling)
    cache_isolation_check = _measure_cache_isolation_drift(
        FGF1V1Coupling,
        sweep_multi_qubit_coupling_strength_vs_flux,
    )
    hotspot_shortlist = _build_hotspot_shortlist(
        warm_path_split,
        cache_isolation_check,
        cold_constructor_probe,
        cold_replay_probe,
    )

    return {
        'benchmark': 'fgf1v1_coupling_flux_sweep_workflow',
        'workflow': (
            'pysuqu.qubit.multi.FGF1V1Coupling construction + '
            'pysuqu.qubit.sweeps.sweep_multi_qubit_coupling_strength_vs_flux'
        ),
        'backend': backend,
        'samples': samples,
        'warmups': warmups,
        'iterations_per_sample': iterations,
        'flux_point_count': len(_COUPLER_FLUX_POINTS),
        'coupler_flux_points': list(_COUPLER_FLUX_POINTS),
        'method': _COUPLING_METHOD,
        'sample_seconds': sample_seconds,
        'mean_seconds': mean_seconds,
        'median_seconds': median_seconds,
        'min_seconds': min(sample_seconds),
        'max_seconds': max(sample_seconds),
        'stdev_seconds': stdev_seconds,
        'workload_checksum': workload_signature['checksum'],
        'workload_signature': workload_signature,
        'warm_path_split': warm_path_split,
        'cold_constructor_probe': cold_constructor_probe,
        'cold_replay_probe': cold_replay_probe,
        'cache_isolation_check': cache_isolation_check,
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
    cold_constructor_probe = result['cold_constructor_probe']
    cold_replay_probe = result['cold_replay_probe']
    cache_isolation_check = result['cache_isolation_check']
    coupling_values = ', '.join(f'{value:.12f}' for value in signature['coupling_values_ghz'])
    cold_values = ', '.join(f'{value:.12f}' for value in cache_isolation_check['cold_coupling_values_ghz'])
    warmed_values = ', '.join(
        f'{value:.12f}'
        for value in cache_isolation_check['warmed_coupling_values_ghz']
    )
    restored_signature = signature['restored_signature']
    restored_flags = ', '.join(
        f'{key}={value}'
        for key, value in restored_signature.items()
    )
    return '\n'.join(
        [
            f"benchmark: {result['benchmark']}",
            f"workflow: {result['workflow']}",
            f"backend: {result['backend']}",
            (
                'config: '
                f"samples={result['samples']}, warmups={result['warmups']}, "
                f"iterations={result['iterations_per_sample']}, "
                f"flux_point_count={result['flux_point_count']}, method={result['method']}"
            ),
            f"mean_seconds: {result['mean_seconds']:.6f}",
            f"median_seconds: {result['median_seconds']:.6f}",
            f"min_seconds: {result['min_seconds']:.6f}",
            f"max_seconds: {result['max_seconds']:.6f}",
            f"stdev_seconds: {result['stdev_seconds']:.6f}",
            f"workload_checksum: {result['workload_checksum']:.12f}",
            (
                'workload_signature: '
                f"baseline_qubit1_f01_ghz={signature['baseline_qubit1_f01_ghz']:.12f}, "
                f"baseline_qubit2_f01_ghz={signature['baseline_qubit2_f01_ghz']:.12f}, "
                f"baseline_coupler_f01_ghz={signature['baseline_coupler_f01_ghz']:.12f}, "
                f"min_coupling_ghz={signature['min_coupling_ghz']:.12f}, "
                f"max_coupling_ghz={signature['max_coupling_ghz']:.12f}, "
                f"turnoff_coupler_flux={signature['turnoff_coupler_flux']:.12f}, "
                f"turnoff_coupling_ghz={signature['turnoff_coupling_ghz']:.12f}, "
                f"restored_coupler_flux={signature['restored_coupler_flux']:.12f}, "
                f"restored_flux_matches_original={signature['restored_flux_matches_original']}"
            ),
            f"warm_constructor_seconds: {warm_path_split['constructor_seconds']:.6f}",
            f"warm_sweep_seconds: {warm_path_split['sweep_seconds']:.6f}",
            f"warm_change_para_seconds: {warm_path_split['change_para_seconds']:.6f}",
            (
                'warm_refresh_basic_metrics_seconds: '
                f"{warm_path_split['refresh_basic_metrics_seconds']:.6f}"
            ),
            (
                'warm_generate_hamiltonian_seconds: '
                f"{warm_path_split['generate_hamiltonian_seconds']:.6f}"
            ),
            f"warm_get_qq_ecouple_seconds: {warm_path_split['get_qq_ecouple_seconds']:.6f}",
            (
                'warm_restore_exact_template_check_seconds: '
                f"{warm_path_split['restore_exact_template_check_seconds']:.6f}"
            ),
            (
                'warm_restore_exact_template_seconds: '
                f"{warm_path_split['restore_exact_template_seconds']:.6f}"
            ),
            (
                'cold_constructor_probe: '
                f"samples={cold_constructor_probe['samples']}, "
                f"constructor_seconds={cold_constructor_probe['constructor_seconds']:.6f}, "
                f"parameterized_init_seconds={cold_constructor_probe['parameterized_init_seconds']:.6f}, "
                f"generate_ematrix_seconds={cold_constructor_probe['generate_ematrix_seconds']:.6f}, "
                f"update_ej_seconds={cold_constructor_probe['update_ej_seconds']:.6f}, "
                f"restore_exact_template_check_seconds="
                f"{cold_constructor_probe['restore_exact_template_check_seconds']:.6f}, "
                f"generate_hamiltonian_seconds={cold_constructor_probe['generate_hamiltonian_seconds']:.6f}, "
                f"refresh_basic_metrics_seconds={cold_constructor_probe['refresh_basic_metrics_seconds']:.6f}, "
                f"set_solver_result_seconds={cold_constructor_probe['set_solver_result_seconds']:.6f}, "
                f"store_exact_template_seconds={cold_constructor_probe['store_exact_template_seconds']:.6f}, "
                f"parameterized_init_other_seconds={cold_constructor_probe['parameterized_init_other_seconds']:.6f}, "
                f"fgf1v1_init_glue_seconds={cold_constructor_probe['fgf1v1_init_glue_seconds']:.6f}, "
                f"constructor_other_seconds={cold_constructor_probe['constructor_other_seconds']:.6f}"
            ),
            (
                'cold_probe: '
                f"baseline_coupler_flux={cold_replay_probe['baseline_coupler_flux']:.12f}, "
                f"target_coupler_flux={cold_replay_probe['target_coupler_flux']:.12f}, "
                f"cold_constructor_seconds={cold_replay_probe['cold_constructor_seconds']:.6f}, "
                f"cold_target_transition_seconds={cold_replay_probe['cold_target_transition_seconds']:.6f}, "
                f"replay_transition_seconds={cold_replay_probe['replay_transition_seconds']:.6f}"
            ),
            (
                'cold_probe_substeps: '
                f"cold_change_para_seconds={cold_replay_probe['cold_change_para_seconds']:.6f}, "
                f"cold_refresh_basic_metrics_seconds={cold_replay_probe['cold_refresh_basic_metrics_seconds']:.6f}, "
                f"cold_generate_hamiltonian_seconds={cold_replay_probe['cold_generate_hamiltonian_seconds']:.6f}, "
                f"cold_get_qq_ecouple_seconds={cold_replay_probe['cold_get_qq_ecouple_seconds']:.6f}"
            ),
            (
                'replay_probe_substeps: '
                f"replay_change_para_seconds={cold_replay_probe['replay_change_para_seconds']:.6f}, "
                f"replay_refresh_basic_metrics_seconds={cold_replay_probe['replay_refresh_basic_metrics_seconds']:.6f}, "
                f"replay_generate_hamiltonian_seconds={cold_replay_probe['replay_generate_hamiltonian_seconds']:.6f}, "
                f"replay_get_qq_ecouple_seconds={cold_replay_probe['replay_get_qq_ecouple_seconds']:.6f}"
            ),
            (
                'cold_replay_consistency: '
                f"replay_target_matches_cold={cold_replay_probe['replay_target_matches_cold']}, "
                f"replay_restored_matches_baseline={cold_replay_probe['replay_restored_matches_baseline']}"
            ),
            f"coupling_values_ghz: [{coupling_values}]",
            f"cache_isolation_cold_values_ghz: [{cold_values}]",
            f"cache_isolation_warmed_values_ghz: [{warmed_values}]",
            f"cache_isolation_warmed_matches_cold: {cache_isolation_check['warmed_matches_cold']}",
            f"restored_signature: {restored_flags}",
            f"sample_seconds: [{sample_seconds}]",
            f"hotspot: {' '.join(result['hotspot_shortlist'])}",
            f"python_version: {result['python_version']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = benchmark_fgf1v1_coupling_flux_sweep_workflow(
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
