"""Compare cold and repeated propagation on reproducible synthetic models."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np
import qutip as qt
import scipy
from scipy.sparse import diags

from pysuqu import __version__
from pysuqu.funclib.transmission import SignalTrace
from pysuqu.qubit import (
    DriveTerm, PreparedPropagation, clear_native_plan_cache, native_backend_available,
)


def make_workload(case, dimension, samples, seed):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 3.0, samples)
    diagonal = np.sort(rng.uniform(-0.6, 0.6, dimension))
    coupling = rng.uniform(0.08, 0.16, dimension - 1)
    hamiltonian = qt.Qobj(diags([coupling, diagonal, coupling], [-1, 0, 1], dtype=complex).tocsr())
    initial_states = [qt.basis(dimension, index) for index in range(min(4, dimension))]
    terms, collapse = [], []
    if case == 'gate':
        envelope = rng.uniform(0.12, 0.22) * np.sin(np.pi * times / times[-1]) ** 2
        trace = SignalTrace(times, envelope.astype(complex), (samples - 1) / times[-1],
                            'iq_complex', 'qubit_iq', float(rng.uniform(0.7, 1.1)))
        control = qt.Qobj(diags([np.ones(dimension - 1)] * 2, [-1, 1], dtype=complex).tocsr())
        terms = [DriveTerm(control, trace)]
    elif case == 'lindblad':
        collapse = [np.sqrt(rng.uniform(0.04, 0.09)) * qt.destroy(dimension)]
        initial_states = [qt.ket2dm(qt.basis(dimension, dimension - 1))]
    return hamiltonian, terms, times, collapse, initial_states


def benchmark_propagation(*, case='gate', dimension=8, samples=121, repeats=3, seed=863591):
    if case not in {'gate', 'sparse', 'lindblad'}:
        raise ValueError('case must be gate, sparse, or lindblad')
    if dimension < 2 or samples < 4 or repeats < 1:
        raise ValueError('dimension >= 2, samples >= 4, and repeats >= 1 are required')
    hamiltonian, terms, times, collapse, states = make_workload(case, dimension, samples, seed)
    options = {
        'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 100000,
        'coefficient_order': 1, 'store_states': False, 'store_final_state': True,
        'extra': {'normalize_output': False},
    }
    backends = ['qutip', 'qutip_compiled']
    if native_backend_available():
        backends.append('cpp')
    rows, reference = [], None
    for backend in backends:
        clear_native_plan_cache()
        start = time.perf_counter()
        prepared = PreparedPropagation(hamiltonian, terms, times, c_ops=collapse,
                                       backend=backend, options=options)
        result = prepared.propagate_batch(states)
        cold_seconds = time.perf_counter() - start
        if reference is None:
            reference = result.final_states
        cold_error = max((state - expected).norm()
                         for state, expected in zip(result.final_states, reference))
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = prepared.propagate_batch(states)
            timings.append(time.perf_counter() - start)
        state_error = max(cold_error, max((state - expected).norm()
                          for state, expected in zip(result.final_states, reference)))
        conservation_error = max(abs((state.tr() if state.isoper else state.norm()) - 1.0)
                                 for state in result.final_states)
        rows.append({
            'backend': backend,
            'cold_seconds': cold_seconds,
            'warm_median_seconds': statistics.median(timings),
            'warm_sample_seconds': timings,
            'native_integration_seconds': result.stats.get('integration_seconds'),
            'state_error': float(state_error),
            'norm_or_trace_error': float(conservation_error),
            'integrator': result.stats.get('integrator'),
            'matrix_format': result.stats.get('matrix_format'),
        })
    for row in rows:
        row['warm_speedup_vs_reference'] = rows[0]['warm_median_seconds'] / row['warm_median_seconds']
    return {
        'benchmark': 'prepared_propagation', 'case': case, 'seed': seed,
        'dimension': dimension, 'output_samples': samples, 'batch_size': len(states),
        'repeats': repeats, 'options': options, 'native_available': native_backend_available(),
        'versions': {'python': platform.python_version(), 'pysuqu': __version__,
                     'numpy': np.__version__, 'scipy': scipy.__version__, 'qutip': qt.__version__},
        'results': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['gate', 'sparse', 'lindblad'], default='gate')
    parser.add_argument('--dimension', type=int, default=8)
    parser.add_argument('--samples', type=int, default=121)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=863591)
    parser.add_argument('--max-error', type=float, default=2e-6)
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_error) or args.max_error <= 0:
        parser.error('--max-error must be finite and positive')
    result = benchmark_propagation(case=args.case, dimension=args.dimension, samples=args.samples,
                                   repeats=args.repeats, seed=args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(any(row['state_error'] > args.max_error or row['norm_or_trace_error'] > args.max_error
                   for row in result['results']))


if __name__ == '__main__':
    raise SystemExit(main())
