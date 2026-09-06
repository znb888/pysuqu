"""Run real numerical checks against an installed wheel without source shadowing."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

import qutip
import pysuqu
from pysuqu import _native
from pysuqu.qubit import native_backend_available


MODULES = (
    'test_propagation_options', 'test_prepared_propagation', 'test_prepared_gates',
    'test_native_propagation', 'test_native_structured_propagation',
    'test_native_open_propagation', 'test_native_krylov_propagation',
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--native', action='store_true')
    mode.add_argument('--python-only', action='store_true')
    parser.add_argument('--version')
    args = parser.parse_args(argv)
    source_package = Path(__file__).resolve().parents[1] / 'pysuqu'
    installed_package = Path(pysuqu.__file__).resolve()
    if source_package in installed_package.parents:
        parser.error('The source checkout is shadowing the installed wheel; run Python with -I.')
    if not getattr(qutip, '__version__', None):
        parser.error('Real QuTiP is required; test stubs are not accepted.')
    if args.version and pysuqu.__version__ != args.version:
        parser.error('Installed version does not match the requested release.')
    if native_backend_available() != args.native:
        parser.error('Installed native availability does not match the wheel under test.')
    if args.native and not all(callable(getattr(_native, name)) for name in _native.__all__):
        parser.error('The native wheel is missing a required propagation kernel.')

    suite = unittest.TestSuite()
    for name in MODULES:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / 'unit' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.native and result.skipped:
        print('Native wheel validation must not skip numerical tests.', file=sys.stderr)
        return 1
    print('Validated pysuqu', pysuqu.__version__, 'with QuTiP', qutip.__version__)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
