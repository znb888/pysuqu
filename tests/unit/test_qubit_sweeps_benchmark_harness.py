import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests.support import install_test_stubs

install_test_stubs()

from benchmarks.qubit_grounded_flux_sweep_workflow import (
    benchmark_grounded_transmon_flux_sweep,
    main,
)


class GroundedTransmonSweepBenchmarkHarnessTests(unittest.TestCase):
    def test_benchmark_helper_returns_repeatable_measurement_metadata(self):
        result = benchmark_grounded_transmon_flux_sweep(
            samples=2,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'grounded_transmon_flux_sweep')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['warmups'], 0)
        self.assertEqual(result['iterations_per_sample'], 1)
        self.assertEqual(result['flux_point_count'], 81)
        self.assertEqual(result['upper_level'], 3)
        self.assertEqual(len(result['sample_seconds']), 2)
        self.assertTrue(all(sample > 0.0 for sample in result['sample_seconds']))
        self.assertGreater(result['mean_seconds'], 0.0)
        self.assertGreater(result['max_seconds'], 0.0)
        self.assertGreater(result['workload_checksum'], 0.0)
        self.assertGreater(result['workload_signature']['baseline_f01_ghz'], 0.0)
        self.assertGreater(result['workload_signature']['max_level_3_ghz'], 0.0)
        self.assertAlmostEqual(result['workload_signature']['restored_flux'], 0.125)

    def test_cli_json_output_reports_same_workflow(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(['--samples', '1', '--warmups', '0', '--iterations', '1', '--use-test-stubs', '--json'])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload['benchmark'], 'grounded_transmon_flux_sweep')
        self.assertEqual(payload['backend'], 'test_stub')
        self.assertEqual(payload['samples'], 1)
        self.assertEqual(payload['iterations_per_sample'], 1)
        self.assertEqual(payload['flux_point_count'], 81)
        self.assertEqual(payload['upper_level'], 3)
        self.assertEqual(len(payload['sample_seconds']), 1)
        self.assertGreater(payload['sample_seconds'][0], 0.0)


if __name__ == '__main__':
    unittest.main()
