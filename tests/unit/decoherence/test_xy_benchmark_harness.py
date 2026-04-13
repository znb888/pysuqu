import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests.support import install_test_stubs

install_test_stubs()

from benchmarks.decoherence_xy_t1_workflow import benchmark_xy_noise_t1_workflow, main


class XYNoiseBenchmarkHarnessTests(unittest.TestCase):
    def test_benchmark_helper_returns_repeatable_measurement_metadata(self):
        result = benchmark_xy_noise_t1_workflow(
            samples=2,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'xy_noise_t1_workflow')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['warmups'], 0)
        self.assertEqual(result['iterations_per_sample'], 1)
        self.assertEqual(result['noise_points'], 2048)
        self.assertEqual(len(result['sample_seconds']), 2)
        self.assertTrue(all(sample > 0.0 for sample in result['sample_seconds']))
        self.assertGreater(result['mean_seconds'], 0.0)
        self.assertGreater(result['max_seconds'], 0.0)
        self.assertGreater(result['workload_checksum'], 0.0)
        self.assertGreater(result['workload_signature']['t1_seconds'], 0.0)
        self.assertGreater(result['workload_signature']['gamma_up'], 0.0)
        self.assertGreater(result['workload_signature']['gamma_down'], 0.0)

    def test_cli_json_output_reports_same_workflow(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(['--samples', '1', '--warmups', '0', '--iterations', '1', '--use-test-stubs', '--json'])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload['benchmark'], 'xy_noise_t1_workflow')
        self.assertEqual(payload['backend'], 'test_stub')
        self.assertEqual(payload['samples'], 1)
        self.assertEqual(payload['iterations_per_sample'], 1)
        self.assertEqual(payload['noise_points'], 2048)
        self.assertEqual(len(payload['sample_seconds']), 1)
        self.assertGreater(payload['sample_seconds'][0], 0.0)


if __name__ == '__main__':
    unittest.main()
