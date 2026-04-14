import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests.support import install_test_stubs

install_test_stubs()

from benchmarks.qubit_qcrfgr_frequency_probe_workflow import (
    benchmark_qcrfgr_frequency_probe_workflow,
    main,
)


class MultiQubitBenchmarkHarnessTests(unittest.TestCase):
    def test_benchmark_helper_returns_repeatable_measurement_metadata(self):
        result = benchmark_qcrfgr_frequency_probe_workflow(
            samples=2,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'qcrfgr_frequency_probe_workflow')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['warmups'], 0)
        self.assertEqual(result['iterations_per_sample'], 1)
        self.assertEqual(result['probe_point_count'], 4)
        self.assertEqual(result['qubit_index'], 0)
        self.assertEqual(len(result['sample_seconds']), 2)
        self.assertTrue(all(sample > 0.0 for sample in result['sample_seconds']))
        self.assertGreater(result['mean_seconds'], 0.0)
        self.assertGreater(result['max_seconds'], 0.0)
        self.assertGreater(result['workload_checksum'], 0.0)
        self.assertGreater(result['workload_signature']['baseline_qubit_f01_ghz'], 0.0)
        self.assertGreater(result['workload_signature']['baseline_coupler_f01_ghz'], 0.0)
        self.assertGreater(result['workload_signature']['probe_span_mhz'], 0.0)
        self.assertEqual(len(result['workload_signature']['probe_values_ghz']), 4)
        self.assertTrue(result['workload_signature']['restored_flux_matches_original'])
        self.assertAlmostEqual(result['workload_signature']['restored_coupler_flux'], 0.11)
        self.assertGreater(result['warm_path_split']['constructor_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['probe_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['refresh_basic_metrics_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['generate_ematrix_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['restore_exact_template_check_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['restore_exact_template_seconds'], 0.0)
        self.assertEqual(len(result['hotspot_shortlist']), 3)
        self.assertTrue(all(isinstance(item, str) and item for item in result['hotspot_shortlist']))

    def test_cli_json_output_reports_same_workflow(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(['--samples', '1', '--warmups', '0', '--iterations', '1', '--use-test-stubs', '--json'])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload['benchmark'], 'qcrfgr_frequency_probe_workflow')
        self.assertEqual(payload['backend'], 'test_stub')
        self.assertEqual(payload['samples'], 1)
        self.assertEqual(payload['iterations_per_sample'], 1)
        self.assertEqual(payload['probe_point_count'], 4)
        self.assertEqual(payload['qubit_index'], 0)
        self.assertEqual(len(payload['sample_seconds']), 1)
        self.assertGreater(payload['sample_seconds'][0], 0.0)
        self.assertGreater(payload['warm_path_split']['constructor_seconds'], 0.0)
        self.assertEqual(len(payload['hotspot_shortlist']), 3)


if __name__ == '__main__':
    unittest.main()
