import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests.support import install_test_stubs

install_test_stubs()

from benchmarks.decoherence_z_tphi2_workflow import benchmark_z_noise_tphi2_workflow, main


class ZNoiseBenchmarkHarnessTests(unittest.TestCase):
    def test_benchmark_helper_returns_repeatable_measurement_metadata(self):
        result = benchmark_z_noise_tphi2_workflow(
            samples=2,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'z_noise_tphi2_workflow')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['warmups'], 0)
        self.assertEqual(result['iterations_per_sample'], 1)
        self.assertEqual(result['noise_points'], 2048)
        self.assertEqual(result['delay_points'], 100)
        self.assertEqual(len(result['sample_seconds']), 2)
        self.assertTrue(all(sample > 0.0 for sample in result['sample_seconds']))
        self.assertGreater(result['mean_seconds'], 0.0)
        self.assertGreater(result['max_seconds'], 0.0)
        self.assertGreater(result['workload_checksum'], 0.0)
        self.assertEqual(result['workload_signature']['method'], 'cal')
        self.assertEqual(result['workload_signature']['experiment'], 'Ramsey')
        self.assertGreater(result['workload_signature']['tphi2_seconds'], 0.0)
        self.assertGreater(result['workload_signature']['white_noise_temperature_kelvin'], 0.0)
        self.assertGreater(result['workload_signature']['flux_sensitivity_rad_per_s_per_wb'], 0.0)
        self.assertEqual(result['workload_signature']['frequency_min_hz'], 10000.0)
        self.assertEqual(result['workload_signature']['frequency_max_hz'], 100000000.0)
        self.assertGreater(result['warm_path_split']['constructor_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['build_qubit_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['build_noise_seconds'], 0.0)
        self.assertGreaterEqual(result['warm_path_split']['update_sensitivity_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['cal_tphi2_seconds'], 0.0)
        self.assertEqual(len(result['hotspot_shortlist']), 3)
        self.assertTrue(all(isinstance(item, str) and item for item in result['hotspot_shortlist']))

    def test_cli_json_output_reports_same_workflow(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ['--samples', '1', '--warmups', '0', '--iterations', '1', '--use-test-stubs', '--json']
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload['benchmark'], 'z_noise_tphi2_workflow')
        self.assertEqual(payload['backend'], 'test_stub')
        self.assertEqual(payload['samples'], 1)
        self.assertEqual(payload['iterations_per_sample'], 1)
        self.assertEqual(payload['noise_points'], 2048)
        self.assertEqual(payload['delay_points'], 100)
        self.assertEqual(len(payload['sample_seconds']), 1)
        self.assertGreater(payload['sample_seconds'][0], 0.0)
        self.assertEqual(payload['workload_signature']['method'], 'cal')
        self.assertEqual(payload['workload_signature']['experiment'], 'Ramsey')


if __name__ == '__main__':
    unittest.main()
