import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests.support import install_test_stubs

install_test_stubs()

from benchmarks.qubit_multi_coupling_flux_sweep_workflow import (
    benchmark_fgf1v1_coupling_flux_sweep_workflow,
    main,
)


class MultiQubitCouplingSweepBenchmarkHarnessTests(unittest.TestCase):
    def test_benchmark_helper_returns_repeatable_measurement_metadata(self):
        result = benchmark_fgf1v1_coupling_flux_sweep_workflow(
            samples=2,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'fgf1v1_coupling_flux_sweep_workflow')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['warmups'], 0)
        self.assertEqual(result['iterations_per_sample'], 1)
        self.assertEqual(result['flux_point_count'], 4)
        self.assertEqual(result['method'], 'ES')
        self.assertEqual(len(result['sample_seconds']), 2)
        self.assertTrue(all(sample > 0.0 for sample in result['sample_seconds']))
        self.assertGreater(result['mean_seconds'], 0.0)
        self.assertGreater(result['max_seconds'], 0.0)
        self.assertNotEqual(result['workload_checksum'], 0.0)
        self.assertGreater(result['workload_signature']['baseline_qubit1_f01_ghz'], 0.0)
        self.assertGreater(result['workload_signature']['baseline_qubit2_f01_ghz'], 0.0)
        self.assertGreater(result['workload_signature']['baseline_coupler_f01_ghz'], 0.0)
        self.assertEqual(len(result['workload_signature']['coupling_values_ghz']), 4)
        self.assertTrue(result['workload_signature']['restored_flux_matches_original'])
        self.assertAlmostEqual(result['workload_signature']['restored_coupler_flux'], 0.11)
        self.assertEqual(
            set(result['workload_signature']['restored_signature']),
            {
                'qubit1_f01_matches_baseline',
                'qubit2_f01_matches_baseline',
                'coupler_f01_matches_baseline',
                'qr_g_matches_baseline',
                'qq_g_matches_baseline',
                'qc_g_matches_baseline',
                'qq_geff_matches_baseline',
            },
        )
        self.assertTrue(
            all(
                isinstance(flag, bool)
                for flag in result['workload_signature']['restored_signature'].values()
            )
        )
        self.assertTrue(all(result['workload_signature']['restored_signature'].values()))
        self.assertGreater(result['warm_path_split']['constructor_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['sweep_seconds'], 0.0)
        self.assertGreaterEqual(result['warm_path_split']['change_para_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['refresh_basic_metrics_seconds'], 0.0)
        self.assertGreaterEqual(result['warm_path_split']['generate_hamiltonian_seconds'], 0.0)
        self.assertGreaterEqual(result['warm_path_split']['get_qq_ecouple_seconds'], 0.0)
        self.assertEqual(
            set(result['cold_constructor_probe']),
            {
                'samples',
                'constructor_seconds',
                'parameterized_init_seconds',
                'generate_ematrix_seconds',
                'update_ej_seconds',
                'restore_exact_template_check_seconds',
                'generate_hamiltonian_seconds',
                'refresh_basic_metrics_seconds',
                'set_solver_result_seconds',
                'store_exact_template_seconds',
                'parameterized_init_other_seconds',
                'fgf1v1_init_glue_seconds',
                'constructor_other_seconds',
            },
        )
        self.assertGreater(result['cold_constructor_probe']['samples'], 0)
        self.assertGreater(result['cold_constructor_probe']['constructor_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['parameterized_init_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['generate_ematrix_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['update_ej_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['restore_exact_template_check_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['generate_hamiltonian_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['refresh_basic_metrics_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['set_solver_result_seconds'], 0.0)
        self.assertGreater(result['cold_constructor_probe']['store_exact_template_seconds'], 0.0)
        self.assertGreaterEqual(result['cold_constructor_probe']['parameterized_init_other_seconds'], 0.0)
        self.assertGreaterEqual(result['cold_constructor_probe']['fgf1v1_init_glue_seconds'], 0.0)
        self.assertGreaterEqual(result['cold_constructor_probe']['constructor_other_seconds'], 0.0)
        self.assertEqual(
            set(result['cold_replay_probe']),
            {
                'baseline_coupler_flux',
                'target_coupler_flux',
                'cold_samples',
                'replay_round_trips',
                'replay_transition_count',
                'baseline_coupling_ghz',
                'seeded_target_coupling_ghz',
                'cold_constructor_seconds',
                'cold_target_transition_seconds',
                'cold_change_para_seconds',
                'cold_refresh_basic_metrics_seconds',
                'cold_generate_hamiltonian_seconds',
                'cold_get_qq_ecouple_seconds',
                'cold_restore_exact_template_check_seconds',
                'cold_restore_exact_template_seconds',
                'cold_target_coupling_ghz',
                'replay_transition_seconds',
                'replay_target_transition_seconds',
                'replay_restore_transition_seconds',
                'replay_change_para_seconds',
                'replay_refresh_basic_metrics_seconds',
                'replay_generate_hamiltonian_seconds',
                'replay_get_qq_ecouple_seconds',
                'replay_restore_exact_template_check_seconds',
                'replay_restore_exact_template_seconds',
                'replay_target_coupling_ghz',
                'replay_restored_coupling_ghz',
                'replay_target_matches_cold',
                'replay_restored_matches_baseline',
            },
        )
        self.assertAlmostEqual(result['cold_replay_probe']['baseline_coupler_flux'], 0.11)
        self.assertAlmostEqual(result['cold_replay_probe']['target_coupler_flux'], 0.095)
        self.assertGreater(result['cold_replay_probe']['cold_samples'], 0)
        self.assertGreater(result['cold_replay_probe']['replay_round_trips'], 0)
        self.assertGreater(result['cold_replay_probe']['replay_transition_count'], 0)
        self.assertGreater(result['cold_replay_probe']['baseline_coupling_ghz'], 0.0)
        self.assertGreater(result['cold_replay_probe']['cold_constructor_seconds'], 0.0)
        self.assertGreater(result['cold_replay_probe']['cold_target_transition_seconds'], 0.0)
        self.assertGreater(result['cold_replay_probe']['replay_transition_seconds'], 0.0)
        self.assertGreater(result['cold_replay_probe']['replay_target_transition_seconds'], 0.0)
        self.assertGreater(result['cold_replay_probe']['replay_restore_transition_seconds'], 0.0)
        self.assertIsInstance(result['cold_replay_probe']['replay_target_matches_cold'], bool)
        self.assertTrue(result['cold_replay_probe']['replay_target_matches_cold'])
        self.assertIsInstance(result['cold_replay_probe']['replay_restored_matches_baseline'], bool)
        self.assertTrue(result['cold_replay_probe']['replay_restored_matches_baseline'])
        self.assertEqual(
            set(result['cache_isolation_check']),
            {
                'cold_coupling_values_ghz',
                'warmed_coupling_values_ghz',
                'warmed_matches_cold',
            },
        )
        self.assertEqual(len(result['cache_isolation_check']['cold_coupling_values_ghz']), 4)
        self.assertEqual(len(result['cache_isolation_check']['warmed_coupling_values_ghz']), 4)
        self.assertIsInstance(result['cache_isolation_check']['warmed_matches_cold'], bool)
        self.assertTrue(result['cache_isolation_check']['warmed_matches_cold'])
        self.assertEqual(len(result['hotspot_shortlist']), 3)
        self.assertTrue(all(isinstance(item, str) and item for item in result['hotspot_shortlist']))

    def test_cli_json_output_reports_same_workflow(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(['--samples', '1', '--warmups', '0', '--iterations', '1', '--use-test-stubs', '--json'])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload['benchmark'], 'fgf1v1_coupling_flux_sweep_workflow')
        self.assertEqual(payload['backend'], 'test_stub')
        self.assertEqual(payload['samples'], 1)
        self.assertEqual(payload['iterations_per_sample'], 1)
        self.assertEqual(payload['flux_point_count'], 4)
        self.assertEqual(payload['method'], 'ES')
        self.assertEqual(len(payload['sample_seconds']), 1)
        self.assertGreater(payload['sample_seconds'][0], 0.0)
        self.assertGreater(payload['warm_path_split']['constructor_seconds'], 0.0)
        self.assertGreater(payload['warm_path_split']['sweep_seconds'], 0.0)
        self.assertGreater(payload['cold_constructor_probe']['constructor_seconds'], 0.0)
        self.assertGreater(payload['cold_constructor_probe']['parameterized_init_seconds'], 0.0)
        self.assertGreater(payload['cold_constructor_probe']['generate_hamiltonian_seconds'], 0.0)
        self.assertGreater(payload['cold_constructor_probe']['update_ej_seconds'], 0.0)
        self.assertGreater(payload['cold_constructor_probe']['set_solver_result_seconds'], 0.0)
        self.assertGreater(payload['cold_constructor_probe']['store_exact_template_seconds'], 0.0)
        self.assertGreater(payload['cold_replay_probe']['cold_constructor_seconds'], 0.0)
        self.assertGreater(payload['cold_replay_probe']['replay_transition_seconds'], 0.0)
        self.assertTrue(payload['cold_replay_probe']['replay_target_matches_cold'])
        self.assertEqual(len(payload['cache_isolation_check']['cold_coupling_values_ghz']), 4)
        self.assertEqual(len(payload['hotspot_shortlist']), 3)


if __name__ == '__main__':
    unittest.main()
