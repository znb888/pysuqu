import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from tests.support import install_test_stubs

install_test_stubs()

from benchmarks.decoherence_r_read_tphi_workflow import (
    _build_hotspot_shortlist,
    benchmark_r_noise_read_tphi_workflow,
    main,
)


class RNoiseBenchmarkHarnessTests(unittest.TestCase):
    def test_benchmark_helper_returns_repeatable_measurement_metadata(self):
        result = benchmark_r_noise_read_tphi_workflow(
            samples=2,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'r_noise_read_tphi_workflow')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['method'], 'cal')
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
        self.assertEqual(result['workload_signature']['source'], 'readout-cavity')
        self.assertGreater(result['workload_signature']['tphi_seconds'], 0.0)
        self.assertGreater(result['workload_signature']['n_bar'], 0.0)
        self.assertGreater(result['workload_signature']['white_noise_temperature_kelvin'], 0.0)
        self.assertEqual(result['workload_signature']['read_freq_hz'], 6.5e9)
        self.assertEqual(result['workload_signature']['kappa_hz'], 6.0e6)
        self.assertEqual(result['workload_signature']['chi_hz'], 1.0e6)
        self.assertEqual(result['workload_signature']['frequency_min_hz'], 1.0)
        self.assertEqual(result['workload_signature']['frequency_max_hz'], 1.0e6)
        self.assertGreater(result['warm_path_split']['constructor_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['build_qubit_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['build_noise_seconds'], 0.0)
        self.assertGreaterEqual(result['warm_path_split']['build_r_analyzer_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['cal_nbar_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['cal_read_tphi_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['cal_readcavity_psd_seconds'], 0.0)
        self.assertGreater(result['warm_path_split']['cal_read_dephase_seconds'], 0.0)
        self.assertEqual(result['warm_path_split']['fit_decay_seconds'], 0.0)
        self.assertEqual(len(result['hotspot_shortlist']), 3)
        self.assertTrue(all(isinstance(item, str) and item for item in result['hotspot_shortlist']))

    def test_benchmark_helper_supports_fit_path_metadata(self):
        result = benchmark_r_noise_read_tphi_workflow(
            method='fit',
            samples=1,
            warmups=0,
            iterations=1,
            use_test_stubs=True,
        )

        self.assertEqual(result['benchmark'], 'r_noise_read_tphi_workflow')
        self.assertEqual(result['backend'], 'test_stub')
        self.assertEqual(result['method'], 'fit')
        self.assertEqual(
            result['workflow'],
            "pysuqu.decoherence.RNoiseDecoherence constructor + "
            "cal_read_tphi(method='fit', experiment='Ramsey')",
        )
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['iterations_per_sample'], 1)
        self.assertGreater(result['sample_seconds'][0], 0.0)
        self.assertEqual(result['workload_signature']['method'], 'fit')
        self.assertEqual(result['workload_signature']['experiment'], 'Ramsey')
        self.assertEqual(result['workload_signature']['source'], 'readout-cavity')
        self.assertGreater(result['workload_signature']['tphi_seconds'], 0.0)
        self.assertGreater(result['workload_signature']['tphi2_seconds'], 0.0)
        self.assertGreater(result['workload_signature']['fit_error_seconds'], 0.0)
        self.assertGreater(result['workload_signature']['tphi2_fit_error_seconds'], 0.0)
        self.assertEqual(result['workload_signature']['readout_psd_points'], 100)
        self.assertEqual(result['workload_signature']['dephase_points'], 100)
        self.assertGreater(result['workload_signature']['readout_psd_mean'], 0.0)
        self.assertGreaterEqual(result['workload_signature']['dephase_terminal'], 0.0)
        self.assertLessEqual(result['workload_signature']['dephase_terminal'], 1.0)
        self.assertEqual(result['workload_signature']['readout_frequency_min_hz'], 0.01)
        self.assertAlmostEqual(result['workload_signature']['readout_frequency_max_hz'], 12000000.0)
        self.assertGreater(result['warm_path_split']['fit_decay_seconds'], 0.0)
        self.assertEqual(len(result['hotspot_shortlist']), 3)

    def test_cli_json_output_reports_fit_workflow(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    '--method',
                    'fit',
                    '--samples',
                    '1',
                    '--warmups',
                    '0',
                    '--iterations',
                    '1',
                    '--use-test-stubs',
                    '--json',
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload['benchmark'], 'r_noise_read_tphi_workflow')
        self.assertEqual(payload['backend'], 'test_stub')
        self.assertEqual(payload['method'], 'fit')
        self.assertEqual(payload['samples'], 1)
        self.assertEqual(payload['iterations_per_sample'], 1)
        self.assertEqual(payload['noise_points'], 2048)
        self.assertEqual(payload['delay_points'], 100)
        self.assertEqual(len(payload['sample_seconds']), 1)
        self.assertGreater(payload['sample_seconds'][0], 0.0)
        self.assertEqual(payload['workload_signature']['method'], 'fit')
        self.assertEqual(payload['workload_signature']['experiment'], 'Ramsey')
        self.assertEqual(payload['workload_signature']['source'], 'readout-cavity')
        self.assertGreater(payload['warm_path_split']['fit_decay_seconds'], 0.0)

    def test_fit_shortlist_prefers_fit_support_boundary_when_fit_decay_dominates(self):
        shortlist = _build_hotspot_shortlist(
            {
                'constructor_seconds': 0.0012,
                'build_qubit_seconds': 0.0006,
                'build_noise_seconds': 0.0003,
                'build_r_analyzer_seconds': 0.0,
                'cal_nbar_seconds': 0.0,
                'cal_read_tphi_seconds': 0.0065,
                'cal_readcavity_psd_seconds': 0.00004,
                'cal_read_dephase_seconds': 0.00002,
                'fit_decay_seconds': 0.0061,
            },
            method='fit',
        )

        self.assertIn('fit_decay', shortlist[1])
        self.assertIn('pysuqu.funclib.mathlib.py', shortlist[2])

    def test_fit_shortlist_switches_to_analyzer_helper_after_fit_decay_drops(self):
        shortlist = _build_hotspot_shortlist(
            {
                'constructor_seconds': 0.0012,
                'build_qubit_seconds': 0.0006,
                'build_noise_seconds': 0.0003,
                'build_r_analyzer_seconds': 0.0,
                'cal_nbar_seconds': 0.0,
                'cal_read_tphi_seconds': 0.0015,
                'cal_readcavity_psd_seconds': 0.00005,
                'cal_read_dephase_seconds': 0.00002,
                'fit_decay_seconds': 0.00001,
            },
            method='fit',
        )

        self.assertIn('cal_readcavity_psd', shortlist[1])
        self.assertIn('pysuqu.decoherence.analysis.py', shortlist[2])

    def test_fit_shortlist_prefers_refresh_when_constructor_reclaims_hotspot(self):
        shortlist = _build_hotspot_shortlist(
            {
                'constructor_seconds': 0.0012,
                'build_qubit_seconds': 0.0006,
                'build_noise_seconds': 0.0003,
                'build_r_analyzer_seconds': 0.0,
                'cal_nbar_seconds': 0.0,
                'cal_read_tphi_seconds': 0.00008,
                'cal_readcavity_psd_seconds': 0.00002,
                'cal_read_dephase_seconds': 0.00001,
                'fit_decay_seconds': 0.00003,
            },
            method='fit',
        )

        self.assertIn('fit_decay', shortlist[1])
        self.assertIn('refresh round', shortlist[2])


if __name__ == '__main__':
    unittest.main()
