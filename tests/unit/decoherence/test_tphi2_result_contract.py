import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence import TphiResult
from pysuqu.decoherence.dequbit import ZNoiseDecoherence


class ZNoiseDecoherenceTphi2ContractTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return ZNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True)

    def test_cal_tphi2_returns_tphi_result_for_fit_and_cal_paths(self):
        z_noise = self._construct()
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.95, 0.84, 0.73])
        global_popt = np.array([10.0e-6, 3.0e-6, 1.0, 0.0])
        global_pcov = np.diag([0.25e-12, 0.16e-12, 1.0e-4, 1.0e-4])
        segment_a_popt = np.array([11.0e-6, 4.0e-6, 1.0, 0.0])
        segment_a_pcov = np.diag([0.36e-12, 0.25e-12, 1.0e-4, 1.0e-4])
        segment_b_popt = np.array([12.0e-6, 5.0e-6, 1.0, 0.0])
        segment_b_pcov = np.diag([0.49e-12, 0.36e-12, 1.0e-4, 1.0e-4])

        with (
            patch.object(z_noise, 'cal_dephase', return_value=fit_curve),
            patch(
                'pysuqu.decoherence.dequbit.fit_decay',
                side_effect=[
                    (global_popt, global_pcov),
                    (segment_a_popt, segment_a_pcov),
                    (segment_b_popt, segment_b_pcov),
                ],
            ),
        ):
            fit_result = z_noise.cal_tphi2(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                cut_point=[1e3],
                is_print=False,
                is_plot=False,
            )

        cal_result = z_noise.cal_tphi2(
            method='cal',
            experiment='SpinEcho',
            delay_list=delay_list,
            is_print=False,
            is_plot=False,
        )

        self.assertIsInstance(fit_result, TphiResult)
        self.assertEqual(fit_result.unit, 's')
        self.assertEqual(fit_result.value, global_popt[1])
        self.assertEqual(fit_result.metadata['method'], 'fit')
        self.assertEqual(fit_result.metadata['experiment'], 'Ramsey')
        self.assertEqual(fit_result.metadata['source'], 'z-control')
        self.assertEqual(fit_result.fit_diagnostics['tphi1'], global_popt[0])
        self.assertEqual(fit_result.fit_diagnostics['fit_error'], np.sqrt(global_pcov[1][1]))
        self.assertEqual(len(fit_result.fit_diagnostics['segments']), 2)

        self.assertIsInstance(cal_result, TphiResult)
        self.assertEqual(cal_result.unit, 's')
        self.assertEqual(cal_result.metadata['method'], 'cal')
        self.assertEqual(cal_result.metadata['experiment'], 'SpinEcho')
        self.assertEqual(cal_result.metadata['source'], 'z-control')
        self.assertEqual(cal_result.fit_diagnostics['segments'], {})
        self.assertGreater(cal_result.value, 0.0)

if __name__ == '__main__':
    unittest.main()
