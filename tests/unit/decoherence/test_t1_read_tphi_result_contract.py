import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence import T1Result, TphiResult
from pysuqu.decoherence.dequbit import RNoiseDecoherence, XYNoiseDecoherence


class DecoherenceRoundEContractTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, cls, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return cls(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def test_cal_t1_returns_seconds_facing_t1_result(self):
        xy_noise = self._construct(XYNoiseDecoherence)

        result = xy_noise.cal_t1(is_print=False)

        self.assertIsInstance(result, T1Result)
        self.assertEqual(result.unit, 's')
        self.assertEqual(result.value, xy_noise.T1)
        self.assertEqual(result.metadata['method'], 'cal')
        self.assertEqual(result.metadata['source'], 'xy-control')
        self.assertEqual(result.fit_diagnostics['gamma_up'], xy_noise.Gamma_up)
        self.assertEqual(result.fit_diagnostics['gamma_down'], xy_noise.Gamma_down)
        self.assertGreater(result.value, 0.0)

    def test_cal_read_tphi_returns_tphi_result_for_cal_and_fit_paths(self):
        r_noise = self._construct(RNoiseDecoherence)
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.97, 0.88, 0.81])
        popt = np.array([4.0e-6, 9.0e-6, 1.0, 0.0])
        pcov = np.diag([0.25e-12, 0.49e-12, 1.0e-4, 1.0e-4])

        cal_result = r_noise.cal_read_tphi(
            method='cal',
            experiment='SpinEcho',
            delay_list=delay_list,
            is_print=False,
            is_plot=False,
        )
        cal_tphi_rc = cal_result.value

        with (
            patch.object(r_noise, 'cal_readcavity_psd', return_value=np.ones_like(delay_list)),
            patch.object(r_noise, 'cal_read_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', return_value=(popt, pcov)),
        ):
            fit_result = r_noise.cal_read_tphi(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                is_print=False,
                is_plot=False,
            )

        self.assertIsInstance(cal_result, TphiResult)
        self.assertEqual(cal_result.unit, 's')
        self.assertEqual(cal_result.value, cal_tphi_rc)
        self.assertEqual(cal_result.metadata['method'], 'cal')
        self.assertEqual(cal_result.metadata['experiment'], 'SpinEcho')
        self.assertEqual(cal_result.metadata['source'], 'readout-cavity')
        self.assertEqual(cal_result.fit_diagnostics, {})
        self.assertGreater(cal_result.value, 0.0)

        self.assertIsInstance(fit_result, TphiResult)
        self.assertEqual(fit_result.unit, 's')
        self.assertEqual(fit_result.value, popt[0])
        self.assertEqual(fit_result.metadata['method'], 'fit')
        self.assertEqual(fit_result.metadata['experiment'], 'Ramsey')
        self.assertEqual(fit_result.metadata['source'], 'readout-cavity')
        self.assertEqual(fit_result.fit_diagnostics['tphi2'], popt[1])
        self.assertEqual(fit_result.fit_diagnostics['fit_error'], np.sqrt(pcov[0][0]))
        self.assertEqual(fit_result.fit_diagnostics['tphi2_fit_error'], np.sqrt(pcov[1][1]))


if __name__ == '__main__':
    unittest.main()
