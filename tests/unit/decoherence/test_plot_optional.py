import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import RNoiseDecoherence, ZNoiseDecoherence


class DecoherenceRoundGPlotOptionalTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs():
        freq = np.logspace(0, 6, 64)
        psd = 1e-18 / np.maximum(freq, 1.0) + 1e-20
        return freq, psd

    def _construct(self, cls, **kwargs):
        freq, psd = self._sample_noise_inputs()
        with redirect_stdout(StringIO()):
            return cls(psd_freq=freq, psd_S=psd, is_spectral=True, **kwargs)

    def _normalize(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, dict):
            return {key: self._normalize(item) for key, item in value.items()}

        if hasattr(value, 'value') and hasattr(value, 'unit') and hasattr(value, 'metadata'):
            return {
                'value': self._normalize(value.value),
                'unit': self._normalize(value.unit),
                'metadata': self._normalize(value.metadata),
                'fit_diagnostics': self._normalize(value.fit_diagnostics),
            }

        return value

    @staticmethod
    def _z_fit_side_effects():
        global_popt = np.array([10.0e-6, 3.0e-6, 1.0, 0.0])
        global_pcov = np.diag([0.25e-12, 0.16e-12, 1.0e-4, 1.0e-4])
        segment_a_popt = np.array([11.0e-6, 4.0e-6, 1.0, 0.0])
        segment_a_pcov = np.diag([0.36e-12, 0.25e-12, 1.0e-4, 1.0e-4])
        segment_b_popt = np.array([12.0e-6, 5.0e-6, 1.0, 0.0])
        segment_b_pcov = np.diag([0.49e-12, 0.36e-12, 1.0e-4, 1.0e-4])
        return [
            (global_popt, global_pcov),
            (segment_a_popt, segment_a_pcov),
            (segment_b_popt, segment_b_pcov),
        ]

    def test_cal_tphi2_plotting_is_optional(self):
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.95, 0.84, 0.73])

        z_without_plot = self._construct(ZNoiseDecoherence)
        with (
            patch.object(z_without_plot, 'cal_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', side_effect=self._z_fit_side_effects()),
            patch('pysuqu.decoherence.dequbit.plot_z_tphi2_fit') as plot_helper,
        ):
            no_plot_result = z_without_plot.cal_tphi2(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                cut_point=[1e3],
                is_print=False,
                is_plot=False,
            )

        plot_helper.assert_not_called()

        z_with_plot = self._construct(ZNoiseDecoherence)
        with (
            patch.object(z_with_plot, 'cal_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', side_effect=self._z_fit_side_effects()),
            patch('pysuqu.decoherence.dequbit.plot_z_tphi2_fit') as plot_helper,
        ):
            with_plot_result = z_with_plot.cal_tphi2(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                cut_point=[1e3],
                is_print=False,
                is_plot=True,
            )

        plot_helper.assert_called_once()
        self.assertEqual(self._normalize(with_plot_result), self._normalize(no_plot_result))

    def test_cal_read_tphi_plotting_is_optional(self):
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.97, 0.88, 0.81])
        popt = np.array([4.0e-6, 9.0e-6, 1.0, 0.0])
        pcov = np.diag([0.25e-12, 0.49e-12, 1.0e-4, 1.0e-4])

        r_without_plot = self._construct(RNoiseDecoherence)
        with (
            patch.object(r_without_plot, 'cal_readcavity_psd', return_value=np.ones_like(delay_list)),
            patch.object(r_without_plot, 'cal_read_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', return_value=(popt, pcov)),
            patch('pysuqu.decoherence.dequbit.plot_read_tphi_fit') as plot_helper,
        ):
            no_plot_result = r_without_plot.cal_read_tphi(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                is_print=False,
                is_plot=False,
            )

        plot_helper.assert_not_called()

        r_with_plot = self._construct(RNoiseDecoherence)
        with (
            patch.object(r_with_plot, 'cal_readcavity_psd', return_value=np.ones_like(delay_list)),
            patch.object(r_with_plot, 'cal_read_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', return_value=(popt, pcov)),
            patch('pysuqu.decoherence.dequbit.plot_read_tphi_fit') as plot_helper,
        ):
            with_plot_result = r_with_plot.cal_read_tphi(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                is_print=False,
                is_plot=True,
            )

        plot_helper.assert_called_once()
        self.assertEqual(self._normalize(with_plot_result), self._normalize(no_plot_result))


if __name__ == '__main__':
    unittest.main()
