import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence import TphiResult
from pysuqu.decoherence.dequbit import ZNoiseDecoherence
from pysuqu.decoherence.plotting import plot_z_tphi2_multipsd_fit


class ZNoiseDecoherenceMultiPsdTests(unittest.TestCase):
    @staticmethod
    def _sample_noise_inputs(scale=1.0):
        frequency = np.logspace(0, 6, 64)
        psd = scale * (1e-18 / np.maximum(frequency, 1.0) + 1e-20)
        return frequency, psd

    def _construct_with_recording_noise_builder(self):
        frequency, psd = self._sample_noise_inputs()
        noise_calls = []

        def noise_builder(**kwargs):
            recorded = dict(kwargs)
            recorded['psd_freq'] = np.array(kwargs['psd_freq'], copy=True)
            recorded['psd_S'] = np.array(kwargs['psd_S'], copy=True)
            noise_calls.append(recorded)
            return SimpleNamespace(
                output_stage=SimpleNamespace(
                    frequency=recorded['psd_freq'],
                    psd_single=recorded['psd_S'],
                    fit_result=SimpleNamespace(
                        fit_diagnostics={'1f_coef': float(recorded['psd_S'][0])}
                    ),
                )
            )

        with redirect_stdout(StringIO()):
            model = ZNoiseDecoherence(
                psd_freq=frequency,
                psd_S=psd,
                is_spectral=True,
                is_print=False,
                noise_builder=noise_builder,
            )
        return model, noise_calls

    def test_multipsd_fit_returns_labeled_results_without_mutating_instance(self):
        model, noise_calls = self._construct_with_recording_noise_builder()
        frequency_a, psd_a = self._sample_noise_inputs(scale=1.0)
        frequency_b = np.logspace(7, 1, 80)
        psd_b = 2.0 * (1e-18 / frequency_b + 1e-20)
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        dephase_a = np.array([0.96, 0.86, 0.76])
        dephase_b = np.array([0.93, 0.78, 0.64])
        popt_a = np.array([10.0e-6, 3.0e-6, 1.0])
        popt_b = np.array([11.0e-6, 2.0e-6, 1.0])
        pcov_a = np.diag([0.25e-12, 0.16e-12, 1.0e-4])
        pcov_b = np.diag([0.36e-12, 0.09e-12, 1.0e-4])
        original_noise = model.noise
        original_frequency = np.array(model.psd_freq, copy=True)
        original_psd = np.array(model.psd_S, copy=True)

        with (
            patch.object(model, 'get_sensitivity_at_idle', return_value=1.23) as sensitivity,
            patch.object(model, 'cal_dephase', side_effect=[dephase_a, dephase_b]),
            patch(
                'pysuqu.decoherence.dequbit.fit_decay',
                side_effect=[(popt_a, pcov_a), (popt_b, pcov_b)],
            ),
            patch(
                'pysuqu.decoherence.dequbit.plot_z_tphi2_multipsd_fit'
            ) as plot_helper,
        ):
            results = model.cal_tphi_multipsd(
                psd_list=[psd_a, psd_b],
                psd_freq_list=[frequency_a, frequency_b],
                labels=['open', 'loopback'],
                delay_list=delay_list,
                idle_freq=4.8e9,
                is_print=False,
                is_plot=True,
            )

        self.assertEqual(list(results), ['open', 'loopback'])
        self.assertTrue(all(isinstance(result, TphiResult) for result in results.values()))
        self.assertEqual(results['open'].value, popt_a[1])
        self.assertEqual(results['loopback'].value, popt_b[1])
        self.assertEqual(results['open'].metadata['label'], 'open')
        self.assertEqual(results['loopback'].metadata['psd_index'], 1)
        sensitivity.assert_called_once_with(4.8e9)

        self.assertEqual(len(noise_calls), 3)
        np.testing.assert_array_equal(noise_calls[1]['psd_freq'], frequency_a)
        np.testing.assert_array_equal(noise_calls[2]['psd_freq'], frequency_b)
        np.testing.assert_array_equal(noise_calls[1]['psd_S'], psd_a)
        np.testing.assert_array_equal(noise_calls[2]['psd_S'], psd_b)
        self.assertFalse(noise_calls[1]['is_print'])
        self.assertFalse(noise_calls[2]['is_print'])

        self.assertIs(model.noise, original_noise)
        np.testing.assert_array_equal(model.psd_freq, original_frequency)
        np.testing.assert_array_equal(model.psd_S, original_psd)
        self.assertFalse(hasattr(model, 'tphi2'))
        plot_helper.assert_called_once()
        self.assertEqual(
            list(plot_helper.call_args.kwargs['series']),
            ['open', 'loopback'],
        )

    def test_multipsd_cal_returns_results_without_drawing_fit_plot(self):
        model, _ = self._construct_with_recording_noise_builder()
        frequency_a, psd_a = self._sample_noise_inputs(scale=1.0)
        frequency_b, psd_b = self._sample_noise_inputs(scale=2.0)

        with (
            patch.object(model, 'get_sensitivity_at_idle', return_value=1.23),
            patch.object(
                model.z_analyzer,
                'calculate_tphi2_cal',
                side_effect=[4.0e-6, 2.0e-6],
            ),
            patch(
                'pysuqu.decoherence.dequbit.plot_z_tphi2_multipsd_fit'
            ) as plot_helper,
        ):
            results = model.cal_tphi_multipsd(
                psd_list=[psd_a, psd_b],
                psd_freq_list=[frequency_a, frequency_b],
                labels=['a', 'b'],
                method='cal',
                experiment='SpinEcho',
                is_print=False,
                is_plot=True,
            )

        self.assertEqual(results['a'].value, 4.0e-6)
        self.assertEqual(results['b'].value, 2.0e-6)
        plot_helper.assert_not_called()

    def test_multipsd_validates_parallel_inputs_and_labels(self):
        model, _ = self._construct_with_recording_noise_builder()
        frequency, psd = self._sample_noise_inputs()

        invalid_calls = (
            ({'psd_list': [], 'psd_freq_list': []}, 'at least one PSD'),
            (
                {'psd_list': [psd, psd], 'psd_freq_list': [frequency]},
                'same number',
            ),
            (
                {
                    'psd_list': [psd, psd],
                    'psd_freq_list': [frequency, frequency],
                    'labels': ['same', 'same'],
                },
                'unique',
            ),
            (
                {'psd_list': [psd[:-1]], 'psd_freq_list': [frequency]},
                'mismatched',
            ),
        )

        for kwargs, message in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model.cal_tphi_multipsd(
                        **kwargs,
                        is_print=False,
                        is_plot=False,
                    )


class ZTphi2MultiPsdPlotTests(unittest.TestCase):
    def test_plot_contains_one_point_and_fit_curve_per_psd(self):
        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        series = {
            'open': {
                'dephase': np.array([0.95, 0.84, 0.73]),
                'fitted_dephase': np.array([0.94, 0.83, 0.72]),
                'tphi2': 3.0e-6,
                'tphi2_fiterror': 0.4e-6,
            },
            'loopback': {
                'dephase': np.array([0.91, 0.74, 0.59]),
                'fitted_dephase': np.array([0.90, 0.73, 0.58]),
                'tphi2': 2.0e-6,
                'tphi2_fiterror': 0.3e-6,
            },
        }

        with patch('pysuqu.decoherence.plotting.plt.show'):
            plot_z_tphi2_multipsd_fit(
                delay_list=delay_list,
                series=series,
                experiment='Ramsey',
                plot_title='Q07 comparison',
            )

        figure = plt.gcf()
        try:
            axes = figure.axes[0]
            self.assertEqual(len(axes.lines), 4)
            self.assertEqual(axes.get_title(), 'Q07 comparison (Ramsey)')
            legend_text = [text.get_text() for text in axes.get_legend().get_texts()]
            self.assertIn('open: Tphi2 = 3.00 +/- 0.40 us', legend_text)
            self.assertIn('loopback: Tphi2 = 2.00 +/- 0.30 us', legend_text)
        finally:
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
