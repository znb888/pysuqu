import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import Decoherence, RNoiseDecoherence, XYNoiseDecoherence, ZNoiseDecoherence
from pysuqu.decoherence.electronics import ElectronicNoise


class DecoherenceRoundFPrintIndependenceTests(unittest.TestCase):
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

        if isinstance(value, tuple):
            return tuple(self._normalize(item) for item in value)

        if isinstance(value, list):
            return [self._normalize(item) for item in value]

        if hasattr(value, 'value') and hasattr(value, 'unit') and hasattr(value, 'metadata'):
            return {
                'value': self._normalize(value.value),
                'unit': self._normalize(value.unit),
                'metadata': self._normalize(value.metadata),
                'fit_diagnostics': self._normalize(value.fit_diagnostics),
            }

        return value

    def _noise_snapshot(self, noise):
        return self._normalize(
            {
                'psd_double_in': noise.psd_double_in,
                'psd_double_out': noise.psd_double_out,
                'psd_single_out': noise.psd_single_out,
                'white_noise_in': noise.white_noise_in,
                'white_noise_out': noise.white_noise_out,
                'white_ref_freq_in': noise.white_ref_freq_in,
                'white_ref_freq_out': noise.white_ref_freq_out,
                'white_noise_temperature_in': noise.white_noise_temperature_in,
                'white_noise_temperature_out': noise.white_noise_temperature_out,
                'noise_fitres_in': noise.noise_fitres_in,
                'noise_fitres_out': noise.noise_fitres_out,
            }
        )

    def _assert_report_output(self, output: str, *snippets: str):
        self.assertTrue(output.strip())
        for snippet in snippets:
            self.assertIn(snippet, output)

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

    def test_electronic_noise_constructor_and_refresh_model_respect_is_print(self):
        freq, psd = self._sample_noise_inputs()
        noisy_stdout = StringIO()
        with redirect_stdout(noisy_stdout):
            noise = ElectronicNoise(psd_freq=freq, psd_S=psd, noise_prop='single')

        self._assert_report_output(
            noisy_stdout.getvalue(),
            'Electronic Noise Summary',
            'Input white noise PSD:',
            'Output white noise temperature:',
            'Total attenuation:',
        )

        quiet_stdout = StringIO()
        with redirect_stdout(quiet_stdout):
            quiet_noise = ElectronicNoise(psd_freq=freq, psd_S=psd, noise_prop='single', is_print=False)

        self.assertEqual(quiet_stdout.getvalue(), '')
        self.assertEqual(self._noise_snapshot(noise), self._noise_snapshot(quiet_noise))

    def test_decoherence_constructor_and_refresh_model_remember_class_is_print(self):
        freq, psd = self._sample_noise_inputs()

        quiet_stdout = StringIO()
        with redirect_stdout(quiet_stdout):
            quiet_model = Decoherence(
                psd_freq=freq,
                psd_S=psd,
                couple_term=1.0,
                couple_type='z',
                is_print=False,
            )

        self.assertEqual(quiet_stdout.getvalue(), '')

        noisy_stdout = StringIO()
        with redirect_stdout(noisy_stdout):
            noisy_model = Decoherence(
                psd_freq=freq,
                psd_S=psd,
                couple_term=1.0,
                couple_type='z',
                is_print=True,
            )

        self._assert_report_output(
            noisy_stdout.getvalue(),
            'Qubit F01:',
            'Electronic Noise Summary',
        )

        quiet_refresh_stdout = StringIO()
        with redirect_stdout(quiet_refresh_stdout):
            quiet_model.refresh_model()
        self.assertEqual(quiet_refresh_stdout.getvalue(), '')

        noisy_refresh_stdout = StringIO()
        with redirect_stdout(noisy_refresh_stdout):
            noisy_model.refresh_model()
        self._assert_report_output(
            noisy_refresh_stdout.getvalue(),
            'Qubit F01:',
            'Electronic Noise Summary',
        )

    def test_r_constructor_respects_class_is_print_for_auto_nbar(self):
        freq, psd = self._sample_noise_inputs()

        quiet_stdout = StringIO()
        with redirect_stdout(quiet_stdout):
            RNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, is_print=False)
        self.assertEqual(quiet_stdout.getvalue(), '')

        noisy_stdout = StringIO()
        with redirect_stdout(noisy_stdout):
            RNoiseDecoherence(psd_freq=freq, psd_S=psd, is_spectral=True, is_print=True)

        self._assert_report_output(
            noisy_stdout.getvalue(),
            'Qubit F01:',
            'Electronic Noise Summary',
            'Readout Thermal Population',
            'n_bar',
        )

    def test_electronic_noise_refresh_model_does_not_depend_on_is_print(self):
        noise = self._construct(ElectronicNoise)

        quiet_stdout = StringIO()
        with redirect_stdout(quiet_stdout):
            noise.refresh_model(is_print=False)
        expected_snapshot = self._noise_snapshot(noise)
        self.assertEqual(quiet_stdout.getvalue(), '')

        noisy_stdout = StringIO()
        with redirect_stdout(noisy_stdout):
            result = noise.refresh_model(is_print=True)

        self._assert_report_output(
            noisy_stdout.getvalue(),
            'Electronic Noise Summary',
            'Input white noise PSD:',
            'Output white noise temperature:',
            'Total attenuation:',
        )
        self.assertTrue(result)
        self.assertEqual(self._noise_snapshot(noise), expected_snapshot)

    def test_z_compute_paths_do_not_depend_on_is_print(self):
        z_tphi1_false = self._construct(ZNoiseDecoherence)
        z_tphi1_true = self._construct(ZNoiseDecoherence)

        expected_tphi1 = z_tphi1_false.cal_tphi1(is_print=False)
        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_tphi1 = z_tphi1_true.cal_tphi1(is_print=True)
        self._assert_report_output(stdout.getvalue(), 'Z Noise Tphi1', 'Tphi1:')
        self.assertEqual(actual_tphi1, expected_tphi1)

        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.95, 0.84, 0.73])
        z_fit_false = self._construct(ZNoiseDecoherence)
        z_fit_true = self._construct(ZNoiseDecoherence)

        with (
            patch.object(z_fit_false, 'cal_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', side_effect=self._z_fit_side_effects()),
        ):
            expected_fit = z_fit_false.cal_tphi2(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                cut_point=[1e3],
                is_print=False,
                is_plot=False,
            )

        with (
            patch.object(z_fit_true, 'cal_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', side_effect=self._z_fit_side_effects()),
        ):
            stdout = StringIO()
            with redirect_stdout(stdout):
                actual_fit = z_fit_true.cal_tphi2(
                    method='fit',
                    experiment='Ramsey',
                    delay_list=delay_list,
                    cut_point=[1e3],
                    is_print=True,
                    is_plot=False,
                )

        self._assert_report_output(stdout.getvalue(), 'Z Noise Tphi2 (fit, Ramsey)', 'Tphi2:')
        self.assertEqual(self._normalize(actual_fit), self._normalize(expected_fit))

        z_cal_false = self._construct(ZNoiseDecoherence)
        z_cal_true = self._construct(ZNoiseDecoherence)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected_cal = z_cal_false.cal_tphi2(
                method='cal',
                experiment='SpinEcho',
                delay_list=delay_list,
                cut_point=[1e3],
                is_print=False,
                is_plot=False,
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stdout = StringIO()
            with redirect_stdout(stdout):
                actual_cal = z_cal_true.cal_tphi2(
                    method='cal',
                    experiment='SpinEcho',
                    delay_list=delay_list,
                    cut_point=[1e3],
                    is_print=True,
                    is_plot=False,
                )

        self._assert_report_output(stdout.getvalue(), 'Z Noise Tphi2 (cal, SpinEcho)', 'Tphi2:')
        self.assertEqual(self._normalize(actual_cal), self._normalize(expected_cal))

    def test_xy_compute_paths_do_not_depend_on_is_print(self):
        xy_t1_false = self._construct(XYNoiseDecoherence)
        xy_t1_true = self._construct(XYNoiseDecoherence)

        expected_t1 = xy_t1_false.cal_t1(is_print=False)
        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_t1 = xy_t1_true.cal_t1(is_print=True)
        self._assert_report_output(stdout.getvalue(), 'XY Noise T1', 'T1:')
        self.assertEqual(self._normalize(actual_t1), self._normalize(expected_t1))

        xy_thermal_false = self._construct(XYNoiseDecoherence)
        xy_thermal_true = self._construct(XYNoiseDecoherence)

        expected_thermal = xy_thermal_false.cal_thermal_exitation(is_print=False)
        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_thermal = xy_thermal_true.cal_thermal_exitation(is_print=True)
        self._assert_report_output(
            stdout.getvalue(),
            'XY Noise T1',
            'XY Thermal Excitation',
            'Thermal excitation probability:',
        )
        self.assertEqual(self._normalize(actual_thermal), self._normalize(expected_thermal))

    def test_r_compute_paths_do_not_depend_on_is_print(self):
        r_nbar_false = self._construct(RNoiseDecoherence)
        r_nbar_true = self._construct(RNoiseDecoherence)

        expected_nbar = r_nbar_false.cal_nbar(is_print=False)
        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_nbar = r_nbar_true.cal_nbar(is_print=True)
        self._assert_report_output(stdout.getvalue(), 'Readout Thermal Population', 'n_bar')
        self.assertEqual(actual_nbar, expected_nbar)

        delay_list = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        fit_curve = np.array([0.97, 0.88, 0.81])
        popt = np.array([4.0e-6, 9.0e-6, 1.0, 0.0])
        pcov = np.diag([0.25e-12, 0.49e-12, 1.0e-4, 1.0e-4])

        r_cal_false = self._construct(RNoiseDecoherence)
        r_cal_true = self._construct(RNoiseDecoherence)

        expected_cal = r_cal_false.cal_read_tphi(
            method='cal',
            experiment='SpinEcho',
            delay_list=delay_list,
            is_print=False,
            is_plot=False,
        )
        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_cal = r_cal_true.cal_read_tphi(
                method='cal',
                experiment='SpinEcho',
                delay_list=delay_list,
                is_print=True,
                is_plot=False,
            )
        self._assert_report_output(stdout.getvalue(), 'Readout-Induced Tphi (cal, SpinEcho)', 'Readout Tphi:')
        self.assertEqual(self._normalize(actual_cal), self._normalize(expected_cal))

        r_fit_false = self._construct(RNoiseDecoherence)
        r_fit_true = self._construct(RNoiseDecoherence)

        with (
            patch.object(r_fit_false, 'cal_readcavity_psd', return_value=np.ones_like(delay_list)),
            patch.object(r_fit_false, 'cal_read_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', return_value=(popt, pcov)),
        ):
            expected_fit = r_fit_false.cal_read_tphi(
                method='fit',
                experiment='Ramsey',
                delay_list=delay_list,
                is_print=False,
                is_plot=False,
            )

        with (
            patch.object(r_fit_true, 'cal_readcavity_psd', return_value=np.ones_like(delay_list)),
            patch.object(r_fit_true, 'cal_read_dephase', return_value=fit_curve),
            patch('pysuqu.decoherence.dequbit.fit_decay', return_value=(popt, pcov)),
        ):
            stdout = StringIO()
            with redirect_stdout(stdout):
                actual_fit = r_fit_true.cal_read_tphi(
                    method='fit',
                    experiment='Ramsey',
                    delay_list=delay_list,
                    is_print=True,
                    is_plot=False,
                )

        self._assert_report_output(
            stdout.getvalue(),
            'Readout-Induced Tphi (fit, Ramsey)',
            'Readout Tphi:',
            'Background Tphi2:',
        )
        self.assertEqual(self._normalize(actual_fit), self._normalize(expected_fit))


if __name__ == '__main__':
    unittest.main()

