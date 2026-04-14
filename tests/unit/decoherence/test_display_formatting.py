import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.decoherence.dequbit import XYNoiseDecoherence, ZNoiseDecoherence
from pysuqu.qubit.base import Phi0
from pysuqu.decoherence.formatting import (
    format_bias_current_voltage_report,
    format_xy_current_voltage_report,
)


class DecoherenceRoundHDisplayFormattingTests(unittest.TestCase):
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

        return value

    def test_cal_bias_current_voltage_keeps_display_surface_stable(self):
        phi_fraction = 0.25
        z_noise = self._construct(ZNoiseDecoherence)
        expected_result = z_noise.cal_bias_current_voltage(phi_fraction=phi_fraction, is_print=False)
        expected_lines = (
            f'------- Bias Current/Voltage Calculation (phi_fraction={phi_fraction}) -------',
            f'Bias flux: {expected_result["phi_bias"] / Phi0:.3f} Phi0 = {expected_result["phi_bias"]*1e3:.3f} mPhi0',
            f'Chip end: {expected_result["chip_current_uA"]:.3f} uA, {expected_result["chip_voltage_mV"]:.3f} mV',
            f'Total attenuation: {expected_result["total_attenuation_dB"]:.2f} dB',
            f'Room end: {expected_result["room_current_mA"]:.3f} mA, {expected_result["room_voltage_mV"]:.3f} mV, {expected_result["room_power_dBm"]:.2f} dBm',
        )

        self.assertEqual(
            format_bias_current_voltage_report(phi_fraction=phi_fraction, results=expected_result),
            expected_lines,
        )

        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_result = z_noise.cal_bias_current_voltage(phi_fraction=phi_fraction, is_print=True)

        self.assertEqual(self._normalize(actual_result), self._normalize(expected_result))
        self.assertEqual(tuple(stdout.getvalue().splitlines()), expected_lines)

    def test_cal_xy_current_voltage_keeps_display_surface_stable(self):
        phi_fraction = 0.010 / (4 * np.pi)
        xy_noise = self._construct(XYNoiseDecoherence)
        expected_result = xy_noise.cal_xy_current_voltage(phi_fraction=phi_fraction, is_print=False)
        expected_lines = (
            f'--- XY Control Line Current/Voltage Calculation (phi_fraction={phi_fraction:.6f}) ---',
            f'Bias flux: {expected_result["phi_bias"] / Phi0:.6f} Phi0 = {expected_result["phi_bias"]*1e3:.6f} mPhi0',
            f'Chip end: {expected_result["chip_current_uA"]:.3f} uA, {expected_result["chip_voltage_uV"]:.3f} uV, {expected_result["chip_power_dBm"]:.2f} dBm',
            f'Total attenuation: {expected_result["total_attenuation_dB"]:.2f} dB',
            f'Room end: {expected_result["room_current_mA"]:.6f} mA, {expected_result["room_voltage_mV"]:.3f} mV, {expected_result["room_power_dBm"]:.2f} dBm',
        )

        self.assertEqual(
            format_xy_current_voltage_report(phi_fraction=phi_fraction, results=expected_result),
            expected_lines,
        )

        stdout = StringIO()
        with redirect_stdout(stdout):
            actual_result = xy_noise.cal_xy_current_voltage(phi_fraction=phi_fraction, is_print=True)

        self.assertEqual(self._normalize(actual_result), self._normalize(expected_result))
        self.assertEqual(tuple(stdout.getvalue().splitlines()), expected_lines)


if __name__ == '__main__':
    unittest.main()
