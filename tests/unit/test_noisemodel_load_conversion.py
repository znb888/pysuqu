import unittest

import numpy as np

from pysuqu.funclib.noisemodel import convert_noise_psd_to_load


class NoisePsdLoadConversionTests(unittest.TestCase):
    def test_equal_resistances_leave_every_supported_mode_unchanged(self):
        values = np.array([0.0, 1.5, 3.0])
        modes = (
            "dBm/Hz",
            "W/Hz",
            "V^2/Hz",
            "V/sqrt(Hz)",
            "A^2/Hz",
            "A/sqrt(Hz)",
        )

        for mode in modes:
            with self.subTest(mode=mode):
                result = convert_noise_psd_to_load(values, mode=mode)
                np.testing.assert_allclose(result, values)

    def test_known_resistance_ratios_are_applied_to_each_quantity(self):
        kwargs = {"R_m": 50.0, "R_s": 50.0, "R_l": 100.0}
        cases = {
            "dBm/Hz": (10.0, 10.0 + 10.0 * np.log10(8.0 / 9.0)),
            "W/Hz": (2.0, 2.0 * 8.0 / 9.0),
            "V^2/Hz": (2.0, 2.0 * 16.0 / 9.0),
            "V/sqrt(Hz)": (2.0, 2.0 * 4.0 / 3.0),
            "A^2/Hz": (2.0, 2.0 * 4.0 / 9.0),
            "A/sqrt(Hz)": (2.0, 2.0 * 2.0 / 3.0),
        }

        for mode, (measured, expected) in cases.items():
            with self.subTest(mode=mode):
                result = convert_noise_psd_to_load(measured, mode=mode, **kwargs)
                self.assertAlmostEqual(result, expected)

    def test_scalar_and_array_like_inputs_keep_the_documented_output_shapes(self):
        scalar = convert_noise_psd_to_load(1.0, mode="W/Hz")
        sequence = convert_noise_psd_to_load([1.0, 2.0], mode="W/Hz")
        matrix = convert_noise_psd_to_load(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            mode="W/Hz",
        )

        self.assertIsInstance(scalar, float)
        self.assertIsInstance(sequence, np.ndarray)
        self.assertEqual(sequence.dtype, np.float64)
        self.assertEqual(matrix.shape, (2, 2))

    def test_non_positive_or_non_finite_resistances_are_rejected(self):
        invalid_values = (0.0, -1.0, np.nan, np.inf)

        for name in ("R_m", "R_s", "R_l"):
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    kwargs = {name: value}
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        convert_noise_psd_to_load(1.0, mode="W/Hz", **kwargs)

    def test_negative_linear_values_are_rejected_but_negative_dbm_is_valid(self):
        linear_modes = (
            "W/Hz",
            "V^2/Hz",
            "V/sqrt(Hz)",
            "A^2/Hz",
            "A/sqrt(Hz)",
        )

        for mode in linear_modes:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "cannot be negative"):
                    convert_noise_psd_to_load([-1.0], mode=mode)

        self.assertEqual(convert_noise_psd_to_load(-120.0), -120.0)

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported mode"):
            convert_noise_psd_to_load(1.0, mode="invalid")


if __name__ == "__main__":
    unittest.main()
