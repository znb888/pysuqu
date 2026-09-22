import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pysuqu.funclib.noisemodel import (
    compare_psd,
    compare_psd_groups,
    plot_psd_groups,
    process_psd,
)


class ProcessPsdTests(unittest.TestCase):
    def test_xy_mode_combines_quadratures_over_bandwidth(self):
        frequency = np.array([1e3, 2e3, 4e3])
        bandwidth = np.array([10.0, 20.0, 40.0])

        freq, values = process_psd(
            frequency,
            bandwidth,
            "xy_stddev2_over_bandwidth",
            xstddev=np.array([1.0, 2.0, 3.0]),
            ystddev=np.array([0.0, 2.0, 4.0]),
        )

        np.testing.assert_array_equal(freq, frequency)
        np.testing.assert_allclose(values, [0.1, 0.4, 0.625])

    def test_mode_specific_fields_and_lengths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "Unknown PSD mode"):
            process_psd([1.0], [1.0], "not_a_mode", rpwr=[1.0])
        with self.assertRaisesRegex(ValueError, "requires xstddev and ystddev"):
            process_psd([1.0, 2.0], [1.0, 1.0], "xy_stddev2_over_bandwidth")
        with self.assertRaisesRegex(ValueError, "same length"):
            process_psd([1.0, 2.0], [1.0], "rpwr_over_bandwidth", rpwr=[1.0, 2.0])

    def test_rpwr_and_rstddev_modes(self):
        frequency = np.array([1e3, 2e3])
        bandwidth = np.array([10.0, 20.0])

        _, values = process_psd(
            frequency, bandwidth, "rpwr_over_bandwidth", rpwr=[2.0, 6.0]
        )
        np.testing.assert_allclose(values, [0.2, 0.3])

        _, raw = process_psd(frequency, bandwidth, "rpwr_raw", rpwr=[2.0, 6.0])
        np.testing.assert_allclose(raw, [2.0, 6.0])

        _, r_values = process_psd(
            frequency, bandwidth, "rstddev2_over_bandwidth", rstddev=[2.0, 4.0]
        )
        np.testing.assert_allclose(r_values, [0.4, 0.8])

    def test_compare_psd_reports_relative_error_statistics(self):
        frequency = np.array([0.0, 1e3, 2e3, 3e3])
        psd_a = np.array([1.0, 1.0, 2.0, 4.0])
        psd_b = np.array([1.0, 1.1, 2.0, 4.4])

        comparison = compare_psd(frequency, psd_a, psd_b)

        self.assertEqual(comparison["all"]["points"], 4)
        self.assertEqual(comparison["nonzero_frequency"]["points"], 3)
        # Errors are normalized by the larger of the two traces: 0.1/1.1 and 0.4/4.4.
        self.assertAlmostEqual(
            comparison["nonzero_frequency"]["max_relative_error"], 1.0 / 11.0
        )
        self.assertAlmostEqual(
            comparison["nonzero_frequency"]["median_relative_error"], 1.0 / 11.0
        )

    def test_compare_psd_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            compare_psd([1.0, 2.0], [1.0], [1.0, 2.0])


class NoiseModelPlottingTests(unittest.TestCase):
    def setUp(self):
        frequency = np.array([100.0, 10.0, 1000.0])
        self.data = {
            "000/dev/sample": {
                "group_id": "000",
                "label": "Q04 P15",
                "freq": frequency,
                "psd": np.array([2.0, 4.0, 1.0]),
                "psd_by_mode": {
                    "xy_stddev2_over_bandwidth": np.array([2.0, 4.0, 1.0]),
                    "rpwr_over_bandwidth": np.array([1.0, 2.0, 0.5]),
                },
            },
            "001/dev/sample": {
                "group_id": "001",
                "label": "Q04 R4",
                "freq": frequency,
                "psd": np.array([3.0, 5.0, 2.0]),
                "psd_by_mode": {
                    "xy_stddev2_over_bandwidth": np.array([3.0, 5.0, 2.0]),
                    "rpwr_over_bandwidth": np.array([1.5, 2.5, 1.0]),
                },
            },
        }

    def tearDown(self):
        plt.close("all")

    def test_plot_psd_groups_selects_groups_and_uses_labels(self):
        _, axis = plot_psd_groups(self.data, group_ids=["001"])

        self.assertEqual(len(axis.lines), 1)
        self.assertEqual(axis.lines[0].get_label(), "Q04 R4")
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [10.0, 100.0, 1000.0])

    def test_plot_psd_groups_can_use_a_mode_from_psd_by_mode(self):
        _, axis = plot_psd_groups(
            self.data, group_ids=["000", "001"], psd_key="rpwr_over_bandwidth"
        )

        self.assertEqual(len(axis.lines), 2)
        self.assertEqual(
            {line.get_label() for line in axis.lines}, {"Q04 P15", "Q04 R4"}
        )

    def test_plot_psd_groups_accepts_custom_group_labels(self):
        _, axis = plot_psd_groups(
            self.data, group_labels=["control", "experiment"]
        )

        self.assertEqual(
            [line.get_label() for line in axis.lines], ["control", "experiment"]
        )

    def test_compare_psd_groups_keeps_one_panel_per_group(self):
        _, axes = compare_psd_groups(
            self.data,
            modes=("xy_stddev2_over_bandwidth", "rpwr_over_bandwidth"),
        )

        self.assertEqual(axes.shape, (2, 1))
        self.assertTrue(all(len(axis.lines) == 2 for axis in axes[:, 0]))

    def test_compare_psd_groups_accepts_a_single_mode_string(self):
        _, axes = compare_psd_groups(self.data, modes="xy_stddev2_over_bandwidth")

        self.assertTrue(all(len(axis.lines) == 1 for axis in axes[:, 0]))

    def test_plot_psd_groups_rejects_unknown_group(self):
        with self.assertRaisesRegex(ValueError, "No requested groups found"):
            plot_psd_groups(self.data, group_ids=["999"])


if __name__ == "__main__":
    unittest.main()
