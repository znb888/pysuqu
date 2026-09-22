import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pysuqu.funclib.noisemodel import find_psd_peaks


class NoiseModelPeakDetectionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _trace(frequency, centers):
        x = np.log10(frequency)
        values = np.ones_like(frequency, dtype=float)
        for center, amplitude, width in centers:
            values += amplitude * np.exp(-0.5 * ((x - np.log10(center)) / width) ** 2)
        return values

    def test_single_trace_returns_peak_frequencies_on_unsorted_grid(self):
        frequency = np.array([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5])
        psd = self._trace(frequency, [(1e4, 100.0, 0.035)])

        peaks = find_psd_peaks(
            frequency[::-1],
            psd[::-1],
            prominence=10.0,
            smooth_window=0,
        )

        self.assertEqual(peaks.shape, (1,))
        self.assertAlmostEqual(np.log10(peaks[0]), 4.0, delta=0.08)

    def test_multiple_traces_accept_mapping_and_custom_labels(self):
        frequency = np.logspace(3, 6, 300)
        psd = {
            "Q04": self._trace(frequency, [(1e4, 30.0, 0.025)]),
            "Q12": self._trace(frequency, [(1e5, 30.0, 0.025)]),
        }

        peaks = find_psd_peaks(frequency, psd, prominence=5.0)

        self.assertEqual(set(peaks), {"Q04", "Q12"})
        self.assertAlmostEqual(np.log10(peaks["Q04"][0]), 4.0, delta=0.03)
        self.assertAlmostEqual(np.log10(peaks["Q12"][0]), 5.0, delta=0.03)

    def test_multiple_traces_accept_2d_array_and_labels(self):
        frequency = np.logspace(3, 6, 300)
        psd = np.vstack(
            [
                self._trace(frequency, [(2e4, 30.0, 0.025)]),
                self._trace(frequency, [(2e5, 30.0, 0.025)]),
            ]
        )

        peaks = find_psd_peaks(
            frequency,
            psd,
            labels=["first", "second"],
            prominence=5.0,
        )

        self.assertEqual(set(peaks), {"first", "second"})
        self.assertAlmostEqual(np.log10(peaks["first"][0]), np.log10(2e4), delta=0.03)
        self.assertAlmostEqual(np.log10(peaks["second"][0]), np.log10(2e5), delta=0.03)

    def test_plot_uses_custom_axis_labels_and_title(self):
        frequency = np.logspace(3, 5, 100)
        psd = self._trace(frequency, [(1e4, 20.0, 0.04)])
        _, axis = plt.subplots()

        find_psd_peaks(
            frequency,
            psd,
            is_plot=True,
            ax=axis,
            title="Noise spectrum",
            frequency_unit="MHz",
            psd_unit="A$^2$/Hz",
            prominence=5.0,
            show_peak_frequencies=True,
        )

        self.assertEqual(axis.get_title(), "Noise spectrum")
        self.assertEqual(axis.get_xlabel(), "Frequency (MHz)")
        self.assertEqual(axis.get_ylabel(), "A$^2$/Hz")
        self.assertEqual(len(axis.collections), 1)
        self.assertTrue(any("kHz" in text.get_text() for text in axis.texts))

    def test_common_peaks_use_one_vertical_line(self):
        frequency = np.logspace(3, 6, 300)
        psd = {
            "first": self._trace(frequency, [(1e4, 30.0, 0.025)]),
            "second": self._trace(frequency, [(1.01e4, 25.0, 0.025)]),
        }
        _, axis = plt.subplots()

        find_psd_peaks(
            frequency,
            psd,
            is_plot=True,
            ax=axis,
            prominence=5.0,
            show_common_peak_lines=True,
            common_peak_tolerance=0.02,
        )

        self.assertEqual(len(axis.lines), 3)
        self.assertEqual(len(axis.collections), 0)
        self.assertEqual(len(axis.texts), 1)
        self.assertIn("kHz", axis.texts[0].get_text())


if __name__ == "__main__":
    unittest.main()
