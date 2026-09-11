import unittest

import numpy as np

from pysuqu.funclib.transmission import (
    MIMOTouchstoneStage,
    SignalBundle,
    SignalTrace,
    TouchstoneNetwork,
    TouchstoneStage,
)


class TouchstoneResponseCacheTests(unittest.TestCase):
    def _case(self, *, cache_size=4):
        frequencies = np.linspace(-2.0, 2.0, 17)
        scattering = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
        scattering[:, 1, 0] = 0.5 + 0.1j * frequencies
        network = TouchstoneNetwork(frequencies, scattering)
        stage = TouchstoneStage(
            network=network,
            input_port=1,
            output_port=2,
            frequency_mode="relative",
            response_cache_size=cache_size,
        )
        values = np.sin(np.linspace(0.0, 4.0 * np.pi, 32))
        trace = SignalTrace(
            t_axis=np.arange(values.size, dtype=float) / 2.0,
            values=values,
            sample_rate=2.0,
            domain="rf_real",
            plane="awg_rf",
        )
        return network, stage, trace

    def test_repeat_uses_cached_response(self):
        _, stage, trace = self._case()
        first = stage.apply(trace)
        second = stage.apply(trace)
        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(len(stage._response_cache), 1)

    def test_mutating_network_invalidates_response_key(self):
        network, stage, trace = self._case()
        before = stage.apply(trace).values.copy()
        network.s_parameters[:, 1, 0] *= 0.25
        after = stage.apply(trace).values
        self.assertGreater(float(np.max(np.abs(before - after))), 1e-9)
        self.assertEqual(len(stage._response_cache), 2)

    def test_cache_can_be_disabled(self):
        _, stage, trace = self._case(cache_size=0)
        stage.apply(trace)
        stage.apply(trace)
        self.assertEqual(len(stage._response_cache), 0)

    def test_cache_size_must_be_nonnegative(self):
        with self.assertRaises(ValueError):
            self._case(cache_size=-1)

    def test_mimo_repeat_uses_cached_response(self):
        frequencies = np.linspace(-2.0, 2.0, 17)
        scattering = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
        scattering[:, 0, 0] = 0.5
        scattering[:, 1, 0] = 0.2
        scattering[:, 0, 1] = 0.1
        scattering[:, 1, 1] = 0.4
        network = TouchstoneNetwork(frequencies, scattering)
        stage = MIMOTouchstoneStage(
            network=network,
            input_ports=(1, 2),
            output_ports=(1, 2),
            frequency_mode="relative",
        )
        values = np.sin(np.linspace(0.0, 4.0 * np.pi, 32))
        bundle = SignalBundle(
            traces={
                "a": SignalTrace(np.arange(32) / 2.0, values, 2.0, "rf_real", "awg_rf"),
                "b": SignalTrace(np.arange(32) / 2.0, values * 0.5, 2.0, "rf_real", "awg_rf"),
            },
            order=("a", "b"),
        )
        first = stage.apply(bundle)
        second = stage.apply(bundle)
        for name in first.order:
            np.testing.assert_array_equal(first[name].values, second[name].values)
        self.assertEqual(len(stage._response_cache), 1)


if __name__ == "__main__":
    unittest.main()
