"""Focused checks for opt-in AWG waveform caching."""

import unittest
from unittest.mock import patch

import numpy as np

from pysuqu.funclib.awgenerator import (
    ChannelSchedule,
    EnvelopeParams,
    MixerParams,
    PulseEvent,
    WaveformCache,
    WaveformGenerator,
)


def _schedule(amplitude=0.6, *, custom=False):
    envelope = EnvelopeParams(
        name="cache_probe",
        duration=2.0,
        peak_amp=amplitude,
        shape_type="custom" if custom else "square",
        custom_func=(lambda t: np.ones_like(t)) if custom else None,
    )
    return ChannelSchedule(
        name="synthetic_drive",
        sampling_rate=4.0,
        mixer_config=MixerParams(lo_freq=0.35),
        mixer_correction=False,
        events=[PulseEvent(start_time=0.0, envelope=envelope)],
    )


class WaveformCacheTests(unittest.TestCase):
    def test_reuses_trace_without_aliasing_arrays(self):
        cache = WaveformCache(max_entries=2)
        generator = WaveformGenerator(total_time=2.0, sample_rate=4.0, cache=cache)
        schedule = _schedule()

        with patch.object(
            generator, "_compile_awg_iq_complex", wraps=generator._compile_awg_iq_complex
        ) as compile_wave:
            first = generator.generate_awg_output(schedule)
            first.values[0] = 99.0
            second = generator.generate_awg_output(schedule)

        self.assertEqual(compile_wave.call_count, 1)
        self.assertEqual(cache.info(), {"size": 1, "max_entries": 2, "hits": 1, "misses": 1})
        self.assertIsNot(first, second)
        np.testing.assert_allclose(second.values, 0.6 * np.ones(8))

    def test_mutating_schedule_invalidates_key_and_callable_bypasses_cache(self):
        cache = WaveformCache()
        generator = WaveformGenerator(total_time=2.0, sample_rate=4.0, cache=cache)
        schedule = _schedule()
        generator.generate_awg_output(schedule)
        schedule.events[0].envelope.peak_amp = 0.25
        changed = generator.generate_awg_output(schedule)

        np.testing.assert_allclose(changed.values, 0.25 * np.ones(8))
        self.assertEqual(cache.info()["hits"], 0)
        self.assertEqual(cache.info()["misses"], 2)

        callable_schedule = _schedule(custom=True)
        generator.generate_awg_output(callable_schedule)
        generator.generate_awg_output(callable_schedule)
        self.assertEqual(cache.info()["size"], 2)
        self.assertEqual(cache.info()["hits"], 0)
        self.assertEqual(cache.info()["misses"], 4)

    def test_cache_is_bounded_and_clear_resets_statistics(self):
        cache = WaveformCache(max_entries=1)
        generator = WaveformGenerator(total_time=2.0, sample_rate=4.0, cache=cache)
        generator.generate_awg_output(_schedule(0.2))
        generator.generate_awg_output(_schedule(0.4))
        self.assertEqual(cache.info()["size"], 1)
        self.assertEqual(cache.info()["misses"], 2)
        cache.clear()
        self.assertEqual(cache.info(), {"size": 0, "max_entries": 1, "hits": 0, "misses": 0})

    def test_constructor_rejects_invalid_cache_options(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            WaveformCache(max_entries=0)
        with self.assertRaises(TypeError):
            WaveformGenerator(total_time=1.0, sample_rate=1.0, cache=object())


if __name__ == "__main__":
    unittest.main()
