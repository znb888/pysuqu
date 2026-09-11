"""Measure opt-in AWG waveform-cache reuse on a synthetic schedule."""

from __future__ import annotations

import argparse
import time

import numpy as np

from pysuqu.funclib.awgenerator import (
    ChannelSchedule,
    EnvelopeParams,
    MixerParams,
    PulseEvent,
    WaveformCache,
    WaveformGenerator,
)


def build_schedule() -> ChannelSchedule:
    events = [
        PulseEvent(
            start_time=1.5,
            envelope=EnvelopeParams(
                name="gaussian_probe",
                duration=8.0,
                peak_amp=0.7,
                shape_type="gaussian",
                sigma=2.0,
                drag_coeff=0.04,
            ),
            if_freq=0.09,
            phase_offset=0.17,
        ),
        PulseEvent(
            start_time=14.0,
            envelope=EnvelopeParams(
                name="cosine_probe",
                duration=6.0,
                peak_amp=0.35,
                shape_type="cosine",
            ),
            if_freq=-0.06,
        ),
    ]
    return ChannelSchedule(
        name="synthetic_benchmark",
        sampling_rate=2.0,
        mixer_config=MixerParams(lo_freq=0.42, gain_ratio=1.01, phase_error=0.012),
        mixer_correction=True,
        events=events,
    )


def measure(generator: WaveformGenerator, schedule: ChannelSchedule, repeats: int):
    start = time.perf_counter()
    output = None
    for _ in range(repeats):
        output = generator.generate_awg_output(schedule, mode="iq")
    return time.perf_counter() - start, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    schedule = build_schedule()
    uncached = WaveformGenerator(total_time=24.0, sample_rate=2.0)
    cache = WaveformCache(max_entries=4)
    cached = WaveformGenerator(total_time=24.0, sample_rate=2.0, cache=cache)

    uncached.generate_awg_output(schedule)
    cached.generate_awg_output(schedule)
    uncached_time, reference = measure(uncached, schedule, args.repeats)
    cached_time, result = measure(cached, schedule, args.repeats)

    print(f"repeats={args.repeats}")
    print(f"uncached_seconds={uncached_time:.9f}")
    print(f"cached_seconds={cached_time:.9f}")
    print(f"speedup={uncached_time / cached_time:.3f}x")
    print(f"max_abs_difference={np.max(np.abs(reference.values - result.values)):.3e}")
    print(f"cache={cache.info()}")


if __name__ == "__main__":
    main()
