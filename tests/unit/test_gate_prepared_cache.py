import unittest
from unittest.mock import patch

import numpy as np
import qutip as qt

from pysuqu.funclib.awgenerator import (
    ChannelSchedule, EnvelopeParams, MixerParams, PulseEvent,
)
from pysuqu.qubit.gate import SingleQubitGate


@unittest.skipUnless(getattr(qt, '__version__', None) and hasattr(qt.Qobj, 'expm'),
                     'real QuTiP is required')
class GatePreparedCacheTests(unittest.TestCase):
    def _case(self):
        gate = SingleQubitGate(total_time=12.0, sample_rate=2.0,
                               qubit_frequency=5.0, energy_trunc_level=3)
        schedule = ChannelSchedule(
            sampling_rate=2.0,
            mixer_config=MixerParams(lo_freq=5.0),
            events=[PulseEvent(
                start_time=1.0,
                envelope=EnvelopeParams(duration=6.0, peak_amp=8e-4,
                                        shape_type="gaussian", sigma=1.5),
                if_freq=0.1,
            )],
        )
        options = {
            "atol": 1e-8, "rtol": 1e-6, "store_states": False,
            "store_final_state": True, "matrix_format": "dense", "frame": "lab",
        }
        return gate, schedule, options

    def test_opt_in_reuse_prepares_once(self):
        gate, schedule, options = self._case()
        with patch.object(gate, "prepare_trace_propagator",
                          wraps=gate.prepare_trace_propagator) as prepare:
            first = gate.run_simulation(channel=schedule, backend="qutip_compiled",
                                        options=options, reuse_prepared=True)
            second = gate.run_simulation(channel=schedule, backend="qutip_compiled",
                                         options=options, reuse_prepared=True)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(gate.prepared_cache_info()["entries"], 1)
        np.testing.assert_allclose(first.final_state.full(), second.final_state.full(),
                                   rtol=0.0, atol=0.0)

    def test_mutating_schedule_invalidates_key(self):
        gate, schedule, options = self._case()
        with patch.object(gate, "prepare_trace_propagator",
                          wraps=gate.prepare_trace_propagator) as prepare:
            gate.run_simulation(channel=schedule, backend="qutip_compiled",
                                options=options, reuse_prepared=True)
            schedule.events.append(PulseEvent(
                start_time=8.0,
                envelope=EnvelopeParams(duration=1.5, peak_amp=2e-4,
                                        shape_type="square"),
            ))
            gate.run_simulation(channel=schedule, backend="qutip_compiled",
                                options=options, reuse_prepared=True)
        self.assertEqual(prepare.call_count, 2)

    def test_clear_resets_entries(self):
        gate, schedule, options = self._case()
        gate.run_simulation(channel=schedule, backend="qutip_compiled",
                            options=options, reuse_prepared=True)
        self.assertEqual(gate.prepared_cache_info()["entries"], 1)
        gate.clear_prepared_cache()
        self.assertEqual(gate.prepared_cache_info(), {"entries": 0, "max_entries": 4})


if __name__ == "__main__":
    unittest.main()
