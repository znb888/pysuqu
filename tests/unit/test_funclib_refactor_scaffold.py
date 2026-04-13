import importlib.util
import importlib
import io
import subprocess
import sys
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tests.support import install_plotly_stub


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FUNCLIB_ROOT = WORKSPACE_ROOT / 'pysuqu' / 'funclib'


def load_module(module_name: str, relative_path: str):
    module_path = FUNCLIB_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=WORKSPACE_ROOT,
        check=False,
    )


class FunclibRefactorScaffoldTests(unittest.TestCase):
    def test_package_root_light_helpers_import_without_qutip_or_plotly(self):
        script = f"""
import importlib.abc
import pathlib
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qutip' or fullname.startswith('qutip.'):
            raise ModuleNotFoundError('blocked qutip for test')
        if fullname == 'plotly' or fullname.startswith('plotly.'):
            raise ModuleNotFoundError('blocked plotly for test')
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, str(pathlib.Path(r'{WORKSPACE_ROOT}')))
from pysuqu.funclib import fft_analysis, thermal_photon

assert callable(fft_analysis)
assert callable(thermal_photon)
"""
        result = run_python(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_awgenerator_non_plotting_helpers_work_without_plotly(self):
        script = f"""
import importlib.abc
import importlib.util
import numpy as np
import pathlib
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'plotly' or fullname.startswith('plotly.'):
            raise ModuleNotFoundError('blocked plotly for test')
        return None

sys.meta_path.insert(0, Blocker())
module_path = pathlib.Path(r'{FUNCLIB_ROOT / "awgenerator.py"}')
spec = importlib.util.spec_from_file_location('funclib_awgenerator_blocked_plotly', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

generator = module.WaveformGenerator(total_time=4.0, sample_rate=1.0)
envelope = module.EnvelopeParams(duration=4.0, peak_amp=1.0, shape_type='square')
schedule = module.ChannelSchedule(
    sampling_rate=1.0,
    mixer_correction=False,
    events=[module.PulseEvent(start_time=0.0, envelope=envelope)],
)
wave, _ = generator.generate_channel_waveform(schedule, return_complex=True)
assert np.allclose(np.real(wave), np.ones(4))
"""
        result = run_python(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_cpmg_transfunc_returns_expected_values_for_ordinary_n(self):
        mathlib = load_module('funclib_mathlib_round_b', 'mathlib.py')
        freq = np.array([0.125, 0.25])

        result_n2 = mathlib.cpmg_transfunc(freq, tau=1.0, N=2, len_pi=0.0)
        result_n3 = mathlib.cpmg_transfunc(freq, tau=1.0, N=3, len_pi=0.0)

        expected_n2 = np.abs(
            1
            + np.exp(1j * 2 * np.pi * freq)
            + 2
            * (
                -np.exp(1j * 2 * np.pi * (1 / 2) * freq)
                + np.exp(1j * 2 * np.pi * (3 / 2) * freq)
            )
        ) / (4 * np.pi * freq)
        expected_n3 = np.abs(
            1
            - np.exp(1j * 2 * np.pi * freq)
            + 2
            * (
                -np.exp(1j * 2 * np.pi * (1 / 3) * freq)
                + np.exp(1j * 2 * np.pi * freq)
                - np.exp(1j * 2 * np.pi * (5 / 3) * freq)
            )
        ) / (4 * np.pi * freq)

        self.assertTrue(np.allclose(result_n2, expected_n2))
        self.assertTrue(np.allclose(result_n3, expected_n3))

    def test_fft_analysis_rejects_short_inputs_with_clear_error(self):
        mathlib = load_module('funclib_mathlib_round_f_fft', 'mathlib.py')

        with self.assertRaisesRegex(ValueError, 'at least 2 samples'):
            mathlib.fft_analysis(np.array([0.0]), np.array([1.0]))

    def test_smooth_data_savgol_clamps_oversize_window(self):
        mathlib = load_module('funclib_mathlib_round_f_smooth', 'mathlib.py')
        data = np.arange(5.0)

        result = mathlib.smooth_data(data, method='savgol', window=7, polyorder=3)
        expected = mathlib.savgol_filter(data, window_length=5, polyorder=3)

        self.assertTrue(np.allclose(result, expected))

    def test_find_knee_point_handles_degenerate_inputs_without_runtime_warning(self):
        mathlib = load_module('funclib_mathlib_round_f_knee', 'mathlib.py')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaisesRegex(ValueError, 'must vary'):
                mathlib.find_knee_point(np.ones(4), np.arange(4.0))
            knee_x, knee_idx = mathlib.find_knee_point(np.arange(4.0), np.ones(4))

        self.assertEqual(knee_x, 0.0)
        self.assertEqual(knee_idx, 0)
        self.assertEqual(caught, [])

    def test_thermal_photon_matches_kelvin_contract(self):
        mathlib = load_module('funclib_mathlib_round_g_thermal', 'mathlib.py')

        temperature_k = 0.05
        frequency_ghz = 6.7

        result = mathlib.thermal_photon(temperature_k, frequency_ghz)
        expected = mathlib.temp2nbar(temperature_k, ff=frequency_ghz * 1e9)

        self.assertAlmostEqual(result, expected)

    def test_import_waveform_preserves_sample_count(self):
        install_plotly_stub()
        awgenerator = load_module('funclib_awgenerator_round_b', 'awgenerator.py')

        time = np.arange(4.0)
        wave = np.ones(4, dtype=complex)

        schedule = awgenerator.import_waveform(time, wave, sampling_rate=1.0)
        generator = awgenerator.WaveformGenerator(total_time=4.0, sample_rate=1.0)
        generated_wave, _ = generator.generate_channel_waveform(schedule, return_complex=True)

        self.assertEqual(np.count_nonzero(np.real(generated_wave)), 4)
        self.assertTrue(np.allclose(np.real(generated_wave), [1.0, 1.0, 1.0, 1.0]))

    def test_triangle_shape_peaks_in_the_middle(self):
        install_plotly_stub()
        awgenerator = load_module('funclib_awgenerator_triangle_round_b', 'awgenerator.py')

        generator = awgenerator.WaveformGenerator(total_time=4.0, sample_rate=1.0)
        envelope = awgenerator.EnvelopeParams(duration=4.0, peak_amp=1.0, shape_type='triangle')

        baseband = generator._generate_baseband(np.arange(4.0), envelope)

        self.assertTrue(np.allclose(np.real(baseband), [0.0, 0.5, 1.0, 0.5]))

    def test_channel_schedule_display_uses_ghz_and_gsas_units_consistently(self):
        install_plotly_stub()
        awgenerator = load_module('funclib_awgenerator_display_round_i', 'awgenerator.py')

        schedule = awgenerator.ChannelSchedule(
            name='drive',
            sampling_rate=2.0,
            mixer_config=awgenerator.MixerParams(lo_freq=5.0),
            events=[
                awgenerator.PulseEvent(
                    start_time=1.0,
                    name='x90',
                    if_freq=0.05,
                    phase_offset=0.125,
                    envelope=awgenerator.EnvelopeParams(duration=4.0, shape_type='square'),
                )
            ],
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            schedule.display()

        output = stdout.getvalue()
        self.assertIn('Mixer LO: 5.000 GHz | SR: 2.00 GSa/s', output)
        self.assertIn('Freq (MHz)', output)
        self.assertIn('x90 | 1.00', output)
        self.assertIn('50.00', output)

    def test_mixer_params_repr_is_complete(self):
        install_plotly_stub()
        awgenerator = load_module('funclib_awgenerator_repr_round_i', 'awgenerator.py')

        mixer = awgenerator.MixerParams(
            lo_freq=5.0,
            gain_ratio=1.1,
            phase_error=0.2,
            lo_leakage_i=0.01,
            lo_leakage_q=-0.02,
        )

        result = repr(mixer)

        self.assertIn('lo_freq=5.00GHz', result)
        self.assertIn('gain_ratio=1.1', result)
        self.assertIn('phase_error=0.2rad', result)
        self.assertIn('lo_leakage_i=0.01V', result)
        self.assertIn('lo_leakage_q=-0.02V', result)
        self.assertTrue(result.endswith('>'))

    def test_sii_dbm2temp_honors_visible_frequency_and_resistance_parameters(self):
        noisemodel = importlib.import_module('pysuqu.funclib.noisemodel')

        sii_dbm = -120.0
        frequency_hz = 5.5e9
        resistance_ohm = 75.0

        result = noisemodel.Sii_dBm2temp(sii_dbm, f=frequency_hz, R=resistance_ohm)
        expected = noisemodel.Sii2T_Double(
            noisemodel.Sii_dBm2A(sii_dbm, R=resistance_ohm),
            f=frequency_hz,
            R=resistance_ohm,
        )

        self.assertAlmostEqual(result, expected)

    def test_sii_dbm2temp_keeps_the_historical_default_frequency(self):
        noisemodel = importlib.import_module('pysuqu.funclib.noisemodel')

        sii_dbm = -118.0
        resistance_ohm = 50.0

        result = noisemodel.Sii_dBm2temp(sii_dbm, R=resistance_ohm)
        expected = noisemodel.Sii2T_Double(
            noisemodel.Sii_dBm2A(sii_dbm, R=resistance_ohm),
            f=6.7e9,
            R=resistance_ohm,
        )

        self.assertAlmostEqual(result, expected)

    def test_double_and_single_sided_current_noise_conversions_round_trip(self):
        noisemodel = importlib.import_module('pysuqu.funclib.noisemodel')

        frequency_hz = np.array([4.5e9, 6.7e9])
        resistance_ohm = 75.0
        sii_double = np.array([1.3e-25, 2.1e-25])

        sii_single = noisemodel.Sii_D2S(sii_double, f=frequency_hz, R=resistance_ohm)
        round_trip = noisemodel.Sii_S2D(sii_single, f=frequency_hz, R=resistance_ohm)

        self.assertTrue(np.allclose(round_trip, sii_double))


if __name__ == '__main__':
    unittest.main()

