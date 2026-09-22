import tempfile
import unittest
from pathlib import Path

import numpy as np

from pysuqu.funclib.awgenerator import import_waveform_from_file, _load_waveform_csv


class AWGFileImportTests(unittest.TestCase):
    def test_legacy_csv_loader_delegates_to_interfile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.csv"
            path.write_text(
                "time,real,imag\n0,1,0\n1,2,1\n2,3,0\n",
                encoding="utf-8",
            )
            time_ns, waveform = _load_waveform_csv(str(path))
            np.testing.assert_allclose(time_ns, [0.0, 1.0, 2.0])
            np.testing.assert_allclose(waveform, [1.0, 2.0 + 1j, 3.0])

    def test_import_waveform_from_file_preserves_event_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.csv"
            path.write_text(
                "time,real,imag\n0,1,0\n1,2,1\n2,3,0\n",
                encoding="utf-8",
            )
            event = import_waveform_from_file(
                path,
                target_type="event",
                if_freq=0.125,
                frame_change=0.25,
            )
            self.assertEqual(event.start_time, 0.0)
            self.assertEqual(event.if_freq, 0.125)
            self.assertEqual(event.frame_change, 0.25)
            self.assertEqual(event.envelope.shape_type, "custom")
            np.testing.assert_allclose(event.envelope.custom_func([0.0, 1.0, 2.0]), [1.0, 2.0 + 1j, 3.0])


if __name__ == "__main__":
    unittest.main()
