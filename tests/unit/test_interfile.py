import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pysuqu.funclib as funclib
from pysuqu.funclib import interfile


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None


class CSVAdapterTests(unittest.TestCase):
    def test_file_adapters_are_exported_from_funclib(self):
        self.assertIs(funclib.read_csv_rows, interfile.read_csv_rows)
        self.assertIs(funclib.read_waveform_csv, interfile.read_waveform_csv)
        self.assertIs(funclib.process_psd_inh5groups, interfile.process_psd_inh5groups)

    def test_write_read_and_append_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rows.csv"
            written = interfile.write_csv_rows(
                path,
                [{"name": "first", "value": 1}, {"name": "second", "extra": "x"}],
            )
            self.assertEqual(written, path)
            self.assertEqual(
                interfile.read_csv_rows(path),
                [
                    {"name": "first", "value": "1", "extra": ""},
                    {"name": "second", "value": "", "extra": "x"},
                ],
            )
            interfile.append_csv_row(path, {"name": "third", "value": 3, "extra": "y"})
            rows = interfile.read_csv_rows(path, converters={"value": int})
            self.assertEqual(rows[-1], {"name": "third", "value": 3, "extra": "y"})

    def test_append_rejects_header_schema_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            interfile.write_csv_rows(path, [{"a": 1, "b": 2}])
            with self.assertRaisesRegex(ValueError, "header mismatch"):
                interfile.append_csv_row(path, {"a": 3}, fieldnames=("a",))

    def test_empty_write_can_emit_explicit_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            interfile.write_csv_rows(path, [], fieldnames=("a", "b"))
            self.assertEqual(path.read_text(encoding="utf-8"), "a,b\n")
            self.assertEqual(interfile.read_csv_rows(path), [])

    def test_waveform_csv_normalizes_columns_and_time_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.csv"
            path.write_text(
                " TIME ; I ; q \n0;1;2\n0.001;3;4\n",
                encoding="utf-8",
            )
            time_ns, waveform = interfile.read_waveform_csv(
                path,
                time_column="time",
                real_column="i",
                imag_column="q",
                time_unit="us",
            )
            np.testing.assert_allclose(time_ns, [0.0, 1.0])
            np.testing.assert_allclose(waveform, [1 + 2j, 3 + 4j])

    def test_waveform_csv_reports_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            missing = directory / "missing.csv"
            missing.write_text("time,real\n0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing waveform CSV column"):
                interfile.read_waveform_csv(missing)

            non_monotonic = directory / "non_monotonic.csv"
            non_monotonic.write_text(
                "time,real,imag\n1,1,0\n0,2,0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "monotonically increasing"):
                interfile.read_waveform_csv(non_monotonic)

            with self.assertRaisesRegex(ValueError, "Unsupported time unit"):
                interfile.read_waveform_csv(missing, time_unit="minute")


class OptionalHDF5Tests(unittest.TestCase):
    def test_missing_h5py_has_actionable_error(self):
        with patch.object(interfile, "_h5py", None):
            with self.assertRaisesRegex(ModuleNotFoundError, r"pysuqu\[hdf5\]"):
                interfile.list_h5_groups("missing.h5")

    @unittest.skipUnless(H5PY_AVAILABLE, "h5py extra is not installed")
    def test_list_and_process_psd_groups(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            h5_path = directory / "noise.h5"
            frequencies = np.array([1.0, 2.0, 3.0])
            bandwidth = np.array([1.0, 2.0, 4.0])
            xstddev = np.array([1.0, 2.0, 3.0])
            ystddev = np.array([0.0, 1.0, 2.0])
            with h5py.File(h5_path, "w") as h5:
                group = h5.create_group("sample_a/trace")
                group.attrs["label"] = np.bytes_("sensor-a")
                group.create_dataset("frequency", data=frequencies)
                group.create_dataset("bandwidth", data=bandwidth)
                group.create_dataset("xstddev", data=xstddev)
                group.create_dataset("ystddev", data=ystddev)
                group.create_dataset("rpwr", data=xstddev**2 + ystddev**2)

            listed = interfile.list_h5_groups(h5_path)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["sample_path"], "sample_a/trace")
            self.assertEqual(listed[0]["label"], "sensor-a")

            output_dir = directory / "csv"
            results = interfile.process_psd_inh5groups(
                h5_path,
                "sample_a",
                output_dir=output_dir,
                save_csv=True,
                verbose=True,
            )
            result = results["sample_a/trace"]
            np.testing.assert_allclose(result["psd"], [1.0, 2.5, 3.25])
            self.assertEqual(result["psd_source"], "xy_stddev")
            self.assertTrue(result["csv_path"].exists())
            self.assertEqual(len(interfile.read_csv_rows(result["csv_path"])), 3)

    @unittest.skipUnless(H5PY_AVAILABLE, "h5py extra is not installed")
    def test_hdf5_dataset_length_validation_can_be_relaxed(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "mismatch.h5"
            with h5py.File(h5_path, "w") as h5:
                group = h5.create_group("sample")
                group.create_dataset("frequency", data=[1.0, 2.0])
                group.create_dataset("bandwidth", data=[1.0])
                group.create_dataset("rpwr", data=[3.0])

            with self.assertRaisesRegex(ValueError, "Dataset length mismatch"):
                interfile.process_h5_groups(h5_path, "sample", save_csv=False)
            result = interfile.process_h5_groups(
                h5_path,
                "sample",
                save_csv=False,
                strict_lengths=False,
            )
            np.testing.assert_allclose(result["sample"]["psd"], [3.0])


if __name__ == "__main__":
    unittest.main()
