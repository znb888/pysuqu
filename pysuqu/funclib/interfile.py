"""Small, reusable adapters for data files used by :mod:`pysuqu`.

The module keeps format mechanics separate from domain objects. CSV support
uses only the standard library. HDF5 support is optional and is imported
lazily so importing :mod:`pysuqu.funclib` does not require ``h5py``.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

try:
    import h5py as _h5py
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency tests
    _h5py = None


PathInput = Union[str, os.PathLike]
CSVRow = Mapping[str, Any]

__all__ = [
    "append_csv_row",
    "list_h5_groups",
    "process_h5_groups",
    "process_psd_inh5groups",
    "print_h5_groups",
    "read_csv_rows",
    "read_h5_dataset",
    "read_waveform_csv",
    "write_csv_rows",
]


def _as_path(path: PathInput) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _detect_csv_delimiter(sample: str) -> str:
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv_table(
    path: PathInput,
    *,
    delimiter: str | None,
    encoding: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    path = _as_path(path)
    with path.open("r", newline="", encoding=encoding) as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = csv.DictReader(
            handle,
            delimiter=delimiter or _detect_csv_delimiter(sample),
        )
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def read_csv_rows(
    path: PathInput,
    *,
    delimiter: str | None = None,
    encoding: str = "utf-8",
    converters: Optional[Mapping[str, Callable[[str], Any]]] = None,
) -> list[dict[str, Any]]:
    """Read a CSV table as rows of strings, with optional column converters.

    Empty files and header-only files return an empty list. Column names are
    preserved exactly; callers that need case-insensitive matching should
    normalize them explicitly.
    """
    _, rows = _read_csv_table(path, delimiter=delimiter, encoding=encoding)
    if not converters:
        return rows

    converted_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value in (None, "") or key not in converters:
                converted[key] = value
                continue
            try:
                converted[key] = converters[key](value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unable to convert CSV column {key!r} on row {row_number} in {path}."
                ) from exc
        converted_rows.append(converted)
    return converted_rows


def _fieldnames_for_rows(rows: Sequence[CSVRow]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def write_csv_rows(
    path: PathInput,
    rows: Iterable[CSVRow],
    *,
    fieldnames: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> Path:
    """Write mapping rows to CSV and return the normalized output path.

    When ``fieldnames`` is omitted, keys are collected in first-seen order.
    Empty input creates an empty file unless an explicit header is supplied.
    """
    path = _as_path(path)
    materialized = [dict(row) for row in rows]
    names = (
        [str(name) for name in fieldnames]
        if fieldnames is not None
        else _fieldnames_for_rows(materialized)
    )
    _ensure_parent(path)
    with path.open("w", newline="", encoding=encoding) as handle:
        if names:
            writer = csv.DictWriter(
                handle,
                fieldnames=names,
                delimiter=delimiter,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(materialized)
    return path


def append_csv_row(
    path: PathInput,
    row: CSVRow,
    *,
    fieldnames: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> Path:
    """Append one CSV row, creating and validating the header as needed."""
    path = _as_path(path)
    row_dict = dict(row)
    _ensure_parent(path)
    needs_header = not path.exists() or path.stat().st_size == 0

    existing_names: list[str] = []
    if not needs_header:
        existing_names, _ = _read_csv_table(
            path,
            delimiter=delimiter,
            encoding=encoding,
        )
        if not existing_names:
            needs_header = True

    names = [str(name) for name in fieldnames] if fieldnames is not None else existing_names
    if not names:
        names = _fieldnames_for_rows([row_dict])
    if existing_names and names != existing_names:
        raise ValueError(
            f"CSV header mismatch for {path}: expected {existing_names}, received {names}."
        )

    with path.open("a", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=names,
            delimiter=delimiter,
            extrasaction="raise",
        )
        if needs_header:
            writer.writeheader()
        writer.writerow(row_dict)
    return path


def read_waveform_csv(
    path: PathInput,
    *,
    time_column: str = "time",
    real_column: str = "real",
    imag_column: str = "imag",
    time_unit: str = "ns",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a time/I/Q CSV and return time in ns plus a complex waveform."""
    path = _as_path(path)
    scales = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
    normalized_unit = time_unit.strip().lower()
    if normalized_unit not in scales:
        raise ValueError(
            f"Unsupported time unit {time_unit!r}; expected one of {sorted(scales)}."
        )

    fieldnames, rows = _read_csv_table(path, delimiter=None, encoding="utf-8")
    if not fieldnames:
        raise ValueError(f"CSV file is empty or has no headers: {path}")

    normalized = [name.strip().lower() for name in fieldnames]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"CSV file has duplicate column names after normalization: {path}")
    column_map = {name: actual for name, actual in zip(normalized, fieldnames)}
    requested = {
        "time": time_column.strip().lower(),
        "real": real_column.strip().lower(),
        "imag": imag_column.strip().lower(),
    }
    missing = [label for label, name in requested.items() if name not in column_map]
    if missing:
        raise ValueError(
            f"Missing waveform CSV column(s) {missing}; available columns: {fieldnames}."
        )
    if not rows:
        raise ValueError(f"CSV file contains no data rows: {path}")

    time_values: list[float] = []
    real_values: list[float] = []
    imag_values: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        values: list[float] = []
        for label in ("time", "real", "imag"):
            column = column_map[requested[label]]
            raw = row.get(column)
            try:
                values.append(float(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value in waveform CSV column {column!r} "
                    f"on row {row_number}: {raw!r}."
                ) from exc
        time_values.append(values[0] * scales[normalized_unit])
        real_values.append(values[1])
        imag_values.append(values[2])

    time_array = np.asarray(time_values, dtype=float)
    complex_wave = np.asarray(real_values, dtype=float) + 1j * np.asarray(imag_values, dtype=float)
    if not np.all(np.diff(time_array) >= 0):
        raise ValueError(f"Time array must be monotonically increasing: {path}")
    return time_array, complex_wave


def _require_h5py():
    if _h5py is None:
        raise ModuleNotFoundError(
            "HDF5 support requires the optional dependency 'h5py'. "
            "Install it with: pip install pysuqu[hdf5]"
        )
    return _h5py


def _decode_h5_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else ""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _group_label(group: Any) -> str:
    label = group.attrs.get("label", group.attrs.get("name", None))
    if label is None and "chunkheader" in group:
        try:
            header = group["chunkheader"][()]
            names = getattr(header.dtype, "names", None)
            if names and "name" in names:
                label = header["name"].reshape(-1)[0]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            label = None
    return "" if label is None else _decode_h5_value(label)


def read_h5_dataset(
    path: PathInput,
    dataset_path: str,
    *,
    dtype: Any = None,
) -> np.ndarray:
    """Read one HDF5 dataset into a detached NumPy array."""
    h5py = _require_h5py()
    with h5py.File(_as_path(path), "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"HDF5 dataset does not exist: {dataset_path}")
        item = h5[dataset_path]
        if not isinstance(item, h5py.Dataset):
            raise TypeError(f"HDF5 path is not a dataset: {dataset_path}")
        values = item[()]
    return np.asarray(values, dtype=dtype)


def list_h5_groups(path: PathInput) -> list[dict[str, Any]]:
    """List HDF5 groups containing the datasets needed for PSD analysis."""
    h5py = _require_h5py()
    path = _as_path(path)
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        sample_paths: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Group):
                return
            keys = set(obj.keys())
            if "frequency" in keys and "bandwidth" in keys and (
                "rpwr" in keys
                or {"xstddev", "ystddev"}.issubset(keys)
                or "rstddev" in keys
            ):
                sample_paths.append(name)

        h5.visititems(visit)
        for sample_path in sorted(sample_paths):
            group = h5[sample_path]
            frequency = np.asarray(group["frequency"][()], dtype=float).ravel()
            finite_frequency = frequency[np.isfinite(frequency)]
            rows.append(
                {
                    "group_id": sample_path.split("/", 1)[0],
                    "sample_path": sample_path,
                    "points": int(frequency.size),
                    "frequency_min_hz": float(np.min(finite_frequency)) if finite_frequency.size else float("nan"),
                    "frequency_max_hz": float(np.max(finite_frequency)) if finite_frequency.size else float("nan"),
                    "label": _group_label(group),
                    "has_rpwr": "rpwr" in group,
                    "has_xy_stddev": {"xstddev", "ystddev"}.issubset(group.keys()),
                    "has_rstddev": "rstddev" in group,
                    "datasets": sorted(group.keys()),
                }
            )
    return rows

def _matches_h5_group(sample_path: str, selectors: Optional[Sequence[str]]) -> bool:
    """Match a sample path against group ids, paths, or path prefixes."""
    if not selectors:
        return True
    group_id = sample_path.split("/", 1)[0]
    for selector in selectors:
        normalized = str(selector).strip().strip("/")
        if not normalized:
            continue
        if (
            sample_path == normalized
            or group_id == normalized
            or sample_path.startswith(normalized + "/")
        ):
            return True
    return False


def _h5_dataset_vector(group: Any, name: str) -> Optional[np.ndarray]:
    """Read one optional HDF5 dataset as a detached finite-agnostic vector."""
    if name not in group:
        return None
    return np.asarray(group[name][()], dtype=float).reshape(-1)


def process_h5_groups(
    path: PathInput,
    group_id: Optional[Union[str, Sequence[str]]] = None,
    *,
    group_selector: Optional[Union[str, Sequence[str]]] = None,
    group_name: Optional[Union[str, Sequence[str]]] = None,
    mode: str = "xy_stddev2_over_bandwidth",
    output_dir: Optional[PathInput] = None,
    save_csv: bool = False,
    strict_lengths: bool = True,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Process Zurich Instruments HDF5 sweeper groups into PSD arrays.

    ``group_id``/``group_selector``/``group_name`` may identify a top-level
    group, an exact sample path, or a sample-path prefix.  The returned mapping
    is keyed by each matching sample path and contains detached arrays under
    ``freq`` and ``psd``.  HDF5 remains an optional dependency and is loaded
    only when this function is called.
    """
    h5py = _require_h5py()
    selectors = [item for item in (group_id, group_selector, group_name) if item is not None]
    if len(selectors) > 1:
        raise ValueError("Specify only one of group_id, group_selector, or group_name.")
    selector_value = selectors[0] if selectors else None
    if selector_value is None:
        normalized_selectors = None
    elif isinstance(selector_value, str):
        normalized_selectors = [selector_value]
    else:
        normalized_selectors = [str(item) for item in selector_value]

    mode_aliases = {
        "xy_stddev": "xy_stddev2_over_bandwidth",
        "rpwr": "rpwr_over_bandwidth",
        "rstddev": "rstddev2_over_bandwidth",
    }
    requested_mode = mode_aliases.get(str(mode), str(mode))
    path_obj = _as_path(path)
    results: dict[str, dict[str, Any]] = {}

    from .noisemodel import compare_psd, process_psd

    with h5py.File(path_obj, "r") as h5:
        sample_paths: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Group):
                return
            keys = set(obj.keys())
            if "frequency" in keys and "bandwidth" in keys and (
                "rpwr" in keys
                or {"xstddev", "ystddev"}.issubset(keys)
                or "rstddev" in keys
            ) and _matches_h5_group(name, normalized_selectors):
                sample_paths.append(name)

        h5.visititems(visit)
        for sample_path in sorted(sample_paths):
            group = h5[sample_path]
            arrays = {
                name: _h5_dataset_vector(group, name)
                for name in ("frequency", "bandwidth", "rpwr", "xstddev", "ystddev", "rstddev")
            }
            required = arrays["frequency"]
            bandwidth = arrays["bandwidth"]
            assert required is not None and bandwidth is not None
            lengths = {
                name: int(values.size)
                for name, values in arrays.items()
                if values is not None
            }
            expected_length = lengths["frequency"]
            mismatched = {
                name: length
                for name, length in lengths.items()
                if length != expected_length
            }
            if mismatched and strict_lengths:
                details = ", ".join(
                    f"{name}={length}" for name, length in sorted(lengths.items())
                )
                raise ValueError(
                    f"Dataset length mismatch in HDF5 group {sample_path}: {details}."
                )
            if mismatched:
                usable_length = min(lengths.values())
                arrays = {
                    name: (None if values is None else values[:usable_length])
                    for name, values in arrays.items()
                }

            frequency = np.asarray(arrays["frequency"], dtype=float)
            bandwidth = np.asarray(arrays["bandwidth"], dtype=float)
            psd_by_mode: dict[str, np.ndarray] = {}
            if arrays["xstddev"] is not None and arrays["ystddev"] is not None:
                _, psd_by_mode["xy_stddev2_over_bandwidth"] = process_psd(
                    frequency,
                    bandwidth,
                    "xy_stddev2_over_bandwidth",
                    xstddev=arrays["xstddev"],
                    ystddev=arrays["ystddev"],
                )
            if arrays["rpwr"] is not None:
                _, psd_by_mode["rpwr_over_bandwidth"] = process_psd(
                    frequency,
                    bandwidth,
                    "rpwr_over_bandwidth",
                    rpwr=arrays["rpwr"],
                )
                _, psd_by_mode["rpwr_raw"] = process_psd(
                    frequency,
                    bandwidth,
                    "rpwr_raw",
                    rpwr=arrays["rpwr"],
                )
            if arrays["rstddev"] is not None:
                _, psd_by_mode["rstddev2_over_bandwidth"] = process_psd(
                    frequency,
                    bandwidth,
                    "rstddev2_over_bandwidth",
                    rstddev=arrays["rstddev"],
                )
            if not psd_by_mode:
                raise ValueError(f"HDF5 group {sample_path} contains no supported PSD fields.")
            resolved_mode = requested_mode
            if resolved_mode not in psd_by_mode and (
                requested_mode == "xy_stddev2_over_bandwidth"
                and "rpwr_over_bandwidth" in psd_by_mode
            ):
                # Many Zurich sweeper exports contain only ``rpwr``.  Keep the
                # XY representation as the preferred mode when available, but
                # use the equivalent power trace when it is not present.
                resolved_mode = "rpwr_over_bandwidth"
            if resolved_mode not in psd_by_mode:
                available = ", ".join(sorted(psd_by_mode))
                raise ValueError(
                    f"PSD mode {requested_mode!r} is unavailable for group {sample_path}; "
                    f"available modes: {available}."
                )

            selected_psd = psd_by_mode[resolved_mode]
            source_labels = {
                "xy_stddev2_over_bandwidth": "xy_stddev",
                "rpwr_over_bandwidth": "rpwr",
                "rpwr_raw": "rpwr_raw",
                "rstddev2_over_bandwidth": "rstddev",
            }
            item: dict[str, Any] = {
                "group_id": sample_path.split("/", 1)[0],
                "sample_path": sample_path,
                "label": _group_label(group),
                "freq": frequency.copy(),
                "frequency": frequency.copy(),
                "bandwidth": bandwidth.copy(),
                "psd": np.asarray(selected_psd, dtype=float).copy(),
                "psd_source": source_labels[resolved_mode],
                "psd_mode": resolved_mode,
                "requested_psd_mode": requested_mode,
                "psd_by_mode": {
                    key: np.asarray(values, dtype=float).copy()
                    for key, values in psd_by_mode.items()
                },
            }
            if "xy_stddev2_over_bandwidth" in psd_by_mode and "rpwr_over_bandwidth" in psd_by_mode:
                item["crosscheck"] = compare_psd(
                    frequency,
                    psd_by_mode["xy_stddev2_over_bandwidth"],
                    psd_by_mode["rpwr_over_bandwidth"],
                )

            if save_csv:
                csv_root = _as_path(output_dir) if output_dir is not None else path_obj.parent / "psd_csv"
                safe_name = sample_path.replace("/", "_").replace("\\", "_")
                csv_path = csv_root / f"{safe_name}.csv"
                write_csv_rows(
                    csv_path,
                    (
                        {"frequency": float(freq), "psd": float(value)}
                        for freq, value in zip(item["freq"], item["psd"])
                    ),
                    fieldnames=("frequency", "psd"),
                )
                item["csv_path"] = csv_path
            if verbose:
                print(
                    f"group={sample_path} | mode={resolved_mode} | "
                    f"points={item['freq'].size} | source={item['psd_source']}"
                )
            results[sample_path] = item

    return results


def process_psd_inh5groups(
    path: PathInput,
    group_id: Optional[Union[str, Sequence[str]]] = None,
    *,
    group_selector: Optional[Union[str, Sequence[str]]] = None,
    group_name: Optional[Union[str, Sequence[str]]] = None,
    mode: str = "xy_stddev2_over_bandwidth",
    output_dir: Optional[PathInput] = None,
    save_csv: bool = False,
    strict_lengths: bool = True,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compatibility spelling for :func:`process_h5_groups`."""
    return process_h5_groups(
        path,
        group_id,
        group_selector=group_selector,
        group_name=group_name,
        mode=mode,
        output_dir=output_dir,
        save_csv=save_csv,
        strict_lengths=strict_lengths,
        verbose=verbose,
    )


def print_h5_groups(path: PathInput) -> list[dict[str, Any]]:
    """Print available PSD groups and return their metadata."""
    rows = list_h5_groups(path)
    print(f"HDF5 file: {_as_path(path)}")
    print(f"groups: {len(rows)}")
    for row in rows:
        print(
            f"group={row['group_id']} | points={row['points']} | "
            f"frequency=[{row['frequency_min_hz']:.9g}, {row['frequency_max_hz']:.9g}] Hz | "
            f"label={row['label'] or '<empty>'} | "
            f"xy_stddev={'yes' if row['has_xy_stddev'] else 'no'} | "
            f"rstddev={'yes' if row['has_rstddev'] else 'no'} | "
            f"rpwr={'yes' if row['has_rpwr'] else 'no'} | "
            f"path={row['sample_path']}"
        )
    return rows


