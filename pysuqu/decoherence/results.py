"""Minimum structured result drafts for the decoherence refactor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np


def _copy_mapping(mapping: Mapping[str, object] | None) -> dict[str, object]:
    if mapping is None:
        return {}

    copied = dict(mapping)
    for key, value in list(copied.items()):
        if isinstance(value, Mapping):
            copied[key] = dict(value)
    return copied


def _normalize_scalar_result(instance) -> None:
    object.__setattr__(instance, 'value', float(instance.value))
    object.__setattr__(instance, 'unit', str(instance.unit))
    object.__setattr__(instance, 'metadata', _copy_mapping(instance.metadata))
    object.__setattr__(instance, 'fit_diagnostics', _copy_mapping(instance.fit_diagnostics))


def _copy_array_value(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)

    if isinstance(value, np.generic):
        return value.item()

    return value


@dataclass(frozen=True)
class TphiResult:
    """Structured draft for decoherence Tphi contracts."""

    value: float
    unit: str = 's'
    metadata: dict[str, object] = field(default_factory=dict)
    fit_diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_scalar_result(self)


@dataclass(frozen=True)
class T1Result:
    """Structured XY/readout relaxation result in seconds."""

    value: float
    unit: str = 's'
    metadata: dict[str, object] = field(default_factory=dict)
    fit_diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_scalar_result(self)


class BiasCurrentVoltageResult(TypedDict):
    """Stable public mapping contract for Z-bias current/voltage summaries."""

    phi_bias: float
    chip_current_uA: float
    chip_voltage_mV: float
    total_attenuation_dB: float
    room_current_mA: float
    room_voltage_mV: float
    room_power_dBm: float


class XYCurrentVoltageResult(TypedDict):
    """Stable public mapping contract for XY current/voltage summaries."""

    phi_bias: float
    chip_current_uA: float
    chip_voltage_uV: float
    chip_power_dBm: float
    total_attenuation_dB: float
    room_current_mA: float
    room_voltage_mV: float
    room_power_dBm: float


@dataclass(frozen=True)
class NoiseFitResult:
    """Structured draft for electronic noise fit contracts."""

    value: float
    unit: str = 'A^2/Hz'
    metadata: dict[str, object] = field(default_factory=dict)
    fit_diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_scalar_result(self)

    @classmethod
    def from_fit_dict(
        cls,
        fit_data: Mapping[str, object],
        *,
        noise_type: str,
        noise_prop: str,
        unit: str = 'A^2/Hz',
        metadata: Mapping[str, object] | None = None,
    ) -> 'NoiseFitResult':
        required_keys = (
            'white_noise',
            '1f_coef',
            'corner_freq',
            'white_ref_freq',
            'white_noise_temperature',
        )
        missing_keys = [key for key in required_keys if key not in fit_data]
        if missing_keys:
            missing_str = ', '.join(missing_keys)
            raise ValueError(f'fit_data is missing required keys: {missing_str}')

        result_metadata = {
            'noise_type': noise_type,
            'noise_prop': noise_prop,
        }
        if metadata is not None:
            result_metadata.update(dict(metadata))

        diagnostics = {key: fit_data[key] for key in required_keys if key != 'white_noise'}

        return cls(
            value=fit_data['white_noise'],
            unit=unit,
            metadata=result_metadata,
            fit_diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class NoisePipelineStage:
    """Immutable snapshot for one electronic-noise pipeline stage.

    `frequency` is expressed in Hz. For spectral workflows, `psd_double`,
    `psd_single`, and `psd_smooth` are in `A^2/Hz`, while `white_noise` keeps
    the same PSD units and `white_noise_temperature` is expressed in K.
    """

    frequency: object
    psd_double: object
    psd_single: object
    psd_smooth: object
    white_noise: object
    white_ref_freq: object
    white_noise_temperature: object
    fit_result: NoiseFitResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'frequency', _copy_array_value(self.frequency))
        object.__setattr__(self, 'psd_double', _copy_array_value(self.psd_double))
        object.__setattr__(self, 'psd_single', _copy_array_value(self.psd_single))
        object.__setattr__(self, 'psd_smooth', _copy_array_value(self.psd_smooth))
        object.__setattr__(self, 'white_noise', _copy_array_value(self.white_noise))
        object.__setattr__(self, 'white_ref_freq', _copy_array_value(self.white_ref_freq))
        object.__setattr__(
            self,
            'white_noise_temperature',
            _copy_array_value(self.white_noise_temperature),
        )


__all__ = [
    'BiasCurrentVoltageResult',
    'NoiseFitResult',
    'NoisePipelineStage',
    'T1Result',
    'TphiResult',
    'XYCurrentVoltageResult',
]
