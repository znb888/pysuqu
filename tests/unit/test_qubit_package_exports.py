import unittest

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu import qubit
from pysuqu.qubit.analysis import SingleQubitSpectrum, analyze_single_qubit_spectrum
from pysuqu.qubit.base import AbstractQubit, ParameterizedQubit, Phi0, pi
from pysuqu.qubit.gate import (
    ChannelSchedule,
    EnvelopeParams,
    GateBase,
    MixerParams,
    PulseEvent,
    SingleQubitGate,
    WaveformGenerator,
)
from pysuqu.qubit.multi import FGF1V1Coupling, FGF2V7Coupling, QCRFGRModel
from pysuqu.qubit.single import FloatingTransmon, GroundedTransmon, SingleQubitBase
from pysuqu.qubit.solver import HamiltonianEvo
from pysuqu.qubit.types import FluxSpec, SpectrumResult


class QubitPackageExportTests(unittest.TestCase):
    def test_package_exports_stable_public_api(self):
        self.assertIs(qubit.AbstractQubit, AbstractQubit)
        self.assertIs(qubit.ChannelSchedule, ChannelSchedule)
        self.assertIs(qubit.EnvelopeParams, EnvelopeParams)
        self.assertIs(qubit.FGF1V1Coupling, FGF1V1Coupling)
        self.assertIs(qubit.FGF2V7Coupling, FGF2V7Coupling)
        self.assertIs(qubit.FloatingTransmon, FloatingTransmon)
        self.assertIs(qubit.FluxSpec, FluxSpec)
        self.assertIs(qubit.GateBase, GateBase)
        self.assertIs(qubit.GroundedTransmon, GroundedTransmon)
        self.assertIs(qubit.HamiltonianEvo, HamiltonianEvo)
        self.assertIs(qubit.MixerParams, MixerParams)
        self.assertIs(qubit.ParameterizedQubit, ParameterizedQubit)
        self.assertEqual(qubit.Phi0, Phi0)
        self.assertIs(qubit.PulseEvent, PulseEvent)
        self.assertIs(qubit.QCRFGRModel, QCRFGRModel)
        self.assertIs(qubit.SingleQubitGate, SingleQubitGate)
        self.assertIs(qubit.SingleQubitBase, SingleQubitBase)
        self.assertIs(qubit.SingleQubitSpectrum, SingleQubitSpectrum)
        self.assertIs(qubit.SpectrumResult, SpectrumResult)
        self.assertIs(qubit.WaveformGenerator, WaveformGenerator)
        self.assertIs(qubit.analyze_single_qubit_spectrum, analyze_single_qubit_spectrum)
        self.assertEqual(qubit.pi, pi)

    def test_package_all_matches_documented_exports(self):
        self.assertEqual(
            qubit.__all__,
            [
                'AbstractQubit',
                'ChannelSchedule',
                'EnvelopeParams',
                'FGF1V1Coupling',
                'FGF2V7Coupling',
                'FloatingTransmon',
                'FluxSpec',
                'GateBase',
                'GroundedTransmon',
                'HamiltonianEvo',
                'MixerParams',
                'ParameterizedQubit',
                'Phi0',
                'PulseEvent',
                'QCRFGRModel',
                'SingleQubitGate',
                'SingleQubitBase',
                'SingleQubitSpectrum',
                'SpectrumResult',
                'WaveformGenerator',
                'analyze_single_qubit_spectrum',
                'pi',
            ],
        )


if __name__ == '__main__':
    unittest.main()

