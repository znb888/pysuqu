import unittest

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu import qubit
from pysuqu.funclib.transmission import (
    AttenuatorStage,
    BundleTransmissionChain,
    BundleTransmissionResult,
    DelayStage,
    DerivativePrecorrectionDesign,
    DerivativePrecorrectionStage,
    FIRFilterStage,
    IIRFilterStage,
    MIMOTouchstoneStage,
    SOSFilterStage,
    SignalBundle,
    SignalTrace,
    TouchstoneNetwork,
    TouchstoneStage,
    TransferFunctionStage,
    TransmissionChain,
    TransmissionResult,
    apply_derivative_precorrection,
    compute_derivative_basis,
    design_derivative_precorrection,
    design_inverse_fir_from_touchstone,
    evaluate_touchstone_response,
    load_touchstone_network,
)
from pysuqu.qubit.analysis import SingleQubitSpectrum, analyze_single_qubit_spectrum
from pysuqu.qubit.base import AbstractQubit, ParameterizedQubit, Phi0, pi
from pysuqu.qubit.circuit import (
    TransmonReflectionModel,
    calculate_loaded_single_port_response,
    resolve_load_reflection_response,
)
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
        self.assertIs(qubit.AttenuatorStage, AttenuatorStage)
        self.assertIs(qubit.BundleTransmissionChain, BundleTransmissionChain)
        self.assertIs(qubit.BundleTransmissionResult, BundleTransmissionResult)
        self.assertIs(qubit.ChannelSchedule, ChannelSchedule)
        self.assertIs(qubit.DelayStage, DelayStage)
        self.assertIs(qubit.DerivativePrecorrectionDesign, DerivativePrecorrectionDesign)
        self.assertIs(qubit.DerivativePrecorrectionStage, DerivativePrecorrectionStage)
        self.assertIs(qubit.EnvelopeParams, EnvelopeParams)
        self.assertIs(qubit.FGF1V1Coupling, FGF1V1Coupling)
        self.assertIs(qubit.FGF2V7Coupling, FGF2V7Coupling)
        self.assertIs(qubit.FIRFilterStage, FIRFilterStage)
        self.assertIs(qubit.FloatingTransmon, FloatingTransmon)
        self.assertIs(qubit.FluxSpec, FluxSpec)
        self.assertIs(qubit.GateBase, GateBase)
        self.assertIs(qubit.GroundedTransmon, GroundedTransmon)
        self.assertIs(qubit.HamiltonianEvo, HamiltonianEvo)
        self.assertIs(qubit.IIRFilterStage, IIRFilterStage)
        self.assertIs(qubit.MIMOTouchstoneStage, MIMOTouchstoneStage)
        self.assertIs(qubit.MixerParams, MixerParams)
        self.assertIs(qubit.ParameterizedQubit, ParameterizedQubit)
        self.assertEqual(qubit.Phi0, Phi0)
        self.assertIs(qubit.PulseEvent, PulseEvent)
        self.assertIs(qubit.QCRFGRModel, QCRFGRModel)
        self.assertIs(qubit.SOSFilterStage, SOSFilterStage)
        self.assertIs(qubit.SignalBundle, SignalBundle)
        self.assertIs(qubit.SignalTrace, SignalTrace)
        self.assertIs(qubit.SingleQubitGate, SingleQubitGate)
        self.assertIs(qubit.SingleQubitBase, SingleQubitBase)
        self.assertIs(qubit.SingleQubitSpectrum, SingleQubitSpectrum)
        self.assertIs(qubit.SpectrumResult, SpectrumResult)
        self.assertIs(qubit.TouchstoneNetwork, TouchstoneNetwork)
        self.assertIs(qubit.TouchstoneStage, TouchstoneStage)
        self.assertIs(qubit.TransferFunctionStage, TransferFunctionStage)
        self.assertIs(qubit.TransmissionChain, TransmissionChain)
        self.assertIs(qubit.TransmissionResult, TransmissionResult)
        self.assertIs(qubit.TransmonReflectionModel, TransmonReflectionModel)
        self.assertIs(qubit.calculate_loaded_single_port_response, calculate_loaded_single_port_response)
        self.assertIs(qubit.resolve_load_reflection_response, resolve_load_reflection_response)
        self.assertIs(qubit.WaveformGenerator, WaveformGenerator)
        self.assertIs(qubit.apply_derivative_precorrection, apply_derivative_precorrection)
        self.assertIs(qubit.analyze_single_qubit_spectrum, analyze_single_qubit_spectrum)
        self.assertIs(qubit.compute_derivative_basis, compute_derivative_basis)
        self.assertIs(qubit.design_derivative_precorrection, design_derivative_precorrection)
        self.assertIs(qubit.design_inverse_fir_from_touchstone, design_inverse_fir_from_touchstone)
        self.assertIs(qubit.evaluate_touchstone_response, evaluate_touchstone_response)
        self.assertIs(qubit.load_touchstone_network, load_touchstone_network)
        self.assertEqual(qubit.pi, pi)

    def test_package_all_matches_documented_exports(self):
        self.assertEqual(
            qubit.__all__,
            [
                'AbstractQubit',
                'AttenuatorStage',
                'BundleTransmissionChain',
                'BundleTransmissionResult',
                'ChannelSchedule',
                'DelayStage',
                'DerivativePrecorrectionDesign',
                'DerivativePrecorrectionStage',
                'EnvelopeParams',
                'FGF1V1Coupling',
                'FGF2V7Coupling',
                'FIRFilterStage',
                'FloatingTransmon',
                'FluxSpec',
                'GateBase',
                'GroundedTransmon',
                'HamiltonianEvo',
                'IIRFilterStage',
                'MIMOTouchstoneStage',
                'MixerParams',
                'ParameterizedQubit',
                'Phi0',
                'PulseEvent',
                'QCRFGRModel',
                'SOSFilterStage',
                'SignalBundle',
                'SignalTrace',
                'SingleQubitGate',
                'SingleQubitBase',
                'SingleQubitSpectrum',
                'SpectrumResult',
                'TouchstoneNetwork',
                'TouchstoneStage',
                'TransferFunctionStage',
                'TransmissionChain',
                'TransmissionResult',
                'TransmonReflectionModel',
                'WaveformGenerator',
                'apply_derivative_precorrection',
                'analyze_single_qubit_spectrum',
                'calculate_loaded_single_port_response',
                'compute_derivative_basis',
                'design_derivative_precorrection',
                'design_inverse_fir_from_touchstone',
                'evaluate_touchstone_response',
                'load_touchstone_network',
                'pi',
                'resolve_load_reflection_response',
            ],
        )


if __name__ == '__main__':
    unittest.main()
