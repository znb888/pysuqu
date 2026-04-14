import inspect
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from qutip import basis, tensor

from pysuqu.qubit import multi as multi_module
from pysuqu.qubit.analysis import (
    analyze_multi_qubit_coupler_sensitivity,
    analyze_multi_qubit_coupler_sensitivity_result,
    calculate_multi_qubit_coupler_self_sensitivity,
    calculate_multi_qubit_sensitivity_analytical,
    find_multi_qubit_coupler_detune,
    get_multi_qubit_frequency_at_coupler_flux,
    plot_multi_qubit_sensitivity_curve,
)
from pysuqu.qubit.base import ParameterizedQubit
from pysuqu.qubit.multi import FGF1V1Coupling, FGF2V7Coupling, QCRFGRModel
from pysuqu.qubit.types import CouplingResult, SensitivityResult, SpectrumResult


class FakeHamiltonian:
    def __init__(self, dims, qubit_frequency_ghz=6.35):
        self.dims = dims
        self.qubit_frequency_ghz = qubit_frequency_ghz

    def eigenstates(self):
        ground = tensor(basis(3, 0), basis(3, 0))
        qubit_excited = tensor(basis(3, 1), basis(3, 0))
        coupler_excited = tensor(basis(3, 0), basis(3, 1))
        return (
            np.array([0.0, 2 * np.pi * self.qubit_frequency_ghz, 2 * np.pi * 9.0]),
            [ground, qubit_excited, coupler_excited],
        )


class MultiQubitAnalysisTests(unittest.TestCase):
    def test_qcrfgr_class_source_no_longer_defines_inline_coupler_sensitivity_methods(self):
        class_source = inspect.getsource(QCRFGRModel)
        self.assertNotIn('def cal_coupler_sensitivity', class_source)
        self.assertNotIn('def _cal_sensitivity_numerical', class_source)
        self.assertNotIn('def _cal_sensitivity_analytical', class_source)
        self.assertNotIn('def _cal_coupler_self_sensitivity', class_source)
        self.assertNotIn('def _get_qubit_frequency_at_coupler_flux', class_source)
        self.assertNotIn('def _plot_sensitivity_curve', class_source)

    def test_fgf1_class_source_no_longer_defines_inline_coupler_sensitivity_methods(self):
        class_source = inspect.getsource(FGF1V1Coupling)
        self.assertNotIn('def cal_coupler_sensitivity', class_source)
        self.assertNotIn('def _cal_sensitivity_numerical', class_source)
        self.assertNotIn('def _cal_sensitivity_analytical', class_source)
        self.assertNotIn('def _cal_coupler_self_sensitivity', class_source)
        self.assertNotIn('def _get_qubit_frequency_at_coupler_flux', class_source)
        self.assertNotIn('def _plot_sensitivity_curve', class_source)

    def test_qcrfgr_print_basic_info_is_display_only(self):
        model = QCRFGRModel.__new__(QCRFGRModel)
        model.qubit_f01 = 5.0
        model.coupler_f01 = 6.1
        model.qubit_anharm = -0.25
        model.rq_g = 2 * np.pi * 12e6
        model.qc_g = 0.018
        model._refresh_basic_metrics = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh metrics')
        )
        model.get_readout_couple = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh readout coupling')
        )
        model.get_coupler_couple = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh coupler coupling')
        )

        QCRFGRModel.print_basic_info(model, is_print=False)

        model._refresh_basic_metrics.assert_not_called()
        model.get_readout_couple.assert_not_called()
        model.get_coupler_couple.assert_not_called()
        self.assertAlmostEqual(model.qubit_f01, 5.0)
        self.assertAlmostEqual(model.coupler_f01, 6.1)
        self.assertAlmostEqual(model.qubit_anharm, -0.25)
        self.assertAlmostEqual(model.rq_g, 2 * np.pi * 12e6)
        self.assertAlmostEqual(model.qc_g, 0.018)

    def test_fgf1_print_basic_info_is_display_only(self):
        model = FGF1V1Coupling.__new__(FGF1V1Coupling)
        model.qubit1_f01 = 5.0
        model.qubit2_f01 = 5.2
        model.coupler_f01 = 6.1
        model.qubit_anharm = -0.24
        model.qr_g = 2 * np.pi * 13e6
        model.qq_g = 0.011
        model.qc_g = 0.017
        model.qq_geff = 0.002
        model._refresh_basic_metrics = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh metrics')
        )
        model.get_readout_couple = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh readout coupling')
        )
        model.get_qq_dcouple = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh qq coupling')
        )
        model.get_qc_couple = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh qc coupling')
        )
        model.get_qq_ecouple = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh effective coupling')
        )

        FGF1V1Coupling.print_basic_info(model, is_print=False)

        model._refresh_basic_metrics.assert_not_called()
        model.get_readout_couple.assert_not_called()
        model.get_qq_dcouple.assert_not_called()
        model.get_qc_couple.assert_not_called()
        model.get_qq_ecouple.assert_not_called()
        self.assertAlmostEqual(model.qubit1_f01, 5.0)
        self.assertAlmostEqual(model.qubit2_f01, 5.2)
        self.assertAlmostEqual(model.coupler_f01, 6.1)
        self.assertAlmostEqual(model.qubit_anharm, -0.24)
        self.assertAlmostEqual(model.qr_g, 2 * np.pi * 13e6)
        self.assertAlmostEqual(model.qq_g, 0.011)
        self.assertAlmostEqual(model.qc_g, 0.017)
        self.assertAlmostEqual(model.qq_geff, 0.002)

    def test_fgf2_print_basic_info_is_display_only(self):
        model = FGF2V7Coupling.__new__(FGF2V7Coupling)
        model.qubit1_f01 = 5.0
        model.qubit2_f01 = 5.2
        model.qubit_f01 = 5.1
        model.coupler_f01 = 6.1
        model.qubit1_anharm = -0.23
        model.qubit2_anharm = -0.25
        model.qubit_anharm = -0.24
        model._refresh_basic_metrics = mock.Mock(
            side_effect=AssertionError('print_basic_info should not refresh metrics')
        )

        FGF2V7Coupling.print_basic_info(model, is_print=False)

        model._refresh_basic_metrics.assert_not_called()
        self.assertAlmostEqual(model.qubit1_f01, 5.0)
        self.assertAlmostEqual(model.qubit2_f01, 5.2)
        self.assertAlmostEqual(model.qubit_f01, 5.1)
        self.assertAlmostEqual(model.coupler_f01, 6.1)
        self.assertAlmostEqual(model.qubit1_anharm, -0.23)
        self.assertAlmostEqual(model.qubit2_anharm, -0.25)
        self.assertAlmostEqual(model.qubit_anharm, -0.24)

    def test_qcrfgr_get_readout_couple_rejects_unsupported_mode(self):
        model = QCRFGRModel.__new__(QCRFGRModel)

        with self.assertRaisesRegex(ValueError, 'Unsupported couple_mode: induc'):
            QCRFGRModel.get_readout_couple(
                model,
                readout_freq=6.5e9,
                couple_mode='induc',
                is_print=False,
            )

    def test_fgf1_get_readout_couple_rejects_unsupported_mode_without_mutating_cache(self):
        model = FGF1V1Coupling.__new__(FGF1V1Coupling)
        model.qr_g = 123.0

        with self.assertRaisesRegex(ValueError, 'Unsupported couple_mode: induc'):
            FGF1V1Coupling.get_readout_couple(
                model,
                readout_freq=6.5e9,
                couple_mode='induc',
                is_print=False,
            )

        self.assertEqual(model.qr_g, 123.0)

    def test_qcrfgr_constructor_refreshes_metrics_before_display(self):
        call_order = []

        with mock.patch.object(ParameterizedQubit, '__init__', return_value=None) as super_init, mock.patch.object(
            QCRFGRModel,
            '_refresh_basic_metrics',
            autospec=True,
            side_effect=lambda instance: call_order.append('refresh'),
        ) as refresh, mock.patch.object(
            QCRFGRModel,
            'print_basic_info',
            autospec=True,
            side_effect=lambda instance, is_print=True: call_order.append(('print', is_print)),
        ) as print_basic_info:
            QCRFGRModel(
                capacitance_list=[1.0, 2.0, 3.0, 4.0, 5.0],
                junc_resis_list=[6.0, 7.0],
                qrcouple=[8.0, 9.0],
                flux_list=[0.1, 0.2, 0.3],
                trunc_ener_level=[10, 8, 10],
            )

        super_init.assert_called_once()
        refresh.assert_called_once()
        print_basic_info.assert_called_once()
        self.assertEqual(call_order, ['refresh', ('print', True)])

    def test_fgf1_constructor_refreshes_metrics_before_display(self):
        call_order = []

        with mock.patch.object(ParameterizedQubit, '__init__', return_value=None) as super_init, mock.patch.object(
            FGF1V1Coupling,
            '_refresh_basic_metrics',
            autospec=True,
            side_effect=lambda instance: call_order.append('refresh'),
        ) as refresh, mock.patch.object(
            FGF1V1Coupling,
            'print_basic_info',
            autospec=True,
            side_effect=lambda instance, is_print=True: call_order.append(('print', is_print)),
        ) as print_basic_info:
            FGF1V1Coupling(
                capacitance_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                junc_resis_list=[11.0, 12.0, 13.0],
                qrcouple=[14.0, 15.0],
                flux_list=[0.1, 0.2, 0.3],
                trunc_ener_level=[10, 8, 10],
                is_print=False,
            )

        super_init.assert_called_once()
        refresh.assert_called_once()
        print_basic_info.assert_called_once()
        self.assertEqual(call_order, ['refresh', ('print', False)])

    def test_fgf2_constructor_refreshes_metrics_before_display(self):
        call_order = []

        with mock.patch.object(ParameterizedQubit, '__init__', return_value=None) as super_init, mock.patch.object(
            FGF2V7Coupling,
            '_refresh_basic_metrics',
            autospec=True,
            side_effect=lambda instance: call_order.append('refresh'),
        ) as refresh, mock.patch.object(
            FGF2V7Coupling,
            'print_basic_info',
            autospec=True,
            side_effect=lambda instance, is_print=True: call_order.append(('print', is_print)),
        ) as print_basic_info:
            FGF2V7Coupling(
                capacitance_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                junc_resis_list=[11.0, 12.0],
                flux_list=[0.1, 0.2],
                trunc_ener_level=[10, 8],
            )

        super_init.assert_called_once()
        refresh.assert_called_once()
        print_basic_info.assert_called_once()
        self.assertEqual(call_order, ['refresh', ('print', True)])

    def make_qcrfgr_like_model(self):
        model = QCRFGRModel.__new__(QCRFGRModel)
        model._flux = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.0, 0.2],
            ],
            dtype=float,
        )
        model._junc_ratio = np.ones_like(model._flux)
        model.SMatrix_retainNodes = [0, 2]
        model._ParameterizedQubit__struct = [2, 1]
        model._ParameterizedQubit__nodes = 3
        model._ParameterizedQubit__capac = np.eye(3)
        model._ParameterizedQubit__resis = np.eye(3)
        model._ParameterizedQubit__induc = np.ones((3, 3))
        model.Ec = np.diag([1.0, 2.0])
        model.El = np.diag([4.0, 5.0])
        model.Ejmax = np.diag([7.0, 8.0])
        baseline_hamiltonian = FakeHamiltonian([[3, 3], [3, 3]], qubit_frequency_ghz=6.2)
        baseline_eigenvalues, baseline_eigenstates = baseline_hamiltonian.eigenstates()
        model._hamiltonian = baseline_hamiltonian
        model._Hamiltonian = baseline_hamiltonian
        model._energylevels = baseline_eigenvalues
        model._eigenstates = baseline_eigenstates
        model._numQubits = 2
        model._Nlevel = [3, 3]
        model._cal_mode = 'Eigen'
        model._charges = np.array([0, 0])
        model._generate_hamiltonian = mock.Mock(
            return_value=FakeHamiltonian([[3, 3], [3, 3]], qubit_frequency_ghz=6.35)
        )
        model._refresh_basic_metrics = mock.Mock(side_effect=lambda: setattr(model, 'qubit_f01', 6.0 + model._flux[2, 2]))
        model.eigenHamiltonian = 'baseline-eigen'
        model.couplingHamiltonian = 'baseline-coupling'
        model.highorderHamiltonian = 'baseline-highorder'
        model.destroyors = ['baseline-destroyor']
        model.n_operators = ['baseline-number']
        model.phi_operators = ['baseline-phase']
        model._solver_result = SpectrumResult(
            hamiltonian=baseline_hamiltonian,
            eigenvalues=np.array(baseline_eigenvalues, copy=True),
            eigenstates=list(baseline_eigenstates),
            destroy_operators=model.destroyors,
            number_operators=model.n_operators,
            phase_operators=model.phi_operators,
        )
        model.change_para = ParameterizedQubit.change_para.__get__(model, QCRFGRModel)
        return model

    def test_frequency_probe_restores_flux_state_via_analysis_helper(self):
        multi_module._clear_qcrfgr_probe_frequency_cache()
        model = self.make_qcrfgr_like_model()
        original_flux = model._flux.copy()
        original_solver_result = model.solver_result
        original_destroyors = model.destroyors
        original_numbers = model.n_operators
        original_phases = model.phi_operators

        frequency = get_multi_qubit_frequency_at_coupler_flux(model, 0.35, qubit_idx=0)

        self.assertAlmostEqual(frequency, 6.35)
        np.testing.assert_allclose(model._flux, original_flux)
        self.assertEqual(model._generate_hamiltonian.call_count, 1)
        self.assertEqual(model._generate_hamiltonian.call_args.kwargs, {'transient': True})
        self.assertEqual(model._refresh_basic_metrics.call_count, 0)
        self.assertIs(model.solver_result, original_solver_result)
        self.assertIs(model.destroyors, original_destroyors)
        self.assertIs(model.n_operators, original_numbers)
        self.assertIs(model.phi_operators, original_phases)
        multi_module._clear_qcrfgr_probe_frequency_cache()

    def test_frequency_probe_reuses_identical_fast_path_result_for_same_inputs(self):
        multi_module._clear_qcrfgr_probe_frequency_cache()
        model = self.make_qcrfgr_like_model()

        first = get_multi_qubit_frequency_at_coupler_flux(model, 0.35, qubit_idx=0)
        second = get_multi_qubit_frequency_at_coupler_flux(model, 0.35, qubit_idx=0)

        self.assertAlmostEqual(first, 6.35)
        self.assertAlmostEqual(second, first)
        self.assertEqual(model._generate_hamiltonian.call_count, 1)
        self.assertEqual(model._refresh_basic_metrics.call_count, 0)
        multi_module._clear_qcrfgr_probe_frequency_cache()

    def test_qcrfgr_wrapper_delegates_sensitivity_analysis(self):
        model = QCRFGRModel.__new__(QCRFGRModel)

        with mock.patch(
            'pysuqu.qubit.multi.analyze_multi_qubit_coupler_sensitivity',
            return_value=0.125,
            create=True,
        ) as analyze:
            sensitivity = QCRFGRModel.cal_coupler_sensitivity(
                model,
                coupler_flux_point=0.2,
                method='numerical',
                flux_step=1e-3,
                qubit_idx=0,
                is_print=False,
            )

        self.assertEqual(sensitivity, 0.125)
        analyze.assert_called_once_with(
            model,
            coupler_flux_point=0.2,
            method='numerical',
            flux_step=1e-3,
            qubit_idx=0,
            qubit_fluxes=None,
            is_print=False,
            is_plot=False,
        )

    def test_fgf1_wrapper_passes_qubit_fluxes_to_analysis(self):
        model = FGF1V1Coupling.__new__(FGF1V1Coupling)

        with mock.patch(
            'pysuqu.qubit.multi.analyze_multi_qubit_coupler_sensitivity',
            return_value=0.25,
            create=True,
        ) as analyze:
            sensitivity = FGF1V1Coupling.cal_coupler_sensitivity(
                model,
                coupler_flux_point=0.3,
                method='numerical',
                flux_step=2e-4,
                qubit_idx=1,
                qubit_fluxes=[0.11, 0.22],
                is_print=False,
            )

        self.assertEqual(sensitivity, 0.25)
        analyze.assert_called_once_with(
            model,
            coupler_flux_point=0.3,
            method='numerical',
            flux_step=2e-4,
            qubit_idx=1,
            qubit_fluxes=[0.11, 0.22],
            is_print=False,
            is_plot=False,
        )

    def test_structured_sensitivity_helper_returns_result_with_metadata(self):
        model = SimpleNamespace()

        with mock.patch(
            'pysuqu.qubit.analysis.calculate_multi_qubit_sensitivity_numerical',
            return_value=np.float64(0.125),
        ) as calculate, mock.patch(
            'pysuqu.qubit.analysis.plot_multi_qubit_sensitivity_curve',
        ) as plot:
            result = analyze_multi_qubit_coupler_sensitivity_result(
                model,
                coupler_flux_point=0.3,
                method='numerical',
                flux_step=2e-4,
                qubit_idx=1,
                qubit_fluxes=[0.11, 0.22],
                is_print=False,
                is_plot=False,
            )

        self.assertIsInstance(result, SensitivityResult)
        self.assertEqual(result.coupler_flux_point, 0.3)
        self.assertEqual(result.sensitivity_value, 0.125)
        self.assertEqual(result.metadata['method'], 'numerical')
        self.assertEqual(result.metadata['flux_step'], 2e-4)
        self.assertEqual(result.metadata['qubit_idx'], 1)
        self.assertEqual(result.metadata['qubit_fluxes'], [0.11, 0.22])
        calculate.assert_called_once_with(
            model,
            0.3,
            2e-4,
            1,
            qubit_fluxes=[0.11, 0.22],
        )
        plot.assert_not_called()

    def test_structured_sensitivity_helper_passes_result_directly_to_plotting(self):
        model = SimpleNamespace()

        with mock.patch(
            'pysuqu.qubit.analysis.calculate_multi_qubit_sensitivity_numerical',
            return_value=np.float64(0.125),
        ) as calculate, mock.patch(
            'pysuqu.qubit.analysis.plot_multi_qubit_sensitivity_curve',
        ) as plot:
            result = analyze_multi_qubit_coupler_sensitivity_result(
                model,
                coupler_flux_point=0.3,
                method='numerical',
                flux_step=2e-4,
                qubit_idx=1,
                qubit_fluxes=[0.11, 0.22],
                is_print=False,
                is_plot=True,
            )

        calculate.assert_called_once_with(
            model,
            0.3,
            2e-4,
            1,
            qubit_fluxes=[0.11, 0.22],
        )
        plot.assert_called_once()
        self.assertIs(plot.call_args.args[4], result)
        self.assertEqual(plot.call_args.args[:4], (model, 0.3, 2e-4, 1))

    def test_plot_multi_qubit_sensitivity_curve_accepts_sensitivity_result(self):
        model = SimpleNamespace()
        freq_values = np.linspace(5.0, 5.19, 20)
        result = SensitivityResult(
            coupler_flux_point=0.3,
            sensitivity_value=0.125,
            metadata={'method': 'numerical'},
        )

        with mock.patch(
            'pysuqu.qubit.analysis.get_multi_qubit_frequency_at_coupler_flux',
            side_effect=freq_values.tolist(),
        ) as get_frequency, mock.patch('pysuqu.qubit.analysis.plt') as plt:
            plot_multi_qubit_sensitivity_curve(
                model,
                coupler_flux_point=0.3,
                flux_step=2e-4,
                qubit_idx=1,
                sensitivity=result,
            )

        expected_flux_range = np.linspace(0.299, 0.301, 20)
        expected_tangent = freq_values[len(expected_flux_range) // 2] + 0.125 * (
            expected_flux_range - 0.3
        )

        self.assertEqual(get_frequency.call_count, 20)
        self.assertEqual(plt.plot.call_count, 2)
        np.testing.assert_allclose(plt.plot.call_args_list[0].args[0], expected_flux_range)
        np.testing.assert_allclose(plt.plot.call_args_list[0].args[1], freq_values)
        np.testing.assert_allclose(plt.plot.call_args_list[1].args[0], expected_flux_range)
        np.testing.assert_allclose(plt.plot.call_args_list[1].args[1], expected_tangent)
        self.assertIn('slope=0.125', plt.plot.call_args_list[1].kwargs['label'])

    def test_public_sensitivity_helper_returns_structured_result(self):
        model = SimpleNamespace()

        with mock.patch(
            'pysuqu.qubit.analysis.calculate_multi_qubit_sensitivity_analytical',
            return_value=np.float64(0.05),
        ) as analyze:
            sensitivity = analyze_multi_qubit_coupler_sensitivity(
                model,
                coupler_flux_point=0.2,
                method='analytical',
                flux_step=1e-4,
                qubit_idx=None,
                qubit_fluxes=None,
                is_print=False,
                is_plot=False,
            )

        self.assertIsInstance(sensitivity, SensitivityResult)
        self.assertEqual(sensitivity.coupler_flux_point, 0.2)
        self.assertEqual(sensitivity.sensitivity_value, 0.05)
        self.assertEqual(sensitivity.metadata['method'], 'analytical')
        analyze.assert_called_once_with(model, 0.2, None)

    def test_find_multi_qubit_coupler_detune_interpolates_target_flux(self):
        with mock.patch('builtins.print') as print_mock:
            target_flux = find_multi_qubit_coupler_detune(
                g_list=[2.0, 1.0, 0.0],
                flux_list=[0.1, 0.2, 0.3],
                coupler_strength=0.5,
            )

        self.assertAlmostEqual(target_flux, 0.25)
        print_mock.assert_called_once_with('Flux when g=0.5: 0.25 Phi0')

    def test_find_multi_qubit_coupler_detune_rejects_non_bracketed_target(self):
        for coupler_strength in (2.5, -0.5):
            with self.subTest(coupler_strength=coupler_strength):
                with self.assertRaisesRegex(ValueError, 'not bracketed'):
                    find_multi_qubit_coupler_detune(
                        g_list=[2.0, 1.0, 0.0],
                        flux_list=[0.1, 0.2, 0.3],
                        coupler_strength=coupler_strength,
                    )

    def test_find_multi_qubit_coupler_detune_accepts_coupling_result(self):
        result = CouplingResult(
            sweep_parameter='coupler_flux',
            sweep_values=[0.1, 0.2, 0.3],
            coupling_values=[2.0, 1.0, 0.0],
            metadata={'method': 'ES'},
        )

        with mock.patch('builtins.print') as print_mock:
            target_flux = find_multi_qubit_coupler_detune(
                g_list=result,
                flux_list=None,
                coupler_strength=0.5,
            )

        self.assertAlmostEqual(target_flux, 0.25)
        print_mock.assert_called_once_with('Flux when g=0.5: 0.25 Phi0')

    def test_analytical_sensitivity_warning_uses_delta_label(self):
        model = SimpleNamespace(
            qc_g=0.2,
            coupler_f01=1.1,
            qubit_f01=0.1,
        )

        with mock.patch(
            'pysuqu.qubit.analysis.calculate_multi_qubit_coupler_self_sensitivity',
            return_value=2.0,
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sensitivity = calculate_multi_qubit_sensitivity_analytical(
                model,
                coupler_flux_point=0.25,
                qubit_idx=None,
            )

        self.assertAlmostEqual(sensitivity, 0.08)
        self.assertEqual(len(caught), 1)
        self.assertIn('g/Delta', str(caught[0].message))

    def test_coupler_self_sensitivity_uses_ejmax_instead_of_flux_biased_ej(self):
        model = SimpleNamespace(
            Ec=np.diag([1.0, 2.0]),
            Ejmax=np.diag([7.0, 8.0]),
            Ej=np.diag([7.0, 2.0]),
        )

        sensitivity = calculate_multi_qubit_coupler_self_sensitivity(model, coupler_flux=0.25)

        ec_ghz = model.Ec[1, 1]
        ejmax_ghz = model.Ejmax[1, 1]
        cos_term = np.cos(np.pi * 0.25)
        sin_term = np.sin(np.pi * 0.25)
        ej = ejmax_ghz * np.abs(cos_term)
        dej_dphi = -ejmax_ghz * np.pi * sin_term * np.sign(cos_term)
        expected = (2 * ec_ghz / np.sqrt(8 * ec_ghz * ej)) * dej_dphi

        self.assertAlmostEqual(sensitivity, expected)

if __name__ == '__main__':
    unittest.main()
