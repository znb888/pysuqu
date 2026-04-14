import unittest

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.types import (
    CouplingResult,
    FluxSpec,
    FluxState,
    SensitivityResult,
    SpectrumResult,
    SweepResult,
)


class QubitTypesModuleTests(unittest.TestCase):
    def test_types_module_exports_structured_objects(self):
        self.assertTrue(hasattr(FluxSpec, 'from_full'))
        self.assertTrue(hasattr(FluxState, 'update_from_full'))
        self.assertEqual(SpectrumResult.__name__, 'SpectrumResult')

    def test_flux_spec_and_flux_state_keep_dual_representation_behavior(self):
        reduced = np.diag([0.125, 0.25])
        spec = FluxSpec.from_reduced(reduced, struct=[1, 2], nodes=3)

        np.testing.assert_allclose(spec.reduced, reduced)
        np.testing.assert_allclose(
            spec.full,
            np.array(
                [
                    [0.125, 0.0, 0.0],
                    [0.0, 0.0, 0.25],
                    [0.0, 0.25, 0.0],
                ]
            ),
        )

        state = FluxState()
        state.update_from_full(spec.full, struct=[1, 2], nodes=3)

        np.testing.assert_allclose(state.full, spec.full)
        np.testing.assert_allclose(state.reduced, reduced)

    def test_sweep_result_preserves_series(self):
        result = SweepResult(
            sweep_parameter='coupler_flux',
            sweep_values=[0.125, 0.25],
            series={
                '|[0, 0, 1]>': [1.0, 2.5],
                '|[1, 0, 0]>': np.array([1.5, 3.0]),
            },
            metadata={'qubits_flux': None},
        )

        self.assertEqual(result.sweep_parameter, 'coupler_flux')
        self.assertEqual(result.sweep_values, [0.125, 0.25])
        self.assertIsNone(result.metadata['qubits_flux'])
        np.testing.assert_allclose(result.series['|[0, 0, 1]>'], np.array([1.0, 2.5]))
        np.testing.assert_allclose(result.series['|[1, 0, 0]>'], np.array([1.5, 3.0]))

    def test_coupling_result_preserves_values(self):
        result = CouplingResult(
            sweep_parameter='coupler_flux',
            sweep_values=[0.125, 0.25],
            coupling_values=np.array([1.0, -0.25]),
            metadata={'method': 'ES'},
        )

        self.assertEqual(result.sweep_parameter, 'coupler_flux')
        self.assertEqual(result.sweep_values, [0.125, 0.25])
        self.assertEqual(result.metadata['method'], 'ES')
        np.testing.assert_allclose(result.coupling_values, np.array([1.0, -0.25]))

    def test_sensitivity_result_preserves_value(self):
        result = SensitivityResult(
            coupler_flux_point=0.125,
            sensitivity_value=np.float64(0.0125),
            metadata={
                'method': 'numerical',
                'flux_step': 1e-4,
                'qubit_idx': 1,
                'qubit_fluxes': [0.1, 0.2],
            },
        )

        self.assertEqual(result.coupler_flux_point, 0.125)
        self.assertEqual(result.sensitivity_value, 0.0125)
        self.assertEqual(result.metadata['method'], 'numerical')
        self.assertEqual(result.metadata['flux_step'], 1e-4)
        self.assertEqual(result.metadata['qubit_idx'], 1)
        self.assertEqual(result.metadata['qubit_fluxes'], [0.1, 0.2])


if __name__ == '__main__':
    unittest.main()
