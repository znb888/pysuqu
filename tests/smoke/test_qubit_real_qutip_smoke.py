import json
import subprocess
import sys
import textwrap
import unittest
from importlib.machinery import PathFinder
from pathlib import Path

import numpy as np


def _has_real_qutip():
    loaded_qutip = sys.modules.get('qutip')
    if loaded_qutip is not None and getattr(loaded_qutip, '__file__', None):
        return True
    return PathFinder.find_spec('qutip') is not None


HAS_REAL_QUTIP = _has_real_qutip()
REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(HAS_REAL_QUTIP, 'real qutip is not installed')
class RealQutipGateSmokeTests(unittest.TestCase):
    def _run_subprocess_and_get_payload(self, script: str):
        completed = subprocess.run(
            [sys.executable, '-c', script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f'stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}',
        )

        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, msg=f'expected JSON payload in stdout, got:\n{completed.stdout}')
        return json.loads(lines[-1])

    def test_parameterized_qubit_constructor_exposes_solver_result_with_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.base import ParameterizedQubit

            with redirect_stdout(StringIO()):
                qubit = ParameterizedQubit(
                    capacitances=[[80e-15]],
                    junctions_resistance=[[10_000]],
                    inductances=[[1e20]],
                    fluxes=[[0.125]],
                    trunc_ener_level=[3],
                    junc_ratio=[[1.2]],
                    structure_index=[1],
                )

            solver_result = qubit.solver_result
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'hamiltonian_is_qobj': isinstance(solver_result.hamiltonian, qt.Qobj),
                'hamiltonian_shape': list(solver_result.hamiltonian.shape),
                'hamiltonian_dims': solver_result.hamiltonian.dims,
                'eigenvalue_count': len(solver_result.eigenvalues),
                'first_eigenstate_is_qobj': isinstance(solver_result.eigenstates[0], qt.Qobj),
                'first_eigenstate_shape': list(solver_result.eigenstates[0].shape),
                'destroy_operator_count': len(solver_result.destroy_operators),
                'destroy_operators_are_qobj': all(
                    isinstance(operator, qt.Qobj) for operator in solver_result.destroy_operators
                ),
                'number_operator_count': len(solver_result.number_operators),
                'phase_operator_count': len(solver_result.phase_operators),
                'energy_keys': sorted(qubit.get_energy_matrices().keys()),
                'element_keys': sorted(qubit.get_element_matrices().keys()),
                'flux_shape': list(qubit.get_element_matrices('flux').shape),
                'ej_value': float(qubit.get_energy_matrices('Ej')[0, 0]),
                'f01_value': float(qubit.get_energylevel(1) / (2 * 3.141592653589793)),
                's_matrix': qubit.SMatrix.tolist(),
                'retain_nodes': list(qubit.SMatrix_retainNodes),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertTrue(payload['hamiltonian_is_qobj'])
        self.assertEqual(payload['hamiltonian_shape'], [3, 3])
        self.assertEqual(payload['hamiltonian_dims'], [[3], [3]])
        self.assertEqual(payload['eigenvalue_count'], 3)
        self.assertTrue(payload['first_eigenstate_is_qobj'])
        self.assertEqual(payload['first_eigenstate_shape'], [3, 1])
        self.assertEqual(payload['destroy_operator_count'], 1)
        self.assertTrue(payload['destroy_operators_are_qobj'])
        self.assertEqual(payload['number_operator_count'], 1)
        self.assertEqual(payload['phase_operator_count'], 1)
        self.assertEqual(payload['energy_keys'], ['Ec', 'Ej', 'Ej_max', 'El'])
        self.assertEqual(payload['element_keys'], ['capac', 'flux', 'induc', 'resis'])
        self.assertEqual(payload['flux_shape'], [1, 1])
        self.assertGreater(payload['ej_value'], 0.0)
        self.assertGreater(payload['f01_value'], 0.0)
        self.assertEqual(payload['s_matrix'], [[1.0]])
        self.assertEqual(payload['retain_nodes'], [0])

    def test_parameterized_qubit_change_para_rebuild_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.base import ParameterizedQubit

            with redirect_stdout(StringIO()):
                qubit = ParameterizedQubit(
                    capacitances=[
                        [1.8, -0.25, -0.1],
                        [-0.25, 1.6, -0.15],
                        [-0.1, -0.15, 1.4],
                    ],
                    junctions_resistance=[
                        [1e20, 11.0, 1e20],
                        [11.0, 1e20, 12.0],
                        [1e20, 12.0, 13.0],
                    ],
                    inductances=[
                        [40.0, 90.0, 100.0],
                        [90.0, 45.0, 95.0],
                        [100.0, 95.0, 50.0],
                    ],
                    fluxes=[
                        [0.0, 0.17, 0.0],
                        [0.17, 0.0, 0.0],
                        [0.0, 0.0, 0.09],
                    ],
                    trunc_ener_level=[2, 2],
                    junc_ratio=[
                        [1.0, 1.15, 0.0],
                        [1.15, 1.0, 0.0],
                        [0.0, 0.0, 1.05],
                    ],
                    structure_index=[2, 1],
                )

                qubit.change_para(
                    capac=[
                        [1.9, -0.28, -0.12],
                        [-0.28, 1.68, -0.18],
                        [-0.12, -0.18, 1.48],
                    ],
                    resis=[
                        [1e20, 10.5, 1e20],
                        [10.5, 1e20, 11.5],
                        [1e20, 11.5, 12.5],
                    ],
                    induc=[
                        [42.0, 92.0, 102.0],
                        [92.0, 47.0, 97.0],
                        [102.0, 97.0, 52.0],
                    ],
                )

            solver_result = qubit.solver_result
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'hamiltonian_is_qobj': isinstance(solver_result.hamiltonian, qt.Qobj),
                'hamiltonian_shape': list(solver_result.hamiltonian.shape),
                'hamiltonian_dims': solver_result.hamiltonian.dims,
                'eigenvalues': [float(value) for value in solver_result.eigenvalues],
                'first_eigenstate_is_qobj': isinstance(solver_result.eigenstates[0], qt.Qobj),
                'first_eigenstate_shape': list(solver_result.eigenstates[0].shape),
                'destroy_operator_count': len(solver_result.destroy_operators),
                'number_operator_count': len(solver_result.number_operators),
                'phase_operator_count': len(solver_result.phase_operators),
                'energy_keys': sorted(qubit.get_energy_matrices().keys()),
                'element_keys': sorted(qubit.get_element_matrices().keys()),
                'capac': qubit.get_element_matrices('capac').tolist(),
                'resis': qubit.get_element_matrices('resis').tolist(),
                'induc': qubit.get_element_matrices('induc').tolist(),
                'ec': qubit.get_energy_matrices('Ec').tolist(),
                'el': qubit.get_energy_matrices('El').tolist(),
                'ej_max': qubit.get_energy_matrices('Ej_max').tolist(),
                'ej': qubit.get_energy_matrices('Ej').tolist(),
                's_matrix': qubit.SMatrix.tolist(),
                'retain_nodes': list(qubit.SMatrix_retainNodes),
                'last_changed_params': sorted(qubit._last_changed_params),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertTrue(payload['hamiltonian_is_qobj'])
        self.assertEqual(payload['hamiltonian_shape'], [4, 4])
        self.assertEqual(payload['hamiltonian_dims'], [[2, 2], [2, 2]])
        np.testing.assert_allclose(
            payload['eigenvalues'],
            [
                -1.069894418873957e-07,
                2.379770209458227e-04,
                3.625978214434978e-04,
                6.006818318312078e-04,
            ],
        )
        self.assertTrue(payload['first_eigenstate_is_qobj'])
        self.assertEqual(payload['first_eigenstate_shape'], [4, 1])
        self.assertEqual(payload['destroy_operator_count'], 2)
        self.assertEqual(payload['number_operator_count'], 2)
        self.assertEqual(payload['phase_operator_count'], 2)
        self.assertEqual(payload['energy_keys'], ['Ec', 'Ej', 'Ej_max', 'El'])
        self.assertEqual(payload['element_keys'], ['capac', 'flux', 'induc', 'resis'])
        np.testing.assert_allclose(
            payload['capac'],
            [
                [1.9, -0.28, -0.12],
                [-0.28, 1.68, -0.18],
                [-0.12, -0.18, 1.48],
            ],
        )
        np.testing.assert_allclose(
            payload['resis'],
            [
                [1e20, 10.5, 1e20],
                [10.5, 1e20, 11.5],
                [1e20, 11.5, 12.5],
            ],
        )
        np.testing.assert_allclose(
            payload['induc'],
            [
                [42.0, 92.0, 102.0],
                [92.0, 47.0, 97.0],
                [102.0, 97.0, 52.0],
            ],
        )
        np.testing.assert_allclose(
            payload['ec'],
            [
                [2.285853058550994e-13, 8.489413009749221e-15],
                [8.489413009749218e-15, 1.059120785461306e-13],
            ],
        )
        np.testing.assert_allclose(
            payload['el'],
            [
                [2.790457108324908e-08, 2.595156093483211e-10],
                [2.595156093483211e-10, 4.040857664945163e-08],
            ],
        )
        np.testing.assert_allclose(
            payload['ej_max'],
            [
                [83220.12099281019, 0.0],
                [0.0, 69904.90163396056],
            ],
        )
        np.testing.assert_allclose(
            payload['ej'],
            [
                [71692.0026477512, 0.0],
                [0.0, 67130.92095206138],
            ],
        )
        self.assertEqual(payload['s_matrix'], [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(payload['retain_nodes'], [0, 2])
        self.assertEqual(payload['last_changed_params'], [])

    def test_grounded_transmon_public_frequency_path_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.analysis import analyze_single_qubit_spectrum
            from pysuqu.qubit.single import GroundedTransmon

            with redirect_stdout(StringIO()):
                qubit = GroundedTransmon(
                    capacitance=80e-15,
                    junction_resistance=10_000,
                    inductance=1e20,
                    flux=0.125,
                    trunc_ener_level=3,
                    junc_ratio=1.2,
                    qr_couple=[3e-15],
                )

            spectrum = analyze_single_qubit_spectrum(qubit)
            solver_result = qubit.solver_result
            f01_value = float(qubit.f01)
            anharmonicity_value = float(qubit.anharmonicity)
            energylevel_f01 = float(qubit.get_energylevel(1) / (2 * 3.141592653589793))
            energylevel_anharmonicity = float(
                qubit.get_energylevel(2) / (2 * 3.141592653589793) - 2 * energylevel_f01
            )
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'hamiltonian_is_qobj': isinstance(solver_result.hamiltonian, qt.Qobj),
                'hamiltonian_shape': list(solver_result.hamiltonian.shape),
                'destroy_operator_count': len(solver_result.destroy_operators),
                'f01_value': f01_value,
                'anharmonicity_value': anharmonicity_value,
                'spectrum_f01': float(spectrum.f01),
                'spectrum_anharmonicity': float(spectrum.anharmonicity),
                'energylevel_f01': energylevel_f01,
                'energylevel_anharmonicity': energylevel_anharmonicity,
                'energy_keys': sorted(qubit.get_energy_matrices().keys()),
                'element_keys': sorted(qubit.get_element_matrices().keys()),
                'flux_shape': list(qubit.get_element_matrices('flux').shape),
                'flux_value': float(qubit.get_element_matrices('flux')[0, 0]),
                'ej_value': float(qubit.get_energy_matrices('Ej')[0, 0]),
                's_matrix': qubit.SMatrix.tolist(),
                'retain_nodes': list(qubit.SMatrix_retainNodes),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertTrue(payload['hamiltonian_is_qobj'])
        self.assertEqual(payload['hamiltonian_shape'], [3, 3])
        self.assertEqual(payload['destroy_operator_count'], 1)
        self.assertAlmostEqual(payload['f01_value'], 4.762644473742976, places=6)
        self.assertAlmostEqual(payload['anharmonicity_value'], -0.21977297130109186, places=6)
        self.assertAlmostEqual(payload['spectrum_f01'], payload['f01_value'], places=12)
        self.assertAlmostEqual(
            payload['spectrum_anharmonicity'],
            payload['anharmonicity_value'],
            places=12,
        )
        self.assertAlmostEqual(payload['energylevel_f01'], payload['f01_value'], places=12)
        self.assertAlmostEqual(
            payload['energylevel_anharmonicity'],
            payload['anharmonicity_value'],
            places=12,
        )
        self.assertEqual(payload['energy_keys'], ['Ec', 'Ej', 'Ej_max', 'El'])
        self.assertEqual(payload['element_keys'], ['capac', 'flux', 'induc', 'resis'])
        self.assertEqual(payload['flux_shape'], [1, 1])
        self.assertAlmostEqual(payload['flux_value'], 0.125)
        self.assertGreater(payload['ej_value'], 0.0)
        self.assertEqual(payload['s_matrix'], [[1.0]])
        self.assertEqual(payload['retain_nodes'], [0])

    def test_grounded_transmon_energy_sweep_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.single import GroundedTransmon
            from pysuqu.qubit.sweeps import sweep_single_qubit_energy_vs_flux_base

            with redirect_stdout(StringIO()):
                qubit = GroundedTransmon(
                    capacitance=80e-15,
                    junction_resistance=10_000,
                    inductance=1e20,
                    flux=0.125,
                    trunc_ener_level=3,
                    junc_ratio=1.2,
                    qr_couple=[3e-15],
                )

            original_flux = np.asarray(qubit.get_element_matrices('flux'), dtype=float)
            baseline_f01 = float(qubit.f01)
            sweep_result = sweep_single_qubit_energy_vs_flux_base(
                qubit,
                [np.array([[0.0]]), np.array([[0.02]])],
                upper_level=2,
            )
            sweep_energies = np.asarray(
                [
                    sweep_result.series['level_1'],
                    sweep_result.series['level_2'],
                ],
                dtype=float,
            )
            restored_flux = np.asarray(qubit.get_element_matrices('flux'), dtype=float)

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_f01': baseline_f01,
                'sweep_energies': sweep_energies.tolist(),
                'restored_f01': float(qubit.f01),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_f01'], 4.762644473742976, places=6)
        np.testing.assert_allclose(
            payload['sweep_energies'],
            [
                [4.762644473742976, 4.693215705956289],
                [9.30551597618486, 9.166958372265105],
            ],
        )
        self.assertAlmostEqual(payload['restored_f01'], payload['baseline_f01'], places=12)
        np.testing.assert_allclose(payload['original_flux'], 0.125)
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_grounded_transmon_structured_energy_sweep_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.single import GroundedTransmon
            from pysuqu.qubit.sweeps import sweep_single_qubit_energy_vs_flux_base_result

            with redirect_stdout(StringIO()):
                qubit = GroundedTransmon(
                    capacitance=80e-15,
                    junction_resistance=10_000,
                    inductance=1e20,
                    flux=0.125,
                    trunc_ener_level=3,
                    junc_ratio=1.2,
                    qr_couple=[3e-15],
                )

            original_flux = np.asarray(qubit.get_element_matrices('flux'), dtype=float)
            baseline_f01 = float(qubit.f01)
            result = sweep_single_qubit_energy_vs_flux_base_result(
                qubit,
                [np.array([[0.0]]), np.array([[0.02]])],
                upper_level=2,
            )
            restored_flux = np.asarray(qubit.get_element_matrices('flux'), dtype=float)

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_f01': baseline_f01,
                'sweep_parameter': result.sweep_parameter,
                'sweep_values': [
                    np.asarray(value, dtype=float).tolist()
                    for value in result.sweep_values
                ],
                'series_keys': sorted(result.series.keys()),
                'level_1': np.asarray(result.series['level_1'], dtype=float).tolist(),
                'level_2': np.asarray(result.series['level_2'], dtype=float).tolist(),
                'metadata_flux_origin': np.asarray(
                    result.metadata['flux_origin'],
                    dtype=float,
                ).tolist(),
                'metadata_upper_level': int(result.metadata['upper_level']),
                'restored_f01': float(qubit.f01),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_f01'], 4.762644473742976, places=6)
        self.assertEqual(payload['sweep_parameter'], 'flux_offset')
        np.testing.assert_allclose(payload['sweep_values'], [[[0.0]], [[0.02]]])
        self.assertEqual(payload['series_keys'], ['level_1', 'level_2'])
        np.testing.assert_allclose(
            payload['level_1'],
            [4.762644473742976, 4.693215705956289],
        )
        np.testing.assert_allclose(
            payload['level_2'],
            [9.30551597618486, 9.166958372265105],
        )
        np.testing.assert_allclose(payload['metadata_flux_origin'], payload['original_flux'])
        self.assertEqual(payload['metadata_upper_level'], 2)
        self.assertAlmostEqual(payload['restored_f01'], payload['baseline_f01'], places=12)
        np.testing.assert_allclose(payload['original_flux'], 0.125)
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_floating_transmon_public_frequency_path_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.analysis import analyze_single_qubit_spectrum
            from pysuqu.qubit.single import FloatingTransmon

            with redirect_stdout(StringIO()):
                qubit = FloatingTransmon(
                    basic_element=[112e-15, 128e-15, 7.5e-15, 9_600],
                    flux=0.11,
                    trunc_ener_level=3,
                    junc_ratio=1.08,
                    qr_couple=[9.5e-15, 1.5e-15],
                )

            spectrum = analyze_single_qubit_spectrum(qubit)
            solver_result = qubit.solver_result
            f01_value = float(qubit.f01)
            anharmonicity_value = float(qubit.anharmonicity)
            energylevel_f01 = float(qubit.get_energylevel(1) / (2 * 3.141592653589793))
            energylevel_anharmonicity = float(
                qubit.get_energylevel(2) / (2 * 3.141592653589793) - 2 * energylevel_f01
            )
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'hamiltonian_is_qobj': isinstance(solver_result.hamiltonian, qt.Qobj),
                'hamiltonian_shape': list(solver_result.hamiltonian.shape),
                'destroy_operator_count': len(solver_result.destroy_operators),
                'f01_value': f01_value,
                'anharmonicity_value': anharmonicity_value,
                'spectrum_f01': float(spectrum.f01),
                'spectrum_anharmonicity': float(spectrum.anharmonicity),
                'energylevel_f01': energylevel_f01,
                'energylevel_anharmonicity': energylevel_anharmonicity,
                'energy_keys': sorted(qubit.get_energy_matrices().keys()),
                'element_keys': sorted(qubit.get_element_matrices().keys()),
                'flux_shape': list(qubit.get_element_matrices('flux').shape),
                'flux_matrix': qubit.get_element_matrices('flux').tolist(),
                'ej_value': float(qubit.get_energy_matrices('Ej')[0, 0]),
                's_matrix': qubit.SMatrix.tolist(),
                'retain_nodes': list(qubit.SMatrix_retainNodes),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertTrue(payload['hamiltonian_is_qobj'])
        self.assertEqual(payload['hamiltonian_shape'], [3, 3])
        self.assertEqual(payload['destroy_operator_count'], 1)
        self.assertAlmostEqual(payload['f01_value'], 5.335002628433321, places=6)
        self.assertAlmostEqual(payload['anharmonicity_value'], -0.2600099960878435, places=6)
        self.assertAlmostEqual(payload['spectrum_f01'], payload['f01_value'], places=12)
        self.assertAlmostEqual(
            payload['spectrum_anharmonicity'],
            payload['anharmonicity_value'],
            places=12,
        )
        self.assertAlmostEqual(payload['energylevel_f01'], payload['f01_value'], places=12)
        self.assertAlmostEqual(
            payload['energylevel_anharmonicity'],
            payload['anharmonicity_value'],
            places=12,
        )
        self.assertEqual(payload['energy_keys'], ['Ec', 'Ej', 'Ej_max', 'El'])
        self.assertEqual(payload['element_keys'], ['capac', 'flux', 'induc', 'resis'])
        self.assertEqual(payload['flux_shape'], [2, 2])
        np.testing.assert_allclose(payload['flux_matrix'], [[0.0, 0.11], [0.11, 0.0]])
        self.assertGreater(payload['ej_value'], 0.0)
        self.assertEqual(payload['s_matrix'], [[1.0, -1.0], [1.0, 1.0]])
        self.assertEqual(payload['retain_nodes'], [0])

    def test_qcrfgr_model_public_frequency_path_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.multi import QCRFGRModel

            c_j = 9.8e-15
            capacitance_list = [70.319e-15, 90.238e-15, 6.304e-15 + c_j, 78e-15, 12.65e-15]
            junction_resistance_list = [10007.92, 10007.92 / 6]

            with redirect_stdout(StringIO()):
                model = QCRFGRModel(
                    capacitance_list=capacitance_list,
                    junc_resis_list=junction_resistance_list,
                    qrcouple=[16.812e-15, 0.0159e-15],
                    flux_list=[0.11, 0.11],
                    trunc_ener_level=[6, 5],
                )

            solver_result = model.solver_result
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'hamiltonian_is_qobj': isinstance(solver_result.hamiltonian, qt.Qobj),
                'hamiltonian_shape': list(solver_result.hamiltonian.shape),
                'hamiltonian_dims': solver_result.hamiltonian.dims,
                'destroy_operator_count': len(solver_result.destroy_operators),
                'number_operator_count': len(solver_result.number_operators),
                'phase_operator_count': len(solver_result.phase_operators),
                'qubit_f01': float(model.qubit_f01),
                'coupler_f01': float(model.coupler_f01),
                'rq_g_mhz': float(model.rq_g / (2 * 3.141592653589793 * 1e6)),
                'qc_g_mhz': float(model.qc_g * 1e3),
                'energy_keys': sorted(model.get_energy_matrices().keys()),
                'element_keys': sorted(model.get_element_matrices().keys()),
                'ec_shape': list(model.get_energy_matrices('Ec').shape),
                'flux_shape': list(model.get_element_matrices('flux').shape),
                'flux_matrix': model.get_element_matrices('flux').tolist(),
                's_matrix': model.SMatrix.tolist(),
                'retain_nodes': list(model.SMatrix_retainNodes),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertTrue(payload['hamiltonian_is_qobj'])
        self.assertEqual(payload['hamiltonian_shape'], [30, 30])
        self.assertEqual(payload['hamiltonian_dims'], [[6, 5], [6, 5]])
        self.assertEqual(payload['destroy_operator_count'], 2)
        self.assertEqual(payload['number_operator_count'], 2)
        self.assertEqual(payload['phase_operator_count'], 2)
        self.assertAlmostEqual(payload['qubit_f01'], 5.488494132366466, places=6)
        self.assertAlmostEqual(payload['coupler_f01'], 11.480703336567979, places=6)
        self.assertAlmostEqual(payload['rq_g_mhz'], 141.00665866929367, places=6)
        self.assertAlmostEqual(payload['qc_g_mhz'], 374.0123048975068, places=6)
        self.assertEqual(payload['energy_keys'], ['Ec', 'Ej', 'Ej_max', 'El'])
        self.assertEqual(payload['element_keys'], ['capac', 'flux', 'induc', 'resis'])
        self.assertEqual(payload['ec_shape'], [2, 2])
        self.assertEqual(payload['flux_shape'], [3, 3])
        np.testing.assert_allclose(
            payload['flux_matrix'],
            [
                [0.0, 0.11, 0.0],
                [0.11, 0.0, 0.0],
                [0.0, 0.0, 0.11],
            ],
        )
        self.assertEqual(payload['s_matrix'], [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(payload['retain_nodes'], [0, 2])

    def test_qcrfgr_analysis_frequency_probe_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.analysis import get_multi_qubit_frequency_at_coupler_flux
            from pysuqu.qubit.multi import QCRFGRModel

            c_j = 9.8e-15
            capacitance_list = [70.319e-15, 90.238e-15, 6.304e-15 + c_j, 78e-15, 12.65e-15]
            junction_resistance_list = [10007.92, 10007.92 / 6]

            with redirect_stdout(StringIO()):
                model = QCRFGRModel(
                    capacitance_list=capacitance_list,
                    junc_resis_list=junction_resistance_list,
                    qrcouple=[16.812e-15, 0.0159e-15],
                    flux_list=[0.11, 0.11],
                    trunc_ener_level=[6, 5],
                )

            original_flux = model.get_element_matrices('flux').copy()
            baseline_qubit_f01 = float(model.qubit_f01)
            probed_qubit_f01 = float(
                get_multi_qubit_frequency_at_coupler_flux(model, 0.125, qubit_idx=0)
            )
            restored_flux = model.get_element_matrices('flux')

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'probed_qubit_f01': probed_qubit_f01,
                'baseline_qubit_f01': baseline_qubit_f01,
                'restored_qubit_f01': float(model.qubit_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['probed_qubit_f01'], 5.488326098729453, places=6)
        self.assertAlmostEqual(payload['baseline_qubit_f01'], 5.488494132366466, places=6)
        self.assertAlmostEqual(payload['restored_qubit_f01'], payload['baseline_qubit_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], 11.480703336567979, places=6)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0],
                [0.11, 0.0, 0.0],
                [0.0, 0.0, 0.11],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_analysis_frequency_probe_with_qubit_flux_overrides_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.analysis import get_multi_qubit_frequency_at_coupler_flux
            from pysuqu.qubit.multi import FGF1V1Coupling

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = model.get_element_matrices('flux').copy()
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            probed_qubit2_f01 = float(
                get_multi_qubit_frequency_at_coupler_flux(
                    model,
                    0.125,
                    qubit_idx=1,
                    qubit_fluxes=[0.09, 0.13],
                )
            )
            restored_flux = model.get_element_matrices('flux')

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'probed_qubit2_f01': probed_qubit2_f01,
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['probed_qubit2_f01'], 5.222727255647733, places=6)
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], 9.13495616294588, places=6)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_analysis_numerical_sensitivity_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.analysis import analyze_multi_qubit_coupler_sensitivity
            from pysuqu.qubit.multi import FGF1V1Coupling

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = model.get_element_matrices('flux').copy()
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            baseline_coupler_f01 = float(model.coupler_f01)
            numerical_sensitivity = float(
                analyze_multi_qubit_coupler_sensitivity(
                    model,
                    0.125,
                    method='numerical',
                    flux_step=1e-4,
                    qubit_idx=1,
                    is_print=False,
                ).sensitivity_value
            )
            restored_flux = model.get_element_matrices('flux')

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'baseline_coupler_f01': baseline_coupler_f01,
                'numerical_sensitivity': numerical_sensitivity,
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['baseline_coupler_f01'], 9.13495616294588, places=6)
        self.assertAlmostEqual(payload['numerical_sensitivity'], -0.01603849609122031, places=6)
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], payload['baseline_coupler_f01'], places=12)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_structured_sensitivity_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.analysis import analyze_multi_qubit_coupler_sensitivity_result
            from pysuqu.qubit.multi import FGF1V1Coupling

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            baseline_coupler_f01 = float(model.coupler_f01)
            baseline_qr_g = float(model.qr_g)
            baseline_qq_g = float(model.qq_g)
            baseline_qc_g = float(model.qc_g)
            baseline_qq_geff = float(model.qq_geff)
            result = analyze_multi_qubit_coupler_sensitivity_result(
                model,
                0.125,
                method='numerical',
                flux_step=1e-4,
                qubit_idx=1,
                is_print=False,
                is_plot=False,
            )
            restored_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'baseline_coupler_f01': baseline_coupler_f01,
                'baseline_qr_g': baseline_qr_g,
                'baseline_qq_g': baseline_qq_g,
                'baseline_qc_g': baseline_qc_g,
                'baseline_qq_geff': baseline_qq_geff,
                'coupler_flux_point': float(result.coupler_flux_point),
                'sensitivity_value': float(result.sensitivity_value),
                'metadata_method': result.metadata['method'],
                'metadata_flux_step': float(result.metadata['flux_step']),
                'metadata_qubit_idx': result.metadata['qubit_idx'],
                'metadata_qubit_fluxes': result.metadata['qubit_fluxes'],
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'restored_qr_g': float(model.qr_g),
                'restored_qq_g': float(model.qq_g),
                'restored_qc_g': float(model.qc_g),
                'restored_qq_geff': float(model.qq_geff),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['baseline_coupler_f01'], 9.13495616294588, places=6)
        self.assertAlmostEqual(payload['baseline_qr_g'], 827349574.7157319, places=3)
        self.assertAlmostEqual(payload['baseline_qq_g'], 0.029717909723577515, places=6)
        self.assertAlmostEqual(payload['baseline_qc_g'], 0.2806047528111849, places=6)
        self.assertAlmostEqual(payload['baseline_qq_geff'], 0.004231978575913273, places=6)
        self.assertAlmostEqual(payload['coupler_flux_point'], 0.125)
        self.assertAlmostEqual(payload['sensitivity_value'], -0.01603849609122031, places=6)
        self.assertEqual(payload['metadata_method'], 'numerical')
        self.assertAlmostEqual(payload['metadata_flux_step'], 1e-4, places=12)
        self.assertEqual(payload['metadata_qubit_idx'], 1)
        self.assertIsNone(payload['metadata_qubit_fluxes'])
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], payload['baseline_coupler_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qr_g'], payload['baseline_qr_g'], places=6)
        self.assertAlmostEqual(payload['restored_qq_g'], payload['baseline_qq_g'], places=12)
        self.assertAlmostEqual(payload['restored_qc_g'], payload['baseline_qc_g'], places=12)
        self.assertAlmostEqual(payload['restored_qq_geff'], payload['baseline_qq_geff'], places=12)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_multi_qubit_energy_sweep_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.multi import FGF1V1Coupling
            from pysuqu.qubit.sweeps import sweep_multi_qubit_energy_vs_flux

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            baseline_coupler_f01 = float(model.coupler_f01)
            energy_result = sweep_multi_qubit_energy_vs_flux(
                model,
                [0.11, 0.125],
                is_plot=False,
            )
            restored_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'baseline_coupler_f01': baseline_coupler_f01,
                'energy_result': {
                    'sweep_parameter': energy_result.sweep_parameter,
                    'sweep_values': list(energy_result.sweep_values),
                    'metadata': dict(energy_result.metadata),
                    'series': {
                        key: value.tolist()
                        for key, value in energy_result.series.items()
                    },
                },
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['baseline_coupler_f01'], 9.13495616294588, places=6)
        self.assertEqual(payload['energy_result']['sweep_parameter'], 'coupler_flux')
        self.assertEqual(payload['energy_result']['sweep_values'], [0.11, 0.125])
        self.assertIsNone(payload['energy_result']['metadata']['qubits_flux'])
        np.testing.assert_allclose(
            payload['energy_result']['series']['|[0, 0, 1]>'],
            [5.224530151681845, 5.224305368895874],
        )
        np.testing.assert_allclose(
            payload['energy_result']['series']['|[1, 0, 0]>'],
            [5.167776586474884, 5.167556712207027],
        )
        np.testing.assert_allclose(
            payload['energy_result']['series']['|[0, 1, 0]>'],
            [9.13495616294588, 9.052061249457607],
        )
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], payload['baseline_coupler_f01'], places=12)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_multi_qubit_structured_energy_sweep_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.multi import FGF1V1Coupling
            from pysuqu.qubit.sweeps import sweep_multi_qubit_energy_vs_flux_result

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            baseline_coupler_f01 = float(model.coupler_f01)
            baseline_qr_g = float(model.qr_g)
            baseline_qq_g = float(model.qq_g)
            baseline_qc_g = float(model.qc_g)
            baseline_qq_geff = float(model.qq_geff)
            result = sweep_multi_qubit_energy_vs_flux_result(
                model,
                [0.11, 0.125],
                is_plot=False,
            )
            restored_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'baseline_coupler_f01': baseline_coupler_f01,
                'baseline_qr_g': baseline_qr_g,
                'baseline_qq_g': baseline_qq_g,
                'baseline_qc_g': baseline_qc_g,
                'baseline_qq_geff': baseline_qq_geff,
                'sweep_parameter': result.sweep_parameter,
                'sweep_values': result.sweep_values,
                'series_keys': sorted(result.series.keys()),
                'state_001': np.asarray(result.series['|[0, 0, 1]>'], dtype=float).tolist(),
                'state_100': np.asarray(result.series['|[1, 0, 0]>'], dtype=float).tolist(),
                'state_010': np.asarray(result.series['|[0, 1, 0]>'], dtype=float).tolist(),
                'metadata_qubits_flux': result.metadata['qubits_flux'],
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'restored_qr_g': float(model.qr_g),
                'restored_qq_g': float(model.qq_g),
                'restored_qc_g': float(model.qc_g),
                'restored_qq_geff': float(model.qq_geff),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['baseline_coupler_f01'], 9.13495616294588, places=6)
        self.assertAlmostEqual(payload['baseline_qr_g'], 827349574.7157319, places=3)
        self.assertAlmostEqual(payload['baseline_qq_g'], 0.029717909723577515, places=6)
        self.assertAlmostEqual(payload['baseline_qc_g'], 0.2806047528111849, places=6)
        self.assertAlmostEqual(payload['baseline_qq_geff'], 0.004231978575913273, places=6)
        self.assertEqual(payload['sweep_parameter'], 'coupler_flux')
        self.assertEqual(payload['sweep_values'], [0.11, 0.125])
        self.assertEqual(
            payload['series_keys'],
            ['|[0, 0, 1]>', '|[0, 1, 0]>', '|[1, 0, 0]>'],
        )
        self.assertIsNone(payload['metadata_qubits_flux'])
        np.testing.assert_allclose(
            payload['state_001'],
            [5.224530151681845, 5.224305368895874],
        )
        np.testing.assert_allclose(
            payload['state_100'],
            [5.167776586474884, 5.167556712207027],
        )
        np.testing.assert_allclose(
            payload['state_010'],
            [9.13495616294588, 9.052061249457607],
        )
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], payload['baseline_coupler_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qr_g'], payload['baseline_qr_g'], places=3)
        self.assertAlmostEqual(payload['restored_qq_g'], payload['baseline_qq_g'], places=12)
        self.assertAlmostEqual(payload['restored_qc_g'], payload['baseline_qc_g'], places=12)
        self.assertAlmostEqual(payload['restored_qq_geff'], payload['baseline_qq_geff'], places=12)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_multi_qubit_structured_coupling_sweep_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.multi import FGF1V1Coupling
            from pysuqu.qubit.sweeps import sweep_multi_qubit_coupling_strength_vs_flux_result

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            baseline_coupler_f01 = float(model.coupler_f01)
            baseline_qr_g = float(model.qr_g)
            baseline_qq_g = float(model.qq_g)
            baseline_qc_g = float(model.qc_g)
            baseline_qq_geff = float(model.qq_geff)

            with redirect_stdout(StringIO()):
                result = sweep_multi_qubit_coupling_strength_vs_flux_result(
                    model,
                    [0.11, 0.125],
                    method='ES',
                    is_plot=False,
                )

            restored_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'baseline_coupler_f01': baseline_coupler_f01,
                'baseline_qr_g': baseline_qr_g,
                'baseline_qq_g': baseline_qq_g,
                'baseline_qc_g': baseline_qc_g,
                'baseline_qq_geff': baseline_qq_geff,
                'sweep_parameter': result.sweep_parameter,
                'sweep_values': result.sweep_values,
                'coupling_values': np.asarray(result.coupling_values, dtype=float).tolist(),
                'metadata_method': result.metadata['method'],
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'restored_qr_g': float(model.qr_g),
                'restored_qq_g': float(model.qq_g),
                'restored_qc_g': float(model.qc_g),
                'restored_qq_geff': float(model.qq_geff),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['baseline_coupler_f01'], 9.13495616294588, places=6)
        self.assertAlmostEqual(payload['baseline_qr_g'], 827349574.7157319, places=3)
        self.assertAlmostEqual(payload['baseline_qq_g'], 0.029717909723577515, places=6)
        self.assertAlmostEqual(payload['baseline_qc_g'], 0.2806047528111849, places=6)
        self.assertAlmostEqual(payload['baseline_qq_geff'], 0.004231978575913273, places=6)
        self.assertEqual(payload['sweep_parameter'], 'coupler_flux')
        self.assertEqual(payload['sweep_values'], [0.11, 0.125])
        self.assertEqual(payload['metadata_method'], 'ES')
        np.testing.assert_allclose(
            payload['coupling_values'],
            [0.004231978575913273, 0.00400676495779306],
        )
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], payload['baseline_coupler_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qr_g'], payload['baseline_qr_g'], places=6)
        self.assertAlmostEqual(payload['restored_qq_g'], payload['baseline_qq_g'], places=12)
        self.assertAlmostEqual(payload['restored_qc_g'], payload['baseline_qc_g'], places=12)
        self.assertAlmostEqual(payload['restored_qq_geff'], payload['baseline_qq_geff'], places=12)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1_multi_qubit_coupling_strength_sweep_helper_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.multi import FGF1V1Coupling
            from pysuqu.qubit.sweeps import sweep_multi_qubit_coupling_strength_vs_flux

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            original_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            baseline_qubit1_f01 = float(model.qubit1_f01)
            baseline_qubit2_f01 = float(model.qubit2_f01)
            baseline_coupler_f01 = float(model.coupler_f01)
            baseline_qr_g = float(model.qr_g)
            baseline_qq_g = float(model.qq_g)
            baseline_qc_g = float(model.qc_g)
            baseline_qq_geff = float(model.qq_geff)

            with redirect_stdout(StringIO()):
                g_result = sweep_multi_qubit_coupling_strength_vs_flux(
                    model,
                    [0.11, 0.125],
                    method='ES',
                    is_plot=False,
                )

            restored_flux = np.asarray(model.get_element_matrices('flux'), dtype=float)
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'baseline_qubit1_f01': baseline_qubit1_f01,
                'baseline_qubit2_f01': baseline_qubit2_f01,
                'baseline_coupler_f01': baseline_coupler_f01,
                'baseline_qr_g': baseline_qr_g,
                'baseline_qq_g': baseline_qq_g,
                'baseline_qc_g': baseline_qc_g,
                'baseline_qq_geff': baseline_qq_geff,
                'g_list': [float(value) for value in g_result.coupling_values],
                'restored_qubit1_f01': float(model.qubit1_f01),
                'restored_qubit2_f01': float(model.qubit2_f01),
                'restored_coupler_f01': float(model.coupler_f01),
                'restored_qr_g': float(model.qr_g),
                'restored_qq_g': float(model.qq_g),
                'restored_qc_g': float(model.qc_g),
                'restored_qq_geff': float(model.qq_geff),
                'original_flux': original_flux.tolist(),
                'restored_flux': restored_flux.tolist(),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertAlmostEqual(payload['baseline_qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['baseline_qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['baseline_coupler_f01'], 9.13495616294588, places=6)
        self.assertAlmostEqual(payload['baseline_qr_g'], 827349574.7157319, places=3)
        self.assertAlmostEqual(payload['baseline_qq_g'], 0.029717909723577515, places=6)
        self.assertAlmostEqual(payload['baseline_qc_g'], 0.2806047528111849, places=6)
        self.assertAlmostEqual(payload['baseline_qq_geff'], 0.004231978575913273, places=6)
        np.testing.assert_allclose(
            payload['g_list'],
            [0.004231978575913273, 0.00400676495779306],
        )
        self.assertAlmostEqual(payload['restored_qubit1_f01'], payload['baseline_qubit1_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qubit2_f01'], payload['baseline_qubit2_f01'], places=12)
        self.assertAlmostEqual(payload['restored_coupler_f01'], payload['baseline_coupler_f01'], places=12)
        self.assertAlmostEqual(payload['restored_qr_g'], payload['baseline_qr_g'], places=6)
        self.assertAlmostEqual(payload['restored_qq_g'], payload['baseline_qq_g'], places=12)
        self.assertAlmostEqual(payload['restored_qc_g'], payload['baseline_qc_g'], places=12)
        self.assertAlmostEqual(payload['restored_qq_geff'], payload['baseline_qq_geff'], places=12)
        np.testing.assert_allclose(
            payload['original_flux'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        np.testing.assert_allclose(payload['restored_flux'], payload['original_flux'])

    def test_fgf1v1_coupling_public_frequency_path_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.qubit.multi import FGF1V1Coupling

            c_j = 9.8e-15
            c_q1_total = 165e-15
            c_q2_total = 165e-15
            c_qc = 23.2e-15
            c_q_ground = 5.2e-15
            c_qq = 2.1e-15
            c_coupler_total = 142e-15
            c_11_ground = c_q1_total - c_qc - c_qq - c_q_ground
            c_12_ground = c_q2_total - c_q_ground
            capacitance_list = [
                c_11_ground,
                c_12_ground,
                c_q_ground + c_j,
                c_coupler_total - 2 * c_qc + 6 * c_j,
                c_q_ground + c_j,
                c_12_ground,
                c_11_ground,
                c_qq,
                c_qc,
                c_qc,
            ]

            with redirect_stdout(StringIO()):
                model = FGF1V1Coupling(
                    capacitance_list=capacitance_list,
                    junc_resis_list=[7400, 7400 / 6, 7400],
                    qrcouple=[18.34e-15, 0.02e-15],
                    flux_list=[0.11, 0.11, 0.11],
                    trunc_ener_level=[3, 2, 3],
                    is_print=False,
                )

            solver_result = model.solver_result
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'hamiltonian_is_qobj': isinstance(solver_result.hamiltonian, qt.Qobj),
                'hamiltonian_shape': list(solver_result.hamiltonian.shape),
                'hamiltonian_dims': solver_result.hamiltonian.dims,
                'destroy_operator_count': len(solver_result.destroy_operators),
                'number_operator_count': len(solver_result.number_operators),
                'phase_operator_count': len(solver_result.phase_operators),
                'qubit1_f01': float(model.qubit1_f01),
                'qubit2_f01': float(model.qubit2_f01),
                'coupler_f01': float(model.coupler_f01),
                'qr_g_mhz': float(model.qr_g / (2 * 3.141592653589793 * 1e6)),
                'qq_g_mhz': float(model.qq_g * 1e3),
                'qc_g_mhz': float(model.qc_g * 1e3),
                'qq_geff_mhz': float(model.qq_geff * 1e3),
                'energy_keys': sorted(model.get_energy_matrices().keys()),
                'element_keys': sorted(model.get_element_matrices().keys()),
                'ec_shape': list(model.get_energy_matrices('Ec').shape),
                'flux_shape': list(model.get_element_matrices('flux').shape),
                'flux_matrix': model.get_element_matrices('flux').tolist(),
                's_matrix': model.SMatrix.tolist(),
                'retain_nodes': list(model.SMatrix_retainNodes),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertTrue(payload['hamiltonian_is_qobj'])
        self.assertEqual(payload['hamiltonian_shape'], [18, 18])
        self.assertEqual(payload['hamiltonian_dims'], [[3, 2, 3], [3, 2, 3]])
        self.assertEqual(payload['destroy_operator_count'], 3)
        self.assertEqual(payload['number_operator_count'], 3)
        self.assertEqual(payload['phase_operator_count'], 3)
        self.assertAlmostEqual(payload['qubit1_f01'], 5.224530151681845, places=6)
        self.assertAlmostEqual(payload['qubit2_f01'], 5.167776586474884, places=6)
        self.assertAlmostEqual(payload['coupler_f01'], 9.13495616294588, places=6)
        self.assertAlmostEqual(payload['qr_g_mhz'], 131.67677448098613, places=6)
        self.assertAlmostEqual(payload['qq_g_mhz'], 29.717909723577513, places=6)
        self.assertAlmostEqual(payload['qc_g_mhz'], 280.60475281118494, places=6)
        self.assertAlmostEqual(payload['qq_geff_mhz'], 4.231978575913272, places=6)
        self.assertEqual(payload['energy_keys'], ['Ec', 'Ej', 'Ej_max', 'El'])
        self.assertEqual(payload['element_keys'], ['capac', 'flux', 'induc', 'resis'])
        self.assertEqual(payload['ec_shape'], [3, 3])
        self.assertEqual(payload['flux_shape'], [5, 5])
        np.testing.assert_allclose(
            payload['flux_matrix'],
            [
                [0.0, 0.11, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.11, 0.0],
            ],
        )
        self.assertEqual(
            payload['s_matrix'],
            [
                [1.0, -1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
            ],
        )
        self.assertEqual(payload['retain_nodes'], [0, 2, 3])

    def test_single_qubit_gate_run_simulation_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.funclib.awgenerator import EnvelopeParams, PulseEvent
            from pysuqu.qubit.gate import SingleQubitGate

            with redirect_stdout(StringIO()):
                gate = SingleQubitGate(total_time=8, sample_rate=2, energy_trunc_level=3)
                pulse = PulseEvent(
                    start_time=1.0,
                    envelope=EnvelopeParams(
                        name='x90',
                        duration=3.0,
                        peak_amp=0.2,
                        shape_type='cosine',
                    ),
                    name='x90',
                    if_freq=0.05,
                )
                channel = gate.load_pulse(pulse)
                result = gate.run_simulation(channel=channel)

            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'state_count': len(result.states),
                'time_count': len(result.times),
                'final_state_is_qobj': isinstance(result.states[-1], qt.Qobj),
                'final_state_shape': list(result.states[-1].shape),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertEqual(payload['state_count'], payload['time_count'])
        self.assertGreater(payload['state_count'], 0)
        self.assertTrue(payload['final_state_is_qobj'])
        self.assertEqual(payload['final_state_shape'], [3, 1])

    def test_single_qubit_gate_run_simulation_wires_real_c_ops_when_decoherence_is_loaded(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO
            from math import sqrt

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.funclib.awgenerator import EnvelopeParams, PulseEvent
            from pysuqu.qubit.gate import SingleQubitGate

            with redirect_stdout(StringIO()):
                gate = SingleQubitGate(total_time=8, sample_rate=2, energy_trunc_level=3)
                pulse = PulseEvent(
                    start_time=1.0,
                    envelope=EnvelopeParams(
                        name='x90',
                        duration=3.0,
                        peak_amp=0.2,
                        shape_type='cosine',
                    ),
                    name='x90',
                    if_freq=0.05,
                )
                channel = gate.load_pulse(pulse)
                gate.load_decoherence(T1=100.0, Tphi1=150.0, Tphi2=200.0)
                c_ops = gate._get_c_ops()
                result = gate.run_simulation(channel=channel)

            final_state = result.states[-1]
            trace = final_state.tr()
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'c_ops_count': len(c_ops),
                'first_two_c_ops_are_qobj': all(isinstance(c_op, qt.Qobj) for c_op in c_ops[:2]),
                'third_c_op_is_time_dependent': isinstance(c_ops[2], list) and callable(c_ops[2][1]),
                'third_c_op_coeff_sample': float(c_ops[2][1](5.0, {'Tphi2': 200.0})),
                'expected_third_c_op_coeff_sample': sqrt(10.0) / 200.0,
                'state_count': len(result.states),
                'time_count': len(result.times),
                'final_state_is_qobj': isinstance(final_state, qt.Qobj),
                'final_state_is_density_matrix': not final_state.isket,
                'final_state_shape': list(final_state.shape),
                'final_state_trace_real': float(trace.real),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertEqual(payload['c_ops_count'], 3)
        self.assertTrue(payload['first_two_c_ops_are_qobj'])
        self.assertTrue(payload['third_c_op_is_time_dependent'])
        self.assertAlmostEqual(
            payload['third_c_op_coeff_sample'],
            payload['expected_third_c_op_coeff_sample'],
        )
        self.assertEqual(payload['state_count'], payload['time_count'])
        self.assertGreater(payload['state_count'], 0)
        self.assertTrue(payload['final_state_is_qobj'])
        self.assertTrue(payload['final_state_is_density_matrix'])
        self.assertEqual(payload['final_state_shape'], [3, 3])
        self.assertAlmostEqual(payload['final_state_trace_real'], 1.0, places=6)

    def test_single_qubit_gate_calculate_fidelity_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            from pysuqu.funclib.awgenerator import EnvelopeParams, PulseEvent
            from pysuqu.qubit.gate import SingleQubitGate

            with redirect_stdout(StringIO()):
                gate = SingleQubitGate(total_time=8, sample_rate=2, energy_trunc_level=3)
                pulse = PulseEvent(
                    start_time=1.0,
                    envelope=EnvelopeParams(
                        name='idle',
                        duration=3.0,
                        peak_amp=0.0,
                        shape_type='cosine',
                    ),
                    name='idle',
                    if_freq=0.05,
                )
                channel = gate.load_pulse(pulse)
                metrics = gate.calculate_fidelity(
                    channel=channel,
                    target_state=qt.basis(2, 0),
                    is_print=False,
                )

            final_state_rot = metrics['final_state_rot']
            trace = final_state_rot.tr()
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'fidelity': float(metrics['fidelity']),
                'leakage': float(metrics['leakage']),
                'phase_error_deg': float(metrics['phase_error_deg']),
                'final_state_rot_is_qobj': isinstance(final_state_rot, qt.Qobj),
                'final_state_rot_is_density_matrix': not final_state_rot.isket,
                'final_state_rot_shape': list(final_state_rot.shape),
                'final_state_rot_trace_real': float(trace.real),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertGreater(payload['fidelity'], 0.99)
        self.assertLessEqual(payload['fidelity'], 1.0)
        self.assertGreaterEqual(payload['leakage'], 0.0)
        self.assertLess(payload['leakage'], 0.01)
        self.assertAlmostEqual(payload['phase_error_deg'], 0.0, places=6)
        self.assertTrue(payload['final_state_rot_is_qobj'])
        self.assertTrue(payload['final_state_rot_is_density_matrix'])
        self.assertEqual(payload['final_state_rot_shape'], [3, 3])
        self.assertAlmostEqual(payload['final_state_rot_trace_real'], 1.0, places=6)

    def test_single_qubit_gate_plot_bloch_evolution_uses_real_qutip_when_available(self):
        script = textwrap.dedent(
            """
            import json
            from contextlib import redirect_stdout
            from io import StringIO

            import numpy as np

            from tests.support import install_plotly_stub

            install_plotly_stub()

            import qutip as qt

            def to_float(value):
                value = np.real_if_close(value)
                if hasattr(value, 'item'):
                    value = value.item()
                if isinstance(value, complex):
                    return float(value.real)
                return float(value)

            class RecordingBloch:
                def __init__(self):
                    self.view = None
                    self.point_marker = []
                    self.point_size = []
                    self.points = []
                    self.shown = False
                    qt._last_bloch = self

                def add_points(self, points, meth=None):
                    self.points.append({'points': points, 'meth': meth})
                    return self

                def show(self):
                    self.shown = True
                    return None

            qt.Bloch = RecordingBloch

            from pysuqu.funclib.awgenerator import EnvelopeParams, PulseEvent
            from pysuqu.qubit.gate import SingleQubitGate

            with redirect_stdout(StringIO()):
                gate = SingleQubitGate(total_time=8, sample_rate=2, energy_trunc_level=3)
                pulse = PulseEvent(
                    start_time=1.0,
                    envelope=EnvelopeParams(
                        name='idle',
                        duration=3.0,
                        peak_amp=0.0,
                        shape_type='cosine',
                    ),
                    name='idle',
                    if_freq=0.05,
                )
                channel = gate.load_pulse(pulse)
                result = gate.run_simulation(channel=channel)
                x, y, z = gate.plot_bloch_evolution(result=result, rotation_omega=0.0)

            bloch = qt._last_bloch
            payload = {
                'qutip_file': getattr(qt, '__file__', ''),
                'time_count': len(result.times),
                'x_len': len(x),
                'y_len': len(y),
                'z_len': len(z),
                'x_abs_max': max(abs(to_float(value)) for value in x),
                'y_abs_max': max(abs(to_float(value)) for value in y),
                'z_min': min(to_float(value) for value in z),
                'z_max': max(to_float(value) for value in z),
                'bloch_points_count': len(bloch.points),
                'line_method': bloch.points[0]['meth'],
                'start_method': bloch.points[1]['meth'],
                'end_method': bloch.points[2]['meth'],
                'line_axis_lengths': [len(axis) for axis in bloch.points[0]['points']],
                'start_point': [to_float(value) for value in bloch.points[1]['points']],
                'end_point': [to_float(value) for value in bloch.points[2]['points']],
                'view': list(bloch.view),
                'point_marker': list(bloch.point_marker),
                'point_size': list(bloch.point_size),
                'shown': bool(bloch.shown),
                'z0': to_float(z[0]),
                'zend': to_float(z[-1]),
            }
            print(json.dumps(payload))
            """
        )

        payload = self._run_subprocess_and_get_payload(script)

        self.assertTrue(payload['qutip_file'])
        self.assertGreater(payload['time_count'], 0)
        self.assertEqual(payload['x_len'], payload['time_count'])
        self.assertEqual(payload['y_len'], payload['time_count'])
        self.assertEqual(payload['z_len'], payload['time_count'])
        self.assertLess(payload['x_abs_max'], 1e-6)
        self.assertLess(payload['y_abs_max'], 1e-6)
        self.assertGreater(payload['z_min'], 0.99)
        self.assertLessEqual(payload['z_max'], 1.0 + 1e-6)
        self.assertEqual(payload['bloch_points_count'], 3)
        self.assertEqual(payload['line_method'], 'l')
        self.assertEqual(payload['start_method'], 's')
        self.assertEqual(payload['end_method'], 's')
        self.assertEqual(payload['line_axis_lengths'], [payload['time_count']] * 3)
        self.assertEqual(payload['start_point'][:2], [0.0, 0.0])
        self.assertEqual(payload['end_point'][:2], [0.0, 0.0])
        self.assertAlmostEqual(payload['start_point'][2], payload['z0'], places=6)
        self.assertAlmostEqual(payload['end_point'][2], payload['zend'], places=6)
        self.assertEqual(payload['view'], [-45, 30])
        self.assertEqual(payload['point_marker'], ['o'])
        self.assertEqual(payload['point_size'], [20])
        self.assertTrue(payload['shown'])
