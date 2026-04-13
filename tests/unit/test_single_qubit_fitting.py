import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.single import GroundedTransmon


class GroundedTransmonFittingTests(unittest.TestCase):
    def test_fit_by_frequency_and_anharmonicity_restores_grounded_transmon_parameters(self):
        with redirect_stdout(StringIO()):
            qubit = GroundedTransmon(
                capacitance=80e-15,
                junction_resistance=10_000,
                inductance=1e20,
                flux=0.125,
                trunc_ener_level=3,
                junc_ratio=1.0,
            )

        original_capac = np.array(qubit.get_element_matrices('capac'), copy=True)
        original_resis = np.array(qubit.get_element_matrices('resis'), copy=True)
        original_freq = qubit.f01
        original_anharm = qubit.anharmonicity

        mocked_result = SimpleNamespace(x=np.array([75e-15, 9800.0]))
        with mock.patch('pysuqu.qubit.single.sp.optimize.minimize', return_value=mocked_result):
            with redirect_stdout(StringIO()):
                fitted = qubit.fit_by_frequency_and_anharmonicity(
                    test_freq=5.0,
                    test_anh=-0.25,
                    guess=[80e-15, 10_000],
                )

        np.testing.assert_allclose(fitted, mocked_result.x)
        np.testing.assert_allclose(qubit.get_element_matrices('capac'), original_capac)
        np.testing.assert_allclose(qubit.get_element_matrices('resis'), original_resis)
        self.assertAlmostEqual(qubit.f01, original_freq)
        self.assertAlmostEqual(qubit.anharmonicity, original_anharm)


if __name__ == '__main__':
    unittest.main()

