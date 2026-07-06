import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.single import RLINE, SingleQubitBase, e, hbar, pi


class SingleQubitDriveLossTests(unittest.TestCase):
    @staticmethod
    def _bare_single_qubit(f01=5.2, ec=0.28):
        qubit = object.__new__(SingleQubitBase)
        qubit.f01 = f01
        qubit.Ec = np.array([[ec]], dtype=float)
        qubit._capac = np.array([[1e-6]], dtype=float)
        return qubit

    @staticmethod
    def _mode_capacitance(ec):
        return e**2 / (2 * hbar * 1e9 * ec)

    def test_drive_loss_uses_ec_capacitance_and_capacitive_rc_correction(self):
        qubit = self._bare_single_qubit()
        freq = qubit.f01 * 1e9 * 2 * pi
        capa_drive = 2.0e-12
        mode_capacitance = self._mode_capacitance(qubit.Ec[0, 0])
        expected_t1_c = mode_capacitance * (1 + (freq * capa_drive * RLINE) ** 2) / (
            freq**2 * capa_drive**2 * RLINE
        )

        with redirect_stdout(StringIO()):
            freq_ghz, t1_i, t1_c, t1_drive = qubit.drive_loss(
                capa_drive=capa_drive,
                indu_drive=None,
            )

        self.assertAlmostEqual(freq_ghz, qubit.f01)
        self.assertTrue(np.isinf(t1_i))
        np.testing.assert_allclose(t1_c, expected_t1_c)
        np.testing.assert_allclose(t1_drive, expected_t1_c)

    def test_drive_loss_uses_ec_capacitance_for_inductive_branch_and_allows_zero_capacitor(self):
        qubit = self._bare_single_qubit()
        freq = qubit.f01 * 1e9 * 2 * pi
        indu_drive = 0.8e-12
        mode_capacitance = self._mode_capacitance(qubit.Ec[0, 0])
        expected_t1_i = RLINE / (freq**4 * indu_drive**2 * mode_capacitance)

        with redirect_stdout(StringIO()):
            _, t1_i, t1_c, t1_drive = qubit.drive_loss(
                capa_drive=0.0,
                indu_drive=indu_drive,
            )

        np.testing.assert_allclose(t1_i, expected_t1_i)
        self.assertTrue(np.isinf(t1_c))
        np.testing.assert_allclose(t1_drive, expected_t1_i)

    def test_drive_loss_returns_infinite_t1_when_both_channels_are_disabled(self):
        qubit = self._bare_single_qubit()

        with redirect_stdout(StringIO()):
            _, t1_i, t1_c, t1_drive = qubit.drive_loss(
                capa_drive=None,
                indu_drive=0.0,
            )

        self.assertTrue(np.isinf(t1_i))
        self.assertTrue(np.isinf(t1_c))
        self.assertTrue(np.isinf(t1_drive))


if __name__ == '__main__':
    unittest.main()
