import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from tests.support import install_test_stubs

install_test_stubs()

from pysuqu.qubit.single import (
    RLINE,
    Phi0,
    SingleQubitBase,
    e,
    hbar,
    pi,
)


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



class SingleQubitDriveStrengthTests(unittest.TestCase):
    """Pin the reported drive strengths to their documented MHz convention."""

    @staticmethod
    def _bare_single_qubit(f01=5.2, ec=0.28, ej=18.0):
        qubit = object.__new__(SingleQubitBase)
        qubit.f01 = f01
        qubit.Ec = np.array([[ec]], dtype=float)
        qubit.Ej = np.array([[ej]], dtype=float)
        qubit._capac = np.array([[1e-6]], dtype=float)
        return qubit

    def test_capacitive_strength_reports_angular_frequency_in_mhz(self):
        qubit = self._bare_single_qubit()
        voltage_amp = 1e-3
        coupl_capac = 2.0e-16

        with redirect_stdout(StringIO()):
            strength = qubit.drive_strength(voltage_amp, coupl_capac, mode='cap')

        expected = (
            coupl_capac * voltage_amp
            * (8 * qubit.Ej[0, 0] * qubit.Ec[0, 0] ** 3) ** (1 / 4)
            / e / (2 * pi) * 1e3
        )
        np.testing.assert_allclose(strength, expected)

    def test_inductive_strength_does_not_add_an_extra_two_pi(self):
        qubit = self._bare_single_qubit()
        voltage_amp = 1e-3
        coupl_induc = 1.0e-12

        with redirect_stdout(StringIO()):
            strength = qubit.drive_strength(voltage_amp, coupl_induc, mode='ind')

        expected = (
            coupl_induc * (voltage_amp / RLINE)
            * (2 * qubit.Ec[0, 0] * qubit.Ej[0, 0] ** 3) ** (1 / 4)
            / Phi0 * 1e3
        )
        np.testing.assert_allclose(strength, expected)

    def test_combined_modes_keep_the_cap_and_ind_order(self):
        qubit = self._bare_single_qubit()
        voltage_amp = 1e-3
        coupl_induc, coupl_capac = 1.0e-12, 2.0e-16

        with redirect_stdout(StringIO()):
            cap_strength = qubit.drive_strength(voltage_amp, coupl_capac, mode='cap')
            ind_strength = qubit.drive_strength(voltage_amp, coupl_induc, mode='ind')
            ind_plus_cap = qubit.drive_strength(
                voltage_amp, (coupl_induc, coupl_capac), mode='ind+cap'
            )
            cap_plus_ind = qubit.drive_strength(
                voltage_amp, (coupl_capac, coupl_induc), mode='cap+ind'
            )

        np.testing.assert_allclose(ind_plus_cap, [ind_strength, cap_strength])
        np.testing.assert_allclose(cap_plus_ind, [cap_strength, ind_strength])


if __name__ == '__main__':
    unittest.main()
