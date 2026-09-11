import unittest

import numpy as np

from pysuqu.decoherence.dequbit import Decoherence
from pysuqu.funclib.mathlib import (
    PreparedFilteredPSD,
    ramsey_transfunc,
)


class PreparedPSDBatchTests(unittest.TestCase):
    def setUp(self):
        self.frequency = np.logspace(1, 6, 24)
        self.psd = 2.5e-12 / self.frequency ** 0.7
        self.delays = np.array([0.2e-6, 0.9e-6, 4.0e-6])

    def test_many_matches_scalar_integrals(self):
        prepared = PreparedFilteredPSD.for_continuous(self.frequency, self.psd)
        batched = prepared.integrate_continuous_many(
            lambda f: ramsey_transfunc(f, self.delays), len(self.delays)
        )
        scalar = np.array([
            prepared.integrate_continuous(lambda f, tau=tau: ramsey_transfunc(f, tau))
            for tau in self.delays
        ])
        np.testing.assert_allclose(batched, scalar, rtol=2e-8, atol=1e-30)

    def test_decoherence_cal_dephase_uses_batch_path(self):
        model = object.__new__(Decoherence)
        actual = model.cal_dephase(
            self.psd,
            sensitivity_factor=3.2e5,
            noise_freq=self.frequency,
            experiment="Ramsey",
            delay_list=self.delays,
            integration_method="continuous",
        )
        expected = np.array([
            np.exp(-integrate * (3.2e5 * 2) ** 2 / 2)
            for integrate in [
                PreparedFilteredPSD.for_continuous(self.frequency, self.psd).integrate_continuous(
                    lambda f, tau=tau: ramsey_transfunc(f, tau)
                )
                for tau in self.delays
            ]
        ])
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-12)

    def test_discrete_preparation_preserves_log_rule(self):
        prepared = PreparedFilteredPSD.for_discrete(self.frequency, self.psd)
        expected = prepared.integrate_discrete(
            lambda f: ramsey_transfunc(f, self.delays[0]), method="log"
        )
        self.assertTrue(np.isfinite(expected))
        self.assertGreaterEqual(expected, 0.0)


if __name__ == "__main__":
    unittest.main()
