import unittest

from pysuqu import _native


class NativeLoaderCompatibilityTests(unittest.TestCase):
    def test_prepared_entry_point_is_optional_for_older_extensions(self):
        self.assertTrue(hasattr(_native, 'propagate_prepared'))
        self.assertTrue(_native.propagate_prepared is None or callable(_native.propagate_prepared))

    def test_existing_kernel_exports_remain_independent(self):
        for name in ('propagate', 'propagate_csr', 'propagate_banded',
                     'propagate_fused_csr', 'propagate_interaction_csr',
                     'propagate_lindblad_csr'):
            self.assertTrue(hasattr(_native, name))


if __name__ == '__main__':
    unittest.main()
