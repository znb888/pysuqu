import os
import unittest
from unittest.mock import patch

from pysuqu._native_build_options import native_build_options


class NativeBuildOptionsTests(unittest.TestCase):
    def test_defaults_are_portable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native_build_options()['pgo'], 'off')
            self.assertFalse(native_build_options()['march_native'])

    def test_explicit_tuning_is_parsed(self):
        with patch.dict(os.environ, {
            'PYSUQU_NATIVE_MARCH_NATIVE': '1',
            'PYSUQU_NATIVE_PGO': 'generate',
            'PYSUQU_NATIVE_PGO_DIR': 'build/profile-data',
        }, clear=True):
            options = native_build_options()
        self.assertTrue(options['march_native'])
        self.assertEqual(options['pgo'], 'generate')
        self.assertTrue(options['profile_dir'].endswith('build\\profile-data') or options['profile_dir'].endswith('build/profile-data'))

    def test_invalid_modes_fail_early(self):
        with patch.dict(os.environ, {'PYSUQU_NATIVE_PGO': 'turbo'}, clear=True):
            with self.assertRaises(ValueError):
                native_build_options()


if __name__ == '__main__':
    unittest.main()
