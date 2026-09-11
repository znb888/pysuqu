"""Import and compatibility checks for the split native backend modules."""

import importlib
import unittest


class CppBackendModuleTests(unittest.TestCase):
    def test_focused_modules_import_without_native_extension(self):
        modules = {
            "native_loader": ["_load_native", "_native_propagate"],
            "plans": ["_NativePlan", "_NativePayload", "NativePropagationResult"],
            "preparation": ["_qobj_matrix", "_qobj_csr", "_prepare_native_trace_payloads"],
            "execution": ["NativeExecution"],
            "results": ["_decode_complex_payload"],
            "lindblad": ["LindbladCppPropagationBackend"],
        }
        for name, symbols in modules.items():
            module = importlib.import_module(f"pysuqu.qubit.backends.cpp.{name}")
            for symbol in symbols:
                with self.subTest(module=name, symbol=symbol):
                    self.assertTrue(hasattr(module, symbol))

    def test_reexports_match_legacy_adapter(self):
        legacy = importlib.import_module("pysuqu.qubit.backends.cpp_backend")
        plans = importlib.import_module("pysuqu.qubit.backends.cpp.plans")
        results = importlib.import_module("pysuqu.qubit.backends.cpp.results")
        lindblad = importlib.import_module("pysuqu.qubit.backends.cpp.lindblad")
        self.assertIs(plans._NativePlan, legacy._NativePlan)
        self.assertIs(plans._NativePayload, legacy._NativePayload)
        self.assertIs(results._decode_complex_payload, legacy._decode_complex_payload)
        self.assertIs(lindblad.LindbladCppPropagationBackend, legacy.LindbladCppPropagationBackend)


if __name__ == "__main__":
    unittest.main()
