"""BTQ1 format, quantization and optional Python/Rust parity tests."""

import ast
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from tiny_student import BINS, MODEL_BYTES, features, load_model, logits, window
from train_magika_qat_student import QAT_ACTIVE, QDense
from train_tiny_student import export


class TinyStudentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "model.bin"
        tf.keras.utils.set_random_seed(2)
        QAT_ACTIVE.assign(True)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(BINS,)),
            QDense(16, 4),
            tf.keras.layers.ReLU(),
            QDense(48, 4),
        ])
        export(self.path, self.model)

    def test_export_matches_quantized_training(self):
        self.assertEqual(self.path.stat().st_size, MODEL_BYTES)
        inputs = np.random.default_rng(2).normal(size=(20, BINS)).astype(np.float32)
        expected = self.model(inputs).numpy()
        np.testing.assert_allclose(logits(inputs, load_model(self.path)), expected, atol=2e-6)

    def test_malformed_models(self):
        valid = self.path.read_bytes()
        malformed = [
            b"", valid[:-1], valid + b"\0", b"INVALID!" + valid[8:],
            valid[:8] + struct.pack("<f", float("nan")) + valid[12:],
            valid[:8] + struct.pack("<f", 0) + valid[12:],
            valid[:-4] + struct.pack("<f", float("inf")),
        ]
        for blob in malformed:
            self.path.write_bytes(blob)
            with self.assertRaises(ValueError):
                load_model(self.path)

    def test_windows_and_short_inputs(self):
        self.assertEqual(window(b"\tshort"), b"")
        self.assertEqual(window(b" " * 4096 + b"fn main() {}"), b"")
        self.assertEqual(len(window(b"a" * 1023)), 1023)
        self.assertEqual(len(window(b"a" * 1024)), 2048)
        self.assertEqual(window(b"a" * 4096 + b"b" * 4096), b"a" * 1024 + b"b" * 1024)
        self.assertFalse(features(b"\tshort").any())

    def test_normalization_and_number_folding(self):
        a = features(b"PRINT(value, 123)")
        b = features(b"print(value, 987654)")
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=6)
        self.assertFalse(np.array_equal(features(b"first second"), features(b"second first")))

    @unittest.skipUnless(os.environ.get("BETLANG_TINY_EXAMPLE"), "set BETLANG_TINY_EXAMPLE")
    def test_rust_parity(self):
        executable = os.environ["BETLANG_TINY_EXAMPLE"]
        source_path = self.root / "source"
        rng = np.random.default_rng(3)
        sources = [
            b"fn main() { println!(\"hello\"); }\n",
            b"\x0b" + b"x" * 1023,
            b"\r\n\t" + b"x" * 4096 + b"\r\n\t",
            "fn caf\u00e9() { /* \u03bb */ }\n".encode(),
        ]
        sources.extend(rng.integers(0, 256, n, dtype=np.uint8).tobytes()
                       for n in [8, 31, 1023, 1024, 1025, 2048, 4097, 10000] * 4)
        for source in sources:
            source_path.write_bytes(source)
            command = [executable, str(self.path), str(source_path)]
            output = subprocess.check_output(command + ["--features"], text=True)
            actual_features = np.array(ast.literal_eval(output.strip()[5:-1]), np.float32)
            np.testing.assert_allclose(actual_features, features(source), atol=1e-6)
            output = subprocess.run(command, text=True, check=True, capture_output=True).stdout
            actual = np.array(ast.literal_eval(output.splitlines()[1][5:-1]), np.float32)
            expected = logits(features(source), load_model(self.path))
            np.testing.assert_allclose(actual, expected, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
