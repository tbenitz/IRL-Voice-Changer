#!/usr/bin/env python3
"""CPU-only regression checks for costly GPU pipeline failure points.

These tests intentionally use fake ONNX modules: they verify packaging
control flow (not neural inference) without requiring GPU/model downloads.
"""
import ast
from html import escape
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def get_function(filepath, name, extras):
    code = filepath.read_text()
    tree = ast.parse(code)
    target = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = dict(extras)
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(filepath), "exec"), namespace)
    return namespace[name]


def fake_onnx():
    sys.modules["onnx"] = types.SimpleNamespace(
        checker=types.SimpleNamespace(check_model=lambda path: None)
    )

    class FakeSession:
        def __init__(self, path, providers):
            if providers != ["CPUExecutionProvider"]:
                raise AssertionError("Model recovery must run on CPU")
        def get_inputs(self):
            return [types.SimpleNamespace(name="audio_in")]
        def get_outputs(self):
            return [types.SimpleNamespace(name="audio_out")]
    sys.modules["onnxruntime"] = types.SimpleNamespace(InferenceSession=FakeSession)


class PipelineSmoke(unittest.TestCase):
    def test_small_json_report_packages_trained_pth_onnx(self):
        fake_onnx()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            output = run / "llvc-trained"
            checkpoint = output / "checkpoints"
            checkpoint.mkdir(parents=True)
            dataset = run / "dataset"
            dataset.mkdir()
            (checkpoint / "G_500.pth").write_bytes(b"p" * 2_000_001)
            (output / "example.onnx").write_bytes(b"o" * 2_000_001)
            report = output / "model_report.json"
            report.write_text(json.dumps({"file": "example.onnx", "steps": 500}))
            self.assertLess(report.stat().st_size, 1024)
            (output / "config.json").write_text("{}")
            (dataset / "manifest.json").write_text(
                json.dumps({"train": 360, "val": 32, "dev": 8})
            )
            (run / "reference_selection.json").write_text("{}")
            # Package-only is the recovery path for an already trained GPU
            # model and must not call CUDA, Seed-VC, or LLVC again.
            original_argv = sys.argv
            try:
                sys.argv = ["one_click_pipeline.py", "--package-only",
                            "--run-dir", str(run)]
                spec = importlib.util.spec_from_file_location(
                    "one_click_pipeline", ROOT / "one_click_pipeline.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.main()
                bundle = run / "example_PTH_ONNX.zip"
                self.assertTrue(bundle.is_file())
                with zipfile.ZipFile(bundle) as archive:
                    self.assertIsNone(archive.testzip())
                    names = set(archive.namelist())
                    self.assertIn("model/G_500.pth", names)
                    self.assertIn("model/example.onnx", names)
                    self.assertIn("model/model_report.json", names)
            finally:
                sys.argv = original_argv

    def test_progress_uses_true_pair_and_step_counts(self):
        app = ROOT / "hf-space" / "app.py"
        progress_html = get_function(app, "progress_html", {"escape": escape})
        markup = progress_html(42, 250, 500, stage="Training")
        self.assertIn("42 / 400", markup)
        self.assertIn("250 / 500", markup)
        self.assertIn('value="42"', markup)
        self.assertIn('value="250"', markup)
        self.assertIn("Bars track completed work", markup)
        tree = ast.parse(app.read_text())
        generator = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "create_voice")
        patterns = [
            n.args[0].value for n in ast.walk(generator)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "search" and n.args
            and isinstance(n.args[0], ast.Constant)
        ]
        self.assertTrue(any(re.search(p, "Teacher conversion: 42/400 accepted; 45 attempted")
                            for p in patterns))
        self.assertTrue(any(re.search(p, "LLVC training 250/500 steps")
                            for p in patterns))

    def test_source_corpus_does_not_transcode_all_800_candidates(self):
        source_corpus = get_function(
            ROOT / "generate_teacher_pairs.py", "source_corpus",
            {"Path": Path, "random": random,
             "command": lambda *args: self.fail("Unexpected ffmpeg preprocessing")}
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(30):
                clip = root / str(i) / "chapter" / f"{i}.flac"
                clip.parent.mkdir(parents=True)
                clip.touch()
            found = source_corpus(root, 25, root / "scratch")
            self.assertEqual(len(found), 25)
            self.assertTrue(all(f.suffix == ".flac" for f in found))


if __name__ == "__main__":
    unittest.main(verbosity=2)
