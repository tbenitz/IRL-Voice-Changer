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
        markup = progress_html({
            "status": "running", "stage": "Training",
            "pairs_done": 462, "pairs_target": 800,
            "updates_done": 725, "steps_target": 5000
        }, "G_500.pth", None)
        self.assertIn("462/800", markup)
        self.assertIn("725/5000", markup)
        self.assertIn('value="462"', markup)
        self.assertIn('value="725"', markup)
        self.assertIn("Ready to download", markup)
        self.assertIn("Refresh", markup)



    def test_completed_model_shortcuts_gpu_and_zip(self):
        """Existing PTH+ONNX must be returned without starting any paid GPU work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / "voice-job"
            out = job / "llvc-trained"
            checkpoints = out / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "G_500.pth").write_bytes(b"p" * 2_000_001)
            (out / "example.onnx").write_bytes(b"o" * 2_000_001)
            (out / "model_report.json").write_text(
                json.dumps({"file": "example.onnx", "steps": 500}))
            (job / "dataset").mkdir()
            (job / "dataset" / "manifest.json").write_text(
                json.dumps({"train": 360, "val": 32, "dev": 8}))
            (job / "input.wav").write_bytes(b"dummy")
            for folder in ("repo", "seed", "llvc"):
                (root / folder).mkdir()
            old_argv = sys.argv
            try:
                sys.argv = [
                    "one_click_pipeline.py", "--audio", str(job / "input.wav"),
                    "--run-dir", str(job), "--repo", str(root / "repo"),
                    "--seed-vc", str(root / "seed"), "--llvc", str(root / "llvc"),
                    "--steps", "500", "--skip-zip",
                ]
                spec = importlib.util.spec_from_file_location(
                    "pipeline_no_zip", ROOT / "one_click_pipeline.py"
                )
                pipeline = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pipeline)
                pipeline.main()
            finally:
                sys.argv = old_argv
            self.assertFalse(list(job.glob("*.zip")))
            self.assertTrue((checkpoints / "G_500.pth").exists())
            self.assertTrue((out / "example.onnx").exists())

    def test_direct_pth_download_available_before_onnx(self):
        saved_files = get_function(
            ROOT / "hf-space" / "app.py", "saved_files",
            {"re": re, "Path": Path, "MODEL_RE": re.compile(r"^[A-Za-z0-9_-]+\.onnx$"),
             "json": json}
        )
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            output = job / "llvc-trained"
            checkpoint = output / "checkpoints"
            checkpoint.mkdir(parents=True)
            pth = checkpoint / "G_100.pth"
            pth.write_bytes(b"p" * 2_000_001)
            state = {"checkpoint": str(pth), "status": "running"}
            self.assertEqual(saved_files(job, state), (str(pth), None))
            onnx = output / "my_voice.onnx"
            onnx.write_bytes(b"o" * 2_000_001)
            self.assertEqual(saved_files(job, state), (str(pth), None))
            report = output / "model_report.json"
            report.write_text(json.dumps({
                "file": onnx.name, "steps": 100,
                "size_bytes": onnx.stat().st_size,
                "onnx_io": "audio_in -> audio_out"
            }))
            self.assertEqual(saved_files(job, state), (str(pth), str(onnx)))
            onnx.write_bytes(b"truncated")
            self.assertEqual(saved_files(job, state), (str(pth), None))



    def test_space_no_longer_requires_zip(self):
        app = (ROOT / "hf-space" / "app.py").read_text()
        worker = (ROOT / "job_worker.py").read_text()
        pipeline = (ROOT / "one_click_pipeline.py").read_text()
        self.assertIn('"--skip-zip"', worker)
        self.assertIn('gr.BrowserState(', app)
        self.assertIn('gr.Timer(value=3.0)', app)
        self.assertIn('start_new_session=True', app)
        self.assertIn('saved_files(job, state)', app)
        self.assertIn('if args.skip_zip:', pipeline)
        self.assertNotIn('def create_voice(', app)
        self.assertNotIn('recover_completed_model(', app)
        self.assertEqual(pipeline.count('parser.add_argument("--pairs"'), 1)



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
