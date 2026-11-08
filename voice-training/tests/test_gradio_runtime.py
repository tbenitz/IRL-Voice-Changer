#!/usr/bin/env python3
"""Construct the real Hugging Face Gradio UI without GPU/network or launching server.

This catches incompatible Gradio API args (such as value= vs default_value=
for gr.BrowserState), output/input wiring and missing controls before a user
starts paying for another T4 run.
"""
import ast
from html import escape
import json
from pathlib import Path
import tempfile
import unittest
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "hf-space" / "app.py"


def extract(name, scope):
    ast_tree = ast.parse(APP.read_text())
    node = next(n for n in ast_tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(scope)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(APP), "exec"), namespace)
    return namespace[name]


class GradioRuntimeSmoke(unittest.TestCase):
    def test_build_actual_gradio_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            jobs = workspace / "voice-jobs"
            jobs.mkdir()
            scope = {
                "gr": gr, "CODE_PROBLEM": "", "WORKSPACE": workspace,
                "JOBS": jobs, "Path": Path, "escape": escape,
                "PROFILES": {
                    "High fidelity — 800 pairs, 5,000 updates": (800, 5000, 20),
                    "Studio extended — 1,200 pairs, 8,000 updates": (1200, 8000, 24),
                    "Lower-cost trial — 400 pairs, 1,000 updates": (400, 1000, 10)
                },
                "DEFAULT_PROFILE": "High fidelity — 800 pairs, 5,000 updates",
                "recent_jobs": lambda: [],
                "start_job": lambda audio, profile, browser_id: ("", "", ""),
                "retry_job": lambda browser_id: ("", "", ""),
                "open_saved_job": lambda selection: ("", "", ""),
                "refresh_choices": lambda: gr.update(choices=[]),
                "job_status": lambda job_id, seen: (
                    "", "", "", None, None, None, None, None, ("",) * 6
                )
            }
            scope["progress_html"] = extract("progress_html", scope)
            make_app = extract("make_app", scope)
            demo = make_app()
            self.assertGreaterEqual(len(demo.blocks), 15)
            self.assertGreaterEqual(len(demo.fns), 6)
            self.assertTrue(any(isinstance(item, gr.BrowserState)
                                for item in demo.blocks.values()))
            self.assertTrue(any(isinstance(item, gr.Timer)
                                for item in demo.blocks.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
