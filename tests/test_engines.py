"""Offline, dependency-free tests: commands are bound to real upstream entry points."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import engines

class EngineTests(unittest.TestCase):
    def test_all_requested_models(self):
        self.assertEqual(set(engines.ENGINES),
                         {"beatrice", "rvc", "seed", "llvc", "meanvc2"})

    def test_live_vs_file_modes_explicit(self):
        self.assertIn("file", engines.ENGINES["llvc"].mode.lower())
        for key in ("rvc", "beatrice", "seed", "meanvc2"):
            self.assertIn("Live", engines.ENGINES[key].mode)

    def test_cannot_launch_noninstalled_engine(self):
        with tempfile.TemporaryDirectory() as t:
            with mock.patch.object(engines, "HOME", Path(t)):
                with self.assertRaises(ValueError):
                    engines.launch_args("seed", {})

    def test_llvc_requires_input_file(self):
        with tempfile.TemporaryDirectory() as t:
            with mock.patch.object(engines, "HOME", Path(t)):
                venv = engines.venv_python("llvc")
                venv.parent.mkdir(parents=True)
                venv.touch()
                engines.source_dir("llvc").mkdir(parents=True)
                with self.assertRaises(ValueError):
                    engines.launch_args("llvc", {}, input_file="")

    def test_real_upstream_commands(self):
        with tempfile.TemporaryDirectory() as t:
            with mock.patch.object(engines, "HOME", Path(t)):
                for key in ("seed", "rvc", "llvc", "meanvc2"):
                    p = engines.venv_python(key)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.touch()
                    engines.source_dir(key).mkdir(parents=True, exist_ok=True)
                inp = Path(t) / "input.wav"
                inp.touch()
                for key, expected in [("seed","real-time-gui.py"),
                                      ("rvc","realtime_gui.py"),
                                      ("llvc","infer.py"),
                                      ("meanvc2","runtime/run_rt.py")]:
                    cmd, cwd = engines.launch_args(
                        key, {}, reference=str(inp), input_file=str(inp))
                    self.assertIn(expected, cmd)
                    self.assertTrue(cwd.is_dir())
                    if key == "llvc":
                        self.assertIn("-s", cmd)
                        adapter = engines.source_dir("llvc") / "irlvc_llvc_mic.py"
                        adapter.touch()
                        live, _ = engines.launch_args("llvc", {}, llvc_live=True)
                        self.assertIn(str(adapter), live)

if __name__ == "__main__":
    unittest.main()
