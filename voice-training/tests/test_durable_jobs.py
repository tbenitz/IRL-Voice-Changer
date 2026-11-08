#!/usr/bin/env python3
"""GPU-free checks of the durable trainer and refreshed-browser recovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("job_worker", ROOT / "job_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def create_job(root):
    job = root / "123456789abc"
    job.mkdir()
    (job / "job_config.json").write_text(json.dumps({
        "audio_filename": "reference.wav", "pairs": 800, "steps": 500,
        "teacher_steps": 20,
    }))
    worker.atomic_json(job / "job_state.json", {
        "job_id": job.name, "status": "queued", "stage": "starting",
        "job_dir": str(job), "pairs_done": 0, "pairs_target": 800,
        "updates_done": 0, "steps_target": 500, "checkpoint": "", "error": "",
    })
    return job


class DurableWorkerTests(unittest.TestCase):
    def test_successful_fake_gpu_run_survives_new_client_read(self):
        with tempfile.TemporaryDirectory() as temp:
            job = create_job(Path(temp))
            script = r"""
import json, sys
from pathlib import Path
job = Path(sys.argv[1])
out = job / "llvc-trained"
folder = out / "checkpoints"
folder.mkdir(parents=True)
pth = folder / "G_500.pth"
with open(pth, "wb") as f: f.truncate(2_000_100)
print(f"Teacher conversion: 800/800 accepted; 810 attempted", flush=True)
print("LLVC training 250/500 steps", flush=True)
print("LLVC training 500/500 steps", flush=True)
print(f"Saved checkpoints to {pth} and {folder / 'D_500.pth'}", flush=True)
onnx = out / "quality_voice.onnx"
with open(onnx, "wb") as f: f.truncate(2_000_100)
(out / "model_report.json").write_text(json.dumps({
    "file": onnx.name, "steps": 500, "onnx_io": "audio_in -> audio_out",
    "size_bytes": onnx.stat().st_size
}))
print(f"REAL trained full-size ONNX exported: {onnx}", flush=True)
print(f"MODEL_FILES_READY: ONNX: {onnx}", flush=True)
"""
            code = worker.run_job(job, [sys.executable, "-u", "-c", script, str(job)])
            self.assertEqual(code, 0)
            # No Gradio session exists here. Re-open from the saved job ID.
            saved = json.loads((job / "job_state.json").read_text())
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(saved["pairs_done"], 800)
            self.assertEqual(saved["updates_done"], 500)
            self.assertTrue(Path(saved["checkpoint"]).is_file())
            self.assertTrue(worker.trained_files(job).is_file())
            self.assertIn("800/800 accepted", (job / "job.log").read_text())
            self.assertFalse(list(job.glob("*.zip")))

    def test_failed_work_preserves_checkpoint_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            job = create_job(Path(temp))
            script = r"""
from pathlib import Path
import sys
job = Path(sys.argv[1])
d = job / "llvc-trained" / "checkpoints"
d.mkdir(parents=True)
pth = d / "G_100.pth"
with open(pth, "wb") as f: f.truncate(2_000_100)
print(f"Saved checkpoints to {pth} and {d / 'D_100.pth'}", flush=True)
print("LLVC training 100/500 steps", flush=True)
print("RuntimeError: simulated OOM after checkpoint", flush=True)
sys.exit(1)
"""
            code = worker.run_job(job, [sys.executable, "-u", "-c", script, str(job)])
            self.assertEqual(code, 1)
            saved = worker.load_json(job / "job_state.json")
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["updates_done"], 100)
            self.assertTrue(Path(saved["checkpoint"]).is_file())
            self.assertIsNone(worker.trained_files(job))
            self.assertIn("simulated OOM", (job / "job.log").read_text())

    def test_worker_is_independent_of_parent_python_process(self):
        with tempfile.TemporaryDirectory() as temp:
            job = create_job(Path(temp))
            script = r"""
import sys, time
from pathlib import Path
job = Path(sys.argv[1])
print("Teacher conversion: 600/800 accepted", flush=True)
time.sleep(0.6)
print("Teacher conversion: 800/800 accepted", flush=True)
sys.exit(1)
"""
            # This separate runner is analogous to the Gradio parent launching
            # a worker and ending its HTTP request immediately.
            launch = (
                "import importlib.util,sys;"
                "spec=importlib.util.spec_from_file_location('worker',sys.argv[1]);"
                "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
                "m.run_job(sys.argv[2],[sys.executable,'-u','-c',sys.argv[3],sys.argv[2]])"
            )
            subprocess.Popen(
                [sys.executable, "-u", "-c", launch, str(ROOT / "job_worker.py"),
                 str(job), script],
                start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = worker.load_json(job / "job_state.json")
                if state["status"] == "failed":
                    break
                time.sleep(.15)
            else:
                self.fail("Detached worker did not persist finished job state")
            self.assertEqual(state["pairs_done"], 800)
            self.assertTrue((job / "job.log").is_file())

    def test_atomically_saved_state_survives_restart_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "state.json"
            worker.atomic_json(file, {"status": "running", "pairs_done": 799})
            self.assertEqual(worker.load_json(file)["pairs_done"], 799)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_real_interface_is_rehydratable_and_browser_detached(self):
        code = (ROOT / "hf-space" / "app.py").read_text()
        compile(code, "app.py", "exec")
        compile((ROOT / "job_worker.py").read_text(), "job_worker.py", "exec")
        compile((ROOT / "generate_teacher_pairs.py").read_text(),
                "generate_teacher_pairs.py", "exec")
        compile((ROOT / "one_click_pipeline.py").read_text(),
                "one_click_pipeline.py", "exec")
        self.assertIn("gr.BrowserState(", code)
        self.assertIn('storage_key="llvc-private-space-last-job-v4"', code)
        self.assertIn("start_new_session=True", code)
        self.assertIn("timer.tick(job_status", code)
        self.assertIn("demo.load(job_status", code)
        self.assertIn("allowed_paths=[str(JOBS)]", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
