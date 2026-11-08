#!/usr/bin/env python3
"""Durable, detached GPU job runner for the Hugging Face LLVC Space.

Training is intentionally NOT executed inside a Gradio callback: canceling
the browser queue, reloading the page, or closing the tab cannot cancel this
process. Runtime files survive tab refreshes, but *not* a Space container reset
unless the workspace is backed by persistent storage.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time
import traceback

VALID_JOB = re.compile(r"^[0-9a-f]{12}$")
PAIR_RE = re.compile(r"Teacher conversion:\s*(\d+)/(\d+) accepted")
STEP_RE = re.compile(r"LLVC training (\d+)/(\d+) steps")
CHECKPOINT_RE = re.compile(r"G_(\d+)\.pth")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def trained_files(job):
    """Check artifact files independently; never require ZIP packaging."""
    output = job / "llvc-trained"
    report = output / "model_report.json"
    onnx = None
    if report.is_file():
        try:
            info = load_json(report)
            name = info.get("file")
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+\.onnx", name):
                file = output / name
                if (file.is_file() and file.stat().st_size > 2_000_000
                        and file.stat().st_size == info.get("size_bytes")
                        and info.get("onnx_io") == "audio_in -> audio_out"):
                    onnx = file
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return onnx


def parse_log_line(state, line):
    """Update counters only from evidence, not elapsed time guesses."""
    match = PAIR_RE.search(line)
    if match:
        state["pairs_done"] = int(match.group(1))
        state["stage"] = "Creating quality-checked voice pairs"
    match = STEP_RE.search(line)
    if match:
        state["updates_done"] = int(match.group(1))
        state["stage"] = "Fine-tuning the LLVC generator"
    if "[4/5]" in line or "Genuine teacher corpus:" in line:
        state["pairs_done"] = int(state.get("pairs_target", 0))
        state["stage"] = "Training the neural model"
    if "LLVC training completed" in line and "gradient updates" in line:
        state["updates_done"] = int(state.get("steps_target", 0))
    if "Saved checkpoints to " in line:
        saved = CHECKPOINT_RE.search(line)
        if saved:
            candidate = (Path(state["job_dir"]) / "llvc-trained"
                         / "checkpoints" / f"G_{saved.group(1)}.pth")
            if candidate.is_file() and candidate.stat().st_size >= 2_000_000:
                state["checkpoint"] = str(candidate)
    if "REAL trained full-size ONNX exported:" in line:
        state["stage"] = "ONNX exported; validating downloads"
    if "MODEL_FILES_READY: " in line:
        state["stage"] = "Model artifacts available"
    if "Traceback" in line or "Error:" in line or "FAILED" in line:
        state["last_error_line"] = line[-500:]
    state["updated_at"] = time.time()
    return state


def command_for_job(job, config):
    root = Path(config.get("repo", "/app/irl-voice"))
    return [
        sys.executable,
        str(root / "voice-training" / "one_click_pipeline.py"),
        "--audio", str(job / "input_audio" / config["audio_filename"]),
        "--run-dir", str(job),
        "--repo", str(root),
        "--seed-vc", str(config.get("seed_vc", "/app/seed-vc")),
        "--llvc", str(config.get("llvc", "/app/llvc-upstream")),
        "--cache", str(config.get("cache", "/home/user/voice-data/voice-cache")),
        "--pairs", str(int(config["pairs"])),
        "--steps", str(int(config["steps"])),
        "--teacher-steps", str(int(config["teacher_steps"])),
        "--name", str(config.get("model_name") or "custom_voice"),
        "--batch", "2", "--skip-zip",
    ]


def run_job(job, command=None):
    job = Path(job).resolve()
    config = load_json(job / "job_config.json")
    state_path = job / "job_state.json"
    state = load_json(state_path)
    state.update(status="running", stage="Launching worker", worker_pid=os.getpid(),
                 updated_at=time.time(), error="", job_dir=str(job))
    atomic_json(state_path, state)
    cmd = command if command is not None else command_for_job(job, config)
    log_path = job / "job.log"

    with open(log_path, "a", encoding="utf-8", buffering=1) as logfile:
        logfile.write(f"\n=== Worker started PID={os.getpid()} at {time.ctime()} ===\n")
        process = subprocess.Popen(
            cmd, cwd=str(job), env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        state["pipeline_pid"] = process.pid
        atomic_json(state_path, state)
        fd = process.stdout.fileno()
        partial = b""
        last_save = time.monotonic()

        while True:
            readable, _, _ = select.select([fd], [], [], 1.5)
            if readable:
                chunk = os.read(fd, 65536)
                if chunk:
                    partial += chunk
                    while b"\n" in partial:
                        raw, partial = partial.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace").strip()
                        logfile.write(line + "\n")
                        parse_log_line(state, line)
                else:
                    break
            now = time.monotonic()
            if now - last_save > 4:
                atomic_json(state_path, state)
                last_save = now

        if partial:
            line = partial.decode("utf-8", errors="replace").strip()
            logfile.write(line + "\n")
            parse_log_line(state, line)
        code = process.wait()
        state["return_code"] = code
        state["updated_at"] = time.time()
        state["finished_at"] = time.time()
        onnx = trained_files(job)
        checkpoint = state.get("checkpoint")
        # If training printed no checkpoint announcement, recover the final
        # checkpoint after a clean pipeline exit.
        final = (job / "llvc-trained" / "checkpoints" /
                 f"G_{int(config['steps'])}.pth")
        if final.is_file() and final.stat().st_size >= 2_000_000:
            checkpoint = str(final)
            state["checkpoint"] = checkpoint
        if code == 0 and onnx and checkpoint:
            state.update(status="complete", stage="Complete: PTH and ONNX ready",
                         updates_done=int(config["steps"]),
                         pairs_done=int(config["pairs"]))
        else:
            state.update(status="failed", stage="Stopped; saved files may still be available",
                         error=(f"Process returned exit code {code}. "
                                "Open the saved job log for details."))
        atomic_json(state_path, state)
        logfile.write(f"=== Worker finished, exit={code} ===\n")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = args.job.resolve()
    if not VALID_JOB.fullmatch(job.name):
        parser.error("Invalid job ID")
    job.mkdir(parents=True, exist_ok=True)
    # The training lock is shared by all jobs in the Space. CPU status
    # requests and downloads are not blocked by it.
    lock_file = job.parent / ".gpu_training.lock"
    with open(lock_file, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = load_json(job / "job_state.json")
            state.update(status="failed", error="Another GPU training job is already running",
                         stage="GPU busy", updated_at=time.time())
            atomic_json(job / "job_state.json", state)
            return 2
        try:
            return run_job(job)
        except Exception as exc:
            with open(job / "job.log", "a", encoding="utf-8") as logfile:
                traceback.print_exc(file=logfile)
            state = load_json(job / "job_state.json")
            state.update(status="failed", error=repr(exc),
                         stage="Worker crashed; saved files remain on disk",
                         updated_at=time.time(), finished_at=time.time())
            atomic_json(job / "job_state.json", state)
            return 1


if __name__ == "__main__":
    sys.exit(main())
