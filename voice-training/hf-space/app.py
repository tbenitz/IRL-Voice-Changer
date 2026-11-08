"""Private Hugging Face LLVC trainer: durable jobs, browser-refresh recovery.

The browser only starts / observes jobs. The detached job_worker.py executes
the expensive conversion/training separately, so closing or reloading a tab
does not cancel an active job. Container restarts still erase ephemeral data.
"""
from collections import deque
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

import gradio as gr

BASE = Path("/app")


def refresh_training_code():
    """Bust Docker's cached git-clone layer once at app startup.

    The existing Dockerfile clones GitHub in a cached image layer BEFORE
    it copies app.py. Changing app.py alone would otherwise ship OLD
    training scripts even though GitHub has the updated worker/quality code.
    """
    checkout = BASE / "irl-voice"
    required = checkout / "voice-training" / "job_worker.py"
    if not (checkout / ".git").is_dir():
        return "Training checkout missing; refusing paid work."
    try:
        fetched = subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth", "1",
             "origin", "main"],
            capture_output=True, text=True, timeout=90,
        )
        if fetched.returncode != 0:
            raise RuntimeError(fetched.stderr[-500:])
        reset = subprocess.run(
            ["git", "-C", str(checkout), "reset", "--hard", "FETCH_HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if reset.returncode != 0:
            raise RuntimeError(reset.stderr[-500:])
    except Exception as exc:
        if not required.is_file():
            return f"Could not load new training code: {exc}"
    if not required.is_file():
        return "Missing detached job_worker.py after sync; refused to start paid job."
    pipeline = checkout / "voice-training" / "one_click_pipeline.py"
    teacher = checkout / "voice-training" / "generate_teacher_pairs.py"
    if ("--skip-zip" not in pipeline.read_text() or
            "quality_metrics" not in teacher.read_text()):
        return "Wrong training source revision; refusing GPU run until it is updated."
    return ""


CODE_PROBLEM = refresh_training_code()
ID_RE = re.compile(r"^[0-9a-f]{12}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_-]+\.onnx$")
PROFILES = {
    "High fidelity — 800 pairs, 5,000 updates": (800, 5000, 20),
    "Studio extended — 1,200 pairs, 8,000 updates": (1200, 8000, 24),
    "Lower-cost trial — 400 pairs, 1,000 updates": (400, 1000, 10),
}
DEFAULT_PROFILE = "High fidelity — 800 pairs, 5,000 updates"


def select_workspace():
    desired = os.environ.get("VOICE_WORKSPACE", "").strip()
    if desired and desired.lower() != "auto":
        target = Path(desired).resolve()
        if not target.is_dir() or not os.access(target, os.W_OK):
            raise RuntimeError(f"VOICE_WORKSPACE is not writable: {target}")
        return target
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return Path("/data")
    return Path("/home/user/voice-data")


WORKSPACE = select_workspace()
for suffix in ("voice-jobs", "voice-cache", "hf-cache", "torch-cache"):
    (WORKSPACE / suffix).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(WORKSPACE / "hf-cache")
os.environ["TORCH_HOME"] = str(WORKSPACE / "torch-cache")
JOBS = WORKSPACE / "voice-jobs"
CACHE = WORKSPACE / "voice-cache"


def safe_dir(job_id):
    job_id = str(job_id or "").strip()
    if not ID_RE.fullmatch(job_id):
        return None
    folder = JOBS / job_id
    if not folder.is_dir():
        return None
    # Recover jobs created by the previous version (before job_config.json).
    if ((folder / "job_config.json").is_file() or
            (folder / "dataset" / "manifest.json").is_file() or
            (folder / "llvc-trained" / "checkpoints").is_dir()):
        return folder
    return None


def load_state(job):
    try:
        return json.loads((job / "job_state.json").read_text())
    except (OSError, ValueError, TypeError):
        # Older voice-training jobs had no saved state. Still offer their
        # finished model artifacts instead of requiring a repeat upload.
        manifest_path = job / "dataset" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            manifest = {}
        target = int(manifest.get("requested", 400))
        count = sum(int(manifest.get(k, 0)) for k in ("train", "val", "dev"))
        return {
            "status": "interrupted", "stage": "Earlier run detected — inspect saved model files",
            "pairs_done": count, "pairs_target": target,
            "updates_done": 0, "steps_target": 500,
            "error": "", "checkpoint": "",
        }


def write_state(job, state):
    temp = job / f"job_state.{os.getpid()}.tmp"
    temp.write_text(json.dumps(state, indent=2))
    os.replace(temp, job / "job_state.json")


def worker_running(state):
    pid = state.get("worker_pid")
    if not isinstance(pid, int) or pid <= 1:
        # The worker gets a short start-up allowance. A failed launcher must
        # not block all future jobs forever with a stale "queued" entry.
        age = time.time() - float(state.get("updated_at", 0) or 0)
        return state.get("status") == "queued" and age < 45
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def saved_files(job, state):
    """Check each artifact separately. Never require a ZIP to exist."""
    output = job / "llvc-trained"
    checkpoints = output / "checkpoints"
    chosen = None
    safe_reported = state.get("checkpoint")
    if isinstance(safe_reported, str):
        candidate = Path(safe_reported)
        if candidate.parent == checkpoints and candidate.is_file() and candidate.stat().st_size >= 2_000_000:
            chosen = candidate
    if chosen is None:
        # Some LLVC builds suppress the logger's "Saved checkpoints" line.
        # A checkpoint is still downloadable while training once it has
        # remained untouched long enough to avoid a partial torch.save file.
        terminal = state.get("status") in ("failed", "complete", "interrupted")
        now = time.time()
        files = sorted(
            ((int(m.group(1)), f) for f in checkpoints.glob("G_*.pth")
             if (m := re.fullmatch(r"G_(\d+)\.pth", f.name))
             and f.stat().st_size >= 2_000_000
             and (terminal or now - f.stat().st_mtime >= 15.0)),
            key=lambda item: item[0]
        )
        chosen = files[-1][1] if files else None
    validated = None
    report_file = output / "model_report.json"
    if report_file.is_file():
        try:
            report = json.loads(report_file.read_text())
            name = report.get("file")
            if (isinstance(name, str) and MODEL_RE.fullmatch(name)
                and type(report.get("steps")) is int
                and report.get("onnx_io") == "audio_in -> audio_out"):
                file = output / name
                companion = checkpoints / f"G_{report['steps']}.pth"
                if (file.is_file() and companion.is_file()
                    and file.stat().st_size > 2_000_000
                    and file.stat().st_size == report.get("size_bytes")):
                    validated = file
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return (str(chosen) if chosen else None,
            str(validated) if validated else None)


def recent_jobs():
    jobs = []
    for job in JOBS.iterdir():
        if not job.is_dir() or not ID_RE.fullmatch(job.name):
            continue
        if not safe_dir(job.name):
            continue
        state = load_state(job)
        jobs.append((job.stat().st_mtime,
                     f"{job.name} | {state.get('status', 'unknown')} | "
                     f"{state.get('pairs_done', 0)}/{state.get('pairs_target', '?')} pairs"))
    jobs.sort(reverse=True)
    return [label for _, label in jobs[:30]]


def tail_logs(job, limit=110):
    file = job / "job.log"
    if not file.is_file():
        return "Job queued. Training logs will appear here."
    with open(file, "rb") as handle:
        handle.seek(0, 2)
        length = handle.tell()
        handle.seek(max(0, length - 24000))
        raw = handle.read().decode("utf-8", errors="replace")
    return "\n".join(raw.splitlines()[-limit:])


def progress_html(state, pth=None, onnx=None):
    pairs = max(0, int(state.get("pairs_done") or 0))
    target = max(1, int(state.get("pairs_target") or 800))
    updates = max(0, int(state.get("updates_done") or 0))
    total = max(1, int(state.get("steps_target") or 5000))
    status = escape(str(state.get("status", "ready")).upper())
    stage = escape(str(state.get("stage", "Upload one recording to begin.")))
    color = "#269e6c" if state.get("status") == "complete" else "#7487ac"
    return (
        f'<div style="display:grid;gap:12px;padding:12px 0">'
        f'<div><b style="color:{color}">{status}</b> — {stage}</div>'
        f'<div>Quality-checked training pairs: <b>{min(pairs,target)}/{target}</b>'
        f'<progress value="{min(pairs,target)}" max="{target}"'
        f' style="width:100%;height:18px"></progress></div>'
        f'<div>Neural weight updates: <b>{min(updates,total)}/{total}</b>'
        f'<progress value="{min(updates,total)}" max="{total}"'
        f' style="width:100%;height:18px"></progress></div>'
        f'<div>Saved checkpoint: <b>{"Ready to download" if pth else "Not yet"}</b>'
        f' | Validated ONNX: <b>{"Ready to download" if onnx else "Not yet"}</b></div>'
        '<small>Progress is counted from saved server-side work. Refreshing this page does not restart training.</small>'
        '</div>'
    )


def audition_files(job, verified_onnx):
    """A/B examples created by the real model exporter, no new inference."""
    if not verified_onnx:
        return (None, None, None)
    directory = job / "llvc-trained"
    names = ("AB_original.wav", "AB_teacher.wav", "AB_trained_ONNX.wav")
    files = [directory / name for name in names]
    return tuple(str(file) if file.is_file() and file.stat().st_size > 1000 else None
                 for file in files)


def job_status(job_id, last_seen=None):
    """Read status and restore artifacts without running a training function."""
    last_seen = last_seen or ("", "", "", "", "", "")
    if len(last_seen) != 6:
        last_seen = ("", "", "", "", "", "")
    job = safe_dir(job_id)
    if job is None:
        return (str(job_id or ""), progress_html({}),
                "No job selected. Select a saved job below if the browser refreshed.",
                None, None, None, None, None,
                ("", "", "", "", "", ""))
    state = load_state(job)
    if state.get("status") in ("running", "queued") and not worker_running(state):
        state.update(status="interrupted",
                     stage="Worker stopped or launch timed out; saved files may remain")
        write_state(job, state)
    pth, onnx = saved_files(job, state)
    original, teacher, student = audition_files(job, onnx)
    current = (job.name, pth or "", onnx or "",
               original or "", teacher or "", student or "")
    update_values = [
        value if current[0] != last_seen[0] or value != previous else gr.update()
        for value, previous in zip((pth, onnx, original, teacher, student),
                                   last_seen[1:])
    ]
    description = tail_logs(job)
    if state.get("error"):
        description += "\n\nLAST ERROR: " + str(state["error"])
    return (job.name, progress_html(state, pth, onnx), description,
            *update_values, current)


def launch_worker(job):
    log_file = job / "launcher.log"
    with open(log_file, "ab") as log:
        subprocess.Popen(
            [sys.executable, str(BASE / "irl-voice" / "voice-training" / "job_worker.py"),
             "--job", str(job)],
            cwd=str(job), env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )


def any_gpu_job_active():
    # A private Space has one T4. Reject accidental concurrent jobs before
    # anyone pays for duplicate GPU work.
    for label in recent_jobs():
        job = safe_dir(label[:12])
        if job:
            state = load_state(job)
            if state.get("status") in ("queued", "running") and worker_running(state):
                return job.name
    return None


def start_job(audio, profile, previous_id):
    if not audio or not Path(audio).is_file():
        return previous_id or "", previous_id or "", "Upload an audio recording first."
    if profile not in PROFILES:
        return previous_id or "", previous_id or "", "Select a valid training profile."
    existing = any_gpu_job_active()
    if existing:
        return existing, existing, (
            f"Job {existing} is already running. Reopened it instead of starting another paid run."
        )
    try:
        import torch
        if not torch.cuda.is_available():
            return previous_id or "", previous_id or "", (
                "No NVIDIA GPU detected. Do not start paid training on CPU-only hardware."
            )
    except Exception as exc:
        return previous_id or "", previous_id or "", f"GPU check failed: {exc}"
    if CODE_PROBLEM:
        return previous_id or "", previous_id or "", (
            "Cannot start paid training: " + CODE_PROBLEM
        )
    pairs, steps, teacher = PROFILES[profile]
    job_id = uuid.uuid4().hex[:12]
    job = JOBS / job_id
    job.mkdir(parents=True, exist_ok=False)
    audio_dir = job / "input_audio"
    audio_dir.mkdir()
    source_name = Path(audio).name
    suffix = Path(source_name).suffix.lower()
    if suffix not in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".mp4", ".aac"):
        return previous_id or "", previous_id or "", "Unsupported audio format."
    filename = "reference" + suffix
    # Keep the user's chosen model name instead of producing reference.onnx
    model_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem).strip("_").lower()
    model_name = (model_name or "voice")[:50] + "_" + job_id[:6]
    try:
        # Uploaded Gradio temp files can disappear after a browser refresh.
        # Copy the voice recording INTO the job before returning job ID.
        shutil.copy2(audio, audio_dir / filename)
        config = {
            "job_id": job_id, "audio_filename": filename,
            "model_name": model_name,
            "pairs": pairs, "steps": steps, "teacher_steps": teacher,
            "repo": str(BASE / "irl-voice"),
            "seed_vc": str(BASE / "seed-vc"),
            "llvc": str(BASE / "llvc-upstream"), "cache": str(CACHE),
            "created_at": time.time(), "profile": profile,
        }
        (job / "job_config.json").write_text(json.dumps(config, indent=2))
        write_state(job, {
            "job_id": job_id, "job_dir": str(job), "status": "queued",
            "stage": "Job saved. Launching detached worker.",
            "pairs_done": 0, "pairs_target": pairs,
            "updates_done": 0, "steps_target": steps,
            "updated_at": time.time(), "checkpoint": "", "error": "",
        })
        launch_worker(job)
        return job_id, job_id, (
            f"Job {job_id} saved and started independently of this browser. "
            "You may refresh and return; select the job in Saved Jobs to see the same progress. "
            "Downloads appear separately as soon as ready."
        )
    except Exception as exc:
        return previous_id or "", previous_id or "", f"Could not save/start job: {exc}"


def retry_job(job_id):
    job = safe_dir(job_id)
    if not job:
        return job_id, job_id, "Saved job no longer exists; no GPU work started."
    state = load_state(job)
    completed_pth, completed_onnx = saved_files(job, state)
    if completed_pth and completed_onnx:
        return job_id, job_id, (
            "PTH and validated ONNX are already saved; use the download controls. "
            "No new training has been started."
        )
    if state.get("status") in ("running", "queued") and worker_running(state):
        return job_id, job_id, "Already running; nothing was restarted."
    existing = any_gpu_job_active()
    if existing:
        return existing, existing, "Another GPU job is still running."
    if not (job / "target_reference.wav").is_file():
        return job_id, job_id, ("The previously selected reference is missing. "
                               "Cannot safely resume this job.")
    if not (job / "dataset" / "manifest.json").is_file():
        return job_id, job_id, ("The generated pairs are missing. "
                               "Cannot safely resume without repeating them.")
    if not (job / "job_config.json").is_file():
        return job_id, job_id, (
            "This is a legacy job. Its downloads can be recovered, but its "
            "training cannot be safely resumed automatically."
        )
    config = json.loads((job / "job_config.json").read_text())
    if not (job / "input_audio" / config["audio_filename"]).is_file():
        return job_id, job_id, "Saved original recording is missing."
    try:
        import torch
        if not torch.cuda.is_available():
            return job_id, job_id, "No T4 GPU available to resume training."
    except Exception as exc:
        return job_id, job_id, f"GPU check failed: {exc}"
    state.update(status="queued", stage="Retry requested on existing saved job",
                 error="", updated_at=time.time(), worker_pid=None)
    write_state(job, state)
    try:
        launch_worker(job)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        write_state(job, state)
        return job_id, job_id, f"Could not resume: {exc}"
    return job_id, job_id, f"Reusing job {job_id}; existing dataset and model files will be checked before work."


def open_saved_job(selection):
    job_id = (selection or "").strip()[:12]
    if not safe_dir(job_id):
        return "", "", "No saved job was found with that ID."
    return job_id, job_id, "Saved job restored. No new training started."


def refresh_choices():
    return gr.update(choices=recent_jobs())


def make_app():
    with gr.Blocks(title="LLVC Neural Voice Studio") as demo:
        browser_id = gr.BrowserState(
            default_value="", storage_key="llvc-private-space-last-job-v4",
            secret="LLVC-private-space-browser-job-id-stable-v4",
        )
        last_seen = gr.State(("", "", "", "", "", ""))
        if CODE_PROBLEM:
            gr.Markdown("**Training disabled before GPU use:** " + CODE_PROBLEM)
        gr.Markdown(
            "# LLVC Neural Voice Studio\n"
            "Your training is a **server-side job**, not a browser session. "
            "Closing the tab or refreshing the page won't cancel a running job. "
            "Your last job ID is saved in this browser and the server lists jobs still on disk."
        )
        with gr.Row():
            upload = gr.File(label="Upload reference voice (2 seconds–2 hours)",
                             file_types=["audio"], type="filepath")
            with gr.Column():
                profile = gr.Radio(
                    choices=list(PROFILES), value=DEFAULT_PROFILE,
                    label="Quality / compute profile",
                    info="Higher pair counts and teacher steps mean significantly more GPU processing. "
                         "Choose the lower-cost trial before an expensive quality run if desired.",
                )
                begin = gr.Button("Start training (save independent job)", variant="primary")
                retry = gr.Button("Resume saved job (only if stopped)", variant="secondary")
        message = gr.Textbox(label="System message", interactive=False)
        with gr.Row():
            job_id = gr.Textbox(label="Saved job ID", interactive=False)
            with gr.Column():
                job_choice = gr.Dropdown(choices=recent_jobs(), label="Saved jobs on this Space")
                with gr.Row():
                    reload_jobs = gr.Button("Refresh jobs")
                    open_job = gr.Button("Open selected job")
        bars = gr.HTML(value=progress_html({}))
        with gr.Row():
            pth = gr.File(label="Download trained PTH (when checkpoint saved)", interactive=False)
            onnx = gr.File(label="Download verified ONNX (when export finishes)", interactive=False)
        with gr.Accordion("Voice quality A/B comparison (once ONNX is validated)", open=False):
            with gr.Row():
                original_audio = gr.Audio(label="Original source speech", interactive=False)
                teacher_audio = gr.Audio(label="Seed-VC target conversion", interactive=False)
                model_audio = gr.Audio(label="Your trained LLVC ONNX conversion", interactive=False)
            gr.Markdown("Compare the three clips by ear. An ONNX export passing inference checks "
                        "does not by itself establish that the target speaker was learned.")
        logs = gr.Textbox(label="Persistent training log (survives browser refresh)",
                          lines=17, max_lines=26, interactive=False)
        gr.Markdown(
            "**No ZIP step. No additional storage purchase required.** "
            + ("Persistent mounted storage is in use. " if WORKSPACE == Path("/data")
               else "Files use temporary Space disk, so pausing/rebuilding the Space can erase "
                    "jobs. Download PTH/ONNX before stopping hardware. ")
            + "The voice's similarity must be checked by listening to comparison samples; "
              "more training data does not guarantee a matching voice."
        )
        timer = gr.Timer(value=3.0)
        # Starting and retrying are SHORT callbacks, never 2–5 hour queued tasks.
        begin.click(start_job, inputs=[upload, profile, browser_id],
                    outputs=[browser_id, job_id, message],
                    queue=False, concurrency_limit=1)
        retry.click(retry_job, inputs=[browser_id],
                    outputs=[browser_id, job_id, message],
                    queue=False, concurrency_limit=1)
        open_job.click(open_saved_job, inputs=[job_choice],
                       outputs=[browser_id, job_id, message], queue=False)
        reload_jobs.click(refresh_choices, outputs=[job_choice], queue=False)
        # Rehydration reads BrowserState from localStorage on each page load.
        # The timer can continue polling after any disconnect/reconnect.
        demo.load(job_status, inputs=[browser_id, last_seen],
                  outputs=[job_id, bars, logs, pth, onnx,
                           original_audio, teacher_audio, model_audio, last_seen], queue=False)
        timer.tick(job_status, inputs=[browser_id, last_seen],
                   outputs=[job_id, bars, logs, pth, onnx,
                            original_audio, teacher_audio, model_audio, last_seen],
                   queue=False, show_progress="hidden")
    return demo


if __name__ == "__main__":
    app = make_app()
    app.queue(default_concurrency_limit=1, max_size=4)
    app.launch(server_name="0.0.0.0", server_port=7860, max_file_size="4gb",
               allowed_paths=[str(JOBS)])
