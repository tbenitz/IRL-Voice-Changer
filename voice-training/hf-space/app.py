"""Hugging Face dedicated GPU Space frontend for the real LLVC voice trainer."""
import os
from html import escape
from collections import deque
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile

import gradio as gr

ROOT = Path("/app")
# /data is not writable unless an appropriate storage volume is mounted.
# Use the pre-created user-owned directory, even on Spaces without /data.
WORKSPACE = Path(os.environ.get("VOICE_WORKSPACE", "/home/user/voice-data"))
JOBS = WORKSPACE / "voice-jobs"
CACHE = WORKSPACE / "voice-cache"


def progress_html(pairs=0, updates=0, total_steps=500, stage="Ready",
                  exported=False, packaged=False):
    """Two real counters, not a fabricated percentage of total wall time."""
    pairs = max(0, min(400, int(pairs)))
    total_steps = max(1, int(total_steps))
    updates = max(0, min(total_steps, int(updates)))
    state = ("Bundle ready" if packaged else
             "ONNX exported" if exported else "Waiting")
    return (
        '<div style="display:grid;gap:14px;padding:12px 2px">'
        f'<strong>{escape(stage)}</strong>'
        '<div style="display:grid;gap:5px">'
        f'<span>Teacher pairs: <b>{pairs} / 400</b></span>'
        f'<progress value="{pairs}" max="400" style="width:100%;height:18px"></progress>'
        '</div><div style="display:grid;gap:5px">'
        f'<span>Neural training: <b>{updates} / {total_steps} updates</b></span>'
        f'<progress value="{updates}" max="{total_steps}" style="width:100%;height:18px"></progress>'
        '</div>'
        f'<span>Export and package: {state}</span>'
        '<small>Bars track completed work, not estimated remaining minutes.</small>'
        '</div>'
    )


def create_voice(uploaded_audio, steps, resume_id):
    steps = int(steps)
    progress = progress_html(total_steps=steps)
    # Resume only an existing job in this exact running container/volume.
    resume_id = (resume_id or "").strip()
    if resume_id:
        if not re.fullmatch(r"[a-f0-9]{12}", resume_id):
            yield "Invalid recovery job ID (expected 12 hexadecimal characters).", None, progress
            return
        job_dir = JOBS / resume_id
        if not (job_dir / "dataset" / "manifest.json").is_file():
            yield ("No saved pairs found for " + resume_id +
                   ". Restart/rebuild may have erased temporary storage."), None, progress
            return
        # A completed dataset contains the processed reference itself, so
        # the user doesn't have to upload another two-hour original.
        saved_reference = job_dir / "target_reference.wav"
        if not saved_reference.is_file():
            yield "Recovered job is missing its reference audio; cannot resume safely.", None, progress
            return
        uploaded_audio = str(saved_reference)
    else:
        if not uploaded_audio:
            yield "Upload an audio recording first, or enter a saved job ID.", None, progress
            return
        if not os.path.isfile(uploaded_audio):
            yield "Uploaded file no longer exists. Try uploading again.", None, progress
            return
        job_dir = JOBS / uuid.uuid4().hex[:12]
    # Test CUDA before starting a long job, never launch GPU training on CPU.
    try:
        import torch
        if not torch.cuda.is_available():
            yield "NO NVIDIA GPU DETECTED. Assign a T4 GPU in Space Settings → Hardware, then restart.", None, progress
            return
    except Exception as exc:
        yield f"GPU preflight failed: {exc}", None, progress
        return
    job_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "/app/irl-voice/voice-training/one_click_pipeline.py",
        "--audio", str(uploaded_audio),
        "--run-dir", str(job_dir),
        "--repo", "/app/irl-voice",
        "--seed-vc", "/app/seed-vc",
        "--llvc", "/app/llvc-upstream",
        "--cache", str(CACHE),
        "--pairs", "400",
        "--steps", str(int(steps)),
        "--batch", "2",
        "--teacher-steps", "10",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    logs = deque(maxlen=90)
    logs.append(f"Job: {job_dir.name}" + (" (RESUMING saved 400 pairs)" if resume_id else ""))
    logs.append("Starting GPU training: 400 accepted pairs, then full LLVC PTH and ONNX.")
    completed_pairs = 400 if resume_id else 0
    completed_steps = 0
    exported = False
    yield "\n".join(logs), None, progress_html(completed_pairs, 0, steps, "Starting GPU job")
    process = subprocess.Popen(
        command, cwd="/app", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    last_update = 0.0
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        logs.append(line)
        counter_changed = False
        pair_match = re.search(r"Teacher conversion:\s*(\d+)/400 accepted", line)
        if pair_match:
            completed_pairs = min(400, int(pair_match.group(1)))
            counter_changed = True
            stage = "Building paired speech (Seed-VC)"
        elif "[4/5]" in line or "Genuine teacher corpus:" in line:
            completed_pairs = 400
            stage = "Training LLVC model"
            counter_changed = True
        step_match = re.search(r"LLVC training (\d+)/(\d+) steps", line)
        if step_match:
            completed_steps = min(steps, int(step_match.group(1)))
            stage = "Training LLVC model"
            counter_changed = True
        if "LLVC training completed" in line and "gradient updates" in line:
            completed_steps = steps
            counter_changed = True
        if "REAL trained full-size ONNX exported:" in line:
            exported = True
            completed_steps = steps
            stage = "ONNX verified; packaging model"
            counter_changed = True
        if "COMPLETE — ZIP:" in line:
            exported = True
            completed_steps = steps
            completed_pairs = 400
            stage = "Model ZIP completed"
            counter_changed = True
        important = (
            line.startswith("[") or
            line.startswith("Teacher conversion") or
            line.startswith("LLVC training ") or
            line.startswith("COMPLETE") or
            "Error" in line or "Traceback" in line
        )
        now = time.monotonic()
        if (counter_changed or important) and now - last_update > 0.8:
            stage = locals().get("stage", "Working")
            yield "\n".join(logs), None, progress_html(
                completed_pairs, completed_steps, steps, stage, exported
            )
            last_update = now
    return_code = process.wait()
    packages = list(job_dir.glob("*_PTH_ONNX.zip"))
    if return_code == 0 and len(packages) == 1:
        logs.append("SUCCESS — PTH and ONNX are bundled in the single downloadable ZIP.")
        yield "\n".join(logs), str(packages[0]), progress_html(
            400, steps, steps, "Complete — ZIP ready for download", True, True
        )
    else:
        logs.append(f"FAILED (exit {return_code}). Saved pairs/checkpoints may remain at {job_dir}.")
        yield "\n".join(logs), None, progress_html(
            completed_pairs, completed_steps, steps, "Stopped — see error in logs", exported
        )

def recover_completed_model(job_id):
    """CPU-only ZIP recovery from already exported PTH/ONNX; no training."""
    job_id = (job_id or "").strip()
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        return "Enter the 12-character ID from the failed job log.", None
    directory = JOBS / job_id
    report = directory / "llvc-trained" / "model_report.json"
    if not report.is_file():
        return ("No saved export report in this job. "
                "The Space may have lost ephemeral files when paused/rebuilt."), None
    command = [
        sys.executable,
        "/app/irl-voice/voice-training/one_click_pipeline.py",
        "--run-dir", str(directory), "--package-only",
    ]
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=240, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    files = list(directory.glob("*_PTH_ONNX.zip"))
    if result.returncode == 0 and len(files) == 1:
        return "Recovered existing PTH and ONNX; no GPU training used.", str(files[0])
    return "ZIP recovery failed:\n" + result.stdout[-5000:], None


def backup_pairs(job_id):
    """Download the already generated teacher corpus before a rebuild."""
    job_id = (job_id or "").strip()
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        return "Enter the 12-character job ID printed in your training logs.", None
    folder = JOBS / job_id
    manifest = folder / "dataset" / "manifest.json"
    if not manifest.is_file():
        return "No pairs found for this job; storage may have been reset.", None
    archive = folder / "teacher_pairs_BACKUP.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for file in sorted((folder / "dataset").rglob("*.wav")):
            bundle.write(file, arcname=str(file.relative_to(folder)))
        bundle.write(manifest, arcname="dataset/manifest.json")
        for name in ("target_reference.wav", "reference_selection.json"):
            f = folder / name
            if f.is_file():
                bundle.write(f, arcname=name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            return "Backup file failed integrity check.", None
    return f"Backup ready for saved job {job_id}.", str(archive)


with gr.Blocks(title="Custom Neural Voice Trainer") as demo:
    gr.Markdown(
        "# Custom neural voice trainer\n"
        "Upload **one recording (2 seconds to 2 hours)**. "
        "The app analyzes **at most its first 3 minutes**, "
        "builds **400 accepted matching voice pairs**, fine-tunes the "
        "pretrained LLVC model, and exports **PTH + ONNX in one ZIP**.\n\n"
        "Runs on dedicated NVIDIA GPU hardware. "
        "For 2–3-second voice references, cloning quality may be poor. "
        "Only upload recordings you have permission to process."
    )
    with gr.Row():
        audio = gr.File(label="Upload your voice recording", file_types=["audio"], type="filepath")
        with gr.Column():
            updates = gr.Dropdown(
                label="Training updates", choices=[500, 1000, 2000, 5000, 10000],
                value=500, info="500 = quick baseline; 5,000 = longer fine-tuning. These are updates on the same 400 pairs."
            )
            start = gr.Button("Train voice and build ZIP", variant="primary")
    with gr.Accordion("Recover an interrupted job (optional)", open=False):
        resume_id = gr.Textbox(
            label="Existing job ID",
            placeholder="e.g. 699d49e13cea",
            info="Resume on the same live container or mounted storage; don't re-upload or regenerate 400 pairs."
        )
        backup = gr.Button("Download saved pairs before rebuilding", variant="secondary")
        backup_message = gr.Textbox(label="Backup status", interactive=False)
        backup_file = gr.File(label="Saved pairs ZIP", interactive=False)
        backup.click(fn=backup_pairs, inputs=resume_id,
                     outputs=[backup_message, backup_file])
        salvage = gr.Button("Recover completed PTH + ONNX as ZIP (no GPU training)")
        salvage_message = gr.Textbox(label="ZIP recovery status", interactive=False)
        salvage_file = gr.File(label="Recovered model ZIP", interactive=False)
        salvage.click(fn=recover_completed_model, inputs=resume_id,
                      outputs=[salvage_message, salvage_file])
    progress_panel = gr.HTML(value=progress_html(), label="Live progress")
    status = gr.Textbox(label="Detailed training log", lines=16, max_lines=28, interactive=False)
    bundle = gr.File(label="Download PTH + ONNX (one ZIP)", interactive=False)
    start.click(
        fn=create_voice, inputs=[audio, updates, resume_id],
        outputs=[status, bundle, progress_panel], concurrency_limit=1,
    )
    gr.Markdown(
        "Checkpoints and intermediate data stay on this Space's disk. "
        "Attach a Storage Bucket for persistence across restarts; "
        "otherwise download your ZIP when it finishes. "
        "ONNX inference is verified, but matching a target speaker still requires listening."
    )

demo.queue(default_concurrency_limit=1, max_size=4)
demo.launch(server_name="0.0.0.0", server_port=7860, max_file_size="4gb")
