"""Hugging Face dedicated GPU Space frontend for the real LLVC voice trainer."""
import os
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


def create_voice(uploaded_audio, steps, resume_id):
    # Resume only an existing job in this exact running container/volume.
    resume_id = (resume_id or "").strip()
    if resume_id:
        if not re.fullmatch(r"[a-f0-9]{12}", resume_id):
            yield "Invalid recovery job ID (expected 12 hexadecimal characters).", None
            return
        job_dir = JOBS / resume_id
        if not (job_dir / "dataset" / "manifest.json").is_file():
            yield ("No saved pairs found for " + resume_id +
                   ". Restart/rebuild may have erased temporary storage."), None
            return
        # A completed dataset contains the processed reference itself, so
        # the user doesn't have to upload another two-hour original.
        saved_reference = job_dir / "target_reference.wav"
        if not saved_reference.is_file():
            yield "Recovered job is missing its reference audio; cannot resume safely.", None
            return
        uploaded_audio = str(saved_reference)
    else:
        if not uploaded_audio:
            yield "Upload an audio recording first, or enter a saved job ID.", None
            return
        if not os.path.isfile(uploaded_audio):
            yield "Uploaded file no longer exists. Try uploading again.", None
            return
        job_dir = JOBS / uuid.uuid4().hex[:12]
    # Test CUDA before starting a long job, never launch GPU training on CPU.
    try:
        import torch
        if not torch.cuda.is_available():
            yield "NO NVIDIA GPU DETECTED. Assign a T4 GPU in Space Settings → Hardware, then restart.", None
            return
    except Exception as exc:
        yield f"GPU preflight failed: {exc}", None
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
    yield "\n".join(logs), None
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
        important = (
            line.startswith("[") or
            line.startswith("Teacher conversion") or
            line.startswith("LLVC training ") or
            line.startswith("COMPLETE") or
            "Error" in line or "Traceback" in line
        )
        now = time.monotonic()
        if important and now - last_update > 1:
            yield "\n".join(logs), None
            last_update = now
    return_code = process.wait()
    packages = list(job_dir.glob("*_PTH_ONNX.zip"))
    if return_code == 0 and len(packages) == 1:
        logs.append("SUCCESS — PTH and ONNX are bundled in the single downloadable ZIP.")
        yield "\n".join(logs), str(packages[0])
    else:
        logs.append(f"FAILED (exit {return_code}). Saved pairs/checkpoints remain at {job_dir}.")
        yield "\n".join(logs), None


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
                value=5000, info="5,000 repeats training on the 400 pairs; it does not generate 5,000 pairs."
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
    status = gr.Textbox(label="Live training and export progress", lines=16, max_lines=28, interactive=False)
    bundle = gr.File(label="Download PTH + ONNX (one ZIP)", interactive=False)
    start.click(
        fn=create_voice, inputs=[audio, updates, resume_id], outputs=[status, bundle],
        concurrency_limit=1,
    )
    gr.Markdown(
        "Checkpoints and intermediate data stay on this Space's disk. "
        "Attach a Storage Bucket for persistence across restarts; "
        "otherwise download your ZIP when it finishes. "
        "ONNX inference is verified, but matching a target speaker still requires listening."
    )

demo.queue(default_concurrency_limit=1, max_size=4)
demo.launch(server_name="0.0.0.0", server_port=7860, max_file_size="4gb")
