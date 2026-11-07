"""Hugging Face dedicated GPU Space frontend for the real LLVC voice trainer."""
import os
from collections import deque
from pathlib import Path
import subprocess
import sys
import time
import uuid

import gradio as gr

ROOT = Path("/app")
JOBS = Path("/data/voice-jobs")
CACHE = Path("/data/voice-cache")


def create_voice(uploaded_audio, steps):
    if not uploaded_audio:
        yield "Upload an audio recording first.", None
        return
    if not os.path.isfile(uploaded_audio):
        yield "Uploaded file no longer exists. Try uploading again.", None
        return
    job_dir = JOBS / uuid.uuid4().hex[:12]
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
    status = gr.Textbox(label="Live training and export progress", lines=16, max_lines=28, interactive=False)
    bundle = gr.File(label="Download PTH + ONNX (one ZIP)", interactive=False)
    start.click(
        fn=create_voice, inputs=[audio, updates], outputs=[status, bundle],
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
