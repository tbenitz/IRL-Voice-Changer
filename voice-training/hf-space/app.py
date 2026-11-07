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


def select_workspace():
    # Honor explicitly mounted read/write persistent storage when configured.
    chosen = os.environ.get("VOICE_WORKSPACE", "").strip()
    if chosen and chosen.lower() != "auto":
        target = Path(chosen).resolve()
        if not target.is_dir() or not os.access(target, os.W_OK):
            raise RuntimeError(f"VOICE_WORKSPACE is not a writable directory: {target}")
        return target
    bucket = Path("/data")
    if bucket.is_dir() and os.access(bucket, os.W_OK):
        return bucket
    return Path("/home/user/voice-data")


WORKSPACE = select_workspace()
for folder_name in ("voice-jobs", "voice-cache", "hf-cache", "torch-cache"):
    (WORKSPACE / folder_name).mkdir(parents=True, exist_ok=True)
# Let all child training processes reuse caches across runs/restarts when a
# persistent HF Storage Bucket is mounted at /data.
os.environ["HF_HOME"] = str(WORKSPACE / "hf-cache")
os.environ["TORCH_HOME"] = str(WORKSPACE / "torch-cache")
JOBS = WORKSPACE / "voice-jobs"
CACHE = WORKSPACE / "voice-cache"


def progress_html(pairs=0, updates=0, total_steps=500, stage="Ready",
                  exported=False, packaged=False):
    """Two real counters, not a fabricated percentage of total wall time."""
    pairs = max(0, min(400, int(pairs)))
    total_steps = max(1, int(total_steps))
    updates = max(0, min(total_steps, int(updates)))
    state = ("Verified ONNX downloadable" if exported else "Waiting for verified ONNX")
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
        f'<span>Export: {state}</span>'
        '<small>Bars track completed work, not estimated remaining minutes.</small>'
        '</div>'
    )


def available_model_files(job_dir):
    """Return existing trained checkpoint and verified ONNX independently.

    PTH may be an intermediate saved checkpoint. ONNX appears only when
    model_report.json has been written after the real inference checks.
    No ZIP and no GPU are needed to access either artifact.
    """
    output = job_dir / "llvc-trained"
    checkpoints = []
    for file in (output / "checkpoints").glob("G_*.pth"):
        match = re.fullmatch(r"G_(\d+)\.pth", file.name)
        if match and file.is_file() and file.stat().st_size >= 2_000_000:
            checkpoints.append((int(match.group(1)), file))
    pth_file = max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None
    report_file = output / "model_report.json"
    onnx_file = None
    if report_file.is_file():
        try:
            import json
            report = json.loads(report_file.read_text())
            name = report.get("file", "")
            if (isinstance(name, str) and
                    re.fullmatch(r"[A-Za-z0-9_-]+\.onnx", name) and
                    isinstance(report.get("steps"), int) and
                    report.get("onnx_io") == "audio_in -> audio_out"):
                candidate = output / name
                if (candidate.is_file() and
                        candidate.stat().st_size >= 2_000_000 and
                        candidate.stat().st_size == report.get("size_bytes") and
                        (output / "checkpoints" / f"G_{report['steps']}.pth").is_file()):
                    onnx_file = candidate
        except (OSError, ValueError, TypeError):
            pass
    return (str(pth_file) if pth_file else None,
            str(onnx_file) if onnx_file else None)


def create_voice(uploaded_audio, steps, resume_id):
    steps = int(steps)
    progress = progress_html(total_steps=steps)
    # Resume only an existing job in this exact running container/volume.
    resume_id = (resume_id or "").strip()
    if resume_id:
        if not re.fullmatch(r"[a-f0-9]{12}", resume_id):
            yield "Invalid recovery job ID (expected 12 hexadecimal characters).", None, None, progress
            return
        job_dir = JOBS / resume_id
        if not (job_dir / "dataset" / "manifest.json").is_file():
            yield ("No saved pairs found for " + resume_id +
                   ". Restart/rebuild may have erased temporary storage."), None, None, progress
            return
        # A completed dataset contains the processed reference itself, so
        # the user doesn't have to upload another two-hour original.
        saved_reference = job_dir / "target_reference.wav"
        if not saved_reference.is_file():
            yield "Recovered job is missing its reference audio; cannot resume safely.", None, None, progress
            return
        uploaded_audio = str(saved_reference)
    else:
        if not uploaded_audio:
            yield "Upload an audio recording first, or enter a saved job ID.", None, None, progress
            return
        if not os.path.isfile(uploaded_audio):
            yield "Uploaded file no longer exists. Try uploading again.", None, None, progress
            return
        job_dir = JOBS / uuid.uuid4().hex[:12]
    # Test CUDA before starting a long job, never launch GPU training on CPU.
    try:
        import torch
        if not torch.cuda.is_available():
            yield "NO NVIDIA GPU DETECTED. Assign a T4 GPU in Space Settings → Hardware, then restart.", None, None, progress
            return
    except Exception as exc:
        yield f"GPU preflight failed: {exc}", None, None, progress
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
        "--skip-zip",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    logs = deque(maxlen=90)
    logs.append(f"Job: {job_dir.name}" + (" (RESUMING saved 400 pairs)" if resume_id else ""))
    logs.append("Training PTH checkpoint and verified ONNX will appear in separate download boxes. NO ZIP required.")
    completed_pairs = 400 if resume_id else 0
    completed_steps = 0
    exported = False
    existing_pth, existing_onnx = available_model_files(job_dir)
    last_pth, last_onnx = existing_pth, existing_onnx
    yield "\n".join(logs), existing_pth, existing_onnx, progress_html(
        completed_pairs, 0, steps, "Starting GPU job"
    )
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
            stage = "ONNX verified; individual file ready for download"
            counter_changed = True
        if "DONE: PTH and ONNX saved separately." in line:
            exported = True
            completed_steps = steps
            completed_pairs = 400
            stage = "PTH and ONNX ready — download each file"
            counter_changed = True
        important = (
            line.startswith("[") or
            line.startswith("Teacher conversion") or
            line.startswith("LLVC training ") or
            line.startswith("COMPLETE") or
            "Error" in line or "Traceback" in line
        )
        now = time.monotonic()
        # The latest checkpoint is shown when its save has finished;
        # the ONNX appears only after its post-export validation report.
        pth_now, onnx_now = available_model_files(job_dir)
        checkpoint_safe = ("Saved checkpoints to " in line or
                           "Requested checkpoint already exists" in line or
                           "Using completed trained checkpoint" in line or
                           "REAL trained full-size ONNX exported" in line or
                           "MODEL_FILES_READY" in line)
        changed_pth = bool(pth_now and pth_now != last_pth and checkpoint_safe)
        changed_onnx = bool(onnx_now and onnx_now != last_onnx)
        if ((counter_changed or important or changed_pth or changed_onnx)
                and (now - last_update > 0.8 or changed_pth or changed_onnx)):
            if changed_pth:
                last_pth = pth_now
            if changed_onnx:
                last_onnx = onnx_now
                exported = True
            stage = locals().get("stage", "Working")
            if changed_pth and not changed_onnx:
                logs.append("DOWNLOAD AVAILABLE: latest saved PTH checkpoint.")
            if changed_onnx:
                logs.append("DOWNLOAD AVAILABLE: verified ONNX (no ZIP).")
            yield ("\n".join(logs),
                   last_pth if changed_pth else gr.update(),
                   last_onnx if changed_onnx else gr.update(),
                   progress_html(completed_pairs, completed_steps, steps, stage, exported))
            last_update = now
    return_code = process.wait()
    pth_file, onnx_file = available_model_files(job_dir)
    if return_code == 0 and pth_file and onnx_file:
        logs.append("SUCCESS: PTH and ONNX available in their own download boxes. NO ZIP.")
        yield ("\n".join(logs), pth_file, onnx_file,
               progress_html(400, steps, steps, "Completed — download both files", True))
    else:
        logs.append(f"JOB STOPPED (exit {return_code}). Any finished PTH/ONNX remains downloadable above.")
        yield ("\n".join(logs), pth_file, onnx_file,
               progress_html(completed_pairs, completed_steps, steps,
                             "Stopped — available model files preserved", bool(onnx_file)))

def recover_saved_files(job_id):
    """Expose finished PTH and ONNX without starting a GPU or ZIP build."""
    job_id = (job_id or "").strip()
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        return "Enter the 12-character job ID.", None, None
    directory = JOBS / job_id
    pth, onnx = available_model_files(directory)
    if not pth and not onnx:
        return ("No saved model files found. Space storage may have been "
                "erased when paused or rebuilt."), None, None
    message = "Saved downloads found: "
    message += ("PTH " if pth else "")
    message += ("ONNX" if onnx else "")
    return message, pth, onnx


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
        "pretrained LLVC model, and offers **separate PTH and ONNX downloads**.\n\n"
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
            start = gr.Button("Train voice and export PTH + ONNX", variant="primary")
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
        salvage = gr.Button("Find saved PTH and ONNX downloads (no GPU)")
        salvage_message = gr.Textbox(label="Recovery status", interactive=False)
        salvage_pth = gr.File(label="Saved PTH checkpoint", interactive=False)
        salvage_onnx = gr.File(label="Saved verified ONNX", interactive=False)
        salvage.click(fn=recover_saved_files, inputs=resume_id,
                      outputs=[salvage_message, salvage_pth, salvage_onnx])
    progress_panel = gr.HTML(value=progress_html(), label="Live progress")
    status = gr.Textbox(label="Detailed training log", lines=16, max_lines=28, interactive=False)
    with gr.Row():
        pth_download = gr.File(
            label="Download PTH checkpoint (available as soon as saved)",
            interactive=False,
        )
        onnx_download = gr.File(
            label="Download verified ONNX (available after export)",
            interactive=False,
        )
    start.click(
        fn=create_voice, inputs=[audio, updates, resume_id],
        outputs=[status, pth_download, onnx_download, progress_panel],
        concurrency_limit=1,
    )
    gr.Markdown(
        f"**Storage location:** `{WORKSPACE}`. "
        + ("A writable /data volume is being used for jobs and model caches. "
           if WORKSPACE == Path("/data") else
           "**Temporary storage** — pausing, stopping or rebuilding the Space "
           "can erase saved pairs and models. Attach a read-write Storage Bucket "
           "at /data before starting a paid training run. ")
        + "PTH is available when a training checkpoint finishes saving; "
          "ONNX becomes available after export validation. "
          "No ZIP is created or required. "
          "ONNX execution checks do not prove the voice sounds like the reference."
    )

demo.queue(default_concurrency_limit=1, max_size=4)
demo.launch(server_name="0.0.0.0", server_port=7860, max_file_size="4gb")
