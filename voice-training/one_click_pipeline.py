#!/usr/bin/env python3
"""One-upload GPU voice pipeline: 2s–2h audio -> 400 aligned pairs -> LLVC PTH -> ONNX -> ZIP.

Run with the Python environment used for Seed-VC/LLVC (Colab voice-venv or
a GPU Hugging Face Docker Space). New uploads get separate --run-dir paths.
The output ZIP is created only if trained PTH + validated ONNX both exist.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import zipfile

MAX_SECONDS = 7200.0
ANALYZE_SECONDS = 180
TARGET_SECONDS = 30


def say(message):
    print(message, flush=True)


def run(*argv, cwd=None):
    cmd = [str(arg) for arg in argv]
    say("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def slug(text):
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower()[:60]
    return clean or "custom_voice"


def duration_seconds(file):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file),
    ], text=True).strip()
    value = float(output)
    if not math.isfinite(value) or value < 2 or value > MAX_SECONDS + 1:
        raise ValueError("Audio must be between 2 seconds and 2 hours long.")
    return value


def select_voice_reference(audio, reference, scratch):
    import numpy as np
    import soundfile as sf

    duration = duration_seconds(audio)
    say(f"[1/5] Source audio length: {duration:.1f}s. Analyzing at most the first 180s.")
    scratch.mkdir(parents=True, exist_ok=True)
    extracted = scratch / "first_three_minutes.wav"
    run("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", audio, "-t", ANALYZE_SECONDS, "-ac", "1", "-ar", "22050",
        "-c:a", "pcm_s16le", extracted)
    samples, rate = sf.read(extracted, dtype="float32")
    if samples.ndim != 1 or rate != 22050 or len(samples) < 2 * rate:
        raise ValueError("Audio decoding failed or provided fewer than 2 seconds.")
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < .003:
        raise ValueError("Reference audio is effectively silent.")
    if len(samples) <= TARGET_SECONDS * rate:
        best = samples
        start = 0
    else:
        # Analyze short-time speech activity throughout the first 3 minutes
        # and pick a contiguous 30-second window with usable voice content.
        hop = rate // 2
        chunks = len(samples) // hop
        energies = np.array([
            np.sqrt(np.mean(samples[i * hop:(i + 1) * hop] ** 2))
            for i in range(chunks)
        ])
        gate = max(.008, float(np.percentile(energies, 75)) * .15)
        active = (energies > gate).astype(np.float32)
        nwin = int(TARGET_SECONDS * rate / hop)
        scores = np.convolve(active, np.ones(nwin), mode="valid")
        # Tiny energy tie-breaker; don't systematically select louder music.
        scores += np.convolve(np.minimum(energies, .16), np.ones(nwin), mode="valid") * .01
        start = int(np.argmax(scores)) * hop
        best = samples[start:start + TARGET_SECONDS * rate]
    reference.parent.mkdir(parents=True, exist_ok=True)
    sf.write(reference, best, 22050, subtype="PCM_16")
    say(f"[1/5] Reference selected at {start/rate:.1f}s, length {len(best)/rate:.1f}s.")
    if len(best) < 3 * rate:
        say("WARNING: 2–3 second references are allowed, but voice similarity may be poor.")
    return {"input_duration_seconds": duration,
            "analyzed_seconds": round(len(samples) / rate, 2),
            "reference_start_seconds": round(start / rate, 2),
            "reference_seconds": round(len(best) / rate, 2)}


def check_gpu_distributed():
    """Fail before 400 teacher conversions if local PyTorch rendezvous cannot bind."""
    import torch.distributed as dist
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    previous_addr = os.environ.get("MASTER_ADDR")
    previous_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group(
            backend="gloo", init_method="env://", rank=0, world_size=1
        )
        say(f"GPU distributed preflight passed on free localhost port {port}.")
    except Exception as exc:
        raise RuntimeError(
            "Distributed training cannot open a localhost port; "
            "aborting before creating expensive teacher pairs."
        ) from exc
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for name, previous in (
            ("MASTER_ADDR", previous_addr), ("MASTER_PORT", previous_port)
        ):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def prepare_corpus(cache):
    folder = cache / "LibriSpeech" / "dev-clean"
    if folder.is_dir() and any(folder.rglob("*.flac")):
        say("[2/5] Using cached LibriSpeech dev-clean.")
        return folder
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "dev-clean.tar.gz"
    say("[2/5] Downloading LibriSpeech source speech...")
    run("wget", "-c", "--show-progress", "-O", archive,
        "https://www.openslr.org/resources/12/dev-clean.tar.gz")
    run("tar", "-xzf", archive, "-C", cache)
    if not folder.is_dir() or not any(folder.rglob("*.flac")):
        raise RuntimeError("LibriSpeech extraction incomplete.")
    return folder


def package_model(output, manifest, reference_info, run_dir, model_name, steps):
    pth = output / "checkpoints" / f"G_{steps}.pth"
    onnx = output / f"{model_name}.onnx"
    report = output / "model_report.json"
    expected = (pth, onnx, report)
    for file in expected:
        if not file.is_file() or file.stat().st_size < 1024:
            raise RuntimeError(f"Missing finished trained artifact: {file}")
    if pth.stat().st_size < 2_000_000 or onnx.stat().st_size < 2_000_000:
        raise RuntimeError("Model artifacts are unexpectedly small. Refusing invalid ZIP.")
    import onnx as onnx_lib
    import onnxruntime as ort
    onnx_lib.checker.check_model(str(onnx))
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "audio_in" or session.get_outputs()[0].name != "audio_out":
        raise RuntimeError("ONNX input/output signature does not match the browser's Fast Neural interface.")
    metadata = {
        "model": model_name,
        "trained_updates": steps,
        "requested_pairs": 400,
        "reference": reference_info,
        "accepted_pairs": sum(manifest.get(k, 0) for k in ("train", "val", "dev")),
        "model_architecture": "KoeAI LLVC HiFi-GAN",
        "note": "Audio resemblance and live performance need an actual listening/device test.",
    }
    metadata_path = run_dir / "training_details.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    archive = run_dir / (model_name + "_PTH_ONNX.zip")
    members = [
        (pth, "model/" + pth.name),
        (onnx, "model/" + onnx.name),
        (report, "model/model_report.json"),
        (output / "config.json", "model/config.json"),
        (run_dir / "dataset" / "manifest.json", "training/manifest.json"),
        (metadata_path, "training/training_details.json"),
    ]
    for name in ("AB_original.wav", "AB_teacher.wav", "AB_trained_ONNX.wav"):
        f = output / name
        if f.is_file():
            members.append((f, "samples/" + f.name))
    say("[5/5] Packaging PTH + ONNX together...")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as bundle:
        for file, name in members:
            if not file.is_file():
                raise RuntimeError(f"Missing package member: {file}")
            bundle.write(file, arcname=name)
    with zipfile.ZipFile(archive) as bundle:
        damaged = bundle.testzip()
        if damaged:
            raise RuntimeError(f"ZIP integrity check failed: {damaged}")
    say(f"COMPLETE — ZIP: {archive}")
    say(f"Pairs accepted: {metadata['accepted_pairs']}/400. Steps: {steps}.")
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/content/irl-voice"))
    parser.add_argument("--seed-vc", type=Path, default=Path("/content/seed-vc"))
    parser.add_argument("--llvc", type=Path, default=Path("/content/llvc-upstream"))
    parser.add_argument("--cache", type=Path, default=Path("/content/voice-cache"))
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--teacher-steps", type=int, default=10)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    if args.pairs != 400:
        parser.error("This one-click workflow generates exactly 400 accepted pairs.")
    if args.steps < 500 or args.steps % 500:
        parser.error("--steps must be 500 or more, in multiples of 500.")
    if not 1 <= args.batch <= 16:
        parser.error("--batch must be between 1 and 16")
    if not 4 <= args.teacher_steps <= 50:
        parser.error("--teacher-steps must be between 4 and 50")
    audio = args.audio.expanduser().resolve()
    work = args.run_dir.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    seed = args.seed_vc.expanduser().resolve()
    llvc = args.llvc.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")
    for path, label in ((repo, "training repo"), (seed, "Seed-VC"), (llvc, "LLVC")):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for teacher conversion and training.")
    say("GPU: " + torch.cuda.get_device_name(0))
    check_gpu_distributed()
    work.mkdir(parents=True, exist_ok=True)
    name = slug(args.name or audio.stem)
    reference = work / "target_reference.wav"
    info_path = work / "reference_selection.json"
    if reference.is_file() and info_path.is_file():
        info = json.loads(info_path.read_text())
        say("[1/5] Reusing existing selected voice reference.")
    else:
        info = select_voice_reference(audio, reference, work / "scratch")
        info_path.write_text(json.dumps(info, indent=2))
    dataset = work / "dataset"
    manifest_file = dataset / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    accepted = sum(int(manifest.get(k, 0)) for k in ("train", "val", "dev"))
    if accepted < args.pairs:
        corpus = prepare_corpus(cache)
        say(f"[3/5] Generating exactly {args.pairs} accepted teacher pairs; currently {accepted}.")
        run(sys.executable, repo / "voice-training" / "generate_teacher_pairs.py",
            "--reference", reference, "--source-dir", corpus,
            "--output", dataset, "--seed-vc", seed,
            "--pairs", args.pairs, "--teacher-steps", args.teacher_steps)
        manifest = json.loads(manifest_file.read_text())
    accepted = sum(int(manifest.get(k, 0)) for k in ("train", "val", "dev"))
    if accepted != args.pairs:
        raise RuntimeError(f"Expected {args.pairs} accepted pairs, got {accepted}.")
    output = work / "llvc-trained"
    saved = output / "checkpoints" / f"G_{args.steps}.pth"
    say(f"[4/5] Fine-tuning on {accepted} pairs for {args.steps} weight updates.")
    if saved.is_file():
        say(f"Using completed trained checkpoint: {saved.name}; not retraining.")
    run(sys.executable, repo / "voice-training" / "train_full_llvc.py",
        "--llvc", llvc, "--dataset", dataset, "--output", output,
        "--steps", args.steps, "--batch", args.batch, "--name", name)
    package_model(output, manifest, info, work, name, args.steps)


if __name__ == "__main__":
    main()
