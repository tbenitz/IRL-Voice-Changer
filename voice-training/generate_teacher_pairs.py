#!/usr/bin/env python3
"""Generate genuine source/teacher-converted LLVC pairs automatically.

Requires the official Plachtaa/seed-vc repository and a CUDA GPU.
The only speaker-specific input is one authorized target voice recording.
No Pocket TTS dual-render pairs are used.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def command(*argv):
    subprocess.run([str(a) for a in argv], check=True)


def source_corpus(source_dir, limit, work):
    """Choose diverse candidates without decoding hundreds we may never use.

    A 400-pair run normally uses ~404 source clips. Previously all 800
    candidates were decoded to WAV before the first teacher conversion.
    """
    originals = sorted(Path(source_dir).rglob("*.flac"))
    originals += sorted(Path(source_dir).rglob("*.wav"))
    if not originals:
        raise RuntimeError("Missing source corpus. Download LibriSpeech dev-clean first.")
    rng = random.Random(42)
    rng.shuffle(originals)
    selected = []
    per_speaker = {}
    for path in originals:
        speaker = path.parent.parent.name if path.parent.parent != Path(source_dir) else path.parent.name
        if per_speaker.get(speaker, 0) >= 70:
            continue
        selected.append(path)
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
        if len(selected) >= limit:
            break
    if len(selected) < 25:
        raise RuntimeError(f"Only {len(selected)} eligible source clips; need at least 25")
    return selected


def decode_source(source, work, index):
    """Decode just the next source clip; remove temporary WAV after use."""
    output = Path(work) / f"{index:04d}_original.wav"
    command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", source, "-ac", "1", "-ar", "16000",
            "-t", "9", "-acodec", "pcm_s16le", output)
    audio, rate = sf.read(output, dtype="float32")
    if rate != 16000 or len(audio) < 2 * 16000:
        raise ValueError("LibriSpeech source shorter than 2 seconds")
    return output


def prepare_pair(source, converted, destination, index):
    """Reject bad timing; DO NOT time-stretch two unrelated recordings."""
    a, ar = sf.read(source, dtype="float32")
    b, br = sf.read(converted, dtype="float32")
    if ar != 16000 or br != 16000:
        raise ValueError(f"Sample rate mismatch: {ar}/{br}")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Expected mono WAVs")
    if abs(len(a) - len(b)) > int(0.06 * len(a)):
        raise ValueError(f"Teacher changed clip duration by >6%: {len(a)} / {len(b)}")
    if len(a) < 32000:
        raise ValueError("Clip too short")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Source or teacher contains non-finite samples")
    src_rms = float(np.sqrt(np.mean(a * a)))
    teacher_rms = float(np.sqrt(np.mean(b * b)))
    if src_rms < .008 or teacher_rms < .008:
        raise ValueError("Source or teacher was near silent")
    if np.mean((a-b) ** 2) < 1e-7:
        raise ValueError("Voice converter produced essentially unchanged audio")
    ratio = teacher_rms / src_rms
    if ratio < 0.12 or ratio > 8.0:
        raise ValueError(f"Teacher gain outside plausible range: {ratio:.2f}")
    # Reject severe clipping; minor peaks near full scale are permitted.
    teacher_clipped = float(np.mean(np.abs(b) > .995))
    if teacher_clipped > .025:
        raise ValueError(f"Converted audio severely clipped: {teacher_clipped:.1%}")
    # Quantify speech-envelope preservation as an auditable diagnostic.
    # This is not speaker-similarity validation.
    frame = 1600  # 100 ms at 16 kHz
    usable = (n // frame) * frame
    src_env = np.mean(np.abs(a[:usable].reshape(-1, frame)), axis=1)
    teacher_env = np.mean(np.abs(b[:usable].reshape(-1, frame)), axis=1)
    env_similarity = float(np.corrcoef(src_env, teacher_env)[0, 1]) if (
        np.std(src_env) > 1e-6 and np.std(teacher_env) > 1e-6) else 0.0
    if not np.isfinite(env_similarity):
        env_similarity = 0.0
    stats = {
        "source_rms": round(src_rms, 5),
        "teacher_rms": round(teacher_rms, 5),
        "gain_ratio": round(ratio, 3),
        "teacher_clipped_fraction": round(teacher_clipped, 5),
        "speech_envelope_correlation": round(env_similarity, 3),
        "seconds": round(n / 16000, 2),
    }
    # Resample identity is preserved; each converted file is generated from its
    # *own original*, with the same speech and approximate frame timing.
    part = "dev" if index % 50 == 0 else "val" if index % 12 == 0 else "train"
    folder = Path(destination) / part
    folder.mkdir(parents=True, exist_ok=True)
    name = f"voice_{index:04d}"
    sf.write(folder / f"{name}_original.wav", a, 16000, subtype="PCM_16")
    sf.write(folder / f"{name}_converted.wav", b, 16000, subtype="PCM_16")
    return part, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Consented speaker WAV/MP3")
    p.add_argument("--source-dir", required=True, help="LibriSpeech corpus folder")
    p.add_argument("--output", required=True, help="LLVC dataset output")
    p.add_argument("--seed-vc", required=True, help="Official Seed-VC clone path")
    p.add_argument("--pairs", type=int, default=400)
    p.add_argument("--teacher-steps", type=int, default=10)
    args = p.parse_args()
    if not 200 <= args.pairs <= 2000:
        p.error("The source count must be 200–2000 accepted clips")
    if not 4 <= args.teacher_steps <= 50:
        p.error("Teacher steps must be 4–50")
    if not Path(args.reference).is_file():
        p.error("Speaker reference audio missing")
    seed_dir = Path(args.seed_vc).resolve()
    sys.path.insert(0, str(seed_dir))
    os.chdir(seed_dir)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher synthesis requires GPU; this job will not pretend CPU training succeeded.")
    import inference as seed
    work = Path(args.output).resolve().parent / "source_wavs"
    temp = Path(args.output).resolve().parent / "teacher_tmp"
    work.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    reference = str(Path(args.reference).resolve())
    # Attempt additional candidates so the user receives the requested number
    # of VALID pairs, not just 400 attempts (which might yield 397 accepted).
    candidates = source_corpus(args.source_dir, min(2500, args.pairs * 2), work)
    args_teacher = SimpleNamespace(
        source="", target=reference, output=str(temp),
        diffusion_steps=args.teacher_steps, length_adjust=1.0,
        inference_cfg_rate=0.7, f0_condition=False,
        auto_f0_adjust=False, semi_tone_shift=0,
        checkpoint=None, config=None, fp16=True)
    # Load the 100+ MB teacher and feature extractors just once.
    preloaded = seed.load_models(args_teacher)
    seed.load_models = lambda _: preloaded
    manifest_path = Path(args.output) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "train": 0, "val": 0, "dev": 0, "rejected": 0, "reasons": [],
        "requested": args.pairs, "attempted": 0, "accepted_source_indices": [],
        "quality_metrics": [],
    }
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get("requested") == args.pairs and isinstance(old.get("accepted_source_indices"), list):
            results = old
            print(f"Resuming accepted pairs: {len(results['accepted_source_indices'])}/{args.pairs}", flush=True)
        else:
            raise RuntimeError("Existing dataset manifest belongs to another job or older format; use a new output directory.")
    already_accepted = set(results["accepted_source_indices"])
    for i, source in enumerate(candidates):
        if len(results["accepted_source_indices"]) >= args.pairs:
            break
        if i in already_accepted:
            continue
        converted = None
        source_wav = None
        try:
            source_wav = decode_source(source, work, i)
            args_teacher.source = str(source_wav)
            for old in temp.glob("*.wav"):
                old.unlink()
            seed.main(args_teacher)
            outputs = list(temp.glob("vc_*.wav"))
            if len(outputs) != 1:
                raise ValueError("Teacher did not produce exactly one recording")
            converted = temp / f"converted_{i:04d}.wav"
            command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", outputs[0], "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le", converted)
            # Use accepted index for exact train/val/dev allocations.
            accepted_index = len(results["accepted_source_indices"])
            part, metrics = prepare_pair(source_wav, converted, args.output, accepted_index)
            results[part] += 1
            results.setdefault("quality_metrics", []).append(metrics)
            results["accepted_source_indices"].append(i)
            # Persist after each successful pair to recover from session loss.
            manifest_path.write_text(json.dumps(results, indent=2))
        except Exception as e:
            results["rejected"] += 1
            if len(results["reasons"]) < 50:
                results["reasons"].append(f"{source.name}: {str(e)[:200]}")
        finally:
            if converted:
                converted.unlink(missing_ok=True)
            # Converted dataset contains its own accepted source WAV;
            # this temporary decode is no longer needed.
            (work / f"{i:04d}_original.wav").unlink(missing_ok=True)
        results["attempted"] = i + 1
        if (i + 1) % 10 == 0 or len(results["accepted_source_indices"]) == args.pairs:
            print(f"Teacher conversion: {len(results['accepted_source_indices'])}/{args.pairs} accepted; "
                  f"{i+1} attempted, {results['rejected']} rejected",
                  flush=True)
            manifest_path.write_text(json.dumps(results, indent=2))
    manifest_path.write_text(json.dumps(results, indent=2))
    if len(results["accepted_source_indices"]) < args.pairs:
        raise RuntimeError(
            f"Only {len(results['accepted_source_indices'])}/{args.pairs} usable teacher pairs; "
            f"see {manifest_path} for rejection details"
        )
    if min(results["train"], results["val"], results["dev"]) < 2:
        raise RuntimeError("Insufficient aligned training/validation clips")
    metrics = results.get("quality_metrics", [])
    quality_summary = {
        "accepted": len(results["accepted_source_indices"]),
        "requested": args.pairs,
        "rejected": results["rejected"],
        "audio_integrity_checked": True,
        "teacher_clipping_rejected_above": 0.025,
        "speaker_similarity_evaluated": False,
        "note": "Amplitude, duration and envelope diagnostics do not prove speaker similarity.",
        "median_envelope_correlation": (
            float(np.median([m["speech_envelope_correlation"] for m in metrics]))
            if metrics else None),
    }
    (Path(args.output) / "quality_report.json").write_text(
        json.dumps(quality_summary, indent=2))
    print(f"QUALITY_REPORT: accepted={quality_summary['accepted']}; "
          f"rejected={quality_summary['rejected']}; "
          f"median_envelope_correlation={quality_summary['median_envelope_correlation']}", flush=True)
    print("Teacher corpus ready. Pairs use SAME source speech; review A/B audio for voice quality.", flush=True)


if __name__ == "__main__":
    main()
