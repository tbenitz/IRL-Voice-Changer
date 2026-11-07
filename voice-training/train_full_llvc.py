#!/usr/bin/env python3
"""Fine-tune the pretrained KoeAI LLVC HFG model and export a browser ONNX.

GPU required. No fake 70 KB audio adapter, no randomly created placeholder
weights, no independent TTS audio pairs. Requires a genuine teacher-generated
parallel dataset, which generate_teacher_pairs.py creates automatically.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def validate_data(path):
    counts = {}
    for subset in ("train", "val", "dev"):
        source = sorted((path / subset).glob("*_original.wav"))
        paired = [p.with_name(p.name.replace("_original.wav", "_converted.wav")) for p in source]
        require(source and all(p.is_file() for p in paired),
                f"Dataset {subset} is missing original or teacher-converted WAVs")
        counts[subset] = len(source)
    require(counts["train"] >= 20 and counts["val"] >= 2 and counts["dev"] >= 2,
            f"Not enough speech data for a real student voice: {counts}")
    return counts

def setup(upstream, output, dataset, steps, batch):
    config_path = upstream / "experiments" / "llvc_hfg" / "config.json"
    require(config_path.is_file(), "Missing official KoeAI LLVC source/config")
    config = json.loads(config_path.read_text())
    # Fine-tune the already trained 39.5 MB HiFi-GAN LLVC PyTorch
    # checkpoint, never train a new 13 MB network from random initialization.
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download(repo_id="KoeAI/llvc_models",
        filename="models/checkpoints/llvc_hfg/G_500000.pth")
    require(Path(checkpoint).is_file(), "Official LLVC pretrained checkpoint missing")
    config["pretrained_generator"] = str(Path(checkpoint).resolve())
    config["data"]["dir"] = str(dataset)
    config["data"]["sr"] = 16000
    config["batch_size"] = batch
    config["eval_batch_size"] = min(4, batch)
    # Use the real LLVC causal encoder/transformer architecture, not the
    # previous 4-layer 65KB Pocket waveform adapter.
    # Reuse official architecture dimensions to preserve compatibility.
    # IMPORTANT: preserve every official llvc_hfg model parameter unchanged.
    # Its pretrained generator is NOT shape-compatible with the v1 Pocket
    # adapter or with arbitrary low-lookahead decoder dimensions.
    config["aux_fairseq"]["c"] = 0
    config["discriminator"] = "hfg"
    config["fp16_run"] = True
    config["optim"]["lr"] = 2e-5  # Gentle fine-tuning, do not overwrite pretrained timbre with 5e-4.
    # Save intermediate checkpoints so GPU interruptions do not erase the run.
    config["checkpoint_interval"] = min(500, steps)
    config["log_interval"] = 100  # Needed: upstream nests checkpoint saving inside this log block.
    config["max_steps"] = steps
    config["test_dir"] = str(upstream / "test_wavs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    # The upstream loop normally runs forever; stop *after* writing a real
    # generator/discriminator checkpoint at the user-chosen update count.
    train = upstream / "train.py"
    original = train.read_text()
    if "IRL_VOICE_TRAIN_FINITE_PATCH" not in original:
        require("import fairseq\n" in original,
                "Official LLVC training source has changed; refusing unsafe patch")
        original = original.replace("import fairseq\n", "# fairseq disabled (aux_fairseq.c=0)\n")
        key = "                torch.cuda.empty_cache()"
        require(original.count(key) == 1,
                "Official LLVC training loop changed; refusing unsafe patch")
        patch = (
            "                # IRL_VOICE_TRAIN_FINITE_PATCH\n"
            "                if global_step >= config['max_steps']:\n"
            "                    if is_main_process:\n"
            "                        print('LLVC training completed '+str(global_step)+' gradient updates', flush=True)\n"
            "                    dist.destroy_process_group()\n"
            "                    return\n"
        )
        original = original.replace(key, patch + key)
        # Print useful progress even when tensorboard is not open.
        progress_marker = "            global_step += 1\n"
        require(original.count(progress_marker) == 1,
                "Official LLVC training progress loop changed; refusing unsafe patch")
        original = original.replace(
            progress_marker,
            progress_marker +
            "            if global_step % 25 == 0 and is_main_process:\n"
            "                print('LLVC training {}/{} steps'.format(global_step, config.get('max_steps')), flush=True)\n"
        )
        # On a fresh run use the published full-size LLVC generator as the
        # initial model. On resume use our saved G/D training checkpoints.
        init_marker = "        epoch = 0\n"
        require(original.count(init_marker) == 1,
                "Official generator initialization changed; refusing unsafe patch")
        original = original.replace(init_marker,
            init_marker +
            "        if config.get('pretrained_generator'):\n"
            "            print('Fine-tuning official pretrained LLVC HFG generator', flush=True)\n"
            "            initial = torch.load(config['pretrained_generator'], map_location='cpu', weights_only=False)\n"
            "            net_g.load_state_dict(initial['model'], strict=True)\n"
            "            del initial\n")
        train.write_text(original)
    return config

def train(upstream, output):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run([sys.executable, "train.py", "-d", str(output)],
                            cwd=upstream, env=env)
    require(result.returncode == 0,
            f"Full LLVC neural training failed with exit code {result.returncode}")

def export(upstream, output, steps, name):
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    sys.path.insert(0, str(upstream))
    from model import Net
    config = json.loads((output / "config.json").read_text())
    checkpoint = output / "checkpoints" / f"G_{steps}.pth"
    require(checkpoint.is_file(),
            "Training did not produce the requested real LLVC checkpoint; no fake ONNX will be exported")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Net(**config["model_params"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval().cpu()
    onnx_path = output / (name + ".onnx")
    test = torch.randn(1, 1, 8192, dtype=torch.float32) * .07
    with torch.inference_mode():
        prediction = model(test)
    require(prediction.ndim == 3 and prediction.shape[-1] == 8192,
            f"Trained LLVC generator failed output shape check: {prediction.shape}")
    try:
        torch.onnx.export(
            model, test, str(onnx_path),
            input_names=["audio_in"], output_names=["audio_out"],
            opset_version=17, export_params=True,
            do_constant_folding=True, dynamo=False,
            dynamic_axes={"audio_in": {2: "samples"},
                          "audio_out": {2: "samples"}})
    except Exception as exc:
        raise RuntimeError(
            "Full LLVC generator checkpoint was trained, but the ONNX exporter FAILED. "
            "Checkpoint is preserved. This is not a successful ONNX build.") from exc
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    require(session.get_inputs()[0].name == "audio_in"
            and session.get_outputs()[0].name == "audio_out",
            "Export is not compatible with Fast Neural microphone input/output names")
    results = []
    for n in (4096, 8192, 12288):
        source = (.09 * np.sin(2*np.pi*190*np.arange(n)/16000)).astype("float32")[None, None, :]
        converted = session.run(["audio_out"], {"audio_in": source})[0]
        require(converted.shape == source.shape and np.isfinite(converted).all(),
                f"Exported ONNX fails 16kHz streaming chunk of {n} samples")
        rms = float(np.sqrt(np.mean(converted * converted)))
        results.append({"samples": n, "output_rms": rms})
    require(onnx_path.stat().st_size > 2_000_000,
            "Full LLVC model unexpectedly tiny; refusing to label it a full trained model")
    # Held-out A/B sound files are mandatory: source vs actual teacher vs
    # trained ONNX output. Listening, not ONNX file size, judges voice quality.
    from scipy.io import wavfile
    validation_dir = Path(config["data"]["dir"]) / "val"
    original = next(iter(sorted(validation_dir.glob("*_original.wav"))), None)
    require(original is not None, "Missing held-out validation speech")
    teacher = original.with_name(original.name.replace("_original.wav", "_converted.wav"))
    require(teacher.is_file(), "Missing teacher output for held-out speech")
    source_sr, dry_pcm = wavfile.read(str(original))
    target_sr, teacher_pcm = wavfile.read(str(teacher))
    require(source_sr == target_sr == 16000, "A/B reference rate must be 16 kHz")
    n = min(len(dry_pcm), len(teacher_pcm), 4*16000)
    require(n >= 16000, "Not enough held-out material to check the trained voice")
    dry = np.asarray(dry_pcm[:n], dtype=np.float32) / 32768
    voice_teacher = np.asarray(teacher_pcm[:n], dtype=np.float32) / 32768
    history = np.zeros(4096, np.float32)
    generated = []
    for start in range(0, n, 4096):
        block = dry[start:start+4096]
        padded = np.pad(block, (0, 4096-len(block)))
        full = np.concatenate([history, padded]).astype(np.float32)[None,None,:]
        result = session.run(["audio_out"], {"audio_in": full})[0].reshape(-1)
        require(np.isfinite(result).all(), "Student ONNX generated invalid validation audio")
        generated.append(np.clip(result[-4096:][:len(block)], -1, 1))
        history = padded
    voice_student = np.concatenate(generated)[:n]
    def save_compare(label, pcm):
        wavfile.write(str(output / ("AB_" + label + ".wav")), 16000,
                      (np.clip(pcm, -1, 1) * 32767).astype(np.int16))
    save_compare("original", dry)
    save_compare("teacher", voice_teacher)
    save_compare("trained_ONNX", voice_student)
    voice_rms = float(np.sqrt(np.mean(voice_student*voice_student)))
    require(voice_rms > 0.003,
            "Trained ONNX failed A/B test: nearly silent output. Checkpoint preserved.")
    report = {
        "type": "pretrained-LLVC-HFG-finetuned-streaming-ONNX", "steps": steps,
        "file": onnx_path.name, "size_bytes": onnx_path.stat().st_size,
        "onnx_io": "audio_in -> audio_out", "sample_rate": 16000,
        "inference_checks": results,
        "held_out_ab_files": ["AB_original.wav", "AB_teacher.wav", "AB_trained_ONNX.wav"],
        "held_out_output_rms": voice_rms,
        "limitations": "ONNX execution verified. Speaker similarity, naturalness, and Chromebook RTF require listening and device testing."}
    (output / "model_report.json").write_text(json.dumps(report, indent=2))
    print("REAL trained full-size ONNX exported:", onnx_path, f"({onnx_path.stat().st_size/1e6:.2f} MB)")
    print(json.dumps(report, indent=2))
    return onnx_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llvc", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--name", default="my_custom_neural_voice")
    p.add_argument("--export-only", action="store_true",
                   help="Export saved G_<steps>.pth to ONNX without retraining")
    a = p.parse_args()
    import torch
    if not a.export_only:
        require(torch.cuda.is_available(), "Full LLVC training requires a CUDA GPU")
    require(500 <= a.steps <= 50000 and a.steps % 500 == 0,
            "Use 500–50000 real training updates in multiples of 500")
    require(1 <= a.batch <= 16, "Batch size must be 1–16")
    print("Genuine teacher corpus:", validate_data(a.dataset.resolve()), flush=True)
    upstream = a.llvc.resolve()
    output = a.output.resolve()
    expected = output / "checkpoints" / f"G_{a.steps}.pth"
    if a.export_only:
        require((output / "config.json").is_file(),
                "Cannot export: no saved LLVC training config")
        require(expected.is_file(),
                f"Cannot export: checkpoint not present: {expected}")
        print("Exporting saved checkpoint without repeating training:", expected, flush=True)
    else:
        setup(upstream, output, a.dataset.resolve(), a.steps, a.batch)
        if expected.is_file():
            print("Requested checkpoint already exists; skipping training:", expected, flush=True)
        else:
            train(upstream, output)
    export(upstream, output, a.steps, a.name)

if __name__ == "__main__":
    main()
