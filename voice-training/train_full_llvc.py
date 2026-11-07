#!/usr/bin/env python3
"""Train the real KoeAI LLVC causal model and export a browser-compatible ONNX.

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
    config_path = upstream / "experiments" / "llvc" / "config.json"
    require(config_path.is_file(), "Missing official KoeAI LLVC source/config")
    config = json.loads(config_path.read_text())
    config["data"]["dir"] = str(dataset)
    config["data"]["sr"] = 16000
    config["batch_size"] = batch
    config["eval_batch_size"] = min(4, batch)
    # Use the real LLVC causal encoder/transformer architecture, not the
    # previous 4-layer 65KB Pocket waveform adapter.
    # Reuse official architecture dimensions to preserve compatibility.
    config["model_params"]["lookahead"] = False
    config["model_params"]["dec_buf_len"] = 100
    config["model_params"]["dec_chunk_size"] = 72
    config["model_params"]["decoder_dropout"] = 0.0
    config["model_params"]["out_buf_len"] = 2
    config["aux_fairseq"]["c"] = 0
    config["discriminator"] = "hfg"
    config["fp16_run"] = True
    # Save intermediate checkpoints so GPU interruptions do not erase the run.
    config["checkpoint_interval"] = min(500, steps)
    config["log_interval"] = max(100, steps)
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
    report = {
        "type": "trained-full-LLVC-streaming-ONNX", "steps": steps,
        "file": onnx_path.name, "size_bytes": onnx_path.stat().st_size,
        "onnx_io": "audio_in -> audio_out", "sample_rate": 16000,
        "inference_checks": results,
        "limitations": "ONNX execution verified. Speaker similarity, naturalness, and Chromebook RTF require listening and device testing."}
    (output / "model_report.json").write_text(json.dumps(report, indent=2))
    print("REAL trained full-size ONNX exported:", onnx_path, f"({onnx_path.stat().st_size()/1e6:.2f} MB)")
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
    a = p.parse_args()
    import torch
    require(torch.cuda.is_available(), "Full LLVC distillation requires a CUDA GPU; no CPU-only pretend run")
    require(500 <= a.steps <= 50000, "Use 500–50000 real training updates")
    require(1 <= a.batch <= 16, "Batch size must be 1–16")
    print("Genuine teacher corpus:", validate_data(a.dataset.resolve()), flush=True)
    upstream = a.llvc.resolve()
    output = a.output.resolve()
    setup(upstream, output, a.dataset.resolve(), a.steps, a.batch)
    train(upstream, output)
    export(upstream, output, a.steps, a.name)

if __name__ == "__main__":
    main()
