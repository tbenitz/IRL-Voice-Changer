#!/usr/bin/env python3
"""Colab bootstrap for a Python 3.13 host: install compatible Python 3.10 venv.

Only use system Python for this bootstrap. All Seed-VC and LLVC model code runs
with /content/voice-venv/bin/python. Keep full pip failure output visible.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/content")
ENV = ROOT / "voice-venv"
PYTHON = ENV / "bin" / "python"
PACKAGES = [
    "numpy==1.26.4", "scipy==1.13.1", "librosa==0.10.2",
    "soundfile==0.12.1", "transformers==4.46.3",
    "huggingface-hub>=0.28.1,<1", "munch==4.0.0",
    "einops==0.8.0", "accelerate", "descript-audio-codec==1.0.0",
    "hydra-core", "pyyaml", "tqdm",
    "speechbrain==1.0.2", "auraloss", "torchmetrics",
    "tensorboard", "onnx", "onnxruntime", "safetensors", "sentencepiece",
]

def run(*args, cwd=None):
    command = [str(a) for a in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("NVIDIA GPU not detected: set Runtime -> Change runtime type -> T4 GPU")
    run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")

    for dest, url in (
        ("irl-voice", "https://github.com/tbenitz/IRL-Voice-Changer.git"),
        ("seed-vc", "https://github.com/Plachtaa/seed-vc.git"),
        ("llvc-upstream", "https://github.com/KoeAI/LLVC.git"),
    ):
        if not (ROOT / dest / ".git").is_dir():
            run("git", "clone", "--depth", "1", url, dest)
    if not shutil.which("ffmpeg"):
        run("apt-get", "update", "-qq")
        run("apt-get", "install", "-y", "ffmpeg")

    if not shutil.which("uv"):
        run(sys.executable, "-m", "pip", "install", "--upgrade", "uv")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv package manager was not installed")
    print("Colab host:", sys.version.split()[0], "-> isolated Python 3.10 training environment", flush=True)
    run(uv, "python", "install", "3.10")
    if not PYTHON.is_file():
        run(uv, "venv", "--python", "3.10", "--seed", str(ENV))
    run(str(PYTHON), "--version")

    # Explicit matching CUDA packages avoid picking mismatched PyTorch and
    # torchaudio wheels. PyTorch 2.4.0 has Python 3.10 + cu121 builds.
    run(uv, "pip", "install", "--python", str(PYTHON),
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0")

    # Do not use -q: show the exact dependency that fails.
    run(uv, "pip", "install", "--python", str(PYTHON), *PACKAGES)

    # Verify the exact interpreter all later cells will use.
    run(str(PYTHON), "-c",
        "import sys,torch,numpy,torchaudio,transformers,onnxruntime,librosa;"
        "print('Python',sys.version.split()[0], 'CUDA',torch.cuda.is_available(),"
        "'Torch',torch.__version__,'NumPy',numpy.__version__);"
        "assert torch.cuda.is_available(), 'CUDA unavailable in isolated Python'")
    print("\nSUCCESS: Python 3.10 GPU training environment installed.")
    print("USE THIS PYTHON FOR ALL TEACHER AND TRAINING STEPS:", PYTHON, flush=True)

if __name__ == "__main__":
    main()
