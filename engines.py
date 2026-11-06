"""Upstream engine setup and launch adapters for IRL Voice Changer.

Each voice engine runs in its own Python environment and *actually executes*
the upstream real-time app or inference script. No placeholder audio is produced.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import webbrowser
import zipfile

HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "IRLVoiceChanger"
HOME.mkdir(parents=True, exist_ok=True)
USER_AGENT = "IRL-Voice-Changer/1.0 (upstream integration)"

@dataclass(frozen=True)
class Engine:
    key: str
    title: str
    tagline: str
    upstream: str
    branch: str
    python_version: str
    mode: str
    information: str

ENGINES = {
    "beatrice": Engine("beatrice", "Beatrice v2", "Very low latency · VCClient", "w-okada/voice-changer", "master", "", "Live · official desktop client",
        "Use the author's VCClient Beatrice v2 build. Import the downloaded EXE or select it under Settings. The upstream packaged client handles audio devices and conversion."),
    "rvc": Engine("rvc", "RVC", "Trained voices · real-time GUI", "RVC-Project/Retrieval-based-Voice-Conversion-WebUI", "main", "3.12", "Live · upstream real-time GUI",
        "Import a compatible RVC .pth and optional .index in the upstream real-time GUI. The required HuBERT and RMVPE weights must be downloaded separately; see upstream README."),
    "seed": Engine("seed", "Seed-VC Realtime", "Reference voice · CUDA recommended", "jiaheguo521/seed-vc-realtime", "main", "3.10", "Live · upstream GUI",
        "Use the upstream reference-audio selector. On first launch the default model may download. An NVIDIA GPU is strongly recommended."),
    "llvc": Engine("llvc", "LLVC", "Efficient conversion · experimental microphone streaming", "KoeAI/LLVC", "main", "3.11", "Live mic adapter + file mode",
        "Run a custom low-latency mic adapter using the real pretrained LLVC model and persistent buffers, or switch to the official infer.py file conversion. CPU may underrun on slower machines."),
    "meanvc2": Engine("meanvc2", "MeanVC2", "Reference voice · 40 ms model option", "ASLP-lab/MeanVC2", "main", "3.11", "Live microphone · command-line",
        "Supply a reference WAV in the hub and start upstream run_rt.py --mode realtime. The upstream code chooses the system's default microphone and speaker."),
}

def engine_dir(key: str) -> Path:
    return HOME / "engines" / key

def source_dir(key: str) -> Path:
    return engine_dir(key) / "source"

def venv_python(key: str) -> Path:
    env = engine_dir(key) / ".venv"
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def config_path() -> Path:
    return HOME / "settings.json"

def load_settings() -> dict:
    path = config_path()
    try:
        d = json.loads(path.read_text("utf-8"))
        if isinstance(d, dict):
            return d
    except (OSError, ValueError):
        pass
    return {}

def save_settings(settings: dict) -> None:
    temp = config_path().with_suffix(".tmp")
    temp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    temp.replace(config_path())

def subprocess_flags():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

def run_logged(args: list[str], cwd: Path | None, log, *, env=None) -> None:
    """Wait for an install command and relay output, raising on failure."""
    log("$ " + " ".join(str(x) for x in args))
    with subprocess.Popen([str(x) for x in args], cwd=str(cwd) if cwd else None,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", bufsize=1,
                          creationflags=subprocess_flags(), env=env) as p:
        for line in p.stdout:
            log(line.rstrip())
        code = p.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}. Review log above.")

def download_url(url: str, log) -> bytes:
    log("Downloading " + url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()

def install_uv(log) -> Path:
    known = shutil.which("uv")
    if known:
        return Path(known)
    folder = HOME / "tools"
    exe_name = "uv.exe" if os.name == "nt" else "uv"
    path = folder / exe_name
    if path.is_file():
        return path
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Automatic runtime setup is currently for Windows x64. Install uv from https://docs.astral.sh/uv/ and retry.")
    url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    data = download_url(url, log)
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        matches = [n for n in bundle.namelist() if n.replace("\\", "/").endswith("/uv.exe") or n == "uv.exe"]
        if not matches:
            raise RuntimeError("Downloaded uv bundle contains no uv.exe.")
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bundle.read(matches[0]))
    log("Installed uv into the user-local tools directory.")
    return path

def download_source(engine: Engine, log) -> Path:
    dest = source_dir(engine.key)
    if (dest / "README.md").is_file():
        log("Using existing upstream source.")
        return dest
    url = f"https://github.com/{engine.upstream}/archive/refs/heads/{engine.branch}.zip"
    data = download_url(url, log)
    folder = engine_dir(engine.key)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(folder)) as tmp:
        target = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                parts = Path(entry.filename).parts
                if len(parts) < 2 or entry.is_dir():
                    continue
                relative = Path(*parts[1:])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Unsafe path inside upstream source archive.")
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as src, output.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(target), str(dest))
    log("Downloaded upstream source: " + engine.upstream)
    return dest

def setup_engine(key: str, log, gpu: bool = False):
    """Provision isolated interpreter and dependencies; no hidden model substitutions."""
    engine = ENGINES[key]
    if key == "beatrice":
        log("Beatrice runs from the official VCClient download. Click Download / Help, download the correct edition, then use Select VCClient EXE.")
        return
    src = download_source(engine, log)
    marker = engine_dir(key) / "setup.done"
    marker.unlink(missing_ok=True)
    uv = install_uv(log)
    run_logged([str(uv), "python", "install", engine.python_version], None, log)
    interpreter = venv_python(key)
    if not interpreter.is_file():
        run_logged([str(uv), "venv", "--python", engine.python_version, str(interpreter.parent.parent)], None, log)
    cmd = [str(uv), "pip", "install", "--python", str(interpreter)]
    if key == "rvc":
        if gpu:
            log("Installing RVC NVIDIA CUDA 11.8 dependencies. For RTX 50 series configure CUDA 12.8 manually per upstream README.")
            run_logged(cmd + ["torch==2.7.1+cu118", "torchaudio==2.7.1+cu118",
                               "--index-url", "https://download.pytorch.org/whl/cu118",
                               "--extra-index-url", "https://pypi.org/simple"], src, log)
            run_logged(cmd + ["-r", "requirments_cu118_py312.txt"], src, log)
        else:
            run_logged(cmd + ["-r", "requirments_cpu_py312.txt"], src, log)
        log("RVC code installed. Before conversion, download HuBERT/RMVPE weights from the upstream README and add your own .pth model.")
    elif key == "seed":
        run_logged(cmd + ["-r", "requirements.txt"], src, log)
        if gpu:
            run_logged(cmd + ["--upgrade", "torch", "torchvision", "torchaudio",
                               "--index-url", "https://download.pytorch.org/whl/cu128"], src, log)
        log("Seed-VC installed; first GUI launch may download checkpoint/config from Hugging Face.")
    elif key == "llvc":
        run_logged(cmd + ["-r", "requirements.txt"], src, log)
        run_logged(cmd + ["sounddevice>=0.5,<1"], src, log)
        live_script = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "llvc_live.py"
        if not live_script.is_file():
            raise RuntimeError("LLVC live adapter was not bundled with the executable.")
        shutil.copyfile(live_script, src / "irlvc_llvc_mic.py")
        log("Downloading LLVC's official pretrained model files.")
        run_logged([str(interpreter), "download_models.py"], src, log)
    elif key == "meanvc2":
        if gpu:
            run_logged(cmd + ["torch==2.5.1", "torchaudio==2.5.1",
                               "--index-url", "https://download.pytorch.org/whl/cu121"], src, log)
        else:
            run_logged(cmd + ["torch==2.5.1", "torchaudio==2.5.1",
                               "--index-url", "https://download.pytorch.org/whl/cpu"], src, log)
        run_logged(cmd + ["-r", "requirements.txt"], src, log)
        log("Running upstream initialization.py --task all (downloads trained weights).")
        run_logged([str(interpreter), "initialization.py", "--task", "all"], src, log)
    marker.write_text("setup completed", encoding="utf-8")
    log("Setup finished for " + engine.title + ". This does not guarantee your audio hardware, model weights and GPU drivers are compatible.")

def engine_is_installed(key: str, settings: dict) -> bool:
    if key == "beatrice":
        return bool(settings.get("beatrice_exe") and Path(settings["beatrice_exe"]).is_file())
    return venv_python(key).is_file() and source_dir(key).is_dir() and (engine_dir(key) / "setup.done").is_file()

def launch_args(key: str, settings: dict, *, reference: str = "", input_file: str = "",
                cpu: bool = True, mean_model: str = "40ms", output_dir: str = "",
                llvc_live: bool = False) -> tuple[list[str], Path]:
    engine = ENGINES[key]
    if key == "beatrice":
        exe = Path(settings.get("beatrice_exe", ""))
        if not exe.is_file():
            raise ValueError("Choose the official VCClient EXE in Settings first.")
        return [str(exe)], exe.parent
    py = venv_python(key)
    src = source_dir(key)
    if not py.is_file() or not src.is_dir():
        raise ValueError(f"{engine.title} is not installed. Choose Set up engine first.")
    if key == "rvc":
        return [str(py), "-u", "realtime_gui.py"], src
    if key == "seed":
        return [str(py), "-u", "real-time-gui.py"], src
    if key == "llvc":
        if llvc_live:
            adapter = src / "irlvc_llvc_mic.py"
            if not adapter.is_file():
                raise ValueError("LLVC microphone adapter missing. Run Set up engine again.")
            return [str(py), "-u", str(adapter)], src
        if not input_file or not Path(input_file).is_file():
            raise ValueError("Choose a WAV/MP3 audio file before running LLVC.")
        out = Path(output_dir or str(HOME / "outputs")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return [str(py), "-u", "infer.py", "-s", "-f", str(Path(input_file).resolve()), "-o", str(out)], src
    if key == "meanvc2":
        if not reference or not Path(reference).is_file():
            raise ValueError("Choose an existing target-speaker WAV for MeanVC2.")
        if mean_model not in ("40ms", "120ms"):
            raise ValueError("Unknown MeanVC2 model setting.")
        return [str(py), "-u", "runtime/run_rt.py", "--mode", "realtime", "--model", mean_model,
                "--target-spk", str(Path(reference).resolve()), "--device", "cpu" if cpu else "cuda"], src
    raise ValueError("Unknown engine.")

def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Kill child helper processes too, without invoking a shell.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess_flags(), check=False)
    else:
        process.terminate()

def official_url(key: str) -> str:
    if key == "beatrice":
        return "https://huggingface.co/wok000/vcclient000/tree/main"
    return "https://github.com/" + ENGINES[key].upstream

def open_upstream(key: str) -> None:
    webbrowser.open(official_url(key))
