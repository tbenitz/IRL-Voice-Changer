# IRL Voice Changer

A Windows desktop control center for **five genuine voice-conversion engines**, plus a separate **Pocket TTS microphone experiment** on GitHub Pages.

Website: https://tbenitz.github.io/IRL-Voice-Changer/  
Pocket TTS experiment: https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/  
Portable Windows EXE: https://github.com/tbenitz/IRL-Voice-Changer/actions/workflows/windows.yml  
All build runs: https://github.com/tbenitz/IRL-Voice-Changer/actions

## Download and start

1. Go to **Actions → Build Windows portable EXE → latest successful run → Artifacts** and download **IRL-Voice-Changer-Windows-x64**. Extract its EXE.
2. Run the EXE on Windows x64. No separate Python installation is necessary *for the launcher itself*.
3. Pick an engine from the left and select **Set up engine**. The launcher downloads the chosen official upstream source and uses Astral uv to provision the required Python version in its own local environment. These installations can be several gigabytes.
4. Click **Start engine**. Choose microphone, output device, and the voice/model inside the original engine's window (or through the explicit input/reference controls in this launcher).
5. For Discord, OBS or a game, route the engine's converted audio to a virtual audio device, which needs to be installed separately. For IRL speakers, route audio to a speaker in the upstream engine. Use headphones for testing to prevent feedback.

If you prefer running the control center from source, it uses the Python standard library:

    python hub.py

Settings and upstream engines are saved under the user-local IRLVoiceChanger data directory (on Windows, %LOCALAPPDATA%\IRLVoiceChanger).

## What each model actually does

| Engine | Real upstream launch | Input/reference | Live microphone conversion? |
| --- | --- | --- | --- |
| [Beatrice v2](https://github.com/w-okada/voice-changer) | Download official VCClient desktop version and select its EXE | Compatible Beatrice voice model via VCClient | **Yes** |
| [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Python 3.12 isolated environment, realtime_gui.py | RVC .pth voice model, optional .index, HuBERT and RMVPE weights | **Yes** |
| [Seed-VC Realtime](https://github.com/jiaheguo521/seed-vc-realtime) | Python 3.10 isolated environment, real-time-gui.py | Reference recording in upstream GUI; default model may download automatically | **Yes, but delayed; GPU recommended** |
| [LLVC](https://github.com/KoeAI/LLVC) | Python 3.11 isolated environment, infer.py with -s streaming-simulation option | Input audio file selected in hub; pretrained checkpoint download during setup | **Not in published CLI** (file conversion) |
| [MeanVC2](https://github.com/ASLP-lab/MeanVC2) | Python 3.11 isolated environment, runtime/run_rt.py with --mode realtime | Reference target-speaker WAV; model setup downloads checkpoints | **Yes, experimental** |

**Scope:** The application is a common **installation, launch and control center** for the real upstream engines. They do NOT share one unified low-level inference pipeline: each engine has its own model format, voice management and audio device implementation. No button fabricates converted audio. An engine must be installed and correctly configured before it can produce a transformed voice.

### Engine-specific instructions

**Beatrice v2:** Select Official download, choose a suitable edition from the developer's [VCClient downloads](https://huggingface.co/wok000/vcclient000/tree/main), unzip it and select the program's EXE through the hub. The author provides prebuilt clients; this hub intentionally does not pretend to compile or automatically package Beatrice.

**RVC:** Set up installs the upstream source and its dependencies, but the upstream README requires separate HuBERT and RMVPE model assets. You must download those assets into the layout specified in [RVC's README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), then import a licensed voice model through its real-time GUI. There are CPU and CUDA setup choices; the initial NVIDIA preset is CUDA 11.8 for pre-RTX-50 cards. RTX 50 cards require upstream CUDA 12.8 adjustments.

**Seed-VC:** Set up installs the upstream environment. Starting its GUI may fetch the official default checkpoint/config. Select a reference audio and configure microphone and speaker in its own UI. Real-time voice conversion requires a suitable GPU for practical latency.

**LLVC:** Pick an input audio file in the hub and launch. This calls the real upstream inference command with the simulated-streaming switch; the result is saved to the chosen output folder. This version deliberately does NOT claim to convert a live microphone feed using LLVC because the published infer.py is a file converter.

**MeanVC2:** Set up runs the upstream initialization script to fetch checkpoints. Supply a target-speaker WAV, select the 40ms or 120ms model, choose CPU/CUDA, and launch. MeanVC2's runtime uses the system-default input/output devices.

### Technical and hardware cautions

- The launcher uses [Astral uv](https://docs.astral.sh/uv/) to download isolated Python runtimes. Installing uv and model weights requires internet access and runs third-party code from official sources.
- The EXE contains only the launcher; it does NOT include all five PyTorch runtimes, GPU drivers, weights, voice packs or virtual audio cable drivers.
- Some upstream requirements may need manual adjustment for your GPU, Windows build or hardware. A successful Python dependency installation does not prove a model runs.
- Every engine has a different expected checkpoint. A Pocket TTS voice profile cannot simply be imported as an RVC or Beatrice model.
- The app shows upstream logs and errors instead of silently using a fallback fake voice.

## Pocket TTS HTML: microphone → real synthesized voice

The separate [pocket-demo/index.html](pocket-demo/index.html) is a turn-by-turn proof of concept that uses the **real Pocket ONNX browser inference** from your existing [A-Portable-Voice-Clone-Exe](https://github.com/tbenitz/A-Portable-Voice-Clone-Exe). On Pages deployment, the workflow copies the working Pocket Studio source to the same origin underneath the new demo.

1. Open the Pocket demo URL and wait for ONNX model initialization.
2. In the embedded Pocket Voice Studio, select a voice or upload a reference recording you own or have permission to use.
3. Click **Speak a phrase**. The browser recognizes the spoken words, and this page passes the actual recognized text to the embedded Pocket Studio's genuine Generate button.
4. Listen to the output, replay it, and export WAV or MP3 via the real Pocket Studio.

Pocket synthesis runs in-browser, but the microphone recognition stage uses the browser's Web Speech API, which **may process speech online** depending on browser and OS. This is not a fully local live microphone converter. It must recognize an utterance before the TTS engine generates a new one. This changes the vocal identity but not the user's exact vocal timing, emotion or pitch contours.

For a fully local speech-to-text frontend, the original Windows Pocket project has a Whisper option. The demonstration is a **separate HTML page** that relies on same-origin Pocket assets; opening the file directly from the computer without the source app files will not work.

## GitHub Actions and deployment

- [Windows build workflow](.github/workflows/windows.yml) validates the Python adapters and produces a portable EXE as an Actions artifact.
- [Pages workflow](.github/workflows/pages.yml) publishes this website and the genuine Pocket studio assets from a pinned commit in the existing Pocket repository.
- To enable the site, go to **Settings → Pages → Build and deployment → Source: GitHub Actions**. If the initial Pages workflow fails because the site is not enabled, set this and then re-run the deployment workflow.

Build the EXE locally (Windows Python):

    python -m pip install pyinstaller
    pyinstaller --onefile --windowed --name IRL-Voice-Changer hub.py

Run source-level tests:

    python -m unittest discover -s tests -v
    python -m compileall -q hub.py engines.py

**Testing scope:** GitHub Actions can test code and compile a Windows executable. It cannot test your actual sound devices, graphics driver, low-latency audio behavior or whether a licensed model weight file is present. A real Windows PC is required for those tests.

## Credits, licenses and voice consent

- [w-okada VCClient](https://github.com/w-okada/voice-changer)
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Seed-VC Realtime](https://github.com/jiaheguo521/seed-vc-realtime)
- [KoeAI LLVC](https://github.com/KoeAI/LLVC)
- [ASLP-lab MeanVC2](https://github.com/ASLP-lab/MeanVC2)
- [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
- [Pocket Voice Clone Studio integration](https://github.com/tbenitz/A-Portable-Voice-Clone-Exe)

Every upstream project and set of model weights has its own license. Check them before redistributing models or binaries. Only clone or convert voices you own or have express permission to use; do not pass synthetic recordings off as authentic evidence or impersonate people deceptively.
