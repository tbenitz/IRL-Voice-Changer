# IRL Voice Changer

**Actual live Chromebook microphone processing:** https://tbenitz.github.io/IRL-Voice-Changer/live-vc/  
**Real ONNX RVC recording conversion (experimental):** https://tbenitz.github.io/IRL-Voice-Changer/live-vc/#rvc  
**Local testing choices (no embedded hosted demos):** https://tbenitz.github.io/IRL-Voice-Changer/online-test/

## One-click RVC voice setup (Chromebook / Chrome)

Open [Live AI RVC](https://tbenitz.github.io/IRL-Voice-Changer/live-vc/#rvc-live), select one of four built-in licensed RVC v2 target voices, and press **Download selected RVC & start mic**. The browser downloads a selected voice checkpoint plus **189 MB browser-compatible FP16 ContentVec** and **181 MB FP16 RMVPE** models. The first download is approximately **430 MB** including the selected voice; files are cached in IndexedDB when storage permits. Subsequent visits reuse downloaded files if the browser retains the cache.

Four preset checkpoints come from [Ampixa Labs Nepali RVC Voices v1](https://huggingface.co/ampixa/nepali-rvc-voices-v1) under **CC BY 4.0**. The ONNX generators have fixed 512-frame inputs, which the local streaming worker pads and trims. The FP16 shared support graphs come from [Ampixa's browser RVC runtime](https://huggingface.co/ampixa/rvc-browser-runtime), whose source models are **GPL-3.0**. The prior Q8 files contained unsupported `ConvInteger` operations and are not used. All voice and model downloads are static assets; **no user microphone audio goes to Hugging Face**.

For a pre-mic recording comparison, use **Test selected voice (first 3 sec)** in the recording section after uploading a short source MP3/WAV. It uses the same local RVC ONNX worker and checkpoints, not an external demo or text-to-speech.

**Performance note:** Browser inference on an 8 GB Chromebook has not been benchmarked end-to-end with these specific weights. The fixed-frame neural generator may run too slowly to support real-time speech. A passed automated deployment or mocked-ONNX test does not prove sound quality, compatibility or latency with the actual models. On any error, the live output stays muted rather than passing through unconverted microphone audio.

## What's implemented—and what still isn't

The live microphone page is **client-side AudioWorklet DSP**: microphone PCM is transformed by granular pitch processing, filtering, and gating and sent to the local output. It **is not an RVC/Beatrice/Seed-VC model**. The fail-closed audio chain never automatically forwards dry microphone audio.

The separate RVC recording panel calls the same persistent-session local RVC v2 ONNX worker as the live microphone for a short recorded test (ContentVec → RMVPE → RVC generator). It accepts an **actual trained ONNX voice checkpoint** plus audio and uses the same browser-compatible FP16 support models. These are heavier than the previous Q8 files, so an 8 GB Chromebook could still run out of memory. This recording preview is **file-to-file inference, not continuous live streaming**. Chromebook memory and runtime compatibility are unverified, and browser processing may fail.

A new **experimental browser neural RVC microphone pipeline** is implemented at [Live AI RVC](live-vc/#rvc-live). It holds ContentVec, RMVPE, and RVC ONNX sessions in a persistent Web Worker, processes microphone blocks, displays lag and processing time, and mutes/drops audio rather than playing the original voice on failure. **This has passed static/synthetic tests but has not passed real model inference or end-to-end latency testing on an 8 GB Chromebook**; it may be too slow or unable to load the model. A 3-second MP3 is reference audio, not a trained RVC checkpoint; zero-shot reference conversion requires a separate compatible voice-conversion engine. The other requested neural engines have not been ported into the local browser and are not represented by embedded Hugging Face iframes. Pocket TTS is a separate text-to-speech experiment, **not voice conversion**.


This repository combines a Windows desktop control center for five upstream neural VC engines, a browser live DSP voice changer, a local experimental RVC ONNX recording converter, and a separate Pocket TTS speech generator.

Website: https://tbenitz.github.io/IRL-Voice-Changer/  
**Try models online:** https://tbenitz.github.io/IRL-Voice-Changer/online-test/  
Pocket TTS experiment: https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/  
Portable Windows EXE: https://github.com/tbenitz/IRL-Voice-Changer/actions/workflows/windows.yml  
All build runs: https://github.com/tbenitz/IRL-Voice-Changer/actions

## Try before installing — online model lab

Visit the [Local Test Lab](https://tbenitz.github.io/IRL-Voice-Changer/online-test/), then open the [actual live voice changer](https://tbenitz.github.io/IRL-Voice-Changer/live-vc/) or the [real RVC model test](https://tbenitz.github.io/IRL-Voice-Changer/live-vc/#rvc). These interfaces do not embed Hugging Face inference pages.

| Model | Online preview |
| --- | --- |
| **Beatrice v2** | [RVC + Beatrice hosted converter](https://huggingface.co/spaces/Luminia/rvc-beatrice-voice-conversion). Real Beatrice conversion with a compatible voice model; not VCClient live routing. |
| **RVC** | Same converter, with an RVC checkpoint or available voice. |
| **Seed-VC** | [Plachta Seed-VC online demo](https://huggingface.co/spaces/Plachta/Seed-VC), source voice + reference voice. It is the Seed-VC family, not the real-time fork GUI. |
| **MeanVC2** | [MeanVC2 ZeroGPU demo](https://huggingface.co/spaces/hugging-apps/meanvc2-voice-conversion) using source and reference recordings. |
| **LLVC** | [Official audio comparisons](https://koeai.github.io/llvc-demo/). **Listening-only:** no verified hosted custom-recording inference endpoint currently connected. |
| **Pocket TTS** | [Real browser Pocket ONNX](https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/) with same-document microphone speech recognition and text-to-speech generation. |

The external demonstration links above are documentation for the upstream projects, not embedded conversion iframes. The local Chromebook testing page uses real browser-side code. None of these demos can create a systemwide virtual microphone in ChromeOS.

### Pocket microphone demo repair

Previously, the example loaded the Pocket Studio inside an iframe, and the new site's default Pages source did not contain the actual inference assets. This could leave it stuck loading or block audio playback.

We now publish the full Pocket ONNX studio JavaScript, tokenizer and inference worker directly in this repository under the pocket-demo/studio folder. The pocket-demo entry redirects to the full studio with direct microphone controls built into the **same document**, avoiding iframe audio permissions.

To test Pocket on a Chromebook, open the full Studio, keep the default **Chromebook / 8 GB — low memory** setting, press **Load Pocket model**, and wait for all four ONNX stages to finish. Select a built-in voice first and generate typed speech, then record or upload a permitted reference voice if desired. The English asset bundle is roughly 200 MB on disk and may take considerably more RAM during initialization; success is device-dependent. Then click **Speak a phrase** in the floating microphone panel. Your spoken words are recognized and handed to the real Pocket ONNX synth. If speech recognition is unavailable, type a phrase and click **Generate words** instead.

Pocket generation is local in the browser, but the speech recognition part may use your browser's online transcription service. It does not preserve exact vocal timing or emotion, and model weight downloads can be large on low-memory devices.

The [Pages workflow](.github/workflows/pages.yml) now validates and deploys the committed Pocket files rather than overwriting the fixed copy during deployment.

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
| [LLVC](https://github.com/KoeAI/LLVC) | Python 3.11 venv; packaged experimental microphone adapter, or official infer.py -s file mode | Pretrained checkpoint downloaded at setup; select live mic or file in hub | **Yes via new experimental adapter; file option too** |
| [MeanVC2](https://github.com/ASLP-lab/MeanVC2) | Python 3.11 isolated environment, runtime/run_rt.py with --mode realtime | Reference target-speaker WAV; model setup downloads checkpoints | **Yes, experimental** |

**Scope:** The application is a common **installation, launch and control center** for the real upstream engines. They do NOT share one unified low-level inference pipeline: each engine has its own model format, voice management and audio device implementation. No button fabricates converted audio. An engine must be installed and correctly configured before it can produce a transformed voice.

### Engine-specific instructions

**Beatrice v2:** Select Official download, choose a suitable edition from the developer's [VCClient downloads](https://huggingface.co/wok000/vcclient000/tree/main), unzip it and select the program's EXE through the hub. The author provides prebuilt clients; this hub intentionally does not pretend to compile or automatically package Beatrice.

**RVC:** Set up installs the upstream source and its dependencies, but the upstream README requires separate HuBERT and RMVPE model assets. You must download those assets into the layout specified in [RVC's README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), then import a licensed voice model through its real-time GUI. There are CPU and CUDA setup choices; the initial NVIDIA preset is CUDA 11.8 for pre-RTX-50 cards. RTX 50 cards require upstream CUDA 12.8 adjustments.

**Seed-VC:** Set up installs the upstream environment. Starting its GUI may fetch the official default checkpoint/config. Select a reference audio and configure microphone and speaker in its own UI. Real-time voice conversion requires a suitable GPU for practical latency.

**LLVC:** By default the hub runs the bundled experimental llvc_live.py microphone adapter, which calls the genuine upstream model with persistent streaming buffers in a dedicated inference thread (not a pitch-shift placebo). It uses your system's default audio devices at 16 kHz. Results depend on CPU speed and hardware; underruns are logged. For the official upstream file converter, uncheck LLVC live microphone mode, browse an audio file and output folder, and start. The upstream infer.py itself remains file-only.

**MeanVC2:** Set up runs the upstream initialization script to fetch checkpoints. Supply a target-speaker WAV, select the 40ms or 120ms model, choose CPU/CUDA, and launch. MeanVC2's runtime uses the system-default input/output devices.

### Technical and hardware cautions

- The launcher uses [Astral uv](https://docs.astral.sh/uv/) to download isolated Python runtimes. Installing uv and model weights requires internet access and runs third-party code from official sources.
- The EXE contains only the launcher; it does NOT include all five PyTorch runtimes, GPU drivers, weights, voice packs or virtual audio cable drivers.
- Some upstream requirements may need manual adjustment for your GPU, Windows build or hardware. A successful Python dependency installation does not prove a model runs.
- Every engine has a different expected checkpoint. A Pocket TTS voice profile cannot simply be imported as an RVC or Beatrice model.
- The app shows upstream logs and errors instead of silently using a fallback fake voice.

## Pocket TTS HTML: microphone → real synthesized voice

The separate [pocket-demo/index.html](pocket-demo/index.html) is a turn-by-turn proof of concept that uses the **real Pocket ONNX browser inference** from your existing [A-Portable-Voice-Clone-Exe](https://github.com/tbenitz/A-Portable-Voice-Clone-Exe). On Pages deployment, the workflow copies the working Pocket Studio source to the same origin underneath the new demo.

1. Open the Pocket demo URL, leave **Chromebook / 8 GB — low memory** selected, and click **Load Pocket model**. It downloads on demand and creates one session at a time instead of loading all the models concurrently.
2. In the full Pocket Voice Studio, select a built-in voice and first generate a short typed sample. Optionally upload or record a short reference from a voice you own or have permission to use.
3. Click **Speak a phrase**. The browser recognizes the spoken words, and this page passes the actual recognized text to the embedded Pocket Studio's genuine Generate button.
4. Listen to the output, replay it, and export WAV or MP3 via the real Pocket Studio.

Pocket synthesis runs in-browser, but the microphone recognition stage uses the browser's Web Speech API, which **may process speech online** depending on browser and OS. This is not a fully local live microphone converter. It must recognize an utterance before the TTS engine generates a new one. This changes the vocal identity but not the user's exact vocal timing, emotion or pitch contours.

For a fully local speech-to-text frontend, the original Windows Pocket project has a Whisper option. The demonstration is a **separate HTML page** that relies on same-origin Pocket assets; opening the file directly from the computer without the source app files will not work.

**Chromebook limitations:** The Pocket ONNX TTS engine is genuine CPU/WebAssembly inference and can run inside ChromeOS without installing Windows or Python. Its on-device performance is not guaranteed on every 8 GB system. Beatrice, RVC, Seed-VC and MeanVC2 use third-party hosted preview pages from the [model lab](online-test/), while LLVC currently offers listening samples only; the GitHub Pages site does not run those five engines locally.

## GitHub Actions and deployment

- [Windows build workflow](.github/workflows/windows.yml) validates the Python adapters and produces a portable EXE as an Actions artifact.
- [Pages workflow](.github/workflows/pages.yml) publishes this website and the genuine Pocket studio assets from a pinned commit in the existing Pocket repository.
- To enable the site, go to **Settings → Pages → Build and deployment → Source: GitHub Actions**. If the initial Pages workflow fails because the site is not enabled, set this and then re-run the deployment workflow.

Build the EXE locally (Windows Python):

    python -m pip install pyinstaller
    pyinstaller --onefile --windowed --add-data "llvc_live.py;." --name IRL-Voice-Changer hub.py

Run source-level tests:

    python -m unittest discover -s tests -v
    python -m compileall -q hub.py engines.py llvc_live.py

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
