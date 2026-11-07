# IRL Voice Changer

## Custom ONNX correction — why the old 65 KB student is NOT a voice clone

**The existing 65–70 KB Pocket ONNX is only a four-layer waveform adapter, not a speaker-trained version of the working 13 MB models.** Its 31-tap convolutions provide only ~5.7 ms of raw speech context. The old dataset paired *independent text-to-speech utterances*, stretching their total lengths but not aligning phonemes or waveform phase. Worse, the old loss penalized changing the input voice. A valid ONNX graph and a changed waveform do not mean the custom target voice was learned.

The [actual LLVC approach](https://github.com/ultrapg/hevoicechanger) used **hundreds of source speech clips passed through an already-trained RVC voice-specific teacher**, producing synchronized original/converted examples, then a substantially larger causal model with thousands of training updates. No client-side toggle can turn independently synthesized Pocket audio or a 6-ms-receptive-field student into the same trained voice model. Simply padding the 65 KB ONNX to 13 MB would do nothing.

**What's corrected in this repo:**
- The previously working pretrained checkpoints and 512-ms microphone chunk / 256-ms history defaults are preserved. For manually chosen 500/1,000-ms history, the browser now tries a shorter context without clearing audio history if inference is too slow; on sustained overload it mutes and reports the error instead of silently replacing captured speech chunks.
- Old student ONNX files trained from independent Pocket TTS renders are **refused as finished custom voice models**; existing saved files and saved Pocket voices are not deleted.
- Pocket Studio now has an advanced importer for **genuine original vs. custom-trained-teacher-converted WAVs from the same original speech**, named `001_original.wav` / `001_converted.wav`. Standard 16/40/48-kHz WAV is decoded and resampled locally to 16-kHz mono without Python, and wrong durations, silence and near-identical pairs are rejected. Independent Pocket TTS pairs are **not accepted for model training**.
- The ONNX waveform adapter training remains **research-only and too small to promise a custom speaker match**, even after real teacher data is imported. A proper full-size causal model, stronger spectral/adversarial training, and an already-trained target voice-conversion teacher or an actual zero-shot VC engine are still needed to deliver the requested quality. **This task is not completed merely because the tests pass.**



## Pocket custom voice → synthetic speech dataset → local experimental student ONNX

**[Open the Custom Voice Training Lab](https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/studio/#pocket-trainer)** in the existing Pocket Studio. It uses the real on-device Pocket ONNX model, not a remote TTS API or fabricated audio.

1. **Load Pocket** (~200 MB). Choose any built-in Pocket voice (no upload needed) or one of your already-saved custom voices. For a brand-new custom voice, upload/record a consented short sample.
2. **Generate recordings silently**: choose a 1, 5, 10 or 30 minute goal. The default **Cloned voice only** mode renders the selected voice **once** for each sentence, without initializing the audio player or requiring an MP3/preview. WAVs and transcripts save after each successful recording and can be resumed. **Experimental two-voice pairs** are optional and synthesize a second speaker only if explicitly selected.
3. **Download WAV dataset ZIP**: by default, the archive has only cloned target WAVs and manifest.json; the experimental paired mode also adds source WAVs. Browser storage can be deleted by Chrome cleanup, so export a copy.
4. **Optional experimental WebGPU student**: only 4+ two-voice pairs can be used by the current TensorFlow.js waveform adapter. One-voice TTS data cannot train that adapter. The independent TTS pairs are not properly aligned voice-conversion teacher outputs and speaker cloning is not demonstrated. WebGPU is attempted first with slower CPU fallback where possible.
5. **Experimental ONNX export and live audit**: JavaScript exports trained adapter weights as ONNX (audio_in → audio_out) and verifies inference parity. The [Fast Neural voice changer](https://tbenitz.github.io/IRL-Voice-Changer/live-vc/fast-voice.html) loads the saved model automatically, but now **rejects essentially unchanged microphone audio instead of presenting it as a cloned voice**. It also reports measured input/output waveform differences and lets you record 8 seconds of the mic and actual ONNX output for manual A/B listening. For a reliable voice-change example use the pretrained 13.7 MB streaming voice.

**Critical scientific limitation:** this is an *experimental timbre adapter*, **not a demonstrated speaker-identity clone**. Pocket TTS creates independent utterances with different phoneme timing and phase; coarse duration-aligned synthetic pairs **are not proper LLVC/RVC knowledge-distillation pairs**. Merely synthesizing 30 minutes from a 3–10-second reference will not create 30 minutes of real recorded speaker diversity. The resulting model may preserve more of the original voice than the desired one, sound poor, or fail to meet real-time latency goals. A high-quality custom streaming voice still generally needs aligned source/target pairs created by a real voice-conversion teacher processing the same source audio and more extensive training. WebGPU gradient compatibility and voice sound quality have not been end-to-end verified on the user's Chromebook.

**What is verified:** automated CI runs real TensorFlow.js gradient steps (CPU), exports learned weights into ONNX, verifies it with the Python ONNX checker, and runs ONNX Runtime Web WASM inference with numerical parity against TensorFlow.js. The generated WAV/ZIP format is verified too. This does **not** prove browser WebGPU compatibility, Pocket dataset generation on a particular device, or speaker similarity.


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


This is a browser-only repository with live DSP and ONNX voice-changing experiments, a local experimental RVC converter, Pocket TTS generation and a small student-model training lab.

Website: https://tbenitz.github.io/IRL-Voice-Changer/  
**Try models online:** https://tbenitz.github.io/IRL-Voice-Changer/online-test/  
Pocket TTS experiment: https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/  
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

## Browser-only project — no Windows EXE

This repository builds the GitHub Pages site: a real streaming ONNX voice changer, a Pocket TTS browser studio, and an experimental synthetic-speech dataset / custom student trainer. **It does not build, embed, or generate Windows executables.** Your separate Windows voice-cloning projects remain independent.

In the [Fast Neural studio](https://tbenitz.github.io/IRL-Voice-Changer/live-vc/fast-voice.html), **Saved trained ONNX** automatically reads the latest experimental Pocket export from this browser’s IndexedDB. **Import ONNX file** is for manually selecting a compatible trained model from Chromebook Downloads. Select one method, not both. For slower Chromebook hardware, microphone chunks default to **512 ms**, with a corresponding latency penalty. An identity-similarity safety guard mutes experimental near-unchanged mic output rather than presenting it as a converted voice.

In the [Pocket Custom Voice Trainer](https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/studio/#pocket-trainer), choose one of your saved custom Pocket voices as the **target** (or use a newly recorded/uploaded voice), choose a different source speaker, and press **Generate / resume**. It automatically synthesizes paired WAV files: **no manual preview and no MP3 export are required.** Local speech model quality is experimental; independent TTS renditions are not correctly aligned teacher–student training data.

## Pocket TTS HTML: microphone → real synthesized voice

The separate [pocket-demo/index.html](pocket-demo/index.html) is a turn-by-turn proof of concept that uses the **real Pocket ONNX browser inference** from your existing [A-Portable-Voice-Clone-Exe](https://github.com/tbenitz/A-Portable-Voice-Clone-Exe). On Pages deployment, the workflow copies the working Pocket Studio source to the same origin underneath the new demo.

1. Open the Pocket demo URL, leave **Chromebook / 8 GB — low memory** selected, and click **Load Pocket model**. It downloads on demand and creates one session at a time instead of loading all the models concurrently.
2. In the full Pocket Voice Studio, select a built-in voice and first generate a short typed sample. Optionally upload or record a short reference from a voice you own or have permission to use.
3. Click **Speak a phrase**. The browser recognizes the spoken words, and this page passes the actual recognized text to the embedded Pocket Studio's genuine Generate button.
4. Listen to the output, replay it, and export WAV or MP3 via the real Pocket Studio.

Pocket synthesis runs in-browser, but the microphone recognition stage uses the browser's Web Speech API, which **may process speech online** depending on browser and OS. This is not a fully local live microphone converter. It must recognize an utterance before the TTS engine generates a new one. This changes the vocal identity but not the user's exact vocal timing, emotion or pitch contours.

The demonstration is a **browser page** relying on same-origin Pocket assets; opening it directly from disk without its source assets will not work.

**Chromebook limitations:** The Pocket ONNX TTS engine is genuine CPU/WebAssembly inference and can run inside ChromeOS without installing Windows or Python. Its on-device performance is not guaranteed on every 8 GB system. Beatrice, RVC, Seed-VC and MeanVC2 use third-party hosted preview pages from the [model lab](online-test/), while LLVC currently offers listening samples only; the GitHub Pages site does not run those five engines locally.

## GitHub Actions and deployment

- [Pages workflow](.github/workflows/pages.yml) publishes the site and checks browser modules.
- [Pocket student validation](.github/workflows/pocket-student-validation.yml) checks real TFJS training and ONNX Runtime Web inference, but cannot verify voice identity or Chromebook WebGPU performance.
- No Windows executable workflow or Windows launcher is included.

## Credits, licenses and voice consent

- [w-okada VCClient](https://github.com/w-okada/voice-changer)
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Seed-VC Realtime](https://github.com/jiaheguo521/seed-vc-realtime)
- [KoeAI LLVC](https://github.com/KoeAI/LLVC)
- [ASLP-lab MeanVC2](https://github.com/ASLP-lab/MeanVC2)
- [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
- [Pocket Voice Clone Studio integration](https://github.com/tbenitz/A-Portable-Voice-Clone-Exe)

Every upstream project and set of model weights has its own license. Check them before redistributing models or binaries. Only clone or convert voices you own or have express permission to use; do not pass synthetic recordings off as authentic evidence or impersonate people deceptively.
