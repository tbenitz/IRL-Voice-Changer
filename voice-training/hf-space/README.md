---
title: One-click Custom Neural Voice Trainer
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: t4-medium
---

# Custom neural voice trainer — GPU Space

## Fix: LLVC port collision after 400/400 pairs (Nov 7, 2026)

The original upstream `KoeAI/LLVC/train.py` forces distributed process group `MASTER_PORT=12355`. In the reported private T4 Space, this produced `RuntimeError: EADDRINUSE` as soon as training launched, even though **400/400 teacher pairs were successfully created**. The patched `voice-training/train_full_llvc.py` now replaces that hard-coded assignment with an environment-aware default and assigns a random available localhost port for each training run.

The Gradio UI now includes **Recover an interrupted job**. Enter an existing 12-character ID (for example `699d49e13cea`) to resume its saved dataset if the same disk remains accessible, *without rerunning teacher conversion*. The **Download saved pairs before rebuilding** button archives existing teacher WAV pairs and reference audio in a ZIP. It works only if the original job files still exist and you have deployed the updated app without losing the original disk. Uploads are not automatically restored after a rebuild.

**Critical:** the default `/home/user/voice-data` is ephemeral on Spaces. Rebuilding, stopping, or changing hardware may erase it. Hugging Face recommends an attached read/write Storage Bucket for persistence. When using a bucket mounted at e.g. `/data`, set the Space's `VOICE_WORKSPACE` environment variable to `/data` after verifying the mount is writable. This will make future resumed jobs persistent; it cannot retroactively recover files that were already deleted. Hugging Face *Dev Mode* (PRO/Team/Enterprise) can sometimes let you patch and resume **in the existing container** without a rebuild; standard code commits always rebuild the underlying image.

## Runtime errors fixed (Nov 7, 2026)

The original Dockerfile configured `HF_HOME=/data/hf-cache` and prepared `/data` during Docker build, which fails in environments without a writable `/data` mount. **The updated Dockerfile and `app.py` no longer require `/data`.** Temporary jobs, downloads and caches use `/home/user/voice-data`, owned by UID 1000. Recopy **both** files to the root of your actual Space and rebuild; updating the GitHub template does **not** update a separate Hugging Face repository automatically.

The message **"NVIDIA Driver was not detected"** means the running container cannot access an NVIDIA GPU. In the Hugging Face Space open **Settings → Space hardware**, assign a dedicated **T4 GPU** and restart. `suggested_hardware` in YAML does not provision a GPU. The interface can start on CPU, but it will refuse training rather than spending time trying to train without CUDA.


**Upload one file, get one download:** WAV, MP3, FLAC, M4A, OGG or OPUS from **2 seconds to 2 hours**. The source is decoded only for its **first 3 minutes** and the best roughly **30-second contiguous speaking window** is selected as the reference (or the full clip if shorter).

The GPU runs:
1. Official **Seed-VC** reference-conditioned voice conversion for **400 accepted** parallel original/conversion pairs (rejected conversions are replaced with new source examples).
2. Fine-tuning the **pretrained full KoeAI LLVC HiFi-GAN network**, normally **5,000 gradient updates** (editable in the Space).
3. Full ONNX export and ONNX Runtime validation with `audio_in → audio_out`.
4. **One downloadable ZIP** with the trained `G_5000.pth`, `.onnx`, config, manifest, report, and A/B WAV recordings.

## Deploy in Hugging Face Spaces

Create a **new Hugging Face Space** with **Docker** SDK and select a dedicated **NVIDIA T4 GPU** (T4 Medium suggested). Copy the **four files in this folder** — `README.md`, `Dockerfile`, `requirements.txt`, and `app.py` — to the **root** of the Space repository. The Dockerfile clones [tbenitz/IRL-Voice-Changer](https://github.com/tbenitz/IRL-Voice-Changer) and the two upstream model projects at image build time. No Colab or local Python is needed.

Space GPU usage is billed while hardware is active. Building the container requires networking and disk space. **The Space template has not been deployed or end-to-end GPU tested in this conversation.** Do not call the service finished until it successfully exports and you audition the ONNX.

## Storage

Temporary Space files are lost on restart unless a Storage Bucket or another persistent store is attached. Uploading a two-hour audio file still transfers the **whole file** before the GPU analyzes at most three minutes. Very large uploads may be constrained by client/network/Space storage. The ZIP becomes available only after BOTH PTH and ONNX validation pass.

Do not upload recordings you lack permission to process. Voice similarity is not guaranteed, especially with 2–3 second samples.
