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
