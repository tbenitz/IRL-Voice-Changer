# Real custom ONNX voice training

**This is the production-architecture training path.** It replaces the old custom 65–70 KB Pocket waveform adapter as the primary recommended workflow. The old experiment is retained only so existing recordings are not deleted.

[Open the GPU training notebook in Google Colab](https://colab.research.google.com/github/tbenitz/IRL-Voice-Changer/blob/main/voice-training/train_real_voice.ipynb)

The user uploads **one authorized reference recording** to their temporary Colab runtime. No custom RVC teacher, second voice recording, manual matched WAV pairs, Python installation on ChromeOS, or separate dataset preparation is required.

The pipeline:

1. Retrieves public LibriSpeech source speech and official Seed-VC weights. It runs **the actual pretrained Seed-VC voice converter**, reference-conditioned on the chosen target voice, over each original source clip. Each teacher-converted WAV has the same spoken source content, rather than a separate Pocket TTS sentence.
2. Rejects near-silent, unchanged, missing, or badly duration-mismatched recordings. Creates LLVC train/dev/val pairs of original/converted WAVs.
3. Downloads the **actual KoeAI LLVC HiFi-GAN pre-trained generator checkpoint**, `models/checkpoints/llvc_hfg/G_500000.pth`, and loads its exactly matching `experiments/llvc_hfg/config.json`. Fine-tunes full generator weights using the official multi-resolution mel + HiFi-GAN adversarial/feature-matching training code.
4. Saves checkpoints every 500 updates, rather than fabricating a tiny ONNX.
5. Exports the resulting full LLVC generator to ONNX and checks input/output names, sample rate and multiple inference lengths with ONNX Runtime before giving users a model to download.
6. Uses **Fast Neural → Import ONNX** to run the exported model locally in Chrome after downloading. The original pretrained female checkpoints remain untouched.

**Hardware:** NVIDIA GPU is required to make teacher clips and fine-tune LLVC; free Colab GPU availability and session length are not guaranteed. ONNX training/export is not realistic inside the user's 8 GB Chromebook browser. Running the *finished* ONNX is a separate inference task.

**Important verification limits:** The scripts and notebook have been committed to GitHub but have **not been trained or benchmarked end to end on a GPU in this conversation**. Static checks are not speech-quality checks. The supplied one-sample voice, the teacher model, number/diversity of pairs, LLVC fine-tuning and test-device real-time speed all determine success. A valid ONNX file is not proof of speaker resemblance; compare held-out speech against the desired voice before regarding it as finished. The exporter fails rather than supplying random or fake weights if training/model validation fails.

Files:

- `generate_teacher_pairs.py`: single-upload automatic teacher conversion and parallel-corpus creation.
- `train_full_llvc.py`: strict matching to official pre-trained HFG architecture, full gradient training/checkpoints, ONNX export and inference checks.
- `train_real_voice.ipynb`: full workflow requiring only a target recording and a Colab GPU.

This pipeline uses the upstream [Seed-VC](https://github.com/Plachtaa/seed-vc) and [KoeAI/LLVC](https://github.com/KoeAI/LLVC) software and the [LibriSpeech corpus](https://www.openslr.org/12). Check their licenses and Colab terms before redistribution or commercial use.
