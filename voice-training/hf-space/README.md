---
title: LLVC Neural Voice Studio
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: t4-medium
---

# LLVC Neural Voice Studio — Hugging Face T4 Docker Space

This repo folder contains files used by the separate Hugging Face Space repository, **not** a live deployment. To update the running private Space `tstone87/LLVC-Voice-Changer`, copy **only the current `app.py` from this folder** to the **root of that Hugging Face Space repository**; the already-installed Dockerfile clones this GitHub repo. At Space startup `app.py` now performs a one-time `git fetch` to avoid an outdated Docker cached clone. If that sync fails or a required worker script is missing, the app **refuses to start paid training** instead of failing after hours.

## Architecture — avoid lost results on page refresh

1. User uploads a consented recording and chooses a profile.
2. App copies the recording into a server-side job with a permanent-in-the-running-container 12-character job ID.
3. It launches `voice-training/job_worker.py` as a **detached process**, not a long-running Gradio queue callback. A refresh, closed tab, or browser stream error does not terminate this worker while the same container keeps running.
4. Worker writes atomic `job_state.json`, a persistent `job.log`, and saves PTH training checkpoints plus ONNX exports under `voice-jobs/<job-id>/llvc-trained`.
5. Browser job ID is stored in Gradio `BrowserState` (stable storage key and secret) so reloading the page reopens the same job. The **Saved jobs** picker also locates jobs on disk if browser storage was lost.
6. A 3-second read-only timer retrieves real accepted pair counts, real training update counts, saved logs, PTH/ONNX paths, and A/B original/Seed-VC/LLVC comparison audio. No upload or retraining is triggered by reopening.
7. Gradio provides **two independent download controls: PTH checkpoints and validated ONNX**. No ZIP is generated in the Hugging Face Space flow.

## Quality profiles (GPU resource use rises with each option)

| Profile | Accepted teacher pairs | LLVC fine-tuning updates | Seed-VC diffusion steps |
|---|---:|---:|---:|
| Lower-cost trial | 400 | 1,000 | 10 |
| High fidelity (default) | 800 | 5,000 | 20 |
| Studio extended | 1,200 | 8,000 | 24 |

These are *not* promises of target-speaker resemblance. The training dataset uses genuine same-utterance source and teacher-converted pairs. Newly added quality filters check silent files, non-finite samples, excessive gain mismatch and severe clipping, and retain an audit summary in `dataset/quality_report.json`. The app provides A/B listening samples so the user can check whether it learned the intended voice.

## What happens on errors

- Pausing/closing the **browser tab**: the detached job continues and the UI can reconnect.
- Failing a late export or packaging step: any saved PTH checkpoint remains available. The Hugging Face Space does not perform ZIP packaging.
- Worker error or GPU failure: the job log and any finished artifacts remain accessible while the container/disk exists. Use **Resume saved job** only when the original dataset and recording are still present.
- **Space pause, hardware change, container restart/rebuild, or full container crash:** Hugging Face erases ephemeral disk. No software can guarantee recovery from those events without writing the files to storage outside that ephemeral disk. An optional mounted persistent bucket is supported but **not required** to run the app. Users should download available PTH/ONNX before shutting down their T4. This limitation should never be hidden.

## Before spending GPU time

1. Copy only the latest `app.py` to root of the existing Hugging Face Space and let it rebuild. The existing Dockerfile and requirements.txt remain unchanged.
2. Open the App screen on CPU if desired. Confirm the new **Saved Jobs** section, **Quality / compute profile**, direct PTH/ONNX controls and the absence of a code revision warning.
3. If the old ephemeral disk survived, use Saved Jobs to look for earlier training outputs (legacy job folders are supported). Do **not** launch a new job until checking.
4. If no saved job remains and you elect to train, do not close or pause the Space before downloading any available artifacts.

## Testing notes

GitHub Actions source-check workflow syntax-checks all training Python and notebooks, runs CPU-only regression tests (including simulated detached jobs, browser refresh scenarios, checkpoint salvage and packaging failure), and constructs the actual Gradio 6 UI on CPU. **A synthetic test cannot establish live GPU compatibility or speaker similarity.** Treat the full T4 run as unverified until an actual successful run demonstrates that PTH and ONNX are both downloadable.

