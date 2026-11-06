"""Experimental real LLVC microphone adapter.

Converts live 16 kHz audio with the ORIGINAL KoeAI/LLVC model and *persistent*
model streaming buffers. This is a separate integration from the upstream
file-based infer.py. A CPU fast enough to process chunks by the audio deadline
is required; an actual low-latency guarantee is not possible across hardware.

File is copied next to the upstream model.py by the IRL Voice Changer installer.
"""
from __future__ import annotations

import argparse
import collections
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from infer import load_model


def run_live(checkpoint: str, config: str, input_device=None, output_device=None, chunk_factor=4):
    model, sample_rate = load_model(checkpoint, config)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    L = int(model.L)
    core_len = int(model.dec_chunk_size) * L * chunk_factor
    if core_len <= 0:
        raise ValueError("Invalid LLVC model streaming chunk size.")

    incoming: queue.Queue = queue.Queue(maxsize=32)
    sound_output: collections.deque = collections.deque()
    output_lock = threading.Lock()
    done = threading.Event()
    diagnostics = {"processed": 0, "input_overruns": 0, "playback_underruns": 0}
    output_tail = np.empty(0, dtype=np.float32)

    def stream_callback(indata, outdata, frames, timing, status):
        nonlocal output_tail
        if status:
            diagnostics["input_overruns"] += 1
        try:
            incoming.put_nowait(indata[:, 0].copy())
        except queue.Full:
            try:
                incoming.get_nowait()
            except queue.Empty:
                pass
            try:
                incoming.put_nowait(indata[:, 0].copy())
            except queue.Full:
                diagnostics["input_overruns"] += 1

        # No neural inference in the PortAudio callback. Silence on underrun
        # instead of blocking a real-time audio thread.
        out = np.zeros(frames, dtype=np.float32)
        copied = 0
        with output_lock:
            while copied < frames:
                if len(output_tail):
                    n = min(len(output_tail), frames - copied)
                    out[copied:copied + n] = output_tail[:n]
                    output_tail = output_tail[n:]
                    copied += n
                elif sound_output:
                    output_tail = sound_output.popleft()
                else:
                    break
        if copied < frames:
            diagnostics["playback_underruns"] += 1
        outdata[:, 0] = out

    def worker():
        pending = np.empty(0, dtype=np.float32)
        previous = np.zeros(L * 2, dtype=np.float32)
        enc_buf, dec_buf, out_buf = model.init_buffers(1, device)
        if hasattr(model, "convnet_pre"):
            convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, device)
        else:
            convnet_pre_ctx = None
        try:
            with torch.inference_mode():
                while not done.is_set():
                    try:
                        block = incoming.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    pending = np.concatenate((pending, block))
                    # Consume one core frame; keep L samples of future context
                    # (the same shift used in upstream infer_stream).
                    while len(pending) >= core_len + L and not done.is_set():
                        shifted = pending[L:L + core_len]
                        input_frame = np.concatenate((previous, shifted)).astype(np.float32)
                        previous = shifted[-2 * L:].copy()
                        pending = pending[core_len:]
                        tensor = torch.from_numpy(input_frame).to(device)[None, None, :]
                        result, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                            tensor, enc_buf, dec_buf, out_buf, convnet_pre_ctx,
                            pad=(not model.lookahead))
                        signal = result.detach().cpu().numpy().reshape(-1).astype(np.float32)
                        if not np.all(np.isfinite(signal)):
                            raise RuntimeError("LLVC produced non-finite samples")
                        # Avoid explosive feedback and normalize to PCM bounds.
                        signal = np.clip(signal, -0.98, 0.98)
                        diagnostics["processed"] += 1
                        with output_lock:
                            if len(sound_output) >= 32:
                                sound_output.popleft()
                            sound_output.append(signal)
        except BaseException as error:
            print("LLVC conversion worker failed:", error, flush=True)
            done.set()

    processing_thread = threading.Thread(target=worker, name="llvc-inference", daemon=True)
    processing_thread.start()
    print(f"LLVC live microphone mode: rate={sample_rate}, model hop={core_len/sample_rate*1000:.0f} ms", flush=True)
    print("Use headphones and choose your OS default microphone/output device, or pass device IDs.", flush=True)
    print("Real inference, not pitch shifting or prerecorded playback. Ctrl+C stops.", flush=True)

    devices = (input_device, output_device)
    try:
        with sd.Stream(samplerate=sample_rate, blocksize=core_len, channels=1,
                       device=devices, callback=stream_callback, dtype="float32",
                       latency="low"):
            while not done.is_set():
                sd.sleep(1000)
                print("LLVC chunks:", diagnostics["processed"],
                      "input statuses:", diagnostics["input_overruns"],
                      "output underruns:", diagnostics["playback_underruns"], flush=True)
    except KeyboardInterrupt:
        print("Stopping LLVC microphone conversion.", flush=True)
    finally:
        done.set()
        processing_thread.join(timeout=2)
    if diagnostics["processed"] == 0:
        raise RuntimeError("LLVC did not process any microphone frames.")


def main():
    parser = argparse.ArgumentParser(description="Experimental LLVC microphone integration")
    parser.add_argument("--checkpoint-path", default="llvc_models/models/checkpoints/llvc/G_500000.pth")
    parser.add_argument("--config-path", default="experiments/llvc/config.json")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--chunk-factor", type=int, default=4)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    run_live(args.checkpoint_path, args.config_path, args.input_device,
             args.output_device, args.chunk_factor)


if __name__ == "__main__":
    main()
