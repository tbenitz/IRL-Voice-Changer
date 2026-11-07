/**
 * Extends the pinned, already-working Pocket Voice app.
 * No mocked voice outputs; saved PCM powers replay and WAV/MP3 exports.
 */
(async () => {
  "use strict";
  const app = window.__pocketApp;
  if (!app) { console.error("Pocket TTS app not initialized"); return; }
  const PCM_RATE = 24000;
  const container = document.querySelector("#app .container");
  if (!container) throw new Error("Upstream Pocket Voice layout changed.");
  const result = await fetch("./studio-ui.html");
  if (!result.ok) throw new Error("Studio download and playback panel missing.");
  container.insertAdjacentHTML("beforeend", await result.text());
  const el = id => document.getElementById(id);
  const original = el("reference-player");
  const generated = el("clone-player");
  const btnWav = el("save-wav");
  const btnMp3 = el("save-mp3");
  const status = el("audio-status");
  let voiceSerial = 0;
  let previewSerial = -1;
  let previewVoice = null;
  let audioPCM = null;
  let wavUrl = null;
  let refUrl = null;
  let busy = false;

  function message(value, isError = false) {
    status.textContent = String(value);
    status.classList.toggle("is-error", isError);
  }
  function customSelected() {
    return app.activeVoice === "custom" && app.el.voiceSelect.value === "custom" &&
           !!app.lastClonedPCM;
  }
  function update() {
    btnWav.disabled = !audioPCM;
    btnMp3.disabled = !audioPCM || busy;
  }
  function mergeChunks(chunks) {
    const length = chunks.reduce((sum, c) => sum + c.length, 0);
    if (length <= 0 || length > PCM_RATE * 60 * 60 * 2) throw new Error("Invalid output length");
    const audio = new Float32Array(length);
    let pos = 0;
    for (const chunk of chunks) { audio.set(chunk, pos); pos += chunk.length; }
    return audio;
  }
  function audioBlob(pcm) {
    return new Blob([app.float32ToWav(pcm, PCM_RATE)], {type: "audio/wav"});
  }
  function audioLink(pcm) { return URL.createObjectURL(audioBlob(pcm)); }
  function showReference() {
    if (refUrl) URL.revokeObjectURL(refUrl);
    refUrl = null;
    original.removeAttribute("src");
    if (customSelected() && app.lastClonedPCM.length) {
      refUrl = audioLink(app.lastClonedPCM);
      original.src = refUrl;
      original.load();
      const health = app.referenceHealth;
      el("reference-note").textContent = health ?
        ("Voice health " + health.score + "/100 (signal estimate) · " + health.originalDuration.toFixed(1) +
         "s uploaded · excerpt starts at " + health.start + "s · " + health.hint) :
        ((app.lastClonedPCM.length / PCM_RATE).toFixed(1) + " seconds of the original reference. Replay any time.");
    } else {
      el("reference-note").textContent = "Choose the cloned voice to hear its original reference recording.";
    }
    update();
  }

  const worker = app.handleWorkerMessage.bind(app);
  app.handleWorkerMessage = (...args) => {
    const answer = worker(...args);
    if (args[0] && (args[0].type === "voice_encoded" || args[0].type === "voice_set")) showReference();
    return answer;
  };
  const process = app.processVoiceClip.bind(app);
  app.processVoiceClip = async (...args) => {
    voiceSerial++;
    previewSerial = -1;
    message("Encoding the new reference locally. Generate a voice example once it is ready.");
    update();
    return process(...args);
  };
  const load = app.loadSavedVoice.bind(app);
  app.loadSavedVoice = async (...args) => {
    voiceSerial++;
    previewSerial = -1;
    app.referenceHealth = null;
    update();
    return load(...args);
  };
  const buildAudio = app.buildDownloadBlob.bind(app);
  app.buildDownloadBlob = (...args) => {
    const output = buildAudio(...args);
    if (app.currentChunks.length && !app.isGenerating) {
      const merged = mergeChunks(app.currentChunks);
      if (wavUrl) URL.revokeObjectURL(wavUrl);
      wavUrl = audioLink(merged);
      generated.src = wavUrl;
      generated.load();
      audioPCM = merged;
      previewVoice = app.activeVoice;
      previewSerial = customSelected() ? voiceSerial : -1;
      el("generated-note").textContent = (merged.length / PCM_RATE).toFixed(1) +
        "s saved. Replay using the controls. Export without another voice generation.";
      message("Audio ready for optional WAV or MP3 export. The training lab generates its own recordings; no manual preview is needed.");
    }
    update();
    return output;
  };
  const generate = app.generate.bind(app);
  app.generate = async (...args) => {
    if (!app.isWorkerReady || app.isGenerating || !app.el.textInput.value.trim()) return;
    message("Generating a real Pocket TTS preview locally...");
    const output = await generate(...args);
    update();
    return output;
  };
  const stop = app.stop.bind(app);
  app.stop = (...args) => {
    const response = stop(...args);
    message("Stopped. The last complete preview remains available.");
    update();
    return response;
  };
  app.el.voiceSelect.addEventListener("change", () => requestAnimationFrame(showReference));

  function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  btnWav.addEventListener("click", () => {
    if (audioPCM) download(audioBlob(audioPCM), "Pocket-Voice-Generated.wav");
  });
  btnMp3.addEventListener("click", async () => {
    if (!audioPCM || busy) return;
    busy = true; update();
    try {
      if (!window.lamejs || !window.lamejs.Mp3Encoder)
        throw new Error("The local MP3 encoder was not loaded.");
      const mp3 = new window.lamejs.Mp3Encoder(1, PCM_RATE, 192);
      const blocks = [];
      for (let i = 0; i < audioPCM.length; i += 1152) {
        const pcm = audioPCM.subarray(i, i + 1152);
        const shorts = new Int16Array(pcm.length);
        for (let j = 0; j < pcm.length; j++) {
          const sample = Math.max(-1, Math.min(1, pcm[j]));
          shorts[j] = Math.round(sample * (sample < 0 ? 32768 : 32767));
        }
        const bytes = mp3.encodeBuffer(shorts);
        if (bytes.length) blocks.push(new Uint8Array(bytes));
        if (i % (1152 * 300) === 0) await new Promise(resolve => setTimeout(resolve, 0));
      }
      const last = mp3.flush();
      if (last.length) blocks.push(new Uint8Array(last));
      if (!blocks.length) throw new Error("Encoder returned no MP3 frames.");
      download(new Blob(blocks, {type:"audio/mpeg"}), "Pocket-Voice-Generated.mp3");
      message("MP3 saved from the existing generated audio, no regeneration needed.");
    } catch (error) {
      message("MP3 export error: " + error.message, true);
    } finally { busy = false; update(); }
  });

  window.addEventListener("beforeunload", () => {
    if (wavUrl) URL.revokeObjectURL(wavUrl);
    if (refUrl) URL.revokeObjectURL(refUrl);
  });
  update();
})();
