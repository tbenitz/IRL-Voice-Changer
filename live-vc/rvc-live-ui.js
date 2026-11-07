/* Real RVC ONNX microphone orchestration for ChromeOS.
 * No speech transcription, TTS, pitch-shift substitution or cloud inference.
 * Outputs ONLY ONNX generator audio. Unsupported/slower-than-realtime devices
 * drop/mute the output. The worker holds persistent ONNX sessions across chunks.
 */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const app={ctx:null,worker:null,stream:null,source:null,capture:null,
    out:null,active:false,loading:false,inFlight:false,pending:null,
    lastCapture:0,nextPlayTime:0,drops:0,sessionToken:0,chunkMs:1500};
  function show(message,bad=false){
    $("rvc-live-status").textContent=message;
    $("rvc-live-status").style.color=bad?"#ffb5bd":"#c4e0ed";
  }
  function badge(text,live=false,bad=false){
    const el=$("rvc-live-badge");el.textContent=text;
    el.className="status-pill"+(live?" live":bad?" bad":"");
  }
  function toggleButtons(loading=false,active=false){
    $("rvc-live-start").disabled=loading||active;
    $("rvc-live-stop").disabled=!loading&&!active;
    $("rvc-live-chunk").disabled=loading||active;
    $("rvc-live-rate").disabled=loading||active;
    const offline=$("rvc-run");
    if(offline)offline.disabled=loading||active;
    const download=$("get-core");
    if(download)download.disabled=loading||active;
  }
  async function stop(reason){
    app.sessionToken++;
    window.__IRL_RVC_cancelPreparation?.();
    app.active=false;app.loading=false;app.inFlight=false;app.pending=null;
    try{app.capture?.port.postMessage({type:"enable",enabled:false})}catch(_){}
    try{app.stream?.getTracks().forEach(t=>t.stop())}catch(_){}
    try{app.source?.disconnect()}catch(_){}
    try{app.capture?.disconnect()}catch(_){}
    try{app.out?.gain.setValueAtTime(0,app.ctx?.currentTime||0)}catch(_){}
    try{app.out?.disconnect()}catch(_){}
    try{app.worker?.terminate()}catch(_){}
    try{await app.ctx?.close()}catch(_){}
    app.worker=null;app.ctx=null;app.stream=null;app.source=null;app.capture=null;app.out=null;
    app.lastCapture=0;app.nextPlayTime=0;
    toggleButtons();
    badge("OFF · MUTED");
    $("rvc-live-state").textContent="Stopped";
    if(reason)show(reason);
  }
  function inputChunk(frames,sampleRate){
    if(!app.active||!app.worker)return;
    const now=performance.now();
    const block={frames,sampleRate,at:now};
    if(app.inFlight){
      if(app.pending)app.drops++;
      app.pending=block; // keep only the most recent microphone block
      $("rvc-live-drops").textContent=String(app.drops);
      return;
    }
    startConvert(block);
  }
  function startConvert(block){
    if(!app.worker||!app.active)return;
    app.inFlight=true;
    app.lastCapture=block.at;
    app.worker.postMessage({type:"convert",pcm:block.frames,sampleRate:block.sampleRate},[block.frames.buffer]);
    $("rvc-live-state").textContent="RVC infer";
  }
  function playSynthesized(message){
    const ctx=app.ctx;
    if(!app.active||!ctx||!app.out||message.silence)return;
    if(!(message.pcm instanceof Float32Array) || message.pcm.length===0)return;
    const inferred=performance.now()-app.lastCapture+app.chunkMs+
      ((ctx.baseLatency||0)+(ctx.outputLatency||0))*1000;
    $("rvc-live-lag").textContent=Math.round(inferred)+" ms";
    $("rvc-live-processing").textContent=Math.round(message.inferMs||0)+" ms";
    // No replay of speech too old to be usable as a live mic feed.
    if(inferred>8000){
      show("Model took "+Math.round(message.inferMs||0)+" ms. Converted voice muted because it was more than 8 seconds behind. No raw microphone was played.",true);
      app.drops++;$("rvc-live-drops").textContent=String(app.drops);return;
    }
    const rate=Number(message.sampleRate);
    if(![32000,40000,48000].includes(rate)){
      show("RVC returned an invalid sample rate: "+rate+". Audio muted.",true);
      return;
    }
    const data=message.pcm;
    const buf=ctx.createBuffer(1,data.length,rate);
    const dst=buf.getChannelData(0);
    const fade=Math.min(Math.round(rate*.024),Math.floor(data.length/4));
    for(let i=0;i<data.length;i++){
      const volume=i<fade?i/Math.max(1,fade):i>data.length-fade?(data.length-i)/Math.max(1,fade):1;
      dst[i]=Math.max(-1,Math.min(1,Number(data[i])||0))*volume;
    }
    const src=ctx.createBufferSource();
    src.buffer=buf;src.connect(app.out);
    const playAt=Math.max(ctx.currentTime+.03,app.nextPlayTime);
    // Do not build up an unbounded playback backlog.
    if(playAt>ctx.currentTime+3){
      src.disconnect();app.drops++;$("rvc-live-drops").textContent=String(app.drops);
      show("Converted audio playback fell behind. Dropping the old block; original mic remains muted.",true);return;
    }
    src.start(playAt);
    app.nextPlayTime=playAt+buf.duration;
    src.onended=()=>{try{src.disconnect()}catch(_){}};
    $("rvc-live-state").textContent="AI audio";
    show("Real RVC output is playing. Estimated mic-to-speaker delay: "+Math.round(inferred)+" ms; inference "+
      Math.round(message.inferMs||0)+" ms for "+Math.round(message.speechMs||app.chunkMs)+" ms of speech.");
  }
  function workerEvent(event,token){
    if(token!==app.sessionToken)return;
    const d=event.data||{};
    if(d.type==="stage"){
      show(d.text||"Loading ONNX models…");
      $("rvc-live-state").textContent="Loading…";
    }else if(d.type==="ready"){
      $("rvc-live-state").textContent="Models ready";
      badge("MODELS LOADED");
      show(d.text||"RVC models initialized. Starting microphone…");
      enableMic(token).catch(err=>{show("Microphone failed: "+(err?.message||err),true);void stop();});
    }else if(d.type==="audio"){
      playSynthesized(d);
    }else if(d.type==="idle"){
      app.inFlight=false;
      if(app.active && app.pending){
        const next=app.pending;app.pending=null;
        startConvert(next);
      }
    }else if(d.type==="dropped"){
      app.drops++;$("rvc-live-drops").textContent=String(app.drops);
    }else if(d.type==="error"){
      show(String(d.text||"Model inference error")+". The microphone has been muted.",true);
      badge("ERROR · MUTED",false,true);
      void stop("Neural RVC failed: "+String(d.text||"unknown error")+". No raw-audio fallback.");
    }
  }
  async function enableMic(token){
    if(token!==app.sessionToken||!app.ctx)return;
    const ctx=app.ctx;
    const stream=await navigator.mediaDevices.getUserMedia({audio:{
      channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
    if(token!==app.sessionToken){stream.getTracks().forEach(t=>t.stop());return}
    app.stream=stream;
    await ctx.audioWorklet.addModule("./rvc-mic-worklet.js?v=20261106-rvc1");
    if(token!==app.sessionToken)return;
    const capture=new AudioWorkletNode(ctx,"rvc-mic-capture",{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]});
    app.capture=capture;
    app.source=ctx.createMediaStreamSource(stream);
    app.source.connect(capture);
    capture.connect(ctx.destination); // processor guarantees silence for this connection
    app.out=ctx.createGain();app.out.gain.value=.78;app.out.connect(ctx.destination);
    capture.onprocessorerror=()=>{void stop("Microphone capture worklet crashed; all audio muted.")};
    capture.port.onmessage=event=>{
      const d=event.data||{};
      if(d.type==="audio" && d.data instanceof Float32Array)inputChunk(d.data,d.sampleRate);
    };
    capture.port.postMessage({type:"config",chunkMs:app.chunkMs});
    await ctx.resume();
    app.loading=false;app.active=true;
    capture.port.postMessage({type:"enable",enabled:true});
    stream.getTracks().forEach(track=>track.onended=()=>{void stop("Microphone disconnected; no audio is playing.")});
    toggleButtons(false,true);
    badge("AI MIC ACTIVE",true);
    $("rvc-live-state").textContent="Listening";
    show("Microphone is flowing into the real RVC ONNX engine. The raw microphone is permanently muted. Wear headphones. Converted speech will only play after successful inference.");
  }
  async function start(){
    if(app.active||app.loading)return;
    const get=window.__IRL_RVC_getModels;
    if(typeof get!=="function"){
      show("Model controls unavailable. Reload the studio.",true);return;
    }
    if(!window.isSecureContext||!window.AudioWorkletNode){
      show("Recent Chrome over HTTPS with AudioWorklet is required.",true);return;
    }
    const before=get();
    if(before.busy){show("A recording conversion is already running. Stop that before starting the live microphone.",true);return;}
    const token=++app.sessionToken;
    app.loading=true;
    app.chunkMs=Number($("rvc-live-chunk").value)||1500;
    app.drops=0;app.inFlight=false;app.pending=null;
    $("rvc-live-drops").textContent="0";
    $("rvc-live-processing").textContent="—";
    $("rvc-live-lag").textContent="—";
    toggleButtons(true);
    badge("PREPARING MODEL");
    try{
      // Unlock sound output while still in the user's activation handler.
      const ctx=new (window.AudioContext||window.webkitAudioContext)({latencyHint:"interactive"});
      app.ctx=ctx;
      await ctx.resume();
      let files=get();
      if(!files.voice||!files.contentVec||!files.rmvpe){
        if(typeof window.__IRL_RVC_prepareModels!=="function")throw Error("Built-in model downloader unavailable.");
        show("Downloading licensed RVC voice and ONNX support models to your device. No microphone audio is uploaded.");
        await window.__IRL_RVC_prepareModels();
        if(token!==app.sessionToken)return;
        files=get();
      }
      if(!files.voice||!files.contentVec||!files.rmvpe)
        throw Error("RVC voice, ContentVec or RMVPE model is missing after setup.");
      if(!files.voice.name.toLowerCase().endsWith(".onnx"))
        throw Error("The selected voice must be a trained RVC ONNX checkpoint.");
      badge("LOADING ONNX");
      show("Reading downloaded ONNX models into a persistent browser worker…");
      app.worker=new Worker("./rvc-stream-worker.js?v=20261106-auto1");
      app.worker.onmessage=event=>workerEvent(event,token);
      app.worker.onerror=event=>{
        if(token!==app.sessionToken)return;
        void stop("RVC worker failed ("+(event.message||"unknown")+"). Original microphone remains muted.");
      };
      const voice=await files.voice.arrayBuffer();
      if(token!==app.sessionToken)return;
      const contentVec=await files.contentVec.arrayBuffer();
      if(token!==app.sessionToken)return;
      const rmvpe=await files.rmvpe.arrayBuffer();
      if(token!==app.sessionToken)return;
      app.worker.postMessage({
        type:"init",voice,contentVec,rmvpe,
        sampleRate:Number($("rvc-live-rate").value)||48000,
        fixedVoiceFrames:files.fixedVoiceFrames||0
      },[voice,contentVec,rmvpe]);
    }catch(e){
      if(token!==app.sessionToken)return;
      const reason="Real RVC model setup failed: "+(e?.message||e)+". No raw microphone was played.";
      await stop(reason);
    }
  }
  $("rvc-live-start").addEventListener("click",()=>{void start()});
  $("rvc-live-stop").addEventListener("click",()=>{void stop("Stopped. Mic tracks closed, neural worker released and output muted.")});
  window.addEventListener("pagehide",()=>{void stop()});
})();
