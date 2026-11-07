/* Browser RVC preview: no hosted demos, no text-to-speech, no DSP substitution.
 * Reuses the real local ONNX worker with a licensed voice checkpoint. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const runtime=()=>window.__IRL_RVC_getModels?.();
  const status=(message,bad=false)=>{
    const el=$("rvc-status");
    el.textContent=message;
    el.style.color=bad?"#ffb3bc":"#ceeaf1";
  };
  let active=null;
  function cleanup(){
    const state=active;active=null;
    window.__IRL_RVC_clipBusy=false;
    if(state){
      state.cancelled=true;
      try{state.worker?.terminate()}catch(_){}
      try{state.reject?.(new Error("Preview cancelled"))}catch(_){}
    }
    window.__IRL_RVC_cancelPreparation?.();
    const rt=runtime();
    // Update the original page's model buttons, while leaving the live mic UI untouched.
    $("rvc-run").disabled=false;
    $("get-core").disabled=false;
    $("rvc-cancel").disabled=true;
    $("rvc-progress").hidden=true;
  }
  function bufferToWav(samples,rate){
    const buf=new ArrayBuffer(44+samples.length*2),v=new DataView(buf);
    const chars=(offset,str)=>{for(let i=0;i<str.length;i++)v.setUint8(offset+i,str.charCodeAt(i));};
    chars(0,"RIFF");v.setUint32(4,36+2*samples.length,true);
    chars(8,"WAVE");chars(12,"fmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);
    v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);
    v.setUint16(32,2,true);v.setUint16(34,16,true);chars(36,"data");v.setUint32(40,2*samples.length,true);
    for(let i=0;i<samples.length;i++){
      const f=Math.max(-1,Math.min(1,Number(samples[i])||0));
      v.setInt16(44+i*2,f<0?Math.round(f*32768):Math.round(f*32767),true);
    }
    return new Blob([buf],{type:"audio/wav"});
  }
  function request(state,message,complete,timeoutMs){
    return new Promise((resolve,reject)=>{
      let chunkAudio=null;
      const timer=setTimeout(()=>{reject(new Error("RVC model "+message.type+" timed out on this device."));},timeoutMs);
      state.reject=reject;
      state.worker.onmessage=e=>{
        const d=e.data||{};
        if(d.type==="stage"){status(d.text||"Loading RVC…");return;}
        if(d.type==="error"){
          clearTimeout(timer);
          reject(new Error(d.text||"RVC model inference error"));return;
        }
        if(d.type==="audio"){chunkAudio=d;}
        if(d.type==="ready" && complete==="ready"){
          clearTimeout(timer);
          state.reject=null;
          resolve(d);
        }else if(d.type==="idle" && complete==="idle"){
          clearTimeout(timer);
          state.reject=null;
          if(!chunkAudio)reject(new Error("RVC finished without returning a model audio result"));
          else resolve(chunkAudio);
        }
      };
      state.worker.onerror=e=>{
        clearTimeout(timer);
        reject(new Error("RVC worker crashed: "+(e.message||"unknown")));
      };
      state.worker.postMessage(message);
    });
  }
  async function pcm16k(file,limitSec=3){
    if(file.size>50e6)throw new Error("For this Chromebook preview choose a source file under 50 MB.");
    const ctx=new(window.AudioContext||window.webkitAudioContext)();
    try{
      const audio=await ctx.decodeAudioData(await file.arrayBuffer());
      const duration=Math.min(limitSec,audio.duration);
      if(duration<.25)throw new Error("Source recording must contain at least 0.25 seconds");
      const inputFrames=Math.ceil(duration*audio.sampleRate);
      const temp=ctx.createBuffer(1,inputFrames,audio.sampleRate);
      const mixed=temp.getChannelData(0),n=Math.min(2,audio.numberOfChannels);
      for(let c=0;c<n;c++){
        const channel=audio.getChannelData(c);
        for(let i=0;i<inputFrames;i++)mixed[i]+=channel[i]/n;
      }
      const offline=new OfflineAudioContext(1,Math.round(inputFrames*16000/audio.sampleRate),16000);
      const src=offline.createBufferSource();src.buffer=temp;
      src.connect(offline.destination);src.start();
      const rendered=await offline.startRendering();
      return new Float32Array(rendered.getChannelData(0));
    }finally{await ctx.close().catch(()=>{})}
  }
  async function convert(){
    if(active)return;
    const file=$("rvc-audio")?.files?.[0];
    if(!file){status("Upload an MP3 or WAV in the source-audio box first. The trained voice model downloads automatically.",true);return}
    const prior=runtime();
    if(prior?.busy){status("Another model operation is already running. Finish it before starting a preview.",true);return}
    active={cancelled:false,worker:null,reject:null};
    window.__IRL_RVC_clipBusy=true;
    const state=active;
    $("rvc-run").disabled=true;$("get-core").disabled=true;$("rvc-cancel").disabled=false;
    $("rvc-output").hidden=true;$("rvc-save").hidden=true;
    $("rvc-progress").hidden=false;$("rvc-progress").value=0;
    try{
      status("Preparing real RVC model files (downloaded once, cached when possible)…");
      let models=runtime();
      if(!models?.voice||!models?.contentVec||!models?.rmvpe){
        if(!window.__IRL_RVC_prepareModels)throw new Error("Local model downloader has not loaded");
        models=await window.__IRL_RVC_prepareModels();
      }
      if(state.cancelled)return;
      $("rvc-progress").value=12;
      status("Decoding first three seconds of speech for a genuine RVC voice comparison…");
      const pcm=await pcm16k(file);
      if(state.cancelled)return;
      const worker=new Worker("./rvc-stream-worker.js?v=20261106-auto1");
      state.worker=worker;
      status("Initializing ContentVec, RMVPE and voice checkpoint in the local RVC worker…");
      const [voice,contentVec,rmvpe]=[
        await models.voice.arrayBuffer(),
        await models.contentVec.arrayBuffer(),
        await models.rmvpe.arrayBuffer()
      ];
      if(state.cancelled)return;
      const sampleRate=Number($("rvc-live-rate")?.value)||48000;
      const ready=await request(state,{
        type:"init",voice,contentVec,rmvpe,
        sampleRate,fixedVoiceFrames:models.fixedVoiceFrames||0
      },"ready",180000);
      if(state.cancelled)return;
      $("rvc-progress").value=52;
      const durationChunk=16000; // 1.0 seconds / chunk; fit fixed 512-frame model
      const chunks=[];
      let done=0;
      for(let offset=0;offset<pcm.length;offset+=durationChunk){
        if(state.cancelled)return;
        const part=pcm.slice(offset,Math.min(pcm.length,offset+durationChunk));
        status("RVC ONNX inference: "+(offset/16000).toFixed(1)+"/"+(pcm.length/16000).toFixed(1)+" seconds converted…");
        const result=await request(state,{
          type:"convert",pcm:part,sampleRate:16000
        },"idle",180000);
        if(result.pcm?.length && !result.silence){
          const resultRate=Number(result.sampleRate);
          if(resultRate!==sampleRate)throw new Error("RVC output rate differs from selected model output rate");
          chunks.push(result.pcm);
        }
        done+=part.length;
        $("rvc-progress").value=52+Math.round(44*done/pcm.length);
      }
      if(state.cancelled)return;
      if(!chunks.length)throw new Error("RVC model generated no audible converted audio.");
      const length=chunks.reduce((n,a)=>n+a.length,0),merged=new Float32Array(length);
      let offset=0;
      for(const part of chunks){merged.set(part,offset);offset+=part.length}
      const blob=bufferToWav(merged,sampleRate);
      const url=URL.createObjectURL(blob);
      const output=$("rvc-output");
      if(output.dataset.url)URL.revokeObjectURL(output.dataset.url);
      output.dataset.url=url;output.src=url;output.hidden=false;
      $("rvc-save").href=url;$("rvc-save").hidden=false;
      $("rvc-progress").value=100;
      status("Actual RVC ONNX conversion produced "+(length/sampleRate).toFixed(2)+" seconds of audio. Play it above or download the WAV.");
    }catch(err){
      if(!state.cancelled)status("RVC model could not convert the clip: "+(err?.message||err)+". No pitch-shifter or TTS substitute was used.",true);
    }finally{
      if(active===state)cleanup();
    }
  }
  $("rvc-run").addEventListener("click",()=>{void convert()});
  $("rvc-cancel").addEventListener("click",()=>{
    cleanup();
    status("Conversion stopped and model worker released. No raw audio substituted.");
  });
  window.addEventListener("pagehide",cleanup);
})();