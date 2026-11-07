/* Fast AI voice changer controller: real causal neural ONNX, local WASM.
   No input->speaker path. Only model-generated Float32 PCM reaches the gain.
   One worker/model, a latest-only pending chunk, and a bounded playback queue.
 */
(() => {
"use strict";
const $=id=>document.getElementById(id);
const MODEL_ROOT="https://raw.githubusercontent.com/ultrapg/hevoicechanger/main/models/";
const FAST_BUILD="20261107-context-preflight-v12";
const voices={
 hfg:{name:"female_voice_hfg_new.onnx",bytes:13756183},
 old_hq:{name:"female_voice_old_hq.onnx",bytes:13718163},
 old:{name:"female_voice_old.onnx",bytes:13756183}
};
const app={token:0,loading:false,active:false,worker:null,stream:null,ctx:null,
 source:null,capture:null,gain:null,sources:new Set(),abort:null,inFlight:false,pending:null,
 nextPlayTime:0,drops:0,chunkFrames:2048,historySamples:4096,lastCapture:0,
 queue:[],slowCount:0,requestedHistory:0,adaptations:0,
 workerBlobUrl:null,verifiedWorkerBuild:false,
 audit:null,inFlightSource:null,auditUrls:[]};
const CACHE="irl-fast-neural-voice-models", STORE="weights";
async function loadVerifiedWorker(){
 // Fetch tiny worker (not voice weights) using no browser/CDN cache.
 // A mismatched old worker will never run model inference or access the mic.
 const workerPath=new URL("./fast-voice-worker.js",window.location.href);
 workerPath.searchParams.set("v",FAST_BUILD);
 workerPath.searchParams.set("fresh",String(Date.now()));
 const response=await fetch(workerPath.toString(),{cache:"no-store"});
 if(!response.ok)throw Error("Voice worker download HTTP "+response.status+
   ". Audio remains muted.");
 const source=await response.text();
 const exactBuild='const WORKER_BUILD="'+FAST_BUILD+'"';
 if(!source.includes(exactBuild))
  throw Error("An old worker came from browser/CDN cache. Its 100 KB model restriction was NOT loaded. "+
    "Use Reload latest engine to refresh this page. Your saved voice is safe.");
 // One-off Blob URL also prevents Worker() reusing a stale disk cache entry.
 const blobUrl=URL.createObjectURL(new Blob(
   [source+"\n//# sourceURL=fast-voice-worker-"+FAST_BUILD+".js"],
   {type:"application/javascript"}));
 try{return {worker:new Worker(blobUrl),blobUrl}}
 catch(error){URL.revokeObjectURL(blobUrl);throw error}
}
function info(text,bad=false){
 const el=$("status-text");
 el.textContent=String(text);
 el.classList.toggle("bad",bad);
}
function badge(text,kind=""){
 const el=$("badge");el.textContent=text;
 el.className="status"+(kind?" "+kind:"");
}
function toggle(){
 $("start").disabled=app.loading||app.active;
 $("stop").disabled=!(app.loading||app.active);
 $("audit-start").disabled=!app.active||!!app.audit;
 for(const id of ["voice","model-file","chunk","history"])$(id).disabled=app.loading||app.active;
}
function meters(id,pcm){
 if(!(pcm instanceof Float32Array)||!pcm.length)return;
 let sum=0;for(let i=0;i<pcm.length;i+=8)sum+=pcm[i]**2;
 const rms=Math.sqrt(sum/Math.ceil(pcm.length/8));
 $(id).style.width=Math.max(0,Math.min(100,100*(rms*4)))+"%";
}
function openDB(){
 return new Promise((resolve,reject)=>{
  if(!("indexedDB" in window))return reject(Error("IndexedDB disabled"));
  const request=indexedDB.open(CACHE,1);
  request.onupgradeneeded=()=>{
   if(!request.result.objectStoreNames.contains(STORE))
    request.result.createObjectStore(STORE,{keyPath:"name"});
  };
  request.onerror=()=>reject(request.error);
  request.onsuccess=()=>resolve(request.result);
 });
}
async function cached(name){
 let db;
 try{
  db=await openDB();
  const result=await new Promise((resolve,reject)=>{
   const r=db.transaction(STORE,"readonly").objectStore(STORE).get(name);
   r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  });
  return result?.blob||null;
 }catch(e){console.debug("Voice cache unavailable",e.message);return null}
 finally{db?.close()}
}
async function cacheModel(name,blob){
 let db;
 try{
  db=await openDB();
  await new Promise((resolve,reject)=>{
   const tx=db.transaction(STORE,"readwrite");
   tx.objectStore(STORE).put({name,blob});
   tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);
  });
 }catch(e){console.warn("Voice cache unavailable; this session still works",e.message)}
 finally{db?.close()}
}
async function savedStudentModel(){
 return await new Promise((resolve,reject)=>{
  const open=indexedDB.open("pocket-streaming-student-v1",2);
  open.onerror=()=>reject(open.error||Error("Cannot open saved trainer models"));
  open.onupgradeneeded=()=>{
   const db=open.result;
   if(!db.objectStoreNames.contains("pairs"))db.createObjectStore("pairs",{keyPath:"id"});
   if(!db.objectStoreNames.contains("meta"))db.createObjectStore("meta",{keyPath:"key"});
  }; // Safe even when Fast Neural is opened before Pocket Trainer.
  open.onsuccess=()=>{
   const db=open.result;
   if(!db.objectStoreNames.contains("meta")){
    db.close();reject(Error("No Pocket-trained model found. Generate pairs and train in Pocket Studio first."));return;
   }
   const req=db.transaction("meta","readonly").objectStore("meta").get("student-onnx");
   req.onsuccess=()=>{db.close();resolve(req.result?.value||null)};
   req.onerror=()=>{db.close();reject(req.error||Error("Could not read saved trained ONNX"))};
  };
 });
}
async function getModel(){
 const choice=$("voice").value;
 if(choice==="saved"){
  const model=await savedStudentModel();
  if(!model?.blob||!model.validation)
   throw Error("No validated custom ONNX saved. Open Pocket Studio, generate training speech and complete local training first.");
  if(model.trainingData!=="teacher-parallel")
   throw Error("Your saved "+(model.blob.size/1024).toFixed(0)+
    " KB model was generated from INDEPENDENT Pocket TTS audio. That training data is not aligned and cannot teach a working custom neural voice. The old file is still saved for research, but will NOT be presented as a voice clone. Import original/converted recordings from a real custom-trained voice-conversion teacher to start a valid training experiment.");
  if(model.validation.nearIdentity)
   throw Error("Your saved experimental adapter failed the voice-change test: nearly unchanged microphone audio. The file was preserved, but it is not a working custom speaker clone.");
  if(model.blob.size>40e6)throw Error("Saved model is unexpectedly large.");
  $("download-status").textContent="Loaded the "+(model.blob.size/1024).toFixed(0)+
    " KB locally trained ONNX model from Pocket Studio (voice quality unverified).";
  return await model.blob.arrayBuffer();
 }
 if(choice==="custom"){
  const selected=$("model-file").files?.[0];
  if(!selected)throw Error("Select a compatible causal 16kHz ONNX voice checkpoint before starting.");
  if(!selected.name.toLowerCase().endsWith(".onnx"))
   throw Error("Custom model must have .onnx extension");
  if(selected.size>200e6)throw Error("Select a smaller model (under 200 MB) for the fast mode");
  $("download-status").textContent="Using your local ONNX file; not downloading any voice weights.";
  return selected.arrayBuffer();
 }
 const model=voices[choice];
 if(!model)throw Error("Unknown voice preset");
 let blob=await cached(model.name);
 if(blob?.size===model.bytes){
  $("download-status").textContent="Loaded "+model.name+" from your browser model cache ("+
   (blob.size/1e6).toFixed(1)+" MB).";
  return blob.arrayBuffer();
 }
 $("download").hidden=false;$("download").value=0;
 $("download-status").textContent="Downloading "+model.name+" (13.7 MB) once…";
 app.abort=new AbortController();
 const resp=await fetch(MODEL_ROOT+model.name,{signal:app.abort.signal});
 if(!resp.ok)throw Error("Model download HTTP "+resp.status);
 if(!resp.body)throw Error("No streaming model download supported");
 const reader=resp.body.getReader();
 const parts=[];
 let total=0;
 try{
  for(;;){
   const {done,value}=await reader.read();if(done)break;
   total+=value.byteLength;
   if(total>model.bytes)throw Error("Model download exceeded published size");
   parts.push(value);
   $("download").value=100*total/model.bytes;
   $("download-status").textContent="Downloading real neural voice model: "+
    (total/1e6).toFixed(1)+" / "+(model.bytes/1e6).toFixed(1)+" MB";
  }
 }finally{reader.releaseLock()}
 if(total!==model.bytes)throw Error("Incomplete model download ("+total+" bytes)");
 blob=new Blob(parts,{type:"application/octet-stream"});
 $("download").value=100;$("download-status").textContent="Model ready. Caching for later visits when storage allows…";
 await cacheModel(model.name,blob);
 $("download-status").textContent="Model ready ("+(total/1e6).toFixed(1)+" MB). No microphone audio was uploaded.";
 return blob.arrayBuffer();
}
async function stop(message){
 app.token++;
 if(app.audit)completeAudit(true);
 app.loading=false;app.active=false;
 try{app.abort?.abort()}catch(_){}
 try{app.capture?.port.postMessage({type:"enable",enabled:false})}catch(_){}
 try{app.gain?.gain.setValueAtTime(0,app.ctx.currentTime)}catch(_){}
 for(const s of app.sources){try{s.stop()}catch(_){}try{s.disconnect()}catch(_){}}
 app.sources.clear();
 try{app.stream?.getTracks().forEach(t=>t.stop())}catch(_){}
 try{app.source?.disconnect()}catch(_){}
 try{app.capture?.disconnect()}catch(_){}
 try{app.gain?.disconnect()}catch(_){}
 try{app.worker?.terminate()}catch(_){}
 if(app.workerBlobUrl){URL.revokeObjectURL(app.workerBlobUrl);app.workerBlobUrl=null;}
 try{await app.ctx?.close()}catch(_){}
 Object.assign(app,{abort:null,worker:null,ctx:null,stream:null,source:null,capture:null,
  gain:null,inFlight:false,pending:null,nextPlayTime:0,lastCapture:0,
  queue:[],slowCount:0,requestedHistory:0,adaptations:0,inFlightSource:null,
  workerBlobUrl:null,verifiedWorkerBuild:false});
 $("download").hidden=true;
 toggle();badge("OFF · MUTED");
 if(message)info(message);
}
const AUDIT_MAX_FRAMES=8*16000;
function wav16k(pcm){
 const buffer=new ArrayBuffer(44+pcm.length*2),v=new DataView(buffer);
 const ascii=(pos,value)=>{for(let i=0;i<value.length;i++)v.setUint8(pos+i,value.charCodeAt(i))};
 ascii(0,"RIFF");v.setUint32(4,36+pcm.length*2,true);ascii(8,"WAVE");ascii(12,"fmt ");
 v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
 v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);
 ascii(36,"data");v.setUint32(40,pcm.length*2,true);
 for(let i=0;i<pcm.length;i++){
  const x=Math.max(-1,Math.min(1,pcm[i]));
  v.setInt16(44+i*2,Math.round((x<0?32768:32767)*x),true);
 }
 return new Blob([buffer],{type:"audio/wav"});
}
function joinAudio(chunks,count){
 const pcm=new Float32Array(count);let at=0;
 for(const chunk of chunks){
  if(at>=count)break;
  const n=Math.min(count-at,chunk.length);
  pcm.set(chunk.subarray(0,n),at);at+=n;
 }
 return pcm;
}
function completeAudit(stopped=false){
 const audit=app.audit;
 if(!audit)return;
 app.audit=null;app.inFlightSource=null;
 if(audit.frames<16000){
  $("audit-status").textContent="Not enough paired converted audio captured. Speak for at least one second and try again.";
  toggle();return;
 }
 for(const url of app.auditUrls)URL.revokeObjectURL(url);
 app.auditUrls=[];
 const frameCount=Math.min(audit.frames,AUDIT_MAX_FRAMES);
 const mic=joinAudio(audit.mic,frameCount);
 const ai=joinAudio(audit.converted,frameCount);
 const micUrl=URL.createObjectURL(wav16k(mic)),aiUrl=URL.createObjectURL(wav16k(ai));
 app.auditUrls=[micUrl,aiUrl];
 $("audit-source").src=micUrl;$("audit-output").src=aiUrl;
 $("audit-download-source").href=micUrl;$("audit-download-output").href=aiUrl;
 $("audit-results").hidden=false;
 $("audit-status").textContent="Captured "+(frameCount/16000).toFixed(1)+
  " sec of paired original microphone and actual ONNX output"+
  (stopped?" before Stop":"")+". Both local WAV clips are ready. The original microphone was NOT played automatically.";
 toggle();
}
function collectAudit(original,converted){
 const audit=app.audit;
 if(!audit||!(original instanceof Float32Array)||!(converted instanceof Float32Array))return;
 const n=Math.min(original.length,converted.length,AUDIT_MAX_FRAMES-audit.frames);
 if(n<=0)return;
 audit.mic.push(original.slice(0,n));
 audit.converted.push(converted.slice(0,n));
 audit.frames+=n;
 $("audit-status").textContent="Capturing actual neural output "+
  (audit.frames/16000).toFixed(1)+" / 8.0 seconds…";
 if(audit.frames>=AUDIT_MAX_FRAMES)completeAudit();
}
function inputChunk(pcm){
 if(!app.active||!app.worker)return;
 meters("input-meter",pcm);
 const block={pcm,at:performance.now()};
 if(app.inFlight){
  // Never replace speech segments with newer ones: that is what caused
  // entire words to disappear at the 500/1000-ms history settings.
  if(app.queue.length>=2){
   void stop("NEURAL OVERLOAD: the model cannot process speech in real time with "+
    Math.round(app.historySamples/16)+" ms previous context. Output muted instead of "+
    "chopping up your words. Choose 256 or 128 ms context, or a faster model.");
   return;
  }
  app.queue.push(block);
  $("performance-note").textContent="Model behind microphone; buffering "+
    app.queue.length+" whole speech chunk(s) without discarding any words.";
  return;
 }
 convert(block);
}
function convert(block){
 if(!app.active||!app.worker)return;
 app.inFlight=true;app.lastCapture=block.at;
 app.inFlightSource=app.audit?block.pcm.slice():null;
 app.worker.postMessage({type:"convert",pcm:block.pcm},[block.pcm.buffer]);
}
function play(d){
 if(!app.active||!app.ctx||!app.gain)return;
 const pcm=d.pcm;
 if(!(pcm instanceof Float32Array)||!pcm.length)return;
 const ctx=app.ctx,rate=16000;
 const rtf=Number(d.rtf)||0,elapsed=Number(d.inferenceMs)||0;
 $("rtf").textContent=rtf.toFixed(2)+"×";
 $("inference").textContent=Math.round(elapsed);
 const voiced=Number(d.inputRms)>.012,correlation=Number(d.correlation);
 const delta=Number(d.changeRatio);
 $("wave-change").textContent=voiced&&Number.isFinite(delta)?(delta*100).toFixed(1)+"%":"—";
 $("wave-corr").textContent=voiced&&Number.isFinite(correlation)?correlation.toFixed(4):"—";
 $("voice-change-note").textContent=!voiced
  ?"Speak naturally to measure how much the ONNX output differs from your microphone."
  :correlation>.97
   ?"Warning: ONNX audio still tracks the original waveform closely. This may NOT sound like another speaker."
   :"ONNX output differs from the input waveform. Compare the A/B recordings to judge audible timbre; this is NOT proof of speaker identity.";
 // RTF is real inference time / audio duration. Context of 500-1000 ms
 // can overload a Chromebook, even though 256 ms runs fine.
 if(rtf>.86)app.slowCount++;else app.slowCount=0;
 if(app.slowCount>=2&&app.historySamples>2048){
  const next=app.historySamples>8000?8000:app.historySamples>4096?4096:2048;
  app.requestedHistory=next;
  app.slowCount=0;
 }
 if(rtf>1.08&&app.historySamples<=2048&&app.slowCount>=3){
  void stop("NEURAL OVERLOAD: the model needs "+Math.round(elapsed)+
    " ms for "+Math.round(app.chunkFrames/16)+" ms of microphone audio, "+
    "even at the minimum 128-ms context. It cannot run continuously on this CPU. "+
    "Output muted instead of playing incomplete speech.");
  return;
 }
 $("performance-note").textContent=app.requestedHistory
  ?"AUTO-RECOVERY: neural inference is too slow with "+Math.round(app.historySamples/16)+
    " ms history. Preserving speech and reducing previous context to "+
    Math.round(app.requestedHistory/16)+" ms."
  :rtf<=.7?"Model processing ahead of live speech."
  :rtf<=1?"Limited processing headroom. Voice history will reduce if sustained."
  :"Too slow for continuous live audio. Automatically shortening history; will mute rather than skip words if recovery fails.";
 const actualDelay=performance.now()-app.lastCapture+app.chunkFrames/16;
 if(actualDelay>1900){
  void stop("NEURAL OVERLOAD: inference arrived over 1.9 seconds late. "+
   "Audio muted rather than skipping the words you spoke. Reduce previous context.");
  return;
 }
 const scheduled=Math.max(ctx.currentTime+.025,app.nextPlayTime);
 if(scheduled>ctx.currentTime+Math.max(1.3,app.chunkFrames/16000*2.5)){
  void stop("NEURAL OVERLOAD: audio playback fell too far behind real-time speech. "+
   "Stopped instead of cutting out words. Choose a shorter previous-audio context.");
  return;
 }
 const buf=ctx.createBuffer(1,pcm.length,rate),channel=buf.getChannelData(0);
 // Short fades avoid harsh clicks if a model chunk boundary is discontinuous.
 const ramp=Math.min(80,Math.floor(pcm.length/8));
 for(let i=0;i<pcm.length;i++){
  const envelope=i<ramp?i/Math.max(1,ramp):
    i>=pcm.length-ramp?(pcm.length-1-i)/Math.max(1,ramp):1;
  channel[i]=Math.max(-1,Math.min(1,pcm[i]))*envelope;
 }
 const src=ctx.createBufferSource();
 src.buffer=buf;src.connect(app.gain);app.sources.add(src);
 src.onended=()=>{app.sources.delete(src);try{src.disconnect()}catch(_){}};
 src.start(scheduled);
 collectAudit(app.inFlightSource,pcm);
 app.inFlightSource=null;
 app.nextPlayTime=scheduled+buf.duration;
 meters("output-meter",pcm);
 $("lag").textContent=Math.round(actualDelay+(scheduled-ctx.currentTime)*1000);
 info("ONNX-generated PCM scheduled: "+Math.round(elapsed)+" ms inference for "+
  Math.round(app.chunkFrames/16)+" ms captured speech. Waveform correlation "+
  (Number.isFinite(correlation)?correlation.toFixed(3):"unknown")+
  "; RTF "+rtf.toFixed(2)+". Listen to the separate A/B recording to verify timbre.");
}
function eventFromWorker(e,token){
 if(token!==app.token)return;
 const d=e.data||{};
 if(d.type==="build"){
   if(d.build!==FAST_BUILD){
    void stop("Wrong neural worker loaded ("+String(d.build||"unknown")+
      "); audio remained muted. Click Reload latest engine.");
    return;
   }
   app.verifiedWorkerBuild=true;
   info("Verified neural worker "+d.build+". Initializing ONNX model…");
 }else if(d.type==="stage")info(d.text||"Loading neural ONNX…");
 else if(d.type==="ready") {
  if(!app.verifiedWorkerBuild){
   void stop("Unverified or stale neural worker refused. Audio stayed muted.");
   return;
  }
  badge("MODEL READY");
  // Do not capture the first words until high history settings have passed
  // a local synthetic real-time throughput check.
  if(app.historySamples>4096){
   info("Checking "+Math.round(app.historySamples/16)+
     " ms context speed before opening the microphone…");
   app.worker.postMessage({type:"preflight",chunkFrames:app.chunkFrames});
  }else{
   info("Model initialized. Starting microphone…");
   void activateMic(token).catch(err=>{void stop("Microphone failure: "+(err?.message||err)+". Input muted.")});
  }
 }else if(d.type==="preflight"){
  if(!app.loading||!app.worker)return;
  const old=app.historySamples;
  app.historySamples=Number(d.historySamples)||old;
  if(app.historySamples!==old){
   app.adaptations++;
   $("history").value=String(app.historySamples);
  }
  const tested=Array.isArray(d.measurements)
    ?d.measurements.map(x=>Math.round(x.historySamples/16)+" ms: "+x.rtf+"×").join(" → "):"";
  $("performance-note").textContent="Pre-mic speed check "+tested+
    (app.historySamples!==old?" | auto-reduced context to "+Math.round(app.historySamples/16)+" ms":"");
  if(!d.sustainable){
   void stop("This neural model cannot sustain real-time speech even after reducing previous audio history. "+
     "Measured "+d.rtf+"× real-time at "+Math.round(app.historySamples/16)+
     " ms context. The microphone was never opened. Select a longer mic chunk or a faster model.");
   return;
  }
  info((app.historySamples!==old
    ?"The selected "+Math.round(old/16)+" ms history was too slow. Reduced to "+
      Math.round(app.historySamples/16)+" ms before any speech was recorded."
    :"The selected context passed the "+d.rtf+"× real-time speed check.")+
    " Starting microphone…");
  void activateMic(token).catch(err=>{void stop("Microphone failure: "+(err?.message||err)+". Input muted.")});
 }else if(d.type==="audio")play(d);
 else if(d.type==="idle"){
  app.inFlight=false;
  if(!app.active||!app.worker)return;
  if(app.requestedHistory){
   // The worker replies "idle" after clearing its busy state; configuration
   // is therefore applied BEFORE the next queued conversion, not silently lost.
   const next=app.requestedHistory;
   app.requestedHistory=0;app.historySamples=next;app.adaptations++;
   $("history").value=String(next);
   app.worker.postMessage({type:"config",historySamples:next});
   $("performance-note").textContent="Long previous-audio context overloaded this device. "+
    "Reduced to "+Math.round(next/16)+" ms without clearing microphone history.";
  }
  if(app.queue.length)convert(app.queue.shift());
 }else if(d.type==="config-applied"){
  if(app.active)$("performance-note").textContent=
   "Neural history reconfigured to "+Math.round(d.historySamples/16)+" ms with speech context preserved.";
 }else if(d.type==="dropped"){
  app.drops++;$("drops").textContent=String(app.drops);
  if(d.reason)info("ONNX output block muted/dropped: "+d.reason,true);
 }else if(d.type==="error"){
  void stop("Neural VC error: "+(d.text||"unknown")+". The raw mic was not played.");
 }
}
async function activateMic(token){
 if(token!==app.token)return;
 const ctx=app.ctx;
 const stream=await navigator.mediaDevices.getUserMedia({audio:{
  channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
 if(token!==app.token){stream.getTracks().forEach(t=>t.stop());return}
 app.stream=stream;
 await ctx.audioWorklet.addModule("./fast-mic-worklet.js?v=20261106-f1");
 if(token!==app.token)return;
 const capture=new AudioWorkletNode(ctx,"fast-neural-mic",{
  numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]
 });
 app.capture=capture;
 app.source=ctx.createMediaStreamSource(stream);
 app.source.connect(capture);
 capture.connect(ctx.destination); // always SILENT worklet output
 const gain=ctx.createGain();
 gain.gain.value=Number($("volume").value)/100;gain.connect(ctx.destination);
 app.gain=gain;
 capture.onprocessorerror=()=>{void stop("AudioWorklet crashed. Everything muted.")};
 capture.port.onmessage=e=>{
  const d=e.data||{};
  if(d.type==="chunk"&&d.pcm instanceof Float32Array)inputChunk(d.pcm);
 };
 capture.port.postMessage({type:"config",chunkFrames:app.chunkFrames});
 await ctx.resume();
 app.loading=false;app.active=true;
 capture.port.postMessage({type:"enable",enabled:true});
 for(const t of stream.getTracks())t.onended=()=>{void stop("Microphone disconnected.")};
 toggle();badge("LIVE NEURAL", "live");
 info(["saved","custom"].includes($("voice").value)
  ? "Experimental ONNX loaded with voice-change protection: nearly unchanged microphone audio will be muted, not mistaken for a converted voice."
  : "Running pre-trained 13.7 MB voice conversion locally at 16 kHz. Headphones recommended.");
}
async function start(){
 if(app.loading||app.active)return;
 if(!window.isSecureContext||!window.AudioWorkletNode||!navigator.mediaDevices?.getUserMedia){
  info("Requires Chrome over HTTPS with microphone and AudioWorklet permission.",true);return;
 }
 app.loading=true;app.drops=0;app.inFlight=false;app.pending=null;
 app.queue=[];app.slowCount=0;app.requestedHistory=0;app.adaptations=0;
 app.chunkFrames=Number($("chunk").value)||8192;
 app.historySamples=Number($("history").value)||4096;
 $("drops").textContent="0";$("lag").textContent="—";$("inference").textContent="—";$("rtf").textContent="—";
 $("wave-change").textContent="—";$("wave-corr").textContent="—";
 $("voice-change-note").textContent="Speak after model initialization to measure the true ONNX output versus input. This is not a speaker identity score.";
 toggle();badge("LOADING AI");
 const token=++app.token;
 try{
  // Output is unlocked by the user's click, before any async download.
  const Ctx=window.AudioContext||window.webkitAudioContext;
  app.ctx=new Ctx({latencyHint:"interactive"});
  await app.ctx.resume();
  const weights=await getModel();
  if(token!==app.token)return;
  info("Fetching and verifying the newest neural worker directly from the server…");
  const {worker,blobUrl}=await loadVerifiedWorker();
  if(token!==app.token){worker.terminate();URL.revokeObjectURL(blobUrl);return}
  app.worker=worker;app.workerBlobUrl=blobUrl;app.verifiedWorkerBuild=false;
  worker.onmessage=e=>eventFromWorker(e,token);
  worker.onerror=e=>{if(token===app.token)void stop("Neural worker crashed: "+(e.message||"unknown"))};
  worker.postMessage({type:"init",model:weights,historySamples:app.historySamples,
    requireVoiceChange:true},[weights]);
 }catch(e){
  if(token===app.token)await stop("Could not initialize the real neural model: "+(e?.message||e));
 }
}
function describeVoiceSelection(){
 const choice=$("voice").value;
 $("custom-field").hidden=choice!=="custom";
 const details=choice==="saved"
  ? "SAVED model: loads your latest Pocket Studio ONNX automatically from this same browser. No file upload. The voice-change guard will mute an almost unchanged mic signal."
  : choice==="custom"
    ? "MANUAL model: choose a compatible audio_in → audio_out .onnx file from Chromebook Downloads. This is separate from the Saved model."
    : "PRETRAINED model: loads a genuine trained 13.7 MB streaming voice converter. You do not need to upload a model.";
 $("voice-mode-detail").textContent=details;
 $("download-status").textContent=details;
}
$("voice").addEventListener("change",describeVoiceSelection);
describeVoiceSelection();
$("volume").addEventListener("input",()=>{
 $("volume-value").textContent=$("volume").value+"%";
 if(app.gain&&app.ctx)app.gain.gain.setTargetAtTime(Number($("volume").value)/100,app.ctx.currentTime,.015);
});
$("start").addEventListener("click",()=>{void start()});
$("audit-start").addEventListener("click",()=>{
 if(!app.active||app.audit)return;
 app.audit={mic:[],converted:[],frames:0};
 $("audit-results").hidden=true;
 $("audit-status").textContent="Recording up to 8 seconds from the input and actual ONNX output. Speak into your mic.";
 toggle();
});
$("engine-build").textContent="Neural runtime build: "+FAST_BUILD+
  " · worker fetched cache-free and version-verified";
$("reload-engine").addEventListener("click",()=>{
 void stop().then(()=>{
  const next=new URL("./fast-voice.html",window.location.href);
  next.searchParams.set("reload",FAST_BUILD+"-"+Date.now());
  window.location.replace(next.toString());
 });
});
$("stop").addEventListener("click",()=>{void stop("Stopped; mic closed and all converted audio muted.")});
window.addEventListener("pagehide",()=>{void stop()});
})();
