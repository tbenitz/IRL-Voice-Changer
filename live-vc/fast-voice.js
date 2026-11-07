/* Fast AI voice changer controller: real causal neural ONNX, local WASM.
   No input->speaker path. Only model-generated Float32 PCM reaches the gain.
   One worker/model, a latest-only pending chunk, and a bounded playback queue.
 */
(() => {
"use strict";
const $=id=>document.getElementById(id);
const MODEL_ROOT="https://raw.githubusercontent.com/ultrapg/hevoicechanger/main/models/";
const voices={
 hfg:{name:"female_voice_hfg_new.onnx",bytes:13756183},
 old_hq:{name:"female_voice_old_hq.onnx",bytes:13718163},
 old:{name:"female_voice_old.onnx",bytes:13756183}
};
const app={token:0,loading:false,active:false,worker:null,stream:null,ctx:null,
 source:null,capture:null,gain:null,sources:new Set(),abort:null,inFlight:false,pending:null,
 nextPlayTime:0,drops:0,chunkFrames:2048,historySamples:4096,lastCapture:0};
const CACHE="irl-fast-neural-voice-models", STORE="weights";
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
async function getModel(){
 const choice=$("voice").value;
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
 try{await app.ctx?.close()}catch(_){}
 Object.assign(app,{abort:null,worker:null,ctx:null,stream:null,source:null,capture:null,
  gain:null,inFlight:false,pending:null,nextPlayTime:0,lastCapture:0});
 $("download").hidden=true;
 toggle();badge("OFF · MUTED");
 if(message)info(message);
}
function inputChunk(pcm){
 if(!app.active||!app.worker)return;
 meters("input-meter",pcm);
 const block={pcm,at:performance.now()};
 if(app.inFlight){
  if(app.pending)app.drops++;
  app.pending=block;
  $("drops").textContent=String(app.drops);
  return;
 }
 convert(block);
}
function convert(block){
 if(!app.active||!app.worker)return;
 app.inFlight=true;app.lastCapture=block.at;
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
 $("performance-note").textContent=rtf<=.7?"Running faster than real time with useful headroom.":
  rtf<1?"Near the processing limit; try a larger chunk if audio skips.":
  "This setting is slower than real-time on your device. Try a shorter history or larger chunk.";
 const actualDelay=performance.now()-app.lastCapture+app.chunkFrames/16;
 if(actualDelay>1900){
  app.drops++;$("drops").textContent=String(app.drops);
  info("Neural audio exceeded 1.9 seconds of delay; dropped, never replaced with original mic.",true);
  return;
 }
 const scheduled=Math.max(ctx.currentTime+.025,app.nextPlayTime);
 if(scheduled>ctx.currentTime+.65){
  app.drops++;$("drops").textContent=String(app.drops);
  app.nextPlayTime=ctx.currentTime;
  info("Processing queue overfull; dropping converted block to catch up.",true);
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
 app.nextPlayTime=scheduled+buf.duration;
 meters("output-meter",pcm);
 $("lag").textContent=Math.round(actualDelay+(scheduled-ctx.currentTime)*1000);
 info("Live neural voice is playing (not the raw mic). "+Math.round(elapsed)+
  " ms model inference for "+Math.round(app.chunkFrames/16)+" ms audio, RTF "+rtf.toFixed(2)+".");
}
function eventFromWorker(e,token){
 if(token!==app.token)return;
 const d=e.data||{};
 if(d.type==="stage")info(d.text||"Loading neural ONNX…");
 else if(d.type==="ready") {
  badge("MODEL READY");info("Model initialized. Starting microphone…");
  void activateMic(token).catch(err=>{void stop("Microphone failure: "+(err?.message||err)+". Input muted.")});
 }else if(d.type==="audio")play(d);
 else if(d.type==="idle"){
  app.inFlight=false;
  if(app.active&&app.pending){const next=app.pending;app.pending=null;convert(next)}
 }else if(d.type==="dropped"){
  app.drops++;$("drops").textContent=String(app.drops);
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
 info("Running genuine causal neural voice conversion locally at 16 kHz. Headphones recommended.");
}
async function start(){
 if(app.loading||app.active)return;
 if(!window.isSecureContext||!window.AudioWorkletNode||!navigator.mediaDevices?.getUserMedia){
  info("Requires Chrome over HTTPS with microphone and AudioWorklet permission.",true);return;
 }
 app.loading=true;app.drops=0;app.inFlight=false;app.pending=null;
 app.chunkFrames=Number($("chunk").value)||2048;
 app.historySamples=Number($("history").value)||4096;
 $("drops").textContent="0";$("lag").textContent="—";$("inference").textContent="—";$("rtf").textContent="—";
 toggle();badge("LOADING AI");
 const token=++app.token;
 try{
  // Output is unlocked by the user's click, before any async download.
  const Ctx=window.AudioContext||window.webkitAudioContext;
  app.ctx=new Ctx({latencyHint:"interactive"});
  await app.ctx.resume();
  const weights=await getModel();
  if(token!==app.token)return;
  info("Initializing the streaming AI model in a dedicated browser worker…");
  const worker=new Worker("./fast-voice-worker.js?v=20261106-f1");
  app.worker=worker;
  worker.onmessage=e=>eventFromWorker(e,token);
  worker.onerror=e=>{if(token===app.token)void stop("Neural worker crashed: "+(e.message||"unknown"))};
  worker.postMessage({type:"init",model:weights,historySamples:app.historySamples},[weights]);
 }catch(e){
  if(token===app.token)await stop("Could not initialize the real neural model: "+(e?.message||e));
 }
}
$("voice").addEventListener("change",()=>{
 $("custom-field").hidden=$("voice").value!=="custom";
 $("download-status").textContent="Selected a neural voice. Download starts only when you click Start.";
});
$("volume").addEventListener("input",()=>{
 $("volume-value").textContent=$("volume").value+"%";
 if(app.gain&&app.ctx)app.gain.gain.setTargetAtTime(Number($("volume").value)/100,app.ctx.currentTime,.015);
});
$("start").addEventListener("click",()=>{void start()});
$("stop").addEventListener("click",()=>{void stop("Stopped; mic closed and all converted audio muted.")});
window.addEventListener("pagehide",()=>{void stop()});
})();
