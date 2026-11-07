/* Actual low-latency causal neural voice conversion.
   Model: ultrapg/hevoicechanger GPL-3.0 streaming 16 kHz ONNX.
   No TTS, pitch-effect masquerade, or raw microphone passthrough.
   Browser WASM inference; main thread owns capture/output and keeps voice muted on errors.
 */
"use strict";
const WORKER_BUILD="20261107-output-audit-v8";
const ORT_DIST="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
let ortReady=false, session=null, history=new Float32Array(4096), busy=false, initialized=false;
let sourceRate=16000;
let requireVoiceChange=false,unchangedSpeechChunks=0;
function send(type,details={},transfer=[]){self.postMessage({type,...details},transfer)}
function err(e){return e?.stack||e?.message||String(e)}
// Handshake proves the exact compiled worker running the audio pipeline.
send("build",{build:WORKER_BUILD});
async function setupRuntime(){
 if(ortReady)return;
 importScripts(ORT_DIST+"ort.min.js");
 ort.env.wasm.wasmPaths=ORT_DIST;
 ort.env.wasm.numThreads=1; // no cross-origin isolation needed; low-memory mode
 ort.env.wasm.proxy=false;
 ort.env.logLevel="error";
 ortReady=true;
}
function resetHistory(count){
 if(!Number.isInteger(count)||count<1024||count>16000)throw Error("History must be 1024–16000 samples");
 history=new Float32Array(count);
}
async function loadModel(bytes){
 // Pocket's locally trained 4-layer ONNX can be ~66 KB. Reject empty data,
 // not small-but-valid models: ONNX Runtime checks the graph itself.
 if(!(bytes instanceof ArrayBuffer)||bytes.byteLength<128)
   throw Error("The streaming ONNX file is empty, invalid, or incomplete ("+
     (bytes?.byteLength??0)+" bytes).");
 await setupRuntime();
 send("stage",{text:"Initializing "+(bytes.byteLength/1024).toFixed(1)+
   " KB neural ONNX model in WebAssembly…"});
 session=await ort.InferenceSession.create(bytes,{
  executionProviders:["wasm"],graphOptimizationLevel:"basic",
  enableMemPattern:false,enableCpuMemArena:false
 });
 const ins=session.inputNames,outs=session.outputNames;
 if(!ins.includes("audio_in")||!outs.includes("audio_out"))
  throw Error("Unsupported model: expected audio_in → audio_out, got "+
   ins.join(",")+" → "+outs.join(","));
 initialized=true;unchangedSpeechChunks=0;
 send("ready",{inputs:ins,outputs:outs,identityGuard:requireVoiceChange});
}
async function infer(chunk){
 if(!initialized||!session)throw Error("Model not initialized");
 if(!(chunk instanceof Float32Array)||chunk.length<256||chunk.length>16000)
  throw Error("Expected 16kHz mono Float32 PCM chunk with 256–16000 samples.");
 const start=performance.now();
 const n=history.length+chunk.length;
 const input=new Float32Array(n);
 input.set(history);
 input.set(chunk,history.length);
 const result=await session.run({audio_in:new ort.Tensor("float32",input,[1,1,n])});
 const generated=result.audio_out;
 if(generated?.data?.length<n||generated.dims?.length!==3)
  throw Error("Causal model produced unexpected output dimensions "+
    (generated?.dims?.join("x")||"none"));
 // The model generates historical context too. Only its NEW block is played.
 const pcm=new Float32Array(chunk.length);
 const tail=generated.data.subarray(generated.data.length-chunk.length);
 for(let i=0;i<pcm.length;i++){
  const v=Number(tail[i]);if(!Number.isFinite(v))throw Error("Model produced invalid audio");
  pcm[i]=Math.min(1,Math.max(-1,v));
 }
 // Measure the ACTUAL model output relative to the current microphone
 // input, not merely whether an ONNX session returned "success".
 // Waveform difference is not a speaker-identity test.
 let deltaSum=0,inputSum=0,outputSum=0,dot=0,dc=0,nMeasured=0;
 for(let i=0;i<pcm.length;i+=4){
  const dry=chunk[i],wet=pcm[i],d=wet-dry;
  deltaSum+=d*d;inputSum+=dry*dry;outputSum+=wet*wet;dot+=dry*wet;
  dc+=wet;nMeasured++;
 }
 const inputRms=Math.sqrt(inputSum/Math.max(1,nMeasured));
 const outputRms=Math.sqrt(outputSum/Math.max(1,nMeasured));
 const changeRatio=Math.sqrt(deltaSum/Math.max(inputSum,1e-12));
 const correlation=dot/Math.sqrt(Math.max(inputSum*outputSum,1e-12));
 const speech=inputRms>.012;
 // Gain-only/phase-identical changes are not a new voice. Require several
 // consecutive voiced blocks before rejecting to avoid startup transients.
 const nearIdentity=speech&&correlation>.995;
 const silentOutput=speech&&outputRms<.003;
 if(requireVoiceChange&&speech&&(nearIdentity||silentOutput)){
  unchangedSpeechChunks++;
  const reason=silentOutput?"Neural ONNX model produced almost silence for audible microphone input":
   "Neural ONNX model returned essentially the same microphone waveform";
  if(unchangedSpeechChunks>=3)throw Error(reason+
   " (waveform correlation "+correlation.toFixed(4)+
   ", normalized change "+changeRatio.toFixed(3)+
   ", input RMS "+inputRms.toFixed(3)+
   ", output RMS "+outputRms.toFixed(3)+
   "). No unmodified audio will be played. "+
   "The 65 KB Pocket student cannot currently be considered a working speaker clone; "+
   "try a pretrained 13.7 MB voice model to test the engine.");
  send("dropped",{reason:reason+"; muting this block pending verification",
   inputRms,outputRms,changeRatio,correlation});
  if(chunk.length>=history.length)history.set(chunk.subarray(chunk.length-history.length));
  else{history.copyWithin(0,chunk.length);history.set(chunk,history.length-chunk.length)}
  return;
 }
 if(speech)unchangedSpeechChunks=0;
 if(chunk.length>=history.length)history.set(chunk.subarray(chunk.length-history.length));
 else {
  history.copyWithin(0,chunk.length);
  history.set(chunk,history.length-chunk.length);
 }
 const inferenceMs=performance.now()-start;
 send("audio",{pcm,inferenceMs,chunkMs:chunk.length/16,rtf:inferenceMs/(chunk.length/16),
  inputRms,outputRms,changeRatio,correlation},[pcm.buffer]);
}
self.onmessage=async event=>{
 const m=event.data||{};
 if(m.type==="stop"){
  initialized=false;session?.release?.().catch(()=>{});session=null;
  history.fill(0);return;
 }
 if(m.type==="init"){
  if(busy)return;busy=true;
  try{
   initialized=false;session?.release?.().catch(()=>{});session=null;
   resetHistory(Number(m.historySamples)||4096);
   requireVoiceChange=m.requireVoiceChange!==false;
   unchangedSpeechChunks=0;
   await loadModel(m.model);
  }catch(e){send("error",{text:"Tiny neural model init: "+err(e)})}
  finally{busy=false}
 }else if(m.type==="config"){
  if(!busy){
   try{resetHistory(Number(m.historySamples)||4096);send("stage",{text:"Neural history window updated"})}
   catch(e){send("error",{text:err(e)})}
  }
 }else if(m.type==="convert"){
  if(!initialized||busy){send("dropped",{reason:"Model busy or not ready"});return}
  busy=true;
  try{await infer(m.pcm)}
  catch(e){send("error",{text:"Neural inference: "+err(e)})}
  finally{busy=false;send("idle")}
 }
};
