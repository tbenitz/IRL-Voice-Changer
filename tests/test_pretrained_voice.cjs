/* Offline real audio test of published HEVoicechanger 13.7 MB pretrained ONNX
 * through the same history + chunk + tail segmentation as browser converter.
 * This does NOT certify a target speaker resemblance. Produces 16kHz real WAVs.
 */
"use strict";
const fs=require("node:fs"),ort=require("onnxruntime-web");
const path=require("node:path");
const DIR=process.env.VOICE_TEST_DIR||"/tmp/irl-voice-audio-test";
const root=path.join(process.cwd(),"node_modules","onnxruntime-web","dist")+path.sep;
ort.env.wasm.wasmPaths=root;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
const from=fs.readFileSync(path.join(DIR,"input.f32"));
const src=new Float32Array(from.buffer.slice(from.byteOffset,from.byteOffset+from.byteLength));
function wav(file,data){
 const b=Buffer.alloc(44+data.length*2);b.write("RIFF",0);b.writeUInt32LE(36+data.length*2,4);
 b.write("WAVEfmt ",8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);
 b.writeUInt32LE(16000,24);b.writeUInt32LE(32000,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);
 b.write("data",36);b.writeUInt32LE(data.length*2,40);
 for(let i=0;i<data.length;i++){const x=Math.max(-1,Math.min(1,data[i]));b.writeInt16LE(Math.round(x*(x<0?32768:32767)),44+i*2)}
 fs.writeFileSync(path.join(DIR,file),b);
}
function stats(x,y){
 let dry=0,wet=0,delta=0,cross=0;
 for(let i=0;i<x.length;i++){
  dry+=x[i]*x[i];wet+=y[i]*y[i];delta+=(x[i]-y[i])**2;cross+=x[i]*y[i];
 }
 return {inputRMS:Math.sqrt(dry/x.length),outputRMS:Math.sqrt(wet/x.length),
  normalizedDiff:Math.sqrt(delta/Math.max(dry,1e-12)),
  corr:cross/Math.sqrt(Math.max(dry*wet,1e-12))};
}
function pitch(pcm){
 const sr=8000,w=640, results=[];
 for(let base=2000;base+w*2<pcm.length;base+=2500){
  const x=new Float32Array(w);
  let avg=0;for(let i=0;i<w;i++){x[i]=pcm[base+2*i];avg+=x[i]}
  avg/=w;for(let i=0;i<w;i++)x[i]-=avg;
  let energy=0;for(const v of x)energy+=v*v;
  if(energy<.005)continue;
  let best=0,bestLag=0;
  for(let lag=18;lag<=130;lag++){
   let cross=0,a=0,b=0;
   for(let i=0;i<w-lag;i++){const x0=x[i],x1=x[i+lag];cross+=x0*x1;a+=x0*x0;b+=x1*x1}
   const score=cross/Math.sqrt(Math.max(1e-12,a*b));
   if(score>best){best=score;bestLag=lag}
  }
  if(best>.66)results.push(sr/bestLag);
 }
 return {n:results.length,median:results.sort((a,b)=>a-b)[Math.floor(results.length/2)]||null};
}
(async()=>{
 const sess=await ort.InferenceSession.create(fs.readFileSync(path.join(DIR,"female_voice_hfg_new.onnx")),{
   executionProviders:["wasm"],graphOptimizationLevel:"basic",
   enableMemPattern:false,enableCpuMemArena:false
 });
 const historySize=16000,chunkSize=8192;
 const history=new Float32Array(historySize);
 const output=new Float32Array(src.length);
 let totalMs=0,chunks=0;
 for(let start=0;start<src.length;start+=chunkSize){
  const part=src.subarray(start,Math.min(start+chunkSize,src.length));
  if(part.length<512)break;
  const data=new Float32Array(historySize+part.length);
  data.set(history);data.set(part,historySize);
  const t=performance.now();
  const result=await sess.run({audio_in:new ort.Tensor("float32",data,[1,1,data.length])});
  totalMs+=performance.now()-t;chunks++;
  const o=result.audio_out?.data;
  if(!o||o.length<data.length)throw Error("ONNX returned invalid output");
  const tail=o.subarray(o.length-part.length);
  output.set(tail,start);
  if(part.length>=historySize)history.set(part.subarray(part.length-historySize));
  else{history.copyWithin(0,part.length);history.set(part,historySize-part.length)}
 }
 if(!output.every(Number.isFinite))throw Error("Nonfinite ONNX speech samples");
 wav("voice-source.wav",src);wav("voice-onnx-13mb.wav",output);
 const result={
  samples:src.length,inputPitch:pitch(src),outputPitch:pitch(output),...stats(src,output),
  model:"Published 13.7 MB female_voice_hfg_new.onnx",historySize,chunkSize,
  totalInferenceMs:Math.round(totalMs),RTF:+(totalMs/(src.length/16)).toFixed(3),
  note:"Pitch estimates and waveform differences are NOT a speaker similarity score."
 };
 fs.writeFileSync(path.join(DIR,"metrics.json"),JSON.stringify(result,null,2));
 console.log("REAL PRETRAINED VOICE CHECK",JSON.stringify(result));
 await sess.release();
})().catch(e=>{console.error(e);process.exitCode=1});
