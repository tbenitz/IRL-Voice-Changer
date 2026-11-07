/* End-to-end RVC worker smoke test with fake ONNX session outputs.
 * Exercises actual JS pipeline wiring only; real model weights NOT tested.
 */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const source=fs.readFileSync(path.join(__dirname,"../live-vc/rvc-stream-worker.js"),"utf8");
const msgs=[],stub={};
class Tensor {
 constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims}
}
const rate=16000;
let sessions=0,genCalls=0,featureCalls=0,pitchCalls=0;
const fakeORT={
 Tensor,env:{wasm:{}},
 InferenceSession:{
  async create(_bytes,_opts){
   const index=sessions++;
   if(index===0)return {
    inputNames:["audio"],inputMetadata:{audio:{dimensions:[1,"samples"]}},outputNames:["units9","unit12","unit12s"],
    async run(feeds){
     featureCalls++;
     assert(feeds.audio instanceof Tensor);
     const n=feeds.audio.data.length,frames=Math.max(1,Math.floor(n/320));
     const v1=new Float32Array(256*frames).fill(.27);
     const v2=new Float32Array(768*frames).fill(.12);
     const v2s=new Float32Array(768*frames).fill(.14);
     return {
      units9:new Tensor("float32",v1,[1,frames,256]),
      unit12:new Tensor("float32",v2,[1,frames,768]),
      unit12s:new Tensor("float32",v2s,[1,frames,768])
     };
    }
   };
   if(index===1)return {
    inputNames:["waveform"],outputNames:["f0"],
    async run(feeds){
     pitchCalls++;
     const n=feeds.waveform.data.length,frames=Math.ceil(n/160);
     return {f0:new Tensor("float32",new Float32Array(frames).fill(210),[1,frames])};
    }
   };
   return {
    inputNames:["phone","phone_lengths","pitch","pitchf","ds","rnd"],
    inputMetadata:{phone:{dimensions:[1,512,768]}},
    outputNames:["audio"],
    async run(feeds){
     genCalls++;
     assert(feeds.phone instanceof Tensor);
     assert(feeds.pitch instanceof Tensor);
     assert.equal(feeds.phone.dims[2],768);
     assert.equal(feeds.phone.dims[1],512,"Fixed-size voice must receive 512 padded frames");
     assert.equal(feeds.rnd.dims[2],512,"Fixed frame noise shape wrong");
     const count=feeds.phone.dims[1]*480,pcm=new Float32Array(count);
     for(let i=0;i<count;i++)pcm[i]=.2*Math.sin(i*2*Math.PI*220/48000);
     return {audio:new Tensor("float32",pcm,[1,1,count])};
    }
   };
  }
 }
};
const self={
 onmessage:null,
 postMessage(message){msgs.push(message)}
};
const context={
 self, Float32Array,Uint8Array,BigInt64Array,Array,Number,Math,BigInt,Promise,
 console,setTimeout,performance:{now:()=>Date.now()},
 importScripts(){context.ort=fakeORT}
};
vm.runInNewContext(source,context);

const normalize=vm.runInNewContext("contentVecTensorToFeatures",context);
{
 // Regression: a 256-frame, 768-channel ContentVec output was incorrectly
 // identified as 256-channel by the previous if(d1===256) branch.
 const timeMajor=new Float32Array(256*768);
 timeMajor[0]=1.5;
 timeMajor[767]=2.5;
 timeMajor[768]=3.5;
 const x=normalize(new Tensor("float32",timeMajor,[1,256,768]),768);
 assert.equal(x.dim,768);
 assert.equal(x.frames,256);
 assert.equal(x.layout,"time-major");
 assert.equal(x.features[767],2.5);
 assert.equal(x.features[768],3.5);
 const channelMajor=new Float32Array(768*256);
 channelMajor[0]=10;
 channelMajor[256]=20;
 channelMajor[1]=30;
 const y=normalize(new Tensor("float32",channelMajor,[1,768,256]),768);
 assert.equal(y.dim,768);
 assert.equal(y.frames,256);
 assert.equal(y.layout,"channel-major");
 assert.equal(y.features[0],10);
 assert.equal(y.features[1],20);
 assert.equal(y.features[768],30);
 assert.throws(()=>normalize(new Tensor("float32",new Float32Array(256*91),[1,91,256]),768),
               /no 768-dimensional feature axis/,"Genuine RVC v1 features must not be disguised as RVC v2");
 console.log("PASS: 256-frame ContentVec axes resolve as genuine 768-D channels (both layouts); incompatible 256-D rejected.");
}

const wait=(ms=80)=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 assert.equal(typeof self.onmessage,"function");
 await self.onmessage({data:{
  type:"init",voice:new ArrayBuffer(12),contentVec:new ArrayBuffer(12),
  rmvpe:new ArrayBuffer(12),sampleRate:48000,fixedVoiceFrames:512
 }});
 assert(msgs.some(x=>x.type==="ready"),"ONNX models failed to load");
 assert.equal(sessions,3,"Sessions should be created only on init");
 for(let c=0;c<2;c++){
  const samples=new Float32Array(rate);
  for(let i=0;i<samples.length;i++)samples[i]=.16*Math.sin(i*2*Math.PI*220/rate);
  await self.onmessage({data:{type:"convert",pcm:samples,sampleRate:rate}});
  const converted=msgs.filter(x=>x.type==="audio"&&!x.silence).at(-1);
  assert(converted,"RVC generator produced no model output");
  assert(converted.pcm instanceof Float32Array,"Output not float32");
  assert(converted.pcm.some(x=>Math.abs(x)>.1),"Converted audio all silent");
  assert(converted.pcm.every(Number.isFinite),"Converted audio contains invalid values");
  assert(converted.pcm.length < 512*480,"Padded output was not trimmed to actual microphone block");
 }
 assert.equal(sessions,3,"ONNX sessions recreated for every block: not streaming");
 assert.equal(featureCalls,2);
 assert.equal(pitchCalls,2);
 assert.equal(genCalls,2);
 assert(msgs.some(x=>x.type==="stage" && /ContentVec unit12 .*768 channels/.test(x.text)),
        "Did not select the correct 768-channel 'unit12' output after the first 256-channel 'units9' output");
 assert(msgs.some(x=>x.type==="idle"),"Worker never released inference busy status");
 console.log("PASS: persistent RVC ONNX sessions and two microphone chunks converted by content, F0 and generator pipeline (mock ONNX only).");
})().catch(e=>{console.error(e);process.exitCode=1});
