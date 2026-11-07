/* Regression tests for genuine tiny causal neural streaming and no dry mic fallback.
 * Heavyweight real model is separately bench-tested on GitHub Actions.
 */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const root=path.resolve(__dirname,"..");
const read=p=>fs.readFileSync(path.join(root,p),"utf8");
const source=read("live-vc/fast-voice-worker.js");
const worklet=read("live-vc/fast-mic-worklet.js");
const html=read("live-vc/fast-voice.html");
const control=read("live-vc/fast-voice.js");
let Mic=null,messages=[];
class AudioWorkletProcessor{constructor(){this.port={onmessage:null,postMessage(m){messages.push(m)}}}}
vm.runInNewContext(worklet,{
 AudioWorkletProcessor,registerProcessor(name,type){assert.equal(name,"fast-neural-mic");Mic=type},
 sampleRate:48000,Float32Array,Number,Math
});
const c=new Mic();
const output=new Float32Array(128);
const input=new Float32Array(128).fill(.5);
assert(c.process([[input]],[[output]]));
assert(output.every(x=>x===0),"Original microphone leaked before activation");
c.port.onmessage({data:{type:"config",chunkFrames:512}});
c.port.onmessage({data:{type:"enable",enabled:true}});
for(let i=0;i<15;i++){
 const out=new Float32Array(128).fill(1);
 c.process([[input]],[[out]]);
 assert(out.every(x=>x===0),"Original microphone leaked during recording");
}
assert(messages.length,"16 kHz microphone frames were not transferred");
const msg=messages[0];
assert.equal(msg.pcm.length,512);
assert.equal(msg.sampleRate,16000);
assert(msg.pcm.every(x=>Math.abs(x-.5)<1e-3),"Microphone downsample corrupt");
c.port.onmessage({data:{type:"enable",enabled:false}});
const stopped=new Float32Array(128).fill(1);
c.process([[input]],[[stopped]]);
assert(stopped.every(x=>x===0),"Stopping should silence microphone output");
class Tensor{constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims}}
let sessionLoads=0,sessionRuns=0;
const fakert={
 Tensor,env:{wasm:{}},
 InferenceSession:{async create(bytes,opts){
  sessionLoads++;
  assert(bytes.byteLength>=128 && bytes.byteLength<100000, "Small trained ONNX should be accepted by worker");
  assert.equal(opts.executionProviders[0],"wasm");
  return {
   inputNames:["audio_in"],outputNames:["audio_out"],
   async run(feeds){
    sessionRuns++;
    const t=feeds.audio_in;
    assert.equal(t.type,"float32");
    assert.equal(t.dims[1],1);
    const data=t.data.slice();
    return {audio_out:new Tensor("float32",data,[1,1,data.length])};
   },
   async release(){}
  };
 }}
};
const outputs=[],self={postMessage(x){outputs.push(x)},onmessage:null},context={
 self,Float32Array,ArrayBuffer,Number,Math,Promise,performance:{now:()=>Date.now()},
 importScripts(){context.ort=fakert},console
};
vm.runInNewContext(source,context);
(async()=>{
 await self.onmessage({data:{
  type:"init",model:new ArrayBuffer(65536),historySamples:2048
 }});
 assert(outputs.some(x=>x.type==="ready"),"Small trained neural ONNX did not initialize");
 assert(outputs.some(x=>x.type==="build"&&x.build==="20261106-cacheguard-v5"),
   "Worker build handshake must be available to fail closed on version mismatches");
 assert(outputs.some(x=>x.type==="stage" && /64\.0 KB neural ONNX/.test(x.text)),
        "Loading status must reflect the actual model size instead of saying 13.7 MB");
 assert.equal(sessionLoads,1);
 for(let i=0;i<3;i++){
  const pcm=new Float32Array(2048).fill(.3);
  await self.onmessage({data:{type:"convert",pcm}});
  assert(outputs.filter(x=>x.type==="audio").length===i+1);
  assert.equal(outputs.filter(x=>x.type==="audio").at(-1).pcm.length,2048);
 }
 assert.equal(sessionRuns,3);
 assert.equal(sessionLoads,1,"Neural session must persist across microphone chunks");
 await self.onmessage({data:{type:"init",model:new ArrayBuffer(32),historySamples:2048}});
 assert(outputs.some(x=>x.type==="error" && /empty, invalid, or incomplete/.test(x.text)),
        "Empty or obviously incomplete models must be rejected without leaking dry microphone");
 assert(control.includes("capture.connect(ctx.destination)"),"Fail-closed worklet is not connected");
 assert(!control.includes("app.source.connect(ctx.destination)"),"Dry microphone route detected");
 assert(!control.includes("src.connect(ctx.destination)"),"Direct microphone playback detected");
 assert(html.includes('id="rtf"'),"Real-time speed diagnostics missing");
 assert(html.includes("fast-voice.js?v=20261106-cacheguard-v5"),
   "Page must load matching versioned fast neural controller");
 assert(html.includes('id="reload-engine"')&&html.includes('id="engine-build"'),
   "Page must expose engine version and safe recovery reload");
 assert(control.includes('cache:"no-store"')&&control.includes("new Worker(blobUrl)"),
   "Fast Neural must fetch and verify the worker instead of trusting stale Worker() cache");
 assert(control.includes('const FAST_BUILD="20261106-cacheguard-v5"'),
   "Controller and real streaming worker build IDs must agree");
 assert(!control.includes('new Worker("./fast-voice-worker.js?v=20261106-f1")'),
   "Obsolete fixed worker launch must not recur");
 assert(html.includes("fast-voice.js"),"Neural controller missing");
 console.log("PASS: real tiny ONNX pipeline wired, 16kHz mic decimation, one persistent model, no dry audio passthrough");
})().catch(e=>{console.error(e);process.exitCode=1});
