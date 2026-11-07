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
  type:"init",model:new ArrayBuffer(65536),historySamples:2048,requireVoiceChange:false
 }});
 assert(outputs.some(x=>x.type==="ready"),"Small trained neural ONNX did not initialize");
 assert(outputs.some(x=>x.type==="build"&&x.build==="20261107-custom-training-fix-v11"),
   "Worker build handshake must be available to fail closed on version mismatches");
 assert(outputs.some(x=>x.type==="stage" && /64\.0 KB neural ONNX/.test(x.text)),
        "Loading status must reflect the actual model size instead of saying 13.7 MB");
 assert.equal(sessionLoads,1);
 for(let i=0;i<3;i++){
  const pcm=new Float32Array(2048).fill(.3);
  await self.onmessage({data:{type:"convert",pcm}});
  assert(outputs.filter(x=>x.type==="audio").length===i+1);
  assert.equal(outputs.filter(x=>x.type==="audio").at(-1).pcm.length,2048);
  assert(Number.isFinite(outputs.filter(x=>x.type==="audio").at(-1).changeRatio),
    "Worker must report measurable ONNX output-vs-input differences");
 }
 assert.equal(sessionRuns,3);
 assert.equal(sessionLoads,1,"Neural session must persist across microphone chunks");
 // Test the user's exact complaint: an experimental ONNX returning input audio
 // must NEVER feed essentially unmodified speech to the converted output.
 const playedBefore=outputs.filter(x=>x.type==="audio").length;
 await self.onmessage({data:{type:"init",model:new ArrayBuffer(66453),
  historySamples:2048,requireVoiceChange:true}});
 for(let i=0;i<3;i++)
  await self.onmessage({data:{type:"convert",pcm:new Float32Array(8192).fill(.25)}});
 assert.equal(outputs.filter(x=>x.type==="audio").length,playedBefore,
  "Near-identical experimental model played the original mic as converted speech");
 assert(outputs.some(x=>x.type==="error"&&/essentially the same microphone waveform|almost silence/.test(x.text)),
  "Custom model should explain why near-identity voice conversion was muted");
 await self.onmessage({data:{type:"init",model:new ArrayBuffer(32),historySamples:2048}});
 assert(outputs.some(x=>x.type==="error" && /empty, invalid, or incomplete/.test(x.text)),
        "Empty or obviously incomplete models must be rejected without leaking dry microphone");
 assert(control.includes("capture.connect(ctx.destination)"),"Fail-closed worklet is not connected");
 assert(!control.includes("app.source.connect(ctx.destination)"),"Dry microphone route detected");
 assert(!control.includes("src.connect(ctx.destination)"),"Direct microphone playback detected");
 assert(html.includes('<option value="8192" selected>512 ms'),
   "Default live microphone chunk should be 512 ms on slow Chromebook hardware");
 assert(html.includes('value="saved"') && html.includes('value="custom"'),
   "Live UI needs distinct saved auto-load and manual-file ONNX modes");
 assert(control.includes("requireVoiceChange:true"),
   "Live controller must enforce no-passthrough protection for saved/manual ONNX models");
 assert(control.includes("app.queue.push(block)")&&!control.includes("app.pending=block"),
   "Never silently replace one captured speech chunk with another");
 assert(control.includes('app.requestedHistory=next')&&control.includes('app.worker.postMessage({type:"config",historySamples:next})'),
   "Long 500/1000ms context must automatically reduce on actual inference overload");
 assert(source.includes("resetHistory(Number(m.historySamples)||4096,true)"),
   "Worker must retain speech history when downshifting long context instead of inserting zeros");
 assert(control.includes("NEURAL OVERLOAD:"),
   "Real-time inference overrun must stop clearly instead of pretending chopped speech is working");
 assert(html.includes('<option value="4096" selected>256 ms'),
   "Restore working pretrained previous-context default; user did not request 1000ms history");
 assert(control.includes('model.trainingData!=="teacher-parallel"'),
   "Old unaligned Pocket pseudo-voice must not play as a real custom neural model");
  assert(html.includes('id="rtf"'),"Real-time speed diagnostics missing");
 assert(html.includes('id="wave-change"')&&html.includes('id="wave-corr"'),
  "The UI must show measured neural waveform changes, not just claim AI output");
 assert(html.includes('id="audit-start"')&&html.includes('id="audit-output"')&&html.includes('id="audit-source"'),
  "Actual ONNX output and original microphone must have local side-by-side recorded previews");
 assert(control.includes("collectAudit(app.inFlightSource,pcm)"),
  "Only the real ONNX PCM received through the worker may be saved as the neural A/B test");
 assert(html.includes("fast-voice.js?v=20261107-custom-training-fix-v11"),
   "Page must load matching versioned fast neural controller");
 assert(html.includes('id="reload-engine"')&&html.includes('id="engine-build"'),
   "Page must expose engine version and safe recovery reload");
 assert(control.includes('cache:"no-store"')&&control.includes("new Worker(blobUrl)"),
   "Fast Neural must fetch and verify the worker instead of trusting stale Worker() cache");
 assert(control.includes('const FAST_BUILD="20261107-custom-training-fix-v11"'),
   "Controller and real streaming worker build IDs must agree");
 assert(!control.includes('new Worker("./fast-voice-worker.js?v=20261106-f1")'),
   "Obsolete fixed worker launch must not recur");
 assert(html.includes("fast-voice.js"),"Neural controller missing");
 console.log("PASS: real tiny ONNX pipeline wired, 16kHz mic decimation, one persistent model, no dry audio passthrough");
})().catch(e=>{console.error(e);process.exitCode=1});
