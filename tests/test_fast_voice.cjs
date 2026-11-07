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
 assert(outputs.some(x=>x.type==="build"&&x.build==="20261107-post-neural-pitch-v9"),
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
 // Exercise the browser's EXPLICIT post-ONNX pitch finish, not a mock
 // pitch-only bypass: even in this test, the model session runs first.
 // A steady 110 Hz input should be pitched up after identity neural fixture.
 function estimatedFrequency(pcm){
  let crossings=0;
  for(let i=1;i<pcm.length;i++)if(pcm[i-1]<=0&&pcm[i]>0)crossings++;
  return crossings/(pcm.length/16000);
 }
 await self.onmessage({data:{type:"init",model:new ArrayBuffer(66453),
  historySamples:2048,requireVoiceChange:false,pitchSemitones:6}});
 let pitched=null;
 for(let block=0;block<4;block++){
  const sine=new Float32Array(8192);
  for(let i=0;i<8192;i++)
   sine[i]=.2*Math.sin(2*Math.PI*110*(block*8192+i)/16000);
  const previous=outputs.filter(x=>x.type==="audio").length;
  await self.onmessage({data:{type:"convert",pcm:sine}});
  const next=outputs.filter(x=>x.type==="audio");
  assert.equal(next.length,previous+1,"Post-pitch DSP must not skip real ONNX inference");
  pitched=next.at(-1);
 }
 assert.equal(pitched.postPitchSemitones,6);
 assert.equal(pitched.pcm.length,8192,"Pitch finishing must not speed up or shorten speech");
 assert(pitched.pcm.every(Number.isFinite),"Pitch finisher produced non-finite samples");
 const outputHz=estimatedFrequency(pitched.pcm);
 assert(outputHz>110*1.20 && outputHz<110*1.65,
  "Post-neural pitch shifter failed to raise 110 Hz test input, measured "+outputHz.toFixed(1)+" Hz");
 await self.onmessage({data:{type:"config",pitchSemitones:0}});
 const pureInput=new Float32Array(8192).fill(.125);
 await self.onmessage({data:{type:"convert",pcm:pureInput}});
 assert.equal(outputs.filter(x=>x.type==="audio").at(-1).postPitchSemitones,0,
  "0 st should run model output without pitch DSP");
 await self.onmessage({data:{type:"init",model:new ArrayBuffer(32),historySamples:2048}});
 assert(outputs.some(x=>x.type==="error" && /empty, invalid, or incomplete/.test(x.text)),
        "Empty or obviously incomplete models must be rejected without leaking dry microphone");
 assert(control.includes("capture.connect(ctx.destination)"),"Fail-closed worklet is not connected");
 assert(!control.includes("app.source.connect(ctx.destination)"),"Dry microphone route detected");
 assert(!control.includes("src.connect(ctx.destination)"),"Direct microphone playback detected");
 assert(html.includes('<option value="8192" selected>512 ms'),
   "Default live microphone chunk should be 512 ms on slow Chromebook hardware");
 assert(html.includes('<option value="16000" selected>1000 ms'),
   "Pretrained voice should default to the native model's 1-second context instead of undersupplying 256ms");
 assert(control.includes('app.historySamples=Number($("history").value)||16000'),
   "History fallback should match native streaming 1-second context");
 assert(html.includes("Speaker identity: NOT VERIFIED"),
   "Changing waveform samples must never be called confirmed speaker conversion");
 assert(html.includes('value="saved"') && html.includes('value="custom"'),
   "Live UI needs distinct saved auto-load and manual-file ONNX modes");
 assert(control.includes("requireVoiceChange:true"),
   "Live controller must enforce no-passthrough protection for saved/manual ONNX models");
  assert(html.includes('id="rtf"'),"Real-time speed diagnostics missing");
 assert(html.includes('id="auto-level"')&&html.includes("Auto-match neural speech loudness"),
   "Auto loudness matching must be clearly labeled as playback processing only");
 assert(control.includes('rawIn/rawOut')&&control.includes('Math.min(3,rawIn/rawOut)'),
   "Output-only level matching must compensate quiet neural audio with a capped factor");
 assert(!control.includes("app.source.connect(app.gain)"),
   "Auto-level must never mix raw microphone into the neural output");
 assert(html.includes('id="pitch-finish"')&&html.includes('after neural ONNX'),
  "Pitch finish must be described as separate post-neural DSP");
 assert(html.includes('id="wave-change"')&&html.includes('id="wave-corr"'),
  "The UI must show measured neural waveform changes, not just claim AI output");
 assert(html.includes('id="audit-start"')&&html.includes('id="audit-output"')&&html.includes('id="audit-source"'),
  "Actual ONNX output and original microphone must have local side-by-side recorded previews");
 assert(control.includes("collectAudit(app.inFlightSource,pcm)"),
  "Only the real ONNX PCM received through the worker may be saved as the neural A/B test");
 assert(html.includes("fast-voice.js?v=20261107-post-neural-pitch-v9"),
   "Page must load matching versioned fast neural controller");
 assert(html.includes('id="reload-engine"')&&html.includes('id="engine-build"'),
   "Page must expose engine version and safe recovery reload");
 assert(control.includes('cache:"no-store"')&&control.includes("new Worker(blobUrl)"),
   "Fast Neural must fetch and verify the worker instead of trusting stale Worker() cache");
 assert(control.includes('const FAST_BUILD="20261107-post-neural-pitch-v9"'),
   "Controller and real streaming worker build IDs must agree");
 assert(!control.includes('new Worker("./fast-voice-worker.js?v=20261106-f1")'),
   "Obsolete fixed worker launch must not recur");
 assert(html.includes("fast-voice.js"),"Neural controller missing");
 console.log("PASS: real tiny ONNX pipeline wired, 16kHz mic decimation, one persistent model, no dry audio passthrough");
})().catch(e=>{console.error(e);process.exitCode=1});
