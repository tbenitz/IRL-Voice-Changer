/* Node smoke tests for fail-closed neural microphone AudioWorklet and RVC wiring.
 * No real ONNX model weights are loaded in CI. Hardware inference requires Chrome.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root=path.resolve(__dirname,"..");
const get = p => fs.readFileSync(path.join(root,p),"utf8");
const captureSource=get("live-vc/rvc-mic-worklet.js");
const workerSource=get("live-vc/rvc-stream-worker.js");
const controllerSource=get("live-vc/rvc-live-ui.js");
const html=get("live-vc/index.html");
const comparisons=get("online-test/index.html");
let Capturer=null, transferred=[];
class AudioWorkletProcessor {
 constructor(){
  this.port={onmessage:null,postMessage(message){transferred.push(message)}};
 }
}
vm.runInNewContext(captureSource,{
 AudioWorkletProcessor, sampleRate:16000, Float32Array, Number,Math,
 registerProcessor(name,klass){assert.equal(name,"rvc-mic-capture");Capturer=klass}
});
assert(Capturer,"Neural microphone processor did not register");
const c=new Capturer(),length=128;
const output=new Float32Array(length),audio=new Float32Array(length).fill(.37);
c.process([[audio]],[[output]]);
assert(output.every(x=>x===0),"Raw mic leaked before start");
c.port.onmessage({data:{type:"config",chunkMs:700}});
c.port.onmessage({data:{type:"enable",enabled:true}});
for(let j=0;j<93;j++){
 const rendered=new Float32Array(length);
 c.process([[audio]],[[rendered]]);
 assert(rendered.every(x=>x===0),"Raw microphone leaked during active capture");
}
assert(transferred.length>0,"Mic capture did not send any audio to the model path");
const message=transferred[0];
assert.equal(message.type,"audio");
assert(message.data.length>=11000 && message.data.length<=12000,"Wrong capture chunk duration");
assert(message.data.every(x=>Math.abs(x-.37)<1e-5),"Captured PCM corrupted");
c.port.onmessage({data:{type:"enable",enabled:false}});
const silence=new Float32Array(length);
c.process([[audio]],[[silence]]);
assert(silence.every(x=>x===0),"Mic leaked after stop");
const self={onmessage:null,postMessage(){}};
vm.runInNewContext(workerSource,{self,Float32Array,Number,Math,BigInt,BigInt64Array,Uint8Array,Int32Array,Array,Promise,performance:{now:()=>0},setTimeout,console});
assert.equal(typeof self.onmessage,"function","Worker did not register message handling");
assert(workerSource.includes('cv=await session(d.contentVec'),"ContentVec session missing");
assert(workerSource.includes('pitchModel=await session(d.rmvpe'),"RMVPE session missing");
assert(workerSource.includes('gen=await session(d.voice'),"RVC generator session missing");
assert(workerSource.includes('await gen.run(synthesizerFeeds('),"Real voice model inference missing");
assert(workerSource.includes('await pitchModel.run('),"Real pitch model inference missing");
assert(workerSource.includes('await cv.run('),"Real content model inference missing");
assert(!workerSource.includes('speechSynthesis'),"TTS is not voice conversion");
for(const id of ["rvc-live-start","rvc-live-stop","rvc-live-lag","rvc-live-processing","rvc-live-state"]){
 assert(html.includes('id="'+id+'"'),"Missing neural RVC UI control "+id);
}
assert(html.includes('src="./rvc-live-ui.js'),'Live RVC controller not mounted');
assert(controllerSource.includes('app.worker.postMessage({type:"convert"'),'Audio chunks not routed to actual neural worker');
assert(!controllerSource.includes('app.source.connect(ctx.destination)'),'Dry mic output route detected');
assert(!html.includes("<iframe")&&!comparisons.includes("<iframe"),"Hosted iframe is forbidden");
console.log("PASS: microphone outputs silence, streams real PCM to RVC worker, persistent ONNX model inference wired, no iframes or TTS substitution.");
