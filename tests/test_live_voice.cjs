/* Node-only source-level/runtime smoke test. Does not prove browser RVC performance. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
const processorSource = fs.readFileSync(path.join(root,"live-vc/live-processor.js"),"utf8");
const html = fs.readFileSync(path.join(root,"live-vc/index.html"),"utf8");
const workflow = fs.readFileSync(path.join(root,".github/workflows/pages.yml"),"utf8");
const home = fs.readFileSync(path.join(root,"index.html"),"utf8");
let Worklet = null;
class AudioWorkletProcessor { constructor(){this.port={onmessage:null,postMessage:()=>{}}} }
vm.runInNewContext(processorSource,{
 AudioWorkletProcessor,
 sampleRate:48000,
 registerProcessor(name,klass){assert.equal(name,"irl-live-voice");Worklet=klass},
 Float32Array,Math,Number
});
assert(Worklet,"Worklet class should register");
const worklet=new Worklet();
const out=new Float32Array(128);
const inp=new Float32Array(128).fill(0.25);
worklet.process([[inp]],[[out]]);
assert(out.every(x=>x===0),"Mic audio must NEVER pass through before activation");
worklet.port.onmessage({data:{type:"configure",pitch:-5,brightness:.52,gate:-49,saturation:.08,volume:.77}});
worklet.port.onmessage({data:{type:"enable",value:true}});
let changed=[];
for(let b=0;b<240;b++){
 const src=new Float32Array(128);
 for(let i=0;i<128;i++) src[i]=.28*Math.sin((b*128+i)*2*Math.PI*220/48000);
 const dst=new Float32Array(128);
 worklet.process([[src]],[[dst]]);
 if(b>90)changed.push(...dst);
}
assert(changed.some(x=>Math.abs(x)>.025),"Transformed stream should produce audible output");
assert(changed.every(Number.isFinite),"Output must remain finite");
assert(changed.some((x,i)=>Math.abs(x-.28*Math.sin((91*128+i)*2*Math.PI*220/48000))>.12),"Transformed output must not be a simple wire-through");
worklet.port.onmessage({data:{type:"enable",value:false}});
const muted=new Float32Array(128);
worklet.process([[inp]],[[muted]]);
assert(muted.every(x=>x===0),"Stop and failure must mute output");
for(const expected of [
 'id="start"','id="stop"','id="live-status"','id="rvc-model"',
 'id="rvc-audio"','id="rvc-vec"','id="rvc-pitch"',
 'source.connect(node);','node.connect(gain);','node.connect(recDest);',
 'api.runPipelineInWorker','Only AudioWorklet-processed audio'
])assert(html.includes(expected),expected+" is required");
assert(!html.includes("source.connect(ctx.destination)"),"No dry microphone route is allowed");
assert(home.includes('href="./live-vc/"'),"Main CTA must open actual live changer");
assert(workflow.includes("cp -a live-vc _site/live-vc"),"Pages must deploy live assets");
console.log("PASS: live worklet muted by default, processes audio, never direct-passes mic, mutes after stop; RVC file lab wired and deploy configured.");
