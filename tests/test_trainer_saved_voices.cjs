/* Pocket saved/builtin voice dataset integration: silence by construction,
 * target-only default, optional two-voice research pairs, no preview/MP3.
 */
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom");
const dom=new JSDOM(
 '<!doctype html><html><head></head><body><div id="model-setup"></div>'+
 '<select id="voice-select"><option value="custom">Custom</option><option value="alba">Alba</option><option value="cosette">Cosette</option></select>'+
 '<select id="language-select"><option value="english_2026-04" selected>English</option></select>'+
 '<textarea id="text-input"></textarea></body></html>',
 {runScripts:"outside-only",url:"https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/studio/"}
);
const win=dom.window,$=id=>win.document.getElementById(id);
win.confirm=()=>true;
const voices=[{id:9,name:"My saved custom voice",pcm:new Float32Array(24000*6).fill(.12)}];
const added=[],calls=[],meta=new Map();let playCount=0;
const app={
 builtinVoices:["alba","cosette"],isWorkerReady:true,isGenerating:false,
 isVoiceEncoding:false,isLoadingSavedVoice:false,activeVoice:"alba",
 activeSavedVoiceId:null,customVoiceLoaded:false,lastClonedPCM:null,
 currentChunks:[],silentBatchMode:false,
 player:{reset(){calls.push("player-reset")},playAudio(){playCount++}},
 el:{voiceSelect:$("voice-select"),langSelect:$("language-select"),textInput:$("text-input")},
 openDB:async()=>({
  transaction(){return {objectStore(){return {getAll(){
   const req={result:null,onsuccess:null,onerror:null};
   win.queueMicrotask(()=>{req.result=voices;req.onsuccess?.()});
   return req
  }}}}},
  close(){}
 }),
 handleWorkerMessage(m){
  if(m.type==="voice_encoded"){this.customVoiceLoaded=true;this.isVoiceEncoding=false;this.isLoadingSavedVoice=false;this.activeVoice="custom"}
  if(m.type==="voice_set")this.activeVoice=m.voiceName;
  if(m.type==="stream_ended")this.isGenerating=false;
 },
 async generate(){
  assert.equal(this.silentBatchMode,true,"Silent batch mode must be active before Pocket synthesis");
  assert(!this.isGenerating);
  calls.push("generate:"+this.activeVoice);
  this.isGenerating=true;
  this.currentChunks=[new Float32Array(48000).fill(.14)];
  win.queueMicrotask(()=>this.handleWorkerMessage({type:"stream_ended"}));
 },
 stop(){this.isGenerating=false}
};
app.worker={postMessage(m){
 calls.push(m.type+":"+(m.data?.voiceName||""));
 if(m.type==="encode_voice")win.queueMicrotask(()=>app.handleWorkerMessage({type:"voice_encoded"}));
 if(m.type==="set_voice")win.queueMicrotask(()=>app.handleWorkerMessage({type:"voice_set",voiceName:m.data.voiceName}));
}};
win.__pocketApp=app;
win.PocketTrainerData={
 SR:16000,
 async listPairs(){return added},
 async getMeta(k){return meta.has(k)?{value:meta.get(k)}:null},
 async putMeta(k,value){meta.set(k,value)},
 async savePair(pair){
  added.push(pair);
  win.queueMicrotask(()=>$("trainer-stop").click()); // Stop after first full recording.
 },
 async clearPairs(){added.length=0;meta.clear()},
 resample(pcm){return new Float32Array(pcm.subarray(0,32000))},
 trimSilence(pcm){return pcm},
 stretch(pcm,length){return Float32Array.from({length},(_,i)=>pcm[Math.min(i,pcm.length-1)]||0)},
 wav(pcm){return new win.Blob([pcm.buffer],{type:"audio/wav"})}
};
const src=fs.readFileSync(path.join(__dirname,"../pocket-demo/studio/trainer-addon.js"),"utf8");
win.eval(src);
const wait=ms=>new Promise(res=>setTimeout(res,ms));
async function one(){
 const prior=added.length;
 $("trainer-generate").click();
 for(let i=0;i<160&&added.length===prior;i++)await wait(20);
 assert.equal(added.length,prior+1,"One recording must be saved automatically");
 for(let i=0;i<50&&$("trainer-stop").disabled===false;i++)await wait(20);
 return added.at(-1);
}
async function reset(){
 $("trainer-reset").click();
 for(let i=0;i<80&&added.length;i++)await wait(20);
 await wait(30);assert.equal(added.length,0);
}
(async()=>{
 await wait(50);
 assert([...$("trainer-target").options].some(o=>o.value==="saved:9"&&o.text.includes("My saved custom voice")));
 assert([...$("trainer-target").options].some(o=>o.value==="builtin:cosette"),
  "Builtin voice should work as target with NO upload");
 assert([...$("trainer-source").options].some(o=>o.value==="saved:9"));
 assert.equal($("trainer-mode").value,"target-only","Target-only must be default; no forced second voice");
 assert($("trainer-source-label").hidden,"Source must be hidden in one-voice mode");
 $("trainer-consent").checked=true;
 $("trainer-target").value="saved:9";
 assert.equal(app.customVoiceLoaded,false,"No manual voice preview is required");
 let record=await one();
 assert.equal(record.targetVoice,"saved:9");
 assert.equal(record.source,null,"Default must NOT synthesize second voice");
 assert.equal(calls.filter(c=>c.startsWith("generate:")).length,1);
 assert(calls.includes("encode_voice:"),"Saved target should be encoded internally");
 assert.equal(app.silentBatchMode,false,"Silent flag must be restored after generation");
 assert.equal(playCount,0,"Dataset must not call the speaker");
 assert(record.text.length>60);
 await reset();
 calls.length=0;
 $("trainer-target").value="builtin:cosette";
 record=await one();
 assert.equal(record.targetVoice,"builtin:cosette");
 assert.equal(record.source,null);
 assert.equal(calls.filter(c=>c.startsWith("generate:")).length,1);
 assert.equal(calls.filter(c=>c==="encode_voice:").length,0,
  "Built-in voice must not require/upload a custom reference sample");
 await reset();
 calls.length=0;
 $("trainer-mode").value="paired-experimental";
 $("trainer-mode").dispatchEvent(new win.Event("change"));
 assert(!$("trainer-source-label").hidden);
 $("trainer-target").value="saved:9";$("trainer-source").value="cosette";
 record=await one();
 assert.equal(record.targetVoice,"saved:9");
 assert.equal(record.sourceVoice,"cosette");
 assert(record.source instanceof win.Blob,"Optional experiment must include source WAV");
 assert.equal(calls.filter(c=>c.startsWith("generate:")).length,2,
  "Two voice synthesis must happen only in explicit experimental mode");
 assert.equal(playCount,0,"Neither voice must play during experimental pairs");
 assert(!win.document.getElementById("download-personal"),"Windows EXE must be absent");
 const appSource=fs.readFileSync(path.join(__dirname,"../pocket-demo/studio/app.js"),"utf8");
 assert(appSource.includes("this.silentBatchMode")&&
  appSource.includes("if (!this.silentBatchMode) this.player.playAudio(pcm)")&&
  appSource.includes("if (this.silentBatchMode) {"),
 "Actual Pocket controller must enforce silence, not a too-early monkey-patch");
 console.log("PASS: saved and built-in voice target-only datasets are silent with NO preview; two voices only when explicitly selected; real Pocket audio gate present");
})().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>win.close());
