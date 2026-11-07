/* Headless integration: a saved Pocket custom voice must be selectable as
 * target and produce real internal synthetic pairs without a preview/MP3.
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
const win=dom.window;
const $=id=>win.document.getElementById(id);
const saved=[{id:9,name:"My uploaded custom voice",pcm:new Float32Array(24000*6).fill(.12)}];
const added=[],calls=[];let stopAtFirstPair=false;
const app={
  builtinVoices:["alba","cosette"],isWorkerReady:true,
  isGenerating:false,isVoiceEncoding:false,isLoadingSavedVoice:false,
  activeVoice:"alba",activeSavedVoiceId:null,customVoiceLoaded:false,
  lastClonedPCM:null,currentChunks:[],player:{playAudio(){}},
  el:{voiceSelect:$("voice-select"),langSelect:$("language-select"),textInput:$("text-input")},
  openDB:async()=>({
    transaction(){return {objectStore(){return {getAll(){
      const req={result:null,onsuccess:null,onerror:null};
      win.queueMicrotask(()=>{req.result=saved;req.onsuccess?.()});
      return req;
    }}}}},
    close(){}
  }),
  handleWorkerMessage(m){
    if(m.type==="voice_encoded"){this.customVoiceLoaded=true;this.isVoiceEncoding=false;this.isLoadingSavedVoice=false;this.activeVoice="custom"}
    if(m.type==="voice_set")this.activeVoice=m.voiceName;
    if(m.type==="stream_ended")this.isGenerating=false;
  },
  async generate(){
    if(this.isGenerating)throw Error("Unexpected simultaneous generation");
    calls.push("generate:"+this.el.textInput.value.slice(0,32)+":"+this.activeVoice);
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
const meta=new Map();
win.__pocketApp=app;
win.PocketTrainerData={
 SR:16000,
 async listPairs(){return added},
 async getMeta(k){return meta.has(k)?{value:meta.get(k)}:null},
 async putMeta(k,value){meta.set(k,value)},
 async savePair(pair){added.push(pair); if(!stopAtFirstPair){stopAtFirstPair=true;win.queueMicrotask(()=>$("trainer-stop").click())}},
 resample(pcm){return new Float32Array(pcm.subarray(0,32000))},
 trimSilence(pcm){return pcm},stretch(pcm,length){return Float32Array.from({length},(_,i)=>pcm[Math.min(i,pcm.length-1)]||0)},
 wav(pcm){return new win.Blob([pcm.buffer],{type:"audio/wav"})}
};
win.eval(fs.readFileSync(path.join(__dirname,"../pocket-demo/studio/trainer-addon.js"),"utf8"));
const wait=ms=>new Promise(res=>setTimeout(res,ms));
(async()=>{
 await wait(50);
 const target=$("trainer-target"),source=$("trainer-source");
 assert(target,"Missing target voice picker");
 assert([...target.options].some(o=>o.value==="saved:9"&&o.text.includes("My uploaded custom voice")),
  "Existing Pocket voice must be listed as a valid custom target");
 assert([...source.options].some(o=>o.value==="saved:9"),
  "Saved voices should also be offered as source speakers");
 target.value="saved:9";source.value="cosette";$("trainer-consent").checked=true;
 assert(!app.customVoiceLoaded,"This must test starting without manually previewing/encoding voice");
 $("trainer-generate").click();
 for(let i=0;i<140&&!added.length;i++)await wait(25);
 assert.equal(added.length,1,"Pair generator must save a completed pair automatically");
 assert.equal(added[0].targetVoice,"saved:9");
 assert.equal(added[0].sourceVoice,"cosette");
 assert(added[0].text.length>60,"Generator must automatically choose text, not require typed preview");
 assert(calls.some(c=>c==="encode_voice:"),"Saved custom reference was not encoded internally");
 assert(calls.filter(c=>c.startsWith("generate:")).length===2,
  "Target + source speech should be synthesized automatically");
 assert(calls.some(c=>c==="set_voice:custom")&&calls.some(c=>c==="set_voice:cosette"));
 assert.equal($("trainer-count").textContent.includes("0:02"),true,
  "Saved pair duration should update UI");
 assert(!win.document.querySelector("#download-personal"),
  "This browser repo must not contain unrelated EXE builder controls");
 console.log("PASS: named saved custom target and source options; auto-encodes reference; generates both WAV pairs without MP3/preview");
})().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>win.close());
