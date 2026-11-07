/* Local Pocket synthetic dataset + browser train/export.
 * Real Pocket worker PCM, IndexedDB, real TensorFlow gradients and ONNX
 * validation. Not a proven voice identity distillation procedure.
 */
(() => {
"use strict";
const app=window.__pocketApp,D=window.PocketTrainerData;
if(!app||!D){console.error("Pocket Trainer: app/storage missing");return}
const anchor=document.getElementById("model-setup");
if(!anchor)throw Error("Missing Pocket model panel");
const css=document.createElement("style");
css.textContent=[
 "#pocket-trainer{margin:25px 0;padding:23px;border:1px solid #357c81;border-radius:16px;background:linear-gradient(140deg,#102737,#121b2b);color:#eefbff;font:14px/1.55 system-ui,sans-serif}",
 "#pocket-trainer *{box-sizing:border-box}#pocket-trainer h3{font-size:clamp(23px,3vw,30px);margin:5px 0 8px}",
 "#pocket-trainer p{color:#b9d1d8;margin:5px 0 14px}#pocket-trainer .ey{font-size:11px;letter-spacing:.14em;color:#64e3d6;font-weight:900;text-transform:uppercase}",
 "#pocket-trainer .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:17px 0}",
 "#pocket-trainer label{font-size:12px;font-weight:800;display:grid;gap:5px}",
 "#pocket-trainer select{font:14px system-ui;border:1px solid #4a7280;padding:11px;color:#fff;background:#0a202d;border-radius:9px;min-width:0;width:100%}",
 "#pocket-trainer .actions{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}",
 "#pocket-trainer button{font:800 13px system-ui;border:1px solid #397982;background:#1c4552;border-radius:10px;color:#f8ffff;padding:12px 15px;cursor:pointer}",
 "#pocket-trainer button.primary{background:#65e6d6;color:#09252c;border:0}#pocket-trainer button:disabled{opacity:.43;cursor:not-allowed}",
 "#pocket-trainer progress{width:100%;accent-color:#68e6d5;height:12px;margin:10px 0}",
 "#pocket-trainer .count{font-variant-numeric:tabular-nums;font-size:23px;font-weight:850}",
 "#pocket-trainer .notice{border:1px solid #79663e;background:#322c20;color:#edd8ae;padding:12px;border-radius:10px;margin:14px 0;font-size:12px}",
 "#pocket-trainer .status{min-height:45px;padding:12px;background:#091c2a;border:1px solid #335567;border-radius:10px;color:#d1e3ec;white-space:pre-wrap;overflow-wrap:anywhere;margin:10px 0;font-size:12px}",
 "#pocket-trainer .status.bad{border-color:#ad5c67;color:#ffc9cc}",
 "#pocket-trainer .sub{border-top:1px solid #365363;padding-top:18px;margin-top:19px}",
 "#pocket-trainer .check{display:flex;align-items:start;gap:10px;margin:13px 0;color:#c5e0e8;font-size:12px}",
 "#pocket-trainer input[type=checkbox]{width:18px;height:18px;flex:none;accent-color:#65e6d6}#pocket-trainer small{font-size:11px;color:#a5c4cf}",
 "@media(max-width:820px){#pocket-trainer .grid{grid-template-columns:1fr 1fr}}",
 "@media(max-width:540px){#pocket-trainer{padding:15px}#pocket-trainer .grid{grid-template-columns:1fr}#pocket-trainer button{flex:1}}"
].join("");
document.head.append(css);
const panel=document.createElement("section");
panel.id="pocket-trainer";
panel.innerHTML=[
 '<span class="ey">Pocket TTS • Synthetic voice recordings • On-device research</span>',
 '<h3>Custom Voice Training Lab</h3>',
 '<p><b>Select a target custom voice below</b> from My Voices, or record/upload a new one below. Then click Generate: the trainer itself synthesizes the paired recordings. <b>You do not need to generate a preview, WAV or MP3 first.</b> Output audio is stored automatically.</p>',
 '<div class="notice"><b>Important:</b> 30 minutes of Pocket-generated speech is not 30 minutes of real or time-aligned RVC teacher conversions. Local WebGPU training here is a <b>genuine but experimental waveform adapter</b>. Its ONNX export is real, but voice identity and naturalness are unverified and may be poor.</div>',
 '<div class="grid">',
 '<label>Target synthetic audio length<select id="trainer-goal"><option value="60">1 minute • test</option><option value="300">5 minutes</option><option value="600">10 minutes</option><option value="1800" selected>30 minutes</option></select></label>',
 '<label>Target voice — who should the model sound like?<select id="trainer-target"><option value="current">Current uploaded/recorded Pocket clone</option></select></label>',
  '<label>Source voice — the input speaker for pairs<select id="trainer-source"><option value="">Load Pocket to choose voices</option></select></label>',
 '<label>Local training updates<select id="trainer-steps"><option value="25">25 • smoke test</option><option value="120" selected>120 • quick</option><option value="500">500 • extended</option><option value="1500">1,500 • experimental</option></select></label>',
 '</div>',
 '<span class="count" id="trainer-count">0:00 / 30:00</span> <small id="trainer-pairs">0 saved speech pairs</small>',
 '<progress id="trainer-progress" max="100" value="0"></progress>',
 '<label class="check"><input type="checkbox" id="trainer-consent"><span>I am using my own voice or a voice I have explicit permission to clone and train.</span></label>',
 '<div class="actions"><button id="trainer-generate" class="primary">Generate 30 minutes with Pocket</button>',
 '<button id="trainer-stop" disabled>Stop; keep saved pairs</button>',
  '<button id="trainer-refresh-voices" type="button">↻ Refresh My Voices</button>',
 '<button id="trainer-export">Download paired WAV dataset ZIP</button>',
 '<button id="trainer-reset">Start a new voice project</button></div>',
 '<div id="trainer-status" class="status" role="status" aria-live="polite">Choose a saved target voice (or the current recording), load the Pocket model, then generate pairs directly. No MP3 or manual preview is needed.</div>',
 '<div class="sub"><span class="ey">WebGPU neural student • Automatic ONNX</span>',
 '<h3 style="font-size:22px">Train locally and export a model</h3>',
 '<p>For lower memory use on an 8 GB Chromebook, you can refresh this page after generating the dataset and train without reloading the 200 MB Pocket model. Saved pairs remain in browser storage.</p><p>After saving four or more paired sentences, run real WebGPU gradient updates on a small causal Conv1D network. The exported ONNX file has <code>audio_in → audio_out</code> and is checked with ONNX Runtime Web before being saved. This is not yet an LLVC-quality or verified speaker conversion pipeline.</p>',
 '<label class="check"><input type="checkbox" id="trainer-cpu" checked><span>Permit slower CPU training if WebGPU is unavailable or unsupported.</span></label>',
 '<div class="actions"><button id="trainer-train" class="primary">Train locally and auto-export ONNX</button>',
 '<button id="trainer-stop-training" disabled>Stop training after current step</button>',
 '<button id="trainer-onnx" disabled>Download trained ONNX</button>',
 '<button id="trainer-open-fast">Open Fast Neural AI</button></div>',
 '<div id="trainer-training-status" class="status" role="status" aria-live="polite">No trained model yet; complete a dataset first.</div></div>'
].join("");
anchor.insertAdjacentElement("afterend",panel);
const $=id=>document.getElementById(id);
const notice=(msg,bad=false)=>{
 const el=$("trainer-status");el.textContent=String(msg);el.classList.toggle("bad",bad)
};
const trainNotice=(msg,bad=false)=>{
 const el=$("trainer-training-status");el.textContent=String(msg);el.classList.toggle("bad",bad)
};
const S={running:false,training:false,cancel:false,cancelTraining:false,
 cursor:0,seconds:0,pairs:0,hash:null,goal:1800};
let pending=null;
const oldHandle=app.handleWorkerMessage.bind(app);
app.handleWorkerMessage=msg=>{
 const ret=oldHandle(msg);
 if(pending){
  if(msg.type==="error"){
   const w=pending;pending=null;clearTimeout(w.timer);
   w.reject(Error(msg.error||"Pocket generation error"));
  }else if(msg.type===pending.type && (!pending.match||pending.match(msg))){
   const w=pending;pending=null;clearTimeout(w.timer);w.resolve(msg);
  }
 }
 return ret;
};
function waitFor(type,trigger,match=null,timeoutMs=360000){
 if(pending)throw Error("A Pocket generation command is already pending");
 return new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>{
   if(pending?.resolve===resolve)pending=null;
   reject(Error("Pocket "+type+" response timed out"));
  },timeoutMs);
  pending={type,resolve,reject,match,timer};
  try{
   const command=trigger();
   // app.generate() does asynchronous AudioContext/worklet setup before
   // posting to the worker. Propagate failures immediately, not in six minutes.
   if(command && typeof command.catch==="function")command.catch(e=>{
    if(pending?.resolve===resolve){pending=null;clearTimeout(timer);reject(e)}
   });
  }catch(e){pending=null;clearTimeout(timer);reject(e)}
 });
}
function rejectPending(){
 if(!pending)return;
 const p=pending;pending=null;clearTimeout(p.timer);p.reject(Error("Stopped by user"))
}
function clock(v){const n=Math.round(v);return Math.floor(n/60)+":"+String(n%60).padStart(2,"0")}
function render(){
 const busy=S.running||S.training;
 $("trainer-generate").disabled=busy;$("trainer-train").disabled=busy;
 $("trainer-stop").disabled=!S.running;
 $("trainer-stop-training").disabled=!S.training;
 $("trainer-export").disabled=busy||!S.pairs;
 $("trainer-reset").disabled=busy;
 $("trainer-goal").disabled=busy;$("trainer-source").disabled=busy;
 $("trainer-target").disabled=busy;$("trainer-refresh-voices").disabled=busy;
 $("trainer-count").textContent=clock(S.seconds)+" / "+clock(S.goal);
 $("trainer-pairs").textContent=S.pairs+" saved pairs";
 $("trainer-progress").value=Math.min(100,100*S.seconds/Math.max(1,S.goal));
 $("trainer-generate").textContent="Generate / resume "+Math.round(S.goal/60)+" min Pocket recordings";
}
const subjects=[
"The quickest way to learn something is to practice slowly and carefully.",
"Yesterday the weather changed without warning, yet the morning remained peaceful.",
"A small blue bird landed near the window and watched the people walking by.",
"Can you tell me which road leads to the old train station after midnight?",
"Every new project begins with questions, experiments, mistakes and improvements.",
"Please remember to leave the lights on near the entrance when you arrive.",
"The river was calm during the afternoon, but the wind grew stronger after sunset.",
"Some people prefer quiet music while others enjoy a lively conversation.",
"We counted seven orange boxes, three broken chairs and twenty silver coins.",
"It surprised everyone when the old clock finally started ticking again.",
"Bright sunlight moved over the garden as the children began their games.",
"I wanted to explain the entire situation before someone drew a conclusion.",
"There was no simple reason for what happened, only a series of choices.",
"She spoke with confidence about the places she hoped to travel next year.",
"They discovered a hidden note beneath the table in the crowded restaurant.",
"Don't forget to charge the batteries before you take the camera outside.",
"His friend suggested an entirely different approach to solving the problem.",
"I could hear people laughing beyond the hallway before I entered the room.",
"Every chapter contains new characters, unexpected turns and moments of suspense.",
"The soup smelled wonderful, and the kitchen was filled with warm afternoon light.",
"I asked three questions, paused, and listened carefully to the answers.",
"The package contained a letter, a photograph and a mysterious wooden key.",
"It took several attempts before the microphone captured the voice clearly.",
"At the center of the square, musicians were performing for a small crowd.",
"We should compare the two results before deciding which model works better.",
"When the doors opened, everyone moved forward in an orderly fashion.",
"Please speak naturally, varying your rhythm, volume, emphasis and intonation.",
"The number was forty-seven, not seventy-four, and the difference mattered.",
"How wonderful it would be to spend another evening beside the quiet lake.",
"They were certain the unusual noise came from somewhere above the ceiling.",
"The neighborhood looked different after the streets were covered in snow.",
"I could not decide between the dark chocolate, the apples and fresh peaches.",
"What happens if we change the settings but leave every other step the same?",
"The distant thunder reminded us that the storm would arrive before dinner.",
"Someone called from the other room and asked whether the guests were ready.",
"The young musician practiced daily until the difficult melody felt natural.",
"We finally agreed to meet at the theater around six thirty on Thursday.",
"Many voices sound similar at first, but their speech patterns are distinct.",
"The instructions were straightforward, yet the smallest details were important.",
"Our latest experiment produced an interesting result worth investigating.",
"This morning I decided to write a letter to a friend across the country.",
"Five minutes later, the conversation had shifted to entirely different subjects.",
"By the end of the week, the team had discovered a faster and simpler solution.",
"The flowers were yellow and violet, and their leaves sparkled with rain.",
"The friendly dog ran across the park, stopped suddenly, and turned around.",
"I will explain what happened in the beginning, the middle, and at the end.",
"Have you ever heard a voice that sounded almost exactly like someone you know?",
"Those tall buildings stood alongside narrow streets and busy neighborhood shops."
];
const endings=[
"The explanation was clear enough for everyone to understand.",
"Unexpected changes can make an ordinary day feel memorable.",
"There is still time to choose another path if we need one.",
"We should check our results and repeat the experiment tomorrow.",
"That is exactly why patience and preparation are so valuable.",
"I would like to hear your thoughts before making a decision.",
"This recording includes a variety of speech sounds and rhythms.",
"Please take your time and pronounce the words comfortably."
];
function words(i){
 return subjects[i%subjects.length]+" "+
 subjects[(i*13+17)%subjects.length]+" "+
 endings[(Math.floor(i/subjects.length)+i)%endings.length];
}
function voiceFingerprint(pcm){
 let h=2166136261,step=Math.max(1,Math.floor(pcm.length/4000));
 for(let i=0;i<pcm.length;i+=step)
  h=Math.imul(h^(Math.round(pcm[i]*32767)&65535),16777619)>>>0;
 return h.toString(16)+"-"+pcm.length;
}
function combine(){
 const n=app.currentChunks.reduce((s,c)=>s+c.length,0);
 if(n<24000||n>24000*110)throw Error("Pocket returned an invalid/empty recording");
 const out=new Float32Array(n);let off=0;
 for(const c of app.currentChunks){out.set(c,off);off+=c.length}
 return out;
}
let savedVoices=[],targetPCM=null,loadedCustomKey=null;
function selectedSaved(key){
 const id=key.slice("saved:".length);
 return savedVoices.find(v=>String(v.id)===id);
}
async function refreshSavedVoices(){
 let db;
 try{
  db=await app.openDB();
  const all=await new Promise((resolve,reject)=>{
   const request=db.transaction("voices","readonly").objectStore("voices").getAll();
   request.onsuccess=()=>resolve(request.result||[]);
   request.onerror=()=>reject(request.error||Error("Cannot read saved Pocket voices"));
  });
  savedVoices=all;
  refreshVoices();
  return all;
 }finally{db?.close()}
}
async function encodeClonedPCM(pcm,key){
 if(app.isGenerating||app.isVoiceEncoding)throw Error("Pocket is already busy");
 // Set the Pocket app to its existing saved-voice path. Do not force a
 // separate TTS preview and do not pop up the save modal.
 const copy=new Float32Array(pcm);
 app.lastClonedPCM=new Float32Array(pcm);
 app.isVoiceEncoding=true;
 app.isLoadingSavedVoice=true;
 try{
  await waitFor("voice_encoded",()=>{
   app.worker.postMessage({type:"encode_voice",data:{audio:copy}},[copy.buffer]);
  },null,180000);
  loadedCustomKey=key;
 }catch(e){
  app.isVoiceEncoding=false;app.isLoadingSavedVoice=false;
  throw e;
 }
}
async function prepareTarget(){
 const choice=$("trainer-target").value;
 if(choice.startsWith("saved:")){
  const record=selectedSaved(choice);
  if(!record||!record.pcm)throw Error("Saved target voice was not found. Click Refresh My Voices.");
  targetPCM=new Float32Array(record.pcm);
  await encodeClonedPCM(targetPCM,choice);
 }else{
  if(!app.customVoiceLoaded||!app.lastClonedPCM||app.lastClonedPCM.length<24000*2)
   throw Error("Select a saved Pocket voice in Target, or record/upload your voice and wait for encoding. No preview generation is required.");
  targetPCM=new Float32Array(app.lastClonedPCM);
  loadedCustomKey="current";
  if(app.activeVoice!=="custom")await waitFor("voice_set",()=>{
   app.worker.postMessage({type:"set_voice",data:{voiceName:"custom"}});
  },m=>m.voiceName==="custom");
 }
 if(targetPCM.length<24000*2)throw Error("Target reference recording needs at least 2 seconds of audio.");
 return voiceFingerprint(targetPCM);
}
async function selectSpeechVoice(voice){
 if(voice==="target"){
  const key=$("trainer-target").value;
  if(loadedCustomKey!==key){
   if(!targetPCM)throw Error("Target voice was not prepared");
   await encodeClonedPCM(targetPCM,key);
  }
  await waitFor("voice_set",()=>app.worker.postMessage({
   type:"set_voice",data:{voiceName:"custom"}
  }),m=>m.voiceName==="custom");
  app.el.voiceSelect.value="custom";app.activeVoice="custom";
 }else if(voice.startsWith("saved:")){
  const record=selectedSaved(voice);
  if(!record||!record.pcm)throw Error("Saved source voice is no longer available");
  if(loadedCustomKey!==voice)await encodeClonedPCM(record.pcm,voice);
  await waitFor("voice_set",()=>app.worker.postMessage({
   type:"set_voice",data:{voiceName:"custom"}
  }),m=>m.voiceName==="custom");
  app.el.voiceSelect.value="custom";app.activeVoice="custom";
 }else{
  await waitFor("voice_set",()=>app.worker.postMessage({
   type:"set_voice",data:{voiceName:voice}
  }),m=>m.voiceName===voice);
  app.el.voiceSelect.value=voice;app.activeVoice=voice;
 }
}
async function synth(text,voice){
 if(S.cancel)throw Error("Stopped by user");
 if(app.isGenerating||app.isVoiceEncoding)throw Error("Pocket is already generating speech");
 await selectSpeechVoice(voice);
 app.el.textInput.value=text;
 app.el.textInput.dispatchEvent(new Event("input",{bubbles:true}));
 await waitFor("stream_ended",()=>app.generate());
 return combine();
}
async function refresh(){
 const pairs=await D.listPairs(),st=await D.getMeta("progress");
 S.pairs=pairs.length;S.seconds=pairs.reduce((s,p)=>s+(p.seconds||0),0);
 S.cursor=st?.value?.cursor??pairs.length;S.hash=st?.value?.hash??null;
 S.goal=Number($("trainer-goal").value);
 const saved=await D.getMeta("student-onnx");
 $("trainer-onnx").disabled=!saved?.value?.blob;
 if(saved?.value?.blob)trainNotice(saved.value.validation?.nearIdentity
  ? "Saved ONNX ("+(saved.value.blob.size/1024).toFixed(0)+
    " KB) is a valid model file, but it produced almost unchanged audio in the training test. It is NOT a working voice clone. You may still download it for inspection."
  : "Saved experimental ONNX ("+(saved.value.blob.size/1024).toFixed(0)+
    " KB). Model export was validated; voice identity has NOT been verified.");
 render();
}
$("trainer-goal").addEventListener("change",()=>{S.goal=Number($("trainer-goal").value);render()});
function refreshVoices(){
 const list=app.builtinVoices||[];
 const source=$("trainer-source"),target=$("trainer-target");
 const prevSource=source.value,prevTarget=target.value;
 source.replaceChildren();target.replaceChildren();
 const targetCurrent=document.createElement("option");
 targetCurrent.value="current";
 targetCurrent.textContent="Current uploaded/recorded Pocket clone (no saved name)";
 target.append(targetCurrent);
 const available=document.createElement("optgroup");available.label="Saved custom voices (your My Voices library)";
 for(const record of savedVoices){
  const key="saved:"+record.id;
  const targetOption=document.createElement("option");targetOption.value=key;
  targetOption.textContent=record.name+" • saved custom voice";
  available.append(targetOption);
 }
 target.append(available);
 const sourceBuiltins=document.createElement("optgroup");
 sourceBuiltins.label="Built-in Pocket input speakers";
 for(const id of list){
  const option=document.createElement("option");option.value=id;
  option.textContent=id+" • built-in source";
  sourceBuiltins.append(option);
 }
 source.append(sourceBuiltins);
 const sourceSaved=document.createElement("optgroup");
 sourceSaved.label="Saved custom input speakers (slower: re-encodes each pair)";
 for(const record of savedVoices){
  const option=document.createElement("option");option.value="saved:"+record.id;
  option.textContent=record.name+" • saved source voice";
  sourceSaved.append(option);
 }
 source.append(sourceSaved);
 const targetKeys=["current",...savedVoices.map(v=>"saved:"+v.id)];
 target.value=targetKeys.includes(prevTarget)?prevTarget:
  (app.activeSavedVoiceId!=null&&targetKeys.includes("saved:"+app.activeSavedVoiceId)
    ?"saved:"+app.activeSavedVoiceId:targetKeys[0]);
 const sourceKeys=[...list,...savedVoices.map(v=>"saved:"+v.id)];
 source.value=sourceKeys.includes(prevSource)?prevSource:(list.includes("alba")?"alba":sourceKeys[0]||"");
}
let voiceCount=0;
setInterval(()=>{
 if(voiceCount!==app.builtinVoices.length){
  voiceCount=app.builtinVoices.length;refreshVoices();
 }
},1600);
$("trainer-refresh-voices").addEventListener("click",()=>{
 refreshSavedVoices().then(all=>notice("Found "+all.length+
  " saved custom voices. Select one in Target, then generate pairs directly; no MP3 needed."))
 .catch(e=>notice("Could not read My Voices: "+e.message,true));
});
refreshVoices();
refreshSavedVoices().catch(e=>notice("Could not load your saved Pocket voice list: "+e.message,true));
async function generateDataset(){
 if(S.running||S.training)return;
 if(!$("trainer-consent").checked){notice("Confirm that you own or have permission to clone the voice.",true);return}
 if(!app.isWorkerReady){notice("Load the Pocket ONNX model in the panel above first.",true);return}
 if(app.el.langSelect.value!=="english_2026-04"){notice("Please switch to the English Pocket model for this synthetic dataset.",true);return}
 const source=$("trainer-source").value;
 if(!app.builtinVoices.includes(source)&&!(source.startsWith("saved:")&&selectedSaved(source))){
  notice("Select an available built-in or saved custom source voice.",true);return;
 }
 if(app.isGenerating||app.isVoiceEncoding){notice("Finish any current Pocket generation before making pairs.",true);return}
 if($("trainer-target").value===source){
  notice("Choose different source and target voices. Identical voices don't create conversion pairs.",true);return;
 }
 S.running=true;S.cancel=false;S.goal=Number($("trainer-goal").value);
 render();
 const originalPlay=app.player?.playAudio;
 if(app.player)app.player.playAudio=()=>{}; // collect actual Pocket PCM without manual playback
 try{
  notice("Preparing selected saved target voice in Pocket encoder. No preview or MP3 required…");
  const hash=await prepareTarget();
  const old=await D.getMeta("progress");
  if(S.hash&&S.hash!==hash)
   throw Error("This project already contains another target voice. Export its ZIP and reset before switching.");
  if(S.pairs&&old?.value?.source&&old.value.source!==source)
   throw Error("This dataset already uses another source voice. Export and reset before switching.");
  S.hash=hash;
  await D.putMeta("progress",{cursor:S.cursor,hash,source,target:$("trainer-target").value});

  while(!S.cancel&&S.seconds<S.goal){
   const index=S.cursor,text=words(index);
   notice("Pair "+(index+1)+": generating the cloned Pocket target speech… "+
    clock(S.seconds)+" / "+clock(S.goal)+" saved.\n"+text);
   let target=await synth(text,"target");
   notice("Pair "+(index+1)+": generating the original Pocket comparison voice "+source+"…");
   let original=await synth(text,source);
   target=D.trimSilence(D.resample(target,24000));
   original=D.trimSilence(D.resample(original,24000));
   if(target.length<D.SR||original.length<D.SR)throw Error("Pocket generated very little usable speech.");
   if(target.length>D.SR*70||original.length>D.SR*70)throw Error("Unexpectedly long Pocket result");
   const aligned=D.stretch(original,target.length);
   const pair={id:"pair-"+String(index).padStart(5,"0"),index,text,
    seconds:target.length/D.SR,source:D.wav(aligned),target:D.wav(target),
    sourceVoice:source,targetVoice:$("trainer-target").value,reference:hash,
    alignment:"Coarse duration stretch only, not phonetic/frame-aligned RVC conversion",
    createdAt:new Date().toISOString()};
   await D.savePair(pair);
   S.cursor++;S.pairs++;S.seconds+=pair.seconds;
   await D.putMeta("progress",{cursor:S.cursor,hash,source,target:$("trainer-target").value});
   render();
   notice("Saved "+S.pairs+" Pocket speech pairs ("+clock(S.seconds)+
    " / "+clock(S.goal)+"). Progress is saved after each completed pair.");
   await new Promise(ok=>setTimeout(ok,10));
  }
  if(!S.cancel)notice("Synthetic speech goal reached: "+clock(S.seconds)+
   ". Download your WAV ZIP or try the experimental student trainer.");
 }catch(e){notice("Paused after "+clock(S.seconds)+" saved. "+e.message,true)}
 finally{
  if(app.player&&originalPlay)app.player.playAudio=originalPlay;
  S.running=false;S.cancel=false;targetPCM=null;loadedCustomKey=null;render();
 }
}
$("trainer-generate").addEventListener("click",()=>void generateDataset());
$("trainer-stop").addEventListener("click",()=>{
 if(!S.running)return;
 S.cancel=true;try{app.stop()}catch(e){}
 rejectPending();
 notice("Stopping the current speech generation; completed pairs remain saved.");
});
function download(blob,name){
 const url=URL.createObjectURL(blob),a=document.createElement("a");
 a.href=url;a.download=name;document.body.append(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),60000);
}
$("trainer-export").addEventListener("click",async()=>{
 if(S.running||S.training)return;
 $("trainer-export").disabled=true;
 try{
  const pairs=await D.listPairs();
  if(!pairs.length)throw Error("No generated WAV files yet");
  const manifest={
   kind:"Pocket synthetic voice dataset, EXPERIMENTAL and NOT phoneme aligned",
   sampleRate:16000,channels:1,totalTargetSeconds:S.seconds,
   issues:["Two independent Pocket outputs of identical text need not have aligned audio.",
    "A compact model trained on these pairs may not reproduce the desired voice.",
    "Clone only speakers who have consented."],
   pairs:pairs.map(p=>({id:p.id,text:p.text,seconds:p.seconds,
    original:"source/"+p.id+".wav",cloned:"target/"+p.id+".wav"}))
  };
  const entries=[{name:"manifest.json",blob:new Blob([JSON.stringify(manifest,null,2)],{type:"application/json"})}];
  for(const p of pairs){
   entries.push({name:"source/"+p.id+".wav",blob:p.source});
   entries.push({name:"target/"+p.id+".wav",blob:p.target});
  }
  notice("Building ZIP from "+pairs.length+" saved, real Pocket speech pairs…");
  const archive=await D.buildZip(entries,(i,n)=>{
   if(i%20===0||i===n)notice("Packing dataset WAV files "+i+"/"+n);
  });
  download(archive,"pocket-synthetic-paired-"+Math.round(S.seconds)+"s.zip");
  notice("ZIP exported ("+(archive.size/1e6).toFixed(1)+" MB), "+
   pairs.length+" speech pairs, "+clock(S.seconds)+" target voice.");
 }catch(e){notice("ZIP export failed: "+e.message,true)}
 finally{render()}
});
$("trainer-reset").addEventListener("click",async()=>{
 if(S.running||S.training)return;
 if(!confirm("Delete this project's saved synthetic speech pairs and experimental ONNX? Your saved Pocket voice references are kept."))return;
 try{
  await D.clearPairs();Object.assign(S,{cursor:0,seconds:0,pairs:0,hash:null});
  await refresh();notice("New voice project created; previously saved Pocket voices were not changed.");
 }catch(e){notice("Project reset failed: "+e.message,true)}
});
$("trainer-train").addEventListener("click",async()=>{
 if(S.running||S.training)return;
 if(!$("trainer-consent").checked){trainNotice("Confirm consent before training.",true);return}
 S.training=true;S.cancelTraining=false;render();
 try{
  if(!window.PocketTrainerWebGPU)throw Error("Local training engine did not load");
  trainNotice("Starting real local gradient training on synthetic pairs (quality unverified)…");
  const out=await window.PocketTrainerWebGPU.runTraining({
   steps:Number($("trainer-steps").value),allowCPU:$("trainer-cpu").checked,
   cancelled:()=>S.cancelTraining,
   onProgress:t=>trainNotice(t),
   onEpoch:({completed,steps,loss,backend})=>
    trainNotice("Training "+completed+"/"+steps+" actual gradient updates on "+
      backend.toUpperCase()+" • loss "+loss.toFixed(6)+
      "\nResult is a limited experimental causal voice adapter.")
  });
  $("trainer-onnx").disabled=false;
  if(out.validation?.nearIdentity)
   trainNotice("ONNX export saved, but the student FAILED voice-change validation: nearly identical microphone output after "+
    out.steps+" gradient updates. It is NOT a trained speaker voice you can use live. You can download the ONNX for research, but try the 13.7 MB pre-trained voice for genuine conversion.",true);
  else trainNotice("Exported and validated "+(out.blob.size/1024).toFixed(0)+
   " KB ONNX after "+out.steps+" updates. The output changes, but speaker identity and sound quality remain unverified.");
 }catch(e){trainNotice("Training/export failed: "+e.message,true)}
 finally{S.training=false;S.cancelTraining=false;render()}
});
$("trainer-stop-training").addEventListener("click",()=>{
 S.cancelTraining=true;trainNotice("Stopping after the current gradient update, then validating and saving partial weights.");
});
$("trainer-onnx").addEventListener("click",async()=>{
 try{
  const model=await D.getMeta("student-onnx");
  if(!model?.value?.blob)throw Error("No validated ONNX model saved");
  download(model.value.blob,"pocket-experimental-student-"+model.value.steps+"steps.onnx");
 }catch(e){trainNotice("Download failed: "+e.message,true)}
});
$("trainer-open-fast").addEventListener("click",()=>{
 window.open("../../live-vc/fast-voice.html","_blank","noopener,noreferrer")
});
refresh().catch(e=>notice("Trainer storage is unavailable: "+e.message,true));
})();
