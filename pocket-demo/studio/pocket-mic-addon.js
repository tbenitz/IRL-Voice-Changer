/* Direct same-document Pocket TTS microphone adapter. Real Pocket ONNX,
   browser SpeechRecognition (may use online STT), no fake conversion. */
(() => {
  "use strict";
  const css = document.createElement("style");
  css.textContent = [
    "#irl-mic-panel{position:fixed;right:14px;bottom:14px;z-index:9999;width:min(405px,calc(100vw - 28px));padding:17px;background:#0d2030;color:#eefbff;border:1px solid #388388;border-radius:16px;box-shadow:0 15px 65px #000b;font:14px/1.5 system-ui,Segoe UI,sans-serif}",
    "#irl-mic-panel *{box-sizing:border-box}#irl-mic-panel h2{font-size:18px;font-weight:750;line-height:1.2;margin:4px 0 9px;color:#fff}",
    "#irl-mic-panel .kicker{font-size:11px;color:#5aded0;letter-spacing:.14em;font-weight:800}",
    "#irl-mic-panel p{color:#b6cbd6;margin:4px 0 12px;font-size:12px}",
    "#irl-mic-panel button{background:#28445a;color:#f4fbff;border:0;border-radius:9px;padding:10px 12px;font:600 13px system-ui;cursor:pointer}",
    "#irl-mic-panel button.primary{background:#5aded0;color:#061922}#irl-mic-panel button:disabled{opacity:.43}",
    "#irl-mic-panel .actions{display:flex;flex-wrap:wrap;gap:7px;margin:9px 0}",
    "#irl-mic-panel textarea{width:100%;min-height:65px;padding:9px;color:#eafbff;background:#0b1826;border:1px solid #456479;border-radius:9px;resize:vertical;font:13px system-ui}",
    "#irl-mic-panel .status{color:#b9dce8;font-size:12px;min-height:30px;margin:9px 0;white-space:pre-wrap;overflow-wrap:anywhere}",
    "#irl-mic-panel #irl-mic-close{float:right;padding:0 7px;font-size:19px;background:transparent}",
    "#irl-mic-toggle{position:fixed;right:14px;bottom:14px;z-index:9998;padding:12px 18px;background:#5aded0;border:0;border-radius:100px;color:#061922;font:750 14px system-ui;cursor:pointer}",
    "@media(max-width:580px){#irl-mic-panel{bottom:5px;right:5px;left:5px;width:auto;max-height:77vh;overflow-y:auto;padding:12px}}"
  ].join("");
  document.head.appendChild(css);

  const panel=document.createElement("section");
  panel.id="irl-mic-panel";
  panel.setAttribute("aria-label","Pocket microphone voice conversion test");
  panel.innerHTML = '<button type="button" id="irl-mic-close" aria-label="Minimize microphone controls">×</button>' +
    '<div class="kicker">REAL POCKET ONNX / MIC EXPERIMENT</div>' +
    '<h2>Speak, then hear the cloned voice</h2>' +
    '<p>Select or clone a Pocket voice below first. We recognize your words, then run genuine Pocket generation. Not zero-delay live conversion.</p>' +
    '<div class="actions"><button type="button" class="primary" id="irl-mic-start">● Speak a phrase</button>' +
    '<button type="button" id="irl-mic-stop" disabled>■ Stop</button><button type="button" id="irl-mic-generate">▶ Generate words</button></div>' +
    '<textarea id="irl-mic-text" aria-label="Spoken words" placeholder="Your transcript appears here. You can also type a test phrase."></textarea>' +
    '<div class="status" id="irl-mic-status" role="status" aria-live="polite">Checking Pocket ONNX engine…</div>' +
    '<p>Browser speech recognition may send audio to its online service. Pocket generation is local. Wear headphones to prevent feedback.</p>';
  document.body.appendChild(panel);
  const toggle=document.createElement("button");
  toggle.type="button"; toggle.id="irl-mic-toggle"; toggle.textContent="🎙 Pocket microphone";
  toggle.hidden=true;
  document.body.appendChild(toggle);
  const $=id=>document.getElementById(id);
  const app=()=>window.__pocketApp;
  const report=s=>{const el=$("irl-mic-status");if(el)el.textContent=String(s)};
  $("irl-mic-close").addEventListener("click",()=>{panel.hidden=true;toggle.hidden=false});
  toggle.addEventListener("click",()=>{panel.hidden=false;toggle.hidden=true});
  const Rec=window.SpeechRecognition||window.webkitSpeechRecognition;
  let recognition=null, active=false, recognized="", consumed=false, pending=false, actuallyGenerated=false;

  function unlock(){
    const p=app();
    if(p && typeof p.ensureAudioContext==="function"){
      p.ensureAudioContext().catch(e=>report("Audio output unavailable: "+e.message));
    }
  }
  function submit(){
    const text=$("irl-mic-text").value.trim();
    if(!text){report("No words available. Speak a phrase or type text.");return}
    const p=app();
    if(!p?.isWorkerReady){report("Pocket ONNX has not loaded yet. Wait for the Studio to say Ready.");return}
    if(p.isGenerating||p.isVoiceEncoding){report("Pocket is still processing. Try when it finishes.");return}
    if(!p.el?.btnGenerate || p.el.btnGenerate.disabled){
      report("Choose an available voice in the Pocket Studio. The main Generate button is disabled.");
      return;
    }
    unlock();
    p.el.textInput.value=text;
    p.el.textInput.dispatchEvent(new Event("input",{bubbles:true}));
    pending=true;actuallyGenerated=false;
    report("Starting actual Pocket ONNX speech generation…");
    p.el.btnGenerate.click();
    setTimeout(()=>{
      if(pending&&!actuallyGenerated&&!p.isGenerating){
        pending=false;
        report("Generation did not begin. Open the Pocket Studio status panel for the model error or press its own Generate button.");
      }
    },4500);
  }

  if(Rec){
    recognition=new Rec();
    recognition.lang="en-US";recognition.continuous=false;recognition.interimResults=true;
    recognition.onresult=e=>{
      let interim="";
      for(let i=0;i<e.results.length;i++){
        if(e.results[i].isFinal){if(!consumed)recognized+=e.results[i][0].transcript+" ";}
        else interim+=e.results[i][0].transcript;
      }
      $("irl-mic-text").value=(recognized+interim).trim();
      if(recognized.trim()&&!consumed){
        consumed=true;active=false;
        $("irl-mic-start").disabled=false;$("irl-mic-stop").disabled=true;
        try{recognition.stop()}catch{}
        report("Phrase recognized. Generating…");
        submit();
      }
    };
    recognition.onerror=e=>{
      active=false;$("irl-mic-start").disabled=false;$("irl-mic-stop").disabled=true;
      report("Speech recognition error: "+e.error+". You can type your words instead.");
    };
    recognition.onend=()=>{active=false;$("irl-mic-start").disabled=false;$("irl-mic-stop").disabled=true};
    report("Wait for Pocket model to load and choose a voice. Then click Speak a phrase.");
  }else{
    $("irl-mic-start").disabled=true;
    report("SpeechRecognition isn't available in this browser. Use typed words instead, or clone/record a voice in the original Studio.");
  }
  $("irl-mic-start").addEventListener("click",()=>{
    if(!recognition||active)return;
    const p=app();
    if(!p?.isWorkerReady){report("Pocket model still loading. Wait until Ready before using microphone.");return}
    if(p.isGenerating){report("Pocket is currently generating. Finish that first.");return}
    recognized="";consumed=false;$("irl-mic-text").value="";
    unlock();
    try{
      recognition.start();active=true;
      $("irl-mic-start").disabled=true;$("irl-mic-stop").disabled=false;
      report("Listening. Speak a phrase and pause; the converted speech will be generated.");
    }catch(e){report("Could not start speech recognition: "+e.message)}
  });
  $("irl-mic-stop").addEventListener("click",()=>{
    if(recognition&&active){try{recognition.stop()}catch{}}
    active=false;$("irl-mic-start").disabled=false;$("irl-mic-stop").disabled=true;
    report("Stopped. You can edit the transcript and click Generate words.");
  });
  $("irl-mic-generate").addEventListener("click",submit);
  const interval=setInterval(()=>{
    const p=app();
    if(!p)return;
    if(pending&&p.isGenerating){actuallyGenerated=true;report("Generating your cloned speech now… Listen in the Studio audio player.");}
    if(pending&&actuallyGenerated&&!p.isGenerating){
      pending=false;actuallyGenerated=false;
      report("Pocket generation complete. Replay or download in the Studio.");
    }
    if(!pending&&!p.isWorkerReady){
      const status=document.getElementById("status-text")?.textContent;
      if(status)report("Pocket ONNX loading: "+status);
    }
  },500);
  window.addEventListener("beforeunload",()=>{
    clearInterval(interval);
    if(recognition&&active)try{recognition.abort()}catch{}
  });
})();
