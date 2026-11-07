/* Built-in licensed ONNX checkpoints for the on-device RVC laboratory.
   No iframe and no remote inference. All downloads are weights only.
   The browser's IndexedDB stores model files for reuse, where allowed. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  // The former q8 exports contain ConvInteger, which ONNX Runtime Web WASM
  // cannot initialize. Use browser-tested FP16 *weights* with FP32 I/O.
  // Cache identity is tied to the exact file + graph format, not just the role.
  const HF_BASE = "https://huggingface.co/ampixa/rvc-browser-runtime/resolve/main/";
  const HF_VOICE = "https://huggingface.co/ampixa/nepali-rvc-voices-v1/resolve/45908909e0e1b01ae45c8254ef939045d221b510/browser/";
  const voicePresets = Object.freeze({
    ne_rvc_m01: {label:"Male 01", file:"ne_rvc_m01.512.fp16.onnx"},
    ne_rvc_f01: {label:"Female 01",file:"ne_rvc_f01.512.fp16.onnx"},
    ne_rvc_m02: {label:"Male 02",file:"ne_rvc_m02.512.fp16.onnx"},
    ne_rvc_f02: {label:"Female 02",file:"ne_rvc_f02.512.fp16.onnx"}
  });
  const baseModels = [
    {key:"contentvec-browser-fp16-v3",kind:"contentVec",url:HF_BASE+"contentvec.fp16.onnx",
      filename:"contentvec.fp16.onnx",minSize:189437475,maxSize:189437475},
    {key:"rmvpe-browser-fp16-v3",kind:"rmvpe",url:HF_BASE+"rmvpe.fp16.onnx",
      filename:"rmvpe.fp16.onnx",minSize:181070368,maxSize:181070368}
  ];
  const local = new Map();
  let preparing = null;
  let controller = null;
  const DB="irl-rvc-voice-models-v3", STORE="weights";
  const progress = $("rvc-model-progress");
  const select = $("rvc-preset");
  const status = $("rvc-model-status");
  const sourceFile = $("rvc-model");

  function report(text,percentage) {
    if(status)status.textContent=text;
    if(progress){
      progress.hidden=false;
      if(Number.isFinite(percentage))progress.value=Math.max(0,Math.min(100,percentage));
      else progress.removeAttribute("value");
    }
    const liveStatus=$("rvc-live-status");
    if(liveStatus)liveStatus.textContent=text;
  }
  function openDB() {
    return new Promise((resolve,reject)=>{
      if(!("indexedDB" in window))return reject(new Error("IndexedDB unavailable"));
      const req=indexedDB.open(DB,1);
      req.onupgradeneeded=()=>{
        if(!req.result.objectStoreNames.contains(STORE))
          req.result.createObjectStore(STORE,{keyPath:"key"});
      };
      req.onerror=()=>reject(req.error||new Error("IndexedDB failed"));
      req.onsuccess=()=>resolve(req.result);
    });
  }
  async function getCached(key) {
    if(local.has(key))return local.get(key);
    let db;
    try{
      db=await openDB();
      const item=await new Promise((resolve,reject)=>{
        const tx=db.transaction(STORE,"readonly");
        const request=tx.objectStore(STORE).get(key);
        request.onsuccess=()=>resolve(request.result||null);
        request.onerror=()=>reject(request.error);
      });
      if(item?.blob?.size && item.filename){
        const file=new File([item.blob],item.filename,{type:"application/octet-stream"});
        local.set(key,file);
        return file;
      }
    }catch(err){console.debug("RVC cache read unavailable:",err.message)}
    finally{db?.close();}
    return null;
  }
  async function persist(key,file) {
    let db;
    try{
      db=await openDB();
      await new Promise((resolve,reject)=>{
        const tx=db.transaction(STORE,"readwrite");
        tx.objectStore(STORE).put({key,filename:file.name,blob:file,at:Date.now()});
        tx.oncomplete=resolve;
        tx.onerror=()=>reject(tx.error||new Error("Model storage failed"));
        tx.onabort=()=>reject(tx.error||new Error("Storage transaction aborted"));
      });
      return true;
    }catch(err){
      console.warn("RVC model not cached; current session can still continue",err.message);
      return false;
    }finally{db?.close();}
  }
  async function obtain(entry,number,total) {
    let file=await getCached(entry.key);
    if(file && file.name===entry.filename && file.size>=entry.minSize && file.size<=entry.maxSize){
      report("Step "+number+"/"+total+": "+entry.filename+" loaded from browser cache.",(number/total)*100);
      return file;
    }
    report("Step "+number+"/"+total+": downloading "+entry.filename+"…",((number-1)/total)*100);
    const resp=await fetch(entry.url,{signal:controller?.signal,redirect:"follow"});
    if(!resp.ok)throw new Error(entry.filename+" HTTP "+resp.status);
    const type=resp.headers.get("content-type")||"";
    if(type.includes("text/html"))throw new Error(entry.filename+" returned an HTML page rather than an ONNX file");
    if(!resp.body)throw new Error(entry.filename+" did not provide a readable download stream");
    const reader=resp.body.getReader();
    const chunks=[];
    let downloaded=0;
    try{
      while(true){
        const part=await reader.read();
        if(part.done)break;
        chunks.push(part.value);
        downloaded+=part.value.byteLength;
        if(downloaded>entry.maxSize)throw new Error("ONNX download exceeded expected model size");
        const share=(downloaded/entry.maxSize);
        report("Step "+number+"/"+total+": "+entry.filename+" — "+(downloaded/1e6).toFixed(1)+" MB",((number-1)+Math.min(1,share))/total*100);
      }
    }finally{reader.releaseLock();}
    if(downloaded<entry.minSize)throw new Error(entry.filename+" was only "+(downloaded/1e6).toFixed(1)+" MB; model download incomplete");
    file=new File(chunks,entry.filename,{type:"application/octet-stream"});
    local.set(entry.key,file);
    report("Caching "+entry.filename+" for repeat visits…",(number/total)*100);
    const saved=await persist(entry.key,file);
    report("Step "+number+"/"+total+" ready"+(saved?" and cached":" (not cached; limited browser storage)")+".",(number/total)*100);
    return file;
  }
  function chosenPreset() {
    return select?.value||"ne_rvc_m01";
  }
  async function prepare() {
    if(preparing)return preparing;
    preparing=(async()=>{
      controller=new AbortController();
      const requested=chosenPreset();
      const chosen=voicePresets[requested];
      const custom=sourceFile?.files?.[0]||null;
      if(requested==="custom" && !custom)throw new Error("Select your own trained RVC .onnx voice model in the upload box, or choose a built-in voice.");
      const entries=[
        ...baseModels.filter(base=>
          !(base.kind==="contentVec"?$("rvc-vec")?.files?.length:$("rvc-pitch")?.files?.length)),
        ...(!custom&&chosen?[{key:"voice-"+requested,kind:"voice",url:HF_VOICE+chosen.file,
          filename:chosen.file,minSize:50e6,maxSize:70e6}]:[])
      ];
      const files={}; // Never reuse already-prepared incompatible model objects from older pages
      const models=entries.length;
      for(let i=0;i<models;i++){
        const entry=entries[i];
        files[entry.kind]=await obtain(entry,i+1,models);
      }
      if(chosen && !custom){
        files.presetId=requested;
        files.fixedVoiceFrames=512;
      }else{
        files.presetId="custom";
        files.fixedVoiceFrames=0;
        files.voice=custom;
      }
      if(!files.contentVec && $("rvc-vec")?.files?.[0])files.contentVec=$("rvc-vec").files[0];
      if(!files.rmvpe && $("rvc-pitch")?.files?.[0])files.rmvpe=$("rvc-pitch").files[0];
      if(!files.voice && custom)files.voice=custom;
      if(!files.voice||!files.contentVec||!files.rmvpe)
        throw new Error("A required voice, ContentVec, or RMVPE ONNX file was not available.");
      if(/_q8[.]onnx|_int8[.]onnx/i.test(files.contentVec.name)||/_q8[.]onnx|_int8[.]onnx/i.test(files.rmvpe.name))
        throw new Error("Quantized q8 ContentVec/RMVPE models are incompatible with this browser runtime (ConvInteger). Remove the custom q8 file or use the built-in FP16 model.");
      window.__IRL_RVC_prepared=files;
      report("All three real ONNX models ready. Initializing neural RVC for microphone conversion…",100);
      return files;
    })();
    try{return await preparing;}
    catch(err){
      report("Model setup failed: "+(err?.message||err));
      throw err;
    }finally{
      preparing=null;
      controller=null;
    }
  }
  window.__IRL_RVC_prepareModels=prepare;
  window.__IRL_RVC_cancelPreparation=()=>controller?.abort();
  select?.addEventListener("change",()=>{
    window.__IRL_RVC_prepared=null;
    if(progress){progress.hidden=true;progress.value=0;}
    if(select.value==="custom"){
      report("Custom voice selected. Upload a trained .onnx checkpoint below; base models will still download automatically.");
    }else{
      report("Selected "+voicePresets[select.value].label+". Press Start to download and load the model locally (cached on later visits).");
    }
  });
  sourceFile?.addEventListener("change",()=>{
    if(sourceFile.files?.length){select.value="custom";window.__IRL_RVC_prepared=null;report("Custom ONNX voice selected. Click Start to prepare its required supporting models.");}
  });
  report("Built-in "+voicePresets[chosenPreset()]?.label+" ready for one-click download. Models download when you press Start; not on page load.");
  if(progress)progress.hidden=true;
})();