/* Experimental RVC v2 streaming ONNX worker.
 * Real ContentVec -> RMVPE -> trained RVC generator. No TTS, no fake
 * audio effects, no raw-microphone fallback. Sessions persist across chunks.
 * This is not yet benchmarked or guaranteed realtime on 8 GB ChromeOS.
 * ONNX Runtime Web is a static JS/WASM library; inference never uses an API.
 */
"use strict";
let ortReady=false, modelReady=false, busy=false;
let cv=null, pitchModel=null, gen=null, sr=48000, shift=0, tail16=new Float32Array(0), fixedVoiceFrames=0;
const ORT_BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
function tell(type,extra={}){self.postMessage({type,...extra});}
function errorText(e){return e?.stack||e?.message||String(e);}
const nap=()=>new Promise(resolve=>setTimeout(resolve,20));
async function setupORT(){
 if(ortReady)return;
 tell("stage",{text:"Loading ONNX Runtime WASM (local inference)…"});
 importScripts(ORT_BASE+"ort.min.js");
 ort.env.wasm.wasmPaths=ORT_BASE;
 ort.env.wasm.numThreads=1;
 ort.env.wasm.proxy=false;
 ort.env.wasm.simd=true;
 ort.env.logLevel="error";
 ortReady=true;
}
function tensor(type,data,dims){return new ort.Tensor(type,data,dims);}
async function session(bytes,label){
 tell("stage",{text:"Initializing "+label+" ONNX session…"});
 const s=await ort.InferenceSession.create(bytes,{
  executionProviders:["wasm"],graphOptimizationLevel:"basic",
  enableCpuMemArena:false,enableMemPattern:false
 });
 tell("stage",{text:label+" ready ("+s.inputNames.join(", ")+")"});
 await nap();
 return s;
}
function downsample(x,rate){
 if(rate===16000)return x;
 const n=Math.max(1,Math.round(x.length*16000/rate));
 const y=new Float32Array(n);
 for(let i=0;i<n;i++){
  const at=i*rate/16000, a=Math.floor(at), k=at-a;
  y[i]=(x[Math.min(a,x.length-1)]||0)*(1-k)+(x[Math.min(a+1,x.length-1)]||0)*k;
 }
 return y;
}
function expectedPhoneDim(){
 const meta=gen?.inputMetadata?.phone;
 const dims=meta?.dimensions||meta?.dims||[];
 const n=Number(dims[2]);
 // RVC v2 uses 768-D; RVC v1 uses 256-D. Never invent 768 dimensions
 // by repeating or padding 256-D features from a mismatched model.
 return [256,768].includes(n)?n:768;
}
function contentVecTensorToFeatures(output,expectedDim){
 if(!output||output.dims?.length!==3||output.dims[0]!==1)
  throw Error("ContentVec output must be [1,frames,channels] or [1,channels,frames]; got "+
    (output?.dims?.join("x")||"no tensor"));
 const [batch,d1,d2]=output.dims;
 const length=output.data?.length||0;
 if(length!==batch*d1*d2)throw Error("ContentVec output has mismatched tensor dimensions");
 // IMPORTANT: inspect the *voice generator's* phone width before choosing axes.
 // The old code checked d1===256 before d2===768, so [1,256,768]
 // was incorrectly interpreted as 256-D even for a genuine 768-D export.
 let frames,dim=expectedDim,features,layout;
 if(d2===dim){
  frames=d1;layout="time-major";
  features=Float32Array.from(output.data);
 }else if(d1===dim){
  frames=d2;layout="channel-major";
  features=new Float32Array(frames*dim);
  for(let f=0;f<frames;f++)
   for(let c=0;c<dim;c++)
    features[f*dim+c]=Number(output.data[c*frames+f]);
 }else throw Error("ContentVec output "+output.dims.join("x")+
  " has no "+dim+"-dimensional feature axis required by the selected RVC generator");
 if(!frames||frames>50000)throw Error("ContentVec returned an invalid frame count "+frames);
 if(features.some(v=>!Number.isFinite(v)))throw Error("ContentVec output contained non-finite features");
 return {features,frames,dim,layout};
}
async function contentVec(audio){
 const names=cv.inputNames,feeds={};
 for(const name of names){
  if(name==="source"||name==="audio"||name==="input"||name==="waveform"){
   const meta=cv.inputMetadata?.[name];
   const dims=meta?.dimensions||meta?.dims||[];
   // Different ContentVec ONNX exporters use [1,samples] or [1,1,samples].
   const shape=dims.length===3?[1,1,audio.length]:[1,audio.length];
   feeds[name]=tensor("float32",audio,shape);
  }else if(name==="padding_mask")feeds[name]=tensor("bool",new Uint8Array(audio.length),[1,audio.length]);
  else if(name==="output_layer")feeds[name]=tensor("int64",new BigInt64Array([12n]),[1]);
  else throw Error("Unsupported ContentVec input "+name);
 }
 const results=await cv.run(feeds);
 const dim=expectedPhoneDim();
 let chosen=null,details=[];
 // Some exports provide several outputs. Prefer an actual 768-D feature
 // tensor rather than accidentally using a 256-D final projection.
 for(const name of cv.outputNames){
  const output=results[name];
  details.push(name+":"+(output?.dims?.join("x")||"none"));
  if(output?.dims?.length!==3)continue;
  const d=output.dims;
  if(d[0]!==1||!(d[1]===dim||d[2]===dim))continue;
  chosen={name,...contentVecTensorToFeatures(output,dim)};
  break;
 }
 if(!chosen)throw Error("ContentVec features incompatible with RVC v2: need "+
   dim+" channels; ONNX outputs are ["+details.join(", ")+
   "]. This is a model/export mismatch, not a microphone problem.");
 tell("stage",{text:"ContentVec "+chosen.name+" "+chosen.frames+" frames × "+chosen.dim+
   " channels ("+chosen.layout+")."});
 const {features,frames}=chosen;
 const doubled=new Float32Array(frames*2*dim);
 for(let f=0;f<frames*2;f++){
  const left=Math.floor(f/2),right=Math.min(frames-1,left+1),t=f%2*.5;
  for(let c=0;c<dim;c++) doubled[f*dim+c]=features[left*dim+c]*(1-t)+features[right*dim+c]*t;
 }
 return {data:doubled,frames:frames*2,dim};
}
const FFT=1024, HOP=160, MELS=128;
let filters=null;
function fftMag(frame){
 const re=Float32Array.from(frame),im=new Float32Array(FFT);
 let j=0;for(let i=1;i<FFT;i++){let bit=FFT>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){const t=re[i];re[i]=re[j];re[j]=t}}
 for(let size=2;size<=FFT;size*=2){
  const step=-2*Math.PI/size;
  for(let base=0;base<FFT;base+=size){
   for(let k=0;k<size/2;k++){
    const wr=Math.cos(k*step),wi=Math.sin(k*step),a=base+k,b=a+size/2;
    const tr=wr*re[b]-wi*im[b],ti=wr*im[b]+wi*re[b];
    re[b]=re[a]-tr;im[b]=im[a]-ti;re[a]+=tr;im[a]+=ti;
   }
  }
 }
 const mag=new Float32Array(FFT/2+1);
 for(let i=0;i<mag.length;i++)mag[i]=Math.hypot(re[i],im[i]);
 return mag;
}
function melFilters(){
 if(filters)return filters;
 const hzToMel=hz=>2595*Math.log10(1+hz/700),melToHz=m=>700*(10**(m/2595)-1);
 const min=hzToMel(30),max=hzToMel(8000);
 const bins=Array.from({length:MELS+2},(_,i)=>Math.max(0,Math.min(FFT/2,Math.floor((FFT+1)*melToHz(min+(max-min)*i/(MELS+1))/16000))));
 filters=Array.from({length:MELS},(_,m)=>{
  const f=new Float32Array(FFT/2+1),[a,b,c]=[bins[m],bins[m+1],bins[m+2]];
  for(let i=a;i<b;i++)f[i]=(i-a)/Math.max(1,b-a);
  for(let i=b;i<c;i++)f[i]=(c-i)/Math.max(1,c-b);
  return f;
 });
 return filters;
}
function melData(audio){
 const frames=Math.ceil(audio.length/HOP),padded=Math.ceil(frames/32)*32;
 const result=new Float32Array(MELS*padded);
 const bank=melFilters(),win=new Float32Array(FFT);
 for(let t=0;t<frames;t++){
  for(let i=0;i<FFT;i++)win[i]=(audio[t*HOP+i]||0)*(.5-.5*Math.cos(2*Math.PI*i/(FFT-1)));
  const spectrum=fftMag(win);
  for(let m=0;m<MELS;m++){
   let power=0;const filt=bank[m];
   for(let i=0;i<filt.length;i++)power+=spectrum[i]*filt[i];
   result[m*padded+t]=Math.log(Math.max(1e-5,power));
  }
 }
 return {data:result,frames,padded};
}
function salienceToF0(data,frames,classes,transpose=false){
 const res=new Float32Array(frames);
 for(let f=0;f<frames;f++){
  let best=0,max=0;
  for(let c=0;c<classes;c++){
   const v=Number(transpose?data[c*frames+f]:data[f*classes+c]);
   if(v>max){max=v;best=c}
  }
  if(max<.03)continue;
  let sum=0,total=0;
  for(let c=Math.max(0,best-4);c<=Math.min(classes-1,best+4);c++){
   const v=Math.max(0,Number(transpose?data[c*frames+f]:data[f*classes+c]));
   sum+=v*(1997.3794084376191+20*c);total+=v;
  }
  const hz=10*Math.pow(2,(sum/Math.max(1e-8,total))/1200);
  res[f]=hz>=50&&hz<=1100?hz:0;
 }
 return res;
}
async function pitchF0(audio){
 const feeds={},meta=pitchModel.inputMetadata||{};
 let nFrames=Math.ceil(audio.length/HOP);
 for(const name of pitchModel.inputNames){
  if(name==="threshold")feeds[name]=tensor("float32",new Float32Array([.03]),[1]);
  else if(name==="sr"||name==="sample_rate")feeds[name]=tensor("int64",new BigInt64Array([16000n]),[1]);
  else if(name==="waveform"||name==="audio"||name==="input_audio")feeds[name]=tensor("float32",audio,[1,audio.length]);
  else {
   const mel=melData(audio);
   // Inspect declared ONNX input shape to choose [1,time,mel] or [1,mel,time].
   const dims=meta[name]?.dimensions||meta[name]?.dims||[];
   let data=mel.data,shape=[1,MELS,mel.padded];
   if(dims.length===3&&(dims[2]===128||dims[2]==="128")){
    const flipped=new Float32Array(mel.padded*MELS);
    for(let m=0;m<MELS;m++)for(let f=0;f<mel.padded;f++)flipped[f*MELS+m]=data[m*mel.padded+f];
    data=flipped;shape=[1,mel.padded,MELS];
   }
   feeds[name]=tensor("float32",data,shape);
  }
 }
 const res=await pitchModel.run(feeds);
 const out=res[pitchModel.outputNames[0]];
 let f0;
 if(out.dims.length===2){
  f0=Float32Array.from(out.data);
 }else if(out.dims.length===3){
  const [batch,d1,d2]=out.dims;
  if(batch!==1)throw Error("Unexpected RMVPE batch size");
  if(d2===360) f0=salienceToF0(out.data,d1,360,false);
  else if(d1===360) f0=salienceToF0(out.data,d2,360,true);
  else throw Error("Unknown RMVPE salience dimensions "+out.dims.join("x"));
 }else throw Error("Unexpected RMVPE output shape "+out.dims);
 const trimmed=new Float32Array(nFrames);
 for(let i=0;i<nFrames;i++){
  const hz=Number(f0[Math.min(i,f0.length-1)])||0;
  trimmed[i]=hz>=50&&hz<=1100?hz:0;
 }
 return trimmed;
}
function synthesizerFeeds(features,f0){
 const validFrames=Math.max(1,Math.min(features.frames,f0.length));
 // Some published browser RVC voices use a fixed [1,512,768] phone shape.
 // Pad the model INPUT while retaining the number of real frames for output trim.
 const detected=gen.inputMetadata?.phone?.dimensions||gen.inputMetadata?.phone?.dims||[];
 const declaredFrames=Number(detected[1]);
 const maxFrames=Number.isInteger(declaredFrames)&&declaredFrames>0?declaredFrames:fixedVoiceFrames;
 const frames=maxFrames>0?maxFrames:validFrames;
 if(validFrames>frames)throw Error("Audio block exceeds voice model's "+frames+" frame input. Lower the microphone block size.");
 const data=new Float32Array(frames*features.dim);
 data.set(features.data.subarray(0,validFrames*features.dim));
 for(let i=validFrames;i<frames;i++)
   data.set(data.subarray((validFrames-1)*features.dim,validFrames*features.dim),i*features.dim);
 const shifted=new Float32Array(frames);
 const cents=new BigInt64Array(frames);
 const min=1127*Math.log(1+50/700),max=1127*Math.log(1+1100/700);
 for(let i=0;i<frames;i++){
  const hz=i<validFrames?f0[i]*Math.pow(2,shift/12):0;
  shifted[i]=hz;
  cents[i]=BigInt(hz<=0?1:Math.round(Math.max(1,Math.min(255,(1127*Math.log(1+hz/700)-min)*254/(max-min)+1))));
 }
 const noise=new Float32Array(192*frames);
 for(let i=0;i<noise.length;i++)noise[i]=(Math.random()+Math.random()+Math.random()+Math.random()-2)*.75;
 const all={
  phone:tensor("float32",data,[1,frames,features.dim]),
  phone_lengths:tensor("int64",new BigInt64Array([BigInt(frames)]),[1]),
  pitch:tensor("int64",cents,[1,frames]),
  pitchf:tensor("float32",shifted,[1,frames]),
  nsff0:tensor("float32",shifted,[1,frames]),
  sid:tensor("int64",new BigInt64Array([0n]),[1]),
  ds:tensor("int64",new BigInt64Array([0n]),[1]),
  rnd:tensor("float32",noise,[1,192,frames])
 };
 const feeds={};
 for(const name of gen.inputNames){
  if(!all[name])throw Error("Unsupported RVC generator input '"+name+"'; expected RVC v2 ONNX");
  feeds[name]=all[name];
 }
 return {feeds,validFrames,frames};
}
async function synth(audio){
 const features=await contentVec(audio);
 const f0=await pitchF0(audio);
 const {feeds,validFrames}=synthesizerFeeds(features,f0);
 const out=await gen.run(feeds);
 const t=out.audio||out[gen.outputNames[0]];
 if(!(t?.data instanceof Float32Array))throw Error("RVC generator did not return Float32 PCM audio");
 // The fixed 512-frame model synthesizes 5.12 seconds even for a one-second mic
 // block. Discard the padded tail or it will cause enormous playback backlog.
 const usable=Math.min(t.data.length,Math.max(1,Math.round(validFrames*sr/100)));
 const wav=new Float32Array(t.data.subarray(0,usable));
 if(!wav.length||wav.length>sr*20)throw Error("Invalid RVC output length");
 for(let i=0;i<wav.length;i++)if(!Number.isFinite(wav[i]))throw Error("RVC produced invalid audio values");
 return wav;
}
async function convert(buffer,rate){
 const start=performance.now();
 const current=downsample(new Float32Array(buffer),rate);
 let energy=0;for(let i=0;i<current.length;i+=8)energy+=current[i]**2;
 const rms=Math.sqrt(energy/Math.ceil(current.length/8));
 const ctxFrames=Math.round(16000*.35);
 const history=tail16.slice(-ctxFrames);
 const input=new Float32Array(history.length+current.length);
 input.set(history,0);input.set(current,history.length);
 tail16=current.slice(-ctxFrames);
 if(rms<.002){tell("audio",{pcm:new Float32Array(0),sampleRate:sr,inferMs:0,silence:true});return}
 const full=await synth(input);
 // RVC outputs samples at its model's sampling rate; trim the actual
 // historical context, leaving only the new block of speech.
 const startFrame=Math.max(0,Math.round(history.length/16000*sr));
 const trimmed=full.slice(Math.min(startFrame,full.length));
 if(!trimmed.length)throw Error("RVC yielded no samples after context trim");
 tell("audio",{
   pcm:trimmed,sampleRate:sr,inferMs:performance.now()-start,
   speechMs:1000*current.length/16000,silence:false
 },[trimmed.buffer]);
}
self.onmessage=async e=>{
 const d=e.data||{};
 if(d.type==="stop"){busy=false;tail16=new Float32Array(0);return}
 if(d.type==="config"){shift=Number(d.pitch)||0;return}
 if(d.type==="init"){
  if(busy)return;busy=true;modelReady=false;
  try{
   sr=[32000,40000,48000].includes(Number(d.sampleRate))?Number(d.sampleRate):48000;
   fixedVoiceFrames=Number.isInteger(d.fixedVoiceFrames)&&d.fixedVoiceFrames>=0?d.fixedVoiceFrames:0;
   await setupORT();
   // One model at a time to avoid simultaneous peak allocation.
   cv=await session(d.contentVec,"ContentVec");
   pitchModel=await session(d.rmvpe,"RMVPE");
   gen=await session(d.voice,"RVC generator");
   modelReady=true;tail16=new Float32Array(0);
   tell("ready",{text:"All three real ONNX models are initialized and reusable across microphone chunks."});
  }catch(err){modelReady=false;tell("error",{text:"Model initialization: "+errorText(err)});}
  finally{busy=false}
  return;
 }
 if(d.type==="convert"){
  if(!modelReady||busy){tell("dropped",{reason:!modelReady?"Models not initialized":"Inference already running"});return}
  busy=true;
  try{await convert(d.pcm,d.sampleRate)}
  catch(err){tell("error",{text:"RVC inference: "+errorText(err)});}
  finally{busy=false;tell("idle");}
 }
};