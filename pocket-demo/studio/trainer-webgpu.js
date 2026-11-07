/* Browser WebGPU student trainer. This trains a genuine *small* bounded-lookahead
 * waveform-to-waveform convolutional model on locally stored speech pairs.
 * IMPORTANT: two Pocket TTS renditions of the same text are not a properly
 * aligned RVC distillation dataset. The resulting voice match is exploratory.
 * Exports the learned Conv weights (not dummy weights) to ONNX, then verifies
 * outputs with ONNX Runtime Web before saving to IndexedDB.
 */
(function(root){
"use strict";
const data=()=>root.PocketTrainerData;
const onnx=()=>root.PocketTrainerONNX;
function loadScript(url){
 return new Promise((resolve,reject)=>{
  const s=document.createElement("script");
  s.src=url;s.async=true;
  s.onload=resolve;s.onerror=()=>reject(Error("Could not load "+url));
  document.head.append(s);
 });
}
async function setupTF(progress,allowCPU){
 if(!root.tf){
  progress("Loading TensorFlow.js runtime for trainable WebGPU tensors…");
  await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js");
 }
 let gpuError=null;
 if(navigator.gpu){
  try{
   if(!root.tf.findBackend("webgpu")){
    progress("Loading WebGPU backend for local gradient-based training…");
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.js");
   }
   if(!await root.tf.setBackend("webgpu"))throw Error("WebGPU backend refused initialization");
   await root.tf.ready();
   const small=root.tf.tensor1d([1,2,3]);
   await small.square().data();small.dispose();
   progress("WebGPU tensors initialized. Testing training kernels…");
   return {tf:root.tf,backend:"webgpu"};
  }catch(e){gpuError=e;progress("WebGPU unavailable: "+e.message)}
 }
 if(!allowCPU)throw Error("WebGPU not available for training on this browser. "+
  (gpuError?.message||"No navigator.gpu adapter")+
  ". Enable CPU fallback to try much slower local training.");
 await root.tf.setBackend("cpu");
 await root.tf.ready();
 progress("Using CPU fallback; training is still local but substantially slower than GPU.");
 return {tf:root.tf,backend:"cpu"};
}
// TFJS 4.22 does not support CAUSAL Conv1D or dilated Conv2D gradients.
// Use trainable SAME non-dilated 31-tap convolutions (at most 45 samples
// = 2.8 ms future lookahead) and export identical weights to ONNX.
function createModel(tf){
 const input=tf.input({shape:[null,1],name:"training_audio"});
 let x=tf.layers.conv1d({filters:16,kernelSize:31,dilationRate:1,
  padding:"same",activation:"tanh",name:"pocket_conv1"}).apply(input);
 x=tf.layers.conv1d({filters:16,kernelSize:31,dilationRate:1,
  padding:"same",activation:"tanh",name:"pocket_conv2"}).apply(x);
 x=tf.layers.conv1d({filters:16,kernelSize:31,dilationRate:1,
  padding:"same",activation:"tanh",name:"pocket_conv3"}).apply(x);
 x=tf.layers.conv1d({filters:1,kernelSize:1,
  padding:"same",activation:"linear",name:"pocket_conv4",
  kernelInitializer:"zeros",biasInitializer:"zeros"}).apply(x);
 const residual=tf.layers.add().apply([input,x]);
 const output=tf.layers.activation({activation:"tanh",name:"pocket_output"}).apply(residual);
 return tf.model({inputs:input,outputs:output,name:"experimental_pocket_streaming_student"});
}
function choosePatch(pairs,cache,windowLength,random=Math.random){
 if(!pairs.length)throw Error("No locally stored speech pairs");
 for(let tries=0;tries<30;tries++){
  const pair=pairs[Math.floor(random()*pairs.length)];
  const d=cache.get(pair.id);if(!d)continue;
  const n=Math.min(d.source.length,d.target.length);
  if(n<windowLength)continue;
  const at=Math.floor(random()*(n-windowLength+1));
  let energy=0;
  for(let i=at;i<at+windowLength;i+=16)energy+=d.source[i]*d.source[i];
  if(energy/(windowLength/16)<0.00015)continue;
  return {source:d.source.subarray(at,at+windowLength),target:d.target.subarray(at,at+windowLength)};
 }
 // A mostly silent dataset is not suitable for speech-to-speech training.
 throw Error("Couldn't find sufficiently loud paired speech windows. Generate clearer Pocket recordings.");
}
async function cacheSamples(pairs,limit=12){
 if(!pairs.length)throw Error("No training pairs available");
 const loaded=new Map(),sample=[...pairs];
 for(let i=sample.length-1;i>0;i--){
  const j=Math.floor(Math.random()*(i+1));[sample[i],sample[j]]=[sample[j],sample[i]];
 }
 for(const pair of sample.slice(0,Math.min(limit,sample.length))){
  const [source,target]=await Promise.all([
   data().decodeWav(pair.source),data().decodeWav(pair.target)]);
  loaded.set(pair.id,{source,target});
 }
 return loaded;
}
async function loadORT(){
 if(!root.ort)await loadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js");
 root.ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
 root.ort.env.wasm.numThreads=1;root.ort.env.wasm.proxy=false;
}
async function validateONNX(blob,model,tf){
 await loadORT();
 const session=await root.ort.InferenceSession.create(await blob.arrayBuffer(),{
  executionProviders:["wasm"],graphOptimizationLevel:"basic",
  enableMemPattern:false,enableCpuMemArena:false
 });
 try{
  if(!session.inputNames.includes("audio_in")||!session.outputNames.includes("audio_out"))
   throw Error("Exported ONNX input/output names do not match the streaming voice changer.");
  const n=2048,pcm=new Float32Array(n);
  for(let i=0;i<n;i++)pcm[i]=.12*Math.sin(i*2*Math.PI*180/16000);
  const output=await session.run({audio_in:new root.ort.Tensor("float32",pcm,[1,1,n])});
  const out=output.audio_out?.data;
  if(!out||out.length!==n||!out.every(Number.isFinite))
   throw Error("Trained ONNX model failed to produce finite 16kHz PCM");
  const predicted=model.predict(tf.tensor3d(pcm,[1,n,1]));
  const direct=Float32Array.from(await predicted.data());predicted.dispose();
  let maxError=0;
  for(let i=0;i<n;i++)maxError=Math.max(maxError,Math.abs(out[i]-direct[i]));
  if(maxError>.035)throw Error("ONNX weights differ from the browser-trained weights, max error "+maxError.toFixed(3));
  return {samples:n,maxError};
 }finally{await session.release()}
}
async function runTraining({steps=240,allowCPU=true,onProgress=()=>{},onEpoch=()=>{},cancelled=()=>false}={}){
 if(!onnx())throw Error("Local ONNX exporter not loaded");
 if(!data())throw Error("Local training database not loaded");
 const pairs=(await data().listPairs()).filter(p=>p.source&&p.target);
 if(pairs.length<4)throw Error("Need at least 4 paired Pocket TTS sentences before training.");
 const loaded=await cacheSamples(pairs,Math.min(pairs.length,12));
 const actualPairs=pairs.filter(p=>loaded.has(p.id));
 const device=await setupTF(onProgress,allowCPU);
 const tf=device.tf;
 let backend=device.backend;
 let model=createModel(tf);
 let optimizer=tf.train.adam(.0005);
 const windowLen=1024,batch=6;
 let finalLoss=NaN,completed=0;
 try{
  for(let step=0;step<steps;step++){
   if(cancelled())break;
   const x=new Float32Array(batch*windowLen),y=new Float32Array(batch*windowLen);
   for(let b=0;b<batch;b++){
    const item=choosePatch(actualPairs,loaded,windowLen);
    x.set(item.source,b*windowLen);
    y.set(item.target,b*windowLen);
   }
   const source=tf.tensor3d(x,[batch,windowLen,1]);
   const target=tf.tensor3d(y,[batch,windowLen,1]);
   let loss=null;
   let retryCPU=false;
   try{
    // Train actual weights; identity preservation discourages output collapse
    // caused by imperfect alignment between independently generated TTS clips.
    loss=optimizer.minimize(()=>tf.tidy(()=>{
     const prediction=model.apply(source);
     const l1=prediction.sub(target).square().mean();
     const preservation=prediction.sub(source).square().mean();
     return l1.add(preservation.mul(.2));
    }),true);
    if(!loss)throw Error("Training produced no loss tensor");
    finalLoss=(await loss.data())[0];
    if(!Number.isFinite(finalLoss))throw Error("Training diverged (nonfinite loss)");
   }catch(e){
    if(backend==="webgpu"&&allowCPU){
     retryCPU=true;
     onProgress("WebGPU gradient execution failed ("+e.message+
      "). Attempting slower local CPU training instead, without claiming GPU success.");
    }else throw Error(backend==="webgpu"?"WebGPU gradient kernels failed: "+e.message:e.message);
   }finally{source.dispose();target.dispose();loss?.dispose()}
   if(retryCPU){
    // WebGPU kernels are not all differentiable in TFJS 4.22.
    // Start from a clean CPU model instead of using partially failed weights.
    optimizer.dispose?.();model.dispose();
    if(!await tf.setBackend("cpu"))throw Error("WebGPU training failed and TensorFlow CPU fallback is unavailable");
    await tf.ready();
    model=createModel(tf);
    optimizer=tf.train.adam(.0005);
    backend="cpu";finalLoss=NaN;completed=0;step=-1;
    onProgress("Restarted real gradient training on CPU; data and voice stays local.");
    continue;
   }
   completed=step+1;
   if(step%5===0||completed===steps){
    onEpoch({completed,steps,loss:finalLoss,backend});
    await tf.nextFrame();
   }
  }
  if(completed===0)throw Error("Training cancelled before the first gradient step");
  onProgress("Converting actual trained WebGPU/CPU weights into ONNX graph…");
  const result=await onnx().exportCausalOnnx(model,{steps:completed});
  onProgress("Running local ONNX Runtime Web validation on exported weights…");
  const validation=await validateONNX(result,model,tf);
  const bundle={blob:result,backend,steps:completed,loss:finalLoss,
   sourcePairs:pairs.length,createdAt:new Date().toISOString(),
   quality:"UNVERIFIED: synthetic TTS voice conversion experiment",
   validation};
  await data().putMeta("student-onnx",bundle);
  onProgress("Saved validated ONNX locally. Open Fast Neural AI and select My saved trained model.");
  return bundle;
 }finally{
  optimizer.dispose?.();model.dispose();loaded.clear();
 }
}
root.PocketTrainerWebGPU={runTraining,createModel,validateONNX};
})(typeof window!=="undefined"?window:globalThis);
