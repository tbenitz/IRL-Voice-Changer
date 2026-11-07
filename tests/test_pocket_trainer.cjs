/* Real Pocket student exporter regression:
 * actual TensorFlow.js gradient update -> ONNX protobuf -> ONNX Runtime Web WASM.
 * Also checks the WAV and ZIP artifacts are genuine, not placeholder downloads.
 */
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const tf=require("@tensorflow/tfjs"),ort=require("onnxruntime-web");
const data=require("../pocket-demo/studio/trainer-data.js");
const exporter=require("../pocket-demo/studio/trainer-onnx.js");
require("../pocket-demo/studio/trainer-webgpu.js");
function signal(n){const x=new Float32Array(n);for(let i=0;i<n;i++)x[i]=.15*Math.sin(i*2*Math.PI*170/16000);return x}
(async()=>{
 const wav=data.wav(signal(16000));
 assert.equal(wav.size,44+16000*2,"Training WAV must contain real PCM, not metadata");
 const audio=await data.decodeWav(wav);
 assert.equal(audio.length,16000);
 assert(Math.abs(audio[500])<1,"Decoded WAV contains corrupt values");
 const archive=await data.buildZip([{name:"target/example.wav",blob:wav},
  {name:"manifest.json",blob:new Blob(['{"pairs":1}'])}]);
 const zip=new Uint8Array(await archive.arrayBuffer());
 const view=new DataView(zip.buffer);
 assert.equal(view.getUint32(0,true),0x04034b50,"ZIP has no valid local file header");
 assert.equal(view.getUint32(zip.length-22,true),0x06054b50,"ZIP central directory absent");
 fs.writeFileSync("/tmp/pocket-trainer-example.zip",zip);
 await tf.setBackend("cpu");await tf.ready();
 const m=globalThis.PocketTrainerWebGPU.createModel(tf);
 const optim=tf.train.adam(.001);
 const x=tf.tensor3d(signal(512),[1,512,1]);
 const y=tf.tensor3d(signal(512),[1,512,1]);
 for(let i=0;i<2;i++){
  const loss=optim.minimize(()=>tf.tidy(()=>{
   const p=m.apply(x);
   return p.sub(y).square().mean();
  }),true);
  assert(Number.isFinite((await loss.data())[0]),"Training loss nonfinite");
  loss.dispose();
 }
 const onnx=await exporter.exportCausalOnnx(m);
 assert(onnx.size>1000&&onnx.size<1000000,"ONNX output empty or unexpectedly large");
 fs.writeFileSync("/tmp/pocket-trained-student.onnx",new Uint8Array(await onnx.arrayBuffer()));
 ort.env.wasm.wasmPaths=path.dirname(require.resolve("onnxruntime-web/dist/ort.min.js"))+path.sep;
 ort.env.wasm.numThreads=1;
 const session=await ort.InferenceSession.create(await onnx.arrayBuffer(),{
  executionProviders:["wasm"],graphOptimizationLevel:"basic"
 });
 assert(session.inputNames.includes("audio_in"),"ONNX input wrong");
 assert(session.outputNames.includes("audio_out"),"ONNX output wrong");
 const pcm=signal(2048);
 const result=await session.run({audio_in:new ort.Tensor("float32",pcm,[1,1,pcm.length])});
 const actual=result.audio_out.data;
 assert.equal(actual.length,pcm.length);
 assert(actual.every(Number.isFinite),"ONNX output contains NaN");
 const tfPrediction=m.predict(tf.tensor3d(pcm,[1,pcm.length,1]));
 const reference=await tfPrediction.data();
 let maximum=0;for(let i=0;i<pcm.length;i++)maximum=Math.max(maximum,Math.abs(reference[i]-actual[i]));
 assert(maximum<.035,"ONNX exported model does not match trained TFJS model. diff="+maximum);
 console.log("PASS: real TFJS weights trained, exported to runnable ONNX Conv1D and confirmed against TFJS output; max diff="+maximum);
 await session.release();tfPrediction.dispose();x.dispose();y.dispose();m.dispose();optim.dispose();
})().catch(e=>{console.error(e);process.exitCode=1});
