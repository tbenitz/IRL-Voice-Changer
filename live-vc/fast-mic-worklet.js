/* Chrome AudioWorklet for fast causal VC: mono 16 kHz microphone chunks.
 * All output channels are ALWAYS silence. Never passes raw microphone audio.
 * Uses a bounded averaging decimator so 48k/44.1k native microphones work
 * without a separate resampler thread or sending 3x data to the worker.
 */
class FastNeuralMic extends AudioWorkletProcessor {
 constructor(){
  super();this.enabled=false;this.chunkFrames=2048;
  this.samples=new Float32Array(this.chunkFrames);this.at=0;
  this.phase=0;this.integrator=0;this.integratorSamples=0;
  this.port.onmessage=e=>{
   const m=e.data||{};
   if(m.type==="config"){
    const c=Number(m.chunkFrames);
    if(Number.isInteger(c)&&c>=512&&c<=8192){
     this.chunkFrames=c;
     this.reset();
    }
   }
   if(m.type==="enable"){
    this.enabled=Boolean(m.enabled);
    this.reset();
   }
  };
 }
 reset(){
  this.samples=new Float32Array(this.chunkFrames);
  this.at=0;this.phase=0;this.integrator=0;this.integratorSamples=0;
 }
 process(inputs,outputs){
  for(const ch of outputs[0]||[])ch.fill(0);
  if(!this.enabled)return true;
  const mic=inputs[0]?.[0];if(!mic)return true;
  const ratio=16000/sampleRate;
  for(let i=0;i<mic.length;i++){
   const x=Number.isFinite(mic[i])?Math.max(-1,Math.min(1,mic[i])):0;
   this.integrator+=x;
   this.integratorSamples++;
   this.phase+=ratio;
   if(this.phase>=1){
    this.phase-=1;
    this.samples[this.at++]=this.integrator/Math.max(1,this.integratorSamples);
    this.integrator=0;this.integratorSamples=0;
    if(this.at===this.chunkFrames){
     const full=this.samples;
     this.samples=new Float32Array(this.chunkFrames);this.at=0;
     this.port.postMessage({type:"chunk",pcm:full,sampleRate:16000},[full.buffer]);
    }
   }
  }
  return true;
 }
}
registerProcessor("fast-neural-mic",FastNeuralMic);
