/* AudioWorklet: captures microphone audio into PCM blocks. The output is
 * always silence. The only audible RVC signal is synthesized model output
 * routed separately by the main thread. */
class RVCMicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.enabled = false;
    this.frames = Math.round(sampleRate * 1.4);
    this.chunk = new Float32Array(this.frames);
    this.position = 0;
    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "config" && Number.isFinite(msg.chunkMs)) {
        this.frames = Math.max(Math.round(sampleRate * .7),
          Math.min(Math.round(sampleRate * 3),Math.round(sampleRate * msg.chunkMs/1000)));
        this.chunk = new Float32Array(this.frames);
        this.position=0;
      } else if (msg.type === "enable") {
        this.enabled = Boolean(msg.enabled);
        this.position = 0;
        this.chunk = new Float32Array(this.frames);
      }
    };
  }
  process(inputs,outputs) {
    const output=outputs[0];
    if(output)for(const ch of output)ch.fill(0); // absolutely no dry audio
    if(!this.enabled)return true;
    const input=inputs[0]?.[0];
    if(!input)return true;
    let offset=0;
    while(offset<input.length){
      const n=Math.min(input.length-offset,this.frames-this.position);
      for(let i=0;i<n;i++){
        const v=input[offset+i];
        this.chunk[this.position+i]=Number.isFinite(v)?Math.max(-1,Math.min(1,v)):0;
      }
      this.position+=n; offset+=n;
      if(this.position===this.frames){
        const chunk=this.chunk;
        this.chunk=new Float32Array(this.frames);
        this.position=0;
        this.port.postMessage({type:"audio",data:chunk,sampleRate,frames:this.frames},[chunk.buffer]);
      }
    }
    return true;
  }
}
registerProcessor("rvc-mic-capture",RVCMicCapture);
