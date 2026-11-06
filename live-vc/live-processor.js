/* Live IRL voice transformation. Only transformed output is routed to speakers.
   This is a DSP granular pitch shifter, not a trained RVC/neural model.
   Failure, silence, or uninitialized state outputs zeros (never raw mic). */
class IRLVoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 32768;
    this.mask = this.size - 1;
    this.buffer = new Float32Array(this.size);
    this.write = 0;
    this.total = 0;
    this.clock = 0;
    this.pitch = Math.pow(2, -5 / 12);
    this.brightness = 0.52;
    this.gate = 0.0035;
    this.saturation = 0.08;
    this.volume = 0.75;
    this.enabled = false;
    this.grain = Math.max(1024, Math.round(sampleRate * 0.055));
    this.lowpass = 0;
    this.envelope = 0;
    this.sampleRateValue = sampleRate;
    this.counter = 0;
    this.inputEnergy = 0;
    this.outputEnergy = 0;
    this.port.onmessage = event => {
      const d = event.data || {};
      if (d.type === "configure") {
        const pitch = Number(d.pitch);
        const brightness = Number(d.brightness);
        const gate = Number(d.gate);
        const saturation = Number(d.saturation);
        const volume = Number(d.volume);
        if ([pitch,brightness,gate,saturation,volume].every(Number.isFinite)) {
          this.pitch = Math.pow(2, Math.max(-12, Math.min(12,pitch)) / 12);
          this.brightness = Math.max(0,Math.min(1,brightness));
          this.gate = Math.pow(10,Math.max(-75,Math.min(-25,gate))/20);
          this.saturation = Math.max(0,Math.min(1,saturation));
          this.volume = Math.max(0,Math.min(1,volume));
        }
      } else if (d.type === "enable") {
        this.enabled = Boolean(d.value);
        if (!this.enabled) this.buffer.fill(0);
        this.write = 0; this.total = 0; this.clock = 0; this.lowpass = 0; this.envelope = 0;
      }
    };
  }
  delayRead(delay) {
    const d = Math.min(this.size - 4, Math.max(1, delay));
    const at = (this.write - d + this.size) % this.size;
    const i = Math.floor(at);
    const f = at - i;
    return this.buffer[i] * (1-f) + this.buffer[(i+1) & this.mask] * f;
  }
  grainRead(phase) {
    const rate = this.pitch;
    const base = this.grain * 1.7;
    const delta = Math.abs(rate - 1) * this.grain;
    const delay = rate >= 1 ? base + (1-phase)*delta : base + phase*delta;
    return this.delayRead(delay);
  }
  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0] && outputs[0][0];
    if (!output) return true;
    output.fill(0); // fail-closed; no automatic dry connection
    if (!this.enabled || !input) return true;
    const grain = this.grain;
    for (let i = 0; i < output.length; i++) {
      const x = Number.isFinite(input[i]) ? Math.max(-1,Math.min(1,input[i])) : 0;
      this.buffer[this.write] = x;
      this.write = (this.write + 1) & this.mask;
      this.total++;
      this.clock++;
      const amp = Math.abs(x);
      this.envelope = this.envelope*0.995 + amp*0.005;
      this.inputEnergy += x*x;
      if (this.total <= grain*3) continue;
      const phase = (this.clock % grain) / grain;
      const p2 = (phase + 0.5) % 1;
      const w1 = 0.5 - 0.5*Math.cos(2*Math.PI*phase);
      const w2 = 1 - w1;
      const pitched = this.grainRead(phase)*w1 + this.grainRead(p2)*w2;
      const cutoff = 400 + this.brightness*this.brightness*13000;
      const alpha = 1 - Math.exp(-2*Math.PI*cutoff/this.sampleRateValue);
      this.lowpass += alpha*(pitched-this.lowpass);
      const altered = this.lowpass*(1 + this.saturation*4);
      const colored = Math.tanh(altered);
      const gateGain = Math.max(0, Math.min(1, (this.envelope - this.gate * 0.55) / (this.gate * 1.3)));
      const y = Number.isFinite(colored) ? this.volume*colored*gateGain : 0;
      output[i] = y;
      this.outputEnergy += y*y;
    }
    this.counter += output.length;
    if (this.counter >= this.sampleRateValue/2) {
      const n = this.counter;
      this.port.postMessage({
        type:"meters", rmsIn:Math.sqrt(this.inputEnergy/n),
        rmsOut:Math.sqrt(this.outputEnergy/n),
        algorithmMs:Math.round(this.grain*2.7/this.sampleRateValue*1000),
        sampleRate:this.sampleRateValue
      });
      this.counter=0;this.inputEnergy=0;this.outputEnergy=0;
    }
    return true;
  }
}
registerProcessor("irl-live-voice", IRLVoiceProcessor);
