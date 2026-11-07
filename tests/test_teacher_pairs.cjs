const assert=require("node:assert/strict");
const D=require("../pocket-demo/studio/trainer-data.js");
const T=require("../pocket-demo/studio/trainer-teacher-pairs.js");
function signal(n,f=170){
 const x=new Float32Array(n);
 for(let i=0;i<n;i++)x[i]=.21*Math.sin(2*Math.PI*f*i/16000);
 return x;
}
function file(name,pcm){
 const wav=D.wav(pcm);
 wav.name=name;return wav;
}
async function bad(files,pattern){
 await assert.rejects(()=>T.prepareFiles(files,D.decodeWav,pcm=>D.wav(pcm)),pattern);
}
(async()=>{
 const originals=signal(32000),converted=signal(32000,240);
 const one=T.groupFiles([file("clip-01_original.wav",originals),
   file("clip-01_converted.wav",converted)]);
 assert.equal(one.length,1);assert.equal(one[0].name,"clip-01");
 const p=await T.prepareFiles([file("clip-01_original.wav",originals),
   file("clip-01_converted.wav",converted)],D.decodeWav,pcm=>D.wav(pcm));
 assert.equal(p.length,1);assert.equal(p[0].source.size,44+originals.length*2);
 assert.equal(p[0].target.size,44+converted.length*2);
 assert(p[0].relativeDifference>.015);
 await bad([file("clip_original.wav",originals)],/lacks/);
 await bad([file("clip_original.wav",originals),
  file("clip_converted.wav",signal(24000,240))],/mismatched durations/);
 await bad([file("clip_original.wav",originals),
  file("clip_converted.wav",originals)],/almost identical/);
 await bad([file("clip_original.wav",originals),
  file("clip_converted.wav",new Float32Array(32000))],/effectively silent/);
 console.log("PASS: genuine teacher-pair importer matches source/converted WAVs, rejects mismatched timing, identical audio, silence, missing files; preserves 16 kHz PCM.");
})().catch(e=>{console.error(e);process.exitCode=1});
