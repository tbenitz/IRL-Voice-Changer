/* Validates explicitly teacher-converted PARALLEL audio examples.
 * The source and converted files MUST derive from the same original utterance,
 * not two independently synthesized Pocket TTS readings of identical text.
 * This is a data-integrity check, NOT proof of target speaker similarity.
 */
(function(root){
"use strict";
function filePairName(filename){
 const base=String(filename).split(/[\\/]/).pop();
 const m=/^(.+?)[_-](original|converted)\.wav$/i.exec(base);
 return m?{id:m[1],role:m[2].toLowerCase()}:null;
}
function groupFiles(files){
 const groups=new Map();
 for(const file of Array.from(files||[])){
  const p=filePairName(file.name);
  if(!p)throw Error("Expected *_original.wav or *_converted.wav, got "+file.name);
  const id=p.id.toLowerCase();
  if(!groups.has(id))groups.set(id,{name:p.id});
  const entry=groups.get(id);
  if(entry[p.role])throw Error("Duplicate "+p.role+" WAV for "+p.id);
  entry[p.role]=file;
 }
 if(!groups.size)throw Error("Select both original and converted WAVs from a trained voice-conversion teacher");
 for(const pair of groups.values())
  if(!pair.original||!pair.converted)
   throw Error("Pair "+pair.name+" lacks its "+(!pair.original?"original":"converted")+" WAV");
 return [...groups.values()];
}
function stats(source,target){
 if(!source.length||!target.length)throw Error("Empty audio");
 let srcE=0,dstE=0,delta=0,valid=0;
 const n=Math.min(source.length,target.length);
 for(let i=0;i<n;i+=8){
  const a=source[i],b=target[i];
  if(!Number.isFinite(a)||!Number.isFinite(b))throw Error("Invalid audio sample");
  srcE+=a*a;dstE+=b*b;delta+=(b-a)*(b-a);valid++;
 }
 return {sourceRms:Math.sqrt(srcE/valid),targetRms:Math.sqrt(dstE/valid),
  relativeDifference:Math.sqrt(delta/Math.max(srcE,1e-12))};
}
async function decodeTeacherWav(blob,fast16kDecoder){
 if(blob.size>100*1024*1024)throw Error("Teacher WAV over 100 MB; split long recordings into clips first");
 try{return await fast16kDecoder(blob)}
 catch(err){
  // RVC teachers commonly write 40 kHz output. Resample decoded PCM locally,
  // without Python, a backend, or trimming away sample alignment.
  if(!/16kHz mono PCM16/i.test(String(err.message)))throw err;
 }
 const bytes=await blob.arrayBuffer(),v=new DataView(bytes);
 const id=o=>o+4<=bytes.byteLength?
  String.fromCharCode(v.getUint8(o),v.getUint8(o+1),v.getUint8(o+2),v.getUint8(o+3)):"";
 if(id(0)!=="RIFF"||id(8)!=="WAVE")throw Error("Not a RIFF WAV file");
 let pos=12,fmt=null,start=-1,size=0;
 while(pos+8<=bytes.byteLength){
  const type=id(pos),len=v.getUint32(pos+4,true),at=pos+8;
  if(at+len>bytes.byteLength)throw Error("Damaged WAV chunk");
  if(type==="fmt "){
   if(len<16)throw Error("Incomplete WAV format header");
   fmt={encoding:v.getUint16(at,true),channels:v.getUint16(at+2,true),
    rate:v.getUint32(at+4,true),bits:v.getUint16(at+14,true)};
  }
  if(type==="data"){start=at;size=len;break}
  pos=at+len+(len%2);
 }
 if(!fmt||start<0||fmt.rate<8000||fmt.rate>96000||
   fmt.channels<1||fmt.channels>2||
   !((fmt.encoding===1&&[16,24,32].includes(fmt.bits))||
     (fmt.encoding===3&&fmt.bits===32)))
  throw Error("Use 8–96 kHz WAV PCM16/24/32 or Float32, mono or stereo. Compressed WAV isn't supported.");
 const width=fmt.bits/8,frames=Math.floor(size/(width*fmt.channels));
 if(frames<=0||frames/fmt.rate>180)throw Error("Teacher clip is empty or longer than 180 seconds");
 const atFrame=(index)=>{
  const i=Math.max(0,Math.min(frames-1,index));let sum=0;
  for(let ch=0;ch<fmt.channels;ch++){
   const p=start+(i*fmt.channels+ch)*width;let value=0;
   if(fmt.encoding===3)value=v.getFloat32(p,true);
   else if(fmt.bits===16)value=v.getInt16(p,true)/32768;
   else if(fmt.bits===24){
    let word=v.getUint8(p)|(v.getUint8(p+1)<<8)|(v.getUint8(p+2)<<16);
    if(word&0x800000)word-=0x1000000;value=word/8388608;
   }else value=v.getInt32(p,true)/2147483648;
   sum+=value;
  }
  return sum/fmt.channels;
 };
 const length=Math.round(frames*16000/fmt.rate),out=new Float32Array(length);
 for(let i=0;i<length;i++){
  const sourceIndex=i*fmt.rate/16000,a=Math.floor(sourceIndex),fraction=sourceIndex-a;
  out[i]=atFrame(a)*(1-fraction)+atFrame(a+1)*fraction;
 }
 return out;
}
async function prepareFiles(files,decodeWav,storeAudio){
 const groups=groupFiles(files),results=[];
 if(typeof decodeWav!=="function")throw Error("WAV decoder missing");
 for(const pair of groups){
  const a=await decodeTeacherWav(pair.original,decodeWav),
        b=await decodeTeacherWav(pair.converted,decodeWav);
  if(a.length<16000||b.length<16000||a.length>16000*180||b.length>16000*180)
   throw Error(pair.name+": clips must be 1 to 180 seconds of 16kHz mono");
  const tolerance=Math.max(320,Math.round(Math.max(a.length,b.length)*.015));
  if(Math.abs(a.length-b.length)>tolerance)
   throw Error(pair.name+": source and teacher output have mismatched durations ("+
    (a.length/16000).toFixed(2)+"s versus "+(b.length/16000).toFixed(2)+
    "s). Pair the SAME original utterance after teacher conversion; do not combine independent TTS renders.");
  const levels=stats(a,b);
  if(levels.sourceRms<.006||levels.targetRms<.006)
   throw Error(pair.name+": one audio track is effectively silent");
  if(levels.relativeDifference<.015)
   throw Error(pair.name+": converted file is almost identical to source. No usable teacher voice change detected.");
  // Keep actual source/converted waveform lengths as recorded; no time-stretch,
  // silence trim or fake padding that would erase teacher alignment.
  const source=typeof storeAudio==="function"?storeAudio(a):pair.original;
  const target=typeof storeAudio==="function"?storeAudio(b):pair.converted;
  results.push({id:pair.name,source,target,length:a.length,
   seconds:a.length/16000,...levels});
 }
 return results;
}
const api={filePairName,groupFiles,prepareFiles,decodeTeacherWav,stats};
if(typeof module!=="undefined"&&module.exports)module.exports=api;
root.PocketTeacherPairs=api;
})(typeof window!=="undefined"?window:globalThis);
