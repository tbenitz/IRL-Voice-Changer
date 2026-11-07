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
async function prepareFiles(files,decodeWav,storeAudio){
 const groups=groupFiles(files),results=[];
 if(typeof decodeWav!=="function")throw Error("WAV decoder missing");
 for(const pair of groups){
  const a=await decodeWav(pair.original),b=await decodeWav(pair.converted);
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
const api={filePairName,groupFiles,prepareFiles,stats};
if(typeof module!=="undefined"&&module.exports)module.exports=api;
root.PocketTeacherPairs=api;
})(typeof window!=="undefined"?window:globalThis);
