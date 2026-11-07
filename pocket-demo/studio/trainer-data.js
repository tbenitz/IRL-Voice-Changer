/* Local training data: 16kHz PCM, resumable IndexedDB pairs and ZIP export.
 * Files never leave the browser. Alignment is only coarse duration alignment:
 * synthetic Pocket TTS targets are NOT waveform-aligned RVC teacher outputs.
 */
(function(root){
"use strict";
const SR=16000, DB="pocket-streaming-student-v1";
const STORE="pairs",META="meta";
function openDB(){
 return new Promise((ok,fail)=>{
  const r=indexedDB.open(DB,1);
  r.onupgradeneeded=()=>{
   if(!r.result.objectStoreNames.contains(STORE))r.result.createObjectStore(STORE,{keyPath:"id"});
   if(!r.result.objectStoreNames.contains(META))r.result.createObjectStore(META,{keyPath:"key"});
  };
  r.onsuccess=()=>ok(r.result);
  r.onerror=()=>fail(r.error||Error("Trainer IndexedDB failed"));
 });
}
async function query(store,mode,fn){
 const db=await openDB();
 try{
  return await new Promise((resolve,reject)=>{
   const tx=db.transaction(store,mode);
   let value=null;
   const req=fn(tx.objectStore(store));
   if(req){req.onsuccess=()=>value=req.result;req.onerror=()=>reject(req.error)}
   tx.oncomplete=()=>resolve(value);
   tx.onerror=()=>reject(tx.error||Error("Trainer storage operation failed"));
   tx.onabort=()=>reject(tx.error||Error("Trainer storage aborted"));
  });
 }finally{db.close()}
}
const getMeta=key=>query(META,"readonly",s=>s.get(key));
const putMeta=(key,value)=>query(META,"readwrite",s=>s.put({key,value}));
const savePair=obj=>query(STORE,"readwrite",s=>s.put(obj));
const getPair=id=>query(STORE,"readonly",s=>s.get(id));
const listPairs=()=>query(STORE,"readonly",s=>s.getAll());
async function clearPairs(){
 const db=await openDB();
 try{await new Promise((resolve,reject)=>{
  const tx=db.transaction([STORE,META],"readwrite");
  tx.objectStore(STORE).clear();
  tx.objectStore(META).clear();
  tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);
 })}finally{db.close()}
}
function resample(pcm,inputRate=24000,outputRate=SR){
 const length=Math.round(pcm.length*outputRate/inputRate);
 if(length>SR*180)throw Error("Single Pocket phrase exceeded 3 min limit");
 const out=new Float32Array(length),ratio=inputRate/outputRate;
 for(let i=0;i<length;i++){
  const p=i*ratio,a=Math.floor(p),f=p-a;
  out[i]=(pcm[Math.min(a,pcm.length-1)]||0)*(1-f)+
   (pcm[Math.min(a+1,pcm.length-1)]||0)*f;
 }
 return out;
}
function trimSilence(signal,rate=SR){
 // Avoid cutting words: 80ms padding around active samples, threshold -47dB.
 const stride=Math.max(1,Math.round(rate*.01));
 let min=signal.length,max=0;
 for(let i=0;i<signal.length;i+=stride){
  let power=0;
  for(let j=i;j<Math.min(signal.length,i+stride);j++)power+=signal[j]*signal[j];
  const rms=Math.sqrt(power/stride);
  if(rms>.0045){min=Math.min(i,min);max=Math.max(max,i+stride)}
 }
 if(min>=max)return signal;
 const pad=Math.round(rate*.08);
 return signal.slice(Math.max(0,min-pad),Math.min(signal.length,max+pad));
}
function stretch(source,length){
 if(source.length===length)return source;
 const out=new Float32Array(length);
 if(!source.length)return out;
 const ratio=(source.length-1)/Math.max(1,length-1);
 for(let i=0;i<length;i++){
  const p=i*ratio,a=Math.floor(p),f=p-a;
  out[i]=source[a]*(1-f)+(source[Math.min(source.length-1,a+1)]||0)*f;
 }
 return out;
}
function wav(pcm,sr=SR){
 const n=pcm.length,b=new ArrayBuffer(44+n*2),v=new DataView(b);
 const id=(offset,x)=>{for(let i=0;i<x.length;i++)v.setUint8(offset+i,x.charCodeAt(i))};
 id(0,"RIFF");v.setUint32(4,36+n*2,true);
 id(8,"WAVE");id(12,"fmt ");v.setUint32(16,16,true);
 v.setUint16(20,1,true);v.setUint16(22,1,true);
 v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);
 v.setUint16(32,2,true);v.setUint16(34,16,true);id(36,"data");
 v.setUint32(40,n*2,true);
 for(let i=0;i<n;i++){const a=Math.max(-1,Math.min(1,pcm[i]));v.setInt16(44+2*i,Math.round(a<0?a*32768:a*32767),true)}
 return new Blob([b],{type:"audio/wav"});
}
async function decodeWav(blob){
 const bytes=await blob.arrayBuffer(),v=new DataView(bytes);
 const read=o=>String.fromCharCode(...new Uint8Array(bytes,o,4));
 if(bytes.byteLength<44||read(0)!=="RIFF"||read(8)!=="WAVE")
  throw Error("Training clip is not a WAV file");
 let pos=12,fmt=null,dataStart=-1,dataLen=0;
 while(pos+8<=bytes.byteLength){
  const key=read(pos),n=v.getUint32(pos+4,true),start=pos+8;
  if(key==="fmt "){
   fmt={format:v.getUint16(start,true),channels:v.getUint16(start+2,true),
    rate:v.getUint32(start+4,true),bits:v.getUint16(start+14,true)};
  }
  if(key==="data"){dataStart=start;dataLen=Math.min(n,bytes.byteLength-start);break}
  pos=start+n+(n%2);
 }
 if(!fmt||fmt.format!==1||fmt.channels!==1||fmt.bits!==16||fmt.rate!==SR||dataStart<0)
  throw Error("Training clip must be 16kHz mono PCM16 WAV");
 const out=new Float32Array(Math.floor(dataLen/2));
 for(let i=0;i<out.length;i++)out[i]=v.getInt16(dataStart+2*i,true)/32768;
 return out;
}
const crcTable=(()=>{const t=new Uint32Array(256);for(let i=0;i<256;i++){let c=i;for(let j=0;j<8;j++)c=(c>>>1)^(c&1?0xedb88320:0);t[i]=c>>>0}return t})();
function crc32(bytes){let c=0xffffffff;for(const x of bytes)c=(c>>>8)^crcTable[(c^x)&255];return (c^0xffffffff)>>>0}
function header(length){return new Uint8Array(length)}
function u16(buf,i,v){buf[i]=v&255;buf[i+1]=(v>>>8)&255}
function u32(buf,i,v){u16(buf,i,v&65535);u16(buf,i+2,Math.floor(v/65536))}
async function buildZip(entries,onProgress=()=>{}){
 const parts=[],central=[];let offset=0,index=0;
 const enc=new TextEncoder();
 for(const entry of entries){
  const name=enc.encode(entry.name);
  const data=new Uint8Array(await entry.blob.arrayBuffer());
  const crc=crc32(data),n=data.length;
  if(n>0xffffffff||offset>0xffffffff)throw Error("Archive too large for browser ZIP32");
  const loc=header(30+name.length);
  u32(loc,0,0x04034b50);u16(loc,4,20);u16(loc,6,0);u16(loc,8,0);
  u32(loc,14,crc);u32(loc,18,n);u32(loc,22,n);u16(loc,26,name.length);
  loc.set(name,30);
  parts.push(loc,data);
  const c=header(46+name.length);
  u32(c,0,0x02014b50);u16(c,4,20);u16(c,6,20);
  u32(c,16,crc);u32(c,20,n);u32(c,24,n);
  u16(c,28,name.length);u32(c,42,offset);c.set(name,46);
  central.push(c);
  offset+=loc.length+n;
  onProgress(++index,entries.length);
 }
 const centralOffset=offset;
 let centralSize=0;for(const c of central){parts.push(c);centralSize+=c.length}
 const end=header(22);
 u32(end,0,0x06054b50);u16(end,8,entries.length);u16(end,10,entries.length);
 u32(end,12,centralSize);u32(end,16,centralOffset);
 parts.push(end);
 return new Blob(parts,{type:"application/zip"});
}
const api={SR,DB,resample,trimSilence,stretch,wav,decodeWav,
 getMeta,putMeta,savePair,getPair,listPairs,clearPairs,buildZip};
if(typeof module!=="undefined"&&module.exports)module.exports=api;
root.PocketTrainerData=api;
})(typeof window!=="undefined"?window:globalThis);
