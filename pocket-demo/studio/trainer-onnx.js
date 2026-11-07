/* Tiny streaming ONNX exporter. 100% local, no Python / conversion server.
 * Converts the SAME trained TFJS causal Conv1D layer weights to ONNX Conv.
 * ONNX input: audio_in [1,1,samples] -> output audio_out [1,1,samples].
 * Source model's quality depends heavily on aligned training data.
 */
(function(root){
"use strict";
const encoder=new TextEncoder();
function pack(chunks){let size=0;for(const c of chunks)size+=c.length;const data=new Uint8Array(size);let p=0;for(const c of chunks){data.set(c,p);p+=c.length}return data}
function v64(value){let x=BigInt(value);const a=[];while(x>127n){a.push(Number((x&127n)|128n));x>>=7n}a.push(Number(x));return Uint8Array.from(a)}
function tag(f,wire){return v64(BigInt(f*8+wire))}
function vi(f,value){return pack([tag(f,0),v64(value)])}
function bytes(f,val){const x=typeof val==="string"?encoder.encode(val):val;return pack([tag(f,2),v64(x.length),x])}
function txt(f,val){return bytes(f,val)}
function child(f,items){return bytes(f,pack(items))}
function dimension(dim){return typeof dim==="string"?child(1,[txt(2,dim)]):child(1,[vi(1,dim)])}
function valueInfo(name,shape){
 const sh=child(2,shape.map(dimension));
 const tensor=child(1,[vi(1,1),sh]);
 return child(11,[txt(1,name),child(2,[tensor])]);
}
function tensor(name,dims,data){
 const raw=new Uint8Array(data.byteLength);
 raw.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
 return child(5,[...dims.map(x=>vi(1,x)),vi(2,1),txt(8,name),bytes(9,raw)]);
}
function attrPads(left){return child(5,[txt(1,"pads"),vi(8,left),vi(8,0),vi(20,7)])}
function node(type,ins,outs,pads=null){
 return child(1,[...ins.map(x=>txt(1,x)),...outs.map(x=>txt(2,x)),txt(4,type),...(pads===null?[]:[attrPads(pads)])]);
}
function convWeights(w){
 // TFJS Conv1D kernel [kernel, input_ch, output_ch] to ONNX
 // Conv1D kernel [output_ch, input_ch, kernel].
 const [k,ic,oc]=w.shape;
 const out=new Float32Array(k*ic*oc);
 for(let o=0;o<oc;o++)
  for(let i=0;i<ic;i++)
   for(let t=0;t<k;t++)out[(o*ic+i)*k+t]=w.values[(t*ic+i)*oc+o];
 return {dims:[oc,ic,k],values:out,k};
}
async function exportCausalOnnx(tfModel,metadata={}){
 const weights=[];
 for(let i=1;i<=3;i++){
  const layer=tfModel.getLayer("pocket_conv"+i);
  if(!layer)throw Error("Training model is missing pocket_conv"+i);
  const raw=layer.getWeights();
  if(raw.length!==2)throw Error("Expected kernel + bias in pocket_conv"+i);
  const [kernel,bias]=await Promise.all(raw.map(async t=>({shape:t.shape,values:Float32Array.from(await t.data())})));
  raw.forEach(t=>t.dispose());
  if(kernel.shape.length!==3||bias.shape.length!==1)throw Error("Unsupported trained Conv1D shape");
  weights.push({kernel:convWeights(kernel),bias});
 }
 const g=[txt(2,"pocket-streaming-timbre-experiment")];
 const graphNodes=[
  node("Conv",["audio_in","W1","B1"],["c1"],weights[0].kernel.k-1),
  node("Tanh",["c1"],["a1"]),
  node("Conv",["a1","W2","B2"],["c2"],weights[1].kernel.k-1),
  node("Tanh",["c2"],["a2"]),
  node("Conv",["a2","W3","B3"],["c3"],0),
  node("Add",["c3","audio_in"],["residual"]),
  node("Tanh",["residual"],["audio_out"])
 ];
 g.push(...graphNodes);
 for(let i=0;i<weights.length;i++){
  g.push(tensor("W"+(i+1),weights[i].kernel.dims,weights[i].kernel.values));
  g.push(tensor("B"+(i+1),weights[i].bias.shape,weights[i].bias.values));
 }
 const type=(field,name,shape)=>{
  const sh=child(2,shape.map(dimension));
  const t=child(1,[vi(1,1),sh]);
  return child(field,[txt(1,name),child(2,[t])]);
 };
 g.push(type(11,"audio_in",[1,1,"samples"]));
 g.push(type(12,"audio_out",[1,1,"samples"]));
 const graph=child(7,g);
 const model=pack([
  vi(1,8),
  txt(2,"Pocket browser training prototype"),
  txt(3,"1.0"),
  txt(5,"Experimental student model trained on synthetic speech pairs; voice quality not certified."),
  graph,
  child(8,[vi(2,17)])
 ]);
 return new Blob([model],{type:"application/octet-stream"});
}
const api={exportCausalOnnx};
if(typeof module!=="undefined"&&module.exports)module.exports=api;
root.PocketTrainerONNX=api;
})(typeof window!=="undefined"?window:globalThis);
