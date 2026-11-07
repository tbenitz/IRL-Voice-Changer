/* Put the real full LLVC builder first. Preserve saved Pocket voices and
 * leave the old research UI in the DOM, but don't ask users to generate
 * pointless independently rendered pairs for a working real-time model.
 */
(() => {
"use strict";
function install(){
 const legacy=document.getElementById("pocket-trainer");
 if(!legacy||document.getElementById("real-neural-training"))return;
 const panel=document.createElement("section");
 panel.id="real-neural-training";
 panel.style.cssText="background:linear-gradient(130deg,#102a3a,#141c31);border:1px solid #4b9a9d;border-radius:17px;margin:25px 0;padding:24px;color:#edfaff;font:14px/1.6 system-ui,sans-serif";
 const colab="https://colab.research.google.com/github/tbenitz/IRL-Voice-Changer/blob/main/voice-training/train_real_voice.ipynb";
 panel.innerHTML=[
  '<span style="color:#6de4d4;font-size:11px;font-weight:900;letter-spacing:.15em">FULL NEURAL MODEL TRAINING</span>',
  '<h2 style="font-size:clamp(23px,3vw,30px);margin:8px 0">Make your own trained ONNX voice</h2>',
  '<p style="max-width:800px;color:#d0e1e8">Upload one authorized reference voice in the training notebook. It automatically runs a real voice-conversion teacher over source recordings, trains the full streaming LLVC neural network, then exports and tests an ONNX file compatible with Fast Neural. <strong>No manual matching pairs, no 65 KB Pocket student.</strong></p>',
  '<p style="color:#bddee4">A cloud NVIDIA GPU is necessary for training this architecture; the Chromebook will use the exported voice, not train it. Google Colab GPU access is subject to availability. ONNX execution is checked, but matching the selected speaker and real-time speed still require listening and measurement.</p>',
  '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:18px 0">',
  '<a href="'+colab+'" target="_blank" rel="noopener noreferrer" style="display:inline-flex;align-items:center;justify-content:center;padding:13px 19px;border-radius:10px;background:#6ce6d7;color:#08272d;text-decoration:none;font-weight:850">Train full custom ONNX on GPU</a>',
  '<a href="../../live-vc/fast-voice.html" style="display:inline-flex;align-items:center;justify-content:center;padding:13px 19px;border-radius:10px;border:1px solid #5faab1;color:#e4fcff;text-decoration:none;font-weight:750">Test downloaded ONNX in Fast Neural</a></div>',
  '<button type="button" id="trainer-legacy-toggle" style="background:transparent;border:0;text-decoration:underline;color:#accbd7;cursor:pointer;font:12px system-ui;padding:8px 0">Show archived Pocket dataset experiments (not a custom neural voice trainer)</button>'
 ].join("");
 legacy.parentNode.insertBefore(panel,legacy);
 legacy.hidden=true;
 document.getElementById("trainer-legacy-toggle").addEventListener("click",()=>{
  legacy.hidden=!legacy.hidden;
  document.getElementById("trainer-legacy-toggle").textContent=legacy.hidden
    ?"Show archived Pocket dataset experiments (not a custom neural voice trainer)"
    :"Hide archived Pocket research";
 });
 document.querySelectorAll('a[href="#pocket-trainer"]').forEach(a=>{
  a.href="#real-neural-training";
 });
 if(location.hash==="#pocket-trainer"){
  history.replaceState(null,"",location.pathname+location.search+"#real-neural-training");
  panel.scrollIntoView({block:"start"});
 }
}
if(document.readyState==="complete")install();
else addEventListener("load",install,{once:true});
})();
