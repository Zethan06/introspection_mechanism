"""HTML renderer for lazily loaded, position-averaged attention tensors."""

from __future__ import annotations

import json


def make_token_attention_browser(metadata: dict) -> str:
    """Return an interactive browser with no attention tensor embedded."""
    data_json = json.dumps(metadata, ensure_ascii=False).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token Attention Browser</title>
<style>
:root{--bg:#f6f8fb;--panel:#fff;--ink:#24272d;--muted:#697386;--line:#d9e0ea;--blue:#5f86f2;--orange:#e8542f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1550px;margin:auto;padding:18px 20px 28px}.titlebar{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}h1{margin:0 0 4px;font-size:22px}.subtitle{color:var(--muted);font-size:13px;margin-bottom:12px}
.export{flex:0 0 auto;border:1px solid #496fd5;border-radius:8px;background:var(--blue);color:#fff;padding:8px 13px;font:600 13px/1.2 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;cursor:pointer}.export:hover{background:#496fd5}.export:focus-visible{outline:3px solid #5f86f255;outline-offset:2px}.export:disabled{cursor:wait;opacity:.55}
.panel{background:var(--panel)}
.controls{display:grid;grid-template-columns:repeat(10,minmax(100px,1fr));gap:10px;padding:12px}
label{display:block;margin:0 0 4px;color:var(--muted);font-size:12px}select,input[type=range],input[type=number]{width:100%}select,input[type=number]{height:34px;border:1px solid var(--line);border-radius:7px;background:#fff;padding:0 8px}
.readout{min-height:34px;display:flex;align-items:center;font-size:13px;color:var(--muted)}
.browser{width:min(calc(100% - 24px),calc(var(--chars-per-line,96) * 1ch + 60px));margin:12px auto;height:min(72vh,820px);overflow:auto;background:#fff;padding:28px 30px;font:24px/1.78 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.tok{position:relative;display:inline;border-radius:6px;padding:1px 2px;margin:0;background:#ffffffb3;cursor:pointer}
.tok:hover{background:#eef2f8!important}.tok.query{background:#dce7ff!important}
.tok[hidden]{display:none}.tok.choice:not(.query),.tok.target:not(.query){box-shadow:none}
.tok.newline{color:#9a3412;font-weight:650}.tok.topkey::after{content:"";display:inline-block;width:.34em;height:.34em;margin:0 .05em 0 .1em;border-radius:50%;background:var(--orange)}
.footer{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 12px 12px}.mini{padding:10px;background:#fbfcfe;min-height:86px}.mini h2{font-size:13px;margin:0 0 7px;color:var(--muted)}
.topList{display:flex;flex-wrap:wrap;gap:6px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.pill{background:#fff;border-radius:999px;padding:4px 7px}
.legend{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px;padding:0 14px 14px}.grad{height:14px;width:210px;border-radius:999px;background:linear-gradient(90deg,#fff0,#fa9e7973,#e8542fe0)}.grad.delta{background:linear-gradient(90deg,#2563b8,#ffffffe6,#e8542fe0)}
@media(max-width:980px){.controls{grid-template-columns:repeat(2,minmax(120px,1fr))}.footer{grid-template-columns:1fr}.browser{font-size:18px;padding:18px}}
@page{size:A4 landscape;margin:0}
@media print{
  *{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}
  body{background:#fff}main{max-width:none;padding:0}.titlebar,.controls,.footer,.legend{display:none}
  .panel{break-inside:avoid}.browser{width:min(100%,calc(var(--chars-per-line,96) * 1ch + 20mm));height:auto;max-height:none;overflow:visible;margin:0 auto;padding:10mm;font-size:12px;line-height:1.55}
  .tok.topkey::after{display:none}
}
</style>
</head>
<body><main>
<div class="titlebar"><div><h1>Token Attention Browser</h1><div class="subtitle" id="subtitle"></div></div><button class="export" id="exportPdf" type="button" disabled>Export PDF</button></div>
<section class="panel">
<div class="controls">
<div><label for="layer">Layer</label><select id="layer"></select></div>
<div><label for="head">Head</label><select id="head"></select></div>
<div><label for="position">Injection Position</label><select id="position"></select></div>
<div><label for="mode">Attention Source</label><select id="mode"><option value="injected">perturbed</option><option value="clean">clean</option><option value="delta">effect (perturbed - clean)</option><option value="absdelta">abs delta</option></select></div>
<div><label for="query">Query Token</label><select id="query"></select></div>
<div><label for="rangeStart">Export First Token</label><select id="rangeStart"></select></div>
<div><label for="rangeEnd">Export Last Token</label><select id="rangeEnd"></select></div>
<div><label for="lineChars">PDF Characters / Line</label><input id="lineChars" type="number" min="20" max="140" step="1" value="96"></div>
<div><label for="topk">Top key dots</label><input id="topk" type="range" min="0" max="20" step="1" value="8"></div>
<div class="readout" id="readout"></div>
</div>
<div class="browser" id="browser"></div>
<div class="footer"><div class="mini"><h2>Selected Query</h2><div id="queryInfo"></div></div><div class="mini"><h2>Top Attended Key Tokens</h2><div class="topList" id="topKeys"></div></div></div>
<div class="legend"><span>key attention</span><span class="grad" id="grad"></span><span>blue fill = query</span><span>orange dot = top key</span></div>
</section></main>
<script>
const meta=__DATA_JSON__;
const seqLen=meta.seq_len,heads=meta.num_heads,layers=meta.layers;
const payloadFiles=meta.effect_payload_files,scales=meta.effect_scales;
const tokenByIdx=new Map(meta.tokens.map(t=>[t.idx,t]));
const $=id=>document.getElementById(id);
const layerSel=$("layer"),headSel=$("head"),posSel=$("position"),modeSel=$("mode"),querySel=$("query"),rangeStartSel=$("rangeStart"),rangeEndSel=$("rangeEnd"),lineChars=$("lineChars"),topk=$("topk"),browser=$("browser"),readout=$("readout"),topKeys=$("topKeys"),queryInfo=$("queryInfo"),grad=$("grad"),exportPdf=$("exportPdf");
let clean=null,effect=null,cleanPromise=null,loadSerial=0;const tokenEls=[];
for(const layer of layers){const o=document.createElement("option");o.value=layers.indexOf(layer);o.textContent="L"+layer;layerSel.appendChild(o)}
for(let h=0;h<heads;h++){const o=document.createElement("option");o.value=h;o.textContent="H"+h;headSel.appendChild(o)}
for(const p of meta.available_target_positions){const o=document.createElement("option");o.value=p;o.textContent="TOKEN "+p;posSel.appendChild(o)}
for(const t of meta.tokens){for(const select of[querySel,rangeStartSel,rangeEndSel]){const o=document.createElement("option");o.value=t.idx;o.textContent=`${t.idx}: ${t.repr}${t.choice!==null?" [TOKEN "+t.choice+"]":""}`;select.appendChild(o)}}
layerSel.value=String(Math.max(0,layers.indexOf(meta.injection_layer)));headSel.value="0";posSel.value=String(meta.target_position);
rangeStartSel.value=String(meta.tokens[0].idx);rangeEndSel.value=String(meta.tokens[meta.tokens.length-1].idx);
const selectedPosition=()=>Number(posSel.value);
const targetIndex=()=>Number(meta.target_token_indices[String(selectedPosition())]);
const targetText=()=>meta.representative_target_tokens[String(selectedPosition())];
function subtitle(){const s=meta.position_summaries[String(selectedPosition())],label=meta.intervention_label||meta.concept,action=meta.intervention_action||"injected",count=meta.num_interventions??meta.num_concepts,countLabel=meta.intervention_count_label||"concept(s)",layerLabel=meta.perturbation_layer_label||meta.injection_layer;$("subtitle").textContent=`${label} ${action} at TOKEN ${selectedPosition()} (${targetText()} in the representative prompt), layer ${layerLabel}; averaged over ${count} ${countLabel} × ${meta.num_clusters} examples. Clean accuracy ${(100*s.clean_accuracy).toFixed(1)}%; perturbed accuracy ${(100*s.injected_accuracy).toFixed(1)}%.`}
function b64bytes(b64){const bin=atob(b64),out=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)out[i]=bin.charCodeAt(i);return out}
async function gunzip(b64,Type){if(typeof DecompressionStream==="undefined")throw new Error("Browser does not support gzip DecompressionStream");const stream=new Blob([b64bytes(b64)]).stream().pipeThrough(new DecompressionStream("gzip"));return new Type(await new Response(stream).arrayBuffer())}
async function loadArray(file,key,Type){const script=document.createElement("script");script.src=file;const done=new Promise((ok,bad)=>{script.onload=ok;script.onerror=()=>bad(new Error("Could not load "+file))});document.head.appendChild(script);try{await done;const all=window.__ATTENTION_COMPACT_PAYLOADS__||{},encoded=all[key];if(!encoded)throw new Error("Missing payload "+key);delete all[key];return await gunzip(encoded,Type)}finally{script.remove()}}
function offset(l,h,q,k){return(((l*heads+h)*seqLen+q)*seqLen+k)}
function cleanValue(l,h,q,k){return clean[offset(l,h,q,k)]/65535}
function effectValue(l,h,q,k){return effect[offset(l,h,q,k)]/32767*Number(scales[String(selectedPosition())])}
function values(l,h,q,mode){const out=[];for(let k=0;k<seqLen;k++){const c=cleanValue(l,h,q,k),d=effectValue(l,h,q,k),inj=Math.max(0,Math.min(1,c+d));out.push(mode==="clean"?c:mode==="injected"?inj:mode==="delta"?d:Math.abs(d))}return out}
function visible(text){return String(text).replace(/\n/g,"\\n\n")}
function createTokens(){browser.textContent="";for(const t of meta.tokens){const s=document.createElement("span");s.className="tok";s.textContent=visible(t.text);s.title=`${t.idx} | id ${t.id} | ${t.repr}`;s.dataset.idx=t.idx;if(t.choice!==null)s.classList.add("choice");if(String(t.text).includes("\n"))s.classList.add("newline");s.onclick=()=>{querySel.value=String(t.idx);draw(true)};tokenEls.push(s);browser.appendChild(s)}updateTarget()}
function updateTarget(){for(const s of tokenEls)s.classList.toggle("target",Number(s.dataset.idx)===targetIndex())}
function rangeBounds(){return[Number(rangeStartSel.value),Number(rangeEndSel.value)]}
function updateVisibleRange(changed){let[start,end]=rangeBounds();if(start>end){if(changed===rangeStartSel){end=start;rangeEndSel.value=String(end)}else{start=end;rangeStartSel.value=String(start)}}for(const s of tokenEls){const i=Number(s.dataset.idx);s.hidden=i<start||i>end}}
function updateLineWidth(){const n=Number(lineChars.value);if(Number.isFinite(n)&&n>0)browser.style.setProperty("--chars-per-line",String(n))}
function normalizeLineWidth(){const n=Math.max(20,Math.min(140,Number(lineChars.value)||96));lineChars.value=String(n);browser.style.setProperty("--chars-per-line",String(n))}
function color(v,max,delta){const t=Math.max(-1,Math.min(1,v/Math.max(max,1e-12)));if(delta&&t<0)return`rgba(37,99,235,${.08+.62*Math.abs(t)})`;return`rgba(232,84,47,${.05+.78*Math.abs(t)})`}
function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function draw(scroll=false){if(!clean||!effect)return;const l=Number(layerSel.value),h=Number(headSel.value),q=Number(querySel.value),mode=modeSel.value,v=values(l,h,q,mode),[start,end]=rangeBounds(),ranked=v.map((value,idx)=>({value,idx})).filter(x=>x.idx>=start&&x.idx<=end).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)),dots=new Set(ranked.slice(0,Number(topk.value)).map(x=>x.idx)),max=Math.max(...ranked.map(x=>Math.abs(x.value)),1e-12);for(const s of tokenEls){const i=Number(s.dataset.idx);s.classList.toggle("query",i===q);s.classList.toggle("topkey",dots.has(i));s.style.backgroundColor=color(v[i],max,mode==="delta")}const qt=tokenByIdx.get(q);readout.textContent=`L${layers[l]} H${h} | keys ${start}–${end} | top dots ${topk.value}`;queryInfo.innerHTML=`<div class="topList"><span class="pill">idx ${q}</span><span class="pill">id ${qt.id}</span><span class="pill">${escapeHtml(qt.repr)}</span></div>`;topKeys.textContent="";for(const x of ranked.slice(0,Math.max(1,Number(topk.value)))){const p=document.createElement("span");p.className="pill";p.textContent=`${x.idx} ${tokenByIdx.get(x.idx).repr}: ${x.value.toExponential(2)}`;topKeys.appendChild(p)}grad.classList.toggle("delta",mode==="delta");exportPdf.disabled=false;if(scroll){const queryToken=tokenEls.find(s=>Number(s.dataset.idx)===q);if(queryToken&&!queryToken.hidden)queryToken.scrollIntoView({block:"center",inline:"center",behavior:"smooth"})}}
function exportCurrentView(){if(exportPdf.disabled)return;normalizeLineWidth();const oldTitle=document.title,l=layers[Number(layerSel.value)],h=Number(headSel.value),p=selectedPosition(),mode=modeSel.value,q=Number(querySel.value),[start,end]=rangeBounds();document.title=`token_attention_L${l}_H${h}_P${p}_${mode}_Q${q}_K${start}-${end}_C${lineChars.value}`;const restore=()=>{document.title=oldTitle;window.removeEventListener("afterprint",restore)};window.addEventListener("afterprint",restore);window.print();setTimeout(restore,1000)}
async function loadPosition(scroll=true){const p=selectedPosition(),serial=++loadSerial;readout.textContent=`loading compressed TOKEN ${p} effect...`;try{if(!cleanPromise)cleanPromise=loadArray(meta.clean_payload_file,"clean",Uint16Array).then(x=>(clean=x));const [c,e]=await Promise.all([cleanPromise,loadArray(payloadFiles[String(p)],`effect_${p}`,Int16Array)]);if(serial!==loadSerial)return;clean=c;effect=e;querySel.value=String(targetIndex());updateTarget();subtitle();draw(scroll)}catch(err){readout.textContent=String(err)}}
for(const el of[layerSel,headSel,modeSel,querySel,topk])el.addEventListener("input",()=>draw(el===querySel));
for(const el of[rangeStartSel,rangeEndSel])el.addEventListener("input",()=>{updateVisibleRange(el);draw(false)});
lineChars.addEventListener("input",updateLineWidth);
lineChars.addEventListener("change",normalizeLineWidth);
posSel.addEventListener("input",()=>loadPosition(true));createTokens();updateVisibleRange();normalizeLineWidth();subtitle();loadPosition(true);
exportPdf.addEventListener("click",exportCurrentView);
</script></body></html>"""
    return template.replace("__DATA_JSON__", data_json)
