const chat = document.querySelector("#chat");
const form = document.querySelector("#composer");
const question = document.querySelector("#question");
const send = document.querySelector("#send");
const errorBox = document.querySelector("#error");
const documents = document.querySelector("#documents");

function showError(message){ errorBox.textContent = message; errorBox.hidden = false; }
function clearError(){ errorBox.hidden = true; errorBox.textContent = ""; }
function addMessage(text, role, sources=[]){
  const row=document.createElement("div"); row.className=`message ${role}`;
  row.innerHTML=`<div class="avatar">${role==="user"?"U":"L"}</div><div class="bubble"></div>`;
  const bubble=row.querySelector(".bubble"); const body=document.createElement("div");
  body.textContent=text; bubble.appendChild(body);
  if(sources.length){const list=document.createElement("div");list.className="sources";list.innerHTML="<strong>Retrieved sources</strong>";
    sources.forEach(s=>{const item=document.createElement("div");item.className="source";item.innerHTML=`📄 <strong>${escapeHtml(s.source)}</strong> · ${s.page?`PDF page ${s.page}`:`text chunk ${escapeHtml(s.chunk_id)}`} <span class="score">retrieval ${s.score.toFixed(3)}</span><br><span>${escapeHtml(s.claim_supported||"Passage-level source")}</span><br><em>${escapeHtml(s.excerpt||"")}</em>${s.source_url?`<br><a href="${escapeHtml(s.source_url)}" target="_blank" rel="noreferrer">Official source</a> · verified ${escapeHtml(s.verified_date||"")}`:""}`;list.appendChild(item)});bubble.appendChild(list)}
  chat.appendChild(row); row.scrollIntoView({behavior:"smooth",block:"nearest"});
}
function intentLabel(intent){return String(intent||"general").replaceAll("_"," ")}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
async function loadDocuments(){
  try{const r=await fetch("/api/documents");if(!r.ok)throw new Error("Backend did not respond");const data=await r.json();
    documents.innerHTML=data.documents.length?data.documents.map(d=>`<div class="doc"><span class="doc-name" title="${escapeHtml(d.name)}">📄 ${escapeHtml(d.name)}</span><span class="${d.status==="warning"?"doc-warning":"doc-status"}"><span class="status-dot"></span>${d.status} · ${d.chunks} chunks${d.pages?` · ${d.pages} pages`:""}</span>${d.message?`<div class="doc-meta">${escapeHtml(d.message)}</div>`:""}</div>`).join(""):'<p class="muted">No PDF or TXT files found.</p>';
  }catch(e){documents.innerHTML='<p class="muted">Could not load document status.</p>'}
}
async function submit(text){text=text.trim();if(!text)return;clearError();addMessage(text,"user");question.value="";send.disabled=true;send.innerHTML="Thinking…";
  const typing=document.createElement("div");typing.className="message assistant typing";typing.innerHTML='<div class="avatar">L</div><div class="bubble">Retrieving sources and asking local Ollama…</div>';chat.appendChild(typing);
  try{const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:text})});const data=await r.json();typing.remove();if(!r.ok)throw new Error(data.detail||"The request failed.");addMessage(`Intent: ${intentLabel(data.intent)}\n\n${data.answer}`,"assistant",data.sources||[])}
  catch(e){typing.remove();showError(e.message);addMessage("I could not complete that request. See the error message below.","assistant")}
  finally{send.disabled=false;send.innerHTML='Send <span>↗</span>';question.focus()}
}
form.addEventListener("submit",e=>{e.preventDefault();submit(question.value)});
question.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
document.querySelector("#clear").addEventListener("click",()=>{chat.innerHTML="";clearError()});
document.querySelector("#reindex").addEventListener("click",async()=>{clearError();try{const r=await fetch("/api/documents/reindex",{method:"POST"});if(!r.ok)throw new Error("Reindex failed");await loadDocuments()}catch(e){showError(e.message)}});
document.querySelectorAll(".samples button").forEach(button=>button.addEventListener("click",()=>submit(button.textContent)));
document.querySelector("#analyze-scam").addEventListener("click",async()=>{const text=document.querySelector("#scam-text").value.trim(),out=document.querySelector("#scam-result");if(!text){out.textContent="Paste a message first.";return}out.textContent="Analyzing locally…";try{const r=await fetch("/api/analyze-scam",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});const data=await r.json();out.className="tool-result";out.innerHTML=`<p><strong>${escapeHtml(data.assessment)}</strong></p>${data.signals.map(s=>`<div class="signal">${escapeHtml(s.strength)}: ${escapeHtml(s.explanation)}</div>`).join("")}<p>${data.safe_steps.map(escapeHtml).join(" · ")}</p>`}catch(e){out.textContent="Could not analyze the message."}});
document.querySelector("#save-incident").addEventListener("click",async()=>{const out=document.querySelector("#incident-result"),payload={incident_category:document.querySelector("#incident-category").value,amount:document.querySelector("#incident-amount").value,transaction_reference:document.querySelector("#incident-reference").value,notes:document.querySelector("#incident-notes").value};out.textContent="Saving locally…";try{const r=await fetch("/api/incidents",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const data=await r.json();if(!r.ok)throw new Error(data.detail);out.textContent=`Saved locally to ${data.local_file}.`}catch(e){out.textContent=e.message||"Could not save notes."}});
loadDocuments();
