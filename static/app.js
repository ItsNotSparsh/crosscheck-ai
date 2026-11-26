const form=document.querySelector('#analyze-form');
const fileInput=document.querySelector('#image-file');
const dropzone=document.querySelector('#dropzone');
const preview=document.querySelector('#image-preview');
const dropMain=document.querySelector('#drop-main');
const dropSub=document.querySelector('#drop-sub');
const removeButton=document.querySelector('#remove-image');
const imageModal=document.querySelector('#image-modal');
const imageModalContent=document.querySelector('#image-modal-content');
const imageModalClose=document.querySelector('#image-modal-close');
let previewUrl=null;
document.querySelector('#check-mode').addEventListener('change', event => {
  const mode=event.target.value;
  fileInput.required=['photo','dataset'].includes(mode);
  document.querySelector('#image-requirement').textContent=fileInput.required?'Required':'Optional for factual claims';
  document.querySelector('.evidence-option').hidden=['photo','article'].includes(mode);
  document.querySelector('#article-source').hidden=mode!=='article';
  const articleUrl=document.querySelector('#article-url');
  articleUrl.disabled=mode!=='article';
  articleUrl.required=mode==='article';
});

function setFile(file){
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){showError('Choose a JPG, PNG, or WEBP image.');fileInput.value='';return;}
  if(file.size>12*1024*1024){showError('That image is over 12 MB. Choose a smaller image.');fileInput.value='';return;}
  clearError();
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(file);preview.src=previewUrl;preview.hidden=false;
  dropzone.classList.add('has-image');dropMain.innerHTML=`<b>${escapeHtml(file.name)}</b>`;dropSub.textContent=`${(file.size/1024/1024).toFixed(2)} MB · click to replace`;
  removeButton.hidden=false;
}
function clearFile(){fileInput.value='';preview.removeAttribute('src');preview.hidden=true;dropzone.classList.remove('has-image');dropMain.innerHTML='Drop an image here or <b>browse files</b>';dropSub.textContent='A photo attached to the claim · up to 12 MB';removeButton.hidden=true;if(previewUrl){URL.revokeObjectURL(previewUrl);previewUrl=null;}}
function closeImageModal(){imageModal.hidden=true;imageModalContent.removeAttribute('src');}
preview.addEventListener('click',e=>{if(!previewUrl)return;e.preventDefault();e.stopPropagation();imageModalContent.src=previewUrl;imageModal.hidden=false;imageModalClose.focus();});
imageModalClose.addEventListener('click',closeImageModal);
imageModal.addEventListener('click',e=>{if(e.target===imageModal)closeImageModal();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!imageModal.hidden)closeImageModal();});
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function showError(message){document.querySelector('#error-message').textContent=message;document.querySelector('#error-state').hidden=false;document.querySelector('#empty-state').hidden=true;document.querySelector('#analysis-result').hidden=true;}
function clearError(){document.querySelector('#error-state').hidden=true;}
fileInput.addEventListener('change',()=>setFile(fileInput.files[0]));
removeButton.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();clearFile();});
for(const eventName of ['dragenter','dragover'])dropzone.addEventListener(eventName,e=>{e.preventDefault();dropzone.classList.add('drag-over');});
for(const eventName of ['dragleave','drop'])dropzone.addEventListener(eventName,e=>{e.preventDefault();dropzone.classList.remove('drag-over');});
dropzone.addEventListener('drop',e=>{const file=e.dataTransfer.files[0];if(file){const transfer=new DataTransfer();transfer.items.add(file);fileInput.files=transfer.files;setFile(file);}});
document.querySelector('#claim-text').addEventListener('input',e=>document.querySelector('#char-count').textContent=`${e.target.value.length} / 1000`);

function setLoading(loading){document.querySelector('#loading-state').hidden=!loading;if(loading){document.querySelector('#empty-state').hidden=true;document.querySelector('#analysis-result').hidden=true;document.querySelector('#error-state').hidden=true;document.querySelector('#result-title').textContent='Checking your claim';}const button=document.querySelector('#analyze-button');button.disabled=loading;button.querySelector('.button-label').textContent=loading?'Analyzing…':'Analyze claim';}
function renderResult(data){
  clearError();document.querySelector('#empty-state').hidden=true;document.querySelector('#analysis-result').hidden=false;document.querySelector('#result-title').textContent='Your screening result';
  document.querySelector('.probability-block').hidden=!data.probabilities.length;
  document.querySelector('.alignment-box').hidden=data.similarity==null;
  document.querySelector('.verdict-copy .section-kicker').textContent=({photo:'VISIBLE CAPTION CHECK',news:'WRITTEN CLAIM CHECK',dataset:'FAKEDDIT SCREENING',article:'AGREEMENT WITH LINKED ARTICLE'})[data.mode];
  const label=data.label;const verdict=document.querySelector('#verdict-label');const icon=document.querySelector('#verdict-icon');
  verdict.textContent=label;icon.className='verdict-icon';icon.textContent=label==='Fake'?'!':label==='Real'?'✓':'?';if(label==='Fake')icon.classList.add('fake');else if(label==='Real')icon.classList.add('real');
  document.querySelector('#verdict-note').textContent=data.decision_basis||'Source comparison required';
  document.querySelector('.confidence').hidden=true;document.querySelector('.confidence-meter').hidden=true;
  const probs=document.querySelector('#probability-list');probs.replaceChildren();for(const item of data.probabilities){const row=document.createElement('div');row.className='prob-row'+(item.label===label?' active':'')+(item.label==='Fake'?' fake':'')+(item.label==='Possibly misleading / uncertain'?' uncertain':'');const name=document.createElement('span');name.className='prob-name';name.textContent=item.label;const track=document.createElement('span');track.className='prob-track';const fill=document.createElement('i');fill.style.width=`${Math.round(item.value*100)}%`;track.append(fill);const value=document.createElement('span');value.className='prob-number';value.textContent=`${Math.round(item.value*100)}%`;row.append(name,track,value);probs.append(row);}
  const sim=data.similarity;document.querySelector('#similarity-value').textContent=sim==null?'':sim.toFixed(3);document.querySelector('#similarity-note').textContent=sim<.2?'Weak semantic match in this example; this is not proof of an image mismatch.':sim>.35?'Some semantic alignment; this does not confirm the image or claim.':'Intermediate semantic alignment; inspect the original context.';
  const reasons=document.querySelector('#reason-list');reasons.replaceChildren();for(const message of data.reasons){const li=document.createElement('li');li.textContent=message;reasons.append(li);}
  renderEvidence(data.evidence,data.evidence_check);
}
function renderEvidence(evidence,check){const panel=document.querySelector('#evidence-block');if(!evidence){panel.hidden=true;return;}panel.hidden=false;document.querySelector('#evidence-provider').textContent=evidence.provider||'SEARCH UNAVAILABLE';const checkText=check?` Evidence comparison: ${check.stance.replaceAll('_',' ')} across ${check.checked} snippets; this is a text-entailment cue, not a truth probability.`:'';document.querySelector('#evidence-message').textContent=(evidence.message||'')+checkText;const list=document.querySelector('#evidence-list');list.replaceChildren();for(const item of evidence.items||[]){try{const url=new URL(item.url);if(!['https:','http:'].includes(url.protocol))continue;const link=document.createElement('a');link.className='evidence-item';link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';const title=document.createElement('strong');title.textContent=item.title||'Open source';const snippet=document.createElement('span');snippet.textContent=item.snippet||'';const source=document.createElement('small');source.textContent=`${item.source||'Search result'}${item.date?' · '+item.date:''} ↗`;link.append(title,snippet,source);list.append(link);}catch(_){}}
}
form.addEventListener('submit',async e=>{e.preventDefault();clearError();const button=document.querySelector('#analyze-button');const data=new FormData(form);data.set('evidence',document.querySelector('#evidence-toggle').checked?'true':'false');setLoading(true);try{const response=await fetch('/api/analyze',{method:'POST',body:data});const payload=await response.json();if(!response.ok)throw new Error(payload.error||'The request could not be completed.');renderResult(payload);}catch(error){showError(error.message||'A connection error occurred. Check that the local project server is running.');document.querySelector('#result-title').textContent='Analysis needs attention';}finally{setLoading(false);button.disabled=false;button.querySelector('.button-label').textContent='Analyze claim';}});
