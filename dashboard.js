'use strict';
const names={chase:'Chase com ruído',chase_no_noise:'Chase sem ruído',striker:'Atacante posicional',keeper:'Goleiro',initial_policy:'Política inicial congelada'};
const colors={chase:'#63b3ff',chase_no_noise:'#70d9dc',striker:'#62d6a3',keeper:'#f0c36a',initial_policy:'#b49cff'};
const state={training:[],evaluations:[],live:{},visible:new Set(Object.keys(names)),connected:false};
const $=id=>document.getElementById(id),fmt=(n,d=0)=>Number.isFinite(n)?n.toLocaleString('pt-BR',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
const percent=n=>Number.isFinite(n)?fmt(n*100,1)+'%':'—';
function merge(target,records){const map=new Map(target.map(r=>[r.iteration,r]));records.forEach(r=>map.set(r.iteration,r));return [...map.values()].sort((a,b)=>a.iteration-b.iteration);}
function group(evaluation){const scenario=$('scenario').value;const result=[];for(const bot of Object.keys(names)){
 const rows=evaluation.cases.filter(r=>r.opponent===bot&&(scenario==='all'||r.scenario===scenario));
 const valid=rows.filter(r=>Number.isFinite(r.score));const sum=k=>rows.reduce((n,r)=>n+r[k],0);
 result.push({bot,score:valid.length?valid.reduce((n,r)=>n+r.score,0)/valid.length:null,gf:sum('gf'),ga:sum('ga'),wins:sum('wins'),draws:sum('draws'),losses:sum('losses'),stationary:rows.length?rows.reduce((n,r)=>n+r.stationary_fraction,0)/rows.length:null});}return result;}
function svgNode(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
const scaleCharts=['performance-chart','stationary-chart','reward-chart','ppo-chart'];
const chartDomains={};
let scalePrefs={};try{scalePrefs=JSON.parse(localStorage.getItem('haxball-x-windows')||'{}');}catch{}
const preference=id=>scalePrefs[id]||{window:'all'};
function saveScale(id,value){scalePrefs[id]=value;try{localStorage.setItem('haxball-x-windows',JSON.stringify(scalePrefs));}catch{}render();}
function control(id){
 const toolbar=document.createElement('div');toolbar.className='scale-controls';toolbar.dataset.chart=id;
 toolbar.innerHTML=`<label>Intervalo X <select data-field="window" aria-label="Intervalo X de ${id}"><option value="all">Toda a execução</option><option value="100">Últimas 100 iterações</option><option value="250">Últimas 250 iterações</option><option value="500">Últimas 500 iterações</option><option value="custom">Escolher início e fim</option></select></label><span class="custom-x" hidden><label>De <input data-field="from" type="number" min="0" step="1" aria-label="Iteração inicial de ${id}"></label><label>Até <input data-field="to" type="number" min="0" step="1" aria-label="Iteração final de ${id}"></label></span><span class="scale-error" role="status"></span>`;
 $(id).before(toolbar);
 const sync=()=>{const pref=preference(id);for(const field of toolbar.querySelectorAll('[data-field]')){if(document.activeElement!==field)field.value=pref[field.dataset.field]??'';}toolbar.querySelector('.custom-x').hidden=pref.window!=='custom';};
 const updateControl=event=>{const field=event.target.dataset.field;if(!field)return;const pref={...preference(id)};pref[field]=event.target.value;const domain=chartDomains[id];if(domain&&field==='window'&&pref.window==='custom'){pref.from??=String(domain.from);pref.to??=String(domain.to);}saveScale(id,pref);sync();};toolbar.addEventListener('change',updateControl);toolbar.addEventListener('input',event=>{if(event.target.tagName==='INPUT')updateControl(event);});sync();return sync;
}
const scaleSync={};
function chartBounds(id,series,defaults){
 const pref=preference(id),all=series.flatMap(s=>s.points),latest=Math.max(0,...all.map(p=>p.x));
 let xMin=null,xMax=null,error='';
 if(pref.window!=='all'&&pref.window!=='custom')xMin=Math.max(0,latest-Number(pref.window));
 if(pref.window==='custom'){
   if(pref.from!==''&&pref.from!==undefined)xMin=Number(pref.from);
   if(pref.to!==''&&pref.to!==undefined)xMax=Number(pref.to);
   if(xMin!==null&&xMax!==null&&xMin>=xMax){error='O fim deve ser maior que o início.';xMin=xMax=null;}
 }
 const filtered=series.map(s=>({...s,points:s.points.filter(p=>(xMin===null||p.x>=xMin)&&(xMax===null||p.x<=xMax))}));
 document.querySelector(`[data-chart="${id}"] .scale-error`).textContent=error;
 return {series:filtered,min:defaults.min,max:defaults.max,xMin,xMax};
}
function plot(id,series,{min=null,max=null,percentage=false}={}){
 const bounds=chartBounds(id,series,{min,max,percentage});series=bounds.series;min=bounds.min;max=bounds.max;
 const root=$(id);root.replaceChildren();const points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y));
 if(!points.length){const div=document.createElement('div');div.className='empty';div.textContent=series.some(s=>s.points.length)?'Sem valores medidos neste intervalo':'Sem dados neste intervalo';root.append(div);return;}
 const w=Math.max(root.clientWidth,280),h=260,m={l:56,r:20,t:12,b:40},xs=points.map(p=>p.x),ys=points.map(p=>p.y);
 const x0=bounds.xMin??Math.min(...xs),x1=bounds.xMax??Math.max(...xs,x0+1);let y0=min??Math.min(...ys,0),y1=max??Math.max(...ys);
 if(y0===y1)y1=y0+1;if(min===null&&max===null){const pad=(y1-y0)*.08;y0-=pad;y1+=pad;}
 chartDomains[id]={min:y0,max:y1,from:x0,to:x1,percentage};
 const x=v=>m.l+(v-x0)/(x1-x0)*(w-m.l-m.r),y=v=>h-m.b-(v-y0)/(y1-y0)*(h-m.b-m.t);
 const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':root.getAttribute('aria-label')});root.append(svg);
 const defs=svgNode('defs'),clip=svgNode('clipPath',{id:id+'-clip'});clip.append(svgNode('rect',{x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b}));defs.append(clip);svg.append(defs);
 const marks=svgNode('g',{'clip-path':`url(#${id}-clip)`});svg.append(marks);
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;svg.append(svgNode('line',{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:'grid'}),svgNode('text',{x:m.l-8,y:y(v)+4,'text-anchor':'end'},percentage?fmt(v*100,(y1-y0)<.05?1:0)+'%':fmt(v,(y1-y0)<.001?5:(y1-y0)<.01?4:(y1-y0)<.1?3:(y1-y0)<1?2:(y1-y0)<5?1:0)));}
 for(const v of [...new Set(Array.from({length:4},(_,i)=>Math.round(x0+(x1-x0)*i/3)))]){svg.append(svgNode('text',{x:x(v),y:h-20,'text-anchor':v===x0?'start':v===x1?'end':'middle'},fmt(v)));}
 svg.append(svgNode('text',{x:(w+m.l)/2,y:h-3,'text-anchor':'middle'},'Iteração'));
 for(const s of series){let d='',previous=false;for(const p of s.points){if(!Number.isFinite(p.y)){previous=false;continue;}d+=(previous?'L':'M')+x(p.x)+','+y(p.y);previous=true;}marks.append(svgNode('path',{d,stroke:s.color,class:'series'}));for(const p of s.points.filter(p=>Number.isFinite(p.y))){if(s.points.length<5)marks.append(svgNode('circle',{cx:x(p.x),cy:y(p.y),r:3,fill:s.color}));}}
 const tip=document.createElement('div');tip.className='tip';tip.hidden=true;root.append(tip);
 const hit=svgNode('rect',{x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b,fill:'transparent'});svg.append(hit);
 hit.addEventListener('pointermove',event=>{const rect=root.getBoundingClientRect(),px=event.clientX-rect.left;const target=x0+(px-m.l)/(w-m.l-m.r)*(x1-x0);const iteration=points.reduce((best,p)=>Math.abs(p.x-target)<Math.abs(best-target)?p.x:best,points[0].x);tip.replaceChildren();const title=document.createElement('strong');title.textContent='Iteração '+fmt(iteration);tip.append(title);for(const s of series){const p=s.points.find(p=>p.x===iteration);if(p&&Number.isFinite(p.y)){const row=document.createElement('div');row.textContent=s.label+': '+(percentage?percent(p.y):fmt(p.y,3));tip.append(row);}}tip.hidden=false;tip.style.left=Math.max(0,Math.min(px+12,w-220))+'px';tip.style.top='20px';});hit.addEventListener('pointerleave',()=>tip.hidden=true);
}
function render(){
 const live=state.live,c=live.config||{},last=state.training.at(-1),evalu=state.evaluations.at(-1);
 $('run-label').textContent=c.out?c.out.split('/').at(-1):'Aguardando execução';
 $('status').textContent=({running:'Em execução',completed:'Concluído',failed:'Falhou',interrupted:'Processo interrompido',stale:'Sem heartbeat recente',waiting:'Aguardando treino'})[live.status]||live.status||'—';
 $('connection').textContent=state.connected?'Dados conectados':'Conexão interrompida';
 $('updated').textContent=live.updated_at?'Último heartbeat: '+new Date(live.updated_at).toLocaleTimeString('pt-BR',{timeZone:'America/Sao_Paulo'}):'—';
 $('iterations').textContent=fmt(live.completed_iterations||0)+' / '+fmt(c.iters);
 $('progress').max=c.iters||1000;$('progress').value=live.completed_iterations||0;
 const phases={rollout:'Coletando partidas',updating:'Atualizando a rede',evaluating:'Avaliando adversários',checkpoint:'Salvando checkpoint',completed:'Execução encerrada',failed:live.error||'Falha'};
 let phase=phases[live.phase]||live.phase||'Aguardando';if(live.phase==='rollout')phase+=' · '+fmt(live.rollout_step)+' / '+fmt(c.rollout_steps)+' decisões';if(live.phase==='evaluating')phase+=' · '+fmt(live.evaluation_completed)+' / '+fmt(live.evaluation_total)+' casos';$('phase').textContent=phase;
 $('samples').textContent=fmt(last?.samples||0);$('setup').textContent=c.envs?`${c.envs} ambientes · ${c.threads} threads PyTorch`:'—';
 $('speed').textContent=last?fmt(last.training_samples_per_second,0)+' amostras/s':'—';
 if(last){const recent=state.training.slice(-20);const seconds=recent.reduce((n,r)=>n+r.iteration_seconds,0)/recent.length*Math.max(0,c.iters-live.completed_iterations);$('eta').textContent='Treino restante: ~'+fmt(seconds/60,0)+' min + avaliações';}else $('eta').textContent='Tempo restante: —';
 $('score').textContent=evalu?percent(evalu.score):'—';$('eval-age').textContent=evalu?'Avaliação completa da iteração '+evalu.iteration:'Ainda sem avaliação concluída';
 $('configuration').textContent=c.reward_weights?`Gol ±${c.reward_weights.goal} · aproximação ${c.reward_weights.approach} · progresso ${c.reward_weights.progress} · seed ${c.seed}`:'—';
 const performance=Object.keys(names).filter(bot=>state.visible.has(bot)).map(bot=>({label:names[bot],color:colors[bot],points:state.evaluations.map(e=>({x:e.iteration,y:group(e).find(r=>r.bot===bot).score}))}));plot('performance-chart',performance,{min:0,max:1,percentage:true});
 const tbody=$('opponents');tbody.replaceChildren();if(evalu){for(const row of group(evalu)){const tr=document.createElement('tr');for(const value of [names[row.bot],percent(row.score),fmt(row.wins),fmt(row.draws),fmt(row.losses),fmt(row.gf)+' × '+fmt(row.ga),percent(row.stationary)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);}renderGoals(group(evalu));}else $('goals-chart').innerHTML='<div class="empty">Aguardando avaliação completa</div>';
 plot('stationary-chart',[{label:'Bola parada',color:'#f0c36a',points:state.training.map(r=>({x:r.iteration,y:r.stationary_fraction}))}],{min:0,max:1,percentage:true});
 const terms=[['MotionPlayerToBall','Aproximação','#63b3ff'],['MotionBallToGoal','Progresso','#62d6a3'],['GoalReward','Gol','#b49cff']];plot('reward-chart',terms.map(([key,label,color])=>({label,color,points:state.training.map(r=>({x:r.iteration,y:r.reward_terms_per_sample[key]}))})));
 const key=$('ppo-metric').value;plot('ppo-chart',[{label:$('ppo-metric').selectedOptions[0].textContent,color:'#70d9dc',points:state.training.map(r=>({x:r.iteration,y:r[key]}))}],{percentage:key==='clip_fraction',min:['entropy','approximate_kl','clip_fraction','value_loss'].includes(key)?0:null});
 $('ppo-detail').textContent=last?`Entropia ${fmt(last.entropy,3)} · KL ${fmt(last.approximate_kl,4)} · clip ${percent(last.clip_fraction)} · critic ${fmt(last.value_loss,3)}`:'Aguardando atualização da rede.';
}
function renderGoals(rows){const root=$('goals-chart');root.replaceChildren();const w=Math.max(root.clientWidth,280),h=260,max=Math.max(1,...rows.flatMap(r=>[r.gf,r.ga]));const left=w<400?116:150,right=60,width=w-left-right;const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':'Gols pró e contra por adversário'});root.append(svg);rows.forEach((r,i)=>{const y=25+i*43;svg.append(svgNode('text',{x:left-8,y:y+12,'text-anchor':'end'},names[r.bot].replace(' congelada','')));for(const [value,color,dy]of [[r.gf,'#62d6a3',0],[r.ga,'#fa8991',16]]){svg.append(svgNode('rect',{x:left,y:y+dy,width:value/max*width,height:10,fill:color}),svgNode('text',{x:Math.min(w-5,left+value/max*width+5),y:y+dy+10},fmt(value)));}});svg.append(svgNode('text',{x:left,y:236},'0'),svgNode('text',{x:w-right,y:236,'text-anchor':'end'},fmt(max)),svgNode('text',{x:left,y:255},'Verde: pró · vermelho: contra'));}
for(const bot of Object.keys(names)){const button=document.createElement('button');button.type='button';button.setAttribute('aria-pressed','true');const mark=document.createElement('span');mark.className='swatch';mark.style.background=colors[bot];button.append(mark,document.createTextNode(names[bot]));button.addEventListener('click',()=>{state.visible.has(bot)?state.visible.delete(bot):state.visible.add(bot);button.setAttribute('aria-pressed',String(state.visible.has(bot)));render();});$('legend').append(button);}
for(const id of scaleCharts)scaleSync[id]=control(id);
$('scenario').addEventListener('change',render);$('ppo-metric').addEventListener('change',()=>{scaleSync['ppo-chart']();render();});let resize;window.addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(render,150);});
async function connect(){try{const [live,history]=await Promise.all([fetch('/api/state').then(r=>r.json()),fetch('/api/history').then(r=>r.json())]);state.live=live;state.training=history.training;state.evaluations=history.evaluations;render();}catch(error){$('connection').textContent='Servidor indisponível';}
 const events=new EventSource('/api/events');events.onmessage=event=>{const data=JSON.parse(event.data);if(state.live.run_path&&state.live.run_path!==data.live.run_path){state.training=[];state.evaluations=[];}state.live=data.live;state.connected=true;state.training=merge(state.training,data.training);state.evaluations=merge(state.evaluations,data.evaluations);render();};events.onerror=()=>{state.connected=false;render();};}
render();connect();
