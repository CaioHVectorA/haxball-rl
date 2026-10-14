const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const P=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const source=fs.readFileSync('headless-bot/bot.js','utf8');
const forward=source.slice(source.indexOf('function linear('),source.indexOf('// ── Room'));
const context={P,CONFIG:{flipY:false},RED:1,PX:P.pos_coef[0],PY:P.pos_coef[1],GOAL_X:P.goal_x,VS:.1};
vm.createContext(context);vm.runInContext(forward,context);
let maxError=0;
for(const c of fixture.cases){
 context.sample=c;
 const obs=vm.runInContext('buildObs(sample.self,sample.ball,sample.opp,sample.team)',context);
 assert.equal(obs.length,16);
 obs.forEach((v,i)=>assert(Math.abs(v-c.obs[i])<2e-6,'Observation mismatch'));
 context.obs=obs;
 const logits=vm.runInContext('(()=>{let h=tanh(linear(obs,P.trunk0));h=tanh(linear(h,P.trunk2));return [linear(h,P.head_x),linear(h,P.head_y),linear(h,P.head_k)];})()',context);
 logits.forEach((head,h)=>head.forEach((v,i)=>{maxError=Math.max(maxError,Math.abs(v-c.logits[h][i]));assert(Math.abs(v-c.logits[h][i])<2e-5,'Logit mismatch');}));
 const action=vm.runInContext('act(obs)',context);
 assert.deepEqual([action.dx+1,action.dy+1,action.kick?1:0],c.action);
}
console.log(`Export parity passed: ${fixture.cases.length} states, both teams; max logit error ${maxError}`);
