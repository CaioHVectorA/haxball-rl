const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('headless-bot/bot.js','utf8');
const control = source.slice(source.indexOf('function controlBot('), source.indexOf('function handleCommand('));
for (const repeat of [false,true]) {
  const sent=[];
  const context={sent,console,P:{repeat_kick:repeat},CONFIG:{tickSkip:8,flipY:false},DEBUG:false,
    RED:1,SPEC:0,BOT_ID:7,Utils:{keyState:(x,y,k)=>8+(k?16:0)},
    after:f=>f(),humans:()=>[{team:{id:2},disc:{ext:{}}}],buildObs:()=>[],
    act:()=>({dx:1,dy:0,kick:true,rocket:false}),
    room:{extrapolate(){},getPlayer(){return {team:{id:1},disc:{ext:{}}}},gameState:{physicsState:{discs:[{}]}},
      fakeSendPlayerInput(key,id){sent.push(key)}}};
  vm.createContext(context);
  vm.runInContext('let tick=0,lastKey=-1,decisions=0;'+control,context);
  vm.runInContext('controlBot(room)',context);
  assert.equal(sent.length,0,'No negative initial input');
  vm.runInContext('for(let i=0;i<7;i++)controlBot(room)',context);
  assert.deepEqual(sent,[8,24],'Decision releases and represses');
  vm.runInContext('controlBot(room)',context);
  assert.deepEqual(sent,repeat?[8,24,8,24]:[8,24],'Only repeat policies rearm between decisions');
}
console.log('Headless kick compatibility passed');
