// Regression against the actual node-haxball engine; no live room is modified.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {Room,Utils}=require('../headless-bot/node_modules/node-haxball')();
const baseline=JSON.parse(JSON.parse(fs.readFileSync('models/gpu_continuation_latest/policy.json','utf8')).stadium_hbs);
const fixed=JSON.parse(fs.readFileSync('maps/futsal_1v1_contained.hbs','utf8'));
const old=structuredClone(baseline),newMap=structuredClone(fixed);
delete old.name;delete newMap.name;
assert.deepEqual(newMap.planes.slice(old.planes.length),[
  {normal:[0,1],dist:-200,bCoef:1,cMask:['ball']},
  {normal:[0,-1],dist:-200,bCoef:1,cMask:['ball']},
]);
newMap.planes=newMap.planes.slice(0,old.planes.length);
let changes=0;
for(let i=0;i<old.segments.length;i++){
  if(['wall_map','wall_red_goal','wall_blue_goal'].includes(old.segments[i].trait)){
    assert.equal(newMap.segments[i].bias,-1000);
    delete newMap.segments[i].bias;changes++;
  }
}
assert.equal(changes,12);assert.deepEqual(newMap,old,'Only name, wall bias and two ball-only containment planes may change');
function make(map){
  const r=Room.sandbox({},{});r.setCurrentStadium(Utils.parseStadium(JSON.stringify(map)),0);
  r.playerJoin(1,'red','','','','');r.setPlayerTeam(1,1,0);
  r.playerJoin(2,'blue','','','','');r.setPlayerTeam(2,2,0);r.startGame(0);return r;
}
const rooms=[make(baseline),make(fixed)];
const starts=[[0,0],[350,150],[-350,-150],[390,185],[-390,185]];
const report={scenarios:[],lowSpeedIdentical:true,geometryAndPhysicsParametersPreserved:true};
for(const speed of [4,8,12,20,40,80]){
  const escapes=[0,0];let cases=0;
  for(const start of starts)for(let angle=0;angle<360;angle+=5){
    cases++;
    const games=rooms.map(r=>r.gameState.copy());const escaped=[false,false];
    for(const g of games){const b=g.physicsState.discs[0];b.pos.x=start[0];b.pos.y=start[1];b.I.x=speed*Math.cos(angle*Math.PI/180);b.I.y=speed*Math.sin(angle*Math.PI/180);}
    for(let tick=0;tick<180;tick++){
      games.forEach(g=>g.runSteps(1));
      const discs=games.map(g=>g.physicsState.discs[0]);
      for(let i=0;i<2;i++){
        const p=discs[i].pos;
        if(Math.abs(p.y)>206||(Math.abs(p.x)>407&&Math.abs(p.y)>77))escaped[i]=true;
      }
      if(speed===4){
        assert(Math.abs(discs[0].pos.x-discs[1].pos.x)<1e-7);
        assert(Math.abs(discs[0].pos.y-discs[1].pos.y)<1e-7);
      }
    }
    escaped.forEach((v,i)=>{if(v)escapes[i]++;});
  }
  report.scenarios.push({speed,cases,baselineEscapes:escapes[0],containedEscapes:escapes[1]});
  assert.equal(escapes[1],0,`Escaped at speed ${speed}`);
}
assert(report.scenarios.find(r=>r.speed===8).baselineEscapes>0,'Must reproduce original bug');
for(const sign of [-1,1]){
  const game=rooms[1].gameState.copy(),ball=game.physicsState.discs[0];
  ball.pos.x=sign*380;ball.pos.y=0;ball.I.x=sign*8;ball.I.y=0;
  for(let i=0;i<30;i++)game.runSteps(1);
  assert.equal(sign>0?game.redScore:game.blueScore,1,'Goal mouth must remain open');
}
report.bothGoalsRemainFunctional=true;
console.log(JSON.stringify(report,null,2));
if(process.argv[2])fs.writeFileSync(process.argv[2],JSON.stringify(report,null,2));
process.exit(0);
