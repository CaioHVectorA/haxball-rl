// Pure inference and observations, shared by the room and parity tests.
function linear(x, layer) {
  return layer.b.map((bias, i) => layer.w[i].reduce((sum, w, j) => sum + w * x[j], bias));
}
function logits(policy, obs) {
  if (obs.length !== policy.obs_dim) throw new Error('Observation dimension mismatch');
  const h = linear(linear(obs, policy.trunk0).map(Math.tanh), policy.trunk2).map(Math.tanh);
  return ['head_x', 'head_y', 'head_k'].map(name => linear(h, policy[name]));
}
function act(policy, obs) {
  const bins = logits(policy, obs).map(values => values.reduce((best, v, i) => v > values[best] ? i : best, 0));
  return { dx: bins[0] - 1, dy: bins[1] - 1, kick: bins[2] >= 1, rocket: bins[2] === 2 };
}
function buildObs(policy, self, ball, teammates, opponents, team, flipY = false) {
  const size = policy.team_size || 1;
  if (teammates.length !== size - 1 || opponents.length !== size) throw new Error('Incomplete teams');
  const [px, py] = policy.pos_coef || [1 / 420, 1 / 200];
  const sx = team === 1 ? 1 : -1, fy = flipY ? -1 : 1;
  const goal = (policy.goal_x == null ? 370 : policy.goal_x) * sx;
  const obs = [sx * self.pos.x * px, self.pos.y * fy * py,
    sx * self.speed.x / 10, self.speed.y * fy / 10,
    sx * (ball.pos.x - self.pos.x) * px, (ball.pos.y - self.pos.y) * fy * py,
    sx * ball.speed.x / 10, ball.speed.y * fy / 10,
    sx * (goal - self.pos.x) * px, -self.pos.y * fy * py,
    sx * (-goal - self.pos.x) * px, -self.pos.y * fy * py];
  let others = opponents;
  if (policy.obs_schema === 'team24_v1') {
    const ordered = opponents.slice().sort((a, b) => {
      const da = (a.pos.x - self.pos.x) ** 2 + (a.pos.y - self.pos.y) ** 2;
      const db = (b.pos.x - self.pos.x) ** 2 + (b.pos.y - self.pos.y) ** 2;
      return da - db || a.id - b.id;
    });
    others = [ordered[0], teammates[0], ordered[1]];
  }
  for (const other of others) obs.push(sx * (other.pos.x - self.pos.x) * px,
    (other.pos.y - self.pos.y) * fy * py, sx * other.speed.x / 10, other.speed.y * fy / 10);
  if (obs.length !== policy.obs_dim) throw new Error('Unsupported observation schema');
  return obs;
}
function validatePolicy(policy) {
  const size = policy.team_size || 1;
  const expected = policy.obs_schema === 'team24_v1' ? 24 : 16;
  if (policy.obs_dim !== expected || size !== (expected === 24 ? 2 : 1)) throw new Error('Invalid policy schema/team size');
  let width = expected;
  for (const name of ['trunk0', 'trunk2']) {
    const layer = policy[name];
    if (!layer || layer.w.length !== layer.b.length || layer.w.some(row => row.length !== width)) throw new Error('Invalid layer shape');
    width = layer.b.length;
  }
  for (const [name, count] of [['head_x', 3], ['head_y', 3], ['head_k', 2]]) {
    const layer = policy[name];
    if (!layer || layer.w.length !== count || layer.b.length !== count || layer.w.some(row => row.length !== width)) throw new Error('Invalid action heads');
  }
  for (const name of ['trunk0', 'trunk2', 'head_x', 'head_y', 'head_k']) {
    if (!policy[name].w.flat().concat(policy[name].b).every(Number.isFinite)) throw new Error('Nonfinite weights');
  }
  return policy;
}
module.exports = { buildObs, logits, act, validatePolicy };
