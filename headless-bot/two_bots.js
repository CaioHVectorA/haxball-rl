// Host a private Haxball room with TWO bots playing each other (1v1 Spectator Arena).
// Bot 1 (Red) and Bot 2 (Blue) battle each other while you watch as spectator!
const fs = require("fs");
const path = require("path");
const { Room, Utils } = require("node-haxball")();

try {
  for (const line of fs.readFileSync(path.join(__dirname, ".env"), "utf8").split("\n")) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!m || process.env[m[1]] !== undefined) continue;
    let val = m[2];
    if (val.length >= 2 && (val[0] === '"' || val[0] === "'") && val[val.length - 1] === val[0]) {
      val = val.slice(1, -1);
    }
    process.env[m[1]] = val;
  }
} catch {}

const TOKEN = process.env.HEADLESS_TOKEN;
if (!TOKEN) {
  console.error("Set HEADLESS_TOKEN.");
  process.exit(1);
}

const CONFIG = {
  roomName: process.env.ROOM_NAME || "Arena Bot vs Bot — 1v1",
  password: process.env.ROOM_PASS || "bot",
  tickSkip: parseInt(process.env.TICK_SKIP || "8", 10),
  flipY: process.env.FLIP_Y === "1",
};

const BOT_RED_ID = 65534;
const BOT_BLUE_ID = 65535;
const RED = 1, BLUE = 2, SPEC = 0;

const VS = 1 / 10;
let PX, PY, GOAL_X;
const POLICY_PATH = path.join(__dirname, "policy.json");
let P = JSON.parse(fs.readFileSync(POLICY_PATH, "utf8"));

function applyObsParams() {
  PX = P.pos_coef ? P.pos_coef[0] : 1 / 420;
  PY = P.pos_coef ? P.pos_coef[1] : 1 / 200;
  GOAL_X = P.goal_x != null ? P.goal_x : 370;
}
applyObsParams();

function linear(x, layer) {
  const out = new Array(layer.b.length);
  for (let i = 0; i < out.length; i++) {
    let sum = layer.b[i];
    const row = layer.w[i];
    for (let j = 0; j < x.length; j++) sum += row[j] * x[j];
    out[i] = sum;
  }
  return out;
}
const tanh = (a) => a.map(Math.tanh);
const argmax = (a) => a.reduce((best, v, i, arr) => (v > arr[best] ? i : best), 0);

function act(obs) {
  let h = tanh(linear(obs, P.trunk0));
  h = tanh(linear(h, P.trunk2));
  const kickVal = argmax(linear(h, P.head_k));
  return {
    dx: argmax(linear(h, P.head_x)) - 1,
    dy: argmax(linear(h, P.head_y)) - 1,
    kick: kickVal >= 1,
  };
}

function buildObs(self, ball, opp, teamId) {
  const sx = teamId === RED ? 1 : -1;
  const fy = CONFIG.flipY ? -1 : 1;
  return [
    sx * self.pos.x * PX, self.pos.y * fy * PY,
    sx * self.speed.x * VS, self.speed.y * fy * VS,
    sx * ball.pos.x * PX, ball.pos.y * fy * PY,
    sx * ball.speed.x * VS, ball.speed.y * fy * VS,
    sx * (ball.pos.x - self.pos.x) * PX, (ball.pos.y - self.pos.y) * fy * PY,
    Math.hypot(ball.pos.x - self.pos.x, ball.pos.y - self.pos.y) * PX,
    (GOAL_X - sx * self.pos.x) * PX,
    sx * opp.pos.x * PX, opp.pos.y * fy * PY,
    sx * opp.speed.x * VS, opp.speed.y * fy * VS,
  ];
}

let tick = 0;
const after = (fn) => Utils.runAfterGameTick(fn);

function controlBothBots(room) {
  if (++tick % CONFIG.tickSkip !== 0) return;
  room.extrapolate();

  const meRed = room.getPlayer(BOT_RED_ID);
  const meBlue = room.getPlayer(BOT_BLUE_ID);
  const discRed = meRed?.disc?.ext;
  const discBlue = meBlue?.disc?.ext;
  const ball = room.gameState?.physicsState?.discs?.[0];

  if (!discRed || !discBlue || !ball) return;

  // Control RED bot
  {
    const { dx, dy, kick } = act(buildObs(discRed, ball, discBlue, RED));
    const key = Utils.keyState(dx, CONFIG.flipY ? -dy : dy, kick);
    after(() => {
      if (kick) {
        room.fakeSendPlayerInput(key & ~16, BOT_RED_ID);
        room.fakeSendPlayerInput(key, BOT_RED_ID);
      } else {
        room.fakeSendPlayerInput(key, BOT_RED_ID);
      }
    });
  }

  // Control BLUE bot
  {
    const { dx, dy, kick } = act(buildObs(discBlue, ball, discRed, BLUE));
    const key = Utils.keyState(-dx, CONFIG.flipY ? -dy : dy, kick);
    after(() => {
      if (kick) {
        room.fakeSendPlayerInput(key & ~16, BOT_BLUE_ID);
        room.fakeSendPlayerInput(key, BOT_BLUE_ID);
      } else {
        room.fakeSendPlayerInput(key, BOT_BLUE_ID);
      }
    });
  }
}

Room.create(
  {
    name: CONFIG.roomName,
    password: CONFIG.password,
    showInRoomList: false,
    maxPlayerCount: 10,
    token: TOKEN,
    noPlayer: true,
  },
  {
    storage: { player_name: "RL-Host" },
    onOpen: (room) => {
      if (P.stadium_hbs) {
        try {
          room.setCurrentStadium(Utils.parseStadium(P.stadium_hbs), 0);
          console.log(`stadium -> ${P.stadium_name || "custom"}`);
        } catch (e) { console.log("stadium load error:", e.message); }
      }

      // Join both bots
      room.fakePlayerJoin(BOT_RED_ID, `🤖 Bot Vermelho (it4350)`, "br", "🔴", "fake-c1", "fake-a1");
      room.fakePlayerJoin(BOT_BLUE_ID, `🤖 Bot Azul (it4350)`, "fr", "🔵", "fake-c2", "fake-a2");

      after(() => {
        room.setPlayerTeam(BOT_RED_ID, RED);
        room.setPlayerTeam(BOT_BLUE_ID, BLUE);
        // Start match automatically
        setTimeout(() => room.startGame(), 1500);
      });

      room.onRoomLink = (link) => {
        console.log(`\n  >>> JOIN: ${link}\n  >>> password: ${CONFIG.password}\n`);
      };

      room.onPlayerJoin = (player) => {
        console.log(`+ Espectador entrou: ${player.name}`);
        after(() => {
          room.setPlayerAdmin(player.id, true);
          room.setPlayerTeam(player.id, SPEC); // Stay as spectator
          if (!room.gameState) room.startGame();
        });
      };

      room.onTeamVictory = (scores) => {
        console.log(`Fim de jogo: Vermelho ${scores.red} x ${scores.blue} Azul`);
        after(() => setTimeout(() => room.startGame(), 2500));
      };

      room.onGameTick = () => controlBothBots(room);
    }
  }
);
