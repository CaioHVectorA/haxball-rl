// Host a private Haxball room with ChaseBot as the opponent.
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
  roomName: process.env.ROOM_NAME || "Humano vs ChaseBot — Baseline",
  password: process.env.ROOM_PASS || "bot",
  tickSkip: parseInt(process.env.TICK_SKIP || "8", 10),
};

const BOT_ID = 65535;
const RED = 1, BLUE = 2, SPEC = 0;
const KICK_DIST = 34.0;

let tick = 0;
const after = (fn) => Utils.runAfterGameTick(fn);

function controlChase(room) {
  if (++tick % CONFIG.tickSkip !== 0) return;
  room.extrapolate();

  const me = room.getPlayer(BOT_ID);
  const self = me?.disc?.ext;
  const ball = room.gameState?.physicsState?.discs?.[0];
  if (!self || !ball) return;

  const dx = ball.pos.x - self.pos.x;
  const dy = ball.pos.y - self.pos.y;
  const dist = Math.hypot(dx, dy);

  const moveX = dx > 15 ? 1 : (dx < -15 ? -1 : 0);
  const moveY = dy > 15 ? 1 : (dy < -15 ? -1 : 0);
  const kick = dist <= KICK_DIST ? 1 : 0;

  const key = Utils.keyState(moveX, moveY, kick);
  after(() => {
    if (kick) {
      room.fakeSendPlayerInput(key & ~16, BOT_ID);
      room.fakeSendPlayerInput(key, BOT_ID);
    } else {
      room.fakeSendPlayerInput(key, BOT_ID);
    }
  });
}

Room.create(
  {
    name: CONFIG.roomName,
    password: CONFIG.password,
    showInRoomList: false,
    maxPlayerCount: 5,
    token: TOKEN,
    noPlayer: true,
  },
  {
    storage: { player_name: "Chase-Host" },
    onOpen: (room) => {
      // Load futsal-classic if available in policy.json
      try {
        const P = JSON.parse(fs.readFileSync(path.join(__dirname, "policy.json"), "utf8"));
        if (P.stadium_hbs) {
          room.setCurrentStadium(Utils.parseStadium(P.stadium_hbs), 0);
        }
      } catch (e) {}

      room.fakePlayerJoin(BOT_ID, "🎯 ChaseBot (Scripted)", "ch", "🎯", "fake-c", "fake-a");
      after(() => room.setPlayerTeam(BOT_ID, RED));

      room.onRoomLink = (link) => {
        console.log(`\n  >>> JOIN CHASEBOT: ${link}\n  >>> password: ${CONFIG.password}\n`);
      };

      room.onPlayerJoin = (player) => {
        console.log(`+ ${player.name} entrou para enfrentar o ChaseBot`);
        after(() => {
          room.setPlayerAdmin(player.id, true);
          room.setPlayerTeam(player.id, BLUE);
          if (!room.gameState) setTimeout(() => room.startGame(), 1000);
        });
      };

      room.onGameTick = () => controlChase(room);
    }
  }
);
