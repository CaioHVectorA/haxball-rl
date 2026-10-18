// Host a private Haxball room and play against our RL bot.
//
//   HEADLESS_TOKEN=<token from https://www.haxball.com/headlesstoken> node bot.js
//
// Optional env: ROOM_NAME, ROOM_PASS (default "bot"), TICK_SKIP (8), FLIP_Y (0/1).
//
// node-haxball is used only as a thin loader for the official Haxball engine. All the
// bot logic (obs + policy) is ours and mirrors the training code; because the Rust
// core follows the same physics, transfer still requires validation in a real room.
const fs = require("fs");
const path = require("path");
const { Room, Utils, OperationType, Errors } = require("node-haxball")();

// Load a local .env (gitignored) so `node bot.js` finds HEADLESS_TOKEN without exporting it.
// Existing env vars win, so `HEADLESS_TOKEN=… node bot.js` still overrides the file.
try {
  for (const line of fs.readFileSync(path.join(__dirname, ".env"), "utf8").split("\n")) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!m || process.env[m[1]] !== undefined) continue;
    let val = m[2];
    // strip one matching pair of surrounding quotes: HEADLESS_TOKEN="abc" -> abc (else the
    // quotes become part of the token and auth fails with a confusing "invalid token").
    if (val.length >= 2 && (val[0] === '"' || val[0] === "'") && val[val.length - 1] === val[0]) {
      val = val.slice(1, -1);
    }
    process.env[m[1]] = val;
  }
} catch { /* no .env — fall back to the real environment */ }

const TOKEN = process.env.HEADLESS_TOKEN;
if (!TOKEN) {
  console.error("Set HEADLESS_TOKEN (get one at https://www.haxball.com/headlesstoken).");
  process.exit(1);
}

const CONFIG = {
  roomName: process.env.ROOM_NAME || "RL Bot — private",
  password: process.env.ROOM_PASS || "bot",
  tickSkip: parseInt(process.env.TICK_SKIP || "8", 10), // decisions every N physics ticks
  flipY: process.env.FLIP_Y === "1", // flip if the bot moves up/down the wrong way
};
const BOT_ID = 65535;
const runtime = require("./policy_runtime");
const RED = 1, BLUE = 2, SPEC = 0;

// ── Policy (the trained MLP; weights from export_policy.py) ───────────────────
// Obs normalization is map-aware and travels in policy.json (pos_coef/goal_x), so the room
// obs matches what the bot trained on (futsal's bigger field needs a different scale than
// classic). Absent fields fall back to the classic defaults. `let` so a reload updates them.
const POLICY_PATH = process.env.POLICY_PATH ? path.resolve(process.env.POLICY_PATH) : path.join(__dirname, "policy.json");
// `let` so we can HOT-RELOAD new weights live (the `!reload` command / file-watch) without
// restarting the room — re-export policy.json, the running bot picks it up, same link.
let P = runtime.validatePolicy(JSON.parse(fs.readFileSync(POLICY_PATH, "utf8")));
const BOT_IDS = Array.from({length: P.team_size || 1}, (_, i) => BOT_ID - i);
let roomRef = null; // set in onOpen; lets the file-watch announce the new bot in-room
function reloadPolicy() {
  const next = runtime.validatePolicy(JSON.parse(fs.readFileSync(POLICY_PATH, "utf8")));
  if ((next.team_size || 1) !== BOT_IDS.length) throw new Error("Restart room to change team size");
  P = next;
  return P.obs_dim;
}
// auto-reload whenever policy.json changes on disk (so a fresh export swaps the bot instantly)
try {
  fs.watchFile(POLICY_PATH, { interval: 1000 }, () => {
    try {
      reloadPolicy();
      console.log(`policy.json changed -> reloaded "${P.name || "bot"}"`);
      if (roomRef) roomRef.sendAnnouncement(`▶ Now playing: "${P.name || "bot"}".`, null, 0x88ccff);
    } catch (e) { console.log("reload failed:", e.message); }
  });
} catch { /* watch unsupported; !reload still works */ }

function act(obs) { return runtime.act(P, obs); }

// ── Room ──────────────────────────────────────────────────────────────────────
let tick = 0, decisions = 0;
const lastKeys = new Map();
const announce = (room, msg, to = null, color = 0x88ccff) =>
  Utils.runAfterGameTick(() => room.sendAnnouncement(msg, to, color, "normal", 0));
const after = (fn) => Utils.runAfterGameTick(fn);

function humans(room) {
  return room.players.filter((p) => !BOT_IDS.includes(p.id));
}

function startIfReady(room) {
  const size = P.team_size || 1;
  const ready = [RED, BLUE].every(team => room.players.filter(p => p.team?.id === team).length === size);
  if (ready && !room.gameState) after(() => room.startGame());
}

function controlBot(room) {
  if (++tick % CONFIG.tickSkip !== 0) {
    for (const id of BOT_IDS) {
      const key = lastKeys.get(id);
      if (P.repeat_kick && key != null && (key & 16)) after(() => {
        room.fakeSendPlayerInput(key & ~16, id);
        room.fakeSendPlayerInput(key, id);
      });
    }
    return;
  }
  room.extrapolate();
  const ball = room.gameState?.physicsState?.discs?.[0];
  if (!ball) return;
  for (const id of BOT_IDS) {
    const me = room.getPlayer(id), self = me?.disc?.ext;
    if (!self) continue;
    const others = room.players.filter(p => p.id !== id && p.team?.id !== SPEC && p.disc?.ext);
    const disc = p => ({id: p.id, pos: p.disc.ext.pos, speed: p.disc.ext.speed});
    const mates = others.filter(p => p.team.id === me.team.id).map(disc);
    const opponents = others.filter(p => p.team.id !== me.team.id).map(disc);
    if (mates.length !== BOT_IDS.length - 1 || opponents.length !== BOT_IDS.length) continue;
    const obs = runtime.buildObs(P, self, ball, mates, opponents, me.team.id, CONFIG.flipY);
    const {dx, dy, kick} = act(obs);
    const sx = me.team.id === RED ? 1 : -1;
    const key = Utils.keyState(sx * dx, CONFIG.flipY ? -dy : dy, kick);
    after(() => {
      if (kick) room.fakeSendPlayerInput(key & ~16, id);
      room.fakeSendPlayerInput(key, id);
      lastKeys.set(id, key);
    });
  }
}

function handleCommand(room, player, text) {
  const [cmd] = text.slice(1).toLowerCase().split(/\s+/);
  switch (cmd) {
    case "restart": after(() => { room.stopGame(); room.startGame(); }); break;
    case "start": startIfReady(room); break;
    case "stop": after(() => room.stopGame()); break;
    case "swap": // swap the bot and you between teams
      after(() => {
        for (const p of room.players.filter(p => p.team?.id === RED || p.team?.id === BLUE)) {
          room.setPlayerTeam(p.id, p.team.id === RED ? BLUE : RED);
        }
      });
      break;
    case "reload": // hot-swap the bot's weights from policy.json — no room restart
      try {
        reloadPolicy();
        announce(room, `▶ Now playing: "${P.name || "bot"}" (obs_dim ${P.obs_dim}).`);
      } catch (e) {
        announce(room, `Reload failed: ${e.message}`, player.id, 0xff8888);
      }
      break;
    case "who": // which bot is loaded?
      announce(room, `▶ Current bot: "${P.name || "bot"}".`, player.id);
      break;
    case "help":
      announce(room, "Commands: !restart  !start  !stop  !swap  !reload  !who  (UI buttons work too)", player.id);
      break;
    default:
      announce(room, `Unknown command: !${cmd}`, player.id, 0xff8888);
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
      roomRef = room; // for file-watch in-room announcements
      // non-classic maps ship in policy.json; load the right pitch (futsal etc.) before play
      if (P.stadium_hbs) {
        try {
          room.setCurrentStadium(Utils.parseStadium(P.stadium_hbs), 0);  // byId 0 = host/system
          console.log(`stadium -> ${P.stadium_name || "custom"}`);
        } catch (e) { console.log("stadium load failed:", e.message); }
      }
      for (const id of BOT_IDS) {
      room.fakePlayerJoin(id, `🤖 RL · ${P.name || "bot"}`, "tr", "🤖", "fake-conn", "fake-auth");
      after(() => room.setPlayerTeam(id, RED));
      }

      room.onRoomLink = (link) =>
        console.log(`\n  >>> JOIN: ${link}\n  >>> password: ${CONFIG.password}\n`);

      room.onPlayerJoin = (player) => {
        if (BOT_IDS.includes(player.id)) return;
        console.log(`+ ${player.name} (id ${player.id})`);
        after(() => {
          room.setPlayerAdmin(player.id, true); // you control the room (restart, teams, …)
          const botTeam = room.getPlayer(BOT_ID).team?.id || RED;
          const humanTeam = botTeam === RED ? BLUE : RED;
          const full = humans(room).filter(p => p.id !== player.id && p.team?.id === humanTeam).length >= BOT_IDS.length;
          room.setPlayerTeam(player.id, full ? SPEC : humanTeam);
          startIfReady(room);
        });
        announce(room, `You're admin. Bot on red, playing "${P.name || "bot"}". Type !help.`, player.id);
        startIfReady(room);
      };

      room.onPlayerLeave = () => {
        const size = P.team_size || 1;
        if (room.gameState && [RED,BLUE].some(team => room.players.filter(p => p.team?.id === team).length !== size)) after(() => room.stopGame());
      };
      room.onPlayerTeamChange = () => startIfReady(room);

      // chat commands (suppress them from the public chat)
      room.onBeforeOperationReceived = (type, msg) => {
        if (type === OperationType.SendChat && msg.byId !== 0 && msg.text?.startsWith("!")) {
          handleCommand(room, room.getPlayer(msg.byId), msg.text);
          return false;
        }
        return true;
      };

      // keep playing: restart shortly after a goal-limit victory
      room.onGameEnd = () => setTimeout(() => startIfReady(room), 3000);
      room.onTeamGoal = () => startIfReady(room);

      room.onGameStart = () => { tick = 0; lastKeys.clear(); decisions = 0; };
      room.onGameTick = () => controlBot(room);
    },
    onClose: (msg) => {
      const code = msg?.code ?? msg;
      const name = Object.keys(Errors.ErrorCodes).find((key) => Errors.ErrorCodes[key] === code);
      console.error("Room closed:", name || "Unknown", code ?? "(unknown)");
      if (code === Errors.ErrorCodes.MissingRecaptchaCallbackError) {
        console.error("Headless token was not accepted or a CAPTCHA is required. Obtain a fresh token at https://www.haxball.com/headlesstoken and update HEADLESS_TOKEN in headless-bot/.env.");
      }
      process.exit(1);
    },
  }
);
