"use strict";
const $ = (s) => document.querySelector(s);
const C = {
  STOP: "#ff5a52", CAUTION: "#f5a524", BLIND: "#b48cff", ADVISORY: "#e8d44d",
  ok: "#41d68a", info: "#57c7f0", muted: "#8a9a90", line: "#24302a", text: "#dfe8e2",
};
const LV = { NONE: 0, ADVISORY: 1, BLIND: 2, CAUTION: 3, STOP: 4 };
let CFG = null, S = null, showGT = false;
const openAlerts = new Set();

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function setHTML(sel, html) { const el = $(sel); if (el._h !== html) { el.innerHTML = html; el._h = html; } }
const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });

// ───────────────────────── canvas ─────────────────────────
const cv = $("#map"), ctx = cv.getContext("2d");
let W = 0, H = 0;
const X0 = -2200, X1 = 15200, YR = 700;
function resize() {
  const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  cv.width = r.width * d; cv.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0);
  W = r.width; H = r.height;
}
const px = (x) => 16 + ((x - X0) / (X1 - X0)) * (W - 32);
// piecewise lateral scale: the 0–200 m band (where decisions happen) gets 72% of the height
const py = (y) => {
  const a = Math.min(Math.abs(y), YR);
  const f = a <= 200 ? (a / 200) * 0.72 : 0.72 + ((a - 200) / (YR - 200)) * 0.28;
  return H / 2 - Math.sign(y) * f * (H / 2 - 34);
};

function draw() {
  if (!CFG || !S) return;
  ctx.clearRect(0, 0, W, H);
  // terrain
  let g = ctx.createLinearGradient(0, 0, 0, H / 2);
  g.addColorStop(0, "#0f2418"); g.addColorStop(1, "#13261b");
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H / 2);
  g = ctx.createLinearGradient(0, H / 2, 0, H);
  g.addColorStop(0, "#14241a"); g.addColorStop(1, "#0e1d15");
  ctx.fillStyle = g; ctx.fillRect(0, H / 2, W, H / 2);
  ctx.fillStyle = "rgba(223,232,226,.35)"; ctx.font = "11px IBM Plex Sans, system-ui, sans-serif";
  ctx.fillText("NORTH · Walayar reserve forest", 22, 46);
  ctx.fillText("SOUTH · water bodies / Madukkarai range", 22, H - 26);

  // water bodies
  for (const w of CFG.water_bodies) {
    ctx.beginPath();
    ctx.ellipse(px(w.x), py(w.y), (w.r / (X1 - X0)) * (W - 32) * 1.6, Math.abs(py(w.y + w.r / 2) - py(w.y - w.r / 2)), 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(60,130,190,.35)"; ctx.fill();
    ctx.strokeStyle = "rgba(87,199,240,.35)"; ctx.stroke();
  }

  // zone risk
  const zmax = {};
  for (const z of S.zone_risk) zmax[z.zone] = Math.max(zmax[z.zone] || 0, LV[z.level]);
  for (const z of CFG.zones) {
    const x0 = px(z.start), x1 = px(z.end), lv = zmax[z.id] || 0;
    if (lv) {
      const col = ["", C.ADVISORY, C.BLIND, C.CAUTION, C.STOP][lv];
      ctx.fillStyle = col + (lv >= 3 ? "1f" : "14"); ctx.fillRect(x0, 22, x1 - x0, H - 44);
    }
    ctx.strokeStyle = "rgba(255,255,255,.05)"; ctx.beginPath(); ctx.moveTo(x0, 22); ctx.lineTo(x0, H - 22); ctx.stroke();
    ctx.fillStyle = lv ? ["", C.ADVISORY, C.BLIND, C.CAUTION, C.STOP][lv] : "rgba(223,232,226,.4)";
    ctx.font = "600 10px IBM Plex Mono, ui-monospace, monospace"; ctx.fillText(z.id, x0 + 4, 16);
  }
  // hotspots
  for (const h of CFG.hotspots) {
    ctx.fillStyle = "rgba(245,165,36,.75)"; ctx.font = "10px IBM Plex Sans, system-ui, sans-serif";
    ctx.fillText(`${h.id} ${h.name}`, px(h.x) - 30, H - 8);
  }

  // camera coverage + towers
  const st = S.health.status;
  for (const t of CFG.towers) {
    const s = st[t.id];
    const col = s === "ONLINE" ? C.ok : s === "DEGRADED" ? C.CAUTION : C.STOP;
    const a = CFG.camera.range_along_m, l = CFG.camera.range_lateral_m;
    ctx.fillStyle = col + (s === "ONLINE" ? "0d" : "22");
    ctx.fillRect(px(t.x - a), py(l), px(t.x + a) - px(t.x - a), py(-l) - py(l));
    ctx.strokeStyle = col + "40"; ctx.setLineDash([3, 3]);
    ctx.strokeRect(px(t.x - a), py(l), px(t.x + a) - px(t.x - a), py(-l) - py(l)); ctx.setLineDash([]);
    const x = px(t.x), y = py(t.y);
    ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x - 5, y + 4); ctx.lineTo(x + 5, y + 4); ctx.closePath();
    ctx.fillStyle = col; ctx.fill();
    ctx.fillStyle = "rgba(223,232,226,.55)"; ctx.font = "9px IBM Plex Mono, ui-monospace, monospace";
    ctx.fillText(t.id, x - 9, t.y > 0 ? y - 9 : y + 15);
    if (s !== "ONLINE") { ctx.fillStyle = col; ctx.font = "700 10px IBM Plex Mono, ui-monospace, monospace"; ctx.fillText(s === "OFFLINE" ? "✕" : "!", x + 6, y); }
  }

  // tracks (rails)
  for (const L of CFG.section.lines) {
    const y = py(L.y);
    ctx.strokeStyle = "#5c6b62"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px(X0), y); ctx.lineTo(px(X1), y); ctx.stroke();
    ctx.fillStyle = "rgba(223,232,226,.6)"; ctx.font = "10px IBM Plex Mono, ui-monospace, monospace";
    ctx.fillText(L.label, px(X0) + 2, y + (L.y < 0 ? 13 : -5));
  }
  // restricted segments on rails
  for (const z of S.zone_risk) {
    if (LV[z.level] < 2) continue;
    const zz = CFG.zones.find((q) => q.id === z.zone), L = CFG.section.lines.find((q) => q.id === z.line);
    ctx.strokeStyle = C[z.level]; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(px(zz.start), py(L.y)); ctx.lineTo(px(zz.end), py(L.y)); ctx.stroke();
  }
  ctx.lineWidth = 1;
  // stations
  for (const s of CFG.section.stations) {
    const x = px(s.x);
    ctx.fillStyle = "#cfd8d2"; ctx.fillRect(x - 2, py(40), 4, py(-40) - py(40));
    ctx.font = "600 11px IBM Plex Sans, system-ui, sans-serif"; ctx.fillText(s.name, x - 22, py(60));
  }

  // ground truth
  if (showGT) {
    ctx.strokeStyle = "rgba(220,220,220,.7)";
    for (const e of S.world.elephants) { ctx.beginPath(); ctx.arc(px(e.x), py(e.y), 3.2, 0, 7); ctx.stroke(); }
    ctx.font = "9px IBM Plex Mono, ui-monospace, monospace"; ctx.fillStyle = "rgba(200,200,200,.75)";
    const sym = { gaur: "G", boar: "b", cattle: "c", human: "p", hotspot: "♨" };
    for (const o of S.world.others) ctx.fillText(sym[o.kind] || "?", px(o.x) - 3, py(o.y) + 3);
  }

  // system tracks
  for (const t of S.tracks) {
    const col = t.cls === "confirmed" ? C.STOP : t.cls === "candidate" ? C.ADVISORY : t.cls === "rejected" ? "#5d6b63" : "#3c4a42";
    const x = px(t.x), y = py(t.y);
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(px(t.x + t.vx * 25), py(t.y + t.vy * 25)); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, t.cls === "confirmed" ? 5 : 4, 0, 7);
    ctx.fillStyle = col; ctx.fill();
    if (t.cls === "rejected") { ctx.strokeStyle = "#0a0e0c"; ctx.beginPath(); ctx.moveTo(x - 3, y - 3); ctx.lineTo(x + 3, y + 3); ctx.stroke(); }
    if (t.source && t.source !== "sim") {   // track from a real detector (perception demo)
      ctx.strokeStyle = C.info; ctx.lineWidth = 2; ctx.strokeRect(x - 9, y - 9, 18, 18);
    }
  }
  const cam = S.tracks.filter((t) => t.source && t.source !== "sim");
  if (cam.length) {
    const top = cam.reduce((m, t) => (t.score > m.score ? t : m));
    ctx.fillStyle = C.info; ctx.font = "600 10px IBM Plex Mono, ui-monospace, monospace";
    ctx.fillText(`REAL CAMERA · ${cam.length} track${cam.length > 1 ? "s" : ""} · best ${top.score}`, px(top.x) + 14, py(top.y) + 22);
  }
  ctx.lineWidth = 1;

  // trains
  for (const tr of S.world.trains) {
    const L = CFG.section.lines.find((q) => q.id === tr.line), y = py(L.y);
    const rear = tr.front - tr.dir * tr.length;
    let a = px(Math.min(tr.front, rear)), b = px(Math.max(tr.front, rear));
    const col = tr.speed_kmph < 1 ? C.STOP : tr.restricted ? C.CAUTION : C.info;
    ctx.fillStyle = col; ctx.fillRect(a, y - 4, Math.max(3, b - a), 8);
    const hx = px(tr.front);
    ctx.beginPath();
    ctx.moveTo(hx + tr.dir * 6, y); ctx.lineTo(hx, y - 6); ctx.lineTo(hx, y + 6); ctx.closePath(); ctx.fill();
    ctx.font = "600 10px IBM Plex Mono, ui-monospace, monospace"; ctx.fillStyle = col;
    const label = `${tr.name.split(" ").slice(0, 2).join(" ")} ${Math.round(tr.speed_kmph)} km/h${tr.limit_kmph === null ? "" : tr.limit_kmph === 0 ? " · STOP" : " ≤" + tr.limit_kmph}`;
    const lx = Math.min(Math.max(a, 20), W - 200);
    ctx.fillText(label, lx, L.y < 0 ? y + 20 : y - 11);
  }

  // km scale
  ctx.fillStyle = "rgba(223,232,226,.4)"; ctx.font = "9px IBM Plex Mono, ui-monospace, monospace";
  for (let k = 0; k <= 13; k++) {
    const x = px(k * 1000);
    ctx.fillRect(x, H / 2 - 1, 1, 3);
    if (k % 2 === 0) ctx.fillText(`km ${k}`, x - 10, py(-55));
  }
  ctx.fillStyle = "rgba(223,232,226,.3)"; ctx.fillText("lateral scale non-linear (0–200 m expanded)", W - 250, H - 26);
}

// ───────────────────────── panels ─────────────────────────
function kpi(k, v, s, cls = "") { return `<div class="kpi ${cls}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`; }
const fmt = (v, u = "") => (v === null || v === undefined ? "—" : `${v}${u}`);

function renderKPIs() {
  const k = S.kpi, act = S.alerts.filter((a) => a.status === "ACTIVE");
  const top = act.reduce((m, a) => Math.max(m, LV[a.level]), 0);
  const restricted = S.world.trains.filter((t) => t.restricted).length;
  $("#kpis").innerHTML =
    kpi("Active alerts", act.length, top ? ["", "advisory", "blind", "caution", "STOP"][top] : "all clear", top >= 3 ? "bad" : "") +
    kpi("Trains restricted", restricted, `${S.world.trains.length} in section`) +
    kpi("Collisions", k.collisions, `${k.near_misses} near-miss`, k.collisions ? "bad" : "good") +
    kpi("In range → alert", fmt(k.detect_to_alert_p50_s, " s"), "p50, herd enters camera range") +
    kpi("Warning lead time", fmt(k.lead_time_p50_s, " s"), "p50, before train reaches zone") +
    kpi("Pilot ack", fmt(k.ack_p95_s, " s"), "p95 incl. escalation") +
    kpi("Towers online", `${k.towers_online}/${k.towers_total}`, S.health.blind_zones.length ? "blind: " + S.health.blind_zones.join(", ") : "full coverage", k.towers_online < k.towers_total ? "bad" : "") +
    kpi("Uplink", `${k.uplink_kb} KB`, `vs ${k.video_equiv_gb} GB video`);
}

function renderAlerts() {
  document.querySelectorAll("#alertList details[open]").forEach((d) => openAlerts.add(d.dataset.id));
  document.querySelectorAll("#alertList details:not([open])").forEach((d) => openAlerts.delete(d.dataset.id));
  const act = S.alerts.filter((a) => a.status === "ACTIVE");
  $("#alertCount").textContent = `${act.length} active`;
  if (!S.alerts.length) { setHTML("#alertList", `<div class="empty">No alerts. Inject a scenario to see the pipeline.</div>`); return; }
  setHTML("#alertList", S.alerts.map((a) => {
    const cls = a.status === "CLEARED" ? "CLEARED" : a.level;
    const ack = a.status !== "ACTIVE" ? "" : a.controller_ack !== null ? `<span class="acked">✓ controller ack</span>`
      : `<button class="ackbtn" data-ack="${a.id}">Acknowledge</button>`;
    return `<div class="alert ${cls}">
      <div class="row"><span class="lvl ${a.status === "CLEARED" ? "NONE" : a.level}">${a.status === "CLEARED" ? "CLEARED" : a.level}</span>
        <span class="where">${a.zone} · line ${a.line}</span><span class="id">${a.id}</span>${ack}</div>
      <div class="reason">${esc(a.reason)}${a.status === "CLEARED" ? ` <span class="id">(peak ${a.max_level})</span>` : ""}</div>
      <details data-id="${a.id}" ${openAlerts.has(a.id) ? "open" : ""}><summary>audit trail (${a.log.length})</summary><div class="log">${a.log.map(esc).join("\n")}</div></details>
    </div>`;
  }).join(""));
}

function renderTrains() {
  const ts = [...S.world.trains].sort((a, b) => a.line.localeCompare(b.line) || a.front - b.front);
  if (!ts.length) { setHTML("#trainList", `<div class="empty">No trains in or approaching the section.</div>`); return; }
  setHTML("#trainList", ts.map((t) => {
    const pct = Math.min(100, (t.speed_kmph / t.vmax_kmph) * 100);
    const lim = t.limit_kmph !== null ? `<div class="lim" style="left:${Math.min(100, (t.limit_kmph / t.vmax_kmph) * 100)}%"></div>` : "";
    const cls = t.speed_kmph < 1 ? "stopped" : t.restricted ? "restricted" : "";
    const msg = t.inbox.length ? t.inbox[t.inbox.length - 1] : "no messages";
    const pos = t.front < 0 ? `${(-t.front / 1000).toFixed(1)} km before ETMD` : t.front > 13000 ? `${((t.front - 13000) / 1000).toFixed(1)} km before WLH` : `km ${(t.front / 1000).toFixed(2)}`;
    return `<div class="train ${cls}">
      <div><div class="nm">${esc(t.name)}</div><div class="meta">${t.kind} · line ${t.line} ${t.dir > 0 ? "→ PGT" : "← CBE"} · ${pos}</div></div>
      <div class="speedbar"><div class="fill" style="width:${pct}%"></div>${lim}</div>
      <div class="spd">${t.speed_kmph.toFixed(0)} km/h<small>${t.emergency && t.speed_kmph >= 1 ? "EMERGENCY BRAKE" : t.limit_kmph !== null ? (t.limit_kmph === 0 ? "stop order" : "limit " + t.limit_kmph) : "line speed " + t.vmax_kmph}</small></div>
      <div class="msg">📟 ${esc(msg)}</div></div>`;
  }).join(""));
}

function renderAgents() {
  const v = S.verifier;
  setHTML("#verifyList", `<div>pending: ${v.pending}</div>` +
    (v.recent.length ? [...v.recent].reverse().map((r) => `<div class="${r.verdict === "confirmed" ? "warn" : r.verdict === "unverified" ? "bad" : "ok"}">${r.t} ${r.track} s=${r.score} → ${r.verdict} <span style="color:#8a9a90">[${esc(r.via || "")}]</span></div>`).join("") : `<div style="color:#8a9a90">no borderline tracks yet</div>`));
  const h = S.health, bad = Object.entries(h.status).filter(([, s]) => s !== "ONLINE");
  setHTML("#healthList", (bad.length ? bad.map(([t, s]) => `<div class="bad">${t} ${s}</div>`).join("") : `<div class="ok">all ${Object.keys(h.status).length} towers online</div>`) +
    (h.blind_zones.length ? `<div class="warn">BLIND zones: ${h.blind_zones.join(", ")} → 40 km/h</div>` : "") +
    h.tickets.slice().reverse().map((t) => `<div class="${t.status === "OPEN" ? "warn" : "ok"}">${t.id} ${t.tower} · ${esc(t.issue)} · ${t.status}</div>`).join(""));
}

function renderFeed() {
  setHTML("#feed", S.feed.map((e) => `<div class="ev ${e.kind} ${e.level || ""}"><span class="t">${e.clock}</span>${esc(e.text)}</div>`).join("") || `<div class="empty">Waiting for events…</div>`);
}

function renderTop() {
  $("#clock").textContent = S.clock;
  const dn = $("#dayNight"); dn.textContent = S.night ? "NIGHT" : "DAY"; dn.classList.toggle("day", !S.night);
  document.querySelectorAll("#modeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === S.mode));
  document.querySelectorAll("#speedSeg button").forEach((b) => b.classList.toggle("on", +b.dataset.speed === S.speed));
  $("#pauseBtn").textContent = S.paused ? "Resume" : "Pause";
}

let SYS = null;
function renderSystem() {
  if (!S.system) return;
  const b = S.system.bus, st = S.system.store, llm = SYS ? SYS.llm : null, vis = S.verifier;
  const bus = b.kind === "mqtt"
    ? `<span class="${b.connected ? "on" : "off"}">● MQTT ${esc(b.url)}</span> <b>${b.sent}</b> sent · <b>${b.received}</b> recv`
    : `<span class="off">● in-process bus</span>`;
  const store = st.kind === "postgres"
    ? `<span class="on">● Postgres</span> ${st.postgis ? "+PostGIS " : ""}${st.timescale ? "+Timescale " : ""}<b>${st.written}</b> rows`
    : `<span class="off">● SQLite</span>`;
  const l = llm ? (llm.enabled ? `<span class="on">● Groq</span> ${esc(llm.model)}` : `<span class="off">● LLM off</span> (set GROQ_API_KEY)`) : "";
  const v = `verifier: sim${vis.vision ? " + <b>" + esc(vis.vision) + "</b>" : ""}`;
  setHTML("#sysbar", `<span>${bus}</span><span>${store}</span><span>${l}</span><span>${v}</span><span>run ${esc(S.system.run_id)}</span>`);
}

function render() { renderTop(); renderSystem(); renderKPIs(); draw(); renderAlerts(); renderTrains(); renderAgents(); renderFeed(); renderChips(); }

// ───────────────────────── assistant ─────────────────────────
const chatHistory = [];
let chipsKey = "";
function renderChips() {
  const qs = ["What's happening right now?"];
  const a = S.alerts.find((x) => x.status === "ACTIVE");
  if (a) qs.push(`Explain alert ${a.id} — who was told and did they acknowledge?`);
  const t = S.world.trains.find((x) => x.restricted);
  if (t) qs.push(`Why is ${t.name} restricted?`);
  qs.push("Which towers need a technician?", "Summarise this shift for the handover", "Is anything near km 7.9 right now?");
  const key = qs.join("|");
  if (key === chipsKey) return;
  chipsKey = key;
  $("#asstChips").innerHTML = qs.map((q) => `<button type="button" data-q="${esc(q)}">${esc(q)}</button>`).join("");
}
function addMsg(html, cls) {
  const c = $("#chat");
  if (c.querySelector(".empty")) c.innerHTML = "";
  const d = document.createElement("div"); d.className = cls; d.innerHTML = html; c.appendChild(d);
  c.scrollTop = c.scrollHeight; return d;
}
async function ask(q) {
  q = (q || "").trim(); if (!q) return;
  $("#askInput").value = ""; $("#askBtn").disabled = true;
  addMsg(esc(q), "msg-q");
  const pending = addMsg("Thinking… (calling tools)", "msg-a");
  try {
    const r = await (await post("/api/assistant", { question: q, history: chatHistory.slice(-6) })).json();
    const tools = (r.trace || []).map((t) => `<span class="tool ${t.error ? "err" : ""}">${esc(t.tool)}(${esc(Object.entries(t.args || {}).map(([k, v]) => k + "=" + v).join(", "))})</span>`).join("");
    const meta = r.degraded ? `⚠ ${esc(r.reason || "LLM unavailable")}` : `${esc(r.model || "")} · ${r.latency_ms ?? "?"} ms · ${(r.trace || []).length} tool call(s)`;
    pending.className = "msg-a" + (r.degraded ? " degraded" : "");
    pending.innerHTML = `${esc(r.answer)}<div class="meta">${meta}<br>${tools}</div>`;
    chatHistory.push({ role: "user", content: q }, { role: "assistant", content: r.answer || "" });
  } catch (err) {
    pending.className = "msg-a degraded"; pending.textContent = "Request failed: " + err;
  } finally { $("#askBtn").disabled = false; $("#chat").scrollTop = $("#chat").scrollHeight; }
}
$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); ask($("#askInput").value); });

// ───────────────────────── wiring ─────────────────────────
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (m) => { S = JSON.parse(m.data); render(); };
  ws.onclose = () => setTimeout(connect, 1000);
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.mode) await post("/api/mode", { mode: b.dataset.mode });
  else if (b.dataset.speed) await post("/api/speed", { speed: +b.dataset.speed });
  else if (b.dataset.sc) await post(`/api/scenario/${b.dataset.sc}`);
  else if (b.dataset.ack) await post(`/api/alerts/${b.dataset.ack}/ack`);
  else if (b.dataset.q) ask(b.dataset.q);
  else if (b.id === "pauseBtn") await post("/api/pause");
  else if (b.id === "resetBtn") await post("/api/reset", { seed: 7, auto_spawn: false, mode: S ? S.mode : "gajarakshak" });
  else if (b.id === "insightsBtn") {
    $("#insightsOut").textContent = "Generating…";
    const r = await (await fetch("/api/insights?llm=true")).json();
    $("#insightsOut").textContent = r.llm_summary || r.summary;
  }
});
$("#gtToggle").addEventListener("change", (e) => { showGT = e.target.checked; draw(); });
window.addEventListener("resize", () => { resize(); draw(); });

(async () => {
  CFG = await (await fetch("/api/config")).json();
  try { SYS = await (await fetch("/api/system")).json();
    $("#asstModel").textContent = SYS.llm.enabled ? `Groq · ${SYS.llm.model} · read-only tools` : "LLM off — set GROQ_API_KEY (deterministic fallback)"; } catch (e) {}
  $("#sectionName").textContent = CFG.section.name;
  resize();
  connect();
})();
