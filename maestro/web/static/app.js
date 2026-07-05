// UBIK Maestro control panel — frontend logic.

// ── In-book Ubik advertisements (chapter epigraphs, PKD's Ubik) ──────────
const ADS = [
  { c: "Friends, this is clean-up time and we're discounting all our silent, electric Ubik by this much money.",
    d: "Safe when used as directed." },
  { c: "Perk up pouting household surfaces with new miracle Ubik, the easy-to-apply, extra-shiny, non-stick plastic coating.",
    d: "Entirely harmless if used as directed." },
  { c: "The best way to ask for beer is to say Ubik.",
    d: "Made from select hops, choice water, slow-aged for perfect flavor. Ubik." },
  { c: "Wake up to a hearty, lip-smacking bowlful of nutritious, nourishing Ubik toasted flakes.",
    d: "If tension or anxiety persists, discontinue use." },
  { c: "I am Ubik. Before the universe was, I am. I made the suns. I made the worlds. I am and I shall always be.",
    d: "(this one has no disclaimer)" },
];
let adIdx = 0;
function rotateAd() {
  const a = ADS[adIdx % ADS.length];
  document.getElementById("ad-copy").textContent = "“" + a.c + "”";
  document.getElementById("ad-disc").textContent = a.d;
  adIdx++;
}

// ── Console output ───────────────────────────────────────────────────────
function log(msg, cls) {
  const el = document.getElementById("console");
  const t = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.innerHTML = `<span class="t">[${t}]</span> <span class="${cls || ''}">${escapeHtml(msg)}</span>`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// ── API helpers ──────────────────────────────────────────────────────────
async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}: ${(await r.text()).slice(0,200)}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${path} → HTTP ${r.status}`);
  return data;
}

function busy(on) {
  document.querySelectorAll("button").forEach(b => b.disabled = on);
}

// ── Config / links / dropdown ────────────────────────────────────────────
async function loadConfig() {
  try {
    const cfg = await apiGet("/api/config");
    document.getElementById("node").textContent = cfg.node;
    document.getElementById("foot-node").textContent = cfg.node + " @ " + cfg.hostname;
    document.getElementById("ips").textContent =
      `H:${cfg.hippocampal_ip} · S:${cfg.somatic_ip}`;

    const sel = document.getElementById("svc");
    sel.innerHTML = "";
    cfg.services.forEach(s => {
      const o = document.createElement("option"); o.value = s; o.textContent = s; sel.appendChild(o);
    });

    const L = document.getElementById("links");
    const mk = (label, url, cls) =>
      `<a class="${cls||''}" href="${url}" target="_blank" rel="noopener">${label} ↗</a>`;
    L.innerHTML =
      mk("Neo4j browser", cfg.links.neo4j, "neo4j") +
      mk("ChromaDB", cfg.links.chromadb) +
      mk("MCP server", cfg.links.mcp) +
      mk("vLLM (Somatic)", cfg.links.vllm) +
      mk("API docs", "/api/docs");
  } catch (e) { log(e.message, "bad"); }
}

// ── Status grid ──────────────────────────────────────────────────────────
function healthClass(s) {
  if (s.healthy) return "healthy";
  const e = (s.error || "").toLowerCase();
  if (e.includes("not yet loaded") || e.includes("degraded") || e.includes("inferred")) return "degraded";
  return "down";
}
function detailText(s) {
  const d = s.details || {};
  if (s.name === "neo4j" && d.node_count != null) return `${d.node_count} nodes`;
  if (s.name === "chromadb" && d.collections) return `${d.collections} collections`;
  if (d.http_status) return `HTTP ${d.http_status}`;
  return "";
}
async function refreshStatus() {
  try {
    const data = await apiGet("/api/status");
    const grid = document.getElementById("grid");
    grid.innerHTML = "";
    data.services.forEach(s => {
      const cls = healthClass(s);
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML =
        `<span class="node">${s.node}</span>` +
        `<div><span class="dot ${cls}"></span><span class="name">${s.name}</span></div>` +
        `<div class="meta">${cls.toUpperCase()} · ${s.latency_ms} ms ${detailText(s) ? "· " + detailText(s) : ""}</div>` +
        (s.error && !s.healthy ? `<div class="err">${escapeHtml(s.error).slice(0,120)}</div>` : "");
      grid.appendChild(card);
    });
    document.getElementById("summary").textContent =
      `— ${data.healthy}/${data.total} healthy`;
  } catch (e) { log(e.message, "bad"); }
}

// ── Actions ──────────────────────────────────────────────────────────────
async function startAll(localOnly) {
  busy(true);
  log(`Starting ${localOnly ? "local" : "all cluster"} services… (vLLM load can take a few minutes)`, "warn");
  const poll = setInterval(refreshStatus, 4000);
  try {
    const r = await apiPost("/api/start", { local_only: localOnly });
    log(r.ok ? "All services running." : "Partial start — failed: " + r.failed.join(", "), r.ok ? "ok" : "bad");
  } catch (e) { log(e.message, "bad"); }
  finally { clearInterval(poll); busy(false); refreshStatus(); }
}
async function startService() {
  const svc = document.getElementById("svc").value;
  busy(true);
  log(`Starting ${svc}…`, "warn");
  const poll = setInterval(refreshStatus, 4000);
  try {
    const r = await apiPost("/api/start", { service: svc });
    log(r.ok ? `${svc} started.` : `${svc} failed to start.`, r.ok ? "ok" : "bad");
  } catch (e) { log(e.message, "bad"); }
  finally { clearInterval(poll); busy(false); refreshStatus(); }
}
async function getHealth() {
  busy(true);
  try {
    const h = await apiGet("/api/health");
    log("Health: " + (h.overall_status || "unknown").toUpperCase(),
        h.overall_status === "healthy" ? "ok" : "warn");
    log(JSON.stringify(h, null, 2));
  } catch (e) { log(e.message, "bad"); } finally { busy(false); }
}
async function getMetrics() {
  busy(true);
  try { log("Metrics:\n" + JSON.stringify(await apiGet("/api/metrics"), null, 2)); }
  catch (e) { log(e.message, "bad"); } finally { busy(false); }
}
async function getLogs() {
  busy(true);
  try {
    const d = await apiGet("/api/logs?lines=60");
    log(`— tail ${d.path} —`, "warn");
    (d.lines.length ? d.lines : [d.note || "(empty)"]).forEach(l => log(l));
  } catch (e) { log(e.message, "bad"); } finally { busy(false); }
}

// ── Shutdown modal ───────────────────────────────────────────────────────
function openShutdown()  { document.getElementById("modal").classList.add("show"); }
function closeShutdown() { document.getElementById("modal").classList.remove("show"); }
async function doShutdown() {
  const dry   = document.getElementById("sd-dry").checked;
  const local = document.getElementById("sd-local").checked;
  const emerg = document.getElementById("sd-emerg").checked;
  closeShutdown(); busy(true);
  log(`Shutdown${dry ? " (dry-run)" : ""}${local ? " local-only" : ""}${emerg ? " EMERGENCY" : ""}…`,
      emerg ? "bad" : "warn");
  const poll = setInterval(refreshStatus, 4000);
  try {
    const r = await apiPost("/api/shutdown",
      { dry_run: dry, local_only: local, emergency: emerg, confirm: true });
    if (r.action === "emergency_shutdown") log("Emergency shutdown complete.", "bad");
    else log(`${dry ? "Would stop" : "Stopped"}: ${(r.stopped || []).join(", ") || "none"}`, "ok");
  } catch (e) { log(e.message, "bad"); }
  finally { clearInterval(poll); busy(false); refreshStatus(); }
}

// ── Boot ─────────────────────────────────────────────────────────────────
rotateAd(); setInterval(rotateAd, 12000);
loadConfig();
refreshStatus();
setInterval(refreshStatus, 15000);   // live auto-refresh
