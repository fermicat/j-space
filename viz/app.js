"use strict";
/* j-space visualization core. Renders from window.JSPACE_BOOT:
   { mode: "embed"|"live", demos: [payload...] }  (payload schema: core.py) */

const BOOT = window.JSPACE_BOOT;
const SLOTS = ["--s1","--s2","--s3","--s4","--s5","--s6","--s7","--s8"];
const $ = (id) => document.getElementById(id);

const state = { demo: null, pos: 0, layerIdx: 0, pinned: [] };

/* ---------- helpers ---------- */
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const vis = (s) => s.replace(/\n/g, "⏎").replace(/\t/g, "⇥");
const word = (d, tid) => vis(d.vocab[tid] ?? "·");
const gloss = (d, tid) => (d.gloss && d.gloss[tid]) || "";
const rankOf = (d, tid, pos, li) => {
  const r = d.ranks[tid];
  return r ? r[pos][li] : null;
};
// 1-based display: "第1名" is the lens's top word.
const fmtRank = (r) => (r === null ? "—" : "#" + (r + 1));

function rampStep(d, rank) {
  if (rank === null || rank === undefined) return 0;
  const t = 1 - Math.log(rank + 1) / Math.log(d.vocab_size || 151936);
  if (t < 0.28) return 0;
  return Math.max(1, Math.min(7, Math.round(1 + (t - 0.28) / (1 - 0.28) * 6)));
}

function focusTid(d) {
  let best = null, bestR = Infinity;
  for (const tid of state.pinned) {
    const rows = d.ranks[tid];
    if (!rows) continue;
    for (const row of rows) for (const r of row) if (r < bestR) { bestR = r; best = tid; }
  }
  return best;
}

const slotOf = (tid) => SLOTS[state.pinned.indexOf(tid) % SLOTS.length];

/* ---------- tooltip ---------- */
const tip = $("tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html; tip.style.display = "block";
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(x + pad, innerWidth - w - 8) + "px";
  tip.style.top = (y + pad + h > innerHeight ? y - h - 6 : y + pad) + "px";
}
const hideTip = () => { tip.style.display = "none"; };

/* ---------- demo chips ---------- */
function renderChips() {
  const el = $("demo-chips");
  el.innerHTML = "";
  BOOT.demos.forEach((d, i) => {
    const b = document.createElement("button");
    b.className = "demo-chip" + (state.demo === d ? " active" : "");
    b.textContent = d.title_zh || d.slug;
    b.onclick = () => loadDemo(i);
    el.appendChild(b);
  });
}

/* ---------- filmstrip (hero signature) ---------- */
function renderFilmstrip() {
  const d = BOOT.demos.find((x) => x.slug === "capital-cn") || BOOT.demos[0];
  if (!d) return;
  const pos = d.tokens.length - 1;
  const pinned = d.pinned || [];
  const answer = (d.answer || "").toLowerCase();
  let ftid = null, bestR = Infinity, hitLi = -1;
  for (const tid of pinned) {
    const rows = d.ranks[tid]; if (!rows) continue;
    // The signature moment is a SILENT thought: skip concepts the model
    // actually said (they win the output layers, not the workspace story).
    const w = (d.vocab[tid] || "").trim().toLowerCase();
    if (w && answer.includes(w)) continue;
    rows[pos].forEach((r, li) => {
      // Search lens layers only, not the model-output row.
      if (li < d.layers.length - 1 && r < bestR) { bestR = r; ftid = tid; hitLi = li; }
    });
  }
  if (ftid === null) return;
  const row = $("fs-row");
  row.innerHTML = "";
  const nL = d.layers.length;
  const idxs = [];
  for (let li = 0; li < nL - 1; li += 2) idxs.push(li);
  if (!idxs.includes(hitLi) && hitLi >= 0) { idxs.push(hitLi); idxs.sort((a, b) => a - b); }
  idxs.push(nL - 1);
  let delay = 0;
  idxs.forEach((li, k) => {
    const isOut = li === nL - 1;
    const isHit = li === hitLi;
    const top = word(d, d.top_ids[pos][li][0]);
    const cell = document.createElement("div");
    cell.className = "fs-cell" + (isHit ? " hit" : "") + (isOut ? " out" : "");
    cell.style.setProperty("--d", (delay += 0.06) + "s");
    const lab = isOut ? "输出 out" : "L" + d.layers[li];
    const w = isHit ? `${word(d, ftid)} ${fmtRank(bestR)}` : top;
    cell.innerHTML = `<span class="l">${esc(lab)}</span><span class="w mono">${esc(w)}</span>`;
    row.appendChild(cell);
    if (k < idxs.length - 1) {
      const a = document.createElement("span");
      a.className = "fs-arrow"; a.textContent = "›";
      row.appendChild(a);
    }
  });
  $("fs-caption").innerHTML =
    `真实读出:<b>${esc(d.title_zh)}</b> — 每格是 J-lens 在该层读出的第 1 名词;` +
    `<span style="color:var(--accent);font-weight:700">紫色格</span>是沉默的中间概念浮现的一层` +
    `(${esc(word(d, ftid))} 升至全词表 ${fmtRank(bestR)},从未被说出)。`;
}

/* ---------- grid ---------- */
function renderGrid() {
  const d = state.demo;
  const f = focusTid(d);
  const nL = d.layers.length;
  let html = "<thead><tr><th></th>";
  d.tokens.forEach((t, p) => {
    html += `<th class="pos${p === state.pos ? " selcol" : ""}" data-pos="${p}">
      <span class="tok" title="${esc(t)}">${esc(vis(t))}</span></th>`;
  });
  html += "</tr></thead><tbody>";
  for (let li = 0; li < nL; li++) {
    const isOut = li === nL - 1;
    html += `<tr${isOut ? ' class="outrow"' : ""}><th class="layer${isOut ? " outrow" : ""}">` +
      (isOut ? "输出 output" : "L" + d.layers[li]) + "</th>";
    for (let p = 0; p < d.tokens.length; p++) {
      const top = d.top_ids[p][li][0];
      const step = f !== null ? rampStep(d, rankOf(d, f, p, li)) : 0;
      const sel = p === state.pos && li === state.layerIdx;
      html += `<td data-pos="${p}" data-li="${li}"${sel ? ' class="sel"' : ""}>` +
        `<span class="cellword cs${step}">${esc(word(d, top))}</span></td>`;
    }
    html += "</tr>";
  }
  html += "</tbody>";
  const grid = $("lensgrid");
  grid.innerHTML = html;

  grid.querySelectorAll("td").forEach((td) => {
    td.onclick = () => { select(+td.dataset.pos, +td.dataset.li); };
    td.onmousemove = (e) => {
      const p = +td.dataset.pos, li = +td.dataset.li;
      const ids = d.top_ids[p][li].slice(0, 3);
      let h = `<b>${esc(vis(d.tokens[p]))}</b> · ` +
        (li === nL - 1 ? "输出层" : "L" + d.layers[li]) + "<br>";
      h += ids.map((t, k) => `${k + 1}. ${esc(word(d, t))}`).join("&nbsp;&nbsp;");
      if (f !== null) {
        const r = rankOf(d, f, p, li);
        h += `<br>${esc(word(d, f))} → ${fmtRank(r)}`;
      }
      showTip(h, e.clientX, e.clientY);
    };
    td.onmouseleave = hideTip;
  });
  grid.querySelectorAll("th.pos").forEach((th) => {
    th.onclick = () => select(+th.dataset.pos, state.layerIdx);
  });

  const leg = $("grid-legend");
  if (f !== null) {
    let ramp = '<span class="ramp">';
    for (let s = 1; s <= 7; s++) ramp += `<i class="cs${s}"></i>`;
    ramp += "</span>";
    leg.innerHTML = `<span>焦点 token <b class="mono">${esc(word(d, f))}</b> 的排名:</span>` +
      `<span>靠后</span>${ramp}<span>第 1 名</span>` +
      `<span style="color:var(--muted)">(钉住其他词可切换焦点)</span>`;
  } else {
    leg.innerHTML = "<span>钉住一个词(点击任意词芯片)即可按其排名给整个网格着色。</span>";
  }
}

/* ---------- inspector ---------- */
function renderInspector() {
  const d = state.demo;
  const p = state.pos, li = state.layerIdx, nL = d.layers.length;
  $("insp-loc").textContent =
    ` — token「${vis(d.tokens[p])}」 × ` + (li === nL - 1 ? "输出层" : "L" + d.layers[li]);
  const el = $("topk");
  el.innerHTML = "";
  d.top_ids[p][li].forEach((tid, k) => {
    const r = d.top_ranks[p][li][k];
    const tracked = !!d.ranks[tid];
    const pin = state.pinned.includes(tid);
    const row = document.createElement("div");
    row.className = "topk-row";
    const g = gloss(d, tid);
    row.innerHTML =
      `<span class="ranknum">${fmtRank(r)}</span>` +
      `<button class="chip${pin ? " pinned" : ""}${tracked ? "" : " notrack"}"` +
      ` style="--pc:var(${pin ? slotOf(tid) : "--accent"})"` +
      ` title="${tracked ? (pin ? "取消钉住" : "钉住,加入轨迹图") : "此词未被追踪,无跨层数据"}">` +
      `${esc(word(d, tid))}</button>` +
      (g ? `<span class="gloss">${esc(g)}</span>` : "");
    if (tracked) row.querySelector("button").onclick = () => togglePin(tid);
    el.appendChild(row);
  });
  renderJspace();
}

function renderJspace() {
  const d = state.demo;
  const p = state.pos;
  const nL = d.layers.length;
  const layerVal = d.layers[Math.min(state.layerIdx, nL - 2)];
  let best = d.jspace_layers[0], dist = Infinity;
  for (const jl of d.jspace_layers) {
    const dd = Math.abs(jl - layerVal);
    if (dd < dist) { dist = dd; best = jl; }
  }
  const cell = d.jspace[p + "," + best];
  $("js-loc").textContent = ` — L${best}${best !== layerVal ? "(最近的已分解层)" : ""}`;
  if (!cell) { $("atoms").innerHTML = ""; $("evlabel").textContent = "该格未分解。"; return; }
  $("evbar").style.width = Math.round(cell.ev * 100) + "%";
  $("evlabel").textContent =
    `${cell.atoms.length}/25 个原子活跃 · 解释传送向量方差 ${(cell.ev * 100).toFixed(1)}%` +
    `(其余为不可言语化成分)`;
  const topSet = new Set(d.top_ids[p][d.layers.indexOf(best)]);
  const maxC = Math.max(...cell.atoms.map((a) => a[1]), 1e-6);
  $("atoms").innerHTML = cell.atoms.slice(0, 12).map(([tid, c]) => {
    const inTop = topSet.has(tid);
    const g = gloss(d, tid);
    return `<div class="atom">
      <span class="chip${inTop ? " intopk" : ""}" title="${esc(g || word(d, tid))}">${esc(word(d, tid))}</span>
      <span><span class="bar" style="width:${Math.max(2, c / maxC * 100)}%"></span></span>
      <span class="val">${c.toFixed(2)}</span></div>`;
  }).join("");
}

/* ---------- trajectory chart ---------- */
function renderTraj() {
  const d = state.demo;
  const p = state.pos;
  const leg = $("traj-legend");
  leg.innerHTML = "";
  state.pinned.forEach((tid) => {
    const chip = document.createElement("button");
    chip.className = "chip"; chip.style.setProperty("--pc", `var(${slotOf(tid)})`);
    chip.innerHTML = `<i></i>${esc(word(state.demo, tid))}<span class="x">✕</span>`;
    chip.onclick = () => togglePin(tid);
    leg.appendChild(chip);
  });
  $("traj-sub").innerHTML =
    `位置「<span class="mono">${esc(vis(d.tokens[p]))}</span>」上,被钉住的词在每一层的排名(对数轴,越高 = 越接近说出口)。`;

  const box = $("traj-chart");
  if (!state.pinned.length) {
    box.innerHTML = `<div class="traj-empty">点击任意词芯片,把它的跨层轨迹画到这里。</div>`;
    return;
  }
  const W = 640, H = 300, mL = 46, mR = 86, mT = 12, mB = 30;
  const iw = W - mL - mR, ih = H - mT - mB;
  const nL = d.layers.length;
  const vmax = Math.log10((d.vocab_size || 151936) + 1);
  const x = (li) => mL + li / (nL - 1) * iw;
  const y = (r) => mT + Math.log10(r + 1) / vmax * ih;

  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">`;
  // y grid: rank 1, 10, 100, 1k, 10k, 100k (displayed 1-based)
  for (const rv of [0, 9, 99, 999, 9999, 99999]) {
    const yy = y(rv);
    s += `<line x1="${mL}" x2="${W - mR}" y1="${yy}" y2="${yy}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${mL - 6}" y="${yy + 3.5}" text-anchor="end" font-size="10" fill="var(--muted)">${fmtRank(rv)}</text>`;
  }
  for (let li = 0; li < nL; li += 4) {
    s += `<text x="${x(li)}" y="${H - 10}" text-anchor="middle" font-size="10" fill="var(--muted)">L${d.layers[li]}</text>`;
  }
  s += `<text x="${x(nL - 1)}" y="${H - 10}" text-anchor="middle" font-size="10" fill="var(--ink-2)">输出</text>`;
  s += `<line x1="${mL}" x2="${W - mR}" y1="${mT + ih}" y2="${mT + ih}" stroke="var(--baseline)"/>`;

  const ends = [];
  state.pinned.forEach((tid) => {
    const rows = d.ranks[tid]; if (!rows) return;
    const pts = rows[p].map((r, li) => [x(li), y(r)]);
    const path = pts.map((pt, i) => (i ? "L" : "M") + pt[0].toFixed(1) + " " + pt[1].toFixed(1)).join("");
    s += `<path d="${path}" fill="none" stroke="var(${slotOf(tid)})" stroke-width="2" stroke-linejoin="round"/>`;
    ends.push({ tid, y: pts[pts.length - 1][1] });
  });
  ends.sort((a, b) => a.y - b.y);
  let lastY = -20;
  for (const e of ends) {
    const yy = Math.max(e.y, lastY + 12);
    lastY = yy;
    s += `<text x="${W - mR + 6}" y="${yy + 3.5}" font-size="10.5" fill="var(${slotOf(e.tid)})">${esc(word(d, e.tid).slice(0, 10))}</text>`;
  }
  s += `<line id="traj-cross" x1="0" x2="0" y1="${mT}" y2="${mT + ih}" stroke="var(--baseline)" stroke-dasharray="3 3" visibility="hidden"/>`;
  s += `<rect x="${mL}" y="${mT}" width="${iw}" height="${ih}" fill="transparent" id="traj-hover"/>`;
  s += "</svg>";
  box.innerHTML = s;

  const svg = box.querySelector("svg");
  const hover = box.querySelector("#traj-hover");
  const cross = box.querySelector("#traj-cross");
  hover.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width * W;
    const li = Math.max(0, Math.min(nL - 1, Math.round((px - mL) / iw * (nL - 1))));
    cross.setAttribute("x1", x(li)); cross.setAttribute("x2", x(li));
    cross.setAttribute("visibility", "visible");
    let h = `<b>${li === nL - 1 ? "输出层" : "L" + d.layers[li]}</b><br>`;
    h += state.pinned.map((tid) =>
      `${esc(word(d, tid))} ${fmtRank(rankOf(d, tid, p, li))}`).join("<br>");
    showTip(h, e.clientX, e.clientY);
  });
  hover.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

/* ---------- state changes ---------- */
function select(pos, li) {
  state.pos = pos; state.layerIdx = li;
  renderGrid(); renderInspector(); renderTraj();
}

function togglePin(tid) {
  const i = state.pinned.indexOf(tid);
  if (i >= 0) state.pinned.splice(i, 1);
  else {
    if (state.pinned.length >= 8) state.pinned.pop();
    state.pinned.push(tid);
  }
  renderGrid(); renderInspector(); renderTraj();
}

function loadDemo(i) {
  const d = BOOT.demos[i];
  state.demo = d;
  state.pinned = (d.pinned || []).slice(0, 8);
  const nL = d.layers.length;
  const fp = d.focus ? d.focus.pos : -1;
  state.pos = fp < 0 ? d.tokens.length + fp : fp;
  const fl = d.focus ? d.focus.layer : d.layers[Math.floor(nL / 2)];
  state.layerIdx = Math.max(0, d.layers.indexOf(fl));
  $("d-title").textContent = d.title_zh || d.slug;
  $("d-desc").textContent = d.desc_zh || "";
  $("d-desc-en").textContent = (d.title_en ? d.title_en + " — " : "") + (d.desc_en || "");
  $("d-answer").textContent = (d.answer || "").trim() || "(空)";
  $("d-note").textContent = d.note_zh || "";
  $("d-note-en").textContent = d.note_en || "";
  renderChips(); renderGrid(); renderInspector(); renderTraj();
  const col = document.querySelector(`th.pos[data-pos="${state.pos}"]`);
  if (col) col.scrollIntoView({ block: "nearest", inline: "center", behavior: "instant" });
}

/* ---------- live mode ---------- */
function setupLive() {
  if (BOOT.mode !== "live") return;
  $("livebar").classList.add("on");
  const input = $("live-input"), go = $("live-go"), status = $("livestatus");
  async function run() {
    const prompt = input.value.trim();
    if (!prompt) { status.style.display = "block"; status.textContent = "请输入非空 prompt。"; return; }
    go.disabled = true;
    status.style.display = "block";
    const t0 = performance.now();
    status.textContent = "计算中…(前向 + 逐层读出 + j-space 分解,约 10–30 秒)";
    try {
      const res = await fetch("/api/readout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const payload = await res.json();
      payload.title_zh = "自定义 · " + (prompt.length > 24 ? prompt.slice(0, 24) + "…" : prompt);
      payload.title_en = "custom";
      payload.desc_zh = prompt;
      payload.note_zh = "自定义 prompt:点击词芯片钉住感兴趣的概念,观察其跨层轨迹。";
      BOOT.demos.push(payload);
      loadDemo(BOOT.demos.length - 1);
      status.textContent = `完成,用时 ${((performance.now() - t0) / 1000).toFixed(1)} 秒。`;
    } catch (err) {
      status.textContent = "失败:" + err.message;
    } finally {
      go.disabled = false;
    }
  }
  go.onclick = run;
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
}

/* ---------- boot ---------- */
renderFilmstrip();
setupLive();
if (BOOT.demos.length) loadDemo(0);
