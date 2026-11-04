"use strict";

const MODELS = ["logistic", "random_forest", "xgboost", "ensemble"];
const LABELS = { logistic: "Logistic Regression", random_forest: "Random Forest", xgboost: "XGBoost", ensemble: "Ensemble" };
const COLOR_VAR = { logistic: "--s1", random_forest: "--s2", xgboost: "--s3", ensemble: "--s4" };

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (m) => css(COLOR_VAR[m]);
const pct = (v, d = 1) => (v == null ? "–" : (100 * v).toFixed(d));
const signed = (v, d = 1, suffix = " pp") => (v == null ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d) + suffix);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

const S = { histModels: new Set(), range: 0, impKind: "permutation", indSel: "term_spread", overrides: {} };

/* ------------------------------------------------------------------ plotly base */
function layout(extra = {}) {
  const grid = css("--grid"), text2 = css("--text-2"), text3 = css("--text-3");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", size: 11.5, color: text2 },
    margin: { l: 48, r: 14, t: 8, b: 34 },
    xaxis: { gridcolor: grid, linecolor: grid, zeroline: false, tickfont: { color: text3 } },
    yaxis: { gridcolor: grid, linecolor: grid, zeroline: false, tickfont: { color: text3 } },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--border"), font: { color: css("--text"), family: "Inter", size: 12 } },
    showlegend: false,
  }, extra);
}
const CFG = { displayModeBar: false, responsive: true };
const plot = (id, data, lay) => Plotly.react(id, data, lay, CFG);

function recessionShapes(periods, from) {
  return periods.filter((p) => !from || p.end >= from).map((p) => ({
    type: "rect", xref: "x", yref: "paper", x0: p.start, x1: p.end, y0: 0, y1: 1,
    fillcolor: css("--rec"), line: { width: 0 }, layer: "below",
  }));
}
const hline = (y, col, dash = "dash") => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: { color: col, width: 1.2, dash } });

/* ------------------------------------------------------------------ formatting */
function fmtFeat(f, v, info, withUnit = true) {
  if (v == null) return "–";
  if (info.unit === "log Δ") return (v >= 0 ? "+" : "−") + Math.abs(100 * v).toFixed(1) + (withUnit ? "%" : "");
  if (info.unit === "pp") return (v >= 0 && f !== "term_spread" && f !== "term_spread_min12" && f !== "credit_spread" ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2) + (withUnit ? " pp" : "");
  return v.toFixed(2);
}

/* ------------------------------------------------------------------ overview */
function renderSummary(s, model) {
  $("#hero-prob").textContent = pct(s.probability);
  const icon = { low: "M5 13l4 4L19 7", elevated: "M12 8v5m0 3.5v.5", high: "M12 7v6m0 3.5v.5" }[s.band.level];
  $("#hero-band").innerHTML = `<span class="pill ${s.band.level}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="${icon}"/></svg>${s.band.label}</span>`;
  $("#meter-fill").style.width = `${100 * s.probability}%`;
  $("#meter-thr").style.left = `${100 * s.threshold}%`;
  $("#thr-label").textContent = `Signal threshold ${pct(s.threshold, 0)}%`;
  for (const [id, v] of [["#d1", s.change_1m], ["#d3", s.change_3m], ["#d12", s.change_12m]]) {
    const el = $(id);
    el.textContent = signed(100 * v);
    el.className = "v " + (v > 0.0005 ? "up" : v < -0.0005 ? "down" : "");
  }
  const m = model.metrics[model.champion];
  $("#k-champ").innerHTML = `<span class="sw-inline" style="background:${color(model.champion)}"></span>${s.model.label}`;
  $("#k-version").textContent = `${s.model.version} · promoted ${s.model.promoted_at.slice(0, 10)}`;
  $("#k-prauc").textContent = m.pr_auc.toFixed(3);
  $("#k-base").textContent = pct(model.data.positive_rate, 0) + "%";
  $("#k-auc").textContent = m.roc_auc.toFixed(3);
  $("#k-flag").textContent = m.recessions_flagged;
  $("#k-lead").textContent = m.avg_lead_months.toFixed(1);
  const lastObs = Object.values(s.data.last_observation).sort().pop();
  $("#asof-chip").innerHTML = `<span class="dot"></span><span>As of <b>${s.as_of.slice(0, 7)}</b> · ${s.data.source.startsWith("FRED API") ? "FRED API" : "FRED public"} · latest print ${lastObs.slice(0, 7)}</span>`;
  $("#asof-chip").title = `Fetched ${s.data.fetched_at}`;
}

function renderTerm(f) {
  const names = Object.keys(f.snapshots);
  const greys = [css("--text-3"), css("--text-3"), css("--text-3")];
  const traces = names.map((n, i) => {
    const snap = f.snapshots[n];
    const ys = f.horizons.map((h) => 100 * snap.curve[h]);
    const now = i === 0;
    return {
      x: f.horizons.map((h) => `${h}m`), y: ys, name: `${n} (${snap.as_of})`, mode: now ? "lines+markers+text" : "lines",
      line: { color: now ? css("--s1") : greys[i - 1], width: now ? 2.5 : 1.3, dash: ["solid", "dot", "dash", "longdash"][i] },
      marker: { size: 8, color: css("--s1"), line: { color: css("--surface"), width: 2 } },
      text: now ? ys.map((v) => v.toFixed(0) + "%") : undefined, textposition: "top center",
      textfont: { color: css("--text"), family: "JetBrains Mono", size: 11 },
      hovertemplate: `${n}: %{y:.1f}%<extra></extra>`,
    };
  });
  plot("term-chart", traces, layout({
    showlegend: true, legend: { orientation: "h", y: -0.18, x: 0, font: { size: 11 } },
    margin: { l: 40, r: 10, t: 18, b: 50 }, hovermode: "x unified",
    yaxis: { ...layout().yaxis, ticksuffix: "%", rangemode: "tozero" },
  }));
}

/* ------------------------------------------------------------------ history */
function renderToggles(h) {
  const box = $("#model-toggles");
  box.innerHTML = "";
  MODELS.forEach((m) => {
    const b = document.createElement("span");
    b.className = "toggle" + (S.histModels.has(m) ? "" : " off");
    b.innerHTML = `<span class="sw" style="background:${color(m)}"></span>${LABELS[m]}${m === h.champion ? ' <span class="tag">PROD</span>' : ""}`;
    b.onclick = () => { S.histModels.has(m) ? S.histModels.delete(m) : S.histModels.add(m); renderToggles(h); renderHistory(h); };
    box.appendChild(b);
  });
}

function renderHistory(h) {
  const traces = MODELS.filter((m) => S.histModels.has(m)).map((m) => ({
    x: h.dates, y: h.oos[m].map((v) => 100 * v), name: LABELS[m], type: "scatter", mode: "lines",
    line: { color: color(m), width: m === h.champion ? 2.4 : 1.4 },
    hovertemplate: `${LABELS[m]}: %{y:.1f}%<extra></extra>`,
  }));
  const last = h.dates[h.dates.length - 1];
  let x0 = h.dates[0];
  if (S.range) { const d = new Date(last); d.setFullYear(d.getFullYear() - S.range); x0 = d.toISOString().slice(0, 10); }
  plot("history-chart", traces, layout({
    hovermode: "x unified",
    xaxis: { ...layout().xaxis, range: [x0, last], type: "date" },
    yaxis: { ...layout().yaxis, range: [0, 100], ticksuffix: "%" },
    shapes: [...recessionShapes(h.recessions), hline(100 * h.threshold, css("--crit"))],
    annotations: [{ xref: "paper", x: 0.005, y: 100 * h.threshold, yshift: 9, text: `Signal ${pct(h.threshold, 0)}%`, showarrow: false, xanchor: "left", font: { size: 10.5, color: css("--crit") } }],
  }));
}

/* ------------------------------------------------------------------ drivers & importance */
function renderDrivers(d, feats) {
  const rows = d.drivers.slice().reverse();
  plot("drivers-chart", [{
    type: "bar", orientation: "h", y: rows.map((r) => r.label), x: rows.map((r) => 100 * r.contribution),
    marker: { color: rows.map((r) => (r.contribution >= 0 ? css("--pos") : css("--neg"))) },
    customdata: rows.map((r) => [fmtFeat(r.feature, r.value, feats[r.feature]), fmtFeat(r.feature, r.baseline, feats[r.feature])]),
    hovertemplate: "<b>%{y}</b><br>Effect: %{x:+.2f} pp<br>Now %{customdata[0]} · expansion median %{customdata[1]}<extra></extra>",
  }], layout({
    margin: { l: 210, r: 20, t: 8, b: 40 }, bargap: 0.35,
    xaxis: { ...layout().xaxis, ticksuffix: " pp", zeroline: true, zerolinecolor: css("--text-3"), title: { text: "← lowers risk      raises risk →", font: { size: 11, color: css("--text-3") } } },
    yaxis: { ...layout().yaxis, gridcolor: "rgba(0,0,0,0)", tickfont: { color: css("--text-2") } },
  }));
}

function renderImportance(imp) {
  const m = $("#imp-model").value;
  const feats = imp.features;
  let data, xTitle, sub;
  const names = Object.keys(feats);
  if (S.impKind === "permutation") {
    const p = imp.importance[m].permutation;
    const rows = names.map((f) => ({ f, v: p.mean[f], e: p.std[f] })).sort((a, b) => a.v - b.v);
    data = [{
      type: "bar", orientation: "h", y: rows.map((r) => feats[r.f].label), x: rows.map((r) => r.v),
      error_x: { type: "data", array: rows.map((r) => r.e), color: css("--text-3"), thickness: 1, width: 3 },
      marker: { color: color(m) }, hovertemplate: "<b>%{y}</b><br>AUC drop when shuffled: %{x:.4f}<extra></extra>",
    }];
    xTitle = "Drop in ROC-AUC when the feature is shuffled";
    sub = `Out-of-sample permutation importance · trained pre-2004, scored ${p.holdout} · ±1 sd over 25 shuffles`;
  } else if (m === "logistic" && imp.importance.logistic.coefficients) {
    const c = imp.importance.logistic.coefficients;
    const rows = names.map((f) => ({ f, v: c[f] })).sort((a, b) => Math.abs(a.v) - Math.abs(b.v));
    data = [{
      type: "bar", orientation: "h", y: rows.map((r) => feats[r.f].label), x: rows.map((r) => r.v),
      marker: { color: rows.map((r) => (r.v >= 0 ? css("--pos") : css("--neg"))) },
      hovertemplate: "<b>%{y}</b><br>Standardised coefficient: %{x:+.3f}<extra></extra>",
    }];
    xTitle = "Standardised log-odds coefficient (+ raises risk)";
    sub = "Logistic regression coefficients on z-scored features, full-sample fit";
  } else {
    const n = imp.importance[m].native;
    const rows = names.map((f) => ({ f, v: n[f] })).sort((a, b) => a.v - b.v);
    data = [{
      type: "bar", orientation: "h", y: rows.map((r) => feats[r.f].label), x: rows.map((r) => 100 * r.v),
      marker: { color: color(m) }, hovertemplate: "<b>%{y}</b><br>Share: %{x:.1f}%<extra></extra>",
    }];
    xTitle = m === "ensemble" ? "Average normalised importance of members (%)" : m === "xgboost" ? "Gain share (%)" : "Impurity-decrease share (%)";
    sub = "Model-native importance from the full-sample fit";
  }
  $("#imp-sub").textContent = sub;
  plot("imp-chart", data, layout({
    margin: { l: 210, r: 20, t: 8, b: 40 }, bargap: 0.35,
    xaxis: { ...layout().xaxis, zeroline: true, zerolinecolor: css("--text-3"), title: { text: xTitle, font: { size: 11, color: css("--text-3") } } },
    yaxis: { ...layout().yaxis, gridcolor: "rgba(0,0,0,0)", tickfont: { color: css("--text-2") } },
  }));
}

/* ------------------------------------------------------------------ models */
function renderMetrics(model) {
  const cols = [
    ["roc_auc", "ROC-AUC", 3, "max"], ["pr_auc", "PR-AUC", 3, "max"], ["brier", "Brier", 3, "min"],
    ["brier_skill", "Brier skill", 3, "max"], ["log_loss", "Log loss", 3, "min"], ["f1", "F1", 3, "max"],
    ["precision", "Precision", 3, "max"], ["recall", "Recall", 3, "max"], ["threshold", "Threshold", 2, null],
    ["recessions_flagged", "Flagged", null, null], ["false_alarms", "False alarms", 0, "min"],
    ["avg_lead_months", "Avg lead (m)", 1, "max"],
  ];
  const best = {};
  cols.forEach(([k, , , dir]) => {
    if (!dir) return;
    const vals = MODELS.map((m) => model.metrics[m][k]);
    best[k] = dir === "max" ? Math.max(...vals) : Math.min(...vals);
  });
  const head = `<tr><th>Model</th>${cols.map(([, l]) => `<th class="n">${l}</th>`).join("")}<th class="n">Avg rank</th></tr>`;
  const ordered = Object.keys(model.avg_rank);
  const body = ordered.map((m) => {
    const r = model.metrics[m];
    const cells = cols.map(([k, , d]) => {
      const v = r[k];
      const txt = d == null ? v : Number(v).toFixed(d);
      return `<td class="n${best[k] !== undefined && v === best[k] ? " best" : ""}">${txt}</td>`;
    }).join("");
    return `<tr class="${m === model.champion ? "champ" : ""}"><td><span class="sw-inline" style="background:${color(m)}"></span>${LABELS[m]}${m === model.champion ? '<span class="tag">PRODUCTION</span>' : ""}</td>${cells}<td class="n">${model.avg_rank[m].toFixed(2)}</td></tr>`;
  }).join("");
  $("#metrics-table").innerHTML = head + body;
  const b = model.backtest;
  $("#models-note").textContent = `Walk-forward backtest ${b.start} → ${b.end} · ${b.n_months} labelled months · ${b.refit} · ${b.embargo_months}-month embargo. Bold = best in column. Sorted by average rank.`;
}

function renderDiagnostics(bt) {
  const diag = { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: css("--text-3"), dash: "dot", width: 1 }, hoverinfo: "skip", showlegend: false };
  plot("roc-chart", [diag, ...MODELS.map((m) => ({
    x: bt.roc[m].fpr, y: bt.roc[m].tpr, name: `${LABELS[m]} (${bt.metrics[m].roc_auc.toFixed(3)})`, mode: "lines",
    line: { color: color(m), width: 2, shape: "hv" }, hovertemplate: `${LABELS[m]}<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>`,
  }))], layout({
    showlegend: true, legend: { x: 0.98, xanchor: "right", y: 0.04, bgcolor: "rgba(0,0,0,0)", font: { size: 10.5 } },
    xaxis: { ...layout().xaxis, title: { text: "False positive rate", font: { size: 11 } }, range: [0, 1] },
    yaxis: { ...layout().yaxis, title: { text: "True positive rate", font: { size: 11 } }, range: [0, 1.02] },
    margin: { l: 52, r: 10, t: 8, b: 44 },
  }));
  plot("cal-chart", [diag, ...MODELS.map((m) => ({
    x: bt.calibration[m].predicted, y: bt.calibration[m].observed, name: LABELS[m], mode: "lines+markers",
    line: { color: color(m), width: 1.6 }, marker: { size: 8, color: color(m), line: { color: css("--surface"), width: 2 } },
    customdata: bt.calibration[m].count,
    hovertemplate: `${LABELS[m]}<br>Predicted %{x:.2f} · observed %{y:.2f}<br>n = %{customdata}<extra></extra>`,
  }))], layout({
    showlegend: true, legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)", font: { size: 10.5 } },
    xaxis: { ...layout().xaxis, title: { text: "Mean predicted probability", font: { size: 11 } }, range: [0, 1] },
    yaxis: { ...layout().yaxis, title: { text: "Observed frequency", font: { size: 11 } }, range: [0, 1.02] },
    margin: { l: 52, r: 10, t: 8, b: 44 },
  }));
  const metrics = ["precision", "recall", "f1"];
  plot("pr-chart", MODELS.map((m) => ({
    type: "bar", name: LABELS[m], x: ["Precision", "Recall", "F1"], y: metrics.map((k) => bt.metrics[m][k]),
    marker: { color: color(m) }, hovertemplate: `${LABELS[m]} · %{x}: %{y:.3f}<extra></extra>`,
  })), layout({
    barmode: "group", bargap: 0.3, bargroupgap: 0.08, showlegend: true,
    legend: { orientation: "h", y: -0.14, font: { size: 10.5 } },
    yaxis: { ...layout().yaxis, range: [0, 1] }, margin: { l: 40, r: 10, t: 8, b: 50 },
  }));
}

function renderEpisodes(bt) {
  const m = $("#ep-model").value;
  const thr = bt.metrics[m].threshold;
  const head = `<tr><th>NBER peak</th><th>Trough</th><th class="n">Duration</th><th class="n">Peak prob. (12m before)</th><th class="n">Prob. at start</th><th class="n">First signal</th><th>Called</th></tr>`;
  const body = bt.episodes[m].map((e) => `<tr>
    <td class="mono">${e.start}</td><td class="mono">${e.end ?? "–"}</td><td class="n">${e.duration_months ?? "–"} mo</td>
    <td class="n">${pct(e.peak_prob_before)}%</td><td class="n">${pct(e.prob_at_start)}%</td>
    <td class="n">${e.detected ? (e.lead_months === 0 ? "at start" : `${e.lead_months} mo before`) : "–"}</td>
    <td>${e.detected ? '<span class="ok">✓ Yes</span>' : '<span class="miss">✕ Missed</span>'}</td></tr>`).join("");
  $("#episodes-table").innerHTML = head + body +
    `<tr><td colspan="7" class="section-note" style="white-space:normal">Signal threshold for ${LABELS[m]}: ${pct(thr, 0)}% · false-alarm episodes in backtest: ${bt.metrics[m].false_alarms}</td></tr>`;
}

/* ------------------------------------------------------------------ indicators */
function sparkline(dates, vals, recessions, years = 15) {
  const W = 240, H = 38;
  const n = vals.length, start = Math.max(0, n - years * 12);
  const v = vals.slice(start), d = dates.slice(start);
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const x = (i) => (i / (v.length - 1)) * W, y = (val) => H - 3 - ((val - lo) / span) * (H - 6);
  const t0 = new Date(d[0]).getTime(), t1 = new Date(d[d.length - 1]).getTime();
  const tx = (s) => ((new Date(s).getTime() - t0) / (t1 - t0)) * W;
  const rects = recessions.filter((r) => r.end >= d[0]).map((r) =>
    `<rect x="${Math.max(0, tx(r.start))}" y="0" width="${Math.max(1, tx(r.end) - Math.max(0, tx(r.start)))}" height="${H}" fill="var(--rec)"/>`).join("");
  const zero = lo < 0 && hi > 0 ? `<line x1="0" x2="${W}" y1="${y(0)}" y2="${y(0)}" stroke="var(--text-3)" stroke-width="0.6" stroke-dasharray="2 2"/>` : "";
  const path = v.map((val, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(val).toFixed(1)}`).join("");
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${rects}${zero}<path d="${path}" fill="none" stroke="var(--text)" stroke-width="1.3" vector-effect="non-scaling-stroke"/><circle cx="${W}" cy="${y(v[v.length - 1])}" r="2.6" fill="var(--s1)"/></svg>`;
}

function renderIndicatorGrid(ind) {
  const grid = $("#ind-grid");
  grid.innerHTML = ind.context.map((c) => {
    const rp = c.risk_percentile;
    const col = rp >= 0.8 ? "var(--crit)" : rp >= 0.6 ? "var(--warn)" : "var(--good)";
    const chg = c.value - c.prev_3m;
    const chgTxt = c.unit === "log Δ" ? signed(100 * chg, 1, " pts") : signed(chg, 2, c.unit === "pp" ? " pp" : "");
    return `<button class="ind${c.feature === S.indSel ? " sel" : ""}" data-f="${c.feature}">
      <div class="top"><div><div class="name">${c.label}</div><div class="grp">${c.group}</div></div></div>
      <div class="val">${fmtFeat(c.feature, c.value, c)}</div>
      <div class="chg">${chgTxt} over 3m</div>
      ${sparkline(ind.dates, ind.series[c.feature], ind.recessions)}
      <div class="riskbar"><span style="width:${(100 * rp).toFixed(0)}%;background:${col}"></span></div>
      <div class="risk-cap"><span>Recession-risk percentile</span><span class="mono">${(100 * rp).toFixed(0)}</span></div>
    </button>`;
  }).join("");
  grid.querySelectorAll(".ind").forEach((b) => (b.onclick = () => { S.indSel = b.dataset.f; renderIndicatorGrid(ind); renderIndicatorChart(ind); }));
}

function renderIndicatorChart(ind) {
  const c = ind.context.find((x) => x.feature === S.indSel);
  const scale = c.unit === "log Δ" ? 100 : 1;
  const suffix = c.unit === "log Δ" ? "%" : c.unit === "pp" ? " pp" : "";
  $("#ind-title").textContent = c.label;
  $("#ind-desc").textContent = c.desc;
  $("#ind-src").textContent = c.source;
  plot("ind-chart", [{
    x: ind.dates, y: ind.series[c.feature].map((v) => v * scale), mode: "lines", line: { color: css("--text"), width: 1.5 },
    hovertemplate: `%{x|%b %Y}: %{y:.2f}${suffix}<extra></extra>`,
  }], layout({
    hovermode: "x",
    xaxis: { ...layout().xaxis, type: "date" },
    yaxis: { ...layout().yaxis, ticksuffix: suffix, zeroline: true, zerolinecolor: css("--text-3") },
    shapes: [...recessionShapes(ind.recessions), hline(c.expansion_median * scale, css("--good"), "dot"), hline(c.recession_median * scale, css("--crit"), "dot")],
    annotations: [
      { xref: "paper", x: 1, y: c.expansion_median * scale, xanchor: "right", yshift: 8, showarrow: false, text: "expansion median", font: { size: 10, color: css("--good") } },
      { xref: "paper", x: 1, y: c.recession_median * scale, xanchor: "right", yshift: -8, showarrow: false, text: "recession median", font: { size: 10, color: css("--crit") } },
    ],
  }));
}

/* ------------------------------------------------------------------ scenario */
const PRESETS = {
  "Replay Sep 2000": { replay: "2000-09" },
  "Replay Jun 2007": { replay: "2007-06" },
  "Deep inversion": { term_spread: -1.2, term_spread_min12: -1.5 },
  "Credit shock": { credit_spread: 3.2, nfci: 0.6 },
  "Broad slowdown": { term_spread: -0.3, term_spread_min12: -0.8, credit_spread: 2.6, nfci: 0.0, permits_yoy: -0.25, claims_chg6: 0.2, indpro_yoy: -0.02 },
  "Aggressive tightening": { fedfunds_chg12: 3.0, term_spread: -0.5 },
  "Labor shock (isolated)": { sahm: 0.6, unrate_chg12: 0.9, claims_chg6: 0.25, payrolls_yoy: 0.0 },
};

function renderSliders(ind, base) {
  const box = $("#sliders");
  box.innerHTML = ind.context.map((c) => {
    const step = (c.p99 - c.p01) / 200;
    const lo = Math.min(c.p01, c.value), hi = Math.max(c.p99, c.value);
    return `<div class="slider"><div class="row"><span>${c.label}</span><span class="v" id="sv-${c.feature}">${fmtFeat(c.feature, c.value, c)}</span></div>
      <input type="range" id="sl-${c.feature}" data-f="${c.feature}" min="${lo}" max="${hi}" step="${step}" value="${c.value}">
      <div class="base">today ${fmtFeat(c.feature, c.value, c)}</div></div>`;
  }).join("");
  const ctx = Object.fromEntries(ind.context.map((c) => [c.feature, c]));
  let timer;
  box.querySelectorAll("input").forEach((el) => (el.oninput = () => {
    const f = el.dataset.f, v = parseFloat(el.value);
    if (Math.abs(v - ctx[f].value) < 1e-9) delete S.overrides[f]; else S.overrides[f] = v;
    const lab = $(`#sv-${f}`);
    lab.textContent = fmtFeat(f, v, ctx[f]);
    lab.classList.toggle("changed", f in S.overrides);
    clearTimeout(timer); timer = setTimeout(() => runScenario(base), 120);
  }));
  const pre = $("#presets");
  pre.innerHTML = Object.keys(PRESETS).map((k) => `<button class="btn" data-p="${k}">${k}</button>`).join("") + `<button class="btn" data-p="">Reset</button>`;
  pre.querySelectorAll("button").forEach((b) => (b.onclick = async () => {
    let preset = PRESETS[b.dataset.p] || {};
    if (preset.replay) preset = (await api(`/api/predict?month=${preset.replay}`)).features;
    S.overrides = {};
    ind.context.forEach((c) => { $(`#sl-${c.feature}`).value = c.value; });
    // Overrides keep the exact preset value; the slider thumb just shows the nearest step.
    Object.entries(preset).forEach(([f, v]) => {
      if (Math.abs(v - ctx[f].value) > 1e-9) { S.overrides[f] = v; $(`#sl-${f}`).value = v; }
    });
    ind.context.forEach((c) => {
      const lab = $(`#sv-${c.feature}`);
      lab.textContent = fmtFeat(c.feature, S.overrides[c.feature] ?? c.value, c);
      lab.classList.toggle("changed", c.feature in S.overrides);
    });
    pre.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b && b.dataset.p));
    runScenario(base);
  }));
}

async function runScenario(base) {
  const r = await api("/api/scenario", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ overrides: S.overrides }) });
  $("#scen-prob").textContent = pct(r.probability);
  const d = r.probability - r.baseline_probability;
  const n = Object.keys(S.overrides).length;
  $("#scen-vs").innerHTML = n
    ? `<span class="${d > 0 ? "up" : "down"} mono">${signed(100 * d)}</span> vs today’s ${pct(r.baseline_probability)}% · ${n} indicator${n > 1 ? "s" : ""} shocked`
    : `Baseline — matches today’s reading`;
  const hs = Object.keys(r.term_structure).map(Number);
  plot("scen-chart", [
    { x: hs.map((h) => `${h}m`), y: hs.map((h) => 100 * base.snapshots.Now.curve[h]), name: "Today", mode: "lines+markers", line: { color: css("--text-3"), width: 1.5, dash: "dot" }, marker: { size: 6 }, hovertemplate: "Today: %{y:.1f}%<extra></extra>" },
    { x: hs.map((h) => `${h}m`), y: hs.map((h) => 100 * r.term_structure[h]), name: "Scenario", mode: "lines+markers", line: { color: css("--s1"), width: 2.5 }, marker: { size: 8, line: { color: css("--surface"), width: 2 } }, hovertemplate: "Scenario: %{y:.1f}%<extra></extra>" },
  ], layout({
    showlegend: true, legend: { orientation: "h", y: -0.2, font: { size: 11 } }, hovermode: "x unified",
    yaxis: { ...layout().yaxis, ticksuffix: "%", range: [0, 100] }, margin: { l: 40, r: 10, t: 8, b: 46 },
  }));
}

/* ------------------------------------------------------------------ methodology */
function renderMethod(model) {
  $("#sel-rule").textContent = model.selection_rule + ` Ranking: ${Object.entries(model.avg_rank).map(([m, r]) => `${LABELS[m]} ${r.toFixed(2)}`).join(" · ")}.`;
  const kv = (obj) => Object.entries(obj).map(([k, v]) => `<dt>${k}</dt><dd>${Array.isArray(v) ? v.join(", ") : v}</dd>`).join("");
  $("#model-kv").innerHTML = kv({
    Algorithm: model.champion_label, Version: model.production.version, Trained: model.trained_at.replace("T", " ").slice(0, 16) + " UTC",
    Promoted: model.production.promoted_at.replace("T", " ").slice(0, 16) + " UTC", "Promotion reason": model.production.reason,
    Features: model.features.length, "Signal threshold": `${pct(model.threshold, 0)}% (F1-optimal on backtest)`,
  });
  $("#hp-kv").innerHTML = kv(model.hyperparameters[model.champion]);
  const d = model.data;
  $("#data-kv").innerHTML = kv({
    Source: d.source, Fetched: d.fetched_at.replace("T", " ").slice(0, 16) + " UTC", Sample: `${d.sample_start} → ${d.sample_end}`,
    "Labelled through": d.labelled_end, "Labelled months": d.n_labelled, "Positive rate": pct(d.positive_rate) + "%",
  });
  $("#versions-table").innerHTML = `<tr><th>Version</th><th>Champion</th><th class="n">PR-AUC</th><th></th></tr>` +
    model.versions.slice(0, 6).map((v) => `<tr><td class="mono">${v.version}</td><td>${LABELS[v.champion]}</td><td class="n">${v.pr_auc.toFixed(3)}</td><td>${v.production ? '<span class="tag">PROD</span>' : ""}</td></tr>`).join("");
}

/* ------------------------------------------------------------------ boot */
let DATA = {};
function renderAll() {
  const { summary, model, hist, fc, drv, imp, bt, ind } = DATA;
  renderSummary(summary, model);
  renderTerm(fc);
  renderToggles(hist);
  renderHistory(hist);
  renderDrivers(drv, imp.features);
  renderImportance(imp);
  renderMetrics(model);
  renderDiagnostics(bt);
  renderEpisodes(bt);
  renderIndicatorChart(ind);
  renderIndicatorGrid(ind);
  renderMethod(model);
}

async function load() {
  const [summary, model, hist, fc, drv, imp, bt, ind] = await Promise.all([
    api("/api/summary"), api("/api/model"), api("/api/history"), api("/api/forecast"),
    api("/api/drivers"), api("/api/importance"), api("/api/backtest"), api("/api/indicators"),
  ]);
  DATA = { summary, model, hist, fc, drv, imp, bt, ind };
  if (!S.histModels.size) S.histModels = new Set([model.champion, "ensemble"]);
  for (const sel of ["#imp-model", "#ep-model"]) {
    const el = $(sel);
    if (!el.options.length) {
      el.innerHTML = MODELS.map((m) => `<option value="${m}">${LABELS[m]}${m === model.champion ? " (production)" : ""}</option>`).join("");
      el.value = model.champion;
    }
  }
  renderAll();
  S.overrides = {};
  renderSliders(ind, fc);
  runScenario(fc);
  if (location.hash && !S.scrolled) { S.scrolled = true; document.querySelector(location.hash)?.scrollIntoView(); }
}

$("#imp-model").onchange = () => renderImportance(DATA.imp);
$("#ep-model").onchange = () => renderEpisodes(DATA.bt);
$("#imp-seg").querySelectorAll("button").forEach((b) => (b.onclick = () => {
  $("#imp-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  S.impKind = b.dataset.k; renderImportance(DATA.imp);
}));
$("#range-seg").querySelectorAll("button").forEach((b) => (b.onclick = () => {
  $("#range-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  S.range = +b.dataset.y; renderHistory(DATA.hist);
}));
$("#theme-btn").onclick = () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}
  renderAll(); runScenario(DATA.fc);
};
$("#refresh-btn").onclick = async () => {
  const b = $("#refresh-btn");
  b.disabled = true; b.lastChild.textContent = " Refreshing…";
  try { await api("/api/refresh", { method: "POST" }); await load(); }
  catch (e) { alert("Refresh failed: " + e.message); }
  finally { b.disabled = false; b.lastChild.textContent = " Refresh data"; }
};

load().catch((e) => {
  document.querySelector("main").innerHTML = `<div class="card err"><b>Could not load the production model.</b><br>${e.message}<br><br>Run <span class="mono">python train.py</span> and restart the server.</div>`;
});
