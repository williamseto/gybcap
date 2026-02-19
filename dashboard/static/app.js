/* ── Swing Strategy Dashboard — Vanilla JS + Plotly.js ───────────── */

const REGIME_COLORS = { 0: '#f85149', 1: '#d29922', 2: '#3fb950' };
const REGIME_LABELS = { 0: 'BEAR', 1: 'BALANCE', 2: 'BULL' };
const RISK_COLORS   = ['#238636', '#d29922', '#f85149', '#b91c1c'];

let _state = null;
let _ws    = null;

// ── Plotly layout defaults ─────────────────────────────────────────────
const LAYOUT_BASE = {
  paper_bgcolor: '#161b22',
  plot_bgcolor:  '#161b22',
  font:   { color: '#c9d1d9', size: 11 },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d', showgrid: true },
  yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d', showgrid: true },
};

function mergeLayout(overrides) {
  return Object.assign({}, LAYOUT_BASE, overrides, {
    xaxis: Object.assign({}, LAYOUT_BASE.xaxis, (overrides.xaxis || {})),
    yaxis: Object.assign({}, LAYOUT_BASE.yaxis, (overrides.yaxis || {})),
  });
}

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

// ── Entry point ────────────────────────────────────────────────────────
async function init() {
  showLoading(true);
  try {
    const res = await fetch('/api/state');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _state = await res.json();
    renderAll(_state);
  } catch (e) {
    console.error('Failed to load state:', e);
    document.getElementById('loading').textContent = 'Failed to load dashboard data.';
  } finally {
    showLoading(false);
  }
  connectWebSocket();
}

function renderAll(state) {
  renderHeader(state);
  renderRegimeBadge(state);
  renderRiskGauge(state);
  renderComponentBars(state);
  renderProbabilityBar(state);
  renderHmmState(state);
  renderCandlestick(state);
  renderRiskHistory(state);
  renderCorrelations(state);
  renderChangeDetection(state);
  renderOosProbabilities(state);
  renderSignalsTable(state.intraday_signals || []);
}

// ── Header ─────────────────────────────────────────────────────────────
function renderHeader(state) {
  const asOf = state.as_of_date || '—';
  const dur  = state.refresh_duration_sec
    ? ` · refreshed in ${state.refresh_duration_sec}s` : '';
  const acc  = state.model_accuracy ? ` · OOS acc ${(state.model_accuracy*100).toFixed(1)}%` : '';
  document.getElementById('header-date').textContent = `As of ${asOf}${dur}${acc}`;
}

// ── Left column ────────────────────────────────────────────────────────
function renderRegimeBadge(state) {
  const today = state.today;
  if (!today) return;
  const reg   = today.predicted_regime;
  const label = REGIME_LABELS[reg] || '—';
  const cls   = label.toLowerCase();

  const el = document.getElementById('regime-badge');
  el.innerHTML = `
    <div>
      <div class="regime-label ${cls}">${label}</div>
      <div class="regime-meta">P(BULL) ${(today.p_bull*100).toFixed(1)}%
        · P(BEAR) ${(today.p_bear*100).toFixed(1)}%</div>
      <div class="regime-meta">HMM: ${['BEAR','BALANCE','BULL'][today.hmm_state] || '—'}</div>
    </div>`;
}

function renderRiskGauge(state) {
  const today = state.today;
  const score = today ? today.risk_score : 0;
  const regimeIdx = today ? today.risk_regime : 0;
  const color = RISK_COLORS[regimeIdx] || RISK_COLORS[0];

  const data = [{
    type: 'indicator',
    mode: 'gauge+number',
    value: +(score * 100).toFixed(1),
    number: { suffix: '%', font: { size: 24 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#8b949e' },
      bar:  { color },
      steps: [
        { range: [0, 25],  color: '#0d1117' },
        { range: [25, 50], color: '#0d1117' },
        { range: [50, 75], color: '#0d1117' },
        { range: [75,100], color: '#0d1117' },
      ],
      threshold: {
        line: { color: '#fff', width: 2 },
        thickness: 0.75,
        value: score * 100,
      },
    },
  }];

  Plotly.newPlot('gauge-container', data, {
    ...LAYOUT_BASE,
    margin: { l: 30, r: 30, t: 20, b: 0 },
    height: 160,
  }, PLOTLY_CONFIG);
}

function renderComponentBars(state) {
  const today = state.today;
  if (!today) return;

  const components = [
    { key: 'range_stress',      label: 'Range Stress',    cls: 'range'   },
    { key: 'anomaly_intensity', label: 'Anomaly',         cls: 'anomaly' },
    { key: 'change_momentum',   label: 'Change Momentum', cls: 'change'  },
  ];

  const container = document.getElementById('component-bars');
  container.innerHTML = components.map(c => {
    const val = today[c.key] || 0;
    const pct = (val * 100).toFixed(1);
    return `
      <div class="component-bar">
        <span class="bar-label">${c.label}</span>
        <div class="bar-track">
          <div class="bar-fill ${c.cls}" style="width:${pct}%"></div>
        </div>
        <span class="bar-value">${pct}%</span>
      </div>`;
  }).join('');
}

function renderProbabilityBar(state) {
  const today = state.today;
  if (!today) return;
  const bear = (today.p_bear * 100).toFixed(1);
  const bal  = (today.p_balance * 100).toFixed(1);
  const bull = (today.p_bull * 100).toFixed(1);

  document.getElementById('proba-bear').style.width = bear + '%';
  document.getElementById('proba-bal').style.width  = bal  + '%';
  document.getElementById('proba-bull').style.width = bull + '%';

  document.getElementById('proba-label-bear').textContent = `BEAR ${bear}%`;
  document.getElementById('proba-label-bal').textContent  = `BAL ${bal}%`;
  document.getElementById('proba-label-bull').textContent = `BULL ${bull}%`;
}

function renderHmmState(state) {
  const today = state.today;
  if (!today) return;
  const hmmNames = ['BEAR', 'BALANCE', 'BULL'];
  const hmmClasses = ['bear', 'balance', 'bull'];
  const s = today.hmm_state;
  const name = hmmNames[s] || '—';
  const cls  = hmmClasses[s] || 'balance';
  const bull = (today.hmm_bull_prob * 100).toFixed(1);
  const bear = (today.hmm_bear_prob * 100).toFixed(1);

  document.getElementById('hmm-row').innerHTML = `
    <span class="hmm-dot ${cls}"></span>
    <span>HMM: <strong>${name}</strong></span>
    <span style="color:var(--muted)">Bull ${bull}% · Bear ${bear}%</span>`;
}

// ── Right column charts ────────────────────────────────────────────────

function renderCandlestick(state) {
  const hist = state.history || [];
  if (hist.length === 0) return;

  const dates  = hist.map(h => h.date);
  const opens  = hist.map(h => h.open);
  const highs  = hist.map(h => h.high);
  const lows   = hist.map(h => h.low);
  const closes = hist.map(h => h.close);

  // Regime background bands
  const shapes = [];
  let bandStart = null;
  let bandReg   = null;

  hist.forEach((h, i) => {
    const reg = h.predicted_regime;
    if (bandReg !== reg) {
      if (bandStart !== null) {
        shapes.push(_regimeBand(hist[bandStart].date, hist[i-1].date, bandReg));
      }
      bandStart = i;
      bandReg   = reg;
    }
  });
  if (bandStart !== null) {
    shapes.push(_regimeBand(hist[bandStart].date, hist[hist.length-1].date, bandReg));
  }

  const traces = [{
    type: 'candlestick',
    x: dates, open: opens, high: highs, low: lows, close: closes,
    increasing: { line: { color: '#3fb950' } },
    decreasing: { line: { color: '#f85149' } },
    name: 'ES',
  }];

  Plotly.newPlot('chart-candle', traces, mergeLayout({
    title: { text: 'ES Futures — Regime Bands', font: { size: 12 } },
    shapes,
    xaxis: { type: 'date', rangeslider: { visible: false } },
    yaxis: { title: 'Price' },
    height: 320,
    margin: { l: 60, r: 20, t: 40, b: 40 },
    showlegend: false,
  }), PLOTLY_CONFIG);
}

function _regimeBand(x0, x1, reg) {
  const fillMap = {
    0: 'rgba(248,81,73,0.08)',
    1: 'rgba(210,153,34,0.06)',
    2: 'rgba(63,185,80,0.08)',
  };
  return {
    type: 'rect', xref: 'x', yref: 'paper',
    x0, x1, y0: 0, y1: 1,
    fillcolor: fillMap[reg] || 'rgba(0,0,0,0)',
    line: { width: 0 },
    layer: 'below',
  };
}

function renderRiskHistory(state) {
  const hist = state.history || [];
  if (hist.length === 0) return;

  const dates = hist.map(h => h.date);

  const traces = [
    {
      type: 'scatter', mode: 'lines', fill: 'tozeroy',
      x: dates, y: hist.map(h => h.range_stress),
      name: 'Range Stress',
      line: { color: '#388bfd', width: 0 },
      fillcolor: 'rgba(56,139,253,0.3)',
      stackgroup: 'risk',
    },
    {
      type: 'scatter', mode: 'lines', fill: 'tonexty',
      x: dates, y: hist.map(h => h.anomaly_intensity),
      name: 'Anomaly',
      line: { color: '#d29922', width: 0 },
      fillcolor: 'rgba(210,153,34,0.3)',
      stackgroup: 'risk',
    },
    {
      type: 'scatter', mode: 'lines', fill: 'tonexty',
      x: dates, y: hist.map(h => h.change_momentum),
      name: 'Change',
      line: { color: '#f85149', width: 0 },
      fillcolor: 'rgba(248,81,73,0.3)',
      stackgroup: 'risk',
    },
    {
      type: 'scatter', mode: 'lines',
      x: dates, y: hist.map(h => h.risk_score),
      name: 'Composite',
      line: { color: '#fff', width: 2 },
    },
  ];

  Plotly.newPlot('chart-risk', traces, mergeLayout({
    title: { text: 'Risk Score History', font: { size: 12 } },
    yaxis: { range: [0, 1], title: 'Score' },
    height: 260,
    margin: { l: 60, r: 20, t: 40, b: 40 },
    legend: { orientation: 'h', y: -0.25 },
  }), PLOTLY_CONFIG);
}

function renderCorrelations(state) {
  const hist = state.history || [];
  if (hist.length === 0) return;
  const dates = hist.map(h => h.date);

  const nqTraces = [
    { x: dates, y: hist.map(h => h.corr_nq_10d), name: '10d', line: { color: '#58a6ff', width: 1.5 } },
    { x: dates, y: hist.map(h => h.corr_nq_20d), name: '20d', line: { color: '#3fb950', width: 1.5 } },
    { x: dates, y: hist.map(h => h.corr_nq_60d), name: '60d', line: { color: '#d29922', width: 1.5 } },
  ].map(t => ({ ...t, type: 'scatter', mode: 'lines' }));

  const znTraces = [
    { x: dates, y: hist.map(h => h.corr_zn_10d), name: '10d', line: { color: '#58a6ff', width: 1.5 } },
    { x: dates, y: hist.map(h => h.corr_zn_20d), name: '20d', line: { color: '#3fb950', width: 1.5 } },
    { x: dates, y: hist.map(h => h.corr_zn_60d), name: '60d', line: { color: '#d29922', width: 1.5 } },
  ].map(t => ({ ...t, type: 'scatter', mode: 'lines' }));

  const zeroLine = {
    type: 'line', xref: 'paper', yref: 'y',
    x0: 0, x1: 1, y0: 0, y1: 0,
    line: { color: '#8b949e', width: 1, dash: 'dot' },
  };

  Plotly.newPlot('chart-corr-nq', nqTraces, mergeLayout({
    title: { text: 'ES–NQ Rolling Correlation', font: { size: 12 } },
    yaxis: { range: [-1, 1], title: 'Corr' },
    height: 200, shapes: [zeroLine],
    legend: { orientation: 'h', y: -0.4 },
  }), PLOTLY_CONFIG);

  Plotly.newPlot('chart-corr-zn', znTraces, mergeLayout({
    title: { text: 'ES–ZN Rolling Correlation', font: { size: 12 } },
    yaxis: { range: [-1, 1], title: 'Corr' },
    height: 200, shapes: [zeroLine],
    legend: { orientation: 'h', y: -0.4 },
  }), PLOTLY_CONFIG);
}

function renderChangeDetection(state) {
  const hist = state.history || [];
  if (hist.length === 0) return;
  const dates = hist.map(h => h.date);

  const traces = [
    {
      type: 'scatter', mode: 'lines',
      x: dates, y: hist.map(h => h.return_cusum_score),
      name: 'Return CUSUM', line: { color: '#388bfd', width: 1.5 },
    },
    {
      type: 'scatter', mode: 'lines',
      x: dates, y: hist.map(h => h.anomaly_ewma_z),
      name: 'Anomaly EWMA-Z', line: { color: '#d29922', width: 1.5 },
    },
    {
      type: 'scatter', mode: 'lines',
      x: dates, y: hist.map(h => h.change_momentum),
      name: 'Change Momentum', line: { color: '#f85149', width: 1.5 },
    },
  ];

  Plotly.newPlot('chart-change', traces, mergeLayout({
    title: { text: 'Change Detection Signals', font: { size: 12 } },
    height: 200,
    legend: { orientation: 'h', y: -0.4 },
  }), PLOTLY_CONFIG);
}

function renderOosProbabilities(state) {
  if (!state.oos_dates || state.oos_dates.length === 0) return;

  const dates  = state.oos_dates;
  const probas = state.oos_probas || [];      // [[p_bear, p_bal, p_bull], ...]
  const preds  = state.oos_predictions || [];
  const acts   = state.oos_actuals || [];

  const pBull    = probas.map(p => p[2]);
  const actColor = acts.map(a => REGIME_COLORS[a] || '#8b949e');

  const traces = [
    {
      type: 'bar', x: dates, y: acts.map(a => 1),
      marker: { color: actColor, opacity: 0.3 },
      name: 'Actual Regime', yaxis: 'y2', showlegend: true,
    },
    {
      type: 'scatter', mode: 'lines',
      x: dates, y: pBull,
      name: 'P(BULL)', line: { color: '#3fb950', width: 2 },
    },
    {
      type: 'scatter', mode: 'lines',
      x: dates, y: probas.map(p => p[0]),
      name: 'P(BEAR)', line: { color: '#f85149', width: 1.5, dash: 'dot' },
    },
  ];

  const acc = state.model_accuracy ? `  OOS Acc: ${(state.model_accuracy*100).toFixed(1)}%` : '';

  Plotly.newPlot('chart-oos', traces, mergeLayout({
    title: { text: `OOS Regime Probabilities${acc}`, font: { size: 12 } },
    yaxis:  { range: [0, 1], title: 'Probability' },
    yaxis2: { range: [0, 3], overlaying: 'y', side: 'right', showgrid: false, visible: false },
    height: 260,
    legend: { orientation: 'h', y: -0.25 },
  }), PLOTLY_CONFIG);
}

// ── Signals table ──────────────────────────────────────────────────────
function renderSignalsTable(signals) {
  const placeholder = document.getElementById('signals-placeholder');
  const table       = document.getElementById('signals-table');
  const tbody       = document.getElementById('signals-tbody');

  if (!signals || signals.length === 0) {
    placeholder.style.display = 'block';
    table.style.display = 'none';
    return;
  }

  placeholder.style.display = 'none';
  table.style.display = 'table';

  tbody.innerHTML = signals.slice().reverse().map(s => {
    const dir   = (s.direction || '').toLowerCase();
    const cls   = dir === 'long' ? 'signal-long' : dir === 'short' ? 'signal-short' : '';
    const ts    = s.timestamp ? s.timestamp.slice(0, 19).replace('T', ' ') : '—';
    const conf  = s.pred_proba != null ? (s.pred_proba * 100).toFixed(1) + '%' : '—';
    return `<tr>
      <td>${ts}</td>
      <td>${s.strategy_name || '—'}</td>
      <td class="${cls}">${dir.toUpperCase() || '—'}</td>
      <td>${s.level_name || '—'}</td>
      <td>${s.entry_price != null ? s.entry_price.toFixed(2) : '—'}</td>
      <td>${conf}</td>
    </tr>`;
  }).join('');
}

function prependSignalRow(sig) {
  if (!_state) return;
  _state.intraday_signals = _state.intraday_signals || [];
  _state.intraday_signals.unshift(sig);
  renderSignalsTable(_state.intraday_signals);
}

// ── WebSocket ──────────────────────────────────────────────────────────
function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url   = `${proto}//${location.host}/ws/signals`;
  _ws = new WebSocket(url);

  const statusDot = document.getElementById('ws-status');

  _ws.onopen = () => {
    statusDot.className = 'connected';
  };

  _ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }

    if (msg.type === 'signal') {
      prependSignalRow(msg.data);
    } else if (msg.type === 'heartbeat') {
      // Update risk gauge live
      if (_state && _state.today) {
        _state.today.risk_score  = msg.risk_score;
        _state.today.risk_regime = msg.risk_regime;
        renderRiskGauge(_state);
      }
    } else if (msg.type === 'state_update') {
      // Re-fetch full state after a refresh completes
      fetch('/api/state').then(r => r.json()).then(s => {
        _state = s;
        renderAll(s);
      }).catch(console.error);
    }
  };

  _ws.onclose = () => {
    statusDot.className = 'disconnected';
    // Reconnect after 10 s
    setTimeout(connectWebSocket, 10000);
  };

  _ws.onerror = () => {
    statusDot.className = 'disconnected';
  };
}

// ── Loading overlay ────────────────────────────────────────────────────
function showLoading(show) {
  document.getElementById('loading').classList.toggle('show', show);
}

// ── Boot ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
