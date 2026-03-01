"""Build the combined grid + chart HTML.

Reads the parquet directly to generate grid data inline — no intermediate grid_data.js needed.
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
PARQUET = os.path.join(BASE, "data", "funding_rates_3m_v2.parquet")

# ── Generate grid data from parquet ──────────────────────────────────────────
df = pd.read_parquet(PARQUET)
df = df[df["funding_event"] == True].copy()
df["funding_rate_bps"] = df["funding_rate_bps"] / 100  # raw → true bps

DEX = {"Hyperliquid", "Lighter", "Aster"}

def normalize_coin(sym):
    for suffix in ["-USDT-SWAP", "-USDC", "-USD", "USDT"]:
        if sym.endswith(suffix):
            return sym[: -len(suffix)].upper()
    return sym.upper()

df["coin"] = df["symbol"].apply(normalize_coin)
df["ts8"] = df["timestamp"].dt.floor("8h")

# keep only 00/08/16 UTC
df = df[df["timestamp"].dt.hour.isin([0, 8, 16])]

coins = sorted(df["coin"].unique())
timestamps = sorted(df["ts8"].unique())
ts_labels = [t.strftime("%b %d %H:%M") for t in pd.to_datetime(timestamps)]

def compute_spreads(sub):
    """Return 2D list [ts_idx][coin_idx] of spreads, plus median dict."""
    data = [[None] * len(coins) for _ in range(len(timestamps))]
    ts_idx = {t: i for i, t in enumerate(timestamps)}
    coin_idx = {c: i for i, c in enumerate(coins)}
    for (ts8, coin), g in sub.groupby(["ts8", "coin"]):
        if len(g["exchange"].unique()) < 2:
            continue
        spread = round(g["funding_rate_bps"].max() - g["funding_rate_bps"].min(), 2)
        data[ts_idx[ts8]][coin_idx[coin]] = spread

    meds = {}
    for ci, coin in enumerate(coins):
        vals = [data[r][ci] for r in range(len(timestamps)) if data[r][ci] is not None]
        meds[coin] = round(float(np.median(vals)), 1) if vals else None
    return data, meds

data_all, meds_all = compute_spreads(df)
data_dex, meds_dex = compute_spreads(df[df["exchange"].isin(DEX)])

grid_data_js = (
    f"const COINS = {json.dumps(coins)};\n"
    f"const TIMESTAMPS = {json.dumps(ts_labels)};\n"
    f"const DATA = {{all:{json.dumps(data_all)},dex:{json.dumps(data_dex)}}};\n"
    f"const MEDS = {{all:{json.dumps(meds_all)},dex:{json.dumps(meds_dex)}}};\n"
)

print(f"Grid: {len(coins)} coins x {len(timestamps)} timestamps")

# ── HTML template ────────────────────────────────────────────────────────────
HTML_TOP = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Funding Rate Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f5f5f0; color: #222; font-family: monospace; font-size: 11px; padding: 0; margin: 0; }
  #grid-view { padding: 16px; }
  h2 { color: #222; font-size: 14px; margin-bottom: 4px; }
  p.sub { color: #666; font-size: 10px; margin-top: 0; margin-bottom: 10px; }
  .toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .toggle-group { display: flex; border: 1px solid #bbb; border-radius: 6px; overflow: hidden; }
  .toggle-btn {
    background: #e8e8e8; color: #444; border: none; padding: 5px 14px;
    cursor: pointer; font-family: monospace; font-size: 11px; font-weight: bold;
    border-right: 1px solid #bbb; transition: background 0.15s, color 0.15s;
  }
  .toggle-btn:last-child { border-right: none; }
  .toggle-btn.active { background: #3a6ea8; color: #fff; }
  .toggle-btn:hover:not(.active) { background: #d4d4d4; }
  .mode-label { color: #666; font-size: 10px; }
  .wrap { overflow: auto; max-height: calc(100vh - 90px); }
  table { border-collapse: collapse; white-space: nowrap; }
  th {
    background: #e0e0d8; color: #333; padding: 4px 3px; text-align: center;
    border: 1px solid #ccc; position: sticky; top: 0; z-index: 2; font-size: 10px; cursor: pointer;
  }
  th.ts { position: sticky; left: 0; z-index: 3; min-width: 90px; cursor: default; }
  th:hover:not(.ts) { background: #c8d8e8; }
  td { padding: 2px 3px; text-align: right; border: 1px solid #ddd; font-size: 10px; min-width: 28px; cursor: pointer; }
  td.ts {
    background: #e8e8e4; color: #444; position: sticky; left: 0; z-index: 1;
    padding: 2px 8px; text-align: left; border-right: 2px solid #bbb; cursor: default;
  }
  tr:hover td { filter: brightness(0.93); }
  #chart-modal {
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85); z-index: 100;
  }
  #chart-modal.visible { display: flex; flex-direction: column; }
  #modal-header {
    display: flex; align-items: center; gap: 16px; padding: 12px 24px;
    background: #161b22; border-bottom: 1px solid #30363d; flex-wrap: wrap;
  }
  #modal-header label { font-size: 13px; color: #8b949e; font-family: sans-serif; }
  #modal-header select, #modal-header input {
    background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 6px 10px; font-size: 14px; font-family: sans-serif;
  }
  .mfield { display: flex; flex-direction: column; gap: 4px; }
  #close-btn {
    margin-left: auto; background: #da3633; color: #fff; border: none;
    border-radius: 6px; padding: 6px 16px; font-size: 14px; cursor: pointer; font-family: sans-serif;
  }
  #close-btn:hover { background: #f85149; }
  #chart { flex: 1; }
  #status { padding: 4px 24px; font-size: 12px; color: #8b949e; font-family: sans-serif; }
</style>
</head>
<body>

<div id="grid-view">
  <h2>Max Cross-Exchange Spread by Coin &mdash; 8h Windows</h2>
  <p class="sub">spread_bps = max(rate) &minus; min(rate) across exchanges &nbsp;|&nbsp; 00:00 / 08:00 / 16:00 UTC &nbsp;|&nbsp; click a column or cell to drill down</p>
  <div class="toolbar">
    <div class="toggle-group">
      <button class="toggle-btn active" id="btn-all" onclick="setMode('all')">CEX + DEX</button>
      <button class="toggle-btn" id="btn-dex" onclick="setMode('dex')">DEX only</button>
    </div>
    <span class="mode-label" id="mode-label">Binance, OKX, Bybit, Hyperliquid, Lighter, Aster</span>
  </div>
  <div class="wrap">
    <table><thead id="thead"></thead><tbody id="tbody"></tbody></table>
  </div>
</div>

<div id="chart-modal">
  <div id="modal-header">
    <div class="mfield"><label>Coin</label><select id="coin"></select></div>
    <div class="mfield">
      <label>Show</label>
      <select id="rateType">
        <option value="both" selected>Implied + Settlement</option>
        <option value="settlement">Settlement Only</option>
      </select>
    </div>
    <div class="mfield">
      <label>Venues</label>
      <select id="venueFilter">
        <option value="all">All (CEX + DEX)</option>
        <option value="dex">DEX Only</option>
        <option value="cex">CEX Only</option>
      </select>
    </div>
    <button id="close-btn" onclick="closeChart()">&times; Close</button>
  </div>
  <div id="chart"></div>
  <div id="status"></div>
</div>

<script>
"""

HTML_BOTTOM = r"""
const MODE_LABELS = {
  all: 'Binance, OKX, Bybit, Hyperliquid, Lighter, Aster',
  dex: 'Hyperliquid, Lighter, Aster'
};
let currentMode = 'all';

function cellStyle(val) {
  if (val === null)  return ['#f5f5f0', '#bbb'];
  if (val < 1)       return ['#f5f5f0', '#999'];
  if (val < 5)       return ['#ffe0a0', '#444'];
  if (val < 10)      return ['#f4a028', '#222'];
  return              ['#c0392b', '#fff'];
}

function sortedCoins(mode) {
  const meds = MEDS[mode];
  return [...COINS].sort((a, b) => {
    const ma = meds[a] !== null ? meds[a] : -Infinity;
    const mb = meds[b] !== null ? meds[b] : -Infinity;
    return mb - ma;
  });
}

function buildHeader(mode, coinOrder) {
  let html = '<tr><th class="ts">Timestamp</th><th style="background:#d0d0c8">MAX</th>';
  for (const coin of coinOrder) {
    html += `<th onclick="openChart('${coin}')">${coin}</th>`;
  }
  return html + '</tr>';
}

function buildBody(mode, coinOrder) {
  const data = DATA[mode];
  const coinIdx = {};
  COINS.forEach((c, i) => coinIdx[c] = i);
  let html = '';
  for (let r = 0; r < TIMESTAMPS.length; r++) {
    // compute row max across all coins
    let rowMax = null;
    for (const coin of coinOrder) {
      const v = data[r][coinIdx[coin]];
      if (v !== null && (rowMax === null || v > rowMax)) rowMax = v;
    }
    const [maxBg, maxFg] = cellStyle(rowMax);
    const maxTxt = rowMax !== null ? Math.round(rowMax) : '';
    html += `<tr><td class="ts">${TIMESTAMPS[r]}</td>`;
    html += `<td style="background:${maxBg};color:${maxFg};font-weight:bold;border-right:2px solid #999">${maxTxt}</td>`;
    for (const coin of coinOrder) {
      const val = data[r][coinIdx[coin]];
      const [bg, fg] = cellStyle(val);
      const txt = val !== null ? Math.round(val) : '';
      html += `<td style="background:${bg};color:${fg}" onclick="openChart('${coin}')">${txt}</td>`;
    }
    html += '</tr>';
  }
  return html;
}

function setMode(mode) {
  currentMode = mode;
  document.getElementById('btn-all').classList.toggle('active', mode === 'all');
  document.getElementById('btn-dex').classList.toggle('active', mode === 'dex');
  document.getElementById('mode-label').textContent = MODE_LABELS[mode];
  const coinOrder = sortedCoins(mode);
  document.getElementById('thead').innerHTML = buildHeader(mode, coinOrder);
  document.getElementById('tbody').innerHTML = buildBody(mode, coinOrder);
}

// ===== CHART =====
const DATA_BASE = 'data/';
const EXCHANGE_COLORS = {
  Binance: '#f0b90b', OKX: '#ffffff', Bybit: '#ff6600',
  Aster: '#a78bfa', Hyperliquid: '#4ade80', Lighter: '#38bdf8',
};
const EXCHANGES_ORDER = ['Binance', 'OKX', 'Bybit', 'Aster', 'Hyperliquid', 'Lighter'];
const DEX_SET = new Set(['Hyperliquid', 'Lighter', 'Aster']);
const CEX_SET = new Set(['Binance', 'OKX', 'Bybit']);

let chartManifest = {};
let coinChartData = {};
let hiddenExchanges = new Set();

async function initChart() {
  const resp = await fetch(DATA_BASE + 'manifest.json');
  chartManifest = await resp.json();
  const coinSel = document.getElementById('coin');
  Object.keys(chartManifest).sort().forEach(c => {
    coinSel.add(new Option(c, c));
  });
  coinSel.addEventListener('change', loadCoinChart);
  document.getElementById('rateType').addEventListener('change', renderChart);
  document.getElementById('venueFilter').addEventListener('change', renderChart);
}

async function loadCoinChart() {
  const coin = document.getElementById('coin').value;
  if (!coin || !chartManifest[coin]) return;
  document.getElementById('status').textContent = 'Loading...';
  hiddenExchanges = new Set();
  const resp = await fetch(DATA_BASE + coin + '.json');
  coinChartData = await resp.json();

  renderChart();
  const totalRows = Object.values(coinChartData).reduce((s, r) => s + r.length, 0);
  document.getElementById('status').textContent =
    `${coin} — ${Object.keys(coinChartData).length} exchanges, ${totalRows.toLocaleString()} rows`;
}

function renderChart() {
  if (!Object.keys(coinChartData).length) return;
  const coin = document.getElementById('coin').value;
  const venueFilter = document.getElementById('venueFilter').value;
  const showBoth = document.getElementById('rateType').value === 'both';

  const eligibleExchanges = EXCHANGES_ORDER.filter(e => {
    if (!coinChartData[e]) return false;
    if (venueFilter === 'dex') return DEX_SET.has(e);
    if (venueFilter === 'cex') return CEX_SET.has(e);
    return true;
  });

  const traces = [];
  const spreadByTs = {};  // ts -> { exch: rate }

  for (const exch of eligibleExchanges) {
    const color = EXCHANGE_COLORS[exch] || '#888';
    const rows = coinChartData[exch];
    const hidden = hiddenExchanges.has(exch);
    const vis = hidden ? 'legendonly' : true;

    // Detect settlement interval
    const allSettTimes = [];
    for (const [t, r, ev] of rows) { if (ev === 1) allSettTimes.push(t); }
    let interval = '';
    if (allSettTimes.length >= 2) {
      const gap = (allSettTimes[1] - allSettTimes[0]) / 3600000;
      interval = ` (${gap}h)`;
    }
    const label = exch + interval;

    const settTs = [], settRate = [];
    for (const [t, r, ev] of rows) {
      if (ev === 1) {
        if (showBoth) {
          settTs.push(new Date(t));
          settRate.push(r);
        } else {
          const h = new Date(t).getUTCHours();
          if (h === 0 || h === 8 || h === 16) {
            settTs.push(new Date(t));
            settRate.push(r);
            if (!hidden) {
              if (!spreadByTs[t]) spreadByTs[t] = {};
              spreadByTs[t][exch] = r;
            }
          }
        }
      }
      if (showBoth && !hidden) {
        if (!spreadByTs[t]) spreadByTs[t] = {};
        spreadByTs[t][exch] = r;
      }
    }

    if (showBoth) {
      const implTs = [], implRate = [];
      for (const [t, r, ev] of rows) { implTs.push(new Date(t)); implRate.push(r); }
      traces.push({
        x: implTs, y: implRate, type: 'scattergl', mode: 'lines',
        name: label, line: { color, width: 1 }, visible: vis,
        legendgroup: exch, xaxis: 'x', yaxis: 'y',
        hovertemplate: '%{y:.1f}<extra>' + label + '</extra>',
      });
      traces.push({
        x: settTs, y: settRate, type: 'scattergl', mode: 'markers',
        name: label + ' settle', marker: { color, size: 5, symbol: 'diamond' },
        legendgroup: exch, showlegend: false, visible: vis, xaxis: 'x', yaxis: 'y',
        hovertemplate: '%{y:.1f}<extra>' + label + ' settle</extra>',
      });
    } else {
      traces.push({
        x: settTs, y: settRate, type: 'scattergl', mode: 'lines+markers',
        name: label, line: { color, width: 1.5 }, marker: { color, size: 4 },
        visible: vis, xaxis: 'x', yaxis: 'y',
        hovertemplate: '%{y:.1f}<extra>' + label + '</extra>',
      });
    }
  }

  const spreadTs = [], spreadVal = [], spreadText = [];
  for (const t of Object.keys(spreadByTs).map(Number).sort((a, b) => a - b)) {
    const byExch = spreadByTs[t];
    const entries = Object.entries(byExch);
    if (entries.length >= 2) {
      let maxE = entries[0], minE = entries[0];
      for (const e of entries) {
        if (e[1] > maxE[1]) maxE = e;
        if (e[1] < minE[1]) minE = e;
      }
      const spread = Math.round((maxE[1] - minE[1]) * 10) / 10;
      spreadTs.push(new Date(t));
      spreadVal.push(spread);
      spreadText.push(`${spread.toFixed(1)} bps — high: ${maxE[0]} (${maxE[1].toFixed(1)}), low: ${minE[0]} (${minE[1].toFixed(1)})`);
    }
  }

  traces.push({
    x: spreadTs, y: spreadVal, type: 'scattergl', mode: 'lines',
    name: 'Max Spread', line: { color: '#f85149', width: 1.5 },
    xaxis: 'x2', yaxis: 'y2', fill: 'tozeroy', fillcolor: 'rgba(248,81,73,0.15)',
    text: spreadText, hovertemplate: '%{text}<extra></extra>',
  });

  Plotly.newPlot('chart', traces, {
    title: { text: `${coin} — Funding Rates Across Exchanges`, font: { color: '#c9d1d9', size: 16 } },
    paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117',
    grid: { rows: 2, columns: 1, subplots: [['xy'], ['x2y2']], roworder: 'top to bottom' },
    xaxis: { color: '#8b949e', gridcolor: '#21262d', domain: [0, 1] },
    yaxis: { title: 'Funding Rate (bps)', color: '#8b949e', gridcolor: '#21262d', zeroline: true, zerolinecolor: '#30363d', domain: [0.28, 1] },
    xaxis2: { color: '#8b949e', gridcolor: '#21262d', matches: 'x', domain: [0, 1] },
    yaxis2: { title: 'Max Spread (bps)', color: '#8b949e', gridcolor: '#21262d', zeroline: true, zerolinecolor: '#30363d', domain: [0, 0.22] },
    legend: { font: { color: '#c9d1d9' }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center' },
    margin: { t: 60, b: 30, l: 60, r: 20 },
    hovermode: 'x unified',
  }, { responsive: true, scrollZoom: true });

  const chartEl = document.getElementById('chart');
  chartEl.on('plotly_legendclick', function(d) {
    const clickedName = d.data[d.curveNumber].name;
    // Extract exchange name (strip interval suffix and " settle")
    const exch = EXCHANGES_ORDER.find(e => clickedName.startsWith(e));
    if (exch) {
      if (hiddenExchanges.has(exch)) hiddenExchanges.delete(exch);
      else hiddenExchanges.add(exch);
      renderChart();
    }
    return false; // prevent default Plotly toggle
  });
}

async function openChart(coin) {
  document.getElementById('chart-modal').classList.add('visible');
  document.getElementById('venueFilter').value = currentMode === 'dex' ? 'dex' : 'all';
  document.getElementById('coin').value = coin;
  await loadCoinChart();
}

function closeChart() {
  document.getElementById('chart-modal').classList.remove('visible');
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeChart(); });

setMode('all');
initChart();
</script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(HTML_TOP)
    f.write(grid_data_js)
    f.write(HTML_BOTTOM)

print("Done - wrote index.html")
