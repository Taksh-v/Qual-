console.log("script.js loaded");
const LS_KEY = 'macro_query_history_v2';

const state = {
  latest: null,
  history: [],
  charts: {},
  generation: {
    running: false,
    section: '',
    startedAt: 0,
    chars: 0,
    timer: null
  }
};

// ── Persist history to localStorage ──────────────────────────────────────
function _saveHistory() {
  try {
    const slim = state.history.slice(0, 20).map(h => ({
      question: h.question, ts: h.ts,
      regime: h.payload?.snapshot?.regime?.regime || '',
      signal: h.payload?.snapshot?.cross_asset?.overall_signal || '',
      quality: h.payload?.quality?.band || '',
      score: h.payload?.quality?.score || 0,
      model: h.payload?.model_used || '',
    }));
    localStorage.setItem(LS_KEY, JSON.stringify(slim));
  } catch (_) { }
}

function _loadHistory() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return;
    const items = JSON.parse(raw);
    // Merge into state; full payload not stored — only metadata
    items.forEach(item => {
      if (!state.history.find(h => h.ts === item.ts)) {
        state.history.push({ ...item, payload: null });
      }
    });
  } catch (_) { }
}

// ── Critical indicator groups (left panel) ───────────────────────────────
const indicatorGroups = [
  { label: 'US INDICES', color: '#44a0ff', keys: ['sp500', 'nasdaq', 'dow', 'russell2000'] },
  { label: 'GLOBAL INDICES', color: '#a078ff', keys: ['nifty50', 'sensex', 'ftse100', 'nikkei225', 'hangseng', 'dax'] },
  { label: 'COMMODITIES', color: '#ffbe55', keys: ['gold', 'silver', 'oil_wti', 'oil_brent', 'natural_gas', 'copper'] },
  { label: 'CRYPTO', color: '#ff8c42', keys: ['btc_usd', 'eth_usd'] },
];

const allLabels = {
  // US indices
  sp500: 'S&P 500', nasdaq: 'Nasdaq', dow: 'Dow Jones', russell2000: 'Russell 2K', vix: 'VIX',
  // Global indices
  nifty50: 'Nifty 50', sensex: 'Sensex', ftse100: 'FTSE 100',
  nikkei225: 'Nikkei 225', hangseng: 'Hang Seng', dax: 'DAX',
  // Commodities
  gold: 'Gold', silver: 'Silver', oil_wti: 'WTI Oil', oil_brent: 'Brent Oil',
  natural_gas: 'Nat Gas', copper: 'Copper',
  // Crypto
  btc_usd: 'Bitcoin', eth_usd: 'Ethereum',
  // Rates & yields (ticker)
  yield_10y: '10Y Yield', yield_2y: '2Y Yield', yield_3m: '3M Yield', yield_30y: '30Y Yield',
  yield_curve: '10Y-2Y Spd', yield_curve_10y3m: '10Y-3M Spd',
  fed_funds_rate: 'Fed Funds', real_rate_10y: '10Y TIPS', real_rate_proxy: 'Real Rate',
  inflation_cpi: 'CPI', breakeven_10y: '10Y BE',
  // FX
  dxy: 'DXY', eur_usd: 'EUR/USD', usd_jpy: 'USD/JPY', usd_inr: 'USD/INR',
  gbp_usd: 'GBP/USD', usd_cny: 'USD/CNY',
  // Credit & risk
  credit_hy: 'HY Spread', credit_ig: 'IG Spread', ted_spread: 'TED Spd',
  mort_rate_30y: '30Y Mtg',
  // Macro
  gdp_growth: 'GDP', unemployment: 'Unemp.', pmi_mfg: 'PMI Mfg',
  consumer_sentiment: 'Sentiment',
  // Additional Ticker labels
  sp500: 'S&P 500', nasdaq: 'NASDAQ', dow: 'DOW JONES', russell2000: 'RUSSELL 2K',
  vix: 'VIX INDEX',
  gold: 'GOLD', silver: 'SILVER', btc_usd: 'BTC/USD', eth_usd: 'ETH/USD',
  oil_wti: 'WTI CRUDE',
};

function renderIndicatorControls(snapshot) {
  const grid = document.getElementById('indicatorGrid');
  if (!grid) return;
  const byKey = {};
  (snapshot.critical_indicators || []).forEach(i => byKey[i.key] = i);

  let html = '';
  indicatorGroups.forEach(group => {
    // Elegant group header with Radiant accent
    html += `<div class="col-span-full text-[10px] font-bold text-primary/50 uppercase tracking-[0.2em] mb-2 mt-4 border-b border-outline-variant/10 pb-1">${group.label}</div>`;
    
    group.keys.forEach(key => {
      const item = byKey[key] || { value: null, direction: 'flat', unit: '' };
      const dir = item.direction || 'flat';
      const val = (item.value === null || item.value === undefined || item.value === '') ? '\u2014' : item.value;
      
      const changeNum = (item.change !== undefined && item.change !== null)
        ? `<div class="text-[10px] font-mono ${dir}">${item.change > 0 ? '+' : ''}${typeof item.change === 'number' ? item.change.toFixed(2) : item.change}%</div>`
        : '';

      html += `
        <div class="indicator-card group hover:border-primary/30 transition-all duration-500" data-key="${key}">
          <div class="ind-name">${allLabels[key] || key}</div>
          <div class="ind-data">
            <div class="ind-val ${dir}">${val}${item.unit ? '&thinsp;' + item.unit : ''}</div>
            ${changeNum}
          </div>
        </div>`;
    });
  });
  grid.innerHTML = html;
}

function renderIndicatorCharts(snapshot) {
  // Chart holders removed — indicators rendered live via SSE stream
  Object.values(state.charts || {}).forEach(c => { try { c.destroy(); } catch (_) { } });
  state.charts = {};
}

function collectOverrides() {
  // Overrides are disabled for now as we are using charts
  return {};
}

function basePayload(questionOverride) {
  return {
    question: questionOverride || document.getElementById('question').value.trim(),
    geography: document.getElementById('geography').value,
    horizon: document.getElementById('horizon').value,
    response_mode: 'institutional',
    indicator_overrides: collectOverrides()
  };
}

function regimeClass(regimeName) {
  const r = (regimeName || '').toUpperCase();
  if (r.includes('GOLDILOCKS') || r.includes('RECOVERY') || r.includes('REFLATION')) return 'bull';
  if (r.includes('STAGFLATION') || r.includes('RECESSION') || r.includes('DEFLATION')) return 'bear';
  return 'warn';
}

function updateSnapshot(snapshot) {
  const regimeBadge = document.getElementById('regimeBadge');
  const signalBadge = document.getElementById('signalBadge');
  const modelBadge = document.getElementById('modelBadge');

  if (regimeBadge) {
    regimeBadge.textContent = `REGIME: ${snapshot.regime.regime || 'DETECTION...'}`;
    regimeBadge.style.borderColor = 'var(--glass-border-luminous)';
  }
  if (signalBadge) {
    signalBadge.textContent = `SIGNAL: ${snapshot.cross_asset.overall_signal || 'CALCULATING...'}`;
    signalBadge.style.borderColor = 'var(--glass-border-luminous)';
  }

  const detected = snapshot.detected_indicators || {};
  document.getElementById('detected').innerHTML = Object.keys(detected).sort().map((k) => `<span class="label">${k}:</span> ${detected[k]}`).join(' | ');
  renderEvidenceCoverage(snapshot.evidence_coverage || {});

  renderIndicatorControls(snapshot);
  if (modelBadge) modelBadge.textContent = 'INTEL: STANDBY';
}

function renderEvidenceCoverage(coverage) {
  const box = document.getElementById('evidenceCoverage');
  const chunks = coverage.context_chunks || 0;
  const overrideTag = coverage.has_overrides ? 'yes' : 'no';
  const sources = (coverage.sources || []).map((s, i) =>
    `${i + 1}. ${esc(s.title)} (${esc(s.source)} | ${esc(s.date)})`
  ).join('<br>');
  box.innerHTML = [
    `context_chunks: ${chunks}`,
    `manual_overrides: ${overrideTag}`,
    `top_sources:<br>${sources || 'No retrieved sources'}`
  ].join('<br>');
}

function resetCards() {
  document.getElementById('responseText').textContent = '';
  const rc = document.getElementById('responseCard');
  if (rc) rc.open = true;
}

function setGenStatus(text, cls) {
  const el = document.getElementById('genStatus');
  el.className = `gen-status ${cls || ''}`.trim();
  el.textContent = text;
}

function startGeneration(stageText) {
  const g = state.generation;
  g.running = true;
  g.section = '';
  g.startedAt = Date.now();
  g.chars = 0;
  if (g.timer) clearInterval(g.timer);
  setGenStatus(stageText || 'Synthesizing macro vectors...', 'running');
  g.timer = setInterval(() => {
    if (!g.running) return;
    const secs = ((Date.now() - g.startedAt) / 1000).toFixed(1);
    const section = g.section ? `${g.section}` : 'Processing';
    setGenStatus(`Radiant Stream: ${section} | ${g.chars} Chars | ${secs}s`, 'running');
  }, 300);
}

function touchGeneration(section, textChunk) {
  const g = state.generation;
  if (!g.running) return;
  g.section = section || g.section;
  g.chars += (textChunk || '').length;
}

function finishGeneration() {
  const g = state.generation;
  g.running = false;
  if (g.timer) clearInterval(g.timer);
  const secs = ((Date.now() - g.startedAt) / 1000).toFixed(1);
  setGenStatus(`Completed in ${secs}s | total streamed: ${g.chars} chars`, '');
}

function failGeneration(msg) {
  const g = state.generation;
  g.running = false;
  if (g.timer) clearInterval(g.timer);
  setGenStatus(`Generation error: ${msg}`, 'error');
}

function stringifyErrorDetail(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);

  if (Array.isArray(value)) {
    return value
      .map((item) => stringifyErrorDetail(item))
      .filter((item) => item && item.trim())
      .join(' | ');
  }

  if (typeof value === 'object') {
    const msg = (typeof value.message === 'string' && value.message.trim())
      ? value.message.trim()
      : ((typeof value.msg === 'string' && value.msg.trim()) ? value.msg.trim() : '');
    if (msg) {
      const loc = Array.isArray(value.loc) ? value.loc.join('.') : '';
      return loc ? `${loc}: ${msg}` : msg;
    }
    if (value.detail !== undefined) {
      const nested = stringifyErrorDetail(value.detail);
      if (nested) return nested;
    }
    try {
      return JSON.stringify(value);
    } catch (_) {
      return String(value);
    }
  }

  return String(value);
}

function toErrorMessage(error, fallback = 'Unknown error') {
  if (error instanceof Error && error.message) return error.message;
  const text = stringifyErrorDetail(error);
  return text || fallback;
}

async function parseError(res) {
  const fallback = `${res.status} ${res.statusText}`.trim();
  let raw = '';
  try {
    raw = await res.text();
  } catch (_) {
    return fallback;
  }
  if (!raw) return fallback;

  try {
    const body = JSON.parse(raw);
    const detail = (body && body.detail !== undefined)
      ? stringifyErrorDetail(body.detail)
      : stringifyErrorDetail(body);
    return detail || fallback;
  } catch (_) {
    return raw.trim() || fallback;
  }
}

function esc(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatCitations(line) {
  return line.replace(/\[S\d+\]/g, (m) => `<span class="cite">${m}</span>`);
}

function formatSectionToHtml(text) {
  if (typeof marked !== 'undefined') {
    return marked.parse(String(text || ''));
  }
  // Fallback if marked is somehow missing
  const lines = String(text || '').split('\n').filter((x) => x.trim().length > 0);
  return lines.map((line) => `<div class="line">${formatCitations(esc(line))}</div>`).join('');
}

function pickLine(text, startsWith) {
  const line = String(text || '').split('\n').find((x) => x.trim().toLowerCase().startsWith(startsWith.toLowerCase()));
  if (!line) return '';
  const ix = line.indexOf(':');
  return ix >= 0 ? line.slice(ix + 1).trim() : line.trim();
}

function renderPlainSummary(finalPayload) {
  const response = finalPayload?.response_text || '';
  const struct = finalPayload?.response_struct || {};
  const keyPoints = Array.isArray(finalPayload?.key_points) ? finalPayload.key_points : [];

  const findKeyPoint = (title) => {
    const hit = keyPoints.find((p) => p && p.title === title);
    return hit?.text || '';
  };

  const big = findKeyPoint('Executive summary') || struct.executive_summary || pickLine(response, 'Executive summary') || 'No clear answer generated.';
  const marketImpact = findKeyPoint('Market impact') || '';
  const actionDirect = findKeyPoint('Direct answer') || struct.direct_answer || pickLine(response, 'Direct answer') || 'Insufficient custom evidence.';
  const actionPlan = findKeyPoint('Action plan') || '';
  const riskMain = findKeyPoint('Main risks') || struct.main_risks || pickLine(response, 'Main risks') || 'Insufficient custom evidence.';
  const watchNext = findKeyPoint('What to watch next') || '';

  const action = [actionDirect, actionPlan].filter(Boolean).join('\n');
  const risk = [riskMain, watchNext].filter(Boolean).join('\n');

  const q = finalPayload?.quality || {};
  const quality = `Model quality ${q.band || 'UNKNOWN'} (${q.score || 0}/100), citations=${q.citation_count || 0}, context_chunks=${q.context_chunks || 0}.`;

  // Live-data clarity (when available)
  const liveMeta = finalPayload?.snapshot?.live_data_meta || {};
  const liveFetchMs = liveMeta.fetch_ms;
  const liveFromCache = liveMeta.from_cache;
  const liveHint = [
    liveFetchMs != null ? `live_fetch=${liveFetchMs}ms` : null,
    liveFromCache === true ? 'live_from_cache=true' : null
  ].filter(Boolean).join(' | ');

  document.getElementById('summaryBigPicture').textContent = marketImpact ? `${big}\n\nMarket impact: ${marketImpact}` : big;
  document.getElementById('summaryAction').textContent = action;
  document.getElementById('summaryRisk').textContent = risk;
  document.getElementById('summaryQuality').textContent = liveHint ? `${quality} | ${liveHint}` : quality;
}

function renderFinal(finalPayload, question) {
  updateSnapshot(finalPayload.snapshot);
  renderPlainSummary(finalPayload);
  const modelBadge = document.getElementById('modelBadge');
  if (modelBadge) {
    const modelUsed = finalPayload?.model_used || 'N/A';
    modelBadge.textContent = `MODEL: ${modelUsed}`;
  }

  const responseRaw = finalPayload.response_text || '';
  document.getElementById('responseText').innerHTML = formatSectionToHtml(responseRaw);
  
  // Smooth scroll to top of narrative on complete
  const scroller = document.querySelector('.viewport-scroller') || window;
  scroller.scrollTo({ top: 300, behavior: 'smooth' });

  // Show feedback bar and clear previous status
  const fb = document.getElementById('feedbackBar');
  if (fb) fb.style.display = 'flex';
  const fbs = document.getElementById('feedbackStatus');
  if (fbs) fbs.textContent = '';

  state.latest = finalPayload;
  state.history.unshift({ question, payload: finalPayload, ts: new Date().toISOString() });
  if (state.history.length > 20) state.history.pop();
  _saveHistory();
  renderHistory();
}

function buildComparePane(payload, question) {
  const summary = payload.response_struct?.executive_summary || pickLine(payload.response_text || '', 'Executive summary') || 'No concise summary generated.';
  return [
    `Question: ${question}`,
    `Model: ${payload.model_used || 'N/A'}`,
    `Summary: ${summary}`,
    `Regime: ${payload.snapshot.regime.regime} (${payload.snapshot.regime.confidence})`,
    `Cross-Asset: ${payload.snapshot.cross_asset.overall_signal}`,
    '',
    'RESPONSE',
    payload.response_text || ''
  ].join('\n');
}

function renderHistory() {
  const histEl = document.getElementById('history');
  if (!histEl) return;
  if (!state.history.length) {
    histEl.innerHTML = '<div style="color:var(--text-dim);font-size:11px;">No history yet.</div>';
    return;
  }
  histEl.innerHTML = state.history.map((h, idx) => {
    const regClass = (() => {
      const r = (h.regime || h.payload?.snapshot?.regime?.regime || '').toUpperCase();
      if (r.includes('GOLDILOCKS') || r.includes('RECOVERY') || r.includes('REFLATION')) return 'bull';
      if (r.includes('STAGFLATION') || r.includes('RECESSION') || r.includes('DEFLATION')) return 'bear';
      return 'warn';
    })();
    const qBand = (h.quality || h.payload?.quality?.band || '').toUpperCase();
    const qClass = qBand === 'HIGH' ? 'hi' : (qBand === 'MEDIUM' ? 'med' : (qBand === 'LOW' ? 'lo' : ''));
    const regime = h.regime || h.payload?.snapshot?.regime?.regime || '';
    const score = h.score ?? h.payload?.quality?.score ?? '';
    const model = h.model || h.payload?.model_used || '';
    const tsShort = h.ts ? new Date(h.ts).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : '';
    return `<div class="history-item" data-idx="${idx}" title="Click to re-run">
          <div>${esc(h.question)}</div>
          <div class="history-meta">
            ${regime ? `<span class="h-badge ${regClass}">${esc(regime)}</span>` : ''}
            ${qBand ? `<span class="h-badge ${qClass}">${qBand}${score ? ' ' + score : ''}</span>` : ''}
            ${model ? `<span class="h-badge">${esc(model.slice(0, 22))}</span>` : ''}
            ${tsShort ? `<span class="h-badge">${tsShort}</span>` : ''}
          </div>
        </div>`;
  }).join('');
  [...histEl.querySelectorAll('.history-item')].forEach((el) => {
    el.addEventListener('click', () => {
      const item = state.history[parseInt(el.dataset.idx, 10)];
      if (!item) return;
      document.getElementById('question').value = item.question;
      if (item.payload) {
        renderFinal(item.payload, item.question);
      } else {
        // Full payload not stored — re-run the query
        runMain().catch(() => { });
      }
    });
  });
}

async function runMain() {
  if (state.generation.running) return;
  const payload = basePayload();
  if (!payload.question) return;

  document.getElementById('compareWrap').style.display = 'none';
  document.getElementById('analysisZone').style.display = 'block';
  resetCards();
  startGeneration('Building regime snapshot...');

  const streamRes = await fetch('/intelligence/stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });

  if (!streamRes.ok || !streamRes.body) {
    throw new Error(`Streaming endpoint failed: ${await parseError(streamRes)}`);
  }

  const reader = streamRes.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let responseRaw = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const chunk of events) {
      const lines = chunk.split('\n');
      let evt = '';
      let data = '';
      for (const line of lines) {
        if (line.startsWith('event:')) evt = line.slice(6).trim();
        if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!evt || !data) continue;
      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch (_) {
        parsed = { message: data };
      }

      if (evt === 'snapshot') updateSnapshot(parsed);
      if (evt === 'section_start') {
        const card = document.getElementById('responseCard');
        if (card) card.open = true;
        touchGeneration(parsed.section, '');
      }
      if (evt === 'progress') {
        const stage = parsed?.stage || parsed?.error || 'working';
        setGenStatus(`Working: ${stage}`, 'running');
      }
      if (evt === 'token') {
        const target = document.getElementById('responseText');
        responseRaw += parsed.text || '';
        if (typeof marked !== 'undefined') {
          target.innerHTML = marked.parse(responseRaw);
        } else {
          target.innerText = responseRaw; 
        }
        touchGeneration(parsed.section, parsed.text);
      }
      if (evt === 'final') {
        renderFinal(parsed, payload.question);
        finishGeneration();
      }
      if (evt === 'error') {
        const msg = toErrorMessage((parsed && parsed.message !== undefined) ? parsed.message : parsed, 'Unknown stream error');
        document.getElementById('responseText').textContent += '\n[ERROR] ' + msg;
        failGeneration(msg);
      }
    }
  }
}

async function runCompare() {
  const q2 = document.getElementById('compareQuestion').value.trim();
  const q1 = document.getElementById('question').value.trim();
  if (!q1 || !q2) return;

  document.getElementById('analysisZone').style.display = 'none';
  document.getElementById('compareWrap').style.display = 'grid';
  document.getElementById('compareA').textContent = 'Running first scenario...';
  document.getElementById('compareB').textContent = 'Running second scenario...';

  const aPayload = basePayload(q1);
  const bPayload = basePayload(q2);

  const [aRes, bRes] = await Promise.all([
    fetch('/intelligence/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(aPayload) }),
    fetch('/intelligence/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(bPayload) })
  ]);
  if (!aRes.ok || !bRes.ok) {
    const msgA = !aRes.ok ? await parseError(aRes) : '';
    const msgB = !bRes.ok ? await parseError(bRes) : '';
    throw new Error(`Compare failed: ${msgA || msgB}`);
  }

  const [a, b] = await Promise.all([aRes.json(), bRes.json()]);
  document.getElementById('compareA').textContent = buildComparePane(a, q1);
  document.getElementById('compareB').textContent = buildComparePane(b, q2);
}

// ── Feedback ──────────────────────────────────────────────────────────────
async function submitFeedback(rating) {
  const question = document.getElementById('question').value.trim();
  const answerEl = document.getElementById('responseText');
  const answerSnippet = (answerEl ? answerEl.textContent : '').slice(0, 300);
  const statusEl = document.getElementById('feedbackStatus');
  if (!question) { if (statusEl) statusEl.textContent = 'No question to attach feedback to.'; return; }
  try {
    const res = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer_snippet: answerSnippet, rating, comment: '' }),
    });
    if (statusEl) statusEl.textContent = res.ok ? '✓ Thanks for your feedback!' : '✗ Could not save feedback.';
    setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 4000);
  } catch (e) {
    if (statusEl) statusEl.textContent = '✗ ' + toErrorMessage(e);
  }
}

// ── Company Fundamentals lookup ───────────────────────────────────────────
async function lookupFundamentals() {
  const tickerEl = document.getElementById('fundamentalsTicker');
  const resultEl = document.getElementById('fundamentalsResult');
  const ticker = (tickerEl ? tickerEl.value.trim().toUpperCase() : '');
  if (!ticker) { if (resultEl) resultEl.textContent = 'Please enter a ticker symbol.'; return; }
  if (resultEl) resultEl.textContent = `Fetching fundamentals for ${ticker}…`;
  try {
    const res = await fetch(`/fundamentals/${encodeURIComponent(ticker)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const msg = stringifyErrorDetail(err?.detail !== undefined ? err.detail : err) || res.statusText;
      resultEl.textContent = `Error: ${msg}`;
      return;
    }
    const d = await res.json();
    const lines = [
      `${d.ticker} — ${d.name}`,
      `Sector: ${d.sector || 'N/A'}  |  Industry: ${d.industry || 'N/A'}`,
      `Market Cap: ${_fmtLarge(d.market_cap)}  |  P/E: ${d.pe_ratio ?? 'N/A'}  |  Fwd P/E: ${d.forward_pe ?? 'N/A'}`,
      `EPS(TTM): ${d.eps_ttm != null ? '$' + d.eps_ttm.toFixed(2) : 'N/A'}  |  Revenue: ${_fmtLarge(d.revenue_ttm)}`,
      `Profit Margin: ${d.profit_margins != null ? (d.profit_margins * 100).toFixed(1) + '%' : 'N/A'}  |  FCF: ${_fmtLarge(d.free_cashflow)}`,
      `Debt/Equity: ${d.debt_to_equity ?? 'N/A'}  |  Beta: ${d.beta ?? 'N/A'}`,
      `52W: $${d['52w_low'] ?? 'N/A'} – $${d['52w_high'] ?? 'N/A'}  |  Current: $${d.current_price ?? 'N/A'}`,
      `Analyst: ${d.analyst_recommendation || 'N/A'} (${d.analyst_count ?? '?'})  |  Target: ${d.analyst_target != null ? '$' + d.analyst_target.toFixed(2) : 'N/A'}`,
      `Fetched: ${d.fetched_at || ''}`,
    ];
    resultEl.textContent = lines.join('\n');
  } catch (e) {
    resultEl.textContent = 'Error: ' + toErrorMessage(e);
  }
}

function _fmtLarge(v) {
  if (v == null) return 'N/A';
  v = Number(v);
  if (Math.abs(v) >= 1e12) return '$' + (v / 1e12).toFixed(2) + 'T';
  if (Math.abs(v) >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
  if (Math.abs(v) >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
  return '$' + v.toFixed(2);
}

// ── Hedge Analysis ────────────────────────────────────────────────────────
async function runHedge() {
  const question = document.getElementById('question').value.trim();
  const tickersRaw = (document.getElementById('hedgeTickers')?.value || '').trim();
  const resultEl = document.getElementById('hedgeResult');
  if (!question) { if (resultEl) resultEl.textContent = 'Enter a question in the Command panel first.'; return; }
  if (resultEl) resultEl.textContent = 'Running hedge analysis…';
  const tickers = tickersRaw ? tickersRaw.split(/[,\s]+/).map(t => t.toUpperCase()).filter(Boolean) : [];
  const geography = document.getElementById('geography')?.value || 'US';
  const horizon = document.getElementById('horizon')?.value || 'MEDIUM_TERM';
  try {
    const res = await fetch('/intelligence/hedge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, tickers, geography, horizon }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const msg = stringifyErrorDetail(err?.detail !== undefined ? err.detail : err) || res.statusText;
      resultEl.textContent = `Error: ${msg}`;
      return;
    }
    const d = await res.json();
    const lines = [
      `Regime: ${d.regime?.regime || 'N/A'} (${d.regime?.confidence || ''})`,
      `Cross-Asset: ${d.cross_asset?.overall_signal || 'N/A'}`,
      `Model: ${d.model_used || 'N/A'}  |  ${d.latency_ms}ms`,
      '',
      d.response_text || 'No response generated.',
    ];
    resultEl.textContent = lines.join('\n');
  } catch (e) {
    resultEl.textContent = 'Error: ' + toErrorMessage(e);
  }
}

async function exportNote() {
  const payload = basePayload();
  if (!payload.question) return;
  const res = await fetch('/intelligence/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const text = await res.text();
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'morning_note.txt';
  a.click();
  URL.revokeObjectURL(url);
}

document.getElementById('runBtn').addEventListener('click', () => {
  console.log("runBtn clicked");
  runMain().catch((e) => {
    const msg = toErrorMessage(e);
    document.getElementById('responseText').textContent += `\n[ERROR] ${msg}`;
    failGeneration(msg);
  });
});
document.getElementById('compareBtn').addEventListener('click', () => {
  console.log("compareBtn clicked");
  runCompare().catch((e) => {
    const msg = toErrorMessage(e);
    document.getElementById('compareA').textContent = '[ERROR] ' + msg;
    document.getElementById('compareB').textContent = '[ERROR] ' + msg;
  });
});
document.getElementById('exportBtn').addEventListener('click', () => {
  console.log("exportBtn clicked");
  exportNote().catch((e) => {
    document.getElementById('responseText').textContent += `\n[ERROR] ${toErrorMessage(e)}`;
  });
});

document.getElementById('question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    runMain().catch((e) => {
      const msg = toErrorMessage(e);
      document.getElementById('responseText').textContent += `\n[ERROR] ${msg}`;
      failGeneration(msg);
    });
  }
});

// ── User timezone & live clock ───────────────────────────────────────────
const _userTZ = Intl.DateTimeFormat().resolvedOptions().timeZone;

/** Format a Date in the user's local timezone, showing time + tz abbreviation */
function _localTime(date) {
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZoneName: 'short', timeZone: _userTZ
  });
}

/** Format a Date as a short date+time in the user's local timezone */
function _localDateTime(date) {
  return date.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZoneName: 'short', timeZone: _userTZ
  });
}

/** Returns session label for a given exchange clock at current UTC */
function _sessionStatus(openHourUTC, closeHourUTC, label) {
  const now = new Date();
  const h = now.getUTCHours() + now.getUTCMinutes() / 60;
  const dayOfWeek = now.getUTCDay(); // 0=Sun, 6=Sat
  if (dayOfWeek === 0 || dayOfWeek === 6) return `${label} CLOSED`;
  // Handle overnight sessions (open > close, e.g. Asian markets relative to UTC)
  let open = openHourUTC <= closeHourUTC
    ? (h >= openHourUTC && h < closeHourUTC)
    : (h >= openHourUTC || h < closeHourUTC);
  return `${label} ${open ? '<span class="sess-open">OPEN</span>' : '<span class="sess-closed">CLOSED</span>'}`;
}

function _buildClockHTML() {
  const now = new Date();
  const timeStr = _localTime(now);
  // Key exchange sessions expressed in UTC hours (approximate, no DST adjustment)
  const sessions = [
    _sessionStatus(13.5, 20, 'NYSE'),    // 09:30–16:00 ET  = 13:30–20:00 UTC
    _sessionStatus(8, 16.5, 'LSE'),     // 08:00–16:30 UTC
    _sessionStatus(3.5, 10, 'BSE'),      // 09:15–15:30 IST = 03:45–10:00 UTC
    _sessionStatus(0, 6, 'TSE'),       // 09:00–15:00 JST = 00:00–06:00 UTC
  ];
  return `<span class="clock-time">${timeStr}</span>&ensp;` +
    sessions.map(s => `<span class="sess-tag">${s}</span>`).join('&ensp;');
}

const _clockEl = document.getElementById('liveClock');
function _tickClock() {
  if (_clockEl) _clockEl.innerHTML = _buildClockHTML();
}
setInterval(_tickClock, 1000);
_tickClock();

// ── Bootstrap: connect SSE directly — partial events populate screen progressively ─
let _liveES = null;
let _nextRefreshTimer = null;
let _nextRefreshEnd = 0;
// ── Ticker & Instant UI Bootstrap ────────────────────────────────────────
(function bootstrapInstantUI() {
    _loadHistory();
    renderHistory();

    // Optimistic UI Data Loading from Cache or Default Fallback
    try {
        const cacheStr = localStorage.getItem('radiantMarketCache');
        if (cacheStr) {
            const cachedData = JSON.parse(cacheStr);
            cachedData.from_cache = true;
            handleLiveUpdate(cachedData); 
        } else {
            // Zero-delay fallback for first-time visits
            const DEFAULT_MARKET_DATA = {
                from_cache: true,
                indicators: {
                    'yield_10y': { label: '10Y Yield', value: '4.42', unit: '%', direction: 'down', change: -0.05 },
                    'inflation_cpi': { label: 'US CPI', value: '3.10', unit: '%', direction: 'flat', change: 0 },
                    'dxy': { label: 'DXY Index', value: '104.20', unit: '', direction: 'up', change: 0.15 },
                    'oil_wti': { label: 'Crude Oil', value: '82.50', unit: '$', direction: 'up', change: 1.2 },
                    'fed_funds_rate': { label: 'Fed Funds', value: '5.25', unit: '%', direction: 'flat', change: 0 },
                    'vix': { label: 'VIX Volatility', value: '14.20', unit: '', direction: 'down', change: -0.8 },
                    'yield_2y': { label: '2Y Yield', value: '4.85', unit: '%', direction: 'flat', change: 0 },
                    'credit_hy': { label: 'HY Spread', value: '3.42', unit: '%', direction: 'up', change: 0.11 }
                }
            };
            handleLiveUpdate(DEFAULT_MARKET_DATA);
        }
    } catch (e) {}

    connectLiveStream();

    // Add Ticker Interaction to the parent container
    const tickerContainer = document.querySelector('.ticker-container') || document.querySelector('.glass-ticker');
    const tickerTrack = document.getElementById('tickerTrack');
    if (tickerContainer && tickerTrack) {
        tickerContainer.addEventListener('click', () => {
            console.log("Ticker click - toggling pause");
            tickerTrack.classList.toggle('paused');
        });
    }
})();

// ── Live market data (SSE stream) ────────────────────────────────────────
// ── Ticker shows macro/rates/FX data (steady updates, not tick-by-tick markets) ────
const tickerOrder = [
  'sp500', 'nasdaq', 'dow', 'vix',
  'yield_10y', 'yield_2y', 'yield_curve',
  'fed_funds_rate', 'inflation_cpi',
  'dxy', 'eur_usd', 'usd_jpy',
  'gold', 'oil_wti', 'btc_usd', 'eth_usd',
  'credit_hy', 'ted_spread'
];

function _dirSign(dir) {
  return dir === 'up' ? '▲' : (dir === 'down' ? '▼' : '■');
}

function updateTickerBar(inds) {
  const track = document.getElementById('tickerTrack');
  if (!track) return;
  const items = tickerOrder.map(key => {
    const v = inds[key];
    if (!v) return '';
    const val = v.value !== null && v.value !== undefined ? v.value : 'N/A';
    const dir = v.direction || 'flat';
    const chgEle = (v.change !== null && v.change !== undefined)
      ? `<span class="ticker-val ${dir}">${v.change > 0 ? '+' : ''}${v.change.toFixed(2)}</span>`
      : `<span class="ticker-val">--</span>`;
    return `<div class="ticker-item">
      <span class="ticker-label">${allLabels[key] || key}</span>
      <span class="ticker-val">${val}${v.unit ? '&thinsp;' + v.unit : ''}</span>
      ${chgEle}
    </div>`;
  }).filter(Boolean).join('');
  
  // Update HTML and reset animation duration
  track.innerHTML = items + items; // 2 copies for -50% loop
  
  // Calculate duration once layout settles
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
        const halfWidth = track.scrollWidth / 2;
        if (halfWidth > 0) {
            const duration = Math.max(30, halfWidth / 50); // 50px/s
            track.style.animationDuration = `${duration}s`;
            console.log(`Ticker duration set: ${duration}s`);
        }
    });
  });
}

function handleLiveUpdate(data) {
  const inds = data.indicators || {};
  const isPartial = data.partial === true;
  const source = data.source || '';
  const completed = data.completed_sources || [];
  const pending = data.pending_sources || [];
  const fromCache = data.from_cache === true;

  // Build synthetic snapshot from whatever indicators we have so far
  const syntheticSnapshot = {
    critical_indicators: Object.entries(inds).map(([key, v]) => ({
      key,
      label: v.label,
      value: v.value,
      unit: v.unit,
      direction: v.direction || 'flat',
      as_of: v.as_of || '',
      change: v.change,
    }))
  };
  renderIndicatorControls(syntheticSnapshot);

  // Flash changed indicators
  Object.entries(inds).forEach(([key, v]) => {
    if (v.direction && v.direction !== 'flat') {
      const el = document.querySelector(`.indicator-card[data-key="${key}"]`);
      if (el) {
        el.style.borderColor = v.direction === 'up' ? 'var(--accent-success)' : 'var(--accent-error)';
        setTimeout(() => { if (el) el.style.borderColor = 'var(--glass-border)'; }, 1000);
      }
    }
  });

  // Update ticker bar with whatever is available
  updateTickerBar(inds);

  // Cache data to LocalStorage for instant optimistic loading on next visit
  if (Object.keys(inds).length > 2) {
      try {
          localStorage.setItem('radiantMarketCache', JSON.stringify({ indicators: inds }));
      } catch (e) {}
  }

  // Update the status label to show source progress
  const stamp = document.querySelector('.status-indicator .label');
  if (stamp) {
    if (fromCache) {
      stamp.innerHTML = `RADIANT LIVE &middot; CACHED`;
      stamp.style.color = 'var(--text-secondary)';
    } else if (isPartial) {
      stamp.innerHTML = `SYNCING: ${source.toUpperCase()}...`;
      stamp.style.color = 'var(--accent-primary)';
    } else {
      stamp.innerHTML = `RADIANT LIVE`;
      stamp.style.color = 'var(--accent-success)';
    }
  }
}

function _startRefreshCountdown(sleepS) {
  // Radiant style: subtle pulse instead of countdown text for cleaner UI
}

function connectLiveStream() {
  if (_liveES) { _liveES.close(); _liveES = null; }
  const stamp = document.querySelector('.status-indicator .label');
  if (stamp) stamp.innerHTML = `CONNECTING...`;
  const es = new EventSource('/market_data/stream');
  _liveES = es;
  es.addEventListener('update', (e) => {
    try { handleLiveUpdate(JSON.parse(e.data)); } catch (err) { console.warn('Live parse error:', err); }
  });
  es.addEventListener('tick', (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.next_refresh_s) _startRefreshCountdown(d.next_refresh_s);
    } catch (_) { }
  });
  es.addEventListener('error', () => {
    console.warn('Live stream interrupted — reconnecting in 10 s');
    es.close(); _liveES = null;
    setTimeout(connectLiveStream, 10000);
  });
}
