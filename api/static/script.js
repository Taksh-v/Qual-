console.log("script.js loaded");
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

    // ── Critical indicator groups (left panel) ───────────────────────────────
    const indicatorGroups = [
      { label: 'US INDICES',     color: '#44a0ff', keys: ['sp500', 'nasdaq', 'dow', 'russell2000'] },
      { label: 'GLOBAL INDICES', color: '#a078ff', keys: ['nifty50', 'sensex', 'ftse100', 'nikkei225', 'hangseng', 'dax'] },
      { label: 'COMMODITIES',    color: '#ffbe55', keys: ['gold', 'silver', 'oil_wti', 'oil_brent', 'natural_gas', 'copper'] },
      { label: 'CRYPTO',         color: '#ff8c42', keys: ['btc_usd', 'eth_usd'] },
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
    };

    function renderIndicatorControls(snapshot) {
      const grid = document.getElementById('indicatorGrid');
      const byKey = {};
      (snapshot.critical_indicators || []).forEach(i => byKey[i.key] = i);

      let html = '';
      indicatorGroups.forEach(group => {
        html += `<div class="ind-group-header" style="--grp:${group.color}">${group.label}</div>`;
        group.keys.forEach(key => {
          const item = byKey[key] || { value: null, direction: 'flat', unit: '' };
          const dir = item.direction || 'flat';
          const arrow = dir === 'up' ? '\u25b2' : (dir === 'down' ? '\u25bc' : '\u2013');
          const val = (item.value === null || item.value === undefined || item.value === '') ? '\u2014' : item.value;
          const changeNum = (item.change !== undefined && item.change !== null)
            ? `<span class="indicator-change ${dir}">${item.change > 0 ? '+' : ''}${typeof item.change === 'number' ? item.change.toFixed(2) : item.change}</span>`
            : '';
          html += `<div class="indicator" data-key="${key}" style="--grp:${group.color}">
            <div class="indicator-label">
              <span class="indicator-name">${allLabels[key] || key}</span>
              <span class="indicator-value">${val}${item.unit ? '<span class="indicator-unit"> ' + item.unit + '</span>' : ''}</span>
              ${changeNum}
            </div>
            <div class="indicator-arrow ${dir}">${arrow}</div>
          </div>`;
        });
      });
      grid.innerHTML = html;
    }

    function renderIndicatorCharts(snapshot) {
      // Chart holders removed — indicators rendered live via SSE stream
      Object.values(state.charts || {}).forEach(c => { try { c.destroy(); } catch (_) {} });
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
        response_mode: document.getElementById('responseMode').value,
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
      document.getElementById('regimeBadge').className = `badge regime ${regimeClass(snapshot.regime.regime)}`;
      document.getElementById('regimeBadge').textContent = `REGIME: ${snapshot.regime.regime}`;
      document.getElementById('signalBadge').textContent = `CROSS-ASSET: ${snapshot.cross_asset.overall_signal}`;

      const detected = snapshot.detected_indicators || {};
      document.getElementById('detected').innerHTML = Object.keys(detected).sort().map((k) => `${k}: ${detected[k]}`).join('<br>');
      renderEvidenceCoverage(snapshot.evidence_coverage || {});

      const banner = document.getElementById('confidenceBanner');
      const zone = document.getElementById('analysisZone');
      if ((snapshot.regime.confidence || '').toUpperCase() === 'LOW') {
        banner.style.display = 'block';
        banner.textContent = `Low confidence: missing ${snapshot.regime.missing_inputs.join(', ') || 'key indicators'}.`;
        zone.classList.add('low-confidence');
      } else {
        banner.style.display = 'none';
        zone.classList.remove('low-confidence');
      }

      renderIndicatorControls(snapshot);
      renderIndicatorCharts(snapshot);
      const modelBadge = document.getElementById('modelBadge');
      if (modelBadge) modelBadge.textContent = 'MODEL: WAITING';
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
      document.getElementById('responseCard').open = true;
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
      setGenStatus(stageText || 'Analyzing...', 'running');
      g.timer = setInterval(() => {
        if (!g.running) return;
        const secs = ((Date.now() - g.startedAt) / 1000).toFixed(1);
        const section = g.section ? ` | section: ${g.section}` : '';
        setGenStatus(`Generating response${section} | streamed: ${g.chars} chars | ${secs}s`, 'running');
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

    async function parseError(res) {
      let detail = '';
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch (_) {
        detail = await res.text();
      }
      return detail || `${res.status} ${res.statusText}`;
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
      const lines = String(text || '').split('\n').filter((x) => x.trim().length > 0);
      return lines.map((line) => {
        const safe = esc(line);
        if (safe.includes(':')) {
          const idx = safe.indexOf(':');
          const head = safe.slice(0, idx + 1);
          const tail = safe.slice(idx + 1).trim();
          return `<div class="line"><span class="line-key">${head}</span> ${formatCitations(tail)}</div>`;
        }
        return `<div class="line">${formatCitations(safe)}</div>`;
      }).join('');
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
      const big = struct.executive_summary || pickLine(response, 'Executive summary') || 'No clear answer generated.';
      const action = (struct.direct_answer || pickLine(response, 'Direct answer') || 'Insufficient custom evidence.').trim();
      const risk = struct.main_risks || pickLine(response, 'Main risks') || 'Insufficient custom evidence.';
      const q = finalPayload?.quality || {};
      const quality = `Model quality ${q.band || 'UNKNOWN'} (${q.score || 0}/100), citations=${q.citation_count || 0}, context_chunks=${q.context_chunks || 0}.`;

      document.getElementById('summaryBigPicture').textContent = big;
      document.getElementById('summaryAction').textContent = action;
      document.getElementById('summaryRisk').textContent = risk;
      document.getElementById('summaryQuality').textContent = quality;
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

      state.latest = finalPayload;
      state.history.unshift({ question, payload: finalPayload, ts: new Date().toISOString() });
      if (state.history.length > 12) state.history.pop();
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
      const history = document.getElementById('history');
      history.innerHTML = state.history.map((h, idx) => `<div class="history-item" data-idx="${idx}">${h.question}<br><small>${h.ts}</small></div>`).join('');
      [...history.querySelectorAll('.history-item')].forEach((el) => {
        el.addEventListener('click', () => {
          const item = state.history[parseInt(el.dataset.idx, 10)];
          if (!item) return;
          document.getElementById('question').value = item.question;
          renderFinal(item.payload, item.question);
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

      const snapRes = await fetch('/intelligence/snapshot', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      if (!snapRes.ok) {
        throw new Error(`Snapshot failed: ${await parseError(snapRes)}`);
      }
      const snap = await snapRes.json();
      updateSnapshot(snap);
      setGenStatus('Snapshot ready. Streaming analysis...', 'running');

      const streamRes = await fetch('/intelligence/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });

      if (!streamRes.ok || !streamRes.body) {
        throw new Error(`Streaming endpoint failed: ${await parseError(streamRes)}`);
      }

      const reader = streamRes.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

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
          const parsed = JSON.parse(data);

          if (evt === 'snapshot') updateSnapshot(parsed);
          if (evt === 'section_start') {
            const card = document.getElementById('responseCard');
            if (card) card.open = true;
            touchGeneration(parsed.section, '');
          }
          if (evt === 'token') {
            const target = document.getElementById('responseText');
            if (target) target.textContent += parsed.text;
            touchGeneration(parsed.section, parsed.text);
          }
          if (evt === 'final') {
            renderFinal(parsed, payload.question);
            finishGeneration();
          }
          if (evt === 'error') {
            document.getElementById('responseText').textContent += '\n[ERROR] ' + parsed.message;
            failGeneration(parsed.message);
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
        document.getElementById('responseText').textContent += `\n[ERROR] ${e.message}`;
        failGeneration(e.message);
      });
    });
    document.getElementById('compareBtn').addEventListener('click', () => {
      console.log("compareBtn clicked");
      runCompare().catch((e) => {
        document.getElementById('compareA').textContent = '[ERROR] ' + e.message;
        document.getElementById('compareB').textContent = '[ERROR] ' + e.message;
      });
    });
    document.getElementById('exportBtn').addEventListener('click', () => {
      console.log("exportBtn clicked");
      exportNote().catch((e) => {
        document.getElementById('responseText').textContent += `\n[ERROR] ${e.message}`;
      });
    });

    document.getElementById('question').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        runMain().catch((e) => {
          document.getElementById('responseText').textContent += `\n[ERROR] ${e.message}`;
          failGeneration(e.message);
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
      const timeStr  = _localTime(now);
      // Key exchange sessions expressed in UTC hours (approximate, no DST adjustment)
      const sessions = [
        _sessionStatus(13.5, 20, 'NYSE'),    // 09:30–16:00 ET  = 13:30–20:00 UTC
        _sessionStatus(8,  16.5, 'LSE'),     // 08:00–16:30 UTC
        _sessionStatus(3.5, 10, 'BSE'),      // 09:15–15:30 IST = 03:45–10:00 UTC
        _sessionStatus(0,   6, 'TSE'),       // 09:00–15:00 JST = 00:00–06:00 UTC
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

    // ── Live market data (SSE stream) ────────────────────────────────────────
    // ── Ticker shows macro/rates/FX data (steady updates, not tick-by-tick markets) ────
    const tickerOrder = [
      'yield_10y', 'yield_2y', 'yield_3m', 'yield_curve',
      'fed_funds_rate', 'real_rate_10y', 'inflation_cpi', 'breakeven_10y',
      'dxy', 'eur_usd', 'usd_jpy', 'usd_inr',
      'credit_hy', 'credit_ig', 'ted_spread',
      'vix', 'mort_rate_30y', 'oil_wti',
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
        const chg = (v.change !== null && v.change !== undefined)
          ? `<span class="t-chg ${dir}">${_dirSign(dir)} ${v.change > 0 ? '+' : ''}${v.change.toFixed(3)}</span>`
          : `<span class="t-chg flat">${_dirSign(dir)}</span>`;
        return `<span class="ticker-item"><span class="t-name">${allLabels[key] || key}</span><span class="t-val">${val}${v.unit ? '&thinsp;' + v.unit : ''}</span>${chg}</span>`;
      }).filter(Boolean).join('');
      // Duplicate for seamless loop scroll
      track.innerHTML = items + items;
      requestAnimationFrame(() => {
        const halfWidth = track.scrollWidth / 2;
        const duration = Math.max(20, halfWidth / 70); // 70px/s
        track.style.animationDuration = `${duration}s`;
      });
    }

    function handleLiveUpdate(data) {
      const inds = data.indicators || {};
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
      // Flash indicators whose value changed
      Object.entries(inds).forEach(([key, v]) => {
        if (v.direction && v.direction !== 'flat') {
          const indicator = document.querySelector(`.indicator[data-key="${key}"]`);
          if (indicator) {
            indicator.classList.remove('flash-up', 'flash-down');
            void indicator.offsetWidth; // force reflow to restart animation
            indicator.classList.add(`flash-${v.direction}`);
          }
        }
      });
      updateTickerBar(inds);
      const stamp = document.getElementById('marketDataStamp');
      if (stamp) {
        const ts = data.timestamp ? _localDateTime(new Date(data.timestamp)) : _localDateTime(new Date());
        stamp.innerHTML = `<span class="live-dot"></span>&nbsp;LIVE&nbsp;&middot;&nbsp;Updated: ${ts}`;
      }
    }

    let _liveES = null;
    function connectLiveStream() {
      if (_liveES) { _liveES.close(); _liveES = null; }
      const stamp = document.getElementById('marketDataStamp');
      if (stamp) stamp.innerHTML = `<span class="live-dot"></span>&nbsp;LIVE&nbsp;&middot;&nbsp;Connecting...`;
      const es = new EventSource('/market_data/stream');
      _liveES = es;
      es.addEventListener('update', (e) => {
        try { handleLiveUpdate(JSON.parse(e.data)); } catch (err) { console.warn('Live parse error:', err); }
      });
      es.addEventListener('error', () => {
        console.warn('Live stream interrupted — reconnecting in 10 s');
        es.close(); _liveES = null;
        setTimeout(connectLiveStream, 10000);
      });
    }

    // Bootstrap: fast initial fetch then open persistent SSE stream
    async function refreshMarketData() {
      try {
        const res = await fetch('/market_data');
        if (!res.ok) return;
        const data = await res.json();
        const inds = data.indicators || {};
        const syntheticSnapshot = {
          critical_indicators: Object.entries(inds).map(([key, v]) => ({
            key, label: v.label, value: v.value, unit: v.unit,
            direction: 'flat', as_of: v.as_of || '',
          }))
        };
        renderIndicatorControls(syntheticSnapshot);
        updateTickerBar(inds);
        const stamp = document.getElementById('marketDataStamp');
        if (stamp) stamp.innerHTML = `<span class="live-dot"></span>&nbsp;LIVE&nbsp;&middot;&nbsp;Loading stream...`;
      } catch (_) {}
    }

    refreshMarketData().then(() => connectLiveStream());

    fetch('/intelligence/snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: 'Initial dashboard status check', geography: 'US', horizon: 'MEDIUM_TERM', indicator_overrides: {} })
    }).then((r) => r.json()).then((snap) => {
      updateSnapshot(snap);
    });
