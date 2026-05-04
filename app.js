function kalmanModelParams(returns, tauL_days = 10) {
      const dt = 1 / 252;
      const tauL = tauL_days / 252;
      const F = 1.0 - dt / tauL;
      const varR = returns.reduce((sum, value) => sum + value * value, 0) / returns.length;
      const alpha = 1 - Math.exp(-dt / tauL);
      const Q = alpha * alpha * varR;
      const R = Math.max(varR * (1 - alpha) ** 2, 1e-10);
      return { dt, tauL, F, varR, alpha, Q, R };
    }

    function kalmanFilter(returns, tauL_days = 10) {
      const { F, varR, Q, R } = kalmanModelParams(returns, tauL_days);
      const v = new Array(returns.length).fill(0);
      let P = varR;
      let vEst = returns[0];

      for (let index = 0; index < returns.length; index += 1) {
        const vPred = F * vEst;
        const PPred = F * F * P + Q;
        const K = PPred / (PPred + R);
        vEst = vPred + K * (returns[index] - vPred);
        P = (1 - K) * PPred;
        v[index] = vEst;
      }

      return v;
    }

    function gaussianPDF(x, mu, sigma) {
      const d = (x - mu) / sigma;
      return Math.exp(-0.5 * d * d) / (sigma * Math.sqrt(2 * Math.PI));
    }

    function fitBimodalEM(data, maxIter = 60) {
      const n = data.length;
      const mean = data.reduce((a, b) => a + b, 0) / n;
      const std = Math.sqrt(data.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n) || 1e-8;
      const sk = data.reduce((sum, value) => sum + ((value - mean) / std) ** 3, 0) / n;

      function runEM(initLam1, initMu1, initS1, initLam2, initMu2, initS2, iters) {
        let lam1 = initLam1;
        let mu1 = initMu1;
        let s1 = initS1;
        let lam2 = initLam2;
        let mu2 = initMu2;
        let s2 = initS2;

        for (let iter = 0; iter < iters; iter += 1) {
          const r1 = new Array(n);
          const r2 = new Array(n);
          for (let index = 0; index < n; index += 1) {
            const p1 = lam1 * gaussianPDF(data[index], mu1, s1);
            const p2 = lam2 * gaussianPDF(data[index], mu2, s2);
            const total = p1 + p2 + 1e-300;
            r1[index] = p1 / total;
            r2[index] = p2 / total;
          }

          const N1 = r1.reduce((a, b) => a + b, 0);
          const N2 = r2.reduce((a, b) => a + b, 0);
          lam1 = N1 / n;
          lam2 = N2 / n;
          mu1 = r1.reduce((sum, weight, index) => sum + weight * data[index], 0) / (N1 + 1e-10);
          mu2 = r2.reduce((sum, weight, index) => sum + weight * data[index], 0) / (N2 + 1e-10);
          s1 = Math.sqrt(r1.reduce((sum, weight, index) => sum + weight * (data[index] - mu1) ** 2, 0) / (N1 + 1e-10));
          s2 = Math.sqrt(r2.reduce((sum, weight, index) => sum + weight * (data[index] - mu2) ** 2, 0) / (N2 + 1e-10));
          s1 = Math.max(s1, 1e-6);
          s2 = Math.max(s2, 1e-6);
        }

        if (mu1 > mu2) {
          return { lam1: lam2, mu1: mu2, s1: s2, lam2: lam1, mu2: mu1, s2: s1 };
        }

        return { lam1, mu1, s1, lam2, mu2, s2 };
      }

      let result = runEM(0.20, mean - Math.abs(sk) * std, std * 0.5, 0.80, mean + 0.1 * std, std * 0.8, maxIter);
      const spread = Math.abs(result.mu2 - result.mu1);
      if (result.lam1 < 0.05 || result.lam2 < 0.05 || spread < 0.15 * std) {
        const sorted = data.slice().sort((a, b) => a - b);
        const q20 = sorted[Math.floor(n * 0.20)];
        const q70 = sorted[Math.floor(n * 0.70)];
        result = runEM(0.20, q20, std * 0.5, 0.80, q70, std * 0.8, maxIter);
      }

      return result;
    }

    function regimeProbabilities(v, pdf) {
      const p1 = pdf.lam1 * gaussianPDF(v, pdf.mu1, pdf.s1);
      const p2 = pdf.lam2 * gaussianPDF(v, pdf.mu2, pdf.s2);
      const total = p1 + p2 + 1e-300;
      return { pCrash: p1 / total, pBull: p2 / total };
    }

    function estimateTauLDays(series) {
      const mean = series.reduce((a, b) => a + b, 0) / series.length;
      let num = 0;
      let den = 0;
      for (let index = 0; index < series.length - 1; index += 1) {
        num += (series[index] - mean) * (series[index + 1] - mean);
        den += (series[index] - mean) ** 2;
      }
      const acf1 = Math.min(0.999, Math.max(num / (den + 1e-10), 0.001));
      return { acf1, tauL_days: -1 / Math.log(acf1) };
    }

    function logPdfGradient(v, pdf) {
      const phi1 = gaussianPDF(v, pdf.mu1, pdf.s1);
      const phi2 = gaussianPDF(v, pdf.mu2, pdf.s2);
      const numerator =
        -pdf.lam1 * ((v - pdf.mu1) / (pdf.s1 ** 2)) * phi1 -
        pdf.lam2 * ((v - pdf.mu2) / (pdf.s2 ** 2)) * phi2;
      const denominator = pdf.lam1 * phi1 + pdf.lam2 * phi2 + 1e-300;
      return numerator / denominator;
    }

    function logPdfCurvature(v, pdf) {
      const phi1 = gaussianPDF(v, pdf.mu1, pdf.s1);
      const phi2 = gaussianPDF(v, pdf.mu2, pdf.s2);
      const p = pdf.lam1 * phi1 + pdf.lam2 * phi2 + 1e-300;
      const z1 = (v - pdf.mu1) / pdf.s1;
      const z2 = (v - pdf.mu2) / pdf.s2;
      const pPrime = -pdf.lam1 * (z1 / pdf.s1) * phi1 - pdf.lam2 * (z2 / pdf.s2) * phi2;
      const pDbl = phi1 * pdf.lam1 * ((z1 * z1 - 1) / (pdf.s1 * pdf.s1)) + phi2 * pdf.lam2 * ((z2 * z2 - 1) / (pdf.s2 * pdf.s2));
      return pDbl / p - (pPrime / p) * (pPrime / p);
    }

    function langevinDrift(v, pdf, tauL_days, processVar) {
      const tauSteps = Math.max(tauL_days, 1);
      return (-v / tauSteps) + 0.5 * processVar * logPdfGradient(v, pdf);
    }

    function refineMomentumEKF(returns, seedSeries, tauL_days, passes = 3) {
      const observationMean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const observationVar = returns.reduce((sum, value) => sum + (value - observationMean) ** 2, 0) / returns.length;
      const measureVar = Math.max(observationVar * 0.45, 1e-8);
      const processVar = Math.max(observationVar * 0.06, 1e-8);
      let vSeries = seedSeries.slice();
      let pdf = fitBimodalEM(vSeries);

      for (let pass = 0; pass < passes; pass += 1) {
        const refined = new Array(returns.length).fill(0);
        let P = observationVar;
        let vEst = pass === 0 ? seedSeries[0] : vSeries[0];

        for (let index = 0; index < returns.length; index += 1) {
          const drift = langevinDrift(vEst, pdf, tauL_days, processVar);
          const tauSteps = Math.max(tauL_days, 1);
          const jacobian = Math.max(0.2, Math.min(1.25, 1 - (1 / tauSteps) + 0.5 * processVar * logPdfCurvature(vEst, pdf)));
          const vPred = vEst + drift;
          const PPred = jacobian * P * jacobian + processVar;
          const K = PPred / (PPred + measureVar);
          vEst = vPred + K * (returns[index] - vPred);
          P = (1 - K) * PPred;
          refined[index] = vEst;
        }

        vSeries = refined.map((value, index) => 0.85 * value + 0.15 * vSeries[index]);
        pdf = fitBimodalEM(vSeries);
      }

      return { vSeries, pdf, processVar, measureVar, filterType: "EKF" };
    }

    function summarizeAnalysis(returns, vSeries, filterType, processVar, knownPdf = null) {
      const pdf = knownPdf || fitBimodalEM(vSeries);
      const vCurrent = vSeries[vSeries.length - 1];
      const probabilities = regimeProbabilities(vCurrent, pdf);
      const confidence = Math.abs(probabilities.pBull - probabilities.pCrash);
      const tau = estimateTauLDays(vSeries);
      const recent = returns.slice(-20);
      const volReal = Math.sqrt(recent.reduce((sum, value) => sum + value * value, 0) / recent.length) * Math.sqrt(252);
      const driftCurrent = (-vCurrent / Math.max(tau.tauL_days, 1)) + 0.5 * processVar * logPdfGradient(vCurrent, pdf);
      const regime = confidence < 0.20 ? "TRANSITION" : probabilities.pCrash > probabilities.pBull ? "BEAR" : "BULL";
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const std = Math.sqrt(returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / returns.length);
      const skewness = returns.reduce((sum, value) => sum + ((value - mean) / std) ** 3, 0) / returns.length;

      return {
        vSeries,
        vCurrent,
        pdf,
        pCrash: probabilities.pCrash,
        pBull: probabilities.pBull,
        confidence,
        regime,
        tauL_est: tau.tauL_days,
        volReal,
        acf1: tau.acf1,
        driftCurrent,
        processVol: Math.sqrt(processVar),
        filterType,
        skewness,
        returns,
        mean,
        std,
      };
    }

    function analyzeReturns(returns, tauL_days = 12) {
      if (!returns || returns.length < 20) {
        return null;
      }

      const seedParams = kalmanModelParams(returns, tauL_days);
      const seedSeries = kalmanFilter(returns, tauL_days);
      const seedTau = estimateTauLDays(seedSeries);
      const refined = refineMomentumEKF(returns, seedSeries, seedTau.tauL_days);
      const linearAnalysis = summarizeAnalysis(returns, seedSeries, "Linear KF", seedParams.Q);
      const ekfAnalysis = summarizeAnalysis(returns, refined.vSeries, refined.filterType, refined.processVar, refined.pdf);

      return {
        ...ekfAnalysis,
        comparison: {
          linear: linearAnalysis,
          ekf: ekfAnalysis,
          deltaMomentum: ekfAnalysis.vCurrent - linearAnalysis.vCurrent,
          deltaBull: ekfAnalysis.pBull - linearAnalysis.pBull,
          deltaTau: ekfAnalysis.tauL_est - linearAnalysis.tauL_est,
        },
      };
    }

    async function fetchMarketData(ticker) {
      const normalized = ticker.trim().toUpperCase();
      const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(normalized)}?range=6mo&interval=1d&includePrePost=false&events=div%2Csplits`;
      const PROXIES = [
        (u) => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
        (u) => `https://thingproxy.freeboard.io/fetch/${u}`,
        (u) => `https://corsproxy.io/?url=${encodeURIComponent(u)}`,
      ];

      async function tryFetch(url) {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response;
      }

      let response;
      let viaProxy = false;
      let lastError;

      try {
        response = await tryFetch(yahooUrl);
      } catch (error) {
        lastError = error;
      }

      if (!response) {
        for (const makeProxy of PROXIES) {
          try {
            response = await tryFetch(makeProxy(yahooUrl));
            viaProxy = true;
            break;
          } catch (error) {
            lastError = error;
          }
        }
      }

      if (!response) {
        throw new Error(`Unable to load market data – all sources failed. ${lastError?.message || ""}`);
      }

      const payload = await response.json();
      const result = payload?.chart?.result?.[0];
      const error = payload?.chart?.error;
      if (error) {
        throw new Error(error.description || `No market data found for ${normalized}`);
      }
      if (!result) {
        throw new Error(`No market data found for ${normalized}`);
      }

      const timestamps = result.timestamp || [];
      const closes = result.indicators?.quote?.[0]?.close || [];
      const rows = timestamps
        .map((timestamp, index) => ({
          date: new Date(timestamp * 1000).toISOString().slice(0, 10),
          close: Number(closes[index]),
        }))
        .filter((row) => Number.isFinite(row.close));

      if (rows.length < 30) {
        throw new Error(`Insufficient history returned for ${normalized}`);
      }

      const recent = rows.slice(-90);
      return {
        ticker: result.meta?.symbol || normalized,
        name: result.meta?.longName || result.meta?.shortName || normalized,
        prices: recent.map((row) => row.close),
        dates: recent.map((row) => row.date),
        currency: result.meta?.currency || "USD",
        lastDate: recent[recent.length - 1].date,
        viaProxy,
      };
    }

  const APP_VERSION = "2026-05-03";
  const QUICK_PICKS = ["SPY", "QQQ", "AAPL", "NVDA", "BTC-USD", "GLD"];
    const RATIONALE_SECTIONS = [
      {
        title: "Why use a Langevin model at all?",
        body: "The app treats market momentum as a latent stochastic state rather than reading price directly from moving averages. In the original physical analogy, a particle moves through a convective boundary layer where vertical velocity is shaped by both deterministic drift and turbulent noise. Here, log-price plays the role of position, while latent momentum v(t) plays the role of velocity. That gives the model memory, persistence, and a physically motivated way to separate fast shocks from slower regimes.",
      },
      {
        title: "Atmospheric analogy behind the interface",
        body: "In a convective boundary layer, the two dominant flow modes are narrow, intense bursts and broader, milder returns. That asymmetry creates a skewed velocity distribution that is well represented by a two-component Gaussian mixture. The financial translation flips the sign intuition: bear moves tend to be fast and concentrated, while bullish phases are often slower and more persistent. The app keeps that asymmetry by fitting a bimodal distribution to the estimated momentum series.",
      },
      {
        title: "What the math is trying to preserve",
        body: "The Thomson well-mixed idea matters because it says the drift term should be consistent with the observed stationary distribution. In plain language: if momentum is empirically distributed in an asymmetric, two-regime way, the dynamics should not assume a single Gaussian world. The mean-reverting part controls persistence through tau_L, while the distribution-sensitive correction keeps the process aligned with the fitted bimodal density. In this standalone version that correction enters an extended-Kalman-style update after the initial linear estimate.",
      },
      {
        title: "Why the dashboard looks the way it does",
        body: "Each panel maps directly to one piece of the model. A linear Kalman pass gives an initial hidden momentum estimate, then an EKF-style nonlinear Langevin update propagates that state using a drift consistent with the fitted bimodal density. The EM fit estimates the two momentum modes. Bayesian responsibilities convert the current v(t) into bear-versus-bull probabilities, while a transition label is derived from low posterior confidence rather than from an invented third Gaussian. The gauge, probability bars, PDF view, and suggested exposure are all different views of the same latent-state inference pipeline.",
      },
    ];
    const ANALOGY_ROWS = [
      ["Vertical position", "Log-price, the running market trajectory"],
      ["Vertical velocity", "Latent momentum v(t) inferred from returns"],
      ["Bear regime", "Fast adverse move or bear-like impulse"],
      ["Bull regime", "Broader, slower bull regime"],
      ["Velocity scale", "Volatility level"],
      ["Lagrangian time", "Memory or mean-reversion horizon tau_L"],
      ["Well-mixed condition", "A consistency constraint between dynamics and empirical distribution"],
    ];

    const state = {
      ticker: "",
      view: "tool",
      status: "idle",
      statusMsg: "Ready. Uses Yahoo Finance chart data directly from the browser.",
      rawData: null,
      analysis: null,
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function regimeColor(regime) {
      return regime === "BEAR" ? "#ff4757" : regime === "BULL" ? "#2ed573" : "#f5a623";
    }

    function adviceText(analysis) {
      if (!analysis) {
        return "";
      }
      if (analysis.regime === "BEAR") {
        return "Bearish pressure detected. Consider trimming long exposure or hedging risk.";
      }
      if (analysis.regime === "BULL") {
        return "Bullish momentum is stable. Long exposure remains favored with risk control.";
      }
      return "Neutral regime. Wait for confirmation before changing exposure.";
    }

    function renderSparkline(data, color, height = 140, showZero = true) {
      if (!data || data.length < 2) {
        return "";
      }
      const width = 100;
      const svgHeight = 100;
      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = max - min || 1;
      const points = data.map((value, index) => `${(index / (data.length - 1)) * width},${svgHeight - ((value - min) / range) * svgHeight}`).join(" ");
      const zeroY = svgHeight - ((0 - min) / range) * svgHeight;
      const gradientId = `grad-${color.replace(/[^a-z0-9]/gi, "")}`;
      const last = data[data.length - 1];
      const lastY = svgHeight - ((last - min) / range) * svgHeight;
      return `
        <svg viewBox="0 0 ${width} ${svgHeight}" width="100%" height="${height}" preserveAspectRatio="none" style="display:block">
          <defs>
            <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="${color}" stop-opacity="0.3"></stop>
              <stop offset="100%" stop-color="${color}" stop-opacity="0"></stop>
            </linearGradient>
          </defs>
          ${showZero ? `<line x1="0" y1="${zeroY}" x2="${width}" y2="${zeroY}" stroke="rgba(255,255,255,0.1)" stroke-width="0.5" stroke-dasharray="2,2"></line>` : ""}
          <polygon points="0,${svgHeight} ${points} ${width},${svgHeight}" fill="url(#${gradientId})"></polygon>
          <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.2"></polyline>
          <circle cx="${width}" cy="${lastY}" r="2.5" fill="${color}" stroke="var(--bg)" stroke-width="1"></circle>
        </svg>
      `;
    }

    function renderRegimeGauge(pCrash, pBull, regime) {
      const R = 90;
      const cx = 100;
      const cy = 100;
      const startAngle = -200;
      const endAngle = 20;
      const totalRange = endAngle - startAngle;
      const score = pBull - pCrash;
      const needleAngle = startAngle + ((score + 1) / 2) * totalRange;
      const currentColor = regimeColor(regime);
      const segments = [
        { start: startAngle, end: startAngle + totalRange * 0.33, color: "#ff4757" },
        { start: startAngle + totalRange * 0.33, end: startAngle + totalRange * 0.67, color: "#4a6280" },
        { start: startAngle + totalRange * 0.67, end: endAngle, color: "#2ed573" },
      ];

      function toRad(degrees) {
        return (degrees * Math.PI) / 180;
      }

      function arcPath(startDeg, endDeg, radius) {
        const start = toRad(startDeg);
        const end = toRad(endDeg);
        const x1 = cx + radius * Math.cos(start);
        const y1 = cy + radius * Math.sin(start);
        const x2 = cx + radius * Math.cos(end);
        const y2 = cy + radius * Math.sin(end);
        const large = endDeg - startDeg > 180 ? 1 : 0;
        return `M ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2}`;
      }

      const nx = cx + R * 0.72 * Math.cos(toRad(needleAngle));
      const ny = cy + R * 0.72 * Math.sin(toRad(needleAngle));
      return `
        <svg class="gauge-svg" viewBox="0 0 200 115" width="200" height="115">
          <path d="${arcPath(startAngle, endAngle, R)}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="12" stroke-linecap="round"></path>
          ${segments.map((segment) => `<path d="${arcPath(segment.start, segment.end, R)}" fill="none" stroke="${segment.color}" stroke-width="12" stroke-linecap="round" opacity="0.7"></path>`).join("")}
          <path d="${arcPath(startAngle, needleAngle, R)}" fill="none" stroke="${currentColor}" stroke-width="3" stroke-linecap="round" opacity="0.9"></path>
          <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#fff" stroke-width="2" stroke-linecap="round"></line>
          <circle cx="${cx}" cy="${cy}" r="5" fill="var(--panel)" stroke="#fff" stroke-width="1.5"></circle>
          <circle cx="${cx}" cy="${cy}" r="2" fill="${currentColor}"></circle>
          <text x="${cx - 82}" y="${cy + 16}" fill="#ff4757" font-size="8" font-family="Space Mono" text-anchor="middle">BEAR</text>
          <text x="${cx + 82}" y="${cy + 16}" fill="#2ed573" font-size="8" font-family="Space Mono" text-anchor="middle">BULL</text>
        </svg>
      `;
    }

    function renderPDFChart(vSeries, pdf, vCurrent) {
      if (!vSeries || !pdf) {
        return "";
      }
      const min = Math.min(...vSeries);
      const max = Math.max(...vSeries);
      const pointsCount = 120;
      const step = (max - min || 1) / pointsCount;
      const xs = Array.from({ length: pointsCount }, (_, index) => min + index * step);
      const totalYs = xs.map((x) => pdf.lam1 * gaussianPDF(x, pdf.mu1, pdf.s1) + pdf.lam2 * gaussianPDF(x, pdf.mu2, pdf.s2));
      const ys1 = xs.map((x) => pdf.lam1 * gaussianPDF(x, pdf.mu1, pdf.s1));
      const ys2 = xs.map((x) => pdf.lam2 * gaussianPDF(x, pdf.mu2, pdf.s2));
      const yMax = Math.max(...totalYs, 1e-9);
      const width = 300;
      const height = 90;
      const denom = max - min || 1;
      const toSvg = (x, y) => [((x - min) / denom) * width, height - (y / yMax) * height * 0.9];
      const totalPoints = xs.map((x, index) => toSvg(x, totalYs[index]).join(",")).join(" ");
      const crashPoints = xs.map((x, index) => toSvg(x, ys1[index]).join(",")).join(" ");
      const bullPoints = xs.map((x, index) => toSvg(x, ys2[index]).join(",")).join(" ");
      const currentX = ((vCurrent - min) / denom) * width;
      const startBase = toSvg(min, 0).join(",");
      const endBase = toSvg(max, 0).join(",");
      return `
        <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" style="display:block">
          <defs>
            <linearGradient id="pdf-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.2"></stop>
              <stop offset="100%" stop-color="#00d4ff" stop-opacity="0"></stop>
            </linearGradient>
          </defs>
          <polygon points="${startBase} ${totalPoints} ${endBase}" fill="url(#pdf-fill)"></polygon>
          <polyline points="${crashPoints}" fill="none" stroke="#ff4757" stroke-width="1" stroke-dasharray="3,2" opacity="0.6"></polyline>
          <polyline points="${bullPoints}" fill="none" stroke="#2ed573" stroke-width="1" stroke-dasharray="3,2" opacity="0.6"></polyline>
          <polyline points="${totalPoints}" fill="none" stroke="#00d4ff" stroke-width="1.5"></polyline>
          <line x1="${currentX}" y1="0" x2="${currentX}" y2="${height}" stroke="#f5a623" stroke-width="1.2" stroke-dasharray="3,2"></line>
          <circle cx="${currentX}" cy="${height / 2}" r="3" fill="#f5a623"></circle>
          <text x="${currentX + 4}" y="${height / 2 - 4}" fill="#f5a623" font-size="7" font-family="Space Mono">v(t)</text>
          <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"></line>
        </svg>
      `;
    }

    function renderExposureRing(pCrash, pBull) {
      const raw = 0.5 + 0.5 * pBull - 1.0 * pCrash;
      const exposure = Math.max(-0.3, Math.min(1.0, raw));
      const pct = (exposure + 0.3) / 1.3;
      const color = exposure > 0.6 ? "#2ed573" : exposure > 0.2 ? "#f5a623" : "#ff4757";
      const R = 46;
      const cx = 55;
      const cy = 55;
      const start = -225;
      const range = 270;
      const endAngle = start + pct * range;

      function toRad(degrees) {
        return (degrees * Math.PI) / 180;
      }

      function arc(s, e) {
        const sx = cx + R * Math.cos(toRad(s));
        const sy = cy + R * Math.sin(toRad(s));
        const ex = cx + R * Math.cos(toRad(e));
        const ey = cy + R * Math.sin(toRad(e));
        const large = e - s > 180 ? 1 : 0;
        return `M ${sx} ${sy} A ${R} ${R} 0 ${large} 1 ${ex} ${ey}`;
      }

      return `
        <svg viewBox="0 0 110 110" width="110" height="110">
          <path d="${arc(start, start + range)}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="8" stroke-linecap="round"></path>
          <path d="${arc(start, endAngle)}" fill="none" stroke="${color}" stroke-width="8" stroke-linecap="round"></path>
          <text x="${cx}" y="${cy - 4}" text-anchor="middle" fill="#fff" font-size="17" font-family="Space Mono" font-weight="700">${exposure > 0 ? "+" : ""}${(exposure * 100).toFixed(0)}%</text>
          <text x="${cx}" y="${cy + 12}" text-anchor="middle" fill="var(--muted)" font-size="8" font-family="Space Mono">EXPO</text>
        </svg>
      `;
    }

    function wilsonCI(hits, n, z = 1.96) {
      if (n === 0) {
        return { lo: 0, hi: 1 };
      }
      const p = hits / n;
      const denom = 1 + z * z / n;
      const center = (p + z * z / (2 * n)) / denom;
      const margin = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom;
      return { lo: Math.max(0, center - margin), hi: Math.min(1, center + margin) };
    }

    function buildHistoricalReplay(prices, dates, tauL_days = 12, warmup = 45, maxRows = 10) {
      if (!prices || prices.length < warmup + 2) {
        return null;
      }
      const rows = [];
      for (let endPriceIndex = warmup; endPriceIndex < prices.length - 1; endPriceIndex += 1) {
        const windowPrices = prices.slice(0, endPriceIndex + 1);
        const windowReturns = [];
        for (let index = 1; index < windowPrices.length; index += 1) {
          windowReturns.push(Math.log(windowPrices[index] / windowPrices[index - 1]));
        }
        const snapshot = analyzeReturns(windowReturns, tauL_days);
        if (!snapshot) {
          continue;
        }
        const nextReturn = Math.log(prices[endPriceIndex + 1] / prices[endPriceIndex]);
        const predictedDirection = snapshot.pBull >= snapshot.pCrash ? 1 : -1;
        const realizedDirection = nextReturn >= 0 ? 1 : -1;
        rows.push({
          asOfDate: dates?.[endPriceIndex] || `t${endPriceIndex}`,
          targetDate: dates?.[endPriceIndex + 1] || `t${endPriceIndex + 1}`,
          regime: snapshot.regime,
          bullWeight: snapshot.pBull,
          crashWeight: snapshot.pCrash,
          confidence: snapshot.confidence,
          predictedDirection,
          nextReturn,
          hit: predictedDirection === realizedDirection,
        });
      }

      const recentRows = rows.slice(-maxRows).reverse();
      const hitCount = recentRows.filter((row) => row.hit).length;
      const hitRate = recentRows.length ? hitCount / recentRows.length : 0;
      const ci = wilsonCI(hitCount, recentRows.length);
      const avgAbsNextMove = recentRows.length ? recentRows.reduce((sum, row) => sum + Math.abs(row.nextReturn), 0) / recentRows.length : 0;
      return {
        rows: recentRows,
        hitRate,
        ci,
        avgAbsNextMove,
        sampleSize: recentRows.length,
      };
    }

    function renderRationalePage() {
      return `
        <div class="rationale-page animate-in">
          <div class="rationale-hero">
            <div class="panel-label">Rationale</div>
            <div class="rationale-title">A market regime tool built from a turbulence analogy, not from a generic trend indicator.</div>
            <div class="rationale-lead">
              This app comes from the idea that market returns can be modeled like the motion of a particle in a turbulent convective boundary layer. Instead of assuming one symmetric market state, it assumes that the hidden momentum process can occupy different modes with different intensity, persistence, and tail behavior. That is the reason the interface estimates a latent state first and classifies regimes second.
            </div>
            <div class="equation-box">
              d log(S_t) = v_t dt<br>
              dv_t = a(v_t) dt + sigma dW_t<br>
              a(v) = -v / tau_L + distribution correction from the fitted bimodal PDF
            </div>
          </div>
          <div class="rationale-grid">
            ${RATIONALE_SECTIONS.map((section) => `
              <div class="rationale-card">
                <h2>${escapeHtml(section.title)}</h2>
                <p>${escapeHtml(section.body)}</p>
              </div>
            `).join("")}
          </div>
          <div class="rationale-card">
            <h2>Atmospheric-to-financial mapping</h2>
            <div class="rationale-list">
              ${ANALOGY_ROWS.map(([key, value]) => `
                <div class="rationale-list-item">
                  <div class="rationale-key">${escapeHtml(key)}</div>
                  <div class="rationale-value">${escapeHtml(value)}</div>
                </div>
              `).join("")}
            </div>
          </div>
          <div class="rationale-grid">
            <div class="rationale-card">
              <h2>Practical use</h2>
              <p>The practical goal is not to forecast exact prices. It is to estimate the current regime more honestly than a Gaussian or single-threshold tool. In that role, the model can be used as a regime filter for discretionary trading, exposure control, stress-aware risk management, and comparison across asset classes such as equities, ETFs, gold, and crypto.</p>
            </div>
            <div class="rationale-card">
              <h2>Limits to keep in mind</h2>
              <p>This remains a compact browser implementation. It uses a short recent window, a simple Kalman filter, and a two-Gaussian EM fit. It does not estimate a full particle filter, a time-varying tau_L, or a risk-neutral pricing model. The rationale page explains the research direction; the dashboard is a lightweight estimator, not a complete institutional calibration stack.</p>
            </div>
          </div>
          <div class="rationale-note">
            The interface is therefore best read as a regime estimation instrument. It translates the original idea from the shared discussion into a concrete workflow: retrieve recent prices, infer latent momentum, fit an asymmetric bimodal distribution, and report which side of that distribution the market currently resembles.
          </div>
        </div>
      `;
    }

    function renderTheoryPage() {
      return `
        <div class="rationale-page animate-in">
          <div class="rationale-hero">
            <div class="panel-label">Theory</div>
            <div class="rationale-title">Theoretical Foundations of the Langevin Regime Model</div>
            <div class="rationale-lead">
              The core theory maps the physical well-mixed condition of turbulent convective boundary layers to financial momentum stochastic equations. This model departs from Gaussian asset pricing by treating the market as a non-equilibrium system where asymmetric structural drift and local noise generate heavy-tailed bimodal distributions.
            </div>
            <div class="equation-box">
              1. Extended Kalman Filter (EKF) State Update<br>
              v_t = a(v_t) dt + EKF_correction(v_t, obs)<br><br>
              2. Analytical Jacobian<br>
              H = d/dv a(v) = -1/tau_L + 0.5 * sigma^2 * d²/dv² ln p(v)
            </div>
          </div>
          <div class="rationale-grid">
            <div class="rationale-card">
              <h2>Stochastic Differential Core</h2>
              <p>The application relies on Thomson's (1987) derivation where the Fokker-Planck equation corresponds to a Langevin process (<i>Thomson D.J, 1987, Criteria for the selection of stochastic models of particle trajectories in turbulent flows, Journal of Fluid Mechanics, 180, 529-556</i>). By imposing the well-mixed condition, a particle starting in a region with specific distribution properties will dynamically remain consistent with that aggregate PDF. In finance, this translates to trend estimators preserving realistic skewness observed in bear/bull asymmetries rather than regressing linearly to an incorrect single mean.</p>
            </div>
            <div class="rationale-card">
              <h2>Limitations &amp; Mathematical Assumptions</h2>
              <p>While the Jacobian derivation and probability updates are exact closed-form expressions, the model operates heuristically in parameter calibration. EKF covariance matrices (Q and R) and the bimodal EM iterations lack longitudinal risk-neutral bounds, acting strictly as dynamic filters. Future theory could couple this with Maximum Likelihood Estimation (MLE) or full Particle Filtering spanning multiple tau_L relaxation bounds.</p>
            </div>
          </div>
        </div>
      `;
    }

    function renderToolView() {
      const analysis = state.analysis;
      const rawData = state.rawData;
      const color = analysis ? regimeColor(analysis.regime) : "#4a6280";
      const advice = adviceText(analysis);
      const dotClass = state.status === "idle" ? "done" : state.status;

      let gridContent = "";
      if (rawData && analysis) {
        gridContent += `
          <div class="ticker-badge animate-in">
            <div class="ticker-sym">${escapeHtml(rawData.ticker)}</div>
            <div class="ticker-name">${escapeHtml(rawData.name || "")}</div>
            <div class="ticker-date">${escapeHtml(rawData.lastDate || "")} · ${rawData.prices.length} days</div>
          </div>
        `;
      }

      if (state.status === "idle") {
        gridContent += `
          <div class="placeholder">
            <div class="placeholder-icon">⚡</div>
            <div class="placeholder-title">Enter a ticker to start</div>
            <div class="placeholder-sub">Direct browser mode · public sources · model runs locally</div>
          </div>
        `;
      }

      if (state.status === "loading") {
        gridContent += `
          <div class="placeholder">
            <div class="placeholder-icon" style="animation:pulse 1s infinite">🌀</div>
            <div class="placeholder-title">${escapeHtml(state.statusMsg)}</div>
            <div class="placeholder-sub">Fetching prices · Kalman filter · Bimodal EM · Bayesian regime</div>
          </div>
        `;
      }

      if (state.status === "error") {
        gridContent += `
          <div class="error-panel animate-in">
            <div style="margin-bottom:8px;font-weight:700">Data loading error</div>
            <div style="color:var(--muted)">${escapeHtml(state.statusMsg)}</div>
          </div>
        `;
      }

      if (state.status === "done" && analysis) {
        const regimeTitle = analysis.regime === "BEAR" ? "↓ BEAR" : analysis.regime === "BULL" ? "↑ BULL" : "◆ TRANSITION";
        const regimeSubtitle = analysis.regime === "BEAR" ? "Bear state" : analysis.regime === "BULL" ? "Bull state" : "Low-confidence transition zone";
        const exposureAction = analysis.regime === "BEAR" ? "REDUCE EXPOSURE" : analysis.regime === "BULL" ? "MAINTAIN / ADD" : "NEUTRAL STANCE";

        gridContent += `
          <div class="panel gauge-panel animate-in">
            <div class="panel-label">Current regime</div>
            ${renderRegimeGauge(analysis.pCrash, analysis.pBull, analysis.regime)}
            <div class="regime-label" style="color:${color}">${regimeTitle}</div>
            <div class="regime-sub">${regimeSubtitle}</div>
            ${rawData ? `<div style="font-family:var(--mono);font-size:13px;font-weight:700;color:var(--accent);margin-top:8px;letter-spacing:0.12em">${escapeHtml(rawData.ticker)}</div>` : ""}
          </div>
          <div class="panel prob-panel animate-in">
            <div class="panel-label">Posterior weights P(regime | v_t)</div>
            ${[
              { label: "Bear", val: analysis.pCrash, color: "#ff4757" },
              { label: "Bull", val: analysis.pBull, color: "#2ed573" },
            ].map((row) => `
              <div class="prob-row">
                <div class="prob-header">
                  <span class="prob-name">${row.label}</span>
                  <span class="prob-val" style="color:${row.color}">${(row.val * 100).toFixed(1)}%</span>
                </div>
                <div class="prob-track"><div class="prob-fill" style="width:${row.val * 100}%;background:${row.color}"></div></div>
              </div>
            `).join("")}
            <div style="margin-top:16px;padding:10px;background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.12);border-radius:6px">
              <div style="display:flex;justify-content:space-between;gap:12px;font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:4px">
                <span>CURRENT MOMENTUM v(t)</span>
                <span>CONFIDENCE ${(analysis.confidence * 100).toFixed(1)}%</span>
              </div>
              <div style="font-family:var(--mono);font-size:16px;font-weight:700;color:${analysis.vCurrent > 0 ? "#2ed573" : "#ff4757"}">${analysis.vCurrent > 0 ? "+" : ""}${(analysis.vCurrent * 100).toFixed(4)}%</div>
            </div>
          </div>
          <div class="panel params-panel animate-in">
            <div class="panel-label">Calibrated CBL parameters</div>
            ${[
              ["tau_L (days)", `${analysis.tauL_est.toFixed(1)} d`],
              ["ACF(lag=1)", analysis.acf1.toFixed(4)],
              ["Realized vol", `${(analysis.volReal * 100).toFixed(2)}%`],
              ["drift(v_t)", `${(analysis.driftCurrent * 100).toFixed(4)}%/day`],
              ["process sigma", `${(analysis.processVol * 100).toFixed(4)}%`],
              ["Skewness", analysis.skewness.toFixed(3)],
              ["lambda_1", analysis.pdf.lam1.toFixed(3)],
              ["mu_1", `${(analysis.pdf.mu1 * 100).toFixed(4)}%`],
              ["sigma_1", `${(analysis.pdf.s1 * 100).toFixed(4)}%`],
              ["lambda_2", analysis.pdf.lam2.toFixed(3)],
              ["mu_2", `${(analysis.pdf.mu2 * 100).toFixed(4)}%`],
              ["sigma_2", `${(analysis.pdf.s2 * 100).toFixed(4)}%`],
            ].map(([key, value]) => `<div class="param-row"><span class="param-key">${key}</span><span class="param-val">${value}</span></div>`).join("")}
          </div>
          <div class="panel compare-panel animate-in">
            <div class="panel-label">Linear KF vs EKF comparison</div>
            <div class="compare-grid">
              ${[analysis.comparison.linear, analysis.comparison.ekf].map((item) => {
                const localColor = regimeColor(item.regime);
                return `
                  <div class="compare-card">
                    <div class="compare-title"><span>${escapeHtml(item.filterType)}</span><span>confidence ${(item.confidence * 100).toFixed(1)}%</span></div>
                    <div class="compare-main">
                      <div class="compare-regime" style="color:${localColor}">${escapeHtml(item.regime)}</div>
                      <div class="compare-momentum">${item.vCurrent > 0 ? "+" : ""}${(item.vCurrent * 100).toFixed(4)}%</div>
                    </div>
                    <div class="compare-stats">
                      <div><div class="compare-stat-label">Bull weight</div><div class="compare-stat-value">${(item.pBull * 100).toFixed(1)}%</div></div>
                      <div><div class="compare-stat-label">Bear weight</div><div class="compare-stat-value">${(item.pCrash * 100).toFixed(1)}%</div></div>
                      <div><div class="compare-stat-label">tau_L</div><div class="compare-stat-value">${item.tauL_est.toFixed(1)} d</div></div>
                      <div><div class="compare-stat-label">drift(v_t)</div><div class="compare-stat-value">${(item.driftCurrent * 100).toFixed(4)}%/day</div></div>
                    </div>
                  </div>
                `;
              }).join("")}
            </div>
            <div class="compare-delta">
              <span>delta v(t): ${analysis.comparison.deltaMomentum > 0 ? "+" : ""}${(analysis.comparison.deltaMomentum * 100).toFixed(4)}%</span>
              <span>delta bull weight: ${analysis.comparison.deltaBull > 0 ? "+" : ""}${(analysis.comparison.deltaBull * 100).toFixed(1)} pts</span>
              <span>delta tau_L: ${analysis.comparison.deltaTau > 0 ? "+" : ""}${analysis.comparison.deltaTau.toFixed(1)} d</span>
            </div>
          </div>
          ${analysis.replay ? `
            <div class="panel replay-panel animate-in">
              <div class="panel-label">Historical one-day-ahead replay</div>
              <div class="replay-summary">
                <span>recent forecasts: ${analysis.replay.sampleSize}</span>
                <span>hit rate: ${(analysis.replay.hitRate * 100).toFixed(1)}% [95% CI ${(analysis.replay.ci.lo * 100).toFixed(0)}–${(analysis.replay.ci.hi * 100).toFixed(0)}%]</span>
                <span>avg next-day move: ${(analysis.replay.avgAbsNextMove * 100).toFixed(2)}%</span>
                <span>rule: bull if P(bull) ≥ P(crash), crash otherwise</span>
              </div>
              <div class="replay-table">
                <div class="replay-row header"><span>As of</span><span>For date</span><span>Predicted</span><span>Confidence</span><span>Realized next return</span></div>
                ${analysis.replay.rows.map((row) => `
                  <div class="replay-row">
                    <span>${escapeHtml(row.asOfDate)}</span>
                    <span>${escapeHtml(row.targetDate)}</span>
                    <span class="replay-pred" style="color:${row.predictedDirection > 0 ? "#2ed573" : "#ff4757"}">${row.predictedDirection > 0 ? "Bull" : "Bear"} · ${escapeHtml(row.regime)}</span>
                    <span>${(row.confidence * 100).toFixed(1)}%</span>
                    <span><span class="replay-hit ${row.hit ? "good" : "bad"}">${row.hit ? "hit" : "miss"}</span> · ${row.nextReturn > 0 ? "+" : ""}${(row.nextReturn * 100).toFixed(2)}%</span>
                  </div>
                `).join("")}
              </div>
            </div>
          ` : ""}
          <div class="panel chart-panel animate-in">
            <div class="panel-label">Lagrangian momentum v(t) - Kalman plus ${escapeHtml(analysis.filterType)} Langevin update (${analysis.vSeries.length} observations)</div>
            <div class="chart-canvas-wrap">${renderSparkline(analysis.vSeries, color, 130, true)}</div>
            <div style="display:flex;gap:16px;margin-top:8px;font-family:var(--mono);font-size:9px;color:var(--muted);flex-wrap:wrap">
              <span>MIN: ${(Math.min(...analysis.vSeries) * 100).toFixed(3)}%</span>
              <span>MAX: ${(Math.max(...analysis.vSeries) * 100).toFixed(3)}%</span>
              <span>sigma: ${(analysis.std * 100).toFixed(3)}%</span>
              <span style="margin-left:auto;color:var(--accent)">v(t): ${analysis.vCurrent > 0 ? "+" : ""}${(analysis.vCurrent * 100).toFixed(4)}%</span>
            </div>
          </div>
          <div class="panel pdf-panel animate-in">
            <div class="panel-label">Bimodal PDF p(v) - EM calibrated momentum distribution</div>
            ${renderPDFChart(analysis.vSeries, analysis.pdf, analysis.vCurrent)}
            <div style="display:flex;gap:20px;margin-top:8px;font-family:var(--mono);font-size:9px;flex-wrap:wrap">
              <span style="color:#ff4757">Bear component λ=${analysis.pdf.lam1.toFixed(2)}</span>
              <span style="color:#2ed573">Bull component λ=${analysis.pdf.lam2.toFixed(2)}</span>
              <span style="color:#00d4ff">Total PDF</span>
              <span style="color:#f5a623">Current v(t)</span>
            </div>
          </div>
          <div class="panel signal-panel animate-in">
            <div class="panel-label">Suggested exposure</div>
            <div class="signal-box">
              ${renderExposureRing(analysis.pCrash, analysis.pBull)}
              <div style="font-size:12px;color:var(--muted);text-align:center;line-height:1.5;max-width:180px">${escapeHtml(advice)}</div>
              <div style="margin-top:8px;padding:8px 14px;background:${color}18;border:1px solid ${color}44;border-radius:6px;font-family:var(--mono);font-size:10px;color:${color};text-align:center">${exposureAction}</div>
            </div>
          </div>
        `;
      }

      return `
        <div class="search-row">
          <input id="ticker-input" class="search-input" placeholder="Enter ticker (AAPL, SPY, BTC-USD)..." value="${escapeHtml(state.ticker)}">
          <button class="search-btn" data-action="analyze" ${state.status === "loading" ? "disabled" : ""}>${state.status === "loading" ? "..." : "Analyze"}</button>
        </div>
        <div class="quick-picks">
          ${QUICK_PICKS.map((symbol) => `<button class="quick-chip" data-pick="${symbol}">${symbol}</button>`).join("")}
        </div>
        <div class="status-bar">
          <div class="status-dot ${dotClass}"></div>
          <span style="font-family:var(--mono);font-size:11px;color:var(--muted)">${escapeHtml(state.statusMsg)}</span>
        </div>
        <div class="grid">${gridContent}</div>
        <div style="text-align:center;margin-top:24px;font-family:var(--mono);font-size:9px;color:var(--muted);opacity:0.5">Standalone HTML demo · Yahoo Finance chart API · Not investment advice</div>
      `;
    }

    function renderDisclaimer() {
      return `
        <div style="max-width:900px;margin:32px auto 0;padding:18px 24px;background:rgba(245,166,35,0.05);border:1px solid rgba(245,166,35,0.18);border-radius:10px;font-family:var(--mono);font-size:10px;color:var(--muted);line-height:1.8">
          <div style="color:#f5a623;font-weight:700;margin-bottom:6px;letter-spacing:0.15em;text-transform:uppercase">Disclaimer</div>
          This tool is provided for <strong style="color:var(--text)">educational and research purposes only</strong>. It does not constitute financial advice, investment recommendations, or an offer or solicitation to buy or sell any financial instrument. All outputs — including regime labels, probability estimates, momentum values, and suggested exposure — are the result of an experimental statistical model and carry no guarantee of accuracy or fitness for any purpose.
          Past model performance, including any hit-rate figures shown in the historical replay panel, is <strong style="color:var(--text)">not indicative of future results</strong>. Markets are inherently unpredictable; no quantitative model eliminates risk.
          Price data is sourced from third-party public APIs and may be delayed, incomplete, or erroneous. The authors accept no liability for decisions made on the basis of this tool. Always consult a qualified financial professional before making investment decisions.
          <div style="margin-top:14px;padding-top:10px;border-top:1px solid rgba(245,166,35,0.12);text-align:center;letter-spacing:0.08em">
            Copyright 2026 Roberto Bianconi
          </div>
        </div>
      `;
    }

    function renderApp() {
      return `
        <div class="app">
          <div class="header">
            <div class="header-eyebrow">CBL · Langevin · Stochastic</div>
            <h1 class="header-title">Regime <span>Estimator</span></h1>
            <div class="header-sub">Standalone HTML · Public market data sources · No build step · Version ${APP_VERSION}</div>
            <div class="view-switch">
              <button class="view-chip ${state.view === "tool" ? "active" : ""}" data-view="tool">Estimator</button>
              <button class="view-chip ${state.view === "rationale" ? "active" : ""}" data-view="rationale">Rationale</button>
              <button class="view-chip ${state.view === "theory" ? "active" : ""}" data-view="theory">Theory</button>
            </div>
          </div>
          ${state.view === "tool" ? renderToolView() : state.view === "rationale" ? renderRationalePage() : renderTheoryPage()}
          ${renderDisclaimer()}
        </div>
      `;
    }

    const root = document.getElementById("root");

    function render() {
      root.innerHTML = renderApp();
      const input = document.getElementById("ticker-input");
      if (input && document.activeElement !== input) {
        input.value = state.ticker;
      }
    }

    async function run(symbol) {
      const currentTicker = (symbol || state.ticker).trim().toUpperCase();
      if (!currentTicker) {
        return;
      }

      state.ticker = currentTicker;
      state.view = "tool";
      state.status = "loading";
      state.statusMsg = `Loading market data for ${currentTicker}...`;
      state.analysis = null;
      state.rawData = null;
      render();

      try {
        const data = await fetchMarketData(currentTicker);
        if (!data.prices || data.prices.length < 20) {
          throw new Error("Insufficient data returned by the public source");
        }

        state.rawData = data;
        state.statusMsg = `Calibrating CBL model for ${currentTicker}...`;
        render();

        const prices = data.prices.map(Number).filter(Number.isFinite);
        const returns = [];
        for (let index = 1; index < prices.length; index += 1) {
          returns.push(Math.log(prices[index] / prices[index - 1]));
        }

        const result = analyzeReturns(returns, 12);
        const replay = buildHistoricalReplay(prices, data.dates, 12);
        state.analysis = { ...result, replay };
        state.status = "done";
        state.statusMsg = `${prices.length} prices loaded · ${returns.length} returns · updated ${data.lastDate || "today"}${data.viaProxy ? " · via proxy" : ""}`;
      } catch (error) {
        state.status = "error";
        state.statusMsg = error.message || "Unknown error while loading data";
      }

      render();
    }

    root.addEventListener("click", (event) => {
      const viewButton = event.target.closest("[data-view]");
      if (viewButton) {
        state.view = viewButton.getAttribute("data-view");
        render();
        return;
      }

      const pickButton = event.target.closest("[data-pick]");
      if (pickButton) {
        run(pickButton.getAttribute("data-pick"));
        return;
      }

      const analyzeButton = event.target.closest("[data-action='analyze']");
      if (analyzeButton && state.status !== "loading") {
        run();
      }
    });

    root.addEventListener("input", (event) => {
      if (event.target.id === "ticker-input") {
        state.ticker = event.target.value.toUpperCase();
      }
    });

    root.addEventListener("keydown", (event) => {
      if (event.target.id === "ticker-input" && event.key === "Enter") {
        run();
      }
    });

    render();

