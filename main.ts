// polymarket-faa-attribution-02 entry (Node 22, standard library only).
//
// Forecast: for each market of the FAA outage attribution family, the five
// quantiles (5/25/50/75/95%) of its Yes price at every minute of the next
// 4 hours (240 minutes), from the prices up to the checkpoint.
//
// Model — recent-window empirical change quantiles, recentered on the last
// price, blended with a fitted Gaussian random walk, and rescaled by the
// most recent hour's realised volatility:
//
//   1. Take each market's last 6 hours (360 minutes) of prices. At anchor
//      horizons h in {1,2,3,5,8,12,20,30,45,60,90,120,180,240} minutes,
//      collect every overlapping h-step change p[i+h] - p[i] and take its
//      empirical quantiles at the five scored levels. Prediction-market
//      prices are sticky (most minutes do not move at all), so the
//      empirical quantiles capture what a Gaussian cannot: inner quantiles
//      that sit exactly on the last price at short horizons, and fat-tailed
//      jump risk at long horizons.
//   2. Recentre every row so the median change is 0. The last price is
//      the best median forecast: market prices are close to martingales,
//      and any drift estimated from the window (momentum, mean reversion
//      to the 6h/24h mean) made both the practice and the live holdout
//      worse in testing, so no drift is used.
//   3. Rescale the empirical offsets by the ratio of the last 60 minutes'
//      realised 1-step volatility to the whole window's, clamped to
//      [0.7, 1.4]. A market that just woke up gets wider quantiles; one
//      that just went quiet gets narrower ones.
//   4. Blend 50/50 with a Gaussian random walk whose sigma is fitted to
//      the empirical interquartile range at h=240. This regularises the
//      noisy empirical tails (only ~120 overlapping windows inform the
//      h=240 quantiles) and keeps the forecast honest when the window
//      was quiet.
//   5. Interpolate the quantile offsets across horizons linearly in
//      sqrt(h), clamp to [0,1], enforce non-decreasing quantiles, round
//      to 6 decimals. Markets with too little history for an anchor fall
//      back to pooled quantiles estimated over every market of the case.
//
// Local validation (own scorer, pinball loss, practice cases p00..p05 plus
// a live holdout case v00 cut 2026-11-13T06:00Z, same practice instance):
//   house persistence baseline: practice 0.363589, v00 0.220800
//   this entry (TypeScript, as built): practice 0.283023, v00 0.076247
// i.e. about 22% below the baseline on the practice and 65% below on the
// live holdout. Longer windows (12h/24h), logit-space changes, tail
// widening and mean reversion were all tested and rejected: each was worse
// on the practice, the holdout, or both.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Anchor horizons (minutes) where empirical change quantiles are estimated. */
const ANCHORS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240];
/** How much recent history informs the empirical quantiles (6 hours). */
const WINDOW_MINUTES = 360;
/** Minimum overlapping windows needed to trust a market's own anchor. */
const MIN_WINDOWS = 50;
/** Stride when sampling changes for the pooled fallback quantiles. */
const POOLED_STRIDE = 5;
/** Blend weight of the Gaussian random walk (0 = pure empirical). */
const GAUSSIAN_BLEND = 0.5;
/** Clamp on the recent-volatility rescaling factor. */
const VOL_FLOOR = 0.7;
const VOL_CEIL = 1.4;
/** Minutes of recent history whose realised volatility rescales the forecast. */
const RECENT_MINUTES = 60;
/** Fallback spread (price units per square-root minute) when nothing else applies. */
const FALLBACK_SIGMA = 0.0025;

interface CaseMarket {
  id: string;
  name: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Linear-interpolation quantile of a sorted array, matching numpy's default. */
function quantileOf(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0] as number;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sorted[lo] as number;
  const b = sorted[hi] as number;
  return a + (b - a) * (pos - lo);
}

/** Root-mean-square of an array of changes (0 when empty). */
function rms(values: number[]): number {
  if (values.length === 0) return 0;
  let s = 0;
  for (const v of values) s += v * v;
  return Math.sqrt(s / values.length);
}

/**
 * Empirical quantiles of h-step changes at the anchor horizons, from one
 * market's recent window. Anchors with too few overlapping windows take
 * the pooled value for that horizon when one exists.
 */
function marketQuantiles(
  prices: number[],
  pooled: Map<number, number[]>,
): Map<number, number[]> {
  const win = prices.slice(-WINDOW_MINUTES);
  const n = win.length;
  const out = new Map<number, number[]>();
  for (const h of ANCHORS) {
    if (n - h < MIN_WINDOWS) {
      const fb = pooled.get(h);
      if (fb) out.set(h, fb.slice());
      continue;
    }
    const diffs: number[] = [];
    for (let i = 0; i <= n - h - 1; i++) diffs.push((win[i + h] as number) - (win[i] as number));
    diffs.sort((a, b) => a - b);
    out.set(h, QUANTILES.map((q) => quantileOf(diffs, q)));
  }
  return out;
}

/**
 * Pooled empirical quantiles over every market of the case, for markets
 * whose own history is too short for an anchor. Changes are sampled with
 * a stride to keep the sort cheap; every market contributes equally.
 */
function pooledQuantiles(markets: CaseMarket[]): Map<number, number[]> {
  const perAnchor = new Map<number, number[]>();
  for (const h of ANCHORS) perAnchor.set(h, []);
  for (const m of markets) {
    const p = m.prices;
    const n = p.length;
    for (const h of ANCHORS) {
      if (n - h < MIN_WINDOWS) continue;
      const arr = perAnchor.get(h) as number[];
      for (let i = 0; i <= n - h - 1; i += POOLED_STRIDE) {
        arr.push((p[i + h] as number) - (p[i] as number));
      }
    }
  }
  const out = new Map<number, number[]>();
  for (const h of ANCHORS) {
    const arr = perAnchor.get(h) as number[];
    if (arr.length === 0) continue;
    arr.sort((a, b) => a - b);
    out.set(h, QUANTILES.map((q) => quantileOf(arr, q)));
  }
  return out;
}

/**
 * Interpolate anchor quantile offsets across horizons, linearly in
 * sqrt(h). Outside the anchor range the nearest anchor is held flat.
 */
function interpolateOffsets(qmap: Map<number, number[]>, horizon: number): number[][] {
  const hs = [...qmap.keys()].sort((a, b) => a - b);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    if (h <= (hs[0] as number)) {
      rows.push((qmap.get(hs[0] as number) as number[]).slice());
    } else if (h >= (hs[hs.length - 1] as number)) {
      rows.push((qmap.get(hs[hs.length - 1] as number) as number[]).slice());
    } else {
      let i = 1;
      while ((hs[i] as number) < h) i++;
      const h0 = hs[i - 1] as number;
      const h1 = hs[i] as number;
      const w = (Math.sqrt(h) - Math.sqrt(h0)) / (Math.sqrt(h1) - Math.sqrt(h0));
      const a = qmap.get(h0) as number[];
      const b = qmap.get(h1) as number[];
      rows.push(a.map((v, j) => (1 - w) * v + w * (b[j] as number)));
    }
  }
  return rows;
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecastMarket(m: CaseMarket, horizon: number, pooled: Map<number, number[]>): number[][] {
  if (m.prices.length === 0) {
    // No history at all: emit a flat 0.5 forecast so the case stays valid.
    return Array.from({ length: horizon }, () => [0.5, 0.5, 0.5, 0.5, 0.5]);
  }
  const x = m.prices[m.prices.length - 1] as number;
  let qmap = marketQuantiles(m.prices, pooled);
  if (qmap.size === 0) {
    // No anchor at all (no pooled values either): plain Gaussian walk.
    qmap = new Map(ANCHORS.map((h) => [h, Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(h))]));
  }

  // Quantile offsets per horizon, recentred so the median change is 0.
  const offsets = interpolateOffsets(qmap, horizon);
  for (let h = 0; h < horizon; h++) {
    const med = (offsets[h] as number[])[2] as number;
    for (let j = 0; j < 5; j++) (offsets[h] as number[])[j] = ((offsets[h] as number[])[j] as number) - med;
  }

  // Rescale by recent realised volatility: last hour vs the whole window.
  const win = m.prices.slice(-WINDOW_MINUTES);
  if (win.length >= 120) {
    const all: number[] = [];
    for (let i = 0; i < win.length - 1; i++) all.push((win[i + 1] as number) - (win[i] as number));
    const recent = all.slice(-RECENT_MINUTES);
    const ratio = rms(recent) / (rms(all) || 1e-9);
    const factor = Math.min(VOL_CEIL, Math.max(VOL_FLOOR, ratio));
    for (let h = 0; h < horizon; h++) {
      for (let j = 0; j < 5; j++) (offsets[h] as number[])[j] = ((offsets[h] as number[])[j] as number) * factor;
    }
  }

  // Gaussian random walk fitted to the empirical IQR at the longest horizon.
  const q240 = offsets[horizon - 1] as number[];
  const iqr = Math.max(((q240[3] as number) - (q240[1] as number)), 1e-6);
  const sigma = iqr / (2 * 0.67449) / Math.sqrt(horizon);

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const off = offsets[h - 1] as number[];
    const raw = off.map((v, j) => {
      const g = (Z[j] as number) * sigma * Math.sqrt(h);
      return x + (1 - GAUSSIAN_BLEND) * v + GAUSSIAN_BLEND * g;
    });
    // Clamp to [0,1], enforce non-decreasing quantiles, round to 6 decimals.
    const q = raw.map((v) => Math.min(1, Math.max(0, v)));
    for (let j = 1; j < 5; j++) q[j] = Math.max(q[j] as number, q[j - 1] as number);
    rows.push(q.map((v) => price6(v)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const pooled = pooledQuantiles(c.markets);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecastMarket(m, c.horizon_minutes, pooled) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
