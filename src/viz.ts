/**
 * Full-screen UMAP exploration of the IAB taxonomy embedding space.
 * Loads precomputed UMAP coordinates from iab-embeddings.json.
 * Pan: click-drag.  Zoom: scroll wheel.
 */

import type { EmbeddedEntry } from './types.ts';

// ── Types ─────────────────────────────────────────────────────────────────────

type PlottedPoint = EmbeddedEntry & { umap: [number, number] };

// ── DOM ───────────────────────────────────────────────────────────────────────

const canvas   = document.getElementById('canvas')   as HTMLCanvasElement;
const ctx      = canvas.getContext('2d')!;
const tooltip  = document.getElementById('tooltip')  as HTMLDivElement;
const legendEl = document.getElementById('legend')   as HTMLDivElement;
const countEl  = document.getElementById('count')    as HTMLSpanElement;
const statusEl = document.getElementById('status')   as HTMLSpanElement;

// ── State ─────────────────────────────────────────────────────────────────────

let points:   PlottedPoint[]  = [];
let normX:    Float64Array;          // UMAP X scaled to 0..1
let normY:    Float64Array;          // UMAP Y scaled to 0..1
let colorMap: Map<string, string>;

let offsetX = 0, offsetY = 0, scale = 1;
let dragging = false;
let lastPX = 0, lastPY = 0;

// ── Colors ────────────────────────────────────────────────────────────────────

function buildColorMap(pts: PlottedPoint[]): Map<string, string> {
  const tier1s = [...new Set(pts.map(p => p.tier1))].sort();
  return new Map(tier1s.map((t1, i) => [
    t1,
    `hsl(${Math.round((i / tier1s.length) * 360)}, 65%, 58%)`,
  ]));
}

// ── Coordinate transforms ─────────────────────────────────────────────────────

const PAD = 40;

function toScreen(i: number): [number, number] {
  const x = PAD + normX[i] * (canvas.width  - PAD * 2);
  const y = PAD + normY[i] * (canvas.height - PAD * 2);
  return [x * scale + offsetX, y * scale + offsetY];
}

function toNorm(screenX: number, screenY: number): [number, number] {
  return [
    ((screenX - offsetX) / scale - PAD) / (canvas.width  - PAD * 2),
    ((screenY - offsetY) / scale - PAD) / (canvas.height - PAD * 2),
  ];
}

// ── Hit-test ──────────────────────────────────────────────────────────────────

function findNearest(mx: number, my: number): PlottedPoint | null {
  const THRESH = 14 / scale;
  const [nx, ny] = toNorm(mx, my);
  let best: PlottedPoint | null = null;
  let bestD = THRESH;
  for (let i = 0; i < points.length; i++) {
    const d = Math.hypot(normX[i] - nx, normY[i] - ny);
    if (d < bestD) { bestD = d; best = points[i]; }
  }
  return best;
}

// ── Draw ──────────────────────────────────────────────────────────────────────

const TIER_RADIUS: Record<number, number> = { 1: 6, 2: 4, 3: 2.5, 4: 1.8 };

function draw(): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Render deeper tiers first so top-level dots sit on top
  for (let tier = 4; tier >= 1; tier--) {
    ctx.globalAlpha = tier === 1 ? 1 : 0.72;
    for (let i = 0; i < points.length; i++) {
      if (points[i].tier !== tier) continue;
      const [sx, sy] = toScreen(i);
      if (sx < -8 || sx > canvas.width + 8 || sy < -8 || sy > canvas.height + 8) continue;
      ctx.beginPath();
      ctx.arc(sx, sy, TIER_RADIUS[tier] ?? 2, 0, Math.PI * 2);
      ctx.fillStyle = colorMap.get(points[i].tier1) ?? '#888';
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

// ── Legend ────────────────────────────────────────────────────────────────────

function buildLegend(): void {
  const counts = new Map<string, number>();
  for (const p of points) counts.set(p.tier1, (counts.get(p.tier1) ?? 0) + 1);
  legendEl.innerHTML = [...colorMap.entries()].map(([t1, color]) => `
    <div class="legend-row">
      <span class="swatch" style="background:${color}"></span>
      <span class="label">${t1}</span>
      <span class="cnt">${counts.get(t1) ?? 0}</span>
    </div>`).join('') + `
    <div class="tier-hint">
      ${([1,2,3,4] as const).map(t => `
        <div class="tier-hint-row">
          <span class="tier-dot" style="width:${TIER_RADIUS[t]*2}px;height:${TIER_RADIUS[t]*2}px"></span>
          <span>Tier ${t}</span>
        </div>`).join('')}
    </div>`;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  statusEl.textContent = 'Loading embeddings…';

  const url   = new URL('iab-embeddings.json', location.href).href;
  const cache = await caches.open('iab-embeddings');
  const hit   = await cache.match(url);
  let all: EmbeddedEntry[];
  if (hit) {
    all = (await hit.json()) as EmbeddedEntry[];
  } else {
    const fresh = await fetch(url);
    if (!fresh.ok) throw new Error('iab-embeddings.json not found — run npm run generate');
    await cache.put(url, fresh.clone());
    all = (await fresh.json()) as EmbeddedEntry[];
  }
  points = all.filter((e): e is PlottedPoint => Array.isArray(e.umap));

  if (!points.length) {
    throw new Error('No UMAP coordinates found — re-run npm run generate to compute them.');
  }

  colorMap = buildColorMap(points);

  const xs = points.map(p => p.umap[0]);
  const ys = points.map(p => p.umap[1]);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  normX = new Float64Array(points.map(p => (p.umap[0] - minX) / (maxX - minX)));
  normY = new Float64Array(points.map(p => (p.umap[1] - minY) / (maxY - minY)));

  buildLegend();
  countEl.textContent = `${points.length} categories`;
  statusEl.textContent = 'Scroll to zoom · drag to pan';

  new ResizeObserver((entries) => {
    const { width, height } = entries[0].contentRect;
    canvas.width  = Math.floor(width);
    canvas.height = Math.floor(height);
    draw();
  }).observe(canvas);
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  draw();

  // ── Scroll zoom ─────────────────────────────────────────────────────────────
  canvas.addEventListener('wheel', (e: WheelEvent) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    offsetX = e.offsetX + (offsetX - e.offsetX) * f;
    offsetY = e.offsetY + (offsetY - e.offsetY) * f;
    scale  *= f;
    draw();
  }, { passive: false });

  // ── Drag pan ─────────────────────────────────────────────────────────────────
  canvas.addEventListener('pointerdown', (e: PointerEvent) => {
    dragging = true; lastPX = e.offsetX; lastPY = e.offsetY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  });
  window.addEventListener('pointerup', () => {
    dragging = false;
    canvas.style.cursor = 'crosshair';
  });

  // ── Hover tooltip ─────────────────────────────────────────────────────────────
  canvas.addEventListener('pointermove', (e: PointerEvent) => {
    if (dragging) {
      offsetX += e.offsetX - lastPX;
      offsetY += e.offsetY - lastPY;
      lastPX = e.offsetX; lastPY = e.offsetY;
      draw();
    }
    const p = findNearest(e.offsetX, e.offsetY);
    if (p) {
      tooltip.style.display = 'block';
      tooltip.style.left    = `${e.clientX + 14}px`;
      tooltip.style.top     = `${e.clientY + 14}px`;
      tooltip.innerHTML     = `<strong>${p.path}</strong><br><code>${p.id}</code> · T${p.tier}`;
    } else {
      tooltip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
}

main().catch((err: unknown) => {
  statusEl.textContent = err instanceof Error ? err.message : String(err);
});
