/**
 * Browser classifier.
 *
 * Asymmetric usage (matches generate-embeddings.ts):
 *   IAB categories (precomputed) → "title: none | text: <path>"
 *   User query                   → "task: search result | query: <text>"
 *
 * Bookmarklet flow: the bookmarklet encodes page text into the URL hash as
 * #classify=<encoded text>. readPageTextFromHash() reads it at module load so
 * the textarea is pre-filled and classification fires automatically once ready.
 */

import { AutoModel, AutoTokenizer, env } from '@huggingface/transformers';
import type { BatchEncoding, EmbeddedEntry, SentenceOutput } from './types.ts';

const MODEL_ID     = 'onnx-community/embeddinggemma-300m-ONNX';
const DTYPE        = 'q8';
const QUERY_PREFIX = 'task: search result | query: ';
const TOP_N        = 20;
const CACHE_FLAG   = `hf-cached:${MODEL_ID}`;

// ── HuggingFace callable wrappers ─────────────────────────────────────────────
// The library's PreTrainedTokenizer / PreTrainedModel extend Callable but their
// call signature is not surfaced through the TypeScript class type. We cast once
// here through `unknown` (the correct TS pattern) rather than sprinkling `as`
// casts throughout the code.
type HFTokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type HFModel     = Awaited<ReturnType<typeof AutoModel.from_pretrained>>;

function callTokenizer(tok: HFTokenizer, text: string): Promise<BatchEncoding> {
  type Fn = (t: string, opts: { padding: boolean; truncation: boolean; max_length: number }) => Promise<BatchEncoding>;
  return (tok as unknown as Fn)(text, { padding: true, truncation: true, max_length: 512 });
}

function callModel(mdl: HFModel, inputs: BatchEncoding): Promise<SentenceOutput> {
  return (mdl as unknown as (i: BatchEncoding) => Promise<SentenceOutput>)(inputs);
}

// ── DOM refs ─────────────────────────────────────────────────────────────────
const statusEl  = document.getElementById('status')!;
const inputEl   = document.getElementById('input') as HTMLTextAreaElement;
const btnEl     = document.getElementById('classify-btn') as HTMLButtonElement;
const resultsEl = document.getElementById('results')!;
const tierSelEl = document.getElementById('tier-filter') as HTMLSelectElement;
const mapCanvas = document.getElementById('map-canvas') as HTMLCanvasElement;
const mapCtx    = mapCanvas.getContext('2d')!;

// ── Bookmarklet entry ─────────────────────────────────────────────────────────
function readPageTextFromHash(): boolean {
  const hash = window.location.hash;
  if (!hash.startsWith('#classify=')) return false;
  try {
    inputEl.value = decodeURIComponent(hash.slice('#classify='.length));
    history.replaceState(null, '', window.location.pathname + window.location.search);
    return true;
  } catch {
    return false;
  }
}

const autoClassify = readPageTextFromHash();

// ── Model cache helpers ───────────────────────────────────────────────────────
function wasModelCached(): boolean {
  try { return localStorage.getItem(CACHE_FLAG) === '1'; }
  catch { return false; }
}

function markModelCached(): void {
  try { localStorage.setItem(CACHE_FLAG, '1'); }
  catch {}
}

// ── Math helpers ──────────────────────────────────────────────────────────────
function normalize(data: Float32Array): number[] {
  let norm = 0;
  for (const x of data) norm += x * x;
  norm = Math.sqrt(norm);
  return norm > 0 ? Array.from(data, x => x / norm) : Array.from(data);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ── State ─────────────────────────────────────────────────────────────────────
let tokenizer: HFTokenizer | null = null;
let model:     HFModel     | null = null;
let taxonomy:  EmbeddedEntry[]    = [];

let selectedId:   string | null                          = null;
let lastQueryVec: number[]                               = [];
let lastScored:   (EmbeddedEntry & { score: number })[] = [];

env.useBrowserCache  = true;
env.allowLocalModels = false;

// ── Embeddings cache ──────────────────────────────────────────────────────────
async function loadEmbeddings(): Promise<EmbeddedEntry[]> {
  const url   = new URL('iab-embeddings.json', location.href).href;
  const cache = await caches.open('iab-embeddings');
  const hit   = await cache.match(url);
  if (hit) return hit.json() as Promise<EmbeddedEntry[]>;
  const fresh = await fetch(url);
  if (!fresh.ok) throw new Error('iab-embeddings.json not found — run `npm run generate` first.');
  await cache.put(url, fresh.clone());
  return fresh.json() as Promise<EmbeddedEntry[]>;
}

// ── Init ─────────────────────────────────────────────────────────────────────
async function init(): Promise<void> {
  setStatus('Loading taxonomy embeddings…');
  taxonomy = await loadEmbeddings();

  const cached = wasModelCached();
  setStatus(cached
    ? 'Loading model from cache…'
    : `Downloading model (first load only, ~300 MB)…`,
  );

  const progress = (p: unknown) => {
    const evt = p as { status?: string; file?: string; loaded?: number; total?: number };
    if (evt.status === 'progress' && evt.file && evt.total) {
      const pct = Math.round(((evt.loaded ?? 0) / evt.total) * 100);
      setStatus(`Downloading ${evt.file}: ${pct}%`);
    }
  };

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: progress });
  model     = await AutoModel.from_pretrained(MODEL_ID, { dtype: DTYPE, progress_callback: progress });
  markModelCached();

  btnEl.disabled = false;

  if (autoClassify && inputEl.value.trim()) {
    setStatus('Page text loaded — classifying…');
    await classify();
  } else {
    setStatus(`Ready. ${taxonomy.length} IAB 3.1 categories loaded.`);
  }
}

// ── Classify ─────────────────────────────────────────────────────────────────
async function classify(): Promise<void> {
  if (!tokenizer || !model) return;
  const text = inputEl.value.trim();
  if (!text) return;

  btnEl.disabled = true;
  setStatus('Generating embedding…');
  resultsEl.innerHTML = '';

  const inputs   = await callTokenizer(tokenizer, `${QUERY_PREFIX}${text}`);
  const output   = await callModel(model, inputs);
  const queryVec = normalize(output.sentence_embedding.data);

  const minTier = Number(tierSelEl.value) || 1;
  const scored  = taxonomy
    .filter(e => e.tier >= minTier)
    .map(e => ({ ...e, score: cosineSimilarity(queryVec, e.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_N);

  lastQueryVec = queryVec;
  lastScored   = scored;
  selectedId   = null;
  renderResults(scored);
  renderMap(queryVec, scored);
  setStatus(`Showing top ${scored.length} (tier ≥ ${minTier}).`);
  btnEl.disabled = false;
}

// ── Mini-map ──────────────────────────────────────────────────────────────────
// Color-codes every IAB point by Tier 1, highlights top matches, and places
// the query at the weighted centroid of its top-10 UMAP positions.
function renderMap(queryVec: number[], scored: (EmbeddedEntry & { score: number })[], selId: string | null = null): void {
  const hasUmap = taxonomy.some(e => e.umap);
  if (!hasUmap) return; // old JSON without UMAP coords — silently skip

  mapCanvas.style.display = 'block';
  const W = mapCanvas.width  = mapCanvas.offsetWidth;
  const H = mapCanvas.height = mapCanvas.offsetHeight;

  // Normalise UMAP coords of all taxonomy points to canvas space
  const withUmap = taxonomy.filter((e): e is EmbeddedEntry & { umap: [number, number] } => Array.isArray(e.umap));
  const xs = withUmap.map(e => e.umap[0]);
  const ys = withUmap.map(e => e.umap[1]);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const PAD = 16;

  function toXY(u: [number, number]): [number, number] {
    return [
      PAD + ((u[0] - minX) / (maxX - minX)) * (W - PAD * 2),
      PAD + ((u[1] - minY) / (maxY - minY)) * (H - PAD * 2),
    ];
  }

  // Build Tier 1 color map (same hue spread used in viz.ts)
  const tier1s = [...new Set(taxonomy.map(e => e.tier1))].sort();
  const colorOf = (t1: string): string => {
    const i = tier1s.indexOf(t1);
    return `hsl(${Math.round((i / tier1s.length) * 360)}, 65%, 58%)`;
  };

  // Build set of top-match IDs for quick lookup
  const topIds = new Set(scored.map(s => s.id));

  mapCtx.clearRect(0, 0, W, H);

  // 1. All background points (small, dim)
  for (const e of withUmap) {
    if (topIds.has(e.id)) continue;
    const [px, py] = toXY(e.umap);
    mapCtx.beginPath();
    mapCtx.arc(px, py, 2, 0, Math.PI * 2);
    mapCtx.fillStyle = colorOf(e.tier1);
    mapCtx.globalAlpha = 0.25;
    mapCtx.fill();
  }

  // 2. Top-match highlights
  mapCtx.globalAlpha = 1;
  for (const s of [...scored].reverse()) {
    const e = s as EmbeddedEntry & { umap?: [number, number] };
    if (!e.umap) continue;
    const [px, py] = toXY(e.umap);
    const r = 3 + s.score * 4; // radius scales with similarity
    mapCtx.beginPath();
    mapCtx.arc(px, py, r, 0, Math.PI * 2);
    mapCtx.fillStyle = colorOf(e.tier1);
    mapCtx.fill();
    mapCtx.strokeStyle = '#fff';
    mapCtx.lineWidth = 0.8;
    mapCtx.stroke();
  }

  // 3. Query point — weighted centroid of top-10 UMAP positions
  const top10 = scored.filter(s => (s as EmbeddedEntry & { umap?: [number, number] }).umap).slice(0, 10);
  if (top10.length) {
    const totalW = top10.reduce((s, p) => s + p.score, 0);
    let qx = 0, qy = 0;
    for (const p of top10) {
      const e = p as EmbeddedEntry & { umap: [number, number] };
      qx += (e.umap[0] * p.score) / totalW;
      qy += (e.umap[1] * p.score) / totalW;
    }
    const [px, py] = toXY([qx, qy]);

    // Outer glow
    mapCtx.beginPath();
    mapCtx.arc(px, py, 9, 0, Math.PI * 2);
    mapCtx.fillStyle = 'rgba(255,255,255,0.15)';
    mapCtx.fill();
    // Inner dot
    mapCtx.beginPath();
    mapCtx.arc(px, py, 5, 0, Math.PI * 2);
    mapCtx.fillStyle = '#fff';
    mapCtx.fill();
    mapCtx.strokeStyle = '#6366f1';
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
  }

  // 4. Selected-entry ring
  if (selId) {
    const sel = scored.find(s => s.id === selId) as (EmbeddedEntry & { umap?: [number, number] }) | undefined;
    if (sel?.umap) {
      const [px, py] = toXY(sel.umap);
      mapCtx.globalAlpha = 0.3;
      mapCtx.beginPath();
      mapCtx.arc(px, py, 11, 0, Math.PI * 2);
      mapCtx.strokeStyle = '#fff';
      mapCtx.lineWidth = 5;
      mapCtx.stroke();
      mapCtx.globalAlpha = 1;
      mapCtx.beginPath();
      mapCtx.arc(px, py, 8, 0, Math.PI * 2);
      mapCtx.strokeStyle = '#fff';
      mapCtx.lineWidth = 1.5;
      mapCtx.stroke();
    }
  }

  mapCtx.globalAlpha = 1;
}

// ── Render results list ───────────────────────────────────────────────────────
function renderResults(scored: (EmbeddedEntry & { score: number })[]): void {
  if (!scored.length) {
    resultsEl.innerHTML = '<p class="muted">No results.</p>';
    return;
  }
  const top = scored[0].score;
  resultsEl.innerHTML = scored
    .map((c, i) => {
      const pct   = Math.round((c.score / top) * 100);
      const badge = i === 0 ? '<span class="badge">Top match</span>' : '';
      return `
        <div class="result${i === 0 ? ' top' : ''}" data-id="${c.id}" style="cursor:pointer">
          <div class="result-header">
            <span class="tier-tag">T${c.tier}</span>
            <span class="result-path">${c.path}</span>
            ${badge}
            <span class="result-score">${(c.score * 100).toFixed(1)}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    })
    .join('');
}

function setStatus(html: string): void { statusEl.innerHTML = html; }

resultsEl.addEventListener('click', (e: Event) => {
  const card = (e.target as Element).closest<HTMLElement>('[data-id]');
  if (!card) return;
  const id = card.dataset.id ?? null;
  selectedId = selectedId === id ? null : id;
  document.querySelectorAll<HTMLElement>('[data-id]').forEach(el => {
    el.classList.toggle('selected', el.dataset.id === selectedId);
  });
  if (lastScored.length) renderMap(lastQueryVec, lastScored, selectedId);
});

btnEl.addEventListener('click', () => { void classify(); });
inputEl.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void classify();
});

init().catch((err: unknown) => {
  const msg = err instanceof Error ? err.message : String(err);
  setStatus(`<span class="error">Error: ${msg}</span>`);
  console.error(err);
});
