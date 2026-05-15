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

env.useBrowserCache  = true;
env.allowLocalModels = false;

// ── Init ─────────────────────────────────────────────────────────────────────
async function init(): Promise<void> {
  setStatus('Loading taxonomy embeddings…');
  const res = await fetch('iab-embeddings.json');
  if (!res.ok) throw new Error('iab-embeddings.json not found — run `npm run generate` first.');
  taxonomy = (await res.json()) as EmbeddedEntry[];

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

  renderResults(scored);
  setStatus(`Showing top ${scored.length} (tier ≥ ${minTier}).`);
  btnEl.disabled = false;
}

// ── Render ────────────────────────────────────────────────────────────────────
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
        <div class="result${i === 0 ? ' top' : ''}">
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

btnEl.addEventListener('click', () => { void classify(); });
inputEl.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void classify();
});

init().catch((err: unknown) => {
  const msg = err instanceof Error ? err.message : String(err);
  setStatus(`<span class="error">Error: ${msg}</span>`);
  console.error(err);
});
