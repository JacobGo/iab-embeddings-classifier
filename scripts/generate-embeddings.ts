/**
 * Downloads IAB Content Taxonomy 3.1, embeds every category using
 * onnx-community/embeddinggemma-300m-ONNX, and writes the result to
 * public/iab-embeddings.json.
 *
 * The model uses asymmetric prefixes:
 *   Documents (IAB categories) → "title: none | text: <path>"
 *   Queries   (browser input)  → "task: search result | query: <text>"
 *
 * Run: npm run generate
 */

import { AutoModel, AutoTokenizer, env } from '@huggingface/transformers';
import { fetchAndParseTaxonomy } from '../src/parse-taxonomy.ts';
import type { BatchEncoding, EmbeddedEntry, SentenceOutput, TaxonomyEntry } from '../src/types.ts';
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_ID   = 'onnx-community/embeddinggemma-300m-ONNX';
const DTYPE      = 'q8' as const;
const DOC_PREFIX = 'title: none | text: ';

const __dirname = dirname(fileURLToPath(import.meta.url));
env.cacheDir = resolve(__dirname, '../.cache');

// ── HuggingFace callable wrappers ─────────────────────────────────────────────
type HFTokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type HFModel     = Awaited<ReturnType<typeof AutoModel.from_pretrained>>;

function callTokenizer(tok: HFTokenizer, text: string): Promise<BatchEncoding> {
  type Fn = (t: string, opts: { padding: boolean; truncation: boolean; max_length: number }) => Promise<BatchEncoding>;
  return (tok as unknown as Fn)(text, { padding: true, truncation: true, max_length: 512 });
}

function callModel(mdl: HFModel, inputs: BatchEncoding): Promise<SentenceOutput> {
  return (mdl as unknown as (i: BatchEncoding) => Promise<SentenceOutput>)(inputs);
}

function normalize(data: Float32Array): number[] {
  let norm = 0;
  for (const x of data) norm += x * x;
  norm = Math.sqrt(norm);
  return norm > 0 ? Array.from(data, x => x / norm) : Array.from(data);
}

async function embedEntry(
  tok:   HFTokenizer,
  mdl:   HFModel,
  entry: TaxonomyEntry,
): Promise<EmbeddedEntry> {
  const inputs = await callTokenizer(tok, `${DOC_PREFIX}${entry.path}`);
  const output = await callModel(mdl, inputs);
  return {
    id:        entry.id,
    name:      entry.name,
    path:      entry.path,
    tier1:     entry.tier1,
    tier:      entry.tier,
    embedding: normalize(output.sentence_embedding.data),
  };
}

async function main(): Promise<void> {
  console.log('Fetching IAB Content Taxonomy 3.1…');
  const taxonomy = await fetchAndParseTaxonomy();
  console.log(`Parsed ${taxonomy.length} entries.\n`);

  console.log(`Loading model: ${MODEL_ID}  dtype=${DTYPE}`);
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const model     = await AutoModel.from_pretrained(MODEL_ID, { dtype: DTYPE });
  console.log('Model ready.\n');

  const results: EmbeddedEntry[] = [];

  for (const entry of taxonomy) {
    process.stdout.write(`  [${entry.id.padEnd(8)}] ${entry.path}…`);
    results.push(await embedEntry(tokenizer, model, entry));
    process.stdout.write(` (${results[results.length - 1].embedding.length}d) ✓\n`);
  }

  const outPath = resolve(__dirname, '../public/iab-embeddings.json');
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(results));
  console.log(`\nWrote ${results.length} embeddings → ${outPath}`);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
