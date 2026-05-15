# IAB Content Classifier

Classifies webpage content against the [IAB Content Taxonomy 3.1](https://github.com/InteractiveAdvertisingBureau/Taxonomies) using semantic embeddings. Runs entirely in the browser — no server, no API calls at classify time.

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│  SETUP (once, Node.js)                                          │
│                                                                 │
│  1. fetch-taxonomy ──► parse ~700 IAB categories from TSV       │
│         │                                                       │
│         ▼                                                       │
│  2. embed each category path using EmbeddingGemma 300M          │
│     "Technology & Computing > Computing > Internet > Cloud"     │
│     with doc prefix: "title: none | text: <path>"              │
│         │                                                       │
│         ▼                                                       │
│  3. UMAP 2D projection of all embeddings (umap-js)              │
│  4. L2-normalise + write ──► public/iab-embeddings.json         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RUNTIME (browser)                                              │
│                                                                 │
│  1. Load iab-embeddings.json  (precomputed, ~10 MB)             │
│     ► cached in Cache Storage after first load                  │
│  2. Load EmbeddingGemma 300M ONNX (q8, WASM)                   │
│     ► cached in Cache Storage after first load                  │
│                                                                 │
│  3. User inputs text (typed or via bookmarklet)                 │
│         │                                                       │
│         ▼                                                       │
│  4. Embed with query prefix: "task: search result | query: …"   │
│  5. Cosine similarity against all ~700 precomputed embeddings   │
│         │                                                       │
│         ▼                                                       │
│  6. Rank and display top 20 matches with similarity bars        │
│  7. Mini-map: query plotted in UMAP space (weighted centroid    │
│     of top-10 matches); click any result to highlight its dot  │
└─────────────────────────────────────────────────────────────────┘
```

### Why asymmetric prefixes?

EmbeddingGemma is trained for asymmetric retrieval. Documents and queries live in different parts of the embedding space. Using the wrong prefix produces lower-quality matches.

| Role | Prefix |
|---|---|
| IAB category (document) | `title: none \| text: <full path>` |
| User input (query) | `task: search result \| query: <text>` |

### Model

[`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) — INT8 quantised (`q8`), 768-dimensional output. The base model is `google/embeddinggemma-300m`. Uses Matryoshka Representation Learning so the embedding can optionally be truncated to 512/256/128d without retraining.

---

## Usage

### Prerequisites

- Node.js 18+
- npm

### 1. Install

```bash
npm install
```

### 2. Generate taxonomy embeddings

Downloads the IAB 3.1 TSV directly from the IAB GitHub repo, embeds every category (~700 entries), computes a UMAP 2D projection, and writes `public/iab-embeddings.json`.

```bash
npm run generate
```

This takes a few minutes on first run because it downloads the EmbeddingGemma model (~300 MB) into `.cache/`. Subsequent runs reuse the cache.

### 3. Serve

Builds `src/app.ts → public/app.js` and `src/viz.ts → public/viz.js`, then starts a local static server.

```bash
npm run serve
# → http://localhost:3000
```

The browser downloads the model on first visit and stores it in Cache Storage. Subsequent visits load from cache instantly.

---

## Embedding space visualisation

The app ships two ways to explore the UMAP projection of the IAB taxonomy embedding space:

**Mini-map** (inline, right of results): appears after the first classification. Shows all taxonomy points coloured by Tier 1 category, with top-20 matches highlighted and the query plotted as a white dot at the weighted centroid of its top-10 matches. Click any result card to ring its point on the map.

**Full-screen explorer** (`/viz.html`): pan with click-drag, zoom with scroll wheel, hover for category path tooltip. Linked via "Full map ↗" in the map panel header.

---

## Bookmarklet

The app includes a one-click bookmarklet to classify any webpage without copy-pasting.

1. Open `http://localhost:3000`
2. Scroll to **Bookmarklet setup** at the bottom of the page
3. Drag **☆ Classify this page** to your bookmarks bar

**How it works:** clicking the bookmarklet on any page grabs `document.body.innerText` (first 8,000 characters), encodes it into the URL hash (`#classify=<text>`), and opens `localhost:3000` in a new tab. The app reads the hash at load time, pre-fills the textarea, and auto-classifies once the model is ready.

The hash fragment is never sent to any server — all processing happens locally in your browser.

---

## Project structure

```
├── scripts/
│   └── generate-embeddings.ts   # Node.js: fetch TSV → embed → UMAP → write JSON
├── src/
│   ├── types.ts                 # Shared TypeScript interfaces
│   ├── parse-taxonomy.ts        # Fetch + parse IAB 3.1 TSV from GitHub
│   ├── app.ts                   # Browser classifier (compiled → public/app.js)
│   └── viz.ts                   # Full-screen UMAP explorer (compiled → public/viz.js)
├── public/
│   ├── index.html               # Classifier UI + import map
│   ├── viz.html                 # Full-screen embedding space explorer
│   ├── favicon.svg              # SVG favicon (embedding scatter plot)
│   ├── app.js                   # Compiled output (generated by npm run build)
│   ├── viz.js                   # Compiled output (generated by npm run build)
│   └── iab-embeddings.json      # Precomputed embeddings + UMAP coords (generated)
└── tsconfig.json
```

---

## Potential optimisations

### Embedding quality

- **Use the full page, not just a text slice** — the bookmarklet truncates to 8,000 characters to fit in a URL hash. A Chrome extension or local proxy could send the full body without the length constraint.
- **Chunk + pool** — for long documents, split into ~400-token chunks, embed each, then average the embeddings rather than truncating at the tokeniser's `max_length`.
- **Weighted text extraction** — extract `<title>`, `<h1>`–`<h3>`, and `<meta name="description">` separately, embed them at higher weight, and blend with body-text embeddings.
- **Ensemble over Matryoshka dimensions** — the model supports 128/256/512/768d. Running at 256d is ~9× faster with modest accuracy loss; useful for interactive re-ranking.

### Classification accuracy

- **Multi-label output** — a page can belong to multiple IAB categories. Threshold similarity scores (e.g. > 0.6) rather than always picking top-N.
- **Hierarchical re-ranking** — first match at Tier 1, then re-embed against the Tier 2/3 children of the winner for faster narrowing without comparing all ~700 entries.
- **Fine-tune on labelled data** — few-shot contrastive fine-tuning with known page → IAB label pairs would significantly improve precision.

### Performance

- **WebGPU** — `@huggingface/transformers` supports WebGPU backends. Set `device: 'webgpu'` and `dtype: 'fp16'` for ~5–10× inference speedup on supported hardware (Chrome 113+, Edge 113+).
- **Precompute in a Web Worker** — move model loading and inference off the main thread entirely to keep the UI responsive during classification.
- **Quantise to q4** — halves model download size (~150 MB vs ~300 MB) with minor accuracy regression; worthwhile if cold-start time matters more than quality.
- **OPFS model cache** — the Origin Private File System API provides more reliable large-file persistence than Cache Storage (which browsers may evict under memory pressure). Store the model weights in OPFS after first download.
- **True UMAP projection for query** — re-run UMAP with the query embedding included (or use parametric UMAP) to get an exact query position rather than the weighted-centroid approximation.

### Developer experience

- **Bundle with esbuild `--bundle`** + tree-shaking to remove unused `@huggingface/transformers` code from the browser payload.
- **Watch mode** — `esbuild --watch` for `src/app.ts` so the browser build rebuilds on save during development.
- **Type-check in CI** — `tsc --noEmit` catches type errors without needing a full build; add to a pre-commit hook or CI step.
