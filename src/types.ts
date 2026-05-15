// Minimal interfaces for the @huggingface/transformers callable API.
// The library's PreTrainedTokenizer / PreTrainedModel extend Callable, which
// TypeScript can't represent as a call signature on the class type, so we cast
// through these once in the callTokenizer / callModel wrappers in app.ts and
// generate-embeddings.ts.
export interface BatchEncoding {
  input_ids: unknown;
  attention_mask: unknown;
  [key: string]: unknown;
}

export interface SentenceOutput {
  sentence_embedding: { data: Float32Array };
}

export interface TaxonomyEntry {
  id: string;
  parentId: string | null;
  name: string;
  tier1: string;
  tier2: string | null;
  tier3: string | null;
  tier4: string | null;
  tier: number;
  path: string; // e.g. "Technology & Computing > Computing > Internet > Cloud Computing"
}

export interface EmbeddedEntry {
  id: string;
  name: string;
  path: string;
  tier1: string;
  tier: number;
  embedding: number[];
}
