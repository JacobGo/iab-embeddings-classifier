import type { TaxonomyEntry } from './types.ts';

export const TSV_URL =
  'https://raw.githubusercontent.com/InteractiveAdvertisingBureau/Taxonomies/develop/Content%20Taxonomies/Content%20Taxonomy%203.1.tsv';

export async function fetchAndParseTaxonomy(): Promise<TaxonomyEntry[]> {
  const res = await fetch(TSV_URL);
  if (!res.ok) throw new Error(`Failed to fetch taxonomy TSV: ${res.status} ${res.statusText}`);
  return parseTsv(await res.text());
}

export function parseTsv(tsv: string): TaxonomyEntry[] {
  const entries: TaxonomyEntry[] = [];

  for (const line of tsv.trim().split('\n')) {
    const cols = line.split('\t');
    const id = cols[0]?.trim();
    // Skip blank lines and the header row if present
    if (!id || id === 'Unique ID') continue;

    const parentId = cols[1]?.trim() || null;
    const name     = cols[2]?.trim();
    const t1       = cols[3]?.trim() || null;
    const t2       = cols[4]?.trim() || null;
    const t3       = cols[5]?.trim() || null;
    const t4       = cols[6]?.trim() || null;

    if (!name || !t1) continue;

    const parts = ([t1, t2, t3, t4] as (string | null)[]).filter((v): v is string => !!v);
    entries.push({
      id,
      parentId,
      name,
      tier1:   t1,
      tier2:   t2,
      tier3:   t3,
      tier4:   t4,
      tier:    parts.length,
      path:    parts.join(' > '),
    });
  }

  return entries;
}
