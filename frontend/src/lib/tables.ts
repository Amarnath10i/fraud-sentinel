// Pull GitHub-style pipe tables out of the generated markdown reports so
// result tables can be charted, not only read.

export interface Table {
  headers: string[];
  rows: string[][];
}

export function parseTables(markdown: string): Table[] {
  const tables: Table[] = [];
  const lines = markdown.split(/\r?\n/);
  for (let i = 0; i < lines.length - 1; i++) {
    const head = lines[i].trim();
    const sep = lines[i + 1].trim();
    if (!head.startsWith("|") || !/^\|[\s:|-]+\|$/.test(sep)) continue;
    const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    const table: Table = { headers: cells(head), rows: [] };
    let j = i + 2;
    while (j < lines.length && lines[j].trim().startsWith("|")) table.rows.push(cells(lines[j++]));
    tables.push(table);
    i = j - 1;
  }
  return tables;
}

/** "0.9903 (0.987-0.994)" -> {point, low, high}; plain numbers have no interval. */
export function parseInterval(cell: string): { point: number; low?: number; high?: number } | null {
  const m = cell.replace(/[$,]/g, "").match(/^(-?[\d.]+)(?:\s*\((-?[\d.]+)\s*-\s*(-?[\d.]+)\))?/);
  if (!m) return null;
  const point = Number(m[1]);
  if (Number.isNaN(point)) return null;
  return m[2] ? { point, low: Number(m[2]), high: Number(m[3]) } : { point };
}
