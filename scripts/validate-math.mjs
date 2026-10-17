import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import katex from 'katex';

const root = 'src/content/docs';
let checked = 0;
const failures = [];
function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) { walk(path); continue; }
    if (!/\.(md|mdx)$/.test(entry.name)) continue;
    let source = readFileSync(path, 'utf8');
    // Starlight/remark-math requires display delimiters on their own lines.
    // An expression like "$E=mc^2$" may render literally despite being valid KaTeX.
    if (path.endsWith('00-prerequisites/transformer.md')) {
      const lines = source.split(/\r?\n/);
      lines.forEach((line, index) => {
        if (line.includes('$') && line.trim() !== '$') {
          failures.push(`${path}:${index + 1}: display math delimiter $ must occupy its own line`);
        }
      });
    }
    // Avoid treating $ or LaTeX characters inside code snippets as equations.
    source = source.replace(/```[\s\S]*?```/g, '');
    source = source.replace(/`[^`\n]*`/g, '');
    source = source.replace(/\$\$([\s\S]*?)\$\$/g, (_, expr) => {
      try { katex.renderToString(expr.trim(), { throwOnError: true, displayMode: true }); checked++; }
      catch (err) { failures.push(`${path}: block formula ${err.message}`); }
      return '';
    });
    source.replace(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g, (_, expr) => {
      try { katex.renderToString(expr, { throwOnError: true }); checked++; }
      catch (err) { failures.push(`${path}: inline formula ${err.message}`); }
      return '';
    });
  }
}
walk(root);
if (failures.length) {
  console.error('Math validation failed:\n' + failures.join('\n'));
  process.exit(1);
}
console.log(`Checked ${checked} LaTeX expressions in ${root}`);
