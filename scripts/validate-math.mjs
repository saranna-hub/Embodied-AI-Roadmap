import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import katex from 'katex';

const root = 'src/content/docs';
const failures = [];
let count = 0;
function checkFile(path) {
  let source = readFileSync(path, 'utf8');
  source = source.replace(/~~~[\\s\\S]*?~~~/g, '').replace(/```[\\s\\S]*?```/g, '');
  source = source.replace(/`[^`\\n]*`/g, '');
  const delim = String.fromCharCode(36, 36);
  source.split(/\\r?\\n/).forEach((line, i) => {
    if (line.includes(delim) && line.trim() !== delim) {
      failures.push(path + ':' + (i + 1) + ': display math delimiter must occupy its own line');
    }
  });
  const blockRe = /\\$\\$([\\s\\S]*?)\\$\\$/g;
  source = source.replace(blockRe, (_, math) => {
    try { katex.renderToString(math.trim(), { throwOnError: true, displayMode: true }); count++; }
    catch (e) { failures.push(path + ': invalid block math: ' + e.message); }
    return '';
  });
  const inlineRe = /(?<!\\\\)\\$([^$\\n]+?)\\$/g;
  source.replace(inlineRe, (_, math) => {
    try { katex.renderToString(math, { throwOnError: true }); count++; }
    catch (e) { failures.push(path + ': invalid inline math: ' + e.message); }
    return '';
  });
}
function walk(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (/\\.(md|mdx)$/.test(e.name)) checkFile(p);
  }
}
walk(root);
if (failures.length) {
  console.error(failures.join('\\n'));
  process.exit(1);
}
console.log('Validated ' + count + ' mathematical expressions.');
