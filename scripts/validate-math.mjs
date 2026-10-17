import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import katex from 'katex';

const failures = [];
let checked = 0;
function inspect(path) {
  let source = readFileSync(path, 'utf8');
  source = source.replace(/~~~[\s\S]*?~~~/g, '');
  source = source.replace(/```[\s\S]*?```/g, '');
  source = source.replace(/`[^`\n]*`/g, '');
  const mark = String.fromCharCode(36,36);
  source.split(/\r?\n/).forEach((line,i) => {
    if (line.includes(mark) && line.trim() !== mark)
      failures.push(path + ':' + (i+1) + ' display math delimiter must be on its own line');
  });
  source = source.replace(/\$\$([\s\S]*?)\$\$/g, (_,expr) => {
    try { katex.renderToString(expr.trim(),{throwOnError:true,displayMode:true}); checked++; }
    catch(e) { failures.push(path + ': block: ' + e.message); }
    return '';
  });
  source.replace(/(?<!\\)\$([^$\n]+?)\$/g, (_,expr) => {
    try { katex.renderToString(expr,{throwOnError:true}); checked++; }
    catch(e) { failures.push(path + ': inline: ' + e.message); }
    return '';
  });
}
function walk(path) {
  for (const entry of readdirSync(path,{withFileTypes:true})) {
    const p = join(path,entry.name);
    if (entry.isDirectory()) walk(p);
    else if (/\.(md|mdx)$/.test(entry.name)) inspect(p);
  }
}
walk('src/content/docs');
if(failures.length){console.error(failures.join('\n'));process.exit(1);}
console.log('Validated '+checked+' math formulas');
