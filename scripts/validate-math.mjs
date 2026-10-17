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
    // Detect display-math delimiters embedded in text lines.
    if (path.endsWith('00-prerequisites/transformer.md')) {
      const delimiter = String.fromCharCode(36, 36);
      const lines = source.split(/\r?\n/);
      lines.forEach((line, index) => {
        if (line.includes(delimiter) && line.trim() !== delimiter) {
          failures.push(path + ':' + (index + 1) + ': display math must be on its own line');
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
 + ') {
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
) && line.trim() !== ') {
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
 + ') {
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
) {
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
