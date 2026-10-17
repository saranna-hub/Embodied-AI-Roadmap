import { readFileSync, existsSync } from 'node:fs';
const path = 'dist/00-prerequisites/transformer/index.html';
if(!existsSync(path)){ console.error('Rendered page missing: '+path); process.exit(1); }
const html = readFileSync(path,'utf8');
const count = (html.match(/class="katex"/g)||[]).length;
if(count < 5){ console.error('Only '+count+' KaTeX nodes found; Markdown math was not rendered.'); process.exit(1); }
// KaTeX may preserve original TeX inside MathML annotations; that is correct.
const bad = /katex-error|class="katex-error"/.test(html);
if(bad){ console.error('Raw/invalid LaTeX remained in rendered HTML'); process.exit(1); }
console.log('Rendered Transformer math: '+count+' KaTeX elements. PASS');
