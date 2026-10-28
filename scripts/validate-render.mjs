import {readFileSync,existsSync,readdirSync} from 'node:fs';
import {join} from 'node:path';

const base='dist';
const pages=[
  ['',0],
  ['00-prerequisites/transformer',5],
  ['02-data-engineering/hifi-umi-deep',1],
  ['02-data-engineering/kiwi-deep',1],
  ['02-data-engineering/monoego-deep',1],
  ['02-data-engineering/ego-oscar-deep',0],
  ['02-data-engineering/capture-systems-comparison',0],
  ['01-foundations/vla/pi0',4],
  ['01-foundations/vla/delta-flow-matching',12],
  ['01-foundations/vla/pi0-joint-attention-code-audit',8],
  ['02-data-engineering/ego-umi-spatiotemporal-calibration',15],
];
const diagrams=[
  ...['hifi-umi','kiwi','monoego','ego-oscar'].flatMap(name => [name+'-system.svg', name+'-detail.svg']),
  'delta-flow-architecture.svg', 'delta-flow-timeline.svg',
  'pi0-joint-attention-dataflow.svg', 'pi0-joint-attention-mask.svg',
  'ego-umi-calibration-chain.svg', 'ego-umi-calibration-qc.svg'
];
let issues=[];
for(const [slug,minMath] of pages){
 const file=join(base,slug,'index.html');
 if(!existsSync(file)){issues.push('MISSING PAGE '+file);continue;}
 const html=readFileSync(file,'utf8');
 if(html.length<1500)issues.push('SUSPICIOUS SHORT HTML '+file);
 if(/katex-error/.test(html))issues.push('KATEX ERROR '+file);
 const count=(html.match(/class="katex"/g)||[]).length;
 if(count<minMath)issues.push('UNRENDERED MATH '+file+' expected '+minMath+' got '+count);
 if(slug.includes('-deep') && !slug.includes('pi0')){
   if(!html.includes('BOM'))issues.push('MISSING BOM '+file);
   if(!html.includes('复现'))issues.push('MISSING REPRODUCTION '+file);
   if(!html.includes('数据'))issues.push('MISSING DATA MODALITIES '+file);
 }
}
for(const name of diagrams){
 const source=join(base,'diagrams',name);
 if(!existsSync(source)){issues.push('MISSING DIAGRAM '+source);continue;}
 const svg=readFileSync(source,'utf8');
 if(!svg.includes('<svg')||!svg.includes('viewBox')||!svg.includes('</svg>'))
   issues.push('INVALID SVG '+source);
}
const favicon=join(base,'favicon.svg');
if(!existsSync(favicon) || !readFileSync(favicon,'utf8').includes('<svg'))issues.push('MISSING OR INVALID FAVICON '+favicon);
// Scan ALL generated documentation pages, not just newly added pages.
function scan(dir){
 for(const item of readdirSync(dir,{withFileTypes:true})){
  const path=join(dir,item.name);
  if(item.isDirectory()){scan(path);continue;}
  if(!path.endsWith('.html'))continue;
  const html=readFileSync(path,'utf8');
  if(/katex-error/.test(html))issues.push('SITE-WIDE MATH ERROR '+path);
  // Check all handbook-owned static image links, including diagrams.
  for(const match of html.matchAll(/(?:src|href)="\/Embodied-AI-Roadmap\/([^"#?]+\.(?:svg|png|jpg|jpeg|webp))"/g)){
   if(!existsSync(join(base,match[1]))) issues.push('MISSING LOCAL IMAGE '+path+' -> '+match[1]);
  }
 }
}
scan(base);
if(issues.length){
 console.error('SITE ACCEPTANCE FAILED\n'+issues.join('\n'));
 process.exit(1);
}
console.log('PASS: '+pages.length+' HTML pages, math, content sections, '+diagrams.length+' diagrams.');
