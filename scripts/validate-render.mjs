import {readFileSync,existsSync} from 'node:fs';
import {join} from 'node:path';

const base='dist';
const pages=[
  ['00-prerequisites/transformer',5],
  ['02-data-engineering/hifi-umi-deep',1],
  ['02-data-engineering/kiwi-deep',1],
  ['02-data-engineering/monoego-deep',1],
  ['02-data-engineering/ego-oscar-deep',0],
  ['02-data-engineering/capture-systems-comparison',0],
  ['01-foundations/vla/pi0',4],
];
const diagrams=[
 'hifi-umi-system.svg','kiwi-system.svg',
 'monoego-system.svg','ego-oscar-system.svg'
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
if(issues.length){
 console.error('SITE ACCEPTANCE FAILED\n'+issues.join('\n'));
 process.exit(1);
}
console.log('PASS: '+pages.length+' HTML pages, math, content sections, '+diagrams.length+' diagrams.');
