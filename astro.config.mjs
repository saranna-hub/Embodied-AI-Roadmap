import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://saranna-hub.github.io',
  base: '/Embodied-AI-Roadmap',
  integrations: [starlight({
    title: 'Embodied AI Roadmap',
    description: '具身智能学习手册：论文原理、训练方法、源码与机器人实践',
    defaultLocale: 'root',
    locales: { root: { label: '简体中文', lang: 'zh-CN' } },
    customCss: ['./src/styles/custom.css'],
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/saranna-hub/Embodied-AI-Roadmap' }],
    sidebar: [
      { label: '开始学习', items: [{label:'阅读指南',slug:'start'}, {label:'技术地图',slug:'taxonomy'}, {label:'研究记录',slug:'radar'}] },
      { label: '00 预备知识', autogenerate: { directory: '00-prerequisites' } },
      { label: '01 具身基础模型', autogenerate: { directory: '01-foundations' } },
      { label: '02 具身数据工程', autogenerate: { directory: '02-data-engineering' } },
      { label: '03 感知与重建', autogenerate: { directory: '03-perception' } },
      { label: '04 Human-to-Robot', autogenerate: { directory: '04-human-to-robot' } },
      { label: '05 控制与执行', autogenerate: { directory: '05-control' } },
      { label: '06 仿真与评测', autogenerate: { directory: '06-simulation' } }
    ]
  })],
  markdown: { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }
});
