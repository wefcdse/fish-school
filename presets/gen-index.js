// 自动生成 presets/index.json：扫描本文件夹下所有 .json（不含 index.json）作为预设清单。
// 部署 workflow 在打包前运行本脚本，保证下拉框里的预设永远与文件夹一致。
const fs = require('fs');
const path = require('path');
const dir = __dirname;
const files = fs.readdirSync(dir)
  .filter(f => f.endsWith('.json') && f !== 'index.json')
  .sort();
fs.writeFileSync(path.join(dir, 'index.json'), JSON.stringify({ presets: files }, null, 2) + '\n');
console.log('presets/index.json ->', files.join(', ') || '(empty)');
