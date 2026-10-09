const fs = require('fs');
const path = require('path');
const src = path.join(__dirname, 'sim/target/wasm32-unknown-unknown/release/fish_sim.wasm');
const out = path.join(__dirname, 'sim_wasm.js');
const b = fs.readFileSync(src);
const b64 = b.toString('base64');
fs.writeFileSync(out, 'window.SIM_WASM_B64 = "' + b64 + '";\n');
console.log('wrote', out, b64.length, 'chars');
