const fs = require('fs');
const src = 'R:/fish-school-wasm/sim/target/wasm32-unknown-unknown/release/fish_sim.wasm';
const out = 'R:/fish-school-wasm/sim_wasm.js';
const b = fs.readFileSync(src);
const b64 = b.toString('base64');
fs.writeFileSync(out, 'window.SIM_WASM_B64 = "' + b64 + '";\n');
console.log('wrote', out, b64.length, 'chars');
