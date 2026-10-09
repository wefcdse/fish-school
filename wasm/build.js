// 编译两份 wasm（SIMD + 标量）并内联进 sim_wasm.js
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const dir = __dirname;
const simDir = path.join(dir, 'sim');
const target = path.join(simDir, 'target/wasm32-unknown-unknown/release/fish_sim.wasm');
const buildDir = path.join(dir, 'build');
fs.mkdirSync(buildDir, { recursive: true });

function build(env) {
  execSync('cargo build --release --target wasm32-unknown-unknown', {
    cwd: simDir, stdio: 'inherit', env: { ...process.env, ...env }
  });
}

// 1) SIMD 版（用 sim/.cargo/config.toml 的 +simd128）
build({});
fs.copyFileSync(target, path.join(buildDir, 'simd.wasm'));

// 2) 标量版（用环境变量覆盖 rustflags 关掉 simd128）
build({ RUSTFLAGS: '-C target-feature=-simd128' });
fs.copyFileSync(target, path.join(buildDir, 'scalar.wasm'));

const simd = fs.readFileSync(path.join(buildDir, 'simd.wasm')).toString('base64');
const scalar = fs.readFileSync(path.join(buildDir, 'scalar.wasm')).toString('base64');
fs.writeFileSync(path.join(dir, 'sim_wasm.js'),
  'window.SIM_WASM_B64_SIMD = "' + simd + '";\n' +
  'window.SIM_WASM_B64_SCALAR = "' + scalar + '";\n');
console.log('embedded  simd=' + simd.length + '  scalar=' + scalar.length + '  chars');
