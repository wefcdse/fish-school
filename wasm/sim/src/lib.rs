// Boids 鱼群模拟 —— 迁移自 index.html 的 JS 逻辑
// 无 wasm-bindgen：纯 C ABI + 导出 memory，便于内联 base64 直接实例化。

const PERCEPTION: f32 = 60.0;
const SEP_RADIUS: f32 = 22.0;
const MOUSE_REPEL_RADIUS: f32 = 70.0;
const SCATTER_RADIUS: f32 = 260.0;
const EDGE_ZONE: f32 = 80.0;
const REF_TICKS: f32 = 60.0;
const CELL_SIZE: f32 = PERCEPTION;
const PARAM_LEN: usize = 24;

#[derive(Default, Clone, Copy)]
struct Acc {
    cdx: f32, cdy: f32, avx: f32, avy: f32, sdx: f32, sdy: f32,
    cnt: i32, scnt: i32,
}

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn hsum_f32(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
}
#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn hsum_i32(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v) + i32x4_extract_lane::<1>(v)
        + i32x4_extract_lane::<2>(v) + i32x4_extract_lane::<3>(v)
}

// 处理邻格区间 [s,e)，把凝聚/对齐/分离的累加值并入 acc
#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn accum_range(
    xp: *const f32, yp: *const f32, vxp: *const f32, vyp: *const f32,
    s: usize, e: usize,
    fx: f32, fy: f32, hw: f32, ww: f32, hh: f32, wh: f32,
    perception2: f32, sep2: f32, wrap: bool,
    acc: &mut Acc,
) {
    let fxv = f32x4_splat(fx); let fyv = f32x4_splat(fy);
    let hwv = f32x4_splat(hw); let wwv = f32x4_splat(ww);
    let hhv = f32x4_splat(hh); let whv = f32x4_splat(wh);
    let nhwv = f32x4_splat(-hw); let nhhv = f32x4_splat(-hh);
    let zv = f32x4_splat(0.0);
    let p2v = f32x4_splat(perception2); let sepv = f32x4_splat(sep2);
    let one = i32x4_splat(1);

    let mut cdxv = zv; let mut cdyv = zv;
    let mut avxv = zv; let mut avyv = zv;
    let mut sdxv = zv; let mut sdyv = zv;
    let mut cntv = i32x4_splat(0); let mut scntv = i32x4_splat(0);

    let mut k = s;
    while k + 4 <= e {
        let mut dxv = f32x4_sub(v128_load(xp.add(k) as *const v128), fxv);
        let mut dyv = f32x4_sub(v128_load(yp.add(k) as *const v128), fyv);
        if wrap {
            let gx = f32x4_gt(dxv, hwv);
            let lx = f32x4_lt(dxv, nhwv);
            let dxa = v128_bitselect(f32x4_sub(dxv, wwv), dxv, gx);
            dxv = v128_bitselect(f32x4_add(dxa, wwv), dxa, lx);
            let gy = f32x4_gt(dyv, hhv);
            let ly = f32x4_lt(dyv, nhhv);
            let dya = v128_bitselect(f32x4_sub(dyv, whv), dyv, gy);
            dyv = v128_bitselect(f32x4_add(dya, whv), dya, ly);
        }
        let d2 = f32x4_add(f32x4_mul(dxv, dxv), f32x4_mul(dyv, dyv));
        let mgt = f32x4_gt(d2, zv);
        let m = v128_and(mgt, f32x4_lt(d2, p2v));
        cdxv = f32x4_add(cdxv, v128_bitselect(dxv, zv, m));
        cdyv = f32x4_add(cdyv, v128_bitselect(dyv, zv, m));
        avxv = f32x4_add(avxv, v128_bitselect(v128_load(vxp.add(k) as *const v128), zv, m));
        avyv = f32x4_add(avyv, v128_bitselect(v128_load(vyp.add(k) as *const v128), zv, m));
        cntv = i32x4_add(cntv, v128_and(m, one));
        let ms = v128_and(mgt, f32x4_lt(d2, sepv));
        sdxv = f32x4_sub(sdxv, v128_bitselect(dxv, zv, ms));
        sdyv = f32x4_sub(sdyv, v128_bitselect(dyv, zv, ms));
        scntv = i32x4_add(scntv, v128_and(ms, one));
        k += 4;
    }
    acc.cdx += hsum_f32(cdxv); acc.cdy += hsum_f32(cdyv);
    acc.avx += hsum_f32(avxv); acc.avy += hsum_f32(avyv);
    acc.sdx += hsum_f32(sdxv); acc.sdy += hsum_f32(sdyv);
    acc.cnt += hsum_i32(cntv); acc.scnt += hsum_i32(scntv);

    while k < e {
        let mut dx = *xp.add(k) - fx;
        let mut dy = *yp.add(k) - fy;
        if wrap {
            if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
            if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
        }
        let d2 = dx * dx + dy * dy;
        if d2 > 0.0 && d2 < perception2 {
            acc.cdx += dx; acc.cdy += dy;
            acc.avx += *vxp.add(k); acc.avy += *vyp.add(k);
            acc.cnt += 1;
            if d2 < sep2 { acc.sdx -= dx; acc.sdy -= dy; acc.scnt += 1; }
        }
        k += 1;
    }
}

#[cfg(not(target_feature = "simd128"))]
#[inline]
unsafe fn accum_range(
    xp: *const f32, yp: *const f32, vxp: *const f32, vyp: *const f32,
    s: usize, e: usize,
    fx: f32, fy: f32, hw: f32, ww: f32, hh: f32, wh: f32,
    perception2: f32, sep2: f32, wrap: bool,
    acc: &mut Acc,
) {
    let mut k = s;
    while k < e {
        let mut dx = *xp.add(k) - fx;
        let mut dy = *yp.add(k) - fy;
        if wrap {
            if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
            if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
        }
        let d2 = dx * dx + dy * dy;
        if d2 > 0.0 && d2 < perception2 {
            acc.cdx += dx; acc.cdy += dy;
            acc.avx += *vxp.add(k); acc.avy += *vyp.add(k);
            acc.cnt += 1;
            if d2 < sep2 { acc.sdx -= dx; acc.sdy -= dy; acc.scnt += 1; }
        }
        k += 1;
    }
}

pub struct Sim {
    x: Vec<f32>, y: Vec<f32>, vx: Vec<f32>, vy: Vec<f32>,
    size: Vec<f32>, hue: Vec<f32>, phase: Vec<f32>, wander: Vec<f32>,
    accx: Vec<f32>, accy: Vec<f32>,
    gx: Vec<i32>, gy: Vec<i32>,
    // 按格重排用的暂存缓冲（每帧排列后与主缓冲交换）
    sx: Vec<f32>, sy: Vec<f32>, svx: Vec<f32>, svy: Vec<f32>,
    ssize: Vec<f32>, shue: Vec<f32>, sphase: Vec<f32>, swander: Vec<f32>,
    sgx: Vec<i32>, sgy: Vec<i32>,
    cell_idx: Vec<u32>, cell_start: Vec<u32>, cell_cursor: Vec<u32>, order: Vec<u32>,
    last_ncells: usize,
    stat_lens: Vec<u32>,
    rng: u32,
}

impl Sim {
    fn new() -> Self {
        Sim {
            x: Vec::new(), y: Vec::new(), vx: Vec::new(), vy: Vec::new(),
            size: Vec::new(), hue: Vec::new(), phase: Vec::new(), wander: Vec::new(),
            accx: Vec::new(), accy: Vec::new(),
            gx: Vec::new(), gy: Vec::new(),
            sx: Vec::new(), sy: Vec::new(), svx: Vec::new(), svy: Vec::new(),
            ssize: Vec::new(), shue: Vec::new(), sphase: Vec::new(), swander: Vec::new(),
            sgx: Vec::new(), sgy: Vec::new(),
            cell_idx: Vec::new(), cell_start: Vec::new(), cell_cursor: Vec::new(), order: Vec::new(),
            last_ncells: 0,
            stat_lens: Vec::new(),
            rng: 0x1234_5678,
        }
    }

    #[inline]
    fn rand(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.rng = x;
        (x >> 8) as f32 / 16_777_216.0
    }

    fn push_fish(&mut self, left: f32, top: f32, right: f32, bottom: f32, speed: f32) {
        let w = right - left;
        let h = bottom - top;
        let (rx, ry, ang, sz, hue, ph, wa) = {
            let rx = self.rand();
            let ry = self.rand();
            let ang = self.rand() * std::f32::consts::TAU;
            let sz = 4.0 + self.rand() * 3.5;
            let hue = 185.0 + self.rand() * 55.0;
            let ph = self.rand() * std::f32::consts::TAU;
            let wa = self.rand() * std::f32::consts::TAU;
            (rx, ry, ang, sz, hue, ph, wa)
        };
        self.x.push(left + rx * w);
        self.y.push(top + ry * h);
        self.vx.push(ang.cos() * speed);
        self.vy.push(ang.sin() * speed);
        self.size.push(sz);
        self.hue.push(hue);
        self.phase.push(ph);
        self.wander.push(wa);
        self.accx.push(0.0);
        self.accy.push(0.0);
        self.gx.push(0);
        self.gy.push(0);
    }

    fn truncate(&mut self, n: usize) {
        self.x.truncate(n); self.y.truncate(n); self.vx.truncate(n); self.vy.truncate(n);
        self.size.truncate(n); self.hue.truncate(n); self.phase.truncate(n); self.wander.truncate(n);
        self.accx.truncate(n); self.accy.truncate(n); self.gx.truncate(n); self.gy.truncate(n);
    }

    fn set_count(&mut self, n: usize, left: f32, top: f32, right: f32, bottom: f32, speed: f32) {
        while self.x.len() < n { self.push_fish(left, top, right, bottom, speed); }
        self.truncate(n);
    }

    fn reset(&mut self, left: f32, top: f32, right: f32, bottom: f32, speed: f32) {
        let n = self.x.len();
        self.x.clear(); self.y.clear(); self.vx.clear(); self.vy.clear();
        self.size.clear(); self.hue.clear(); self.phase.clear(); self.wander.clear();
        self.accx.clear(); self.accy.clear(); self.gx.clear(); self.gy.clear();
        for _ in 0..n { self.push_fish(left, top, right, bottom, speed); }
    }

    fn rescale(&mut self, cx: f32, cy: f32, ratio: f32) {
        for i in 0..self.x.len() {
            self.x[i] = cx + (self.x[i] - cx) * ratio;
            self.y[i] = cy + (self.y[i] - cy) * ratio;
        }
    }

    fn paint(&mut self, wx: f32, wy: f32, r: f32, hue: f32, p: &[f32; PARAM_LEN]) {
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let ww = right - left; let wh = bottom - top;
        let hw = ww * 0.5; let hh = wh * 0.5;
        let r2 = r * r;
        for i in 0..self.x.len() {
            let mut dx = self.x[i] - wx;
            let mut dy = self.y[i] - wy;
            if wrap {
                if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
                if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
            }
            if dx * dx + dy * dy <= r2 { self.hue[i] = hue; }
        }
    }

    fn step(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let nstep = dt * REF_TICKS;

        let cohesion = p[0]; let separation = p[1]; let align = p[2]; let speed = p[3];
        let mouse_field = p[4] > 0.5;
        let mx = p[5]; let my = p[6]; let mradius = p[7]; let mattract = p[8]; let mrepel = p[9];
        let left_down = p[10] > 0.5; let scatter_f = p[11];
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let zoom = p[17];
        let wander_scale = p[18];
        let ww = right - left; let wh = bottom - top;
        let hw = ww * 0.5; let hh = wh * 0.5;

        let perception2 = PERCEPTION * PERCEPTION;
        let sep2 = SEP_RADIUS * SEP_RADIUS;

        // 网格几何：统一到 cell 坐标 [0,cols)×[0,rows)
        let (cols, rows, cw, ch, min_gx, min_gy);
        if wrap {
            let ci = ((ww / CELL_SIZE) as i32).max(3);
            let ri = ((wh / CELL_SIZE) as i32).max(3);
            cols = ci; rows = ri;
            cw = ww / ci as f32; ch = wh / ri as f32;
            min_gx = 0; min_gy = 0;
        } else {
            min_gx = (left / CELL_SIZE).floor() as i32;
            min_gy = (top / CELL_SIZE).floor() as i32;
            let max_gx = (right / CELL_SIZE).floor() as i32;
            let max_gy = (bottom / CELL_SIZE).floor() as i32;
            cols = (max_gx - min_gx + 1).max(1);
            rows = (max_gy - min_gy + 1).max(1);
            cw = CELL_SIZE; ch = CELL_SIZE;
        }
        let cols_u = cols.max(1) as usize;
        let rows_u = rows.max(1) as usize;
        let ncells = cols_u * rows_u;
        self.last_ncells = ncells;

        // 计算每条鱼的格号
        if self.cell_idx.len() < n { self.cell_idx.resize(n, 0); }
        if self.order.len() < n { self.order.resize(n, 0); }
        if self.cell_start.len() < ncells + 1 { self.cell_start.resize(ncells + 1, 0); }
        if self.cell_cursor.len() < ncells { self.cell_cursor.resize(ncells, 0); }

        for i in 0..n {
            let (gx, gy) = if wrap {
                let gx0 = ((self.x[i] - left) / cw).floor() as i32;
                let gy0 = ((self.y[i] - top) / ch).floor() as i32;
                (gx0.rem_euclid(cols), gy0.rem_euclid(rows))
            } else {
                (
                    ((self.x[i] / CELL_SIZE).floor() as i32 - min_gx).clamp(0, cols - 1),
                    ((self.y[i] / CELL_SIZE).floor() as i32 - min_gy).clamp(0, rows - 1),
                )
            };
            self.gx[i] = gx; self.gy[i] = gy;
            self.cell_idx[i] = ((gy as usize) * cols_u + gx as usize) as u32;
        }

        // 计数排序：把鱼索引按格号排进连续数组（CSR）
        self.cell_start[..=ncells].fill(0);
        for i in 0..n { self.cell_start[self.cell_idx[i] as usize] += 1; }
        let mut running = 0u32;
        for c in 0..ncells {
            let cnt = self.cell_start[c];
            self.cell_start[c] = running;
            self.cell_cursor[c] = running;
            running += cnt;
        }
        self.cell_start[ncells] = running;
        for i in 0..n {
            let c = self.cell_idx[i] as usize;
            let pos = self.cell_cursor[c];
            self.order[pos as usize] = i as u32;
            self.cell_cursor[c] = pos + 1;
        }

        // 按格重排鱼数据：邻格读取由随机 gather 变为连续内存
        self.sx.resize(n, 0.0); self.sy.resize(n, 0.0);
        self.svx.resize(n, 0.0); self.svy.resize(n, 0.0);
        self.ssize.resize(n, 0.0); self.shue.resize(n, 0.0);
        self.sphase.resize(n, 0.0); self.swander.resize(n, 0.0);
        self.sgx.resize(n, 0); self.sgy.resize(n, 0);
        for pos in 0..n {
            let i = self.order[pos] as usize;
            self.sx[pos] = self.x[i];
            self.sy[pos] = self.y[i];
            self.svx[pos] = self.vx[i];
            self.svy[pos] = self.vy[i];
            self.ssize[pos] = self.size[i];
            self.shue[pos] = self.hue[i];
            self.sphase[pos] = self.phase[i];
            self.swander[pos] = self.wander[i];
            self.sgx[pos] = self.gx[i];
            self.sgy[pos] = self.gy[i];
        }
        std::mem::swap(&mut self.x, &mut self.sx);
        std::mem::swap(&mut self.y, &mut self.sy);
        std::mem::swap(&mut self.vx, &mut self.svx);
        std::mem::swap(&mut self.vy, &mut self.svy);
        std::mem::swap(&mut self.size, &mut self.ssize);
        std::mem::swap(&mut self.hue, &mut self.shue);
        std::mem::swap(&mut self.phase, &mut self.sphase);
        std::mem::swap(&mut self.wander, &mut self.swander);
        std::mem::swap(&mut self.gx, &mut self.sgx);
        std::mem::swap(&mut self.gy, &mut self.sgy);

        // 逐鱼求力（Jacobi：先全部算力，再统一积分）
        for i in 0..n {
            let fx = self.x[i]; let fy = self.y[i];
            let gx = self.gx[i]; let gy = self.gy[i];
            let mut acc = Acc::default();

            // 邻格列/行索引：wrap 用条件回绕（无整数除法），非 wrap 用 -1 表示越界
            let xc0 = if wrap { if gx == 0 { cols - 1 } else { gx - 1 } } else { gx - 1 };
            let xc2 = if wrap { if gx == cols - 1 { 0 } else { gx + 1 } } else { gx + 1 };
            let yc0 = if wrap { if gy == 0 { rows - 1 } else { gy - 1 } } else { gy - 1 };
            let yc2 = if wrap { if gy == rows - 1 { 0 } else { gy + 1 } } else { gy + 1 };
            let xcs = [xc0, gx, xc2];
            let ycs = [yc0, gy, yc2];

            unsafe {
                let xp = self.x.as_ptr(); let yp = self.y.as_ptr();
                let vxp = self.vx.as_ptr(); let vyp = self.vy.as_ptr();
                for &cx in xcs.iter() {
                    if cx < 0 || cx >= cols { continue; }
                    let cxb = cx as usize;
                    for &cy in ycs.iter() {
                        if cy < 0 || cy >= rows { continue; }
                        let c = cy as usize * cols_u + cxb;
                        let s = self.cell_start[c] as usize;
                        let e = self.cell_start[c + 1] as usize;
                        if s == e { continue; }
                        accum_range(xp, yp, vxp, vyp, s, e, fx, fy, hw, ww, hh, wh, perception2, sep2, wrap, &mut acc);
                    }
                }
            }

            let cdx = acc.cdx; let cdy = acc.cdy;
            let avx = acc.avx; let avy = acc.avy;
            let sdx = acc.sdx; let sdy = acc.sdy;
            let cnt = acc.cnt; let scnt = acc.scnt;

            let mut accx = 0.0f32; let mut accy = 0.0f32;
            if cnt > 0 {
                let nf = cnt as f32;
                let tx = cdx / nf; let ty = cdy / nf;
                let td = (tx * tx + ty * ty).sqrt().max(1e-6);
                accx += tx / td * cohesion * 0.32;
                accy += ty / td * cohesion * 0.32;
                let nvx = avx / nf; let nvy = avy / nf;
                let ad = (nvx * nvx + nvy * nvy).sqrt().max(1e-6);
                accx += nvx / ad * align * 0.10;
                accy += nvy / ad * align * 0.10;
            }
            if scnt > 0 {
                let sd = (sdx * sdx + sdy * sdy).sqrt().max(1e-6);
                accx += sdx / sd * separation * 0.35;
                accy += sdy / sd * separation * 0.35;
            }

            let dw = (self.rand() - 0.5) * 0.5 * nstep;
            self.wander[i] += dw;
            accx += self.wander[i].cos() * 0.02 * wander_scale;
            accy += self.wander[i].sin() * 0.02 * wander_scale;

            if mouse_field {
                let mut dx = mx - fx; let mut dy = my - fy;
                if wrap {
                    if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
                    if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
                }
                let d = (dx * dx + dy * dy).sqrt().max(1e-6);
                if d < mradius {
                    let pull = (1.0 - d / mradius) * mattract;
                    accx += dx / d * pull; accy += dy / d * pull;
                }
                if d < MOUSE_REPEL_RADIUS {
                    let push = (1.0 - d / MOUSE_REPEL_RADIUS) * mrepel;
                    accx -= dx / d * push; accy -= dy / d * push;
                }
            }

            if left_down {
                let mut dx = fx - mx; let mut dy = fy - my;
                if wrap {
                    if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
                    if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
                }
                let d = (dx * dx + dy * dy).sqrt().max(1e-6);
                let pp = (1.0 - d / SCATTER_RADIUS).max(0.0);
                accx += dx / d * pp * scatter_f;
                accy += dy / d * pp * scatter_f;
            }

            self.accx[i] = accx;
            self.accy[i] = accy;
        }

        // 积分 + 边界
        let edge = EDGE_ZONE / zoom;
        let ezk = 0.5 * nstep;
        let damp = 0.94f32.powf(nstep);
        for i in 0..n {
            let mut vx = self.vx[i] + self.accx[i] * nstep;
            let mut vy = self.vy[i] + self.accy[i] * nstep;

            if !wrap {
                if self.x[i] < left + edge { vx += (1.0 - (self.x[i] - left) / edge) * ezk; }
                if self.x[i] > right - edge { vx -= (1.0 - (right - self.x[i]) / edge) * ezk; }
                if self.y[i] < top + edge { vy += (1.0 - (self.y[i] - top) / edge) * ezk; }
                if self.y[i] > bottom - edge { vy -= (1.0 - (bottom - self.y[i]) / edge) * ezk; }
            }

            let sp = (vx * vx + vy * vy).sqrt().max(1e-6);
            let newsp = speed + (sp - speed) * damp;
            vx = vx / sp * newsp;
            vy = vy / sp * newsp;
            self.vx[i] = vx; self.vy[i] = vy;

            self.phase[i] += (0.3 + sp * 0.05) * nstep;

            let mut nx = self.x[i] + vx * nstep;
            let mut ny = self.y[i] + vy * nstep;
            if wrap {
                nx = left + (((nx - left) % ww) + ww) % ww;
                ny = top + (((ny - top) % wh) + wh) % wh;
            } else {
                if nx < left { nx = left; } else if nx > right { nx = right; }
                if ny < top { ny = top; } else if ny > bottom { ny = bottom; }
            }
            self.x[i] = nx; self.y[i] = ny;
        }
    }
}

/* ---------------- 全局状态 & C ABI ---------------- */
static mut SIM_PTR: *mut Sim = std::ptr::null_mut();
static mut PARAMS: [f32; PARAM_LEN] = [0.0; PARAM_LEN];

#[inline]
fn sim() -> &'static mut Sim {
    unsafe { &mut *(std::ptr::read(std::ptr::addr_of!(SIM_PTR))) }
}
#[inline]
fn params() -> &'static [f32; PARAM_LEN] {
    unsafe { &*std::ptr::addr_of!(PARAMS) }
}

#[no_mangle]
pub extern "C" fn sim_new() {
    unsafe { SIM_PTR = Box::into_raw(Box::new(Sim::new())); }
}

#[no_mangle]
pub extern "C" fn sim_len() -> i32 { sim().x.len() as i32 }

#[no_mangle]
pub extern "C" fn sim_set_count(n: i32, left: f32, top: f32, right: f32, bottom: f32, speed: f32) {
    sim().set_count(n.max(0) as usize, left, top, right, bottom, speed);
}

#[no_mangle]
pub extern "C" fn sim_reset(left: f32, top: f32, right: f32, bottom: f32, speed: f32) {
    sim().reset(left, top, right, bottom, speed);
}

#[no_mangle]
pub extern "C" fn sim_update(dt: f32) { sim().step(dt, params()); }

#[no_mangle]
pub extern "C" fn sim_paint(wx: f32, wy: f32, r: f32, hue: f32) {
    sim().paint(wx, wy, r, hue, params());
}

#[no_mangle]
pub extern "C" fn sim_rescale(cx: f32, cy: f32, ratio: f32) { sim().rescale(cx, cy, ratio); }

#[no_mangle]
pub extern "C" fn sim_params_ptr() -> *mut f32 { unsafe { std::ptr::addr_of_mut!(PARAMS) as *mut f32 } }

#[no_mangle]
pub extern "C" fn sim_x_ptr() -> *const f32 { sim().x.as_ptr() }
#[no_mangle]
pub extern "C" fn sim_y_ptr() -> *const f32 { sim().y.as_ptr() }
#[no_mangle]
pub extern "C" fn sim_vx_ptr() -> *const f32 { sim().vx.as_ptr() }
#[no_mangle]
pub extern "C" fn sim_vy_ptr() -> *const f32 { sim().vy.as_ptr() }
#[no_mangle]
pub extern "C" fn sim_size_ptr() -> *const f32 { sim().size.as_ptr() }
#[no_mangle]
pub extern "C" fn sim_hue_ptr() -> *const f32 { sim().hue.as_ptr() }
#[no_mangle]
pub extern "C" fn sim_phase_ptr() -> *const f32 { sim().phase.as_ptr() }

/* 网格密度统计：最密的前 k 格的平均鱼数 */
static mut STATS: [f32; 8] = [0.0; 8];

#[no_mangle]
pub extern "C" fn sim_compute_stats() {
    let s = sim();
    let ncells = s.last_ncells;
    s.stat_lens.clear();
    for c in 0..ncells {
        let cnt = s.cell_start[c + 1] - s.cell_start[c];
        if cnt > 0 { s.stat_lens.push(cnt); }
    }
    s.stat_lens.sort_unstable_by(|a, b| b.cmp(a));
    let lens = &s.stat_lens;
    let avg_top = |k: usize| -> f32 {
        if lens.is_empty() { return 0.0; }
        let kk = k.min(lens.len());
        let sum: u32 = lens[..kk].iter().sum();
        sum as f32 / kk as f32
    };
    let cells = lens.len();
    let total = s.x.len() as f32;
    unsafe {
        let st = std::ptr::addr_of_mut!(STATS);
        (*st)[0] = avg_top(10);
        (*st)[1] = avg_top(20);
        (*st)[2] = avg_top(50);
        (*st)[3] = cells as f32;
        (*st)[4] = if cells > 0 { total / cells as f32 } else { 0.0 };
    }
}

#[no_mangle]
pub extern "C" fn sim_stats_ptr() -> *const f32 { unsafe { std::ptr::addr_of!(STATS) as *const f32 } }
