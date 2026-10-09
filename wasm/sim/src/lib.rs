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

// 一组 8 个向量累加器（凝聚/对齐/分离 + 计数）
#[cfg(target_feature = "simd128")]
#[derive(Clone, Copy)]
struct V8 { cdx: v128, cdy: v128, avx: v128, avy: v128, sdx: v128, sdy: v128, cnt: v128, scnt: v128 }

// 处理邻格区间 [s,e)：向量主体累加进 v，标量尾巴累加进 tail（hsum 由调用方每鱼只做一次）
#[cfg(target_feature = "simd128")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn proc_range(
    s: usize, e: usize,
    xp: *const f32, yp: *const f32, vxp: *const f32, vyp: *const f32,
    fxv: v128, fyv: v128, hwv: v128, wwv: v128, hhv: v128, whv: v128,
    nhwv: v128, nhhv: v128, zv: v128, p2v: v128, sepv: v128, one: v128, wrap: bool,
    v: &mut V8, tail: &mut Acc,
    fx: f32, fy: f32, hw: f32, ww: f32, hh: f32, wh: f32, perception2: f32, sep2: f32,
) {
    let mut cdxv = v.cdx; let mut cdyv = v.cdy;
    let mut avxv = v.avx; let mut avyv = v.avy;
    let mut sdxv = v.sdx; let mut sdyv = v.sdy;
    let mut cntv = v.cnt; let mut scntv = v.scnt;

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
    v.cdx = cdxv; v.cdy = cdyv; v.avx = avxv; v.avy = avyv;
    v.sdx = sdxv; v.sdy = sdyv; v.cnt = cntv; v.scnt = scntv;

    while k < e {
        let mut dx = *xp.add(k) - fx;
        let mut dy = *yp.add(k) - fy;
        if wrap {
            if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
            if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
        }
        let d2 = dx * dx + dy * dy;
        if d2 > 0.0 && d2 < perception2 {
            tail.cdx += dx; tail.cdy += dy;
            tail.avx += *vxp.add(k); tail.avy += *vyp.add(k);
            tail.cnt += 1;
            if d2 < sep2 { tail.sdx -= dx; tail.sdy -= dy; tail.scnt += 1; }
        }
        k += 1;
    }
}

// 成对法：把一对 (a,b) 的贡献同时累加写到双方的标量累加数组（原始指针版，避免借用冲突）
#[cfg(target_feature = "simd128")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_add_raw(
    a: usize, b: usize,
    xp: *const f32, yp: *const f32, vxp: *const f32, vyp: *const f32,
    cdx: *mut f32, cdy: *mut f32, avx: *mut f32, avy: *mut f32,
    sdx: *mut f32, sdy: *mut f32, cnt: *mut i32, scnt: *mut i32,
    wrap: bool, hw: f32, ww: f32, hh: f32, wh: f32, p2: f32, sep2: f32,
) {
    let xa = *xp.add(a); let ya = *yp.add(a); let vxa = *vxp.add(a); let vya = *vyp.add(a);
    let mut dx = *xp.add(b) - xa;
    let mut dy = *yp.add(b) - ya;
    if wrap {
        if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
        if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
    }
    let d2 = dx * dx + dy * dy;
    if d2 > 0.0 && d2 < p2 {
        *cdx.add(a) += dx; *cdy.add(a) += dy;
        *cdx.add(b) -= dx; *cdy.add(b) -= dy;
        *avx.add(a) += *vxp.add(b); *avy.add(a) += *vyp.add(b);
        *avx.add(b) += vxa; *avy.add(b) += vya;
        *cnt.add(a) += 1; *cnt.add(b) += 1;
        if d2 < sep2 {
            *sdx.add(a) -= dx; *sdy.add(a) -= dy;
            *sdx.add(b) += dx; *sdy.add(b) += dy;
            *scnt.add(a) += 1; *scnt.add(b) += 1;
        }
    }
}

// 分块成对：A 块（4 条，向量）逐条对 B 区间的每条鱼；A 侧向量累加，B 侧 hsum 写标量数组
#[cfg(target_feature = "simd128")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_cross(
    a: usize, sb: usize, eb: usize,
    xp: *const f32, yp: *const f32, vxp: *const f32, vyp: *const f32,
    hwv: v128, wwv: v128, hhv: v128, whv: v128, nhwv: v128, nhhv: v128,
    zv: v128, p2v: v128, sepv: v128, one: v128, wrap: bool,
    av: &mut V8,
    cdx: *mut f32, cdy: *mut f32, avx: *mut f32, avy: *mut f32,
    sdx: *mut f32, sdy: *mut f32, cnt: *mut i32, scnt: *mut i32,
) {
    let xav = v128_load(xp.add(a) as *const v128);
    let yav = v128_load(yp.add(a) as *const v128);
    let vxav = v128_load(vxp.add(a) as *const v128);
    let vyav = v128_load(vyp.add(a) as *const v128);
    let mut cdxv = av.cdx; let mut cdyv = av.cdy;
    let mut avxv = av.avx; let mut avyv = av.avy;
    let mut sdxv = av.sdx; let mut sdyv = av.sdy;
    let mut cntv = av.cnt; let mut scntv = av.scnt;
    for m in sb..eb {
        let mut dxv = f32x4_sub(f32x4_splat(*xp.add(m)), xav);
        let mut dyv = f32x4_sub(f32x4_splat(*yp.add(m)), yav);
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
        let m1 = v128_and(mgt, f32x4_lt(d2, p2v));
        let dmx = v128_bitselect(dxv, zv, m1);
        let dmy = v128_bitselect(dyv, zv, m1);
        // A 侧（向量累加，4 条一起）
        cdxv = f32x4_add(cdxv, dmx);
        cdyv = f32x4_add(cdyv, dmy);
        avxv = f32x4_add(avxv, v128_bitselect(f32x4_splat(*vxp.add(m)), zv, m1));
        avyv = f32x4_add(avyv, v128_bitselect(f32x4_splat(*vyp.add(m)), zv, m1));
        cntv = i32x4_add(cntv, v128_and(m1, one));
        let m2 = v128_and(mgt, f32x4_lt(d2, sepv));
        let smx = v128_bitselect(dxv, zv, m2);
        let smy = v128_bitselect(dyv, zv, m2);
        sdxv = f32x4_sub(sdxv, smx);
        sdyv = f32x4_sub(sdyv, smy);
        scntv = i32x4_add(scntv, v128_and(m2, one));
        // B 侧（对 A 的 4 条做水平求和，写标量数组）
        *cdx.add(m) -= hsum_f32(dmx);
        *cdy.add(m) -= hsum_f32(dmy);
        *avx.add(m) += hsum_f32(v128_bitselect(vxav, zv, m1));
        *avy.add(m) += hsum_f32(v128_bitselect(vyav, zv, m1));
        *sdx.add(m) += hsum_f32(smx);
        *sdy.add(m) += hsum_f32(smy);
        *cnt.add(m) += hsum_i32(v128_and(m1, one));
        *scnt.add(m) += hsum_i32(v128_and(m2, one));
    }
    av.cdx = cdxv; av.cdy = cdyv; av.avx = avxv; av.avy = avyv;
    av.sdx = sdxv; av.sdy = sdyv; av.cnt = cntv; av.scnt = scntv;
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
    last_ncells: usize, last_cols: i32, last_rows: i32, last_cols_u: usize,
    // 成对法的每鱼累加数组
    cdx: Vec<f32>, cdy: Vec<f32>, avx: Vec<f32>, avy: Vec<f32>,
    sdx: Vec<f32>, sdy: Vec<f32>, cnt: Vec<i32>, scnt: Vec<i32>,
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
            last_ncells: 0, last_cols: 0, last_rows: 0, last_cols_u: 0,
            cdx: Vec::new(), cdy: Vec::new(), avx: Vec::new(), avy: Vec::new(),
            sdx: Vec::new(), sdy: Vec::new(), cnt: Vec::new(), scnt: Vec::new(),
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

    // 阶段一：建网格（计格号 + 计数排序 + 按格重排）
    fn build(&mut self, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let ww = right - left; let wh = bottom - top;

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
        self.last_cols = cols; self.last_rows = rows; self.last_cols_u = cols_u;

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
    }

    // 阶段二：逐鱼求力（Jacobi：先全部算力，再统一积分）→ accx/accy
    // 单条鱼的邻居累加：每条鱼只 splat 一次、累加进同一组向量、最后只 hsum 一次
    #[cfg(target_feature = "simd128")]
    fn accum_fish(&self, i: usize, cols: i32, rows: i32, cols_u: usize, wrap: bool,
                  hw: f32, ww: f32, hh: f32, wh: f32, perception2: f32, sep2: f32) -> Acc {
        let fx = self.x[i]; let fy = self.y[i];
        let gx = self.gx[i]; let gy = self.gy[i];
        let xc0 = if wrap { if gx == 0 { cols - 1 } else { gx - 1 } } else { gx - 1 };
        let xc2 = if wrap { if gx == cols - 1 { 0 } else { gx + 1 } } else { gx + 1 };
        let yc0 = if wrap { if gy == 0 { rows - 1 } else { gy - 1 } } else { gy - 1 };
        let yc2 = if wrap { if gy == rows - 1 { 0 } else { gy + 1 } } else { gy + 1 };
        let xcs = [xc0, gx, xc2];
        let ycs = [yc0, gy, yc2];
        unsafe {
            let fxv = f32x4_splat(fx); let fyv = f32x4_splat(fy);
            let hwv = f32x4_splat(hw); let wwv = f32x4_splat(ww);
            let hhv = f32x4_splat(hh); let whv = f32x4_splat(wh);
            let nhwv = f32x4_splat(-hw); let nhhv = f32x4_splat(-hh);
            let zv = f32x4_splat(0.0);
            let p2v = f32x4_splat(perception2); let sepv = f32x4_splat(sep2);
            let one = i32x4_splat(1);
            let mut v = V8 { cdx: zv, cdy: zv, avx: zv, avy: zv, sdx: zv, sdy: zv, cnt: i32x4_splat(0), scnt: i32x4_splat(0) };
            let mut tail = Acc::default();
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
                    proc_range(s, e, xp, yp, vxp, vyp, fxv, fyv, hwv, wwv, hhv, whv, nhwv, nhhv, zv, p2v, sepv, one, wrap, &mut v, &mut tail, fx, fy, hw, ww, hh, wh, perception2, sep2);
                }
            }
            Acc {
                cdx: hsum_f32(v.cdx) + tail.cdx,
                cdy: hsum_f32(v.cdy) + tail.cdy,
                avx: hsum_f32(v.avx) + tail.avx,
                avy: hsum_f32(v.avy) + tail.avy,
                sdx: hsum_f32(v.sdx) + tail.sdx,
                sdy: hsum_f32(v.sdy) + tail.sdy,
                cnt: hsum_i32(v.cnt) + tail.cnt,
                scnt: hsum_i32(v.scnt) + tail.scnt,
            }
        }
    }

    #[cfg(not(target_feature = "simd128"))]
    fn accum_fish(&self, i: usize, cols: i32, rows: i32, cols_u: usize, wrap: bool,
                  hw: f32, ww: f32, hh: f32, wh: f32, perception2: f32, sep2: f32) -> Acc {
        let fx = self.x[i]; let fy = self.y[i];
        let gx = self.gx[i]; let gy = self.gy[i];
        let xc0 = if wrap { if gx == 0 { cols - 1 } else { gx - 1 } } else { gx - 1 };
        let xc2 = if wrap { if gx == cols - 1 { 0 } else { gx + 1 } } else { gx + 1 };
        let yc0 = if wrap { if gy == 0 { rows - 1 } else { gy - 1 } } else { gy - 1 };
        let yc2 = if wrap { if gy == rows - 1 { 0 } else { gy + 1 } } else { gy + 1 };
        let xcs = [xc0, gx, xc2];
        let ycs = [yc0, gy, yc2];
        let mut acc = Acc::default();
        for &cx in xcs.iter() {
            if cx < 0 || cx >= cols { continue; }
            let cxb = cx as usize;
            for &cy in ycs.iter() {
                if cy < 0 || cy >= rows { continue; }
                let c = cy as usize * cols_u + cxb;
                let s = self.cell_start[c] as usize;
                let e = self.cell_start[c + 1] as usize;
                for k in s..e {
                    if k == i { continue; }
                    let mut dx = self.x[k] - fx;
                    let mut dy = self.y[k] - fy;
                    if wrap {
                        if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
                        if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
                    }
                    let d2 = dx * dx + dy * dy;
                    if d2 > 0.0 && d2 < perception2 {
                        acc.cdx += dx; acc.cdy += dy;
                        acc.avx += self.vx[k]; acc.avy += self.vy[k];
                        acc.cnt += 1;
                        if d2 < sep2 { acc.sdx -= dx; acc.sdy -= dy; acc.scnt += 1; }
                    }
                }
            }
        }
        acc
    }

    fn forces(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let nstep = dt * REF_TICKS;
        let cohesion = p[0]; let separation = p[1]; let align = p[2];
        let mouse_field = p[4] > 0.5;
        let mx = p[5]; let my = p[6]; let mradius = p[7]; let mattract = p[8]; let mrepel = p[9];
        let left_down = p[10] > 0.5; let scatter_f = p[11];
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let wander_scale = p[18];
        let ww = right - left; let wh = bottom - top;
        let hw = ww * 0.5; let hh = wh * 0.5;
        let perception2 = PERCEPTION * PERCEPTION;
        let sep2 = SEP_RADIUS * SEP_RADIUS;
        let cols = self.last_cols; let rows = self.last_rows; let cols_u = self.last_cols_u;

        for i in 0..n {
            let fx = self.x[i]; let fy = self.y[i];
            // 内部格永远不会跨边界：跳过 min-image 开销，只有边界格才可能 wrap
            let gxf = self.gx[i]; let gyf = self.gy[i];
            let wrap_i = wrap && (gxf == 0 || gxf == cols - 1 || gyf == 0 || gyf == rows - 1);
            let acc = self.accum_fish(i, cols, rows, cols_u, wrap_i, hw, ww, hh, wh, perception2, sep2);

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
    }

    // 处理一对 (a,b)：只算一次，双方都累加
    #[inline]
    fn pair_add(&mut self, a: usize, b: usize, wrap: bool, hw: f32, ww: f32, hh: f32, wh: f32, p2: f32, sep2: f32) {
        let xa = self.x[a]; let ya = self.y[a]; let vxa = self.vx[a]; let vya = self.vy[a];
        let mut dx = self.x[b] - xa;
        let mut dy = self.y[b] - ya;
        if wrap {
            if dx > hw { dx -= ww; } else if dx < -hw { dx += ww; }
            if dy > hh { dy -= wh; } else if dy < -hh { dy += wh; }
        }
        let d2 = dx * dx + dy * dy;
        if d2 > 0.0 && d2 < p2 {
            self.cdx[a] += dx; self.cdy[a] += dy;
            self.cdx[b] -= dx; self.cdy[b] -= dy;
            self.avx[a] += self.vx[b]; self.avy[a] += self.vy[b];
            self.avx[b] += vxa; self.avy[b] += vya;
            self.cnt[a] += 1; self.cnt[b] += 1;
            if d2 < sep2 {
                self.sdx[a] -= dx; self.sdy[a] -= dy;
                self.sdx[b] += dx; self.sdy[b] += dy;
                self.scnt[a] += 1; self.scnt[b] += 1;
            }
        }
    }

    // 成对法（标量实验版）：每个无序对只算一次，双方都累加，再统一后处理
    fn forces_pair(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let ww = right - left; let wh = bottom - top;
        let hw = ww * 0.5; let hh = wh * 0.5;
        let perception2 = PERCEPTION * PERCEPTION;
        let sep2 = SEP_RADIUS * SEP_RADIUS;
        let cols = self.last_cols; let rows = self.last_rows; let cols_u = self.last_cols_u;

        self.cdx.resize(n, 0.0); self.cdy.resize(n, 0.0);
        self.avx.resize(n, 0.0); self.avy.resize(n, 0.0);
        self.sdx.resize(n, 0.0); self.sdy.resize(n, 0.0);
        self.cnt.resize(n, 0); self.scnt.resize(n, 0);
        self.cdx[..n].fill(0.0); self.cdy[..n].fill(0.0);
        self.avx[..n].fill(0.0); self.avy[..n].fill(0.0);
        self.sdx[..n].fill(0.0); self.sdy[..n].fill(0.0);
        self.cnt[..n].fill(0); self.scnt[..n].fill(0);

        let half: [(i32, i32); 4] = [(0, 1), (1, -1), (1, 0), (1, 1)];
        for gy in 0..rows {
            for gx in 0..cols {
                let c = (gy as usize) * cols_u + gx as usize;
                let s = self.cell_start[c] as usize;
                let e = self.cell_start[c + 1] as usize;
                // 同格内 a<b
                let mut a = s;
                while a < e {
                    let mut b = a + 1;
                    while b < e { self.pair_add(a, b, wrap, hw, ww, hh, wh, perception2, sep2); b += 1; }
                    a += 1;
                }
                // 半邻域的 4 个方向（每个格对只访问一次）
                for &(ox, oy) in half.iter() {
                    let nx = gx + ox; let ny = gy + oy;
                    let (cx, cy) = if wrap {
                        (nx.rem_euclid(cols), ny.rem_euclid(rows))
                    } else if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
                        continue;
                    } else {
                        (nx, ny)
                    };
                    let c2 = (cy as usize) * cols_u + cx as usize;
                    let s2 = self.cell_start[c2] as usize;
                    let e2 = self.cell_start[c2 + 1] as usize;
                    let mut ii = s;
                    while ii < e {
                        let mut jj = s2;
                        while jj < e2 { self.pair_add(ii, jj, wrap, hw, ww, hh, wh, perception2, sep2); jj += 1; }
                        ii += 1;
                    }
                }
            }
        }

        self.post_process(dt, p);
    }

    // 后处理：由每鱼累加数组算出 accx/accy（含 wander/鼠标/驱散）
    fn post_process(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let nstep = dt * REF_TICKS;
        let cohesion = p[0]; let separation = p[1]; let align = p[2];
        let mouse_field = p[4] > 0.5;
        let mx = p[5]; let my = p[6]; let mradius = p[7]; let mattract = p[8]; let mrepel = p[9];
        let left_down = p[10] > 0.5; let scatter_f = p[11];
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let wander_scale = p[18];
        let ww = right - left; let wh = bottom - top;
        let hw = ww * 0.5; let hh = wh * 0.5;
        for i in 0..n {
            let fx = self.x[i]; let fy = self.y[i];
            let cdx = self.cdx[i]; let cdy = self.cdy[i];
            let avx = self.avx[i]; let avy = self.avy[i];
            let sdx = self.sdx[i]; let sdy = self.sdy[i];
            let cnt = self.cnt[i]; let scnt = self.scnt[i];
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
    }

    // 成对法（SIMD 分块实验版）：交叉格用 4×4 分块向量化，同格内用标量三角
    #[cfg(target_feature = "simd128")]
    fn forces_pair_simd(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let ww = right - left; let wh = bottom - top;
        let hw = ww * 0.5; let hh = wh * 0.5;
        let perception2 = PERCEPTION * PERCEPTION;
        let sep2 = SEP_RADIUS * SEP_RADIUS;
        let cols = self.last_cols; let rows = self.last_rows; let cols_u = self.last_cols_u;

        self.cdx.resize(n, 0.0); self.cdy.resize(n, 0.0);
        self.avx.resize(n, 0.0); self.avy.resize(n, 0.0);
        self.sdx.resize(n, 0.0); self.sdy.resize(n, 0.0);
        self.cnt.resize(n, 0); self.scnt.resize(n, 0);
        self.cdx[..n].fill(0.0); self.cdy[..n].fill(0.0);
        self.avx[..n].fill(0.0); self.avy[..n].fill(0.0);
        self.sdx[..n].fill(0.0); self.sdy[..n].fill(0.0);
        self.cnt[..n].fill(0); self.scnt[..n].fill(0);

        let half: [(i32, i32); 4] = [(0, 1), (1, -1), (1, 0), (1, 1)];
        unsafe {
            let hwv = f32x4_splat(hw); let wwv = f32x4_splat(ww);
            let hhv = f32x4_splat(hh); let whv = f32x4_splat(wh);
            let nhwv = f32x4_splat(-hw); let nhhv = f32x4_splat(-hh);
            let zv = f32x4_splat(0.0);
            let p2v = f32x4_splat(perception2); let sepv = f32x4_splat(sep2);
            let one = i32x4_splat(1);

            let xp = self.x.as_ptr(); let yp = self.y.as_ptr();
            let vxp = self.vx.as_ptr(); let vyp = self.vy.as_ptr();
            let cdx = self.cdx.as_mut_ptr(); let cdy = self.cdy.as_mut_ptr();
            let avx = self.avx.as_mut_ptr(); let avy = self.avy.as_mut_ptr();
            let sdx = self.sdx.as_mut_ptr(); let sdy = self.sdy.as_mut_ptr();
            let cnt = self.cnt.as_mut_ptr(); let scnt = self.scnt.as_mut_ptr();

            for gy in 0..rows {
                for gx in 0..cols {
                    let c = (gy as usize) * cols_u + gx as usize;
                    let s = self.cell_start[c] as usize;
                    let e = self.cell_start[c + 1] as usize;
                    // 同格内（标量三角 a<b）
                    let mut i = s;
                    while i < e {
                        let mut j = i + 1;
                        while j < e {
                            pair_add_raw(i, j, xp, yp, vxp, vyp, cdx, cdy, avx, avy, sdx, sdy, cnt, scnt, wrap, hw, ww, hh, wh, perception2, sep2);
                            j += 1;
                        }
                        i += 1;
                    }
                    // 预先解析 4 个半邻单格的区间
                    let mut nbrs = [(0usize, 0usize); 4]; let mut nn = 0usize;
                    for &(ox, oy) in half.iter() {
                        let nx = gx + ox; let ny = gy + oy;
                        let (cx, cy) = if wrap {
                            (nx.rem_euclid(cols), ny.rem_euclid(rows))
                        } else if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
                            continue;
                        } else { (nx, ny) };
                        let c2 = (cy as usize) * cols_u + cx as usize;
                        nbrs[nn] = (self.cell_start[c2] as usize, self.cell_start[c2 + 1] as usize);
                        nn += 1;
                    }
                    // 交叉格：A 整块(4) 向量累加
                    let mut a = s;
                    while a + 4 <= e {
                        let mut av = V8 {
                            cdx: v128_load(cdx.add(a) as *const v128),
                            cdy: v128_load(cdy.add(a) as *const v128),
                            avx: v128_load(avx.add(a) as *const v128),
                            avy: v128_load(avy.add(a) as *const v128),
                            sdx: v128_load(sdx.add(a) as *const v128),
                            sdy: v128_load(sdy.add(a) as *const v128),
                            cnt: v128_load(cnt.add(a) as *const v128),
                            scnt: v128_load(scnt.add(a) as *const v128),
                        };
                        for k in 0..nn {
                            let (s2, e2) = nbrs[k];
                            if s2 == e2 { continue; }
                            tile_cross(a, s2, e2, xp, yp, vxp, vyp, hwv, wwv, hhv, whv, nhwv, nhhv, zv, p2v, sepv, one, wrap, &mut av, cdx, cdy, avx, avy, sdx, sdy, cnt, scnt);
                        }
                        v128_store(cdx.add(a) as *mut v128, av.cdx);
                        v128_store(cdy.add(a) as *mut v128, av.cdy);
                        v128_store(avx.add(a) as *mut v128, av.avx);
                        v128_store(avy.add(a) as *mut v128, av.avy);
                        v128_store(sdx.add(a) as *mut v128, av.sdx);
                        v128_store(sdy.add(a) as *mut v128, av.sdy);
                        v128_store(cnt.add(a) as *mut v128, av.cnt);
                        v128_store(scnt.add(a) as *mut v128, av.scnt);
                        a += 4;
                    }
                    // A 尾巴
                    while a < e {
                        for k in 0..nn {
                            let (s2, e2) = nbrs[k];
                            let mut m = s2;
                            while m < e2 {
                                pair_add_raw(a, m, xp, yp, vxp, vyp, cdx, cdy, avx, avy, sdx, sdy, cnt, scnt, wrap, hw, ww, hh, wh, perception2, sep2);
                                m += 1;
                            }
                        }
                        a += 1;
                    }
                }
            }
        }
        self.post_process(dt, p);
    }

    #[cfg(not(target_feature = "simd128"))]
    fn forces_pair_simd(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        self.forces_pair(dt, p);
    }

    // 阶段三：积分 + 边界
    fn integrate(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        let n = self.x.len();
        if n == 0 { return; }
        let nstep = dt * REF_TICKS;
        let speed = p[3];
        let wrap = p[12] > 0.5;
        let left = p[13]; let top = p[14]; let right = p[15]; let bottom = p[16];
        let zoom = p[17];
        let ww = right - left; let wh = bottom - top;

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

    fn step(&mut self, dt: f32, p: &[f32; PARAM_LEN]) {
        self.build(p);
        self.forces(dt, p);
        self.integrate(dt, p);
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

// 分阶段导出（用于性能剖析）
#[no_mangle]
pub extern "C" fn sim_phase_build() { sim().build(params()); }
#[no_mangle]
pub extern "C" fn sim_phase_forces(dt: f32) { sim().forces(dt, params()); }
#[no_mangle]
pub extern "C" fn sim_phase_forces_pair(dt: f32) { sim().forces_pair(dt, params()); }
#[no_mangle]
pub extern "C" fn sim_phase_forces_pair_simd(dt: f32) { sim().forces_pair_simd(dt, params()); }
#[no_mangle]
pub extern "C" fn sim_phase_integrate(dt: f32) { sim().integrate(dt, params()); }

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
