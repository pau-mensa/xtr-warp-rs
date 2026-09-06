/// Standalone benchmark comparing f32, scalar-int8, and SIMD-int8 scoring.
///
/// Compile and run:
///   rustc -O benches/score_residual.rs -o /tmp/bench_score && /tmp/bench_score
///
/// No external dependencies -- all kernels are inlined.
///
/// Reference: https://chaochunhsu.github.io/blog/slow-half-of-plaid/

use std::time::Instant;

// ---- f32 scoring (baseline) ----

fn build_reversed_bit_map() -> [u8; 256] {
    let mut reversed = [0u8; 256];
    for byte_val in 0..256u32 {
        let mut reversed_bits = 0u32;
        let mut bit_pos: u8 = 8;
        while bit_pos >= 4 {
            let segment = (byte_val >> (bit_pos - 4)) & 0x0F;
            let mut rev_seg = 0u32;
            for k in 0..4u8 {
                if (segment & (1 << k)) != 0 {
                    rev_seg |= 1 << (3 - k);
                }
            }
            reversed_bits |= rev_seg;
            if bit_pos > 4 {
                reversed_bits <<= 4;
            }
            bit_pos -= 4;
        }
        reversed[byte_val as usize] = (reversed_bits & 0xFF) as u8;
    }
    reversed
}

fn score_f32(residual: &[u8], rmap: &[u8; 256], lut: &[f32]) -> f32 {
    let mut score = 0.0f32;
    for (i, &packed) in residual.iter().enumerate() {
        let packed = rmap[packed as usize];
        let d0 = i << 1;
        let d1 = d0 + 1;
        let hi = (packed >> 4) as usize;
        let lo = (packed & 0x0F) as usize;
        score += lut[(d0 << 4) | hi] + lut[(d1 << 4) | lo];
    }
    score
}

// ---- int8 scoring (scalar) ----

fn build_int8_lut(f32_lut: &[f32], dim: usize) -> (Vec<i8>, f32) {
    let stride = dim * 16;
    let abs_max = f32_lut[..stride].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max > 1e-10 { abs_max / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;

    let mut code_rev = [0u8; 16];
    for n in 0..16u8 {
        let mut r = 0u8;
        for b in 0..4 {
            if n & (1 << b) != 0 {
                r |= 1 << (3 - b);
            }
        }
        code_rev[n as usize] = r;
    }

    let mut lut = vec![0i8; stride];
    for d in 0..dim {
        for raw in 0..16usize {
            let rev = code_rev[raw] as usize;
            let val = f32_lut[d * 16 + rev];
            lut[d * 16 + raw] = (val * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (lut, scale)
}

#[inline]
fn score_i8_scalar(residual: &[u8], lut: &[i8]) -> i32 {
    let mut sum: i32 = 0;
    for (i, &packed) in residual.iter().enumerate() {
        let d0 = i << 1;
        let d1 = d0 + 1;
        sum += lut[(d0 << 4) | (packed >> 4) as usize] as i32;
        sum += lut[(d1 << 4) | (packed & 0x0F) as usize] as i32;
    }
    sum
}

// ---- int8 scoring (SIMD batched) ----

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub fn score_batch(residuals: &[u8], n: usize, bpe: usize, lut: &[i8]) -> Vec<i32> {
        let mut scores = vec![0i32; n];
        let full = n / 16;
        for b in 0..full {
            unsafe { score_16(residuals, b * 16, bpe, lut, &mut scores[b * 16..]) };
        }
        for doc in full * 16..n {
            scores[doc] = super::score_i8_scalar(&residuals[doc * bpe..(doc + 1) * bpe], lut);
        }
        scores
    }

    #[inline]
    unsafe fn score_16(res: &[u8], base: usize, bpe: usize, lut: &[i8], out: &mut [i32]) {
        let mut acc_lo = vdupq_n_s16(0);
        let mut acc_hi = vdupq_n_s16(0);

        for bp in 0..bpe {
            let d0 = bp << 1;
            let d1 = d0 + 1;
            let tbl0 = vld1q_s8(lut.as_ptr().add(d0 << 4));
            let tbl1 = vld1q_s8(lut.as_ptr().add(d1 << 4));

            let mut g = [0u8; 16];
            for doc in 0..16usize {
                *g.get_unchecked_mut(doc) = *res.get_unchecked((base + doc) * bpe + bp);
            }
            let packed = vld1q_u8(g.as_ptr());
            let hi = vshrq_n_u8(packed, 4);
            let lo = vandq_u8(packed, vdupq_n_u8(0x0F));

            let s0 = vqtbl1q_s8(tbl0, hi);
            let s1 = vqtbl1q_s8(tbl1, lo);

            acc_lo = vaddq_s16(acc_lo, vmovl_s8(vget_low_s8(s0)));
            acc_lo = vaddq_s16(acc_lo, vmovl_s8(vget_low_s8(s1)));
            acc_hi = vaddq_s16(acc_hi, vmovl_s8(vget_high_s8(s0)));
            acc_hi = vaddq_s16(acc_hi, vmovl_s8(vget_high_s8(s1)));
        }

        vst1q_s32(out.as_mut_ptr(),         vmovl_s16(vget_low_s16(acc_lo)));
        vst1q_s32(out.as_mut_ptr().add(4),  vmovl_s16(vget_high_s16(acc_lo)));
        vst1q_s32(out.as_mut_ptr().add(8),  vmovl_s16(vget_low_s16(acc_hi)));
        vst1q_s32(out.as_mut_ptr().add(12), vmovl_s16(vget_high_s16(acc_hi)));
    }
}

#[cfg(target_arch = "x86_64")]
mod sse {
    use std::arch::x86_64::*;

    pub fn score_batch(residuals: &[u8], n: usize, bpe: usize, lut: &[i8]) -> Vec<i32> {
        if !is_x86_feature_detected!("ssse3") || !is_x86_feature_detected!("sse4.1") {
            return (0..n)
                .map(|d| super::score_i8_scalar(&residuals[d * bpe..(d + 1) * bpe], lut))
                .collect();
        }
        unsafe { score_batch_inner(residuals, n, bpe, lut) }
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn score_batch_inner(res: &[u8], n: usize, bpe: usize, lut: &[i8]) -> Vec<i32> {
        let mut scores = vec![0i32; n];
        let full = n / 16;
        for b in 0..full {
            score_16(res, b * 16, bpe, lut, &mut scores[b * 16..]);
        }
        for doc in full * 16..n {
            scores[doc] = super::score_i8_scalar(&res[doc * bpe..(doc + 1) * bpe], lut);
        }
        scores
    }

    #[inline]
    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn score_16(res: &[u8], base: usize, bpe: usize, lut: &[i8], out: &mut [i32]) {
        let mut acc_lo = _mm_setzero_si128();
        let mut acc_hi = _mm_setzero_si128();
        let mask = _mm_set1_epi8(0x0F);

        for bp in 0..bpe {
            let d0 = bp << 1;
            let d1 = d0 + 1;
            let tbl0 = _mm_loadu_si128(lut.as_ptr().add(d0 << 4) as *const __m128i);
            let tbl1 = _mm_loadu_si128(lut.as_ptr().add(d1 << 4) as *const __m128i);

            let mut g = [0u8; 16];
            for doc in 0..16usize {
                *g.get_unchecked_mut(doc) = *res.get_unchecked((base + doc) * bpe + bp);
            }
            let packed = _mm_loadu_si128(g.as_ptr() as *const __m128i);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let lo = _mm_and_si128(packed, mask);

            let s0 = _mm_shuffle_epi8(tbl0, hi);
            let s1 = _mm_shuffle_epi8(tbl1, lo);

            acc_lo = _mm_add_epi16(acc_lo, _mm_cvtepi8_epi16(s0));
            acc_lo = _mm_add_epi16(acc_lo, _mm_cvtepi8_epi16(s1));
            acc_hi = _mm_add_epi16(acc_hi, _mm_cvtepi8_epi16(_mm_srli_si128(s0, 8)));
            acc_hi = _mm_add_epi16(acc_hi, _mm_cvtepi8_epi16(_mm_srli_si128(s1, 8)));
        }

        _mm_storeu_si128(out.as_mut_ptr()        as *mut __m128i, _mm_cvtepi16_epi32(acc_lo));
        _mm_storeu_si128(out.as_mut_ptr().add(4) as *mut __m128i, _mm_cvtepi16_epi32(_mm_srli_si128(acc_lo, 8)));
        _mm_storeu_si128(out.as_mut_ptr().add(8) as *mut __m128i, _mm_cvtepi16_epi32(acc_hi));
        _mm_storeu_si128(out.as_mut_ptr().add(12)as *mut __m128i, _mm_cvtepi16_epi32(_mm_srli_si128(acc_hi, 8)));
    }
}

fn simd_score_batch(residuals: &[u8], n: usize, bpe: usize, lut: &[i8]) -> Vec<i32> {
    #[cfg(target_arch = "aarch64")]
    { return neon::score_batch(residuals, n, bpe, lut); }

    #[cfg(target_arch = "x86_64")]
    { return sse::score_batch(residuals, n, bpe, lut); }

    #[allow(unreachable_code)]
    (0..n).map(|d| score_i8_scalar(&residuals[d * bpe..(d + 1) * bpe], lut)).collect()
}

// ---- benchmark harness ----

const WARMUP: usize = 5;
const RUNS: usize = 20;

fn bench<F: Fn() -> f32>(label: &str, n: usize, f: F) -> f64 {
    for _ in 0..WARMUP { std::hint::black_box(f()); }
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        std::hint::black_box(f());
        times.push(t.elapsed().as_nanos() as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[RUNS / 2] / n as f64;
    eprintln!("  {label:>16}: {med:6.1} ns/emb  (median of {RUNS})");
    med
}

fn main() {
    let dim = 128;
    let bpe = dim / 2;     // bytes per embedding for nbits=4
    let n = 8192;           // embeddings per batch
    let stride = dim * 16;

    eprintln!("Int8 LUT scoring benchmark");
    eprintln!("Reference: https://chaochunhsu.github.io/blog/slow-half-of-plaid/");

    eprintln!("\n=== 4-bit, dim={dim}, {n} embeddings, 1 token LUT ===");
    eprintln!("  f32 LUT: {} bytes   i8 LUT: {} bytes", stride * 4, stride);

    let f32_lut: Vec<f32> = (0..stride)
        .map(|i| (i as f32 * 0.7123).sin() * 0.5)
        .collect();
    let rmap = build_reversed_bit_map();

    let (i8_lut, scale) = build_int8_lut(&f32_lut, dim);

    let residuals: Vec<u8> = (0..n * bpe)
        .map(|i| ((i * 37 + 13) & 0xFF) as u8)
        .collect();

    // ---- correctness check: scalar int8 vs SIMD int8 ----
    let scalar_scores: Vec<i32> = (0..n)
        .map(|idx| score_i8_scalar(&residuals[idx * bpe..(idx + 1) * bpe], &i8_lut))
        .collect();
    let simd_scores = simd_score_batch(&residuals, n, bpe, &i8_lut);
    let mut max_diff = 0i32;
    for idx in 0..n {
        let diff = (scalar_scores[idx] - simd_scores[idx]).abs();
        max_diff = max_diff.max(diff);
    }
    if max_diff != 0 {
        eprintln!("  CORRECTNESS FAIL: max |scalar - simd| = {max_diff}");
        std::process::exit(1);
    }
    eprintln!("  correctness: PASS (scalar == SIMD for all {n} embeddings)");

    // ---- benchmarks: each scores n docs against 1 token's LUT ----

    let f32_ns = bench("f32 LUT", n, || {
        let mut total = 0.0f32;
        for idx in 0..n {
            total += score_f32(&residuals[idx * bpe..(idx + 1) * bpe], &rmap, &f32_lut);
        }
        total
    });

    let scalar_ns = bench("int8 scalar", n, || {
        let mut total = 0.0f32;
        for idx in 0..n {
            let raw = score_i8_scalar(&residuals[idx * bpe..(idx + 1) * bpe], &i8_lut);
            total += raw as f32 * scale;
        }
        total
    });

    let simd_ns = bench("int8 SIMD batch", n, || {
        let batch = simd_score_batch(&residuals, n, bpe, &i8_lut);
        let mut total = 0.0f32;
        for idx in 0..n {
            total += batch[idx] as f32 * scale;
        }
        total
    });

    eprintln!();
    eprintln!("  f32 -> scalar int8:  {:.2}x", f32_ns / scalar_ns);
    eprintln!("  f32 -> SIMD int8:    {:.2}x", f32_ns / simd_ns);
    eprintln!("  scalar -> SIMD int8: {:.2}x", scalar_ns / simd_ns);

    // ---- multi-token benchmark: shows cache pressure effect ----
    let num_tokens = 32;
    eprintln!("\n=== Multi-token: {num_tokens} tokens x {n} docs (cache pressure) ===");
    eprintln!("  f32 LUTs total: {}KB   i8 LUTs total: {}KB",
        num_tokens * stride * 4 / 1024, num_tokens * stride / 1024);

    let f32_luts: Vec<f32> = (0..num_tokens * stride)
        .map(|i| (i as f32 * 0.7123).sin() * 0.5)
        .collect();
    let mut i8_luts = vec![0i8; num_tokens * stride];
    let mut all_scales = vec![0.0f32; num_tokens];
    for t in 0..num_tokens {
        let off = t * stride;
        let (lut_t, sc) = build_int8_lut(&f32_luts[off..off + stride], dim);
        i8_luts[off..off + stride].copy_from_slice(&lut_t);
        all_scales[t] = sc;
    }

    let total_scores = n * num_tokens;

    let f32_mt = bench("f32 LUT", total_scores, || {
        let mut total = 0.0f32;
        for t in 0..num_tokens {
            let off = t * stride;
            for idx in 0..n {
                total += score_f32(&residuals[idx * bpe..(idx + 1) * bpe], &rmap, &f32_luts[off..off + stride]);
            }
        }
        total
    });

    let scalar_mt = bench("int8 scalar", total_scores, || {
        let mut total = 0.0f32;
        for t in 0..num_tokens {
            let off = t * stride;
            for idx in 0..n {
                let raw = score_i8_scalar(&residuals[idx * bpe..(idx + 1) * bpe], &i8_luts[off..off + stride]);
                total += raw as f32 * all_scales[t];
            }
        }
        total
    });

    let simd_mt = bench("int8 SIMD batch", total_scores, || {
        let mut total = 0.0f32;
        for t in 0..num_tokens {
            let off = t * stride;
            let batch = simd_score_batch(&residuals, n, bpe, &i8_luts[off..off + stride]);
            for idx in 0..n {
                total += batch[idx] as f32 * all_scales[t];
            }
        }
        total
    });

    eprintln!();
    eprintln!("  f32 -> scalar int8:  {:.2}x", f32_mt / scalar_mt);
    eprintln!("  f32 -> SIMD int8:    {:.2}x", f32_mt / simd_mt);
    eprintln!("  scalar -> SIMD int8: {:.2}x", scalar_mt / simd_mt);
    eprintln!();
}
