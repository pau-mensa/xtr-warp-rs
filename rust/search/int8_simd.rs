/// SIMD-accelerated int8 LUT scoring via batched table lookup.
///
/// Instead of scoring one document at a time (scalar lookup per dimension),
/// this module scores 16 documents simultaneously per dimension pair:
///
///   1. Gather one packed byte from each of 16 documents
///   2. Split into hi/lo nibbles with SIMD AND/shift
///   3. `tbl`/`pshufb`: look up all 16 scores from the dimension's
///      16-entry LUT in one instruction
///   4. Widen to i16 and accumulate
///
/// This replaces 16 x 2 = 32 scalar lookups with 2 SIMD lookups per
/// dimension pair per batch.
///
/// Reference: <https://chaochunhsu.github.io/blog/slow-half-of-plaid/>

const BATCH: usize = 16;

/// Largest `dim` the i16 accumulators can hold without overflow.
///
/// Each lane sums `dim` looked-up values, each in `[-127, 127]`, so the
/// worst case is `dim * 127`. That must fit i16: `32767 / 127 = 258`.
/// Above this the kernel would wrap silently (and inconsistently, since
/// the scalar tail accumulates in i32), so we fall back to scalar.
const MAX_SIMD_DIM: usize = 258;

/// Score `capacity` residuals against a single query token's int8 LUT.
///
/// Returns one i32 partial score per document (multiply by the token's
/// scale factor to recover the f32 residual score).
///
/// The function dispatches to the best SIMD kernel available at compile
/// time (aarch64 NEON, x86-64 SSSE3) and falls back to scalar otherwise.
pub fn score_batch_4bit(
    residuals: &[u8],
    capacity: usize,
    bytes_per_emb: usize,
    lut: &[i8],
) -> Vec<i32> {
    // Memory-safety backstop for the `get_unchecked` reads below. These must
    // be real checks, not `debug_assert!`, which compiles out under
    // `[profile.release]` -- the builds that ship.
    //
    // This is defense in depth, NOT the primary validation: a residual row
    // width that disagrees with the index dim is rejected with an error by
    // `decompress_centroids_for_shard` before any scoring happens. Returning
    // zeros here would rank documents by centroid score alone, which looks
    // plausible and is wrong, so callers must not rely on this path to
    // report bad data. It exists only so that a buffer that is somehow still
    // short (e.g. `process_cell_impl`'s `.unwrap_or_default()` yielding an
    // empty Vec) cannot read out of bounds.
    if residuals.len() < capacity * bytes_per_emb || lut.len() < bytes_per_emb * 2 * 16 {
        return vec![0i32; capacity];
    }

    // i16 accumulators cannot represent dim > MAX_SIMD_DIM (see const).
    if bytes_per_emb * 2 > MAX_SIMD_DIM {
        return scalar_fallback(residuals, capacity, bytes_per_emb, lut);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64.
        return unsafe { neon::score_batch(residuals, capacity, bytes_per_emb, lut) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1") {
            return unsafe { x86::score_batch(residuals, capacity, bytes_per_emb, lut) };
        }
    }

    #[allow(unreachable_code)]
    scalar_fallback(residuals, capacity, bytes_per_emb, lut)
}

// ---- scalar fallback (any platform) ----

fn score_one(residual: &[u8], lut: &[i8]) -> i32 {
    let mut sum: i32 = 0;
    for (i, &packed) in residual.iter().enumerate() {
        let d0 = i << 1;
        let d1 = d0 + 1;
        sum += lut[(d0 << 4) | (packed >> 4) as usize] as i32;
        sum += lut[(d1 << 4) | (packed & 0x0F) as usize] as i32;
    }
    sum
}

fn scalar_fallback(
    residuals: &[u8],
    capacity: usize,
    bytes_per_emb: usize,
    lut: &[i8],
) -> Vec<i32> {
    (0..capacity)
        .map(|doc| {
            let off = doc * bytes_per_emb;
            score_one(&residuals[off..off + bytes_per_emb], lut)
        })
        .collect()
}

// ---- aarch64 NEON ----

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Score all residuals using NEON `tbl` for batched lookup.
    ///
    /// # Safety
    /// Requires aarch64 NEON (baseline on the architecture).
    pub unsafe fn score_batch(
        residuals: &[u8],
        capacity: usize,
        bpe: usize,
        lut: &[i8],
    ) -> Vec<i32> {
        let mut scores = vec![0i32; capacity];
        let full = capacity / super::BATCH;

        for b in 0..full {
            score_16(residuals, b * super::BATCH, bpe, lut, &mut scores[b * super::BATCH..]);
        }

        let tail = full * super::BATCH;
        for doc in tail..capacity {
            scores[doc] = super::score_one(&residuals[doc * bpe..(doc + 1) * bpe], lut);
        }
        scores
    }

    /// Score exactly 16 documents starting at `base`.
    #[inline]
    unsafe fn score_16(
        residuals: &[u8],
        base: usize,
        bpe: usize,
        lut: &[i8],
        out: &mut [i32],
    ) {
        // Two i16x8 accumulators cover 16 documents.
        // Max per lane: 128 dims x 127 = 16_256, fits i16.
        let mut acc_lo = vdupq_n_s16(0);
        let mut acc_hi = vdupq_n_s16(0);

        for bp in 0..bpe {
            let d0 = bp << 1;
            let d1 = d0 + 1;

            // Load the two 16-entry LUTs for this dimension pair.
            let tbl0 = vld1q_s8(lut.as_ptr().add(d0 << 4));
            let tbl1 = vld1q_s8(lut.as_ptr().add(d1 << 4));

            // Gather one packed byte from each of the 16 documents.
            let mut g = [0u8; 16];
            for doc in 0..16usize {
                *g.get_unchecked_mut(doc) =
                    *residuals.get_unchecked((base + doc) * bpe + bp);
            }
            let packed = vld1q_u8(g.as_ptr());

            // Split nibbles.
            let hi = vshrq_n_u8(packed, 4);
            let lo = vandq_u8(packed, vdupq_n_u8(0x0F));

            // 16-wide table lookup: one score per document per dimension.
            let s0 = vqtbl1q_s8(tbl0, hi);
            let s1 = vqtbl1q_s8(tbl1, lo);

            // Widen i8 -> i16 and accumulate.
            acc_lo = vaddq_s16(acc_lo, vmovl_s8(vget_low_s8(s0)));
            acc_lo = vaddq_s16(acc_lo, vmovl_s8(vget_low_s8(s1)));
            acc_hi = vaddq_s16(acc_hi, vmovl_s8(vget_high_s8(s0)));
            acc_hi = vaddq_s16(acc_hi, vmovl_s8(vget_high_s8(s1)));
        }

        // Widen i16 -> i32 and store.
        vst1q_s32(out.as_mut_ptr(),        vmovl_s16(vget_low_s16(acc_lo)));
        vst1q_s32(out.as_mut_ptr().add(4), vmovl_s16(vget_high_s16(acc_lo)));
        vst1q_s32(out.as_mut_ptr().add(8), vmovl_s16(vget_low_s16(acc_hi)));
        vst1q_s32(out.as_mut_ptr().add(12),vmovl_s16(vget_high_s16(acc_hi)));
    }
}

// ---- x86-64 SSSE3 + SSE4.1 ----

#[cfg(target_arch = "x86_64")]
mod x86 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Score all residuals using `pshufb` for batched lookup.
    ///
    /// # Safety
    /// Requires SSSE3 (pshufb) and SSE4.1 (pmovsxbw, pmovsxwd).
    #[target_feature(enable = "ssse3,sse4.1")]
    pub unsafe fn score_batch(
        residuals: &[u8],
        capacity: usize,
        bpe: usize,
        lut: &[i8],
    ) -> Vec<i32> {
        let mut scores = vec![0i32; capacity];
        let full = capacity / super::BATCH;

        for b in 0..full {
            score_16(residuals, b * super::BATCH, bpe, lut, &mut scores[b * super::BATCH..]);
        }

        let tail = full * super::BATCH;
        for doc in tail..capacity {
            scores[doc] = super::score_one(&residuals[doc * bpe..(doc + 1) * bpe], lut);
        }
        scores
    }

    #[inline]
    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn score_16(
        residuals: &[u8],
        base: usize,
        bpe: usize,
        lut: &[i8],
        out: &mut [i32],
    ) {
        let mut acc_lo = _mm_setzero_si128(); // docs 0-7, i16
        let mut acc_hi = _mm_setzero_si128(); // docs 8-15, i16
        let mask_lo = _mm_set1_epi8(0x0F);

        for bp in 0..bpe {
            let d0 = bp << 1;
            let d1 = d0 + 1;

            let tbl0 = _mm_loadu_si128(lut.as_ptr().add(d0 << 4) as *const __m128i);
            let tbl1 = _mm_loadu_si128(lut.as_ptr().add(d1 << 4) as *const __m128i);

            let mut g = [0u8; 16];
            for doc in 0..16usize {
                *g.get_unchecked_mut(doc) =
                    *residuals.get_unchecked((base + doc) * bpe + bp);
            }
            let packed = _mm_loadu_si128(g.as_ptr() as *const __m128i);

            // pshufb: byte shuffle = 16-entry table lookup.
            // High bit of index zeroes the lane, but nibbles are [0,15].
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_lo);
            let lo = _mm_and_si128(packed, mask_lo);

            let s0 = _mm_shuffle_epi8(tbl0, hi);
            let s1 = _mm_shuffle_epi8(tbl1, lo);

            // Widen i8 -> i16 (SSE4.1 pmovsxbw).
            let s0_lo = _mm_cvtepi8_epi16(s0);
            let s0_hi = _mm_cvtepi8_epi16(_mm_srli_si128(s0, 8));
            let s1_lo = _mm_cvtepi8_epi16(s1);
            let s1_hi = _mm_cvtepi8_epi16(_mm_srli_si128(s1, 8));

            acc_lo = _mm_add_epi16(_mm_add_epi16(acc_lo, s0_lo), s1_lo);
            acc_hi = _mm_add_epi16(_mm_add_epi16(acc_hi, s0_hi), s1_hi);
        }

        // Widen i16 -> i32 (SSE4.1 pmovsxwd) and store.
        _mm_storeu_si128(out.as_mut_ptr()        as *mut __m128i, _mm_cvtepi16_epi32(acc_lo));
        _mm_storeu_si128(out.as_mut_ptr().add(4) as *mut __m128i, _mm_cvtepi16_epi32(_mm_srli_si128(acc_lo, 8)));
        _mm_storeu_si128(out.as_mut_ptr().add(8) as *mut __m128i, _mm_cvtepi16_epi32(acc_hi));
        _mm_storeu_si128(out.as_mut_ptr().add(12)as *mut __m128i, _mm_cvtepi16_epi32(_mm_srli_si128(acc_hi, 8)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes, no rand dependency in tests.
    fn residual_bytes(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 13) & 0xFF) as u8).collect()
    }

    fn lut_for(dim: usize) -> Vec<i8> {
        (0..dim * 16)
            .map(|i| (((i * 31 + 7) % 255) as i32 - 127) as i8)
            .collect()
    }

    /// The SIMD kernel must agree with the scalar reference exactly, for
    /// capacities on both sides of the 16-doc batch boundary. The standalone
    /// benchmark only ever used n=8192 (divisible by 16), so the tail path
    /// was previously untested.
    #[test]
    fn simd_matches_scalar_including_tail() {
        let dim = 128;
        let bpe = dim / 2;
        let lut = lut_for(dim);

        for &cap in &[0usize, 1, 3, 15, 16, 17, 31, 32, 33, 100, 1000] {
            let res = residual_bytes(cap * bpe);
            let got = score_batch_4bit(&res, cap, bpe, &lut);
            let want = scalar_fallback(&res, cap, bpe, &lut);
            assert_eq!(got, want, "mismatch at capacity {cap}");
        }
    }

    /// dim > MAX_SIMD_DIM would overflow the i16 accumulators. The guard
    /// must route to scalar so results stay correct (and, critically, so the
    /// SIMD body and the scalar tail don't disagree within one cell).
    #[test]
    fn large_dim_falls_back_to_scalar_without_overflow() {
        let dim = 384; // > MAX_SIMD_DIM (258)
        let bpe = dim / 2;
        let cap = 20; // spans one full 16-batch plus a 4-doc tail
        // All-max LUT maximizes the accumulator: 384 * 127 = 48_768 > i16::MAX.
        let lut = vec![127i8; dim * 16];
        let res = residual_bytes(cap * bpe);

        let got = score_batch_4bit(&res, cap, bpe, &lut);
        let want = scalar_fallback(&res, cap, bpe, &lut);
        assert_eq!(got, want);
        // Every doc sums dim entries of +127; nothing may wrap negative.
        for (i, &s) in got.iter().enumerate() {
            assert_eq!(s, (dim as i32) * 127, "doc {i} wrapped: {s}");
        }
    }

    /// A residual buffer shorter than capacity * bytes_per_emb must not be
    /// read out of bounds. This is reachable in release builds because
    /// `process_cell_impl` uses `.unwrap_or_default()`, which can hand us an
    /// empty Vec while capacity > 0.
    #[test]
    fn short_or_empty_residuals_do_not_read_out_of_bounds() {
        let dim = 128;
        let bpe = dim / 2;
        let lut = lut_for(dim);

        // Empty buffer, non-zero capacity (the unwrap_or_default case).
        assert_eq!(score_batch_4bit(&[], 32, bpe, &lut), vec![0i32; 32]);

        // Buffer truncated mid-way: still must not over-read.
        let res = residual_bytes(20 * bpe);
        assert_eq!(score_batch_4bit(&res, 40, bpe, &lut), vec![0i32; 40]);
    }

    /// A LUT shorter than dim*16 must be rejected rather than over-read.
    #[test]
    fn short_lut_is_rejected() {
        let dim = 128;
        let bpe = dim / 2;
        let short = vec![0i8; dim * 16 - 1];
        let res = residual_bytes(16 * bpe);
        assert_eq!(score_batch_4bit(&res, 16, bpe, &short), vec![0i32; 16]);
    }
}
