/*
 * benchmark_sve_ld2_vs_ld1.c
 *
 * Compare SVE structured loads/stores vs manual deinterleave/interleave:
 *
 *   LOAD:
 *     Structured:  ld2d {z1.d,z2.d}, p0/z, [x1]
 *     Manual:      ld1d z3.d,p0/z,[x1] ; ld1d z4.d,p0/z,[x1,#1,mul vl]
 *                  uzp1 z1.d,z3.d,z4.d ; uzp2 z2.d,z3.d,z4.d
 *
 *   STORE:
 *     Structured:  st2d {z1.d,z2.d}, p0, [x1]
 *     Manual:      zip1 z3.d,z0.d,z1.d ; zip2 z4.d,z0.d,z1.d
 *                  st1d z3.d,p0,[x1] ; st1d z4.d,p0,[x1,#1,mul vl]
 *
 * Compile:
 *   clang -O2 -march=armv9-a+sve2 -o bench_ld2 benchmark_sve_ld2_vs_ld1.c -lm
 *
 * Run:
 *   ./bench_ld2 > results.csv
 *
 * NOTE: Load kernels use 4× unrolling with 8 independent accumulators
 * to avoid bottlenecking on the add-chain latency, ensuring we measure
 * true load-port throughput rather than accumulator dependency.
 */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Tuning                                                            */
/* ------------------------------------------------------------------ */
#define L1_SIZE (48 * 1024)         /* 48 KiB — fits in L1D         */
#define COLD_SIZE (8 * 1024 * 1024) /* 8 MiB  — exceeds L1+L2      */
#define NUM_STEPS 1024              /* load/store steps per iter    */
#define ITERS_HOT 200000
#define ITERS_COLD 20000
#define WARMUP_ITERS 1000

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
static inline uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint64_t rng_state = 0x12345678ABCDEF01ULL;
static inline uint64_t xorshift64(void) {
  uint64_t x = rng_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  rng_state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

static volatile int64_t sink64;
static volatile int32_t sink32;

/* Fill offset arrays.  Each offset is an *element* index into data[]
 * where a 2×VL-element structured load/store begins.
 * stride = 2 * vl  (elements consumed per step).                     */
static void fill_offsets_same(uint32_t *off, int n) {
  for (int i = 0; i < n; i++)
    off[i] = 0;
}

static void fill_offsets_linear(uint32_t *off, int n, uint32_t stride,
                                uint32_t max_elems) {
  uint32_t usable = (max_elems / stride) * stride; /* round down */
  if (usable == 0)
    usable = stride; /* safety     */
  for (int i = 0; i < n; i++)
    off[i] = (uint32_t)(((uint64_t)i * stride) % usable);
}

static void fill_offsets_random(uint32_t *off, int n, uint32_t stride,
                                uint32_t max_elems) {
  uint32_t num_slots = max_elems / stride;
  if (num_slots == 0)
    num_slots = 1;
  for (int i = 0; i < n; i++)
    off[i] = (uint32_t)(xorshift64() % num_slots) * stride;
}

static void flush_array(void *p, size_t bytes) {
  volatile char *c = (volatile char *)p;
  for (size_t i = 0; i < bytes; i += 64)
    c[i] = (char)(c[i] + 1);
}

/* ================================================================== */
/*  i64 LOAD kernels                                                  */
/* ================================================================== */

static uint64_t bench_ld2d_load(const int64_t *data, const uint32_t *off,
                                int n_steps, int iters) {
  svbool_t pg = svptrue_b64();
  /* 8 independent accumulators — break the dependency chain so the
   * benchmark measures load throughput, not add-chain latency.       */
  svint64_t a0 = svdup_s64(0), a1 = svdup_s64(0), a2 = svdup_s64(0),
            a3 = svdup_s64(0), a4 = svdup_s64(0), a5 = svdup_s64(0),
            a6 = svdup_s64(0), a7 = svdup_s64(0);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s += 4) {
      svint64x2_t p0 = svld2_s64(pg, &data[off[s]]);
      a0 = svadd_s64_x(pg, a0, svget2_s64(p0, 0));
      a1 = svadd_s64_x(pg, a1, svget2_s64(p0, 1));

      svint64x2_t p1 = svld2_s64(pg, &data[off[s + 1]]);
      a2 = svadd_s64_x(pg, a2, svget2_s64(p1, 0));
      a3 = svadd_s64_x(pg, a3, svget2_s64(p1, 1));

      svint64x2_t p2 = svld2_s64(pg, &data[off[s + 2]]);
      a4 = svadd_s64_x(pg, a4, svget2_s64(p2, 0));
      a5 = svadd_s64_x(pg, a5, svget2_s64(p2, 1));

      svint64x2_t p3 = svld2_s64(pg, &data[off[s + 3]]);
      a6 = svadd_s64_x(pg, a6, svget2_s64(p3, 0));
      a7 = svadd_s64_x(pg, a7, svget2_s64(p3, 1));
    }
  }
  uint64_t t1 = now_ns();
  a0 = svadd_s64_x(pg, a0, a1);
  a2 = svadd_s64_x(pg, a2, a3);
  a4 = svadd_s64_x(pg, a4, a5);
  a6 = svadd_s64_x(pg, a6, a7);
  a0 = svadd_s64_x(pg, a0, a2);
  a4 = svadd_s64_x(pg, a4, a6);
  a0 = svadd_s64_x(pg, a0, a4);
  sink64 = svaddv_s64(pg, a0);
  return t1 - t0;
}

static uint64_t bench_ld1d_uzp_load(const int64_t *data, const uint32_t *off,
                                    int n_steps, int iters) {
  svbool_t pg = svptrue_b64();
  uint32_t vl = svcntd();
  svint64_t a0 = svdup_s64(0), a1 = svdup_s64(0), a2 = svdup_s64(0),
            a3 = svdup_s64(0), a4 = svdup_s64(0), a5 = svdup_s64(0),
            a6 = svdup_s64(0), a7 = svdup_s64(0);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s += 4) {
      {
        const int64_t *p = &data[off[s]];
        svint64_t lo = svld1_s64(pg, p);
        svint64_t hi = svld1_s64(pg, p + vl);
        a0 = svadd_s64_x(pg, a0, svuzp1_s64(lo, hi));
        a1 = svadd_s64_x(pg, a1, svuzp2_s64(lo, hi));
      }
      {
        const int64_t *p = &data[off[s + 1]];
        svint64_t lo = svld1_s64(pg, p);
        svint64_t hi = svld1_s64(pg, p + vl);
        a2 = svadd_s64_x(pg, a2, svuzp1_s64(lo, hi));
        a3 = svadd_s64_x(pg, a3, svuzp2_s64(lo, hi));
      }
      {
        const int64_t *p = &data[off[s + 2]];
        svint64_t lo = svld1_s64(pg, p);
        svint64_t hi = svld1_s64(pg, p + vl);
        a4 = svadd_s64_x(pg, a4, svuzp1_s64(lo, hi));
        a5 = svadd_s64_x(pg, a5, svuzp2_s64(lo, hi));
      }
      {
        const int64_t *p = &data[off[s + 3]];
        svint64_t lo = svld1_s64(pg, p);
        svint64_t hi = svld1_s64(pg, p + vl);
        a6 = svadd_s64_x(pg, a6, svuzp1_s64(lo, hi));
        a7 = svadd_s64_x(pg, a7, svuzp2_s64(lo, hi));
      }
    }
  }
  uint64_t t1 = now_ns();
  a0 = svadd_s64_x(pg, a0, a1);
  a2 = svadd_s64_x(pg, a2, a3);
  a4 = svadd_s64_x(pg, a4, a5);
  a6 = svadd_s64_x(pg, a6, a7);
  a0 = svadd_s64_x(pg, a0, a2);
  a4 = svadd_s64_x(pg, a4, a6);
  a0 = svadd_s64_x(pg, a0, a4);
  sink64 = svaddv_s64(pg, a0);
  return t1 - t0;
}

/* ================================================================== */
/*  i64 STORE kernels                                                 */
/* ================================================================== */

static uint64_t bench_st2d_store(int64_t *data, const uint32_t *off,
                                 int n_steps, int iters) {
  svbool_t pg = svptrue_b64();
  svint64_t z1 = svdup_s64(42);
  svint64_t z2 = svdup_s64(84);
  svint64x2_t pair = svcreate2_s64(z1, z2);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int64_t *p = &data[off[s]];
      svst2_s64(pg, p, pair);
    }
  }
  uint64_t t1 = now_ns();
  sink64 = data[0];
  return t1 - t0;
}

static uint64_t bench_zip_st1d_store(int64_t *data, const uint32_t *off,
                                     int n_steps, int iters) {
  svbool_t pg = svptrue_b64();
  uint32_t vl = svcntd();
  svint64_t z0 = svdup_s64(42);
  svint64_t z1 = svdup_s64(84);
  /* zip done inside loop — mirrors real usage where data varies */
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int64_t *p = &data[off[s]];
      svint64_t z3 = svzip1_s64(z0, z1);
      svint64_t z4 = svzip2_s64(z0, z1);
      svst1_s64(pg, p, z3);
      svst1_s64(pg, p + vl, z4);
    }
  }
  uint64_t t1 = now_ns();
  sink64 = data[0];
  return t1 - t0;
}

/* ================================================================== */
/*  i64 LOAD+STORE kernels                                            */
/* ================================================================== */

static uint64_t bench_ld2d_st2d(int64_t *data, const uint32_t *off, int n_steps,
                                int iters) {
  svbool_t pg = svptrue_b64();
  svint64_t inc = svdup_s64(1);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int64_t *p = &data[off[s]];
      svint64x2_t pair = svld2_s64(pg, p);
      svint64_t z1 = svadd_s64_x(pg, svget2_s64(pair, 0), inc);
      svint64_t z2 = svadd_s64_x(pg, svget2_s64(pair, 1), inc);
      svst2_s64(pg, p, svcreate2_s64(z1, z2));
    }
  }
  uint64_t t1 = now_ns();
  sink64 = data[0];
  return t1 - t0;
}

static uint64_t bench_ld1d_uzp_zip_st1d(int64_t *data, const uint32_t *off,
                                        int n_steps, int iters) {
  svbool_t pg = svptrue_b64();
  uint32_t vl = svcntd();
  svint64_t inc = svdup_s64(1);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int64_t *p = &data[off[s]];
      /* load */
      svint64_t z3 = svld1_s64(pg, p);
      svint64_t z4 = svld1_s64(pg, p + vl);
      svint64_t z1 = svuzp1_s64(z3, z4);
      svint64_t z2 = svuzp2_s64(z3, z4);
      /* modify */
      z1 = svadd_s64_x(pg, z1, inc);
      z2 = svadd_s64_x(pg, z2, inc);
      /* store */
      z3 = svzip1_s64(z1, z2);
      z4 = svzip2_s64(z1, z2);
      svst1_s64(pg, p, z3);
      svst1_s64(pg, p + vl, z4);
    }
  }
  uint64_t t1 = now_ns();
  sink64 = data[0];
  return t1 - t0;
}

/* ================================================================== */
/*  i32 LOAD kernels                                                  */
/* ================================================================== */

static uint64_t bench_ld2w_load(const int32_t *data, const uint32_t *off,
                                int n_steps, int iters) {
  svbool_t pg = svptrue_b32();
  svint32_t a0 = svdup_s32(0), a1 = svdup_s32(0), a2 = svdup_s32(0),
            a3 = svdup_s32(0), a4 = svdup_s32(0), a5 = svdup_s32(0),
            a6 = svdup_s32(0), a7 = svdup_s32(0);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s += 4) {
      svint32x2_t p0 = svld2_s32(pg, &data[off[s]]);
      a0 = svadd_s32_x(pg, a0, svget2_s32(p0, 0));
      a1 = svadd_s32_x(pg, a1, svget2_s32(p0, 1));

      svint32x2_t p1 = svld2_s32(pg, &data[off[s + 1]]);
      a2 = svadd_s32_x(pg, a2, svget2_s32(p1, 0));
      a3 = svadd_s32_x(pg, a3, svget2_s32(p1, 1));

      svint32x2_t p2 = svld2_s32(pg, &data[off[s + 2]]);
      a4 = svadd_s32_x(pg, a4, svget2_s32(p2, 0));
      a5 = svadd_s32_x(pg, a5, svget2_s32(p2, 1));

      svint32x2_t p3 = svld2_s32(pg, &data[off[s + 3]]);
      a6 = svadd_s32_x(pg, a6, svget2_s32(p3, 0));
      a7 = svadd_s32_x(pg, a7, svget2_s32(p3, 1));
    }
  }
  uint64_t t1 = now_ns();
  a0 = svadd_s32_x(pg, a0, a1);
  a2 = svadd_s32_x(pg, a2, a3);
  a4 = svadd_s32_x(pg, a4, a5);
  a6 = svadd_s32_x(pg, a6, a7);
  a0 = svadd_s32_x(pg, a0, a2);
  a4 = svadd_s32_x(pg, a4, a6);
  a0 = svadd_s32_x(pg, a0, a4);
  sink32 = (int32_t)svaddv_s32(pg, a0);
  return t1 - t0;
}

static uint64_t bench_ld1w_uzp_load(const int32_t *data, const uint32_t *off,
                                    int n_steps, int iters) {
  svbool_t pg = svptrue_b32();
  uint32_t vl = svcntw();
  svint32_t a0 = svdup_s32(0), a1 = svdup_s32(0), a2 = svdup_s32(0),
            a3 = svdup_s32(0), a4 = svdup_s32(0), a5 = svdup_s32(0),
            a6 = svdup_s32(0), a7 = svdup_s32(0);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s += 4) {
      {
        const int32_t *p = &data[off[s]];
        svint32_t lo = svld1_s32(pg, p);
        svint32_t hi = svld1_s32(pg, p + vl);
        a0 = svadd_s32_x(pg, a0, svuzp1_s32(lo, hi));
        a1 = svadd_s32_x(pg, a1, svuzp2_s32(lo, hi));
      }
      {
        const int32_t *p = &data[off[s + 1]];
        svint32_t lo = svld1_s32(pg, p);
        svint32_t hi = svld1_s32(pg, p + vl);
        a2 = svadd_s32_x(pg, a2, svuzp1_s32(lo, hi));
        a3 = svadd_s32_x(pg, a3, svuzp2_s32(lo, hi));
      }
      {
        const int32_t *p = &data[off[s + 2]];
        svint32_t lo = svld1_s32(pg, p);
        svint32_t hi = svld1_s32(pg, p + vl);
        a4 = svadd_s32_x(pg, a4, svuzp1_s32(lo, hi));
        a5 = svadd_s32_x(pg, a5, svuzp2_s32(lo, hi));
      }
      {
        const int32_t *p = &data[off[s + 3]];
        svint32_t lo = svld1_s32(pg, p);
        svint32_t hi = svld1_s32(pg, p + vl);
        a6 = svadd_s32_x(pg, a6, svuzp1_s32(lo, hi));
        a7 = svadd_s32_x(pg, a7, svuzp2_s32(lo, hi));
      }
    }
  }
  uint64_t t1 = now_ns();
  a0 = svadd_s32_x(pg, a0, a1);
  a2 = svadd_s32_x(pg, a2, a3);
  a4 = svadd_s32_x(pg, a4, a5);
  a6 = svadd_s32_x(pg, a6, a7);
  a0 = svadd_s32_x(pg, a0, a2);
  a4 = svadd_s32_x(pg, a4, a6);
  a0 = svadd_s32_x(pg, a0, a4);
  sink32 = (int32_t)svaddv_s32(pg, a0);
  return t1 - t0;
}

/* ================================================================== */
/*  i32 STORE kernels                                                 */
/* ================================================================== */

static uint64_t bench_st2w_store(int32_t *data, const uint32_t *off,
                                 int n_steps, int iters) {
  svbool_t pg = svptrue_b32();
  svint32_t z1 = svdup_s32(42);
  svint32_t z2 = svdup_s32(84);
  svint32x2_t pair = svcreate2_s32(z1, z2);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int32_t *p = &data[off[s]];
      svst2_s32(pg, p, pair);
    }
  }
  uint64_t t1 = now_ns();
  sink32 = data[0];
  return t1 - t0;
}

static uint64_t bench_zip_st1w_store(int32_t *data, const uint32_t *off,
                                     int n_steps, int iters) {
  svbool_t pg = svptrue_b32();
  uint32_t vl = svcntw();
  svint32_t z0 = svdup_s32(42);
  svint32_t z1 = svdup_s32(84);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int32_t *p = &data[off[s]];
      svint32_t z3 = svzip1_s32(z0, z1);
      svint32_t z4 = svzip2_s32(z0, z1);
      svst1_s32(pg, p, z3);
      svst1_s32(pg, p + vl, z4);
    }
  }
  uint64_t t1 = now_ns();
  sink32 = data[0];
  return t1 - t0;
}

/* ================================================================== */
/*  i32 LOAD+STORE kernels                                            */
/* ================================================================== */

static uint64_t bench_ld2w_st2w(int32_t *data, const uint32_t *off, int n_steps,
                                int iters) {
  svbool_t pg = svptrue_b32();
  svint32_t inc = svdup_s32(1);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int32_t *p = &data[off[s]];
      svint32x2_t pair = svld2_s32(pg, p);
      svint32_t z1 = svadd_s32_x(pg, svget2_s32(pair, 0), inc);
      svint32_t z2 = svadd_s32_x(pg, svget2_s32(pair, 1), inc);
      svst2_s32(pg, p, svcreate2_s32(z1, z2));
    }
  }
  uint64_t t1 = now_ns();
  sink32 = data[0];
  return t1 - t0;
}

static uint64_t bench_ld1w_uzp_zip_st1w(int32_t *data, const uint32_t *off,
                                        int n_steps, int iters) {
  svbool_t pg = svptrue_b32();
  uint32_t vl = svcntw();
  svint32_t inc = svdup_s32(1);
  uint64_t t0 = now_ns();
  for (int it = 0; it < iters; it++) {
    for (int s = 0; s < n_steps; s++) {
      int32_t *p = &data[off[s]];
      svint32_t z3 = svld1_s32(pg, p);
      svint32_t z4 = svld1_s32(pg, p + vl);
      svint32_t z1 = svuzp1_s32(z3, z4);
      svint32_t z2 = svuzp2_s32(z3, z4);
      z1 = svadd_s32_x(pg, z1, inc);
      z2 = svadd_s32_x(pg, z2, inc);
      z3 = svzip1_s32(z1, z2);
      z4 = svzip2_s32(z1, z2);
      svst1_s32(pg, p, z3);
      svst1_s32(pg, p + vl, z4);
    }
  }
  uint64_t t1 = now_ns();
  sink32 = data[0];
  return t1 - t0;
}

/* ================================================================== */
/*  Generic runners                                                   */
/* ================================================================== */

typedef uint64_t (*bench_fn_64)(int64_t *, const uint32_t *, int, int);
typedef uint64_t (*bench_fn_32)(int32_t *, const uint32_t *, int, int);
/* cast-compatible load variants (const data ptr) handled via cast    */

static void run64(const char *cache, const char *pat, const char *op,
                  const char *method, bench_fn_64 fn, int64_t *data,
                  size_t data_bytes, const uint32_t *off, int n_steps,
                  int iters, int is_cold, uint32_t vl) {
  /* warmup */
  fn(data, off, n_steps, WARMUP_ITERS);
  if (is_cold)
    flush_array(data, data_bytes);

  uint64_t ns = fn(data, off, n_steps, iters);
  double ms = ns / 1e6;

  /* Each step loads/stores 2*vl int64 elements = 2*vl*8 bytes */
  double bytes_per_step = 2.0 * vl * 8.0;
  double dir = 1.0;
  if (strstr(op, "Load+Store"))
    dir = 2.0;
  double total_bytes = (double)iters * n_steps * bytes_per_step * dir;
  double gbps = (total_bytes / (ns / 1e9)) / 1e9;

  printf("%-5s,%-7s,%-11s,i64,%-10s,%.3f,%.2f\n", cache, pat, op, method, ms,
         gbps);
}

static void run32(const char *cache, const char *pat, const char *op,
                  const char *method, bench_fn_32 fn, int32_t *data,
                  size_t data_bytes, const uint32_t *off, int n_steps,
                  int iters, int is_cold, uint32_t vl) {
  fn(data, off, n_steps, WARMUP_ITERS);
  if (is_cold)
    flush_array(data, data_bytes);

  uint64_t ns = fn(data, off, n_steps, iters);
  double ms = ns / 1e6;

  double bytes_per_step = 2.0 * vl * 4.0;
  double dir = 1.0;
  if (strstr(op, "Load+Store"))
    dir = 2.0;
  double total_bytes = (double)iters * n_steps * bytes_per_step * dir;
  double gbps = (total_bytes / (ns / 1e9)) / 1e9;

  printf("%-5s,%-7s,%-11s,i32,%-10s,%.3f,%.2f\n", cache, pat, op, method, ms,
         gbps);
}

/* ================================================================== */
/*  main                                                              */
/* ================================================================== */
int main(void) {
  /* ---- discover VL ---- */
  uint32_t vl_d = svcntd(); /* doublewords per vector */
  uint32_t vl_w = svcntw(); /* words per vector       */
  uint32_t stride_d = 2 * vl_d;
  uint32_t stride_w = 2 * vl_w;

  fprintf(stderr, "SVE VL: %u bits  (%u x i64, %u x i32 per register)\n",
          vl_d * 64, vl_d, vl_w);

  /* ---- data arrays ---- */
  int64_t *d64_hot = aligned_alloc(64, L1_SIZE);
  int64_t *d64_cold = aligned_alloc(64, COLD_SIZE);
  int32_t *d32_hot = aligned_alloc(64, L1_SIZE);
  int32_t *d32_cold = aligned_alloc(64, COLD_SIZE);
  memset(d64_hot, 1, L1_SIZE);
  memset(d64_cold, 1, COLD_SIZE);
  memset(d32_hot, 1, L1_SIZE);
  memset(d32_cold, 1, COLD_SIZE);

  uint32_t hot_elems_64 = L1_SIZE / sizeof(int64_t);
  uint32_t cold_elems_64 = COLD_SIZE / sizeof(int64_t);
  uint32_t hot_elems_32 = L1_SIZE / sizeof(int32_t);
  uint32_t cold_elems_32 = COLD_SIZE / sizeof(int32_t);

  /* ---- offset arrays (one set per cache×pattern×dtype) ---- */
  uint32_t *off = aligned_alloc(64, NUM_STEPS * sizeof(uint32_t));

  /* We'll fill offsets on the fly for each combo.  Use a macro. */

  printf(
      "Cache,Pattern,Operation,DataType,Method,Duration_ms,Bandwidth_Gbps\n");

  /* ---- convenience macros ---- */
#define DO_LOAD_64(C, PAT, DATA, DBYTES, OFF, ITERS, COLD)                     \
  run64(C, PAT, "Load", "ld2d", (bench_fn_64)bench_ld2d_load, DATA, DBYTES,    \
        OFF, NUM_STEPS, ITERS, COLD, vl_d);                                    \
  run64(C, PAT, "Load", "ld1d+uzp", (bench_fn_64)bench_ld1d_uzp_load, DATA,    \
        DBYTES, OFF, NUM_STEPS, ITERS, COLD, vl_d);

#define DO_STORE_64(C, PAT, DATA, DBYTES, OFF, ITERS, COLD)                    \
  run64(C, PAT, "Store", "st2d", bench_st2d_store, DATA, DBYTES, OFF,          \
        NUM_STEPS, ITERS, COLD, vl_d);                                         \
  run64(C, PAT, "Store", "zip+st1d", bench_zip_st1d_store, DATA, DBYTES, OFF,  \
        NUM_STEPS, ITERS, COLD, vl_d);

#define DO_LS_64(C, PAT, DATA, DBYTES, OFF, ITERS, COLD)                       \
  run64(C, PAT, "Load+Store", "ld2d+st2d", bench_ld2d_st2d, DATA, DBYTES, OFF, \
        NUM_STEPS, ITERS, COLD, vl_d);                                         \
  run64(C, PAT, "Load+Store", "ld1d+st1d", bench_ld1d_uzp_zip_st1d, DATA,      \
        DBYTES, OFF, NUM_STEPS, ITERS, COLD, vl_d);

#define DO_LOAD_32(C, PAT, DATA, DBYTES, OFF, ITERS, COLD)                     \
  run32(C, PAT, "Load", "ld2w", (bench_fn_32)bench_ld2w_load, DATA, DBYTES,    \
        OFF, NUM_STEPS, ITERS, COLD, vl_w);                                    \
  run32(C, PAT, "Load", "ld1w+uzp", (bench_fn_32)bench_ld1w_uzp_load, DATA,    \
        DBYTES, OFF, NUM_STEPS, ITERS, COLD, vl_w);

#define DO_STORE_32(C, PAT, DATA, DBYTES, OFF, ITERS, COLD)                    \
  run32(C, PAT, "Store", "st2w", bench_st2w_store, DATA, DBYTES, OFF,          \
        NUM_STEPS, ITERS, COLD, vl_w);                                         \
  run32(C, PAT, "Store", "zip+st1w", bench_zip_st1w_store, DATA, DBYTES, OFF,  \
        NUM_STEPS, ITERS, COLD, vl_w);

#define DO_LS_32(C, PAT, DATA, DBYTES, OFF, ITERS, COLD)                       \
  run32(C, PAT, "Load+Store", "ld2w+st2w", bench_ld2w_st2w, DATA, DBYTES, OFF, \
        NUM_STEPS, ITERS, COLD, vl_w);                                         \
  run32(C, PAT, "Load+Store", "ld1w+st1w", bench_ld1w_uzp_zip_st1w, DATA,      \
        DBYTES, OFF, NUM_STEPS, ITERS, COLD, vl_w);

  /* ============================================================== */
  /*  i64                                                           */
  /* ============================================================== */

  /* -- Hot / Same -- */
  fill_offsets_same(off, NUM_STEPS);
  DO_LOAD_64("Hot", "Same", d64_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_STORE_64("Hot", "Same", d64_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_LS_64("Hot", "Same", d64_hot, L1_SIZE, off, ITERS_HOT, 0);

  /* -- Hot / Linear -- */
  fill_offsets_linear(off, NUM_STEPS, stride_d, hot_elems_64);
  DO_LOAD_64("Hot", "Linear", d64_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_STORE_64("Hot", "Linear", d64_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_LS_64("Hot", "Linear", d64_hot, L1_SIZE, off, ITERS_HOT, 0);

  /* -- Hot / Random -- */
  rng_state = 0xDEADBEEF;
  fill_offsets_random(off, NUM_STEPS, stride_d, hot_elems_64);
  DO_LOAD_64("Hot", "Random", d64_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_STORE_64("Hot", "Random", d64_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_LS_64("Hot", "Random", d64_hot, L1_SIZE, off, ITERS_HOT, 0);

  /* -- Cold / Same -- */
  fill_offsets_same(off, NUM_STEPS);
  DO_LOAD_64("Cold", "Same", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_STORE_64("Cold", "Same", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_LS_64("Cold", "Same", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);

  /* -- Cold / Linear -- */
  fill_offsets_linear(off, NUM_STEPS, stride_d, cold_elems_64);
  DO_LOAD_64("Cold", "Linear", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_STORE_64("Cold", "Linear", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_LS_64("Cold", "Linear", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);

  /* -- Cold / Random -- */
  rng_state = 0xCAFEBABE;
  fill_offsets_random(off, NUM_STEPS, stride_d, cold_elems_64);
  DO_LOAD_64("Cold", "Random", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_STORE_64("Cold", "Random", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_LS_64("Cold", "Random", d64_cold, COLD_SIZE, off, ITERS_COLD, 1);

  /* ============================================================== */
  /*  i32                                                           */
  /* ============================================================== */

  /* -- Hot / Same -- */
  fill_offsets_same(off, NUM_STEPS);
  DO_LOAD_32("Hot", "Same", d32_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_STORE_32("Hot", "Same", d32_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_LS_32("Hot", "Same", d32_hot, L1_SIZE, off, ITERS_HOT, 0);

  /* -- Hot / Linear -- */
  fill_offsets_linear(off, NUM_STEPS, stride_w, hot_elems_32);
  DO_LOAD_32("Hot", "Linear", d32_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_STORE_32("Hot", "Linear", d32_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_LS_32("Hot", "Linear", d32_hot, L1_SIZE, off, ITERS_HOT, 0);

  /* -- Hot / Random -- */
  rng_state = 0xFEEDFACE;
  fill_offsets_random(off, NUM_STEPS, stride_w, hot_elems_32);
  DO_LOAD_32("Hot", "Random", d32_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_STORE_32("Hot", "Random", d32_hot, L1_SIZE, off, ITERS_HOT, 0);
  DO_LS_32("Hot", "Random", d32_hot, L1_SIZE, off, ITERS_HOT, 0);

  /* -- Cold / Same -- */
  fill_offsets_same(off, NUM_STEPS);
  DO_LOAD_32("Cold", "Same", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_STORE_32("Cold", "Same", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_LS_32("Cold", "Same", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);

  /* -- Cold / Linear -- */
  fill_offsets_linear(off, NUM_STEPS, stride_w, cold_elems_32);
  DO_LOAD_32("Cold", "Linear", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_STORE_32("Cold", "Linear", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_LS_32("Cold", "Linear", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);

  /* -- Cold / Random -- */
  rng_state = 0xBAADF00D;
  fill_offsets_random(off, NUM_STEPS, stride_w, cold_elems_32);
  DO_LOAD_32("Cold", "Random", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_STORE_32("Cold", "Random", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);
  DO_LS_32("Cold", "Random", d32_cold, COLD_SIZE, off, ITERS_COLD, 1);

  /* ---- cleanup ---- */
  free(d64_hot);
  free(d64_cold);
  free(d32_hot);
  free(d32_cold);
  free(off);
  return 0;
}
