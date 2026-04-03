/*
 * benchmark.c — Gather/Scatter microbenchmark: NEON (emulated) vs SVE (native)
 *
 * Compile (pick ONE target at a time):
 *
 *   NEON build:
 *     clang -O2 -march=armv8-a+simd -DUSE_NEON -o bench_neon benchmark.c -lm
 *
 *   SVE build  (fixed 256-bit VL for reproducibility; drop the vector-bits
 *               flag to let hardware choose):
 *     clang -O2 -march=armv9-a+sve2 -DUSE_SVE -o bench_sve benchmark.c -lm
 *
 *   SVE build  (hardware VL):
 *     clang -O2 -march=armv9-a+sve2 -DUSE_SVE -o bench_sve benchmark.c -lm
 *
 * Run:
 *     ./bench_neon        # prints CSV to stdout
 *     ./bench_sve
 *
 * Combine results:
 *     (head -1 neon.csv; tail -n+2 neon.csv; tail -n+2 sve.csv) > all.csv
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_SVE
#include <arm_sve.h>
#endif

/* ------------------------------------------------------------------ */
/*  Tuning knobs                                                      */
/* ------------------------------------------------------------------ */
#define L1_SIZE       (48 * 1024)          /* 48 KiB — typical Arm L1D    */
#define COLD_SIZE     (8 * 1024 * 1024)    /* 8 MiB  — blow past L1/L2   */
#define NUM_INDICES   1024                 /* number of gather/scatter idx*/
#define ITERS_HOT     200000               /* iterations for L1-hot       */
#define ITERS_COLD    20000                /* iterations for L1-cold      */
#define WARMUP_ITERS  1000                 /* warmup before timing        */

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* xorshift64* PRNG — fast, decent quality */
static uint64_t rng_state = 0x12345678ABCDEF01ULL;
static inline uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Fill index arrays for the three access patterns.
 * max_elem = number of elements in the data array.                    */
static void fill_indices_same(uint32_t *idx32, uint64_t *idx64, int n, int max_elem) {
    /* Every lane hits element 0 */
    for (int i = 0; i < n; i++) { idx32[i] = 0; idx64[i] = 0; }
}

static void fill_indices_linear(uint32_t *idx32, uint64_t *idx64, int n, int max_elem) {
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)(i % max_elem);
        idx32[i] = v;
        idx64[i] = (uint64_t)v;
    }
}

static void fill_indices_random(uint32_t *idx32, uint64_t *idx64, int n, int max_elem) {
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)(xorshift64() % (uint64_t)max_elem);
        idx32[i] = v;
        idx64[i] = (uint64_t)v;
    }
}

/* Prevent the compiler from optimising away the result */
static volatile uint64_t sink;

/* ------------------------------------------------------------------ */
/*  NEON emulated gather / scatter                                    */
/* ------------------------------------------------------------------ */
#ifdef USE_NEON

/* --- i32 gather (4 elements) --- */
static inline uint32x4_t neon_gather_i32(const int32_t *base, const uint32_t *idx) {
    int32_t tmp[4];
    tmp[0] = base[idx[0]];
    tmp[1] = base[idx[1]];
    tmp[2] = base[idx[2]];
    tmp[3] = base[idx[3]];
    return vld1q_u32((const uint32_t *)tmp);
}

/* --- i32 scatter (4 elements) --- */
static inline void neon_scatter_i32(int32_t *base, const uint32_t *idx, uint32x4_t val) {
    int32_t tmp[4];
    vst1q_s32(tmp, vreinterpretq_s32_u32(val));
    base[idx[0]] = tmp[0];
    base[idx[1]] = tmp[1];
    base[idx[2]] = tmp[2];
    base[idx[3]] = tmp[3];
}

/* --- i64 gather (2 elements) --- */
static inline uint64x2_t neon_gather_i64(const int64_t *base, const uint64_t *idx) {
    int64_t tmp[2];
    tmp[0] = base[idx[0]];
    tmp[1] = base[idx[1]];
    return vld1q_u64((const uint64_t *)tmp);
}

/* --- i64 scatter (2 elements) --- */
static inline void neon_scatter_i64(int64_t *base, const uint64_t *idx, uint64x2_t val) {
    int64_t tmp[2];
    vst1q_s64(tmp, vreinterpretq_s64_u64(val));
    base[idx[0]] = tmp[0];
    base[idx[1]] = tmp[1];
}

#define ARCH_NAME "NEON"

/* ---------- i32 benchmark kernels ---------- */

static uint64_t bench_neon_load_i32(int32_t *data, const uint32_t *indices, int n_idx, int iters) {
    uint32x4_t acc = vdupq_n_u32(0);
    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += 4) {
            uint32x4_t v = neon_gather_i32(data, &indices[i]);
            acc = vaddq_u32(acc, v);
        }
    }
    uint64_t t1 = now_ns();
    sink = vgetq_lane_u32(acc, 0);
    return t1 - t0;
}

static uint64_t bench_neon_store_i32(int32_t *data, const uint32_t *indices, int n_idx, int iters) {
    uint32x4_t val = vdupq_n_u32(42);
    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += 4) {
            neon_scatter_i32(data, &indices[i], val);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

static uint64_t bench_neon_loadstore_i32(int32_t *data, const uint32_t *indices, int n_idx, int iters) {
    uint32x4_t inc = vdupq_n_u32(1);
    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += 4) {
            uint32x4_t v = neon_gather_i32(data, &indices[i]);
            v = vaddq_u32(v, inc);
            neon_scatter_i32(data, &indices[i], v);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

/* ---------- i64 benchmark kernels ---------- */

static uint64_t bench_neon_load_i64(int64_t *data, const uint64_t *indices, int n_idx, int iters) {
    uint64x2_t acc = vdupq_n_u64(0);
    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += 2) {
            uint64x2_t v = neon_gather_i64(data, &indices[i]);
            acc = vaddq_u64(acc, v);
        }
    }
    uint64_t t1 = now_ns();
    sink = vgetq_lane_u64(acc, 0);
    return t1 - t0;
}

static uint64_t bench_neon_store_i64(int64_t *data, const uint64_t *indices, int n_idx, int iters) {
    uint64x2_t val = vdupq_n_u64(42);
    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += 2) {
            neon_scatter_i64(data, &indices[i], val);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

static uint64_t bench_neon_loadstore_i64(int64_t *data, const uint64_t *indices, int n_idx, int iters) {
    uint64x2_t inc = vdupq_n_u64(1);
    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += 2) {
            uint64x2_t v = neon_gather_i64(data, &indices[i]);
            v = vaddq_u64(v, inc);
            neon_scatter_i64(data, &indices[i], v);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

#endif /* USE_NEON */

/* ------------------------------------------------------------------ */
/*  SVE native gather / scatter                                       */
/* ------------------------------------------------------------------ */
#ifdef USE_SVE

#define ARCH_NAME "SVE"

/* ---------- i32 benchmark kernels ---------- */

static uint64_t bench_sve_load_i32(int32_t *data, const uint32_t *indices, int n_idx, int iters) {
    svbool_t pg = svptrue_b32();
    uint32_t vl = svcntw();                       /* elements per vector */
    svuint32_t acc = svdup_u32(0);

    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += vl) {
            svbool_t mask = svwhilelt_b32_u32((uint32_t)i, (uint32_t)n_idx);
            svuint32_t idx = svld1_u32(mask, &indices[i]);
            /* byte offsets = element indices * 4 */
            svuint32_t off = svlsl_n_u32_x(pg, idx, 2);
            svint32_t  v   = svld1_gather_u32offset_s32(mask, data, off);
            acc = svadd_u32_x(pg, acc, svreinterpret_u32_s32(v));
        }
    }
    uint64_t t1 = now_ns();
    sink = svaddv_u32(pg, acc);
    return t1 - t0;
}

static uint64_t bench_sve_store_i32(int32_t *data, const uint32_t *indices, int n_idx, int iters) {
    svbool_t pg = svptrue_b32();
    uint32_t vl = svcntw();
    svint32_t val = svdup_s32(42);

    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += vl) {
            svbool_t mask = svwhilelt_b32_u32((uint32_t)i, (uint32_t)n_idx);
            svuint32_t idx = svld1_u32(mask, &indices[i]);
            svuint32_t off = svlsl_n_u32_x(pg, idx, 2);
            svst1_scatter_u32offset_s32(mask, data, off, val);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

static uint64_t bench_sve_loadstore_i32(int32_t *data, const uint32_t *indices, int n_idx, int iters) {
    svbool_t pg = svptrue_b32();
    uint32_t vl = svcntw();
    svint32_t inc = svdup_s32(1);

    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += vl) {
            svbool_t mask = svwhilelt_b32_u32((uint32_t)i, (uint32_t)n_idx);
            svuint32_t idx = svld1_u32(mask, &indices[i]);
            svuint32_t off = svlsl_n_u32_x(pg, idx, 2);
            svint32_t  v   = svld1_gather_u32offset_s32(mask, data, off);
            v = svadd_s32_x(pg, v, inc);
            svst1_scatter_u32offset_s32(mask, data, off, v);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

/* ---------- i64 benchmark kernels ---------- */

static uint64_t bench_sve_load_i64(int64_t *data, const uint64_t *indices, int n_idx, int iters) {
    svbool_t pg = svptrue_b64();
    uint32_t vl = svcntd();
    svuint64_t acc = svdup_u64(0);

    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += vl) {
            svbool_t mask = svwhilelt_b64_u32((uint32_t)i, (uint32_t)n_idx);
            svuint64_t idx = svld1_u64(mask, &indices[i]);
            svuint64_t off = svlsl_n_u64_x(pg, idx, 3);
            svint64_t  v   = svld1_gather_u64offset_s64(mask, data, off);
            acc = svadd_u64_x(pg, acc, svreinterpret_u64_s64(v));
        }
    }
    uint64_t t1 = now_ns();
    sink = svaddv_u64(pg, acc);
    return t1 - t0;
}

static uint64_t bench_sve_store_i64(int64_t *data, const uint64_t *indices, int n_idx, int iters) {
    svbool_t pg = svptrue_b64();
    uint32_t vl = svcntd();
    svint64_t val = svdup_s64(42);

    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += vl) {
            svbool_t mask = svwhilelt_b64_u32((uint32_t)i, (uint32_t)n_idx);
            svuint64_t idx = svld1_u64(mask, &indices[i]);
            svuint64_t off = svlsl_n_u64_x(pg, idx, 3);
            svst1_scatter_u64offset_s64(mask, data, off, val);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

static uint64_t bench_sve_loadstore_i64(int64_t *data, const uint64_t *indices, int n_idx, int iters) {
    svbool_t pg = svptrue_b64();
    uint32_t vl = svcntd();
    svint64_t inc = svdup_s64(1);

    uint64_t t0 = now_ns();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_idx; i += vl) {
            svbool_t mask = svwhilelt_b64_u32((uint32_t)i, (uint32_t)n_idx);
            svuint64_t idx = svld1_u64(mask, &indices[i]);
            svuint64_t off = svlsl_n_u64_x(pg, idx, 3);
            svint64_t  v   = svld1_gather_u64offset_s64(mask, data, off);
            v = svadd_s64_x(pg, v, inc);
            svst1_scatter_u64offset_s64(mask, data, off, v);
        }
    }
    uint64_t t1 = now_ns();
    sink = data[0];
    return t1 - t0;
}

#endif /* USE_SVE */

/* ------------------------------------------------------------------ */
/*  Generic runner                                                    */
/* ------------------------------------------------------------------ */

typedef uint64_t (*bench_fn_i32)(int32_t *, const uint32_t *, int, int);
typedef uint64_t (*bench_fn_i64)(int64_t *, const uint64_t *, int, int);

static void flush_array(void *p, size_t bytes) {
    /* Write + read every cache line to evict previous contents from L1 */
    volatile char *c = (volatile char *)p;
    for (size_t i = 0; i < bytes; i += 64) {
        c[i] = (char)(c[i] + 1);
    }
}

static void run_i32(const char *cache_label, const char *pattern,
                    const char *op_label, bench_fn_i32 fn,
                    int32_t *data, size_t data_bytes,
                    const uint32_t *indices, int n_idx,
                    int iters, int is_cold)
{
    /* warmup */
    fn(data, indices, n_idx, WARMUP_ITERS);

    if (is_cold) flush_array(data, data_bytes);

    uint64_t ns = fn(data, indices, n_idx, iters);
    double ms = ns / 1e6;

    /* total bytes touched = iters * n_idx * sizeof(i32)
     * For load+store count both directions */
    double bytes_factor = 1.0;
    if (strcmp(op_label, "Load+Store") == 0) bytes_factor = 2.0;
    double total_bytes = (double)iters * n_idx * 4.0 * bytes_factor;
    double gbps = (total_bytes / (ns / 1e9)) / 1e9;

    printf("%-5s,%-7s,%-10s,i32,%s,%.3f,%.2f\n",
           cache_label, pattern, op_label, ARCH_NAME, ms, gbps);
}

static void run_i64(const char *cache_label, const char *pattern,
                    const char *op_label, bench_fn_i64 fn,
                    int64_t *data, size_t data_bytes,
                    const uint64_t *indices, int n_idx,
                    int iters, int is_cold)
{
    fn(data, indices, n_idx, WARMUP_ITERS);

    if (is_cold) flush_array(data, data_bytes);

    uint64_t ns = fn(data, indices, n_idx, iters);
    double ms = ns / 1e6;

    double bytes_factor = 1.0;
    if (strcmp(op_label, "Load+Store") == 0) bytes_factor = 2.0;
    double total_bytes = (double)iters * n_idx * 8.0 * bytes_factor;
    double gbps = (total_bytes / (ns / 1e9)) / 1e9;

    printf("%-5s,%-7s,%-10s,i64,%s,%.3f,%.2f\n",
           cache_label, pattern, op_label, ARCH_NAME, ms, gbps);
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */
int main(void) {

    /* ---- allocate data arrays ---- */
    int hot_elems_32  = L1_SIZE / sizeof(int32_t);    /* fits in L1     */
    int cold_elems_32 = COLD_SIZE / sizeof(int32_t);  /* blows L1       */
    int hot_elems_64  = L1_SIZE / sizeof(int64_t);
    int cold_elems_64 = COLD_SIZE / sizeof(int64_t);

    int32_t *data32_hot  = aligned_alloc(64, L1_SIZE);
    int32_t *data32_cold = aligned_alloc(64, COLD_SIZE);
    int64_t *data64_hot  = aligned_alloc(64, L1_SIZE);
    int64_t *data64_cold = aligned_alloc(64, COLD_SIZE);

    memset(data32_hot,  0, L1_SIZE);
    memset(data32_cold, 0, COLD_SIZE);
    memset(data64_hot,  0, L1_SIZE);
    memset(data64_cold, 0, COLD_SIZE);

    /* ---- index arrays for each pattern ---- */
    uint32_t *idx32_same   = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint32_t *idx32_lin    = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint32_t *idx32_rand_h = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint32_t *idx32_rand_c = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));

    uint64_t *idx64_same   = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    uint64_t *idx64_lin    = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    uint64_t *idx64_rand_h = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    uint64_t *idx64_rand_c = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));

    /* Same */
    fill_indices_same(idx32_same, idx64_same, NUM_INDICES, hot_elems_32);

    /* Linear — hot uses hot_elems, cold uses cold_elems */
    fill_indices_linear(idx32_lin, idx64_lin, NUM_INDICES, hot_elems_32);
    /* (for cold, we create separate linear arrays below) */
    uint32_t *idx32_lin_c  = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint64_t *idx64_lin_c  = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    fill_indices_linear(idx32_lin_c, idx64_lin_c, NUM_INDICES, cold_elems_32);

    /* Random — hot range vs cold range */
    rng_state = 0xDEADBEEF;
    fill_indices_random(idx32_rand_h, idx64_rand_h, NUM_INDICES, hot_elems_32);
    rng_state = 0xCAFEBABE;
    fill_indices_random(idx32_rand_c, idx64_rand_c, NUM_INDICES, cold_elems_32);

    /* Separate random index sets for i64 with correct element counts */
    uint32_t *idx32_rand_h64 = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint64_t *idx64_rand_h64 = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    uint32_t *idx32_rand_c64 = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint64_t *idx64_rand_c64 = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    uint32_t *idx32_lin_h64  = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint64_t *idx64_lin_h64  = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));
    uint32_t *idx32_lin_c64  = aligned_alloc(64, NUM_INDICES * sizeof(uint32_t));
    uint64_t *idx64_lin_c64  = aligned_alloc(64, NUM_INDICES * sizeof(uint64_t));

    fill_indices_linear(idx32_lin_h64, idx64_lin_h64, NUM_INDICES, hot_elems_64);
    fill_indices_linear(idx32_lin_c64, idx64_lin_c64, NUM_INDICES, cold_elems_64);
    rng_state = 0xFEEDFACE;
    fill_indices_random(idx32_rand_h64, idx64_rand_h64, NUM_INDICES, hot_elems_64);
    rng_state = 0xBAADF00D;
    fill_indices_random(idx32_rand_c64, idx64_rand_c64, NUM_INDICES, cold_elems_64);

    /* ---- function pointer tables ---- */
#ifdef USE_NEON
    bench_fn_i32 load32  = bench_neon_load_i32;
    bench_fn_i32 store32 = bench_neon_store_i32;
    bench_fn_i32 ls32    = bench_neon_loadstore_i32;
    bench_fn_i64 load64  = bench_neon_load_i64;
    bench_fn_i64 store64 = bench_neon_store_i64;
    bench_fn_i64 ls64    = bench_neon_loadstore_i64;
#else
    bench_fn_i32 load32  = bench_sve_load_i32;
    bench_fn_i32 store32 = bench_sve_store_i32;
    bench_fn_i32 ls32    = bench_sve_loadstore_i32;
    bench_fn_i64 load64  = bench_sve_load_i64;
    bench_fn_i64 store64 = bench_sve_store_i64;
    bench_fn_i64 ls64    = bench_sve_loadstore_i64;
#endif

    /* ---- header ---- */
    printf("Cache,Pattern,Operation,DataType,Arch,Duration_ms,Bandwidth_Gbps\n");

    /* ================================================================
     *  Macro to run all 3 ops for a given (cache, pattern, dtype)
     * ================================================================ */
#define RUN32(clbl, pat, data, dbytes, idx, iters, cold) \
    run_i32(clbl, pat, "Load",       load32,  data, dbytes, idx, NUM_INDICES, iters, cold); \
    run_i32(clbl, pat, "Store",      store32, data, dbytes, idx, NUM_INDICES, iters, cold); \
    run_i32(clbl, pat, "Load+Store", ls32,    data, dbytes, idx, NUM_INDICES, iters, cold);

#define RUN64(clbl, pat, data, dbytes, idx, iters, cold) \
    run_i64(clbl, pat, "Load",       load64,  data, dbytes, idx, NUM_INDICES, iters, cold); \
    run_i64(clbl, pat, "Store",      store64, data, dbytes, idx, NUM_INDICES, iters, cold); \
    run_i64(clbl, pat, "Load+Store", ls64,    data, dbytes, idx, NUM_INDICES, iters, cold);

    /* ---- i32, Hot ---- */
    RUN32("Hot", "Same",    data32_hot, L1_SIZE, idx32_same,    ITERS_HOT, 0);
    RUN32("Hot", "Linear",  data32_hot, L1_SIZE, idx32_lin,     ITERS_HOT, 0);
    RUN32("Hot", "Random",  data32_hot, L1_SIZE, idx32_rand_h,  ITERS_HOT, 0);

    /* ---- i32, Cold ---- */
    RUN32("Cold", "Same",   data32_cold, COLD_SIZE, idx32_same,    ITERS_COLD, 1);
    RUN32("Cold", "Linear", data32_cold, COLD_SIZE, idx32_lin_c,   ITERS_COLD, 1);
    RUN32("Cold", "Random", data32_cold, COLD_SIZE, idx32_rand_c,  ITERS_COLD, 1);

    /* ---- i64, Hot ---- */
    RUN64("Hot", "Same",    data64_hot, L1_SIZE, idx64_same,     ITERS_HOT, 0);
    RUN64("Hot", "Linear",  data64_hot, L1_SIZE, idx64_lin_h64,  ITERS_HOT, 0);
    RUN64("Hot", "Random",  data64_hot, L1_SIZE, idx64_rand_h64, ITERS_HOT, 0);

    /* ---- i64, Cold ---- */
    RUN64("Cold", "Same",   data64_cold, COLD_SIZE, idx64_same,     ITERS_COLD, 1);
    RUN64("Cold", "Linear", data64_cold, COLD_SIZE, idx64_lin_c64,  ITERS_COLD, 1);
    RUN64("Cold", "Random", data64_cold, COLD_SIZE, idx64_rand_c64, ITERS_COLD, 1);

    /* ---- cleanup ---- */
    free(data32_hot);  free(data32_cold);
    free(data64_hot);  free(data64_cold);
    free(idx32_same);  free(idx32_lin);  free(idx32_rand_h); free(idx32_rand_c);
    free(idx64_same);  free(idx64_lin);  free(idx64_rand_h); free(idx64_rand_c);
    free(idx32_lin_c); free(idx64_lin_c);
    free(idx32_rand_h64); free(idx64_rand_h64);
    free(idx32_rand_c64); free(idx64_rand_c64);
    free(idx32_lin_h64);  free(idx64_lin_h64);
    free(idx32_lin_c64);  free(idx64_lin_c64);

    return 0;
}