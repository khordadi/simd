/*
 * SVE ld1d+uzp vs ld2d / st1d+zip vs st2d benchmark
 *
 * Modes:  same-address, linear, random
 * Ops:    load-only, store-only, load+store
 * Variants: 1d (ld1d/st1d + uzp/zip), 2d (ld2d/st2d)
 *
 * Compile: clang-22 -O2 -march=native -o benchmark benchmark.c
 * Run:     ./benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ---------- configuration ---------- */
#define ARRAY_BYTES    (32u * 1024u)           /* 32 KB – fits L1d */
#define NUM_DOUBLES    (ARRAY_BYTES / 8)       /* 4096 doubles     */
#define ITERS          10000000ULL             /* outer-loop iters */
#define WARMUP_ITERS   10000ULL
#define RAND_MASK      0x7FFULL                /* doubles 0-2047, safe for any VL≤512b */
#define MAX_RESULTS    32

/* ---------- helpers ---------- */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    const char *mode;
    const char *op;
    const char *variant;
    uint64_t    iters;
    uint64_t    bytes;
    double      time_s;
    double      bw_gbps;
} result_t;

/* ---------- clobber lists ---------- */
#define Z_CLOBBER \
    "z0","z1","z2","z3","z4","z5","z6","z7",       \
    "z8","z9","z10","z11","z12","z13","z14","z15",  \
    "z16","z17","z18","z19","z20","z21","z22","z23",\
    "z24","z25","z26","z27","z28","z29","z30","z31"

#define COMMON_CLOBBER "memory","cc","p0","x3","x4","x5","x6","x7","x8", Z_CLOBBER

/* ================================================================
 *  SAME-ADDRESS macros  (16× unroll, no pointer arithmetic)
 * ================================================================ */

/* -- load -- */
#define LD2D_SAME  "ld2d {z0.d, z1.d}, p0/z, [%[a]]\n"
#define LD1D_SAME  \
    "ld1d z0.d, p0/z, [%[a]]\n"            \
    "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n"\
    "uzp1 z2.d, z0.d, z1.d\n"              \
    "uzp2 z3.d, z0.d, z1.d\n"

/* -- store -- */
#define ST2D_SAME  "st2d {z0.d, z1.d}, p0, [%[a]]\n"
#define ST1D_SAME  \
    "zip1 z2.d, z0.d, z1.d\n"              \
    "zip2 z3.d, z0.d, z1.d\n"              \
    "st1d z2.d, p0, [%[a]]\n"              \
    "st1d z3.d, p0, [%[a], #1, mul vl]\n"

/* -- load+store -- */
#define LDST2D_SAME \
    "ld2d {z0.d, z1.d}, p0/z, [%[a]]\n"    \
    "st2d {z0.d, z1.d}, p0, [%[a]]\n"
#define LDST1D_SAME \
    "ld1d z0.d, p0/z, [%[a]]\n"            \
    "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n"\
    "uzp1 z2.d, z0.d, z1.d\n"              \
    "uzp2 z3.d, z0.d, z1.d\n"              \
    "zip1 z4.d, z2.d, z3.d\n"              \
    "zip2 z5.d, z2.d, z3.d\n"              \
    "st1d z4.d, p0, [%[a]]\n"              \
    "st1d z5.d, p0, [%[a], #1, mul vl]\n"

#define R16(x) x x x x x x x x x x x x x x x x

/* ================================================================
 *  LINEAR macros  (16× unroll with mul-vl offsets, batched)
 *  ld2d batch = 8 loads (offsets #0,#2,...,#14), stride = 16·VL
 *  ld1d batch = 4 pairs (offsets #0..#7),       stride =  8·VL
 * ================================================================ */

/* ---- ld2d linear: 2 batches of 8 = 16 loads ---- */
#define LD2D_LIN_B8 \
    "ld2d {z0.d,  z1.d},  p0/z, [x3, #0,  mul vl]\n" \
    "ld2d {z2.d,  z3.d},  p0/z, [x3, #2,  mul vl]\n" \
    "ld2d {z4.d,  z5.d},  p0/z, [x3, #4,  mul vl]\n" \
    "ld2d {z6.d,  z7.d},  p0/z, [x3, #6,  mul vl]\n" \
    "ld2d {z8.d,  z9.d},  p0/z, [x3, #8,  mul vl]\n" \
    "ld2d {z10.d, z11.d}, p0/z, [x3, #10, mul vl]\n"  \
    "ld2d {z12.d, z13.d}, p0/z, [x3, #12, mul vl]\n"  \
    "ld2d {z14.d, z15.d}, p0/z, [x3, #14, mul vl]\n"  \
    "add x3, x3, x6\n"

/* ---- ld1d linear: 4 batches of 4 pairs = 16 pairs ---- */
#define LD1D_LIN_B4 \
    "ld1d z0.d, p0/z, [x3, #0, mul vl]\n" \
    "ld1d z1.d, p0/z, [x3, #1, mul vl]\n" \
    "uzp1 z16.d, z0.d, z1.d\n"            \
    "uzp2 z17.d, z0.d, z1.d\n"            \
    "ld1d z2.d, p0/z, [x3, #2, mul vl]\n" \
    "ld1d z3.d, p0/z, [x3, #3, mul vl]\n" \
    "uzp1 z18.d, z2.d, z3.d\n"            \
    "uzp2 z19.d, z2.d, z3.d\n"            \
    "ld1d z4.d, p0/z, [x3, #4, mul vl]\n" \
    "ld1d z5.d, p0/z, [x3, #5, mul vl]\n" \
    "uzp1 z20.d, z4.d, z5.d\n"            \
    "uzp2 z21.d, z4.d, z5.d\n"            \
    "ld1d z6.d, p0/z, [x3, #6, mul vl]\n" \
    "ld1d z7.d, p0/z, [x3, #7, mul vl]\n" \
    "uzp1 z22.d, z6.d, z7.d\n"            \
    "uzp2 z23.d, z6.d, z7.d\n"            \
    "add x3, x3, x6\n"

/* ---- st2d linear: 2 batches of 8 = 16 stores ---- */
#define ST2D_LIN_B8 \
    "st2d {z0.d,  z1.d},  p0, [x3, #0,  mul vl]\n" \
    "st2d {z0.d,  z1.d},  p0, [x3, #2,  mul vl]\n" \
    "st2d {z0.d,  z1.d},  p0, [x3, #4,  mul vl]\n" \
    "st2d {z0.d,  z1.d},  p0, [x3, #6,  mul vl]\n" \
    "st2d {z0.d,  z1.d},  p0, [x3, #8,  mul vl]\n" \
    "st2d {z0.d,  z1.d},  p0, [x3, #10, mul vl]\n"  \
    "st2d {z0.d,  z1.d},  p0, [x3, #12, mul vl]\n"  \
    "st2d {z0.d,  z1.d},  p0, [x3, #14, mul vl]\n"  \
    "add x3, x3, x6\n"

/* ---- st1d linear: 4 batches of 4 pairs = 16 stores ---- */
#define ST1D_LIN_B4 \
    "zip1 z2.d, z0.d, z1.d\n"             \
    "zip2 z3.d, z0.d, z1.d\n"             \
    "st1d z2.d, p0, [x3, #0, mul vl]\n"   \
    "st1d z3.d, p0, [x3, #1, mul vl]\n"   \
    "zip1 z4.d, z0.d, z1.d\n"             \
    "zip2 z5.d, z0.d, z1.d\n"             \
    "st1d z4.d, p0, [x3, #2, mul vl]\n"   \
    "st1d z5.d, p0, [x3, #3, mul vl]\n"   \
    "zip1 z6.d, z0.d, z1.d\n"             \
    "zip2 z7.d, z0.d, z1.d\n"             \
    "st1d z6.d, p0, [x3, #4, mul vl]\n"   \
    "st1d z7.d, p0, [x3, #5, mul vl]\n"   \
    "zip1 z8.d, z0.d, z1.d\n"             \
    "zip2 z9.d, z0.d, z1.d\n"             \
    "st1d z8.d, p0, [x3, #6, mul vl]\n"   \
    "st1d z9.d, p0, [x3, #7, mul vl]\n"   \
    "add x3, x3, x6\n"

/* ---- ld2d+st2d linear: 2 batches of 8 = 16 load+store ---- */
#define LDST2D_LIN_B8 \
    "ld2d {z0.d,  z1.d},  p0/z, [x3, #0,  mul vl]\n" \
    "st2d {z0.d,  z1.d},  p0,   [x3, #0,  mul vl]\n" \
    "ld2d {z2.d,  z3.d},  p0/z, [x3, #2,  mul vl]\n" \
    "st2d {z2.d,  z3.d},  p0,   [x3, #2,  mul vl]\n" \
    "ld2d {z4.d,  z5.d},  p0/z, [x3, #4,  mul vl]\n" \
    "st2d {z4.d,  z5.d},  p0,   [x3, #4,  mul vl]\n" \
    "ld2d {z6.d,  z7.d},  p0/z, [x3, #6,  mul vl]\n" \
    "st2d {z6.d,  z7.d},  p0,   [x3, #6,  mul vl]\n" \
    "ld2d {z8.d,  z9.d},  p0/z, [x3, #8,  mul vl]\n" \
    "st2d {z8.d,  z9.d},  p0,   [x3, #8,  mul vl]\n" \
    "ld2d {z10.d, z11.d}, p0/z, [x3, #10, mul vl]\n"  \
    "st2d {z10.d, z11.d}, p0,   [x3, #10, mul vl]\n"  \
    "ld2d {z12.d, z13.d}, p0/z, [x3, #12, mul vl]\n"  \
    "st2d {z12.d, z13.d}, p0,   [x3, #12, mul vl]\n"  \
    "ld2d {z14.d, z15.d}, p0/z, [x3, #14, mul vl]\n"  \
    "st2d {z14.d, z15.d}, p0,   [x3, #14, mul vl]\n"  \
    "add x3, x3, x6\n"

/* ---- ld1d+st1d linear: 4 batches of 4 pairs = 16 ld+st ---- */
#define LDST1D_LIN_B4 \
    "ld1d z0.d, p0/z, [x3, #0, mul vl]\n"   \
    "ld1d z1.d, p0/z, [x3, #1, mul vl]\n"   \
    "uzp1 z2.d, z0.d, z1.d\n"               \
    "uzp2 z3.d, z0.d, z1.d\n"               \
    "zip1 z4.d, z2.d, z3.d\n"               \
    "zip2 z5.d, z2.d, z3.d\n"               \
    "st1d z4.d, p0, [x3, #0, mul vl]\n"     \
    "st1d z5.d, p0, [x3, #1, mul vl]\n"     \
    "ld1d z0.d, p0/z, [x3, #2, mul vl]\n"   \
    "ld1d z1.d, p0/z, [x3, #3, mul vl]\n"   \
    "uzp1 z2.d, z0.d, z1.d\n"               \
    "uzp2 z3.d, z0.d, z1.d\n"               \
    "zip1 z4.d, z2.d, z3.d\n"               \
    "zip2 z5.d, z2.d, z3.d\n"               \
    "st1d z4.d, p0, [x3, #2, mul vl]\n"     \
    "st1d z5.d, p0, [x3, #3, mul vl]\n"     \
    "ld1d z0.d, p0/z, [x3, #4, mul vl]\n"   \
    "ld1d z1.d, p0/z, [x3, #5, mul vl]\n"   \
    "uzp1 z2.d, z0.d, z1.d\n"               \
    "uzp2 z3.d, z0.d, z1.d\n"               \
    "zip1 z4.d, z2.d, z3.d\n"               \
    "zip2 z5.d, z2.d, z3.d\n"               \
    "st1d z4.d, p0, [x3, #4, mul vl]\n"     \
    "st1d z5.d, p0, [x3, #5, mul vl]\n"     \
    "ld1d z0.d, p0/z, [x3, #6, mul vl]\n"   \
    "ld1d z1.d, p0/z, [x3, #7, mul vl]\n"   \
    "uzp1 z2.d, z0.d, z1.d\n"               \
    "uzp2 z3.d, z0.d, z1.d\n"               \
    "zip1 z4.d, z2.d, z3.d\n"               \
    "zip2 z5.d, z2.d, z3.d\n"               \
    "st1d z4.d, p0, [x3, #6, mul vl]\n"     \
    "st1d z5.d, p0, [x3, #7, mul vl]\n"     \
    "add x3, x3, x6\n"

/* linear loop skeleton: init x3=arr, x6=batch_stride, x7=arr_end; wrap x3 */
#define LIN_LOOP_PRE \
    "mov x3, %[a]\n"   \
    "mov x6, %[stride]\n" \
    "mov x7, %[end]\n" \
    "ptrue p0.d\n"      \
    "1:\n"

#define LIN_WRAP_AND_BRANCH \
    "cmp x3, x7\n"       \
    "csel x3, %[a], x3, hs\n" \
    "subs %[n], %[n], #1\n" \
    "bne 1b\n"

/* ================================================================
 *  RANDOM macros (xorshift64 in x4, address in x8)
 * ================================================================ */

#define XORSHIFT \
    "eor x4, x4, x4, lsl #13\n" \
    "eor x4, x4, x4, lsr #7\n"  \
    "eor x4, x4, x4, lsl #17\n" \
    "and x8, x4, %[mask]\n"

/* ld2d uses [Xn, Xm, LSL #3] so x8 is a doubleword offset */
#define LD2D_RAND \
    XORSHIFT \
    "ld2d {z0.d, z1.d}, p0/z, [%[a], x8, lsl #3]\n"

/* ld1d needs a byte-address base for #N,mul vl offsets */
#define LD1D_RAND \
    XORSHIFT \
    "add x5, %[a], x8, lsl #3\n"           \
    "ld1d z0.d, p0/z, [x5]\n"              \
    "ld1d z1.d, p0/z, [x5, #1, mul vl]\n"  \
    "uzp1 z2.d, z0.d, z1.d\n"              \
    "uzp2 z3.d, z0.d, z1.d\n"

#define ST2D_RAND \
    XORSHIFT \
    "st2d {z0.d, z1.d}, p0, [%[a], x8, lsl #3]\n"

#define ST1D_RAND \
    XORSHIFT \
    "add x5, %[a], x8, lsl #3\n"           \
    "zip1 z2.d, z0.d, z1.d\n"              \
    "zip2 z3.d, z0.d, z1.d\n"              \
    "st1d z2.d, p0, [x5]\n"                \
    "st1d z3.d, p0, [x5, #1, mul vl]\n"

#define LDST2D_RAND \
    XORSHIFT \
    "ld2d {z0.d, z1.d}, p0/z, [%[a], x8, lsl #3]\n" \
    "st2d {z0.d, z1.d}, p0, [%[a], x8, lsl #3]\n"

#define LDST1D_RAND \
    XORSHIFT \
    "add x5, %[a], x8, lsl #3\n"           \
    "ld1d z0.d, p0/z, [x5]\n"              \
    "ld1d z1.d, p0/z, [x5, #1, mul vl]\n"  \
    "uzp1 z2.d, z0.d, z1.d\n"              \
    "uzp2 z3.d, z0.d, z1.d\n"              \
    "zip1 z4.d, z2.d, z3.d\n"              \
    "zip2 z5.d, z2.d, z3.d\n"              \
    "st1d z4.d, p0, [x5]\n"                \
    "st1d z5.d, p0, [x5, #1, mul vl]\n"

#define RAND_LOOP_PRE \
    "mov x4, #0xBEEF\n" \
    "ptrue p0.d\n"       \
    "1:\n"

#define RAND_LOOP_END \
    "subs %[n], %[n], #1\n" \
    "bne 1b\n"

/* ================================================================
 *  Benchmark runner macro
 * ================================================================ */
#define RUN(MODE, OP, VAR, ASM_BODY, CONSTRAINTS_OUT, CONSTRAINTS_IN, BMUL) \
    do {                                                                    \
        /* warmup */                                                        \
        { uint64_t _w = WARMUP_ITERS;                                      \
          asm volatile(ASM_BODY : CONSTRAINTS_OUT : CONSTRAINTS_IN          \
                       : COMMON_CLOBBER); }                                 \
        uint64_t _n = ITERS;                                                \
        double _t0 = now_sec();                                             \
        asm volatile(ASM_BODY : CONSTRAINTS_OUT : CONSTRAINTS_IN            \
                     : COMMON_CLOBBER);                                     \
        double _t1 = now_sec();                                             \
        double _dt = _t1 - _t0;                                            \
        uint64_t _bytes = ITERS * 16ULL * 2 * vl_bytes * (BMUL);           \
        double _bw = (double)_bytes / _dt / 1e9;                           \
        results[ri++] = (result_t){MODE, OP, VAR, ITERS, _bytes, _dt, _bw};\
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n",               \
               MODE, OP, VAR, _dt*1e3, _bw);                               \
    } while(0)

/* ================================================================ */
int main(void)
{
    /* allocate & init */
    double *arr;
    if (posix_memalign((void**)&arr, 4096, ARRAY_BYTES)) {
        perror("posix_memalign"); return 1;
    }
    for (uint64_t i = 0; i < NUM_DOUBLES; i++) arr[i] = (double)i;

    uint64_t vl_bytes;
    asm volatile("cntb %0" : "=r"(vl_bytes));

    uint64_t stride_ld2d = 16 * vl_bytes;   /* bytes per batch-of-8 ld2d */
    uint64_t stride_ld1d =  8 * vl_bytes;   /* bytes per batch-of-4 ld1d-pairs */
    double  *arr_end     = (double*)((char*)arr + ARRAY_BYTES);
    uint64_t rand_mask   = RAND_MASK;

    result_t results[MAX_RESULTS];
    int ri = 0;

    printf("========================================\n");
    printf(" SVE ld1d+uzp / st1d+zip  vs  ld2d / st2d\n");
    printf("========================================\n");
    printf("SVE VL         = %lu bytes (%lu bits)\n", vl_bytes, vl_bytes*8);
    printf("Array          = %u bytes (%u doubles)\n", ARRAY_BYTES, NUM_DOUBLES);
    printf("Iterations     = %llu\n", (unsigned long long)ITERS);
    printf("Unroll         = 16\n\n");

    /* ==============================================================
     *  1. SAME ADDRESS
     * ============================================================== */
    printf("--- same address ---\n");

    /* load-only 2d */
    {
        uint64_t _n = ITERS;
        /* warmup */ { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                "ptrue p0.d\n1:\n" R16(LD2D_SAME)
                "subs %[n], %[n], #1\nbne 1b\n"
                : [n] "+r"(_w) : [a] "r"(arr) : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            "ptrue p0.d\n1:\n" R16(LD2D_SAME)
            "subs %[n], %[n], #1\nbne 1b\n"
            : [n] "+r"(_n) : [a] "r"(arr) : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"same","load","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","same","load","2d",dt*1e3,bytes/dt/1e9);
    }
    /* load-only 1d */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                "ptrue p0.d\n1:\n" R16(LD1D_SAME)
                "subs %[n], %[n], #1\nbne 1b\n"
                : [n] "+r"(_w) : [a] "r"(arr) : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            "ptrue p0.d\n1:\n" R16(LD1D_SAME)
            "subs %[n], %[n], #1\nbne 1b\n"
            : [n] "+r"(_n) : [a] "r"(arr) : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"same","load","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","same","load","1d",dt*1e3,bytes/dt/1e9);
    }
    /* store-only 2d */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                "ptrue p0.d\n1:\n" R16(ST2D_SAME)
                "subs %[n], %[n], #1\nbne 1b\n"
                : [n] "+r"(_w) : [a] "r"(arr) : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            "ptrue p0.d\n1:\n" R16(ST2D_SAME)
            "subs %[n], %[n], #1\nbne 1b\n"
            : [n] "+r"(_n) : [a] "r"(arr) : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"same","store","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","same","store","2d",dt*1e3,bytes/dt/1e9);
    }
    /* store-only 1d */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                "ptrue p0.d\n1:\n" R16(ST1D_SAME)
                "subs %[n], %[n], #1\nbne 1b\n"
                : [n] "+r"(_w) : [a] "r"(arr) : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            "ptrue p0.d\n1:\n" R16(ST1D_SAME)
            "subs %[n], %[n], #1\nbne 1b\n"
            : [n] "+r"(_n) : [a] "r"(arr) : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"same","store","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","same","store","1d",dt*1e3,bytes/dt/1e9);
    }
    /* load+store 2d */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                "ptrue p0.d\n1:\n" R16(LDST2D_SAME)
                "subs %[n], %[n], #1\nbne 1b\n"
                : [n] "+r"(_w) : [a] "r"(arr) : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            "ptrue p0.d\n1:\n" R16(LDST2D_SAME)
            "subs %[n], %[n], #1\nbne 1b\n"
            : [n] "+r"(_n) : [a] "r"(arr) : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes * 2;
        results[ri++] = (result_t){"same","load+store","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","same","load+store","2d",dt*1e3,bytes/dt/1e9);
    }
    /* load+store 1d */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                "ptrue p0.d\n1:\n" R16(LDST1D_SAME)
                "subs %[n], %[n], #1\nbne 1b\n"
                : [n] "+r"(_w) : [a] "r"(arr) : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            "ptrue p0.d\n1:\n" R16(LDST1D_SAME)
            "subs %[n], %[n], #1\nbne 1b\n"
            : [n] "+r"(_n) : [a] "r"(arr) : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes * 2;
        results[ri++] = (result_t){"same","load+store","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","same","load+store","1d",dt*1e3,bytes/dt/1e9);
    }

    /* ==============================================================
     *  2. LINEAR
     * ============================================================== */
    printf("\n--- linear ---\n");

    /* load-only 2d linear */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                LIN_LOOP_PRE
                LD2D_LIN_B8 LD2D_LIN_B8
                LIN_WRAP_AND_BRANCH
                : [n] "+r"(_w)
                : [a] "r"(arr), [stride] "r"(stride_ld2d), [end] "r"(arr_end)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            LIN_LOOP_PRE
            LD2D_LIN_B8 LD2D_LIN_B8
            LIN_WRAP_AND_BRANCH
            : [n] "+r"(_n)
            : [a] "r"(arr), [stride] "r"(stride_ld2d), [end] "r"(arr_end)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"linear","load","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","linear","load","2d",dt*1e3,bytes/dt/1e9);
    }
    /* load-only 1d linear */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                LIN_LOOP_PRE
                LD1D_LIN_B4 LD1D_LIN_B4 LD1D_LIN_B4 LD1D_LIN_B4
                LIN_WRAP_AND_BRANCH
                : [n] "+r"(_w)
                : [a] "r"(arr), [stride] "r"(stride_ld1d), [end] "r"(arr_end)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            LIN_LOOP_PRE
            LD1D_LIN_B4 LD1D_LIN_B4 LD1D_LIN_B4 LD1D_LIN_B4
            LIN_WRAP_AND_BRANCH
            : [n] "+r"(_n)
            : [a] "r"(arr), [stride] "r"(stride_ld1d), [end] "r"(arr_end)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"linear","load","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","linear","load","1d",dt*1e3,bytes/dt/1e9);
    }
    /* store-only 2d linear */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                LIN_LOOP_PRE
                ST2D_LIN_B8 ST2D_LIN_B8
                LIN_WRAP_AND_BRANCH
                : [n] "+r"(_w)
                : [a] "r"(arr), [stride] "r"(stride_ld2d), [end] "r"(arr_end)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            LIN_LOOP_PRE
            ST2D_LIN_B8 ST2D_LIN_B8
            LIN_WRAP_AND_BRANCH
            : [n] "+r"(_n)
            : [a] "r"(arr), [stride] "r"(stride_ld2d), [end] "r"(arr_end)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"linear","store","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","linear","store","2d",dt*1e3,bytes/dt/1e9);
    }
    /* store-only 1d linear */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                LIN_LOOP_PRE
                ST1D_LIN_B4 ST1D_LIN_B4 ST1D_LIN_B4 ST1D_LIN_B4
                LIN_WRAP_AND_BRANCH
                : [n] "+r"(_w)
                : [a] "r"(arr), [stride] "r"(stride_ld1d), [end] "r"(arr_end)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            LIN_LOOP_PRE
            ST1D_LIN_B4 ST1D_LIN_B4 ST1D_LIN_B4 ST1D_LIN_B4
            LIN_WRAP_AND_BRANCH
            : [n] "+r"(_n)
            : [a] "r"(arr), [stride] "r"(stride_ld1d), [end] "r"(arr_end)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"linear","store","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","linear","store","1d",dt*1e3,bytes/dt/1e9);
    }
    /* load+store 2d linear */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                LIN_LOOP_PRE
                LDST2D_LIN_B8 LDST2D_LIN_B8
                LIN_WRAP_AND_BRANCH
                : [n] "+r"(_w)
                : [a] "r"(arr), [stride] "r"(stride_ld2d), [end] "r"(arr_end)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            LIN_LOOP_PRE
            LDST2D_LIN_B8 LDST2D_LIN_B8
            LIN_WRAP_AND_BRANCH
            : [n] "+r"(_n)
            : [a] "r"(arr), [stride] "r"(stride_ld2d), [end] "r"(arr_end)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes * 2;
        results[ri++] = (result_t){"linear","load+store","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","linear","load+store","2d",dt*1e3,bytes/dt/1e9);
    }
    /* load+store 1d linear */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                LIN_LOOP_PRE
                LDST1D_LIN_B4 LDST1D_LIN_B4 LDST1D_LIN_B4 LDST1D_LIN_B4
                LIN_WRAP_AND_BRANCH
                : [n] "+r"(_w)
                : [a] "r"(arr), [stride] "r"(stride_ld1d), [end] "r"(arr_end)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            LIN_LOOP_PRE
            LDST1D_LIN_B4 LDST1D_LIN_B4 LDST1D_LIN_B4 LDST1D_LIN_B4
            LIN_WRAP_AND_BRANCH
            : [n] "+r"(_n)
            : [a] "r"(arr), [stride] "r"(stride_ld1d), [end] "r"(arr_end)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes * 2;
        results[ri++] = (result_t){"linear","load+store","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","linear","load+store","1d",dt*1e3,bytes/dt/1e9);
    }

    /* ==============================================================
     *  3. RANDOM
     * ============================================================== */
    printf("\n--- random ---\n");

    /* load-only 2d random */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                RAND_LOOP_PRE R16(LD2D_RAND) RAND_LOOP_END
                : [n] "+r"(_w)
                : [a] "r"(arr), [mask] "r"(rand_mask)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            RAND_LOOP_PRE R16(LD2D_RAND) RAND_LOOP_END
            : [n] "+r"(_n)
            : [a] "r"(arr), [mask] "r"(rand_mask)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"random","load","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","random","load","2d",dt*1e3,bytes/dt/1e9);
    }
    /* load-only 1d random */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                RAND_LOOP_PRE R16(LD1D_RAND) RAND_LOOP_END
                : [n] "+r"(_w)
                : [a] "r"(arr), [mask] "r"(rand_mask)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            RAND_LOOP_PRE R16(LD1D_RAND) RAND_LOOP_END
            : [n] "+r"(_n)
            : [a] "r"(arr), [mask] "r"(rand_mask)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"random","load","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","random","load","1d",dt*1e3,bytes/dt/1e9);
    }
    /* store-only 2d random */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                RAND_LOOP_PRE R16(ST2D_RAND) RAND_LOOP_END
                : [n] "+r"(_w)
                : [a] "r"(arr), [mask] "r"(rand_mask)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            RAND_LOOP_PRE R16(ST2D_RAND) RAND_LOOP_END
            : [n] "+r"(_n)
            : [a] "r"(arr), [mask] "r"(rand_mask)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"random","store","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","random","store","2d",dt*1e3,bytes/dt/1e9);
    }
    /* store-only 1d random */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                RAND_LOOP_PRE R16(ST1D_RAND) RAND_LOOP_END
                : [n] "+r"(_w)
                : [a] "r"(arr), [mask] "r"(rand_mask)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            RAND_LOOP_PRE R16(ST1D_RAND) RAND_LOOP_END
            : [n] "+r"(_n)
            : [a] "r"(arr), [mask] "r"(rand_mask)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes;
        results[ri++] = (result_t){"random","store","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","random","store","1d",dt*1e3,bytes/dt/1e9);
    }
    /* load+store 2d random */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                RAND_LOOP_PRE R16(LDST2D_RAND) RAND_LOOP_END
                : [n] "+r"(_w)
                : [a] "r"(arr), [mask] "r"(rand_mask)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            RAND_LOOP_PRE R16(LDST2D_RAND) RAND_LOOP_END
            : [n] "+r"(_n)
            : [a] "r"(arr), [mask] "r"(rand_mask)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes * 2;
        results[ri++] = (result_t){"random","load+store","2d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","random","load+store","2d",dt*1e3,bytes/dt/1e9);
    }
    /* load+store 1d random */
    {
        uint64_t _n = ITERS;
        { uint64_t _w = WARMUP_ITERS;
            asm volatile(
                RAND_LOOP_PRE R16(LDST1D_RAND) RAND_LOOP_END
                : [n] "+r"(_w)
                : [a] "r"(arr), [mask] "r"(rand_mask)
                : COMMON_CLOBBER); }
        double t0 = now_sec();
        asm volatile(
            RAND_LOOP_PRE R16(LDST1D_RAND) RAND_LOOP_END
            : [n] "+r"(_n)
            : [a] "r"(arr), [mask] "r"(rand_mask)
            : COMMON_CLOBBER);
        double dt = now_sec() - t0;
        uint64_t bytes = ITERS * 16ULL * 2 * vl_bytes * 2;
        results[ri++] = (result_t){"random","load+store","1d",ITERS,bytes,dt,bytes/dt/1e9};
        printf("  %-8s %-12s %-4s  %9.3f ms  %8.2f GB/s\n","random","load+store","1d",dt*1e3,bytes/dt/1e9);
    }

    /* ==============================================================
     *  Summary & CSV
     * ============================================================== */
    printf("\n========================================\n");
    printf(" Summary (ratio = time_1d / time_2d)\n");
    printf("========================================\n");
    printf("  %-8s %-12s  %9s  %9s  %6s\n", "mode", "op", "2d(ms)", "1d(ms)", "ratio");
    for (int i = 0; i < ri; i += 2) {
        double r = results[i+1].time_s / results[i].time_s;
        printf("  %-8s %-12s  %9.3f  %9.3f  %6.3f\n",
               results[i].mode, results[i].op,
               results[i].time_s*1e3, results[i+1].time_s*1e3, r);
    }

    /* write CSV */
    FILE *fp = fopen("result.csv", "w");
    if (fp) {
        fprintf(fp, "mode,op,variant,iterations,bytes,time_s,bandwidth_gbps\n");
        for (int i = 0; i < ri; i++) {
            fprintf(fp, "%s,%s,%s,%llu,%llu,%.9f,%.4f\n",
                    results[i].mode, results[i].op, results[i].variant,
                    (unsigned long long)results[i].iters,
                    (unsigned long long)results[i].bytes,
                    results[i].time_s, results[i].bw_gbps);
        }
        fclose(fp);
        printf("\nResults written to result.csv\n");
    } else {
        perror("fopen result.csv");
    }

    free(arr);
    return 0;
}