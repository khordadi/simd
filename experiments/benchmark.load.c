#include <arm_neon.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define N 10000000
#define ITERATIONS 1000

// Prevent compiler optimization
void use_data32(uint32_t sum) { __asm__ volatile ("" : : "r" (sum) : "memory"); }
void use_data64(uint64_t sum) { __asm__ volatile ("" : : "r" (sum) : "memory"); }

// ==========================================
// 16-BIT BENCHMARKS
// ==========================================
__attribute__((target("+nosve")))
void neon_gather_16(const uint16_t* data, const uint32_t* indices) {
    uint16_t sum = 0;
    for (int i = 0; i < N; i += 8) {
        uint16x8_t vec;
        vec = vsetq_lane_u16(data[indices[i + 0]], vec, 0);
        vec = vsetq_lane_u16(data[indices[i + 1]], vec, 1);
        vec = vsetq_lane_u16(data[indices[i + 2]], vec, 2);
        vec = vsetq_lane_u16(data[indices[i + 3]], vec, 3);
        vec = vsetq_lane_u16(data[indices[i + 4]], vec, 4);
        vec = vsetq_lane_u16(data[indices[i + 5]], vec, 5);
        vec = vsetq_lane_u16(data[indices[i + 6]], vec, 6);
        vec = vsetq_lane_u16(data[indices[i + 7]], vec, 7);
        sum += vaddvq_u16(vec);
    }
    use_data32(sum);
}

void sve_gather_16(const uint16_t* data, const uint32_t* indices) {
    uint32_t sum = 0;
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b32(i, N);
        svuint32_t v_idx = svld1_u32(pg, &indices[i]);
        svuint32_t v_data = svld1uh_gather_u32index_u32(pg, data, v_idx);
        sum += svaddv_u32(pg, v_data);
        i += svcntw();
    }
    use_data32(sum);
}

// ==========================================
// 32-BIT BENCHMARKS
// ==========================================
__attribute__((target("+nosve")))
void neon_gather_32(const uint32_t* data, const uint32_t* indices) {
    uint32_t sum = 0;
    for (int i = 0; i < N; i += 4) {
        uint32x4_t vec;
        vec = vsetq_lane_u32(data[indices[i + 0]], vec, 0);
        vec = vsetq_lane_u32(data[indices[i + 1]], vec, 1);
        vec = vsetq_lane_u32(data[indices[i + 2]], vec, 2);
        vec = vsetq_lane_u32(data[indices[i + 3]], vec, 3);
        sum += vaddvq_u32(vec);
    }
    use_data32(sum);
}

void sve_gather_32(const uint32_t* data, const uint32_t* indices) {
    uint32_t sum = 0;
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b32(i, N);
        svuint32_t v_idx = svld1_u32(pg, &indices[i]);
        svuint32_t v_data = svld1_gather_u32index_u32(pg, data, v_idx);
        sum += svaddv_u32(pg, v_data);
        i += svcntw(); 
    }
    use_data32(sum);
}

// ==========================================
// 64-BIT BENCHMARKS
// ==========================================
__attribute__((target("+nosve")))
void neon_gather_64(const uint64_t* data, const uint64_t* indices) {
    uint64_t sum = 0;
    for (int i = 0; i < N; i += 2) {
        uint64x2_t vec;
        vec = vsetq_lane_u64(data[indices[i + 0]], vec, 0);
        vec = vsetq_lane_u64(data[indices[i + 1]], vec, 1);
        sum += vaddvq_u64(vec);
    }
    use_data64(sum);
}

void sve_gather_64(const uint64_t* data, const uint64_t* indices) {
    uint64_t sum = 0;
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b64(i, N);
        svuint64_t v_idx = svld1_u64(pg, &indices[i]);
        svuint64_t v_data = svld1_gather_u64index_u64(pg, data, v_idx);
        sum += svaddv_u64(pg, v_data);
        i += svcntd();
    }
    use_data64(sum);
}

// ==========================================
// MAIN FUNCTION & TIMING
// ==========================================
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    struct timespec start, end;
    
    // Allocate memory
    uint16_t* data16 = malloc(N * sizeof(uint16_t));
    uint32_t* data32 = malloc(N * sizeof(uint32_t));
    uint64_t* data64 = malloc(N * sizeof(uint64_t));
    
    uint32_t* idx32 = malloc(N * sizeof(uint32_t));
    uint64_t* idx64 = malloc(N * sizeof(uint64_t));

    // Initialize data
    for (int i = 0; i < N; i++) {
        data16[i] = i % 65535;
        data32[i] = i;
        data64[i] = i;
        idx32[i] = rand() % N; 
        idx64[i] = rand() % N;
    }

    printf("--- 16-BIT TEST ---\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) neon_gather_16(data16, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("NEON 16-bit Time: %f s\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) sve_gather_16(data16, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SVE 16-bit Time:  %f s\n\n", get_time_diff(start, end));

    printf("--- 32-BIT TEST ---\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) neon_gather_32(data32, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("NEON 32-bit Time: %f s\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) sve_gather_32(data32, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SVE 32-bit Time:  %f s\n\n", get_time_diff(start, end));

    printf("--- 64-BIT TEST ---\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) neon_gather_64(data64, idx64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("NEON 64-bit Time: %f s\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) sve_gather_64(data64, idx64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SVE 64-bit Time:  %f s\n", get_time_diff(start, end));

    // Cleanup
    free(data16); free(data32); free(data64);
    free(idx32); free(idx64);

    return 0;
}