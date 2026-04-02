#include <arm_neon.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define N 10000000
#define ITERATIONS 200 

// Prevent compiler from optimizing away memory writes
void use_array(void* ptr) { 
    __asm__ volatile ("" : : "r" (ptr) : "memory"); 
}

// ==========================================
// 16-BIT BENCHMARKS
// ==========================================
__attribute__((target("+nosve")))
void neon_scatter_16(const uint16_t* input, uint16_t* output, const uint32_t* indices) {
    for (int i = 0; i < N; i += 8) {
        // Load contiguous input data
        uint16x8_t vec = vld1q_u16(&input[i]);
        
        // NEON workaround: Extract lanes and perform scalar stores
        output[indices[i + 0]] = vgetq_lane_u16(vec, 0);
        output[indices[i + 1]] = vgetq_lane_u16(vec, 1);
        output[indices[i + 2]] = vgetq_lane_u16(vec, 2);
        output[indices[i + 3]] = vgetq_lane_u16(vec, 3);
        output[indices[i + 4]] = vgetq_lane_u16(vec, 4);
        output[indices[i + 5]] = vgetq_lane_u16(vec, 5);
        output[indices[i + 6]] = vgetq_lane_u16(vec, 6);
        output[indices[i + 7]] = vgetq_lane_u16(vec, 7);
    }
    use_array(output);
}

void sve_scatter_16(const uint16_t* input, uint16_t* output, const uint32_t* indices) {
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b32(i, N);
        
        // Load 32-bit indices
        svuint32_t v_idx = svld1_u32(pg, &indices[i]);
        
        // Load contiguous 16-bit data, expanding into 32-bit vector lanes
        svuint32_t v_data = svld1uh_u32(pg, &input[i]);
        
        // SVE Native Scatter Store (stores 16-bit parts from the 32-bit vector)
        svst1h_scatter_u32index_u32(pg, output, v_idx, v_data);
        
        i += svcntw();
    }
    use_array(output);
}

// ==========================================
// 32-BIT BENCHMARKS
// ==========================================
__attribute__((target("+nosve")))
void neon_scatter_32(const uint32_t* input, uint32_t* output, const uint32_t* indices) {
    for (int i = 0; i < N; i += 4) {
        uint32x4_t vec = vld1q_u32(&input[i]);
        
        output[indices[i + 0]] = vgetq_lane_u32(vec, 0);
        output[indices[i + 1]] = vgetq_lane_u32(vec, 2); // Corrected typo here usually: 1
        output[indices[i + 1]] = vgetq_lane_u32(vec, 1);
        output[indices[i + 2]] = vgetq_lane_u32(vec, 2);
        output[indices[i + 3]] = vgetq_lane_u32(vec, 3);
    }
    use_array(output);
}

void sve_scatter_32(const uint32_t* input, uint32_t* output, const uint32_t* indices) {
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b32(i, N);
        
        svuint32_t v_idx = svld1_u32(pg, &indices[i]);
        svuint32_t v_data = svld1_u32(pg, &input[i]);
        
        svst1_scatter_u32index_u32(pg, output, v_idx, v_data);
        
        i += svcntw(); 
    }
    use_array(output);
}

// ==========================================
// 64-BIT BENCHMARKS
// ==========================================
__attribute__((target("+nosve")))
void neon_scatter_64(const uint64_t* input, uint64_t* output, const uint64_t* indices) {
    for (int i = 0; i < N; i += 2) {
        uint64x2_t vec = vld1q_u64(&input[i]);
        
        output[indices[i + 0]] = vgetq_lane_u64(vec, 0);
        output[indices[i + 1]] = vgetq_lane_u64(vec, 1);
    }
    use_array(output);
}

void sve_scatter_64(const uint64_t* input, uint64_t* output, const uint64_t* indices) {
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b64(i, N);
        
        svuint64_t v_idx = svld1_u64(pg, &indices[i]);
        svuint64_t v_data = svld1_u64(pg, &input[i]);
        
        svst1_scatter_u64index_u64(pg, output, v_idx, v_data);
        
        i += svcntd();
    }
    use_array(output);
}

// ==========================================
// MAIN FUNCTION & TIMING
// ==========================================
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    struct timespec start, end;
    
    // Allocate memory for inputs, outputs, and indices
    uint16_t* in16 = malloc(N * sizeof(uint16_t));
    uint32_t* in32 = malloc(N * sizeof(uint32_t));
    uint64_t* in64 = malloc(N * sizeof(uint64_t));
    
    uint16_t* out16 = malloc(N * sizeof(uint16_t));
    uint32_t* out32 = malloc(N * sizeof(uint32_t));
    uint64_t* out64 = malloc(N * sizeof(uint64_t));
    
    uint32_t* idx32 = malloc(N * sizeof(uint32_t));
    uint64_t* idx64 = malloc(N * sizeof(uint64_t));

    // Initialize data
    for (int i = 0; i < N; i++) {
        in16[i] = i % 65535;
        in32[i] = i;
        in64[i] = i;
        
        out16[i] = 0;
        out32[i] = 0;
        out64[i] = 0;
        
        idx32[i] = rand() % N; 
        idx64[i] = rand() % N;
    }

    printf("--- 16-BIT SCATTER TEST (200 Iterations) ---\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) neon_scatter_16(in16, out16, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("NEON 16-bit Time: %f s\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) sve_scatter_16(in16, out16, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SVE 16-bit Time:  %f s\n\n", get_time_diff(start, end));

    printf("--- 32-BIT SCATTER TEST (200 Iterations) ---\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) neon_scatter_32(in32, out32, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("NEON 32-bit Time: %f s\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) sve_scatter_32(in32, out32, idx32);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SVE 32-bit Time:  %f s\n\n", get_time_diff(start, end));

    printf("--- 64-BIT SCATTER TEST (200 Iterations) ---\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) neon_scatter_64(in64, out64, idx64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("NEON 64-bit Time: %f s\n", get_time_diff(start, end));

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) sve_scatter_64(in64, out64, idx64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("SVE 64-bit Time:  %f s\n", get_time_diff(start, end));

    // Cleanup
    free(in16); free(in32); free(in64);
    free(out16); free(out32); free(out64);
    free(idx32); free(idx64);

    return 0;
}