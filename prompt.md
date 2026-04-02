# ARM SVE Load/Store Benchmark Prompt

## Instruction Equivalence

The following instruction sequences are functionally equivalent:

### ld1d-based implementation
```
ld1d z3.d, p0/z, [x1]
ld1d z4.d, p0/z, [x1, #1, mul vl]
uzp1 z1.d, z3.d, z4.d
uzp2 z2.d, z3.d, z4.d
```

### ld2d-based implementation
```
ld2d { z1.d, z2.d }, p0/z, [x1]
```

### st2d-based implementation
```
st2d { z1.d, z2.d }, p0, [x1]
```

### st1d-based implementation
```
zip1 z3.d, z0.d, z1.d
zip2 z4.d, z0.d, z1.d
st1d z3.d, p0, [x1]
st1d z4.d, p0, [x1, #1, mul vl]
```

---

## Task

Write a high-performance microbenchmark in C to compare `1d` vs `2d` variants.

---

## Benchmark Scenarios

Measure:
1. Load only
2. Store only
3. Load + Store combined

---

## Access Patterns

Implement three modes:
1. Same address repeatedly
2. Linear access over an array
3. Random access over an array  
   - Random generation must use the fastest, minimal assembly instructions (non-cryptographic)

---

## Performance Metrics

- End-to-end execution time
- Effective bandwidth

---

## Constraints

- Array must fit entirely in L1 data cache  
  - L1d: 64 KB  
  - Cache line: 64 bytes  

- Loop unrolling factor: 16  
- Avoid:
  - Function calls inside hot loops  
  - Extra instructions in the benchmark path  

- Keep benchmark assembly-dominant and low-overhead

---

## Output Requirements

- Single file: `benchmark.c`
- Use inline SVE assembly where appropriate
- Print results to CLI
- Generate CSV file:
  ```
  result.csv
  ```

---

## Build & Run

Use clang-22.

Provide bash commands to:
1. Compile
2. Run

---

## Notes

- Ensure fair comparison between ld1d/st1d and ld2d/st2d
- Keep instruction count and register pressure comparable
- Focus on isolating instruction performance differences
