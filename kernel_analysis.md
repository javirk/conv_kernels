# Best Kernel by Channel Count

Benchmark run: 2026-03-14, RTX A6000 (Ampere SM 8.6)

| Channels | Best Kernel | Resolution dependent? |
|----------|-------------|----------------------|
| 8 | 04_MoreOC | Slightly (02_FilterReg at 128) |
| 16 | 38_NHWC_FP16Vec8 | No |
| 32 | 39_NHWC_FP16_BK32 | Mostly no (38 at 768) |
| 64+ | 40_NHWC_FP16_BK64 | Mostly no (43 at 768) |

With CUTLASS TF32, all custom kernels beat CUTLASS across the board — up to 3.6x faster at C=256.

## CUTLASS FP16 comparison (2026-03-14)

After switching CUTLASS to FP16, the gap narrows significantly at mid-range channels but custom kernels still win at low and high channels.

Best custom kernel vs CUTLASS FP16 at 1024x1024:

| Channels | Best Custom | Custom (ms) | CUTLASS FP16 (ms) | Speedup |
|----------|-------------|-------------|--------------------|---------|
| 8 | 05_VecInput | 0.30 | 1.48 | 4.9x |
| 16 | 38_FP16Vec8 | 0.44 | 1.45 | 3.3x |
| 32 | 39_FP16_BK32 | 0.76 | 1.50 | 2.0x |
| 64 | 40_FP16_BK64 | 1.30 | 1.50 | 1.15x |
| 128 | 40_FP16_BK64 | 2.57 | 2.71 | 1.06x |
| 256 | 40_FP16_BK64 | 5.12 | 10.10 | 2.0x |

Key findings:
- C=8-32: Custom kernels still dominate (2-5x faster). CUTLASS FP16 actually got *slower* than CUTLASS TF32 at low channels — the FP16 tile config has higher overhead for small problems.
- C=64-128: Near parity. CUTLASS FP16 is even slightly faster at some resolutions (e.g., C=128 at 128x128: CUTLASS 0.053ms vs custom 0.059ms). The generic implicit GEMM becomes competitive when compute dominates and K is large enough.
- C=256: Custom kernels still 1.5-2x faster. The 3x3-aware tiling and BK=64 single-tile strategy still outperform CUTLASS's generic K-tiling at this scale.
