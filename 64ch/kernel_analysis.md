# Best Kernel by Channel Count

Benchmark run: 2026-03-14, RTX A6000 (Ampere SM 8.6), CUTLASS FP16

| Channels | Best Kernel | Custom (ms) @1024 | CUTLASS FP16 (ms) | Speedup | Resolution dependent? |
|----------|-------------|--------------------|--------------------|---------|----------------------|
| 8 | 05_VecInput | 0.29 | 1.44 | 5.0x | No |
| 16 | 38_NHWC_FP16Vec8 | 0.43 | 1.41 | 3.3x | No |
| 32 | 39_NHWC_FP16_BK32 | 0.74 | 1.45 | 2.0x | No |
| 64 | 40_NHWC_FP16_BK64 | 1.30 | 1.47 | 1.13x | No |
