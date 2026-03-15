#pragma once

// Unified 3x3 Conv2D dispatcher.
//
// Selects the best kernel based on C_in (which dictates BK alignment):
//   C_in < 16  -> kernel 04 (scalar, NCHW, any channel count)
//   C_in < 32  -> kernel 38 (FP16 WMMA BK=16, best occupancy for small C_in)
//   C_in < 64  -> kernel 39 (FP16 WMMA BK=32, double-buffered)
//   C_in == 64 -> kernel 40 (FP16 WMMA BK=64, single ic tile, no ic loop)
//   C_in > 64  -> kernel 39 (FP16 WMMA BK=32, has ic loop for arbitrary C_in)
//
// Interface:
//   Input:  [C_in, H, W]       (NCHW, float32)
//   Filter: [C_out, C_in, 3, 3] (NCHW, float32)
//   Output: [C_out, out_H, out_W] (NCHW, float32)
//   No padding, stride 1. out_H = H-2, out_W = W-2.

#include "helpers.cuh"
#include "kernel_04_more_oc.cuh"
#include "kernel_38_fp16_vec8.cuh"
#include "kernel_39_fp16_bk32.cuh"
#include "kernel_40_fp16_bk64.cuh"

inline void launch_conv2d_3x3(float* d_output, float const* d_input, float const* d_filter,
                               size_t C_in, size_t C_out, size_t H, size_t W,
                               cudaStream_t stream)
{
    if (C_in < 16 || C_out < 16) {
        launch_conv2d_04(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    } else if (C_in < 32) {
        launch_conv2d_38(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    } else if (C_in < 64) {
        launch_conv2d_39(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    } else if (C_in == 64 && C_out >= 16) {
        // Kernel 40 has no ic loop — only correct when C_in fits in BK=64
        launch_conv2d_40(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    } else {
        // C_in > 64: fall back to kernel 39 which has an ic loop. This is suboptimal
        launch_conv2d_39(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    }
}
