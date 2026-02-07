---
slug: "conv-transpose3d-multiply-max-globalavgpool-clamp"
title: "ConvTranspose3D Multiply Max GlobalAvgPool Clamp"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "pooling"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "conv_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "scale"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "maxpool_kernel"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "batch_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "in_channels"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "out_channels"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "depth"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "height"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "width"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "kernel_d"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "kernel_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "kernel_w"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "stride_d"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "stride_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "stride_w"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "padding_d"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "padding_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "padding_w"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform ConvTranspose3D, scale, max pool, global average pool, and clamp.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_d, K_h, K_w)$
- Tensor `conv_bias` of shape $(C_{out})$

## Output
- Tensor `output` after clamp

## Notes
- Clamp is applied to $[0, 1]$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp.py)
