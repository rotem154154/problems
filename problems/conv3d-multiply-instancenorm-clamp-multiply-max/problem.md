---
slug: "conv3d-multiply-instancenorm-clamp-multiply-max"
title: "Conv3D Multiply InstanceNorm Clamp Multiply Max"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "normalization", "reduction"]
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

  - name: "multiplier"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "clamp_min"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "clamp_max"
    type: "double"
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
---

Perform Conv3D, multiply, InstanceNorm, clamp, multiply, and max over channels.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `kernel` of shape $(C_{out}, C_{in}, K_d, K_h, K_w)$
- Tensor `conv_bias` of shape $(C_{out})$
- Tensor `multiplier` of shape $(C_{out}, 1, 1, 1)$

## Output
- Tensor `output` of shape $(B, D, H, W)$

## Notes
- InstanceNorm uses input stats without affine parameters.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max.py)
