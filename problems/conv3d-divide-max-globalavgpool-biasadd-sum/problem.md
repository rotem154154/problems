---
slug: "conv3d-divide-max-globalavgpool-biasadd-sum"
title: "Conv3D Divide Max GlobalAvgPool BiasAdd Sum"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "pooling", "reduction"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

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

  - name: "divisor"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "pool_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "sum_dim"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform Conv3D, divide by a constant, max pool, global average pool, bias add, and sum.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `kernel` of shape $(C_{out}, C_{in}, K_d, K_h, K_w)$
- Tensor `bias` of shape $(C_{out}, 1, 1, 1)$

## Output
- Tensor `output` after global pooling and summation

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py)
