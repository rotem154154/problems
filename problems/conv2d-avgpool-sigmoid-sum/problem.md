---
slug: "conv2d-avgpool-sigmoid-sum"
title: "Conv2D AvgPool Sigmoid Sum"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "pooling", "activation"]
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

  - name: "pool_kernel"
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

  - name: "height"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "width"
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

Perform Conv2D, AvgPool, sigmoid, and sum over all spatial dimensions.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `kernel` of shape $(C_{out}, C_{in}, K_h, K_w)$
- Tensor `bias` of shape $(C_{out})$

## Output
- Tensor `output` of shape $(B)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/65_Conv2d_AvgPool_Sigmoid_Sum.py)
