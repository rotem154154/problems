---
slug: "conv-transpose2d-maxpool-hardtanh-mean-tanh"
title: "ConvTranspose2D MaxPool Hardtanh Mean Tanh"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "pooling", "tanh"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel"
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

  - name: "stride_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "stride_w"
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

  - name: "output_padding_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "output_padding_w"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "maxpool_kernel"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "maxpool_stride"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "hardtanh_min"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "hardtanh_max"
    type: "double"
    pointer: "false"
    constant: "false"
---

Perform ConvTranspose2D, max pool, Hardtanh, mean over spatial dims, and tanh.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_h, K_w)$

## Output
- Tensor `output` of shape $(B, C_{out}, 1, 1)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py)
