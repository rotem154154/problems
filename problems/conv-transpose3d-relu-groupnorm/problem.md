---
slug: "conv-transpose3d-relu-groupnorm"
title: "ConvTranspose3D ReLU GroupNorm"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "normalization", "activation"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "gn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "gn_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "groups"
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
---

Perform ConvTranspose3D, ReLU, and GroupNorm.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_d, K_h, K_w)$
- GroupNorm weights/biases of shape $(C_{out})$

## Output
- Tensor `output` after GroupNorm

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/61_ConvTranspose3d_ReLU_GroupNorm.py)
