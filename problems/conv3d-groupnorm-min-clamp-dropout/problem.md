---
slug: "conv3d-groupnorm-min-clamp-dropout"
title: "Conv3D GroupNorm Min Clamp Dropout"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "normalization", "dropout"]
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

  - name: "num_groups"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "min_value"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "max_value"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "dropout_p"
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

Perform Conv3D, GroupNorm, min, clamp, and dropout.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `kernel` of shape $(C_{out}, C_{in}, K_d, K_h, K_w)$
- Tensor `conv_bias` of shape $(C_{out})$
- GroupNorm weights/biases of shape $(C_{out})$

## Output
- Tensor `output` after dropout

## Notes
- Dropout is disabled in reference for determinism.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout.py)
