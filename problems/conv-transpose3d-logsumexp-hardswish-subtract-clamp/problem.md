---
slug: "conv-transpose3d-logsumexp-hardswish-subtract-clamp"
title: "ConvTranspose3D LogSumExp HardSwish Subtract Clamp"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "activation", "reduction"]
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

  - name: "subtract_bias"
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

Perform ConvTranspose3D, LogSumExp, HardSwish, subtract, and clamp.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_d, K_h, K_w)$
- Tensor `conv_bias` of shape $(C_{out})$
- Tensor `subtract_bias` of shape $(1, 1, 1, 1)$

## Output
- Tensor `output` after clamp

## Notes
- HardSwish uses $x * \\sigma(x + 3) / 6$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp.py)
