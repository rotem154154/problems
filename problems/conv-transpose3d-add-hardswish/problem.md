---
slug: "conv-transpose3d-add-hardswish"
title: "ConvTranspose3D Add HardSwish"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "transpose", "activation"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "add_input"
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

  - name: "output_padding_d"
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
---

Perform ConvTranspose3D, add a residual input, and apply a HardSwish-based gating.

## Input
- Tensor `input` of shape $(B, C_{in}, D, H, W)$
- Tensor `add_input` of shape $(B, C_{out}, D_{out}, H_{out}, W_{out})$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_d, K_h, K_w)$

## Output
- Tensor `output` of shape $(B, C_{out}, D_{out}, H_{out}, W_{out})$

## Notes
- The activation uses `x * hardswish(x)` after the addition.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/26_ConvTranspose3d_Add_HardSwish.py)
