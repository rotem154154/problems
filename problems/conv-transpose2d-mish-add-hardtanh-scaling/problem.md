---
slug: "conv-transpose2d-mish-add-hardtanh-scaling"
title: "ConvTranspose2D Mish Add Hardtanh Scaling"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "activation"]
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

  - name: "add_value"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "scale"
    type: "double"
    pointer: "false"
    constant: "false"
---

Perform ConvTranspose2D, Mish, add value, Hardtanh, and scaling.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_h, K_w)$

## Output
- Tensor `output` after scaling

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py)
