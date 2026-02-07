---
slug: "conv-depthwise-2d-square-input-asym-kernel"
title: "Depthwise Conv2D Square Input, Asymmetric Kernel"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "depthwise"]
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

  - name: "dilation_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "dilation_w"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "groups"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a depthwise 2D convolution on a square input with an asymmetric kernel.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `kernel` of shape $(C_{out}, C_{in}/groups, K_h, K_w)$ with $groups = C_{in}$

## Output
- Tensor `output` of shape $(B, C_{out}, H_{out}, W_{out})$

## Notes
- Depthwise convolution uses `groups = C_{in}`.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/83_conv_depthwise_2D_square_input_asymmetric_kernel.py)
