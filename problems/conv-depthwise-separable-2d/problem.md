---
slug: "conv-depthwise-separable-2d"
title: "Depthwise Separable Conv2D"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "depthwise"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "depthwise_kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "pointwise_kernel"
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

Perform a depthwise-separable 2D convolution: a depthwise convolution followed by
a pointwise $1 \times 1$ convolution.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `depthwise_kernel` of shape $(C_{in}, 1, K_h, K_w)$ with $groups = C_{in}$
- Tensor `pointwise_kernel` of shape $(C_{out}, C_{in}, 1, 1)$

## Output
- Tensor `output` of shape $(B, C_{out}, H_{out}, W_{out})$

## Notes
- Depthwise convolution uses `groups = C_{in}`.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/86_conv_depthwise_separable_2D.py)
