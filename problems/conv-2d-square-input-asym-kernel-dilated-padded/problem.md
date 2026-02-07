---
slug: "conv-2d-square-input-asym-kernel-dilated-padded"
title: "Conv2D Square Input, Asymmetric Kernel (Dilated/Padded)"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution"]
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
---

Perform a standard 2D convolution on a square input with an asymmetric kernel, dilation, and padding.

## Input
- Tensor `input` of shape $(B, C_{\text{in}}, H, W)$
- Tensor `kernel` of shape $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$

## Output
- Tensor `output` of shape $(B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

## Notes
- Dilation and padding are non-default
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__.py)
