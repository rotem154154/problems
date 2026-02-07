---
slug: "conv-transpose-2d-asym-input-asym-kernel-strided-grouped"
title: "ConvTranspose2D Asymmetric Input, Asymmetric Kernel (Strided/Grouped)"
difficulty: "HARD"
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

  - name: "output_padding_h"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "output_padding_w"
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

Perform a transposed 2D convolution on an asymmetric input with asymmetric kernel, stride, padding, dilation, and grouping.

## Input
- Tensor `input` of shape $(B, C_{\text{in}}, H, W)$
- Tensor `kernel` of shape $(C_{\text{in}}, C_{\text{out}}/g, K_h, K_w)$

## Output
- Tensor `output` of shape $(B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__.py)
