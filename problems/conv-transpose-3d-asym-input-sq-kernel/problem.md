---
slug: "conv-transpose-3d-asym-input-sq-kernel"
title: "ConvTranspose3D Asymmetric Input, Square Kernel"
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

  - name: "dilation_d"
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

Perform a transposed 3D convolution on an asymmetric input with a square kernel.

## Input
- Tensor `input` of shape $(B, C_{\text{in}}, D, H, W)$
- Tensor `kernel` of shape $(C_{\text{in}}, C_{\text{out}}/g, K, K, K)$

## Output
- Tensor `output` of shape $(B, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/70_conv_transposed_3D__asymmetric_input__square_kernel.py)
