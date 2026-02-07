---
slug: "conv-2d-asym-input-sq-kernel"
title: "Conv2D Asymmetric Input, Square Kernel"
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

Perform a standard 2D convolution on an asymmetric input with a square kernel:
$$
\text{output} = \text{conv2d}(\text{input}, \text{kernel})
$$

## Input
- Tensor `input` of shape $(B, C_{\text{in}}, H, W)$
- Tensor `kernel` of shape $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$

## Output
- Tensor `output` of shape $(B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

## Notes
- Asymmetric input uses $H \ne W$
- Stride, padding, and dilation are configurable
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/55_conv_standard_2D__asymmetric_input__square_kernel.py)
