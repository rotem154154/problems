---
slug: "conv-transpose-1d-dilated"
title: "ConvTranspose1D Dilated"
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

  - name: "length"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "kernel_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "stride"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "padding"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "output_padding"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "dilation"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "groups"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a transposed 1D convolution with dilation.

## Input
- Tensor `input` of shape $(B, C_{\text{in}}, L)$
- Tensor `kernel` of shape $(C_{\text{in}}, C_{\text{out}}/g, K)$

## Output
- Tensor `output` of shape $(B, C_{\text{out}}, L_{\text{out}})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/74_conv_transposed_1D_dilated.py)
