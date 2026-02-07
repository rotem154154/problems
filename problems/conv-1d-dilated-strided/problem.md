---
slug: "conv-1d-dilated-strided"
title: "Conv1D Dilated and Strided"
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

  - name: "dilation"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a standard 1D convolution with dilation and stride.

## Input
- Tensor `input` of shape $(B, C_{\text{in}}, L)$
- Tensor `kernel` of shape $(C_{\text{out}}, C_{\text{in}}, K)$

## Output
- Tensor `output` of shape $(B, C_{\text{out}}, L_{\text{out}})$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/76_conv_standard_1D_dilated_strided__.py)
