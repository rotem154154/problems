---
slug: "conv-transpose2d-softmax-biasadd-scaling-sigmoid"
title: "ConvTranspose2D Softmax BiasAdd Scaling Sigmoid"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "softmax", "activation"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "conv_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "add_bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "scaling_factor"
    type: "double"
    pointer: "false"
    constant: "false"

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
---

Perform ConvTranspose2D, softmax, bias add, scaling, and sigmoid.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `kernel` of shape $(C_{in}, C_{out}, K_h, K_w)$
- Tensor `conv_bias` of shape $(C_{out})$
- Tensor `add_bias` of shape $(C_{out}, 1, 1)$

## Output
- Tensor `output` after sigmoid

## Notes
- Softmax is applied over channels.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid.py)
