---
slug: "conv2d-activation-batchnorm"
title: "Conv2D Activation BatchNorm"
difficulty: "MEDIUM"
author: "rotem"
tags: ["convolution", "activation", "normalization"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "kernel"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bn_weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bn_bias"
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
---

Perform Conv2D, activation, and BatchNorm.

## Input
- Tensor `input` of shape $(B, C_{in}, H, W)$
- Tensor `kernel` of shape $(C_{out}, C_{in}, K_h, K_w)$
- Tensor `bias` of shape $(C_{out})$
- BatchNorm weights/biases of shape $(C_{out})$

## Output
- Tensor `output` after BatchNorm

## Notes
- Activation uses $x * \\tanh(\\mathrm{softplus}(x))$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/52_Conv2d_Activation_BatchNorm.py)
