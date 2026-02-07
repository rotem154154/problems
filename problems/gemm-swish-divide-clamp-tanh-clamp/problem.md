---
slug: "gemm-swish-divide-clamp-tanh-clamp"
title: "GEMM Swish Divide Clamp Tanh Clamp"
difficulty: "MEDIUM"
author: "rotem"
tags: ["matmul", "activation"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "weight"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "bias"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "divisor"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "clamp_min"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "clamp_max"
    type: "double"
    pointer: "false"
    constant: "false"

  - name: "batch_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "in_features"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "out_features"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform GEMM, Swish, divide, clamp, tanh, and clamp.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$

## Output
- Tensor `output` after final clamp

## Notes
- Swish is $x * \\sigma(x)$.
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py)
