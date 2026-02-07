---
slug: "matmul-scale-residualadd-clamp-logsumexp-mish"
title: "Matmul Scale ResidualAdd Clamp LogSumExp Mish"
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

  - name: "scale_factor"
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

  - name: "input_size"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "hidden_size"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform a linear projection, scale, residual add, clamp, logsumexp, and Mish-based scaling.

## Input
- Tensor `input` of shape $(B, F_{in})$
- Matrix `weight` of shape $(F_{out}, F_{in})$
- Vector `bias` of shape $(F_{out})$

## Output
- Tensor `output` of shape $(B, 1)$

## Notes
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py)
